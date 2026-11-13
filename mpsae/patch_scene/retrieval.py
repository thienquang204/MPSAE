"""Exact cosine search: sparse queries touch K gallery rows, not all 8192."""
import numpy as np
import torch
from torch.nn import functional as F

from ..evaluate.knn import fp32_matmul


def upsample_prediction(probs, empty, grid, shape, tile_rows=64):
    """Bilinear probabilities then argmax, bounded native-resolution workspace."""
    classes, device = probs.shape[1], probs.device
    h, w = map(int, shape)
    source = probs.T.reshape(1, classes, *grid)
    columns = 2*(torch.arange(w, device=device, dtype=torch.float32)+0.5)/w-1
    result = []
    for start in range(0, h, tile_rows):
        stop = min(h, start+tile_rows)
        rows = 2*(torch.arange(start, stop, device=device, dtype=torch.float32)+0.5)/h-1
        yy, xx = torch.meshgrid(rows, columns, indexing='ij')
        coords = torch.stack((xx, yy), -1)[None]
        values = F.grid_sample(source, coords, mode='bilinear', padding_mode='border', align_corners=False)
        prediction = values.argmax(1)[0]
        # Match F.interpolate(..., mode='nearest') for the abstention mask.
        iy = torch.arange(start, stop, device=device)*grid[0]//h
        ix = torch.arange(w, device=device)*grid[1]//w
        prediction[empty.reshape(*grid)[iy[:, None], ix[None, :]]] = classes
        result.append(prediction.cpu())
    return torch.cat(result).numpy()


def stable_topk(scores, k):
    """Score descending, gallery position ascending, including boundary ties."""
    k = min(k, scores.shape[1])
    threshold = scores.topk(k, dim=1, sorted=False).values.min(1, keepdim=True).values
    above, tied = scores > threshold, scores == threshold
    needed = k-above.sum(1, keepdim=True)
    keep = above | (tied & (tied.cumsum(1) <= needed))
    positions = torch.arange(scores.shape[1], device=scores.device).expand_as(scores)
    selected = positions.masked_fill(~keep, scores.shape[1]).topk(k, largest=False, sorted=True).values
    values = scores.gather(1, selected)
    order = values.argsort(dim=1, descending=True, stable=True)
    return values.gather(1, order), selected.gather(1, order)


@torch.inference_mode()
def codes(model, features, k, device, batch=1024):
    indices, values = [], []
    with fp32_matmul():
        for start in range(0, len(features), batch):
            x = torch.as_tensor(features[start:start+batch], device=device, dtype=torch.float32)
            selected = model.pre_activations(x).topk(k, dim=-1)
            z = selected.values.relu()
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite sparse code')
            indices.append(selected.indices)
            values.append(z)
    return torch.cat(indices), torch.cat(values)


class Search:
    """One GPU-resident gallery per evaluation; compact codes remain the representation.

    The transposed dense lookup is a search workspace (2 GiB at 65536 x 8192),
    not claimed as compact index storage. At K=12 it avoids 8192-d GEMMs.
    """
    def __init__(self, gallery, labels, device, *, dim, sparse, neighbors, temperature, query_block):
        self.device, self.sparse = torch.device(device), sparse
        self.labels = torch.as_tensor(labels, device=device, dtype=torch.float32)
        self.neighbors, self.temperature, self.block = neighbors, temperature, query_block
        if len(labels) < neighbors:
            raise ValueError('gallery smaller than neighbor count')
        if sparse:
            idx, val = gallery
            idx, val = idx.to(device), val.to(device)
            val = F.normalize(val.float(), dim=1)
            n = len(val)
            needed = dim*n*4
            if self.device.type == 'cuda':
                free, _ = torch.cuda.mem_get_info(self.device)
                if needed + (512 << 20) > free:
                    raise RuntimeError('Insufficient GPU memory for exact lookup; reduce support_patches')
            self.lookup = torch.zeros((dim, n), device=device, dtype=torch.float32)
            self.lookup.scatter_(0, idx.T, val.T)
        else:
            self.lookup = F.normalize(torch.as_tensor(gallery, device=device, dtype=torch.float32), dim=1).T.contiguous()
        self.workspace_bytes = self.lookup.numel()*self.lookup.element_size()

    @torch.inference_mode()
    def predict(self, query, grid, shape):
        if self.sparse:
            idx, val = query
            val = F.normalize(val.to(self.device).float(), dim=1)
            idx = idx.to(self.device)
            empty = ~val.ne(0).any(1)
            rows = len(val)
        else:
            q = F.normalize(torch.as_tensor(query, device=self.device, dtype=torch.float32), dim=1)
            empty = ~q.ne(0).any(1)
            rows = len(q)
        probabilities = []
        with fp32_matmul():
            for start in range(0, rows, self.block):
                stop = min(start+self.block, rows)
                if self.sparse:
                    scores = torch.zeros((stop-start, self.lookup.shape[1]), device=self.device)
                    for j in range(val.shape[1]):
                        scores.addcmul_(self.lookup.index_select(0, idx[start:stop, j]), val[start:stop, j, None])
                else:
                    scores = q[start:stop] @ self.lookup
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('nonfinite search score')
                best, positions = stable_topk(scores, self.neighbors)
                weights = (best/self.temperature).softmax(1)
                probabilities.append((weights[..., None]*self.labels[positions]).sum(1))
        probs = torch.cat(probabilities)
        probs[empty] = 0
        return upsample_prediction(probs, empty, grid, shape), int(empty.sum())
