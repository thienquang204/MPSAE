"""Patch-label distributions, paper sampling, and soft-label retrieval decoding."""
import numpy as np
import torch
from torch.nn import functional as F

from ..evaluate.ranking import exact_topk

CLASSES = 21
IGNORE = 255


def label_distributions(mask, grid, *, classes=CLASSES):
    """Average one-hot labels, excluding void; never average class ID numbers."""
    mask = torch.as_tensor(np.array(mask, copy=True), dtype=torch.long)
    if mask.ndim != 2 or not bool(((mask >= 0) & ((mask < classes) | (mask == IGNORE))).all()):
        raise ValueError('masks must be 2-D semantic class IDs or void 255; not RGB')
    if classes != CLASSES and mask.shape[0] % grid[0] == 0 and mask.shape[1] % grid[1] == 0:
        # ADE: avoid a H*W*150 int64 one-hot allocation for each training image.
        gh, gw = grid
        cells = mask.numpy().reshape(gh, mask.shape[0]//gh, gw, mask.shape[1]//gw)
        cells = cells.transpose(0, 2, 1, 3).reshape(gh*gw, -1)
        counts = np.stack([np.bincount(row[row != IGNORE], minlength=classes) for row in cells]).astype(np.float32)
        return counts / counts.sum(1, keepdims=True).clip(min=1)
    valid = mask != IGNORE
    onehot = F.one_hot(mask.masked_fill(~valid, 0), classes).float()
    onehot *= valid[..., None]
    pooled = F.adaptive_avg_pool2d(onehot.permute(2, 0, 1)[None], grid)[0]
    pooled = pooled.permute(1, 2, 0).reshape(-1, classes)
    mass = pooled.sum(1, keepdim=True)
    return (pooled / mass.clamp_min(1e-12)).numpy()


def sample_patches(labels, count, rng):
    """Appendix A.1 class-frequency scores; all-void patches are excluded."""
    present = labels > 0
    valid = present.any(1)
    frequencies = present.sum(0)
    score = (present * frequencies).sum(1) * rng.random(len(labels))
    score[~valid] = np.inf
    return np.argsort(score, kind='stable')[:min(count, int(valid.sum()))]


def resnet_spatial(network, images):
    """Original layer4 output, before avgpool; no stride/dilation changes."""
    x = network.maxpool(network.relu(network.bn1(network.conv1(images))))
    for layer in (network.layer1, network.layer2, network.layer3, network.layer4):
        x = layer(x)
    return x.permute(0, 2, 3, 1)


@torch.inference_mode()
def encode(sae, features, k, device, batch=1024):
    """Compact float32 codes; the frozen SAE receives unnormalised features."""
    indices, values = [], []
    for start in range(0, len(features), batch):
        x = torch.tensor(np.array(features[start:start+batch], copy=True), device=device,
                         dtype=torch.float32)
        selected = sae.pre_activations(x).topk(k, dim=-1)
        z = selected.values.relu()
        if not bool(torch.isfinite(z).all()):
            raise ValueError('nonfinite sparse embedding')
        indices.append(selected.indices.cpu().numpy().astype(np.int32))
        values.append(z.cpu().numpy())
    return {'indices': np.concatenate(indices), 'values': np.concatenate(values)}


def predict(gallery, labels, queries, *, dim, grid, output_size, device, neighbors=30,
            temperature=0.02, gallery_chunk=4096, query_block=256):
    """Softmax-weighted soft labels, bilinear upsampling, then class argmax."""
    classes = np.asarray(labels).shape[1]
    scores, positions, seconds = exact_topk(
        gallery, queries, dim=dim, device=device, depth=neighbors, normalize=True,
        gallery_chunk=gallery_chunk, query_block=query_block, progress_stage='scene retrieval')
    weights = np.exp((scores - scores.max(1, keepdims=True)) / temperature)
    weights /= weights.sum(1, keepdims=True)
    probs = np.einsum('nk,nkc->nc', weights, np.asarray(labels)[positions])
    qvalues = queries['values'] if isinstance(queries, dict) else queries
    empty = ~np.any(np.asarray(qvalues) != 0, axis=1)
    # Empty sparse codes carry no evidence. Abstain wherever the nearest grid cell is empty.
    probs[empty] = 0
    probs = torch.from_numpy(probs.T.copy()).reshape(1, classes, *grid)
    upsampled = F.interpolate(probs, size=output_size, mode='bilinear', align_corners=False)
    prediction = upsampled.argmax(1)[0].numpy()
    abstain = F.interpolate(torch.from_numpy(empty.astype(np.float32)).reshape(1, 1, *grid),
                            size=output_size, mode='nearest')[0, 0].bool().numpy()
    prediction[abstain] = classes
    return prediction, {'search_seconds': seconds, 'empty_query_patches': int(empty.sum())}


def confusion(prediction, truth, *, classes=CLASSES):
    truth, prediction = np.asarray(truth, dtype=np.int64), np.asarray(prediction, dtype=np.int64)
    if truth.shape != prediction.shape:
        raise ValueError('prediction and original mask sizes differ')
    valid = truth != IGNORE
    if not np.all((truth[valid] >= 0) & (truth[valid] < classes)):
        raise ValueError('invalid ground-truth mask')
    if not np.all((prediction >= 0) & (prediction <= classes)):
        raise ValueError('prediction must be a VOC class or abstention')
    # Extra prediction column counts abstentions as false negatives, never discards them.
    return np.bincount((truth[valid]*(classes+1) + prediction[valid]).astype(np.int64),
                       minlength=classes*(classes+1)).reshape(classes, classes+1)


def metrics(matrix, *, include_absent=False):
    matrix = np.asarray(matrix)
    classes = matrix.shape[0]
    if matrix.shape != (classes, classes+1):
        raise ValueError('confusion matrix must include one abstention column')
    tp = np.diag(matrix[:, :classes])
    union = matrix.sum(1) + matrix[:, :classes].sum(0) - tp
    iou = np.divide(tp, union, out=np.zeros(classes, dtype=float), where=union > 0)
    selected = np.ones(classes, dtype=bool) if include_absent else union > 0
    return {'miou': float(iou[selected].mean()) if np.any(union) else None,
            'pixel_accuracy': float(tp.sum()/matrix.sum()) if matrix.sum() else None,
            'class_iou': [float(iou[c]) if union[c] or include_absent else None for c in range(classes)],
            'classes_scored': int(selected.sum()), 'valid_pixels': int(matrix.sum()),
            'abstained_pixels': int(matrix[:, classes].sum()), 'units': 'fraction'}
