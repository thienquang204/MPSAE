"""Patch-label distributions, paper sampling, and soft-label retrieval decoding."""
import numpy as np
import torch
from torch.nn import functional as F

from ..evaluate.ranking import exact_topk

CLASSES = 21
IGNORE = 255


def label_distributions(mask, grid):
    """Average one-hot labels, excluding void; never average class ID numbers."""
    mask = torch.as_tensor(np.array(mask, copy=True), dtype=torch.long)
    if mask.ndim != 2 or not bool(((mask >= 0) & ((mask < CLASSES) | (mask == IGNORE))).all()):
        raise ValueError('VOC masks must be 2-D class IDs 0..20 or void 255; not RGB')
    valid = mask != IGNORE
    onehot = F.one_hot(mask.masked_fill(~valid, 0), CLASSES).float()
    onehot *= valid[..., None]
    pooled = F.adaptive_avg_pool2d(onehot.permute(2, 0, 1)[None], grid)[0]
    pooled = pooled.permute(1, 2, 0).reshape(-1, CLASSES)
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
    probs = torch.from_numpy(probs.T.copy()).reshape(1, CLASSES, *grid)
    upsampled = F.interpolate(probs, size=output_size, mode='bilinear', align_corners=False)
    prediction = upsampled.argmax(1)[0].numpy()
    abstain = F.interpolate(torch.from_numpy(empty.astype(np.float32)).reshape(1, 1, *grid),
                            size=output_size, mode='nearest')[0, 0].bool().numpy()
    prediction[abstain] = CLASSES
    return prediction, {'search_seconds': seconds, 'empty_query_patches': int(empty.sum())}


def confusion(prediction, truth):
    truth, prediction = np.asarray(truth, dtype=np.int64), np.asarray(prediction, dtype=np.int64)
    if truth.shape != prediction.shape:
        raise ValueError('prediction and original mask sizes differ')
    valid = truth != IGNORE
    if not np.all((truth[valid] >= 0) & (truth[valid] < CLASSES)):
        raise ValueError('invalid ground-truth mask')
    if not np.all((prediction >= 0) & (prediction <= CLASSES)):
        raise ValueError('prediction must be a VOC class or abstention')
    # Extra prediction column counts abstentions as false negatives, never discards them.
    return np.bincount((truth[valid]*(CLASSES+1) + prediction[valid]).astype(np.int64),
                       minlength=CLASSES*(CLASSES+1)).reshape(CLASSES, CLASSES+1)


def metrics(matrix):
    matrix = np.asarray(matrix)
    tp = np.diag(matrix[:, :CLASSES])
    union = matrix.sum(1) + matrix[:, :CLASSES].sum(0) - tp
    iou = np.divide(tp, union, out=np.zeros(CLASSES, dtype=float), where=union > 0)
    return {'miou': float(iou[union > 0].mean()) if np.any(union) else None,
            'pixel_accuracy': float(tp.sum()/matrix.sum()) if matrix.sum() else None,
            'class_iou': [float(iou[c]) if union[c] else None for c in range(CLASSES)],
            'classes_scored': int((union > 0).sum()), 'valid_pixels': int(matrix.sum()),
            'abstained_pixels': int(matrix[:, CLASSES].sum()), 'units': 'fraction'}
