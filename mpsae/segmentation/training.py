"""Batch coverage and semantic patch targets shared by SAE trainers."""
import numpy as np

def batch_ranges(rows, batch_size):
    if rows < 2:
        raise ValueError('SAE adaptation needs at least two training patches')
    ends = list(range(batch_size, rows, batch_size)) + [rows]
    if len(ends) > 1 and ends[-1] - ends[-2] == 1:
        ends.pop(-2)  # merge a singleton into the preceding batch, never drop a patch
    return list(zip([0] + ends[:-1], ends))


def patch_targets(labels, *, classes=21):
    labels = np.asarray(labels)
    if (labels.ndim != 2 or labels.shape[1] != classes or not np.isfinite(labels).all()
            or np.any(labels < 0) or not np.allclose(labels.sum(1), 1, atol=1e-5)):
        raise ValueError('training patches need valid nonvoid semantic label distributions')
    target = labels.argmax(1).astype(np.int64)
    purity = labels.max(1)
    stats = {'label_rule': 'dominant nonvoid class; ties choose lowest class ID',
        'class_counts': np.bincount(target, minlength=classes).tolist(),
        'mean_purity': float(purity.mean()), 'mixed_fraction': float((purity < 1-1e-6).mean()),
        'background_fraction': float((target == 0).mean())}
    if classes != 21:
        stats.pop('background_fraction')  # ADE class zero is wall, not background
    return target, stats


