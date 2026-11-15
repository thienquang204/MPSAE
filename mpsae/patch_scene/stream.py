"""Full-coverage training and support sampling with bounded memory."""
import numpy as np
import torch

from ..io import read_json, canonical_hash, atomic_json
from ..scene.adapt import batch_ranges, patch_targets


class ShardTraining:
    def __init__(self, manifest, hub):
        self.hub, self.classes = hub, manifest['classes']
        self.entries = []
        for entry in manifest['shards']:
            if entry['split'] != 'train':
                continue
            folder = hub.pull(entry['path'])
            meta = read_json(folder/'metadata.json') or {}
            rows = meta.get('training_rows')
            if rows is None:
                with np.load(folder/'data.npz', allow_pickle=False) as data:
                    rows = len(data['features'])
            bundle = read_json(folder/'bundle.json')
            self.entries.append({'path': entry['path'], 'rows': rows,
                                 'sha256': bundle['files']['data.npz']})
        self.rows = sum(e['rows'] for e in self.entries)
        if self.rows < 2:
            raise ValueError('full training needs at least two valid patches')
        self.identity = {'protocol': 'full-shard-stream-v1', 'rows': self.rows, 'shards': self.entries}

    def steps(self, batch_size):
        return len(batch_ranges(self.rows, batch_size))

    def batches(self, seed, epoch, batch_size, first_batch, device):
        ranges = batch_ranges(self.rows, batch_size)
        if first_batch >= len(ranges):
            return
        skip, b = ranges[first_batch][0], first_batch
        need = ranges[b][1]-ranges[b][0]
        xs, ys, held = [], [], 0
        for shard_index in np.random.default_rng([seed, epoch]).permutation(len(self.entries)):
            entry = self.entries[shard_index]
            if skip >= entry['rows']:
                skip -= entry['rows']
                continue
            folder = self.hub.pull(entry['path'])
            with np.load(folder/'data.npz', allow_pickle=False) as data:
                x = data['features']
                y, _ = patch_targets(data['labels'], classes=self.classes)
            if len(x) != entry['rows'] or not np.isfinite(x).all():
                raise ValueError('invalid training shard')
            indices = np.random.default_rng([seed, epoch, int(shard_index), 1]).permutation(len(x))
            pos, skip = skip, 0
            while pos < len(x):
                take = min(need-held, len(x)-pos)
                idx = indices[pos:pos+take]
                xs.append(x[idx]); ys.append(y[idx])
                pos += take; held += take
                if held == need:
                    a = torch.from_numpy(np.concatenate(xs).astype(np.float32))
                    target = torch.from_numpy(np.concatenate(ys))
                    if str(device).startswith('cuda'):
                        a, target = a.pin_memory(), target.pin_memory()
                    yield b, a.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    b += 1
                    xs, ys, held = [], [], 0
                    if b == len(ranges):
                        return
                    need = ranges[b][1]-ranges[b][0]
        raise RuntimeError('incomplete full-data epoch')


def support_bank(manifest, hub, config):
    """Same frequency-score rule as sample_patches, applied to ALL training rows.

    First pass computes global class frequencies, second keeps only the best scores.
    The preparation job is the only writer; later jobs reuse its small bank.
    """
    from .features import save_npz
    identity = {'features': manifest['identity'], 'count': config.support_patches,
                'seed': config.seed, 'sampler': 'full-frequency-score-v1'}
    relative = 'support/'+canonical_hash(identity)[:20]
    folder = hub.pull(relative)
    receipt = read_json(folder/'manifest.json')
    if receipt is not None:
        if receipt['identity'] != identity:
            raise RuntimeError('support bank identity mismatch')
        with np.load(folder/'data.npz', allow_pickle=False) as data:
            return data['features'], data['labels'], np.arange(len(data['features']))
    entries = [e for e in manifest['shards'] if e['split'] == 'train']
    frequencies = np.zeros(manifest['classes'], dtype=np.int64)
    for entry in entries:
        with np.load(hub.pull(entry['path'])/'data.npz', allow_pickle=False) as data:
            frequencies += (data['labels'] > 0).sum(0)
    rng = np.random.default_rng(config.seed)
    keys, ids, xs, ys, offset = np.empty(0), np.empty(0, dtype=np.int64), None, None, 0
    for entry in entries:
        with np.load(hub.pull(entry['path'])/'data.npz', allow_pickle=False) as data:
            x, y = data['features'], data['labels']
        present = y > 0
        scores = (present*frequencies).sum(1)*rng.random(len(y))
        scores[~present.any(1)] = np.inf
        # Exact global top-count via incremental selection; ties use original order.
        candidate = np.argsort(scores, kind='stable')[:config.support_patches]
        keys = np.concatenate([keys, scores[candidate]])
        ids = np.concatenate([ids, offset+candidate])
        xs = x[candidate] if xs is None else np.concatenate([xs, x[candidate]])
        ys = y[candidate] if ys is None else np.concatenate([ys, y[candidate]])
        keep = np.lexsort((ids, keys))[:config.support_patches]
        keys, ids, xs, ys = keys[keep], ids[keep], xs[keep], ys[keep]
        offset += len(y)
    valid = np.isfinite(keys)
    xs, ys, ids = xs[valid].astype(np.float32), ys[valid], ids[valid]
    if len(xs) < config.neighbors:
        raise ValueError('insufficient support patches')
    save_npz(folder/'data.npz', features=xs, labels=ys, global_indices=ids)
    atomic_json(folder/'manifest.json', {'identity': identity, 'training_rows': offset,
                                       'support_rows': len(xs)})
    hub.push(relative)
    return xs, ys, np.arange(len(xs))
