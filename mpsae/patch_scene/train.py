from dataclasses import asdict
import time

import numpy as np
import torch

from .. import methods
from ..io import atomic_json, canonical_hash, read_json, append_jsonl, array_hash, sha256_file
from ..model import SparseAutoencoder
from ..evaluate.knn import fp32_matmul
from ..scene.adapt import batch_ranges, patch_targets
from .artifacts import save_torch
from .config import VERSION


def fit(config, method, features, labels, feature_identity, hub, device, *, preset=None, stop_after=None):
    preset = preset or config.preset()
    targets, label_stats = patch_targets(labels, classes=labels.shape[1])
    ranges = batch_ranges(len(features), preset.batch_size)
    total = len(ranges)*preset.epochs
    identity = {'version': VERSION, 'features': feature_identity,
                'feature_sha256': array_hash(features), 'label_sha256': array_hash(labels),
                'method': method, 'preset': asdict(preset), 'seed': config.seed, 'k': config.k,
                'objective': methods.describe(method, config.k, preset), 'total_steps': total,
                'initialization': 'fresh SAE; never load an image-level SAE',
                'label_policy': label_stats['label_rule'] if method == 'csrv2' else 'not used by loss',
                'selection': 'final epoch', 'precision': 'fp32_tf32_disabled'}
    # Normalize tuples for serialization and exact identity comparisons.
    import json
    identity = json.loads(json.dumps(identity))
    relative = f'fits/{canonical_hash(identity)[:20]}'
    folder = hub.pull(relative)
    torch.manual_seed(config.seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed)
    model = SparseAutoencoder(features.shape[1], preset.hidden_dim,
                              dead_threshold=preset.dead_threshold).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=preset.lr, betas=preset.adam_betas,
                                 eps=preset.adam_eps, weight_decay=preset.weight_decay)
    step, elapsed = 0, 0.0
    path = folder/'checkpoint.pt'
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise RuntimeError('training resume identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step, elapsed = state['next_step'], state['train_seconds']
        if not 0 <= step <= total:
            raise ValueError('invalid resume step')
        torch.set_rng_state(state['rng_cpu'].cpu())
        if str(device).startswith('cuda') and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state(state['rng_cuda'].cpu(), device)
        if step == total and (folder/'summary.json').exists():
            summary = read_json(folder/'summary.json')
            if summary['checkpoint_sha256'] != sha256_file(path):
                raise RuntimeError('completed checkpoint hash mismatch')
            return model.eval().requires_grad_(False), summary
    atomic_json(folder/'config.json', {'identity': identity, 'patch_label_statistics': label_stats})
    # Keep one bank resident; no image decoder/backbone remains during SAE training.
    x = torch.as_tensor(features, device=device)
    y = torch.as_tensor(targets, device=device)
    started = time.perf_counter()

    def checkpoint(remote=True):
        save_torch(path, {'identity': identity, 'next_step': step, 'complete': step == total,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None,
            'train_seconds': elapsed+time.perf_counter()-started})
        hub.push(relative, remote=remote)

    model.train()
    with fp32_matmul():
        for epoch in range(step//len(ranges), preset.epochs):
            order = np.random.default_rng([config.seed, epoch]).permutation(len(features))
            for b, (start, stop) in enumerate(ranges):
                if epoch*len(ranges)+b < step:
                    continue
                idx = torch.as_tensor(order[start:stop], device=device)
                optimizer.zero_grad(set_to_none=True)
                collect = step == 0 or (step+1) % config.log_every == 0 or step+1 == total
                out = methods.compute_step(method, model, x[idx], k=config.k, preset=preset,
                    step=step, total_steps=total, labels=y[idx] if method == 'csrv2' else None,
                    collect_metrics=collect)
                if not torch.isfinite(out.loss):
                    raise FloatingPointError('nonfinite training loss')
                out.loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite gradient')
                optimizer.step()
                step += 1
                if collect:
                    print(f'[fit {method}] {step}/{total} K={out.training_k} loss={out.loss.item():.5f}', flush=True)
                    atomic_json(folder/'progress.json', {'step': step, 'total': total, 'loss': out.loss.item(),
                        'training_k': out.training_k, 'metrics': out.metrics})
                if step % config.checkpoint_every == 0 or b+1 == len(ranges) or step == stop_after:
                    checkpoint(remote=b+1 == len(ranges) or step == stop_after)
                if step == stop_after and step < total:
                    return None
    if not path.exists():
        checkpoint()
    summary = {'complete': True, 'identity': identity, 'path': relative,
               'checkpoint_sha256': sha256_file(path), 'steps': total,
               'train_seconds': elapsed+time.perf_counter()-started}
    atomic_json(folder/'summary.json', summary)
    hub.push(relative)
    del optimizer, x, y
    return model.eval().requires_grad_(False), summary
