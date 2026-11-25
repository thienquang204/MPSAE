from dataclasses import asdict
import time
import math

import numpy as np
import torch

from .. import methods
from ..io import atomic_json, canonical_hash, read_json, append_jsonl, array_hash, sha256_file
from ..model import SparseAutoencoder
from ..evaluate.knn import fp32_matmul
from ..segmentation.training import batch_ranges, patch_targets
from .artifacts import save_torch
from .config import VERSION


def fit(config, method, features, labels, feature_identity, hub, device, *, preset=None, stop_after=None, training_source=None, allow_training=True):
    preset = preset or config.preset(method, feature_identity.get('encoder'))
    targets, label_stats = patch_targets(labels, classes=labels.shape[1])
    ranges = batch_ranges(len(features), preset.batch_size)
    if training_source is not None:
        ranges = range(training_source.steps(preset.batch_size))
    total = len(ranges)*preset.epochs
    track_epochs = config.early_stopping or training_source is not None
    identity = {'version': VERSION, 'features': feature_identity,
                'feature_sha256': array_hash(features), 'label_sha256': array_hash(labels),
                'method': method, 'preset': asdict(preset), 'seed': config.seed, 'k': config.k,
                'objective': methods.describe(method, config.k, preset), 'total_steps': total,
                'initialization': 'fresh SAE; never load an image-level SAE',
                'label_policy': label_stats['label_rule'] if method == 'csrv2' else 'not used by loss',
                'selection': 'final epoch', 'precision': 'fp32_tf32_disabled'}
    if training_source is not None:
        identity['full_training'] = training_source.identity
        identity['sampling'] = 'all valid patches once per epoch; shuffle shards then shuffle patches'
        label_stats = {'label_rule': 'dominant nonvoid class; ties choose lowest class ID',
                       'training_rows': training_source.rows}
    if config.early_stopping:
        identity['early_stopping'] = {
            'monitor': 'sample_weighted_epoch_training_objective',
            'min_epochs': config.early_min_epochs, 'patience': config.early_patience,
            'relative_min_delta': config.early_min_delta,
            'start': 'after minimum epochs and a full epoch at target K',
            'selection': 'last completed epoch; no validation selection'}
        identity['selection'] = 'last epoch at plateau or epoch limit'
    # Normalize tuples for serialization and exact identity comparisons.
    import json
    identity = json.loads(json.dumps(identity))
    relative = f'fits/{canonical_hash(identity)[:20]}'
    folder = hub.pull(relative)
    if not allow_training:
        completed = read_json(folder/"summary.json") or {}
        if not completed.get("complete") or completed.get("identity") != identity or not (folder/"checkpoint.pt").is_file():
            raise FileNotFoundError(f"{folder}: evaluation-only requires a completed compatible checkpoint")
    torch.manual_seed(config.seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed)
    model = SparseAutoencoder(features.shape[1], preset.hidden_dim,
                              dead_threshold=preset.dead_threshold).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=preset.lr, betas=preset.adam_betas,
                                 eps=preset.adam_eps, weight_decay=preset.weight_decay)
    step, elapsed = 0, 0.0
    tracker = {'best': None, 'bad_epochs': 0, 'loss_sum': 0.0, 'samples': 0,
               'history': [], 'stopped': False}
    path = folder/'checkpoint.pt'
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise RuntimeError('training resume identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step, elapsed = state['next_step'], state['train_seconds']
        tracker = state.get('convergence', tracker)
        if not 0 <= step <= total:
            raise ValueError('invalid resume step')
        torch.set_rng_state(state['rng_cpu'].cpu())
        if str(device).startswith('cuda') and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state(state['rng_cuda'].cpu(), device)
        if (step == total or tracker['stopped']) and (folder/'summary.json').exists():
            summary = read_json(folder/'summary.json')
            if summary['checkpoint_sha256'] != sha256_file(path):
                raise RuntimeError('completed checkpoint hash mismatch')
            return model.eval().requires_grad_(False), summary
    if not allow_training:
        raise RuntimeError(f"{folder}: completed summary has an incomplete checkpoint; evaluation-only cannot resume training")
    atomic_json(folder/'config.json', {'identity': identity, 'patch_label_statistics': label_stats})
    # Keep one bank resident; no image decoder/backbone remains during SAE training.
    x = torch.as_tensor(features, device=device) if training_source is None else None
    y = torch.as_tensor(targets, device=device) if training_source is None else None
    started = time.perf_counter()

    def checkpoint(remote=True):
        save_torch(path, {'identity': identity, 'next_step': step, 'complete': step == total or tracker['stopped'],
            'convergence': tracker,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None,
            'train_seconds': elapsed+time.perf_counter()-started})
        hub.push(relative, remote=remote)

    model.train()
    with fp32_matmul():
        for epoch in range(step//len(ranges), preset.epochs):
            if tracker['stopped']:
                break
            first_batch = max(0, step-epoch*len(ranges))
            if training_source is not None:
                batches = training_source.batches(config.seed, epoch, preset.batch_size, first_batch, device)
            else:
                order = np.random.default_rng([config.seed, epoch]).permutation(len(features))
                def resident_batches():
                    for b, (start, stop) in enumerate(ranges):
                        if b < first_batch:
                            continue
                        idx = torch.as_tensor(order[start:stop], device=device)
                        yield b, x[idx], y[idx]
                batches = resident_batches()
            for b, batch_x, batch_y in batches:
                batch_rows = len(batch_x)
                optimizer.zero_grad(set_to_none=True)
                collect = step == 0 or (step+1) % config.log_every == 0 or step+1 == total
                out = methods.compute_step(method, model, batch_x, k=config.k, preset=preset,
                    step=step, total_steps=total, labels=batch_y if method == 'csrv2' else None,
                    collect_metrics=collect)
                if not torch.isfinite(out.loss):
                    raise FloatingPointError('nonfinite training loss')
                out.loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite gradient')
                optimizer.step()
                if track_epochs:
                    tracker['loss_sum'] += float(out.loss.detach()) * batch_rows
                    tracker['samples'] += batch_rows
                step += 1
                if collect:
                    print(f'[fit {method}] {step}/{total} K={out.training_k} loss={out.loss.item():.5f}', flush=True)
                    atomic_json(folder/'progress.json', {'step': step, 'total': total, 'loss': out.loss.item(),
                        'training_k': out.training_k, 'metrics': out.metrics})
                if track_epochs and b+1 == len(ranges):
                    mean_loss = tracker['loss_sum']/tracker['samples']
                    # CSRv2 changes its objective's K during the first 70% of training.
                    first_k = methods.plan_budget(method, config.k, preset, step=epoch*len(ranges), total_steps=total)
                    eligible = config.early_stopping and epoch+1 >= config.early_min_epochs and first_k == config.k
                    update_plateau(tracker, mean_loss, eligible, config.early_patience, config.early_min_delta)
                    tracker['history'].append({'epoch': epoch+1, 'mean_loss': mean_loss,
                        'eligible': eligible, 'bad_epochs': tracker['bad_epochs'],
                        'stopped': tracker['stopped']})
                    tracker['loss_sum'], tracker['samples'] = 0.0, 0
                    atomic_json(folder/'convergence.json', tracker)
                    print(f"[convergence {method}] epoch={epoch+1} mean_loss={mean_loss:.6g} "
                          f"eligible={eligible} patience={tracker['bad_epochs']}/{config.early_patience} "
                          f"stopped={tracker['stopped']}", flush=True)
                if step % config.checkpoint_every == 0 or b+1 == len(ranges) or step == stop_after:
                    checkpoint(remote=b+1 == len(ranges) or step == stop_after)
                if step == stop_after and step < total and not tracker['stopped']:
                    return None
    if not path.exists():
        checkpoint()
    summary = {'complete': True, 'identity': identity, 'path': relative,
               'checkpoint_sha256': sha256_file(path), 'steps': step,
               'train_seconds': elapsed+time.perf_counter()-started,
               'training_patches': training_source.rows if training_source is not None else len(features),
               'epochs_completed': step//len(ranges), 'max_epochs': preset.epochs}
    if track_epochs:
        summary.update(epochs_completed=step//len(ranges), max_epochs=preset.epochs,
                       stop_reason='training_loss_plateau' if tracker['stopped'] else 'max_epochs',
                       convergence=tracker)
    atomic_json(folder/'summary.json', summary)
    hub.push(relative)
    del optimizer, x, y
    return model.eval().requires_grad_(False), summary


def update_plateau(tracker, loss, eligible, patience, delta):
    """Relative improvement over the last significant best; never uses evaluation labels."""
    if not math.isfinite(loss):
        raise FloatingPointError('nonfinite epoch loss')
    if not eligible:
        tracker['best'], tracker['bad_epochs'] = None, 0
        return
    best = tracker['best']
    if best is None or best-loss > delta*max(abs(best), 1e-12):
        tracker['best'], tracker['bad_epochs'] = loss, 0
    else:
        tracker['bad_epochs'] += 1
    tracker['stopped'] = tracker['bad_epochs'] >= patience
