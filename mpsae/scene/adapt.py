"""SAE-only adaptation on cached training patches; no image/backbone/validation access."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import numpy as np
import torch

from .. import methods
from ..config import BACKBONES
from ..evaluate.knn import fp32_matmul
from ..io import (append_jsonl, atomic_json, canonical_hash, exclusive_lock,
                  read_json, read_jsonl, sha256_file, truncate_jsonl)
from ..train import load_fitted

VERSION = 'voc-sae-adaptation-v1'


def source_preset(cell, record, settings):
    """Read objective/Adam parameters from the actual source, not today's defaults."""
    identity = record['training_identity']
    objective, optimizer = identity['objective'], identity['optimizer']
    if optimizer['name'] != 'Adam' or objective['reconstruction'] != ['L(k)', 'L(4k)/8']:
        raise ValueError('unsupported source optimizer or reconstruction recipe')
    aux = objective['aux']
    values = dict(hidden_dim=objective['hidden_dim'], k_aux=aux['k_aux'],
        aux_coef=aux['coef'], dead_threshold=aux['dead_threshold'],
        nested_coef=objective.get('nested_coef', 0.125), lr=settings.lr,
        epochs=settings.epochs, batch_size=settings.batch_size,
        adam_betas=tuple(optimizer['betas']), adam_eps=optimizer['eps'],
        weight_decay=optimizer['weight_decay'])
    if cell.method in methods.CL_METHODS:
        cl = objective['contrastive']
        expected = 'label_based_cl' if cell.method == 'csrv2' else 'inbatch_cl'
        if cl['loss'] != expected:
            raise ValueError('source contrastive objective mismatch')
        values.update(cl_coef=cl['weight'], cl_temperature=cl['temperature'])
    if cell.method in methods.OT_METHODS:
        ot = objective['ot']
        if ot['cost'] != 'angular' or ot['views'] != ['k', '2k', '4k']:
            raise ValueError('source OT objective mismatch')
        values.update(ot_weight=ot['weight'], ot_mass=ot['mass'], ot_eta=ot['eta'],
            ot_microbatch=ot['microbatch'], ot_max_iters=ot['max_iters'],
            ot_tol=ot['tol'], ot_max_refine_iters=ot.get('max_refine_iters', 2000))
    return replace(BACKBONES[cell.backbone].preset, **values)


def batch_ranges(rows, batch_size):
    if rows < 2:
        raise ValueError('SAE adaptation needs at least two training patches')
    ends = list(range(batch_size, rows, batch_size)) + [rows]
    if len(ends) > 1 and ends[-1] - ends[-2] == 1:
        ends.pop(-2)  # merge a singleton into the preceding batch, never drop a patch
    return list(zip([0] + ends[:-1], ends))


def patch_targets(labels):
    labels = np.asarray(labels)
    if (labels.ndim != 2 or labels.shape[1] != 21 or not np.isfinite(labels).all()
            or np.any(labels < 0) or not np.allclose(labels.sum(1), 1, atol=1e-5)):
        raise ValueError('training patches need valid nonvoid VOC label distributions')
    target = labels.argmax(1).astype(np.int64)
    purity = labels.max(1)
    return target, {'label_rule': 'dominant nonvoid class; ties choose lowest class ID',
        'class_counts': np.bincount(target, minlength=21).tolist(),
        'mean_purity': float(purity.mean()), 'mixed_fraction': float((purity < 1-1e-6).mean()),
        'background_fraction': float((target == 0).mean())}


def _save(path, payload):
    temporary = path.with_suffix('.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def _checked_checkpoint(path, identity, signature, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if (saved.get('identity') != identity or saved.get('signature') != signature
            or canonical_hash(saved['identity']) != signature):
        raise RuntimeError('adaptation checkpoint recipe mismatch')
    return saved


def fit(config, cell, record, features, labels, train_identity, workspace, *, device):
    """A separate fit identity; initializing weights never resumes the ImageNet optimizer."""
    settings = config.adaptation
    if not settings.enabled:
        raise ValueError('adaptation is disabled')
    target, label_stats = patch_targets(labels)
    if (features.dtype != np.float32 or features.ndim != 2 or len(features) != len(target)
            or not np.isfinite(features).all()):
        raise ValueError('adaptation needs finite float32 training patch features')
    preset = source_preset(cell, record, settings)
    ranges = batch_ranges(len(features), settings.batch_size)
    total = len(ranges) * settings.epochs
    objective = methods.describe(cell.method, cell.k, preset)
    objective.pop('annealing', None)
    objective['budget_policy'] = 'fixed target K; no restart of pretraining annealing'
    identity = {'version': VERSION, 'cell': asdict(cell),
        'source_checkpoint_id': record['checkpoint_id'], 'source_signature': record['signature'],
        'training_data': train_identity, 'rows': len(features), 'objective': objective,
        'preset': asdict(preset), 'batch_policy': 'shuffle all patches; merge final singleton',
        'initialization': 'source SAE weights; fresh Adam; reset activity counters',
        'seed': config.seed, 'total_steps': total, 'selection': 'final epoch',
        'label_policy': label_stats['label_rule'] if cell.method == 'csrv2' else 'not used by loss',
        'precision': 'float32_tf32_disabled'}
    identity = json.loads(json.dumps(identity))  # identical tuples/lists after JSON resume
    signature = canonical_hash(identity)
    directory = Path(workspace) / 'adaptations' / signature[:20]
    with exclusive_lock(directory / 'fit.lock', timeout=0):
        return _fit(directory, config, cell, record, features, target, label_stats,
                    preset, identity, signature, ranges, device)


def _fit(directory, config, cell, record, features, target, label_stats,
         preset, identity, signature, ranges, device):
    device = torch.device(device)
    settings, total = config.adaptation, identity['total_steps']
    checkpoint = directory / 'checkpoint.pt'
    completed = read_json(directory / 'summary.json')
    if completed and completed.get('complete'):
        if (completed.get('signature') != signature or completed.get('identity') != identity
                or completed.get('steps') != total or not checkpoint.is_file()
                or completed.get('checkpoint_id') != sha256_file(checkpoint)):
            raise RuntimeError(f'inconsistent completed adaptation: {directory}')
        return {**completed, 'path': str(checkpoint)}
    saved = _checked_checkpoint(checkpoint, identity, signature, device) if checkpoint.exists() else None
    model = load_fitted(cell, Path(record['path']).parent, BACKBONES[cell.backbone], device=device)
    model.train().requires_grad_(True)
    if model.hidden_dim != preset.hidden_dim or model.input_dim != features.shape[1]:
        raise ValueError('source SAE dimensions do not match patch features')
    optimizer = torch.optim.Adam(model.parameters(), lr=preset.lr, betas=preset.adam_betas,
                                 eps=preset.adam_eps, weight_decay=preset.weight_decay)
    step, elapsed, last, prior_peak = 0, 0.0, {}, 0
    if saved:
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        step, elapsed, last = saved['next_step'], saved['train_seconds'], saved['last_metrics']
        prior_peak = saved.get('peak_memory_bytes') or 0
        if type(step) is not int or not 0 <= step <= total:
            raise RuntimeError('invalid adaptation resume position')
    else:
        model.stats_last_nonzero.zero_()
    atomic_json(directory / 'config.json', {'identity': identity, 'signature': signature,
        'backbone_weights': record['backbone_weights'], 'patch_labels': label_stats})
    log_path = directory / 'train.jsonl'
    records = read_jsonl(log_path)
    truncate_jsonl(log_path, sum(r['step'] <= step for r in records))
    # <=0.5 GiB for the default ResNet bank. Kept once per fit; no loader workers/images.
    residence = device if settings.residency == 'device' else torch.device('cpu')
    x = torch.as_tensor(features, device=residence)
    y = torch.as_tensor(target, device=residence) if cell.method == 'csrv2' else None
    print(f'[scene adapt] {cell.backbone}/{cell.method}: {step}/{total} updates; '
          f'K={cell.k}, batch={preset.batch_size}, lr={preset.lr:g}', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    def save():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        _save(checkpoint, {'identity': identity, 'signature': signature, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'next_step': step, 'last_metrics': last,
            'peak_memory_bytes': max(prior_peak, torch.cuda.max_memory_allocated(device))
                                 if device.type == 'cuda' else None,
            'train_seconds': elapsed + time.perf_counter() - start})

    with fp32_matmul(), torch.enable_grad():
        while step < total:
            epoch, within = divmod(step, len(ranges))
            # Epoch-local RNG makes resume and method order independent; objective is deterministic.
            generator = torch.Generator(device='cpu').manual_seed(config.seed + epoch)
            order = torch.randperm(len(x), generator=generator).to(residence)
            for b in range(within, len(ranges)):
                lo, hi = ranges[b]
                indices = order[lo:hi]
                batch = x[indices].to(device)
                targets = y[indices].to(device) if y is not None else None
                optimizer.zero_grad(set_to_none=True)
                collect = (step == 0 or (step+1) % settings.log_every == 0 or step+1 == total)
                output = methods.compute_step(cell.method, model, batch, k=cell.k, preset=preset,
                    step=step, total_steps=total, labels=targets, fixed_budget=True, collect_metrics=collect)
                output.loss.backward()
                if not bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                                         if p.grad is not None]).all()):
                    raise FloatingPointError('nonfinite SAE adaptation gradients; update refused')
                optimizer.step()
                step += 1
                if collect:
                    last = output.metrics
                    append_jsonl(log_path, {**last, 'step': step, 'epoch': epoch+1,
                        'rows': len(batch), 'lr': preset.lr,
                        'elapsed': elapsed + time.perf_counter()-start})
                    print(f'[scene adapt] {cell.method}: {step}/{total} '
                          f'loss={last["loss_total"]:.5f} nnz={last["nnz_mean"]:.2f}', flush=True)
                if step % settings.checkpoint_every == 0 or b == len(ranges)-1:
                    save()
    checkpoint_id = sha256_file(checkpoint)
    summary = {'complete': True, 'identity': identity, 'signature': signature,
        'checkpoint_id': checkpoint_id, 'steps': step, 'train_seconds': elapsed+time.perf_counter()-start,
        'final_metrics': last, 'patch_labels': label_stats,
        'peak_memory_bytes': max(prior_peak, torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None,
        'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                    'device': str(device), 'residency': settings.residency}}
    atomic_json(directory / 'summary.json', summary)
    return {**summary, 'path': str(checkpoint)}


def load_adapted(record, *, device):
    from ..model import SparseAutoencoder
    path = Path(record['path'])
    if sha256_file(path) != record['checkpoint_id']:
        raise RuntimeError('adapted checkpoint bytes changed')
    saved = _checked_checkpoint(path, record['identity'], record['signature'], device)
    if saved['next_step'] != record['identity']['total_steps']:
        raise RuntimeError('adaptation is incomplete')
    state = saved['model']
    h, d = state['encoder.weight'].shape
    model = SparseAutoencoder(d, h, dead_threshold=record['identity']['preset']['dead_threshold']).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
