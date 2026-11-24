"""Frozen patch SAE linear segmentation, isolated from nearest-neighbor evaluation."""
from dataclasses import asdict, dataclass, replace
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..io import atomic_json, canonical_hash, read_json, array_hash, sha256_file, exclusive_lock
from ..evaluate.knn import fp32_matmul
from .artifacts import Hub, save_torch
from .config import Config, require_supported

VERSION = 'frozen-linear-seg-v1'


@dataclass(frozen=True)
class ProbeConfig:
    epochs: int = 20
    batch_size: int = 4096
    lr: float = .001
    weight_decay: float = 0.
    encode_batch: int = 1024
    seed: int = 1227
    residency: str = 'auto'

    def __post_init__(self):
        for key in ('epochs', 'batch_size', 'encode_batch', 'seed'):
            if type(getattr(self, key)) is not int or getattr(self, key) < (0 if key == 'seed' else 1):
                raise ValueError(f'invalid probe {key}')
        for key in ('lr', 'weight_decay'):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key == 'lr' and value == 0):
                raise ValueError(f'invalid probe {key}')
        if self.residency not in ('auto', 'host', 'device'):
            raise ValueError('probe residency must be auto, host or device')


def load(path):
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if set(data) != {'scene', 'probe'}:
        raise ValueError('linear config requires scene and probe mappings')
    config, probe = Config(**data['scene']), ProbeConfig(**data['probe'])
    require_supported(config)
    if config.all_train_patches:
        raise ValueError('This linear protocol uses a declared cached training-patch pool; all_train_patches is unsupported')
    if config.evaluation_mode != 'matched':
        raise ValueError('Linear study trains and evaluates each SAE at matched K')
    return config, probe


def plan_for(config, probe):
    cells = [{'dataset': d, 'encoder': e, 'method': method, 'train_k': k, 'inference_k': k}
             for d in config.dataset_names for e in config.encoder_names
             for method, k in ([('dense', None)] if config.include_dense else []) +
             [(m, k) for m in config.methods for k in config.training_ks]]
    identity = {'protocol': VERSION, 'scene': config.identity(), 'probe': asdict(probe)}
    return {**identity, 'campaign_id': canonical_hash(identity)[:20], 'cells': cells,
            'evaluations': len(cells), 'probe_fits': len(cells),
            'max_sae_fits': sum(c['method'] != 'dense' for c in cells),
            'readout': 'affine on L2-normalized features; soft-label patch cross entropy; bilinear logits',
            'training_data': 'all rows of declared training patch pool, not the retrieval support bank',
            'selection': 'final epoch; no validation-based model/hyperparameter selection'}


def logits(head, values, indices=None):
    """Exactly Wz+b without materializing [batch,8192] sparse feature matrices."""
    if indices is None:
        return head(values)
    weights = F.embedding(indices.long(), head.weight.T)  # batch x K x classes
    return (weights * values.unsqueeze(-1)).sum(1) + head.bias


@torch.no_grad()
def encode_batch(model, features, k, device):
    x = torch.tensor(np.asarray(features), dtype=torch.float32, device=device)
    if model is None:
        return F.normalize(x, dim=-1), None
    top = model.pre_activations(x).topk(k, dim=-1)
    values = top.values.relu()
    if not torch.isfinite(values).all():
        raise FloatingPointError('nonfinite sparse features')
    return F.normalize(values, dim=-1), top.indices


def training_codes(features, identity, model, k, folder, device, batch):
    """Persistent, atomic compact codes. Memory-map one representation at a time."""
    folder.mkdir(parents=True, exist_ok=True)
    receipt = read_json(folder/'codes.json')
    if receipt:
        if receipt['identity'] != identity:
            raise RuntimeError('code-cache identity mismatch')
        for name, digest in receipt['files'].items():
            if sha256_file(folder/name) != digest:
                raise RuntimeError('corrupt probe code cache')
    else:
        width = features.shape[1] if model is None else k
        val = np.lib.format.open_memmap(folder/'values.partial.npy', mode='w+', dtype='float32', shape=(len(features), width))
        idx = None if model is None else np.lib.format.open_memmap(folder/'indices.partial.npy', mode='w+', dtype='int32', shape=val.shape)
        with fp32_matmul():
            for start in range(0, len(features), batch):
                values, indices = encode_batch(model, features[start:start+batch], k, device)
                val[start:start+len(values)] = values.cpu().numpy()
                if idx is not None:
                    idx[start:start+len(values)] = indices.cpu().numpy()
        val.flush(); del val
        names = ['values.npy']
        if idx is not None:
            idx.flush(); del idx
            names.append('indices.npy')
        for name in names:
            os.replace(folder/name.replace('.npy', '.partial.npy'), folder/name)
        atomic_json(folder/'codes.json', {'identity': identity, 'files': {n: sha256_file(folder/n) for n in names}})
    return (np.load(folder/'values.npy', mmap_mode='r'),
            np.load(folder/'indices.npy', mmap_mode='r') if model is not None else None)


def train_head(values, indices, targets, dim, identity, config, folder, device):
    if len(targets) != len(values) or not np.isfinite(targets).all() or np.any(targets < 0) or not np.allclose(targets.sum(1), 1, atol=1e-5):
        raise ValueError('probe requires nonvoid soft class distributions for every training row')
    head = nn.Linear(dim, targets.shape[1]).to(device)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    path = folder/'head.pt'
    epoch, history = 0, []
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved['identity'] != identity:
            raise RuntimeError('linear probe checkpoint mismatch')
        head.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        epoch, history = saved['next_epoch'], saved['history']
    if epoch < config.epochs:
        size = values.nbytes + targets.nbytes + (indices.nbytes if indices is not None else 0)
        resident = config.residency == 'device' or (config.residency == 'auto' and
            str(device).startswith('cuda') and size + (2 << 30) < torch.cuda.mem_get_info(device)[0])
        bank = [values, targets, indices]
        if resident:
            bank = [torch.tensor(np.asarray(a), device=device) if a is not None else None for a in bank]
        print(f'[linear probe] {len(values):,} patches; {"device" if resident else "host"} bank; {dim} inputs', flush=True)
        with fp32_matmul():
            for epoch in range(epoch, config.epochs):
                order = np.random.default_rng([config.seed, epoch]).permutation(len(values))
                loss_sum = torch.zeros((), device=device)
                for start in range(0, len(values), config.batch_size):
                    selection = order[start:start+config.batch_size]
                    if resident:
                        selection = torch.as_tensor(selection, device=device)
                    batch = []
                    for a in bank:
                        batch.append(None if a is None else a[selection] if resident else
                                     torch.tensor(a[selection], device=device))
                    v, y, idx = batch
                    optimizer.zero_grad(set_to_none=True)
                    loss = -(y * F.log_softmax(logits(head, v, idx), dim=-1)).sum(1).mean()
                    loss.backward(); optimizer.step()
                    loss_sum += loss.detach() * len(v)
                mean_loss = float(loss_sum / len(values))
                if not math.isfinite(mean_loss) or any(not torch.isfinite(p).all() for p in head.parameters()):
                    raise FloatingPointError('nonfinite probe optimization')
                history.append({'epoch': epoch+1, 'loss': mean_loss})
                save_torch(path, {'identity': identity, 'model': head.state_dict(), 'optimizer': optimizer.state_dict(),
                    'next_epoch': epoch+1, 'history': history, 'complete': epoch+1 == config.epochs})
                print(f'[linear probe] epoch {epoch+1}/{config.epochs} loss={mean_loss:.6f}', flush=True)
        del bank
    return head.eval().requires_grad_(False), {'history': history, 'checkpoint_sha256': sha256_file(path),
                                            'head_parameters': sum(p.numel() for p in head.parameters())}


@torch.no_grad()
def evaluate_head(config, model, head, k, manifest, hub, device, identity, folder):
    from .features import queries
    from ..segmentation.core import confusion, metrics
    completed = read_json(folder/'result.json')
    if completed and completed.get('identity') == identity and completed.get('complete'):
        return completed
    progress = read_json(folder/'progress.json') or {}
    if progress and progress['identity'] != identity:
        raise RuntimeError('linear evaluation resume mismatch')
    classes = manifest['classes']
    matrix = np.asarray(progress.get('confusion', np.zeros((classes, classes+1))), dtype=np.int64)
    next_image = progress.get('next_image', 0)
    patches, nnz_sum, empties = (progress.get(key, 0) for key in ('patches', 'nnz_sum', 'empty_patches'))
    with fp32_matmul():
        for i, features, truth in queries(manifest, hub, skip=next_image):
            values, indices = encode_batch(model, features, k, device)
            counts = values.count_nonzero(dim=1)
            if model is not None and bool((counts > k).any()):
                raise AssertionError('K-sparsity violated')
            nnz_sum += int(counts.sum()); empties += int((counts == 0).sum()); patches += len(values)
            scores = logits(head, values, indices).T.reshape(1, classes, *manifest['grid'])
            prediction = F.interpolate(scores, size=truth.shape, mode='bilinear', align_corners=False).argmax(1)[0]
            matrix += confusion(prediction.cpu().numpy(), truth, classes=classes)
            next_image = i+1
            if next_image % config.shard_images == 0 or next_image == manifest['counts']['validation']:
                atomic_json(folder/'progress.json', {'identity': identity, 'next_image': next_image,
                    'confusion': matrix.tolist(), 'patches': patches, 'nnz_sum': nnz_sum, 'empty_patches': empties})
                print(f'[linear eval] {next_image}/{manifest["counts"]["validation"]}', flush=True)
    if next_image != manifest['counts']['validation'] or patches == 0:
        raise RuntimeError('incomplete linear segmentation evaluation')
    result = {'identity': identity, 'complete': True, **metrics(matrix, include_absent=manifest['identity']['dataset'] == 'nyuv2'),
              'confusion': matrix.tolist(), 'mean_nnz': nnz_sum/patches, 'empty_patches': empties,
              'patches': patches, 'images': next_image,
              'empty_policy': 'standard affine classifier: zero features predict learned bias; counted explicitly'}
    atomic_json(folder/'result.json', result)
    return result


def run(config, probe, workspace, device, *, smoke=False, export_only=False):
    from .linear_report import export
    plan = {**plan_for(config, probe), 'scientific': not smoke}
    root = Path(workspace)/('SMOKE_ONLY' if smoke else '')
    folder = root/'linear-studies'/plan['campaign_id']
    with exclusive_lock(root/'.linear-probe.lock', timeout=0):
        atomic_json(folder/'plan.json', plan)
        prior = read_json(folder/'results.json') or {}
        def key(c):
            return c['dataset'], c['encoder'], c['method'], c['train_k']
        previous = {key(r): r for r in prior.get('results', [])}
        rows = [{**c, **previous.get(key(c), {}), 'status': previous.get(key(c), {}).get('status', 'pending')} for c in plan['cells']]
        if export_only:
            return export(root, folder, plan, rows)
        hub = Hub(root)  # Always local; no uploads or HF artifact dependency.
        from .features import resolve_sources, prepare, training_bank
        from .train import fit
        from .smoke import fixture
        state = {'status': 'running'}
        atomic_json(folder/'run-state.json', state)
        try:
            sources = {} if smoke else resolve_sources(config, hub)
            for dataset in config.dataset_names:
                for encoder in config.encoder_names:
                    if smoke:
                        manifest, features, labels, _ = fixture(hub, config, encoder, dataset)
                    else:
                        _, manifest = prepare(config, encoder, dataset, sources, hub, device)
                        features, labels, _ = training_bank(manifest, hub, config)
                    hashes = {'features': array_hash(features), 'targets': array_hash(labels)}
                    for row in rows:
                        if (row['dataset'], row['encoder']) != (dataset, encoder):
                            continue
                        model = head = values = indices = None
                        row['status'] = 'running'
                        export(root, folder, plan, rows)
                        try:
                            k, method = row['train_k'], row['method']
                            train_config = config.for_training(k) if k is not None else config
                            fit_record = None
                            if method != 'dense':
                                preset = replace(train_config.preset(method, encoder), hidden_dim=64,
                                    epochs=1, batch_size=16, k_aux=32, ot_microbatch=4) if smoke else None
                                model, fit_record = fit(train_config, method, features, labels, manifest['identity'], hub, device, preset=preset)
                            code_identity = {'protocol': VERSION, 'features': manifest['identity'], **hashes,
                                'sae': fit_record['checkpoint_sha256'] if fit_record else None, 'k': k,
                                'normalization': 'l2-eps1e-12', 'precision': 'fp32-tf32-disabled'}
                            code_folder = root/'linear-codes'/canonical_hash(code_identity)[:20]
                            values, indices = training_codes(features, code_identity, model, k, code_folder, device, probe.encode_batch)
                            effective = replace(probe, epochs=2, batch_size=16) if smoke else probe
                            identity = {'codes': code_identity, 'probe': asdict(effective), 'classes': labels.shape[1],
                                        'initialization': 'zero weights/bias', 'optimizer': 'Adam(.9,.999,eps=1e-8)',
                                        'objective': 'mean soft-target patch CE; no class reweighting'}
                            head_folder = root/'linear-heads'/canonical_hash(identity)[:20]
                            dim = model.hidden_dim if model is not None else features.shape[1]
                            head, record = train_head(values, indices, labels, dim, identity, effective, head_folder, device)
                            eval_identity = {'protocol': VERSION, 'head': record['checkpoint_sha256'],
                                'training': identity, 'features': manifest['identity'], 'upsample': 'bilinear_logits_align_corners_false'}
                            eval_folder = root/'linear-evaluations'/canonical_hash(eval_identity)[:20]
                            result = evaluate_head(config, model, head, k, manifest, hub, device, eval_identity, eval_folder)
                            row.update(status='complete', **result, **record,
                                training_rows=len(features), sae_checkpoint=fit_record['path'] if fit_record else None)
                            row.pop('error', None)
                        except BaseException as error:
                            row.update(status='failed', error=f'{type(error).__name__}: {error}')
                            raise
                        finally:
                            for a in (values, indices):
                                if a is not None and getattr(a, '_mmap', None) is not None:
                                    a._mmap.close()
                            del model, head, values, indices
                            export(root, folder, plan, rows)
                    del features, labels
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if state['status'] == 'running':
                state['status'] = 'finished'
            atomic_json(folder/'run-state.json', state)
            result = export(root, folder, plan, rows)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/linear_scene_voc.yaml')
    parser.add_argument('--workspace', default='work/linear-scene')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args(argv)
    config, probe = load(args.config)
    if args.plan:
        print(json.dumps(plan_for(config, probe), indent=2)); return 0
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    result = run(config, probe, args.workspace, args.device, smoke=args.smoke, export_only=args.export_only)
    return 0 if result['complete'] or args.export_only else 1


if __name__ == '__main__':
    raise SystemExit(main())
