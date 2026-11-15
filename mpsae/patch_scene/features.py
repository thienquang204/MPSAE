from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import math
import os

import numpy as np
import torch

from ..io import atomic_json, canonical_hash, read_json
from ..segmentation.core import label_distributions, sample_patches
from ..segmentation.data import ImageRows, collate
from ..segmentation.datasets import dataset_spec, load_splits, row_ids
from .config import DATASETS, ENCODERS, VERSION
from .encoders import load_encoder, patch_tokens


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.partial').open('wb') as handle:
        np.savez(handle, **arrays)
    os.replace(path.with_suffix('.partial'), path)


def storage_cast(values, dtype):
    result = values.astype(dtype)
    if not np.isfinite(result).all():
        raise ValueError('feature storage overflow; use cache_dtype=float32')
    error = float(np.sum((result.astype(np.float32)-values)**2, dtype=np.float64)
                  / max(float(np.sum(values**2, dtype=np.float64)), 1e-20))
    if error > 1e-5:
        raise ValueError('feature quantization error exceeds the declared tolerance')
    return result, error


def resolve_sources(config, hub):
    from huggingface_hub import HfApi
    relative = f'campaigns/{canonical_hash(config.identity())[:20]}/sources'
    folder = hub.pull(relative)
    sources = read_json(folder/'sources.json')
    original = f'campaigns/{canonical_hash(config.source_identity())[:20]}/sources'
    if sources is None:
        # An inference-only sweep must reuse the original pinned model revisions.
        sources = read_json(hub.pull(original)/'sources.json')
    sources = sources or {'models': {}, 'datasets': {}}
    missing_models = [e for e in config.encoder_names if e not in sources['models']]
    if 'nyuv2' in config.dataset_names:
        from .nyuv2 import load_local
        _, revision = load_local(fingerprint=True)
        previous = sources['datasets'].get('nyuv2')
        if previous is not None and previous != revision:
            raise RuntimeError('NYUv2 source changed; use a new workspace to avoid mixing results')
        sources['datasets']['nyuv2'] = revision
    if 'cityscapes' in config.dataset_names:
        if config.cityscapes_source == 'chris1':
            from .cityscapes_hf import REVISION
            revision = REVISION
        else:
            from .cityscapes import load_local
            _, revision = load_local(fingerprint=True)
        previous = sources['datasets'].get('cityscapes')
        if previous is not None and previous != revision:
            raise RuntimeError('Cityscapes source changed; use a new workspace to avoid mixing results')
        sources['datasets']['cityscapes'] = revision
    missing_data = [d for d in config.dataset_names if d not in sources['datasets']]
    if missing_models or missing_data:
        api = HfApi()
        for name in missing_models:
            sources['models'][name] = api.model_info(ENCODERS[name][0]).sha
        for name in missing_data:
            repo, revision, _ = DATASETS[name]
            sources['datasets'][name] = api.dataset_info(repo, revision=revision).sha
        atomic_json(hub.root/original/'sources.json', sources)
        hub.push(original)
    atomic_json(folder/'sources.json', sources)
    hub.push(relative)
    return sources


def feature_identity(config, encoder, dataset, sources, device):
    return {'version': VERSION, 'encoder': encoder, 'model': ENCODERS[encoder][0],
            'model_revision': sources['models'][encoder], 'dataset': dataset,
            'dataset_revision': sources['datasets'][dataset],
            'image_size': list(config.image_hw) if config.image_width else config.image_size, 'seed': config.seed,
            'train_patches': 'all_valid' if config.all_train_patches else config.train_patches, 'dtype': config.cache_dtype,
            'extraction_batch': config.extraction_batch, 'shard_images': config.shard_images,
            'precision': 'fp16_autocast' if str(device).startswith('cuda') else 'fp32',
            'geometry': ('whole_image_rectangular_bilinear;mask_nearest;no_crop' if config.image_width else
                         'whole_image_square_bilinear;mask_nearest;no_center_crop'),
            'tokens': 'last_layer_spatial_post_layernorm_no_cls_no_projection',
            'sampling': 'all_valid_patches' if config.all_train_patches else 'uniform_valid_patch_equal_image_quota;no_replacement',
            'versions': {p: importlib.metadata.version(p) for p in
                         ('torch', 'transformers', 'numpy', 'pillow', 'datasets')}}


def prepare(config, encoder, dataset, sources, hub, device):
    identity = feature_identity(config, encoder, dataset, sources, device)
    relative = f'features/{canonical_hash(identity)[:20]}'
    root = hub.pull(relative+'/index')
    manifest = read_json(root/'manifest.json')
    if manifest is not None:
        if manifest['identity'] != identity:
            raise RuntimeError('feature identity mismatch')
        return relative, manifest
    repo, _, subset = DATASETS[dataset]
    if dataset == 'nyuv2':
        from .nyuv2 import NYUSpec, load_local
        spec = NYUSpec()
        data, _ = load_local()
    elif dataset == 'cityscapes':
        if config.cityscapes_source == 'chris1':
            from .cityscapes_hf import HFCitySpec, load_hf
            spec = HFCitySpec()
            data = load_hf(sources['datasets'][dataset])
        else:
            from .cityscapes import CitySpec, load_local
            spec = CitySpec()
            data, _ = load_local()
    else:
        spec = dataset_spec(dataset, subset)
        args = SimpleNamespace(dataset=dataset, dataset_repo=repo, dataset_config=subset)
        data = load_splits(args, {'repo': repo, 'revision': sources['datasets'][dataset]},
                           Path(os.environ.get('HF_DATASETS_CACHE', str(hub.root/'hf_datasets'))))
    counts = {'train': spec.train_rows, 'validation': spec.validation_rows}
    ids = {}
    for split in counts:
        if len(data[split]) != counts[split]:
            raise ValueError(f'wrong {dataset} {split} row count')
        ids[split] = row_ids(data[split])
        if len(set(ids[split])) != len(ids[split]):
            raise ValueError('duplicate image IDs')
    if set(ids['train']) & set(ids['validation']):
        raise ValueError('train/validation image IDs overlap')
    from torch.utils.data import DataLoader
    model, norm = load_encoder(encoder, sources['models'][encoder], device)
    grid = tuple(v//ENCODERS[encoder][1] for v in config.image_hw)
    entries = []
    for split, count in counts.items():
        pending, plans, batches = [], [], []
        for start in range(0, count, config.shard_images):
            stop = min(count, start+config.shard_images)
            shard = f'{relative}/{split}/{start:06d}'
            folder = hub.pull(shard)
            if (folder/'data.npz').is_file() and (folder/'bundle.json').is_file():
                entries.append({'split': split, 'start': start, 'stop': stop, 'path': shard})
                continue
            offset = len(pending)
            pending.extend(range(start, stop))
            batches.extend([list(range(i, min(i+config.extraction_batch, offset+stop-start)))
                            for i in range(offset, offset+stop-start, config.extraction_batch)])
            plans.append((start, stop, shard, folder))
        # A single worker pool per split, with batches respecting shard boundaries.
        loader = DataLoader(ImageRows(data[split], pending, config.image_hw, norm, spec),
                            batch_sampler=batches, num_workers=config.workers, collate_fn=collate,
                            pin_memory=str(device).startswith('cuda'))
        stream = iter(loader)
        for start, stop, shard, folder in plans:
            features, labels, image_indices, patch_indices, truths, shapes = [], [], [], [], [], []
            max_error = 0.0
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16,
                                                        enabled=str(device).startswith('cuda')):
                for _ in range(math.ceil((stop-start)/config.extraction_batch)):
                    positions, images, original, masks = next(stream)
                    tokens = patch_tokens(model, images.to(device, non_blocking=True), encoder).cpu().numpy()
                    for i, x, truth, mask in zip(positions, tokens, original, masks):
                        if split == 'train':
                            soft = label_distributions(mask, grid, classes=spec.classes)
                            valid = np.flatnonzero(soft.sum(1) > 0)
                            quota = config.train_patches//count + (i < config.train_patches % count)
                            chosen = valid if config.all_train_patches else np.random.default_rng([config.seed, i]).permutation(valid)[:quota]
                            x = x[chosen]
                            labels.append(soft[chosen])
                            image_indices.append(np.full(len(chosen), i, dtype=np.int32))
                            patch_indices.append(chosen.astype(np.int32))
                        else:
                            truths.append(truth.astype(np.uint8).ravel())
                            shapes.append(truth.shape)
                        x, error = storage_cast(x, config.cache_dtype)
                        max_error = max(max_error, error)
                        features.append(x)
            if split == 'train':
                save_npz(folder/'data.npz', features=np.concatenate(features),
                         labels=np.concatenate(labels), images=np.concatenate(image_indices),
                         patches=np.concatenate(patch_indices))
            else:
                offsets = np.cumsum([0]+[len(t) for t in truths])
                save_npz(folder/'data.npz', features=np.stack(features), truth=np.concatenate(truths),
                         offsets=offsets, shapes=np.asarray(shapes, dtype=np.int32))
            atomic_json(folder/'metadata.json', {'identity': identity, 'split': split,
                'start': start, 'stop': stop, 'ids': ids[split][start:stop],
                'training_rows': sum(len(x) for x in features) if split == 'train' else None,
                'max_storage_relative_squared_error': max_error})
            hub.push(shard)  # interruption loses at most one extraction shard
            entries.append({'split': split, 'start': start, 'stop': stop, 'path': shard})
            print(f'[extract {dataset}/{encoder}] {split} {stop}/{count}', flush=True)
    del model, data
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    entries.sort(key=lambda entry: (entry['split'], entry['start']))
    manifest = {'identity': identity, 'grid': grid, 'classes': spec.classes,
                'counts': counts, 'shards': entries, 'complete': True}
    atomic_json(root/'manifest.json', manifest)
    hub.push(relative+'/index')
    return relative, manifest


def training_bank(manifest, hub, config):
    if config.all_train_patches:
        from .stream import support_bank
        return support_bank(manifest, hub, config)
    xs, ys = [], []
    for entry in manifest['shards']:
        if entry['split'] == 'train':
            folder = hub.pull(entry['path'])
            with np.load(folder/'data.npz', allow_pickle=False) as data:
                xs.append(data['features'])
                ys.append(data['labels'])
    x, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys)
    if len(x) < config.neighbors or not np.isfinite(x).all():
        raise ValueError('insufficient/nonfinite training patches')
    selected = sample_patches(y, config.support_patches, np.random.default_rng(config.seed))
    return x, y, selected


def queries(manifest, hub, *, skip=0):
    for entry in manifest['shards']:
        if entry['split'] != 'validation' or entry['stop'] <= skip:
            continue
        folder = hub.pull(entry['path'])
        with np.load(folder/'data.npz', allow_pickle=False) as data:
            xs, truths, offsets, shapes = data['features'], data['truth'], data['offsets'], data['shapes']
        for j in range(max(0, skip-entry['start']), len(xs)):
            yield entry['start']+j, xs[j].astype(np.float32), truths[offsets[j]:offsets[j+1]].reshape(shapes[j])
