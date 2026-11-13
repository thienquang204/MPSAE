"""Frozen or patch-adapted SAEs evaluated with the same retrieval decoder."""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from ..benchmark.run import checkpoint_record
from ..config import BACKBONES
from ..io import atomic_json, canonical_hash, read_json, array_hash
from ..train import load_fitted
from .config import VERSION, Adaptation
from .core import CLASSES, confusion, encode, metrics, predict
from .data import prepare, support_bank


def source_records(config, source_root):
    if hasattr(config, 'groups'):
        return [record for group in config.groups for record in source_records(group, source_root)]
    records = [checkpoint_record(cell, Path(source_root)) for cell in config.cells()]
    if any(r['backbone_weights'] != records[0]['backbone_weights'] for r in records):
        raise ValueError('all compared methods must use the exact same backbone weights')
    return records


def run(config, workspace, source_root, *, device='cuda:0', prepare_only=False, on_adapted=None):
    config.plan()
    records = source_records(config, source_root)
    directory, manifest = prepare(config, workspace, records[0]['backbone_weights'], device=device)
    if prepare_only:
        return {'feature_cache': str(directory), 'complete': True}
    features, labels = support_bank(directory, manifest)
    if len(features) < config.neighbors:
        raise ValueError('too few valid support patches for configured neighbor count')
    adapted = []
    frozen_rows = []
    if config.adaptation.enabled:
        from .adapt import fit
        # Only training rows reach fit. Validation arrays are opened by evaluation below.
        train_identity = {'feature_cache': manifest['identity'],
                          'features_sha256': array_hash(features), 'labels_sha256': array_hash(labels),
                          'sampling': 'shared VOC train support; label-informed; all-void excluded'}
        for cell, record in zip(config.cells(), records):
            fitted = fit(config, cell, record, features, labels, train_identity, workspace, device=device)
            adapted.append(fitted)
            if on_adapted is not None:
                on_adapted(fitted)  # back up each completed fit before the expensive sweep
        frozen = replace(config, adaptation=Adaptation())
        baseline = _evaluate(frozen, workspace, records, directory, manifest, features, labels,
                             device=device, dense_only=not config.adaptation.compare_frozen)
        frozen_rows = [{**row, 'variant': 'dense' if row['method'] == 'dense' else 'frozen',
                        'query_images': row.get('query_images', manifest['query_images'])}
                       for row in baseline['results']]
    return _evaluate(config, workspace, records, directory, manifest, features, labels,
                     device=device, adapted=adapted, frozen_rows=frozen_rows)


def _evaluate(config, workspace, records, directory, manifest, features, labels, *,
              device, adapted=(), frozen_rows=(), dense_only=False):
    campaign = {'version': VERSION, 'plan': config.plan(), 'features': manifest['identity'],
                'sources': [{k: r[k] for k in ('checkpoint_id', 'signature', 'cell')} for r in records],
                'device': str(device), 'sae_precision': 'float32_tf32_disabled',
                'search': 'exact_cosine_score_desc_gallery_position_asc',
                'empty_code_policy': 'abstain_nearest_grid_mask_count_false_negative'}
    if adapted:
        campaign.update(version='voc-spatial-adapted-v1',
            adaptations=[{'checkpoint_id': r['checkpoint_id'], 'signature': r['signature'],
                          'identity': r['identity']} for r in adapted])
    if dense_only:
        campaign['dense_only'] = True
    output = Path(workspace) / 'reports' / canonical_hash(campaign)[:20]
    atomic_json(output / 'manifest.json', campaign)
    results = list(frozen_rows)
    model_records = adapted or records
    points = ([] if frozen_rows else [(None, None, None)]) + [(cell, record, k) for cell, record in zip(config.cells(), model_records)
                                     for k in config.inference_k]
    if dense_only:
        points = [(None, None, None)]
    from ..evaluate.knn import fp32_matmul
    loaded_key, loaded_sae = None, None
    with torch.inference_mode(), fp32_matmul():
        for cell, record, k in points:
            method = cell.method if cell else 'dense'
            variant = 'adapted' if adapted and cell else 'frozen' if cell else 'dense'
            slug = f'{method}_k{k}' if cell else 'dense'
            point_path = output / f'{slug}.json'
            previous = read_json(point_path)
            if previous and previous.get('complete'):
                results.append(previous)
                continue
            key = (variant, method, record['checkpoint_id']) if cell else None
            if key != loaded_key:
                if adapted:
                    from .adapt import load_adapted
                    loaded_sae = load_adapted(record, device=torch.device(device))
                else:
                    loaded_sae = load_fitted(cell, Path(record['path']).parent, BACKBONES[cell.backbone],
                                            device=torch.device(device)).requires_grad_(False)
                loaded_key = key
            sae = loaded_sae if cell else None
            dim = sae.hidden_dim if sae else features.shape[1]
            bank = encode(sae, features, k, device, config.encode_batch) if sae else features
            values = bank['values'] if sae else bank
            bank_nnz = np.count_nonzero(values, axis=1)
            matrix = np.zeros((CLASSES, CLASSES+1), dtype=np.int64)
            nnz_count = nnz_total = empty_count = 0
            nnz_max = 0
            seconds = 0.0
            start = time.perf_counter()
            for index in range(manifest['query_images']):
                # Per-image scoring makes interrupted evaluation resumable.
                saved_path = output / 'per_image' / slug / f'{index:05d}.json'
                saved = read_json(saved_path)
                if saved is None:
                    with np.load(directory / 'validation' / f'{index:05d}.npz', allow_pickle=False) as row:
                        raw, truth = row['features'], row['truth']
                    query = encode(sae, raw, k, device, config.encode_batch) if sae else raw
                    prediction, timing = predict(bank, labels, query, dim=dim, grid=manifest['grid'],
                        output_size=truth.shape, device=device, neighbors=config.neighbors,
                        temperature=config.temperature, gallery_chunk=config.gallery_chunk,
                        query_block=config.query_block)
                    qvalues = query['values'] if sae else query
                    nnz = np.count_nonzero(qvalues, axis=1)
                    saved = {'confusion': confusion(prediction, truth).tolist(), **timing,
                             'nnz_sum': int(nnz.sum()), 'patches': len(nnz), 'nnz_max': int(nnz.max())}
                    atomic_json(saved_path, saved)
                matrix += np.asarray(saved['confusion'], dtype=np.int64)
                nnz_total += saved['nnz_sum']
                nnz_count += saved['patches']
                nnz_max = max(nnz_max, saved['nnz_max'])
                empty_count += saved['empty_query_patches']
                seconds += saved['search_seconds']
                if (index+1) % 25 == 0:
                    print(f'[scene eval] {slug}: {index+1}/{manifest["query_images"]}', flush=True)
            if sae and (bank_nnz.max() > k or nnz_max > k):
                raise AssertionError('SAE sparsity budget violated')
            record_out = {'complete': True, 'method': method, 'train_k': cell.k if cell else None,
                          'variant': variant,
                          'inference_k': k, 'checkpoint_id': record['checkpoint_id'] if cell else None,
                          **metrics(matrix), 'confusion': matrix.tolist(),
                          'bank_rows': len(features), 'bank_nnz_mean': float(bank_nnz.mean()),
                          'bank_empty_patches': int(np.count_nonzero(bank_nnz == 0)),
                          'query_nnz_mean': nnz_total/nnz_count, 'query_nnz_max': nnz_max,
                          'query_empty_patches': empty_count, 'query_patches': nnz_count,
                          'query_images': manifest['query_images'],
                          'search_seconds': seconds, 'invocation_seconds': time.perf_counter()-start,
                          'timing_note': 'search_seconds includes cached per-image measurements; not end-to-end latency',
                          'index_bytes': int(bank['indices'].nbytes + bank['values'].nbytes) if sae else int(bank.nbytes)}
            atomic_json(point_path, record_out)
            results.append(record_out)
            del bank
    artifacts = [{'signature': r['signature'], 'checkpoint_id': r['checkpoint_id'],
                  'directory': f'adaptations/{r["signature"][:20]}'} for r in adapted]
    atomic_json(output / 'results.json', {'complete': True, 'campaign': campaign,
                                         'adaptations': artifacts, 'results': results})
    from .report import plot
    plot(output / 'results.json')
    return {'complete': True, 'report': str(output), 'results': results}
