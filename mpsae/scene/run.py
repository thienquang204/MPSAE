"""Evaluate completed source fits. No fitting or ImageNet extraction paths."""
from pathlib import Path
import time

import numpy as np
import torch

from ..benchmark.run import checkpoint_record
from ..config import BACKBONES
from ..io import atomic_json, canonical_hash, read_json
from ..train import load_fitted
from .config import VERSION
from .core import CLASSES, confusion, encode, metrics, predict
from .data import prepare, support_bank


def source_records(config, source_root):
    records = [checkpoint_record(cell, Path(source_root)) for cell in config.cells()]
    if any(r['backbone_weights'] != records[0]['backbone_weights'] for r in records):
        raise ValueError('all compared methods must use the exact same backbone weights')
    return records


def run(config, workspace, source_root, *, device='cuda:0', prepare_only=False):
    config.plan()
    records = source_records(config, source_root)
    directory, manifest = prepare(config, workspace, records[0]['backbone_weights'], device=device)
    if prepare_only:
        return {'feature_cache': str(directory), 'complete': True}
    features, labels = support_bank(directory, manifest)
    if len(features) < config.neighbors:
        raise ValueError('too few valid support patches for configured neighbor count')
    campaign = {'version': VERSION, 'plan': config.plan(), 'features': manifest['identity'],
                'sources': [{k: r[k] for k in ('checkpoint_id', 'signature', 'cell')} for r in records],
                'device': str(device), 'sae_precision': 'float32_tf32_disabled',
                'search': 'exact_cosine_score_desc_gallery_position_asc',
                'empty_code_policy': 'abstain_nearest_grid_mask_count_false_negative'}
    output = Path(workspace) / 'reports' / canonical_hash(campaign)[:20]
    atomic_json(output / 'manifest.json', campaign)
    results = []
    points = [(None, None, None)] + [(cell, record, k) for cell, record in zip(config.cells(), records)
                                     for k in config.inference_k]
    from ..evaluate.knn import fp32_matmul
    with torch.inference_mode(), fp32_matmul():
        for cell, record, k in points:
            method = cell.method if cell else 'dense'
            slug = f'{method}_k{k}' if cell else 'dense'
            point_path = output / f'{slug}.json'
            previous = read_json(point_path)
            if previous and previous.get('complete'):
                results.append(previous)
                continue
            sae = load_fitted(cell, Path(record['path']).parent, BACKBONES[cell.backbone],
                              device=torch.device(device)).requires_grad_(False) if cell else None
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
                          'inference_k': k, 'checkpoint_id': record['checkpoint_id'] if cell else None,
                          **metrics(matrix), 'confusion': matrix.tolist(),
                          'bank_rows': len(features), 'bank_nnz_mean': float(bank_nnz.mean()),
                          'bank_empty_patches': int(np.count_nonzero(bank_nnz == 0)),
                          'query_nnz_mean': nnz_total/nnz_count, 'query_nnz_max': nnz_max,
                          'query_empty_patches': empty_count, 'query_patches': nnz_count,
                          'search_seconds': seconds, 'invocation_seconds': time.perf_counter()-start,
                          'timing_note': 'search_seconds includes cached per-image measurements; not end-to-end latency',
                          'index_bytes': int(bank['indices'].nbytes + bank['values'].nbytes) if sae else int(bank.nbytes)}
            atomic_json(point_path, record_out)
            results.append(record_out)
            del sae, bank
    atomic_json(output / 'results.json', {'complete': True, 'campaign': campaign, 'results': results})
    from .report import plot
    plot(output / 'results.json')
    return {'complete': True, 'report': str(output), 'results': results}
