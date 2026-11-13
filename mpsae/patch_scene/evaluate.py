import time
import numpy as np
import torch

from ..io import atomic_json, canonical_hash, read_json, array_hash
from ..scene.core import confusion, metrics
from .features import queries
from .retrieval import Search, codes
from .config import VERSION


def evaluate(config, encoder, dataset, method, model, fit_record, features, labels, selected,
             manifest, hub, device):
    sparse = model is not None
    identity = {'version': VERSION, 'features': manifest['identity'], 'method': method,
                'fit': fit_record['checkpoint_sha256'] if sparse else None,
                'support_indices': array_hash(selected), 'support_rows': len(selected),
                'k': config.k if sparse else None, 'neighbors': config.neighbors,
                'temperature': config.temperature, 'query_block': config.query_block,
                'search': 'fp32_exact_coordinate_lookup_v1' if sparse else 'fp32_exact_gemm_v1',
                'tie_rule': 'score_desc_position_asc', 'ignore': 255}
    relative = f'evaluations/{canonical_hash(identity)[:20]}'
    folder = hub.pull(relative)
    result = read_json(folder/'result.json')
    if result is not None:
        if result['identity'] != identity or not result['complete']:
            raise ValueError('evaluation identity mismatch')
        return result
    state = read_json(folder/'progress.json') or {}
    if state and state['identity'] != identity:
        raise ValueError('evaluation resume mismatch')
    classes = manifest['classes']
    matrix = np.asarray(state.get('confusion', np.zeros((classes, classes+1), dtype=np.int64)), dtype=np.int64)
    next_image = state.get('next_image', 0)
    nnz_sum, patches, empties = (state.get(k, 0) for k in ('nnz_sum', 'patches', 'empty'))
    prior_seconds = state.get('seconds', 0.0)
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    if sparse:
        gallery = codes(model, features[selected], config.k, device)
        gallery_nnz = float(gallery[1].ne(0).sum(1).float().mean())
        dim = model.hidden_dim
    else:
        gallery, dim = features[selected], features.shape[1]
        gallery_nnz = float(np.count_nonzero(gallery, axis=1).mean())
    search = Search(gallery, labels[selected], device, dim=dim, sparse=sparse,
                    neighbors=config.neighbors, temperature=config.temperature, query_block=config.query_block)
    del gallery
    started = time.perf_counter()
    for i, x, truth in queries(manifest, hub, skip=next_image):
        query = codes(model, x, config.k, device) if sparse else x
        if sparse:
            counts = query[1].ne(0).sum(1)
            if int(counts.max()) > config.k:
                raise AssertionError('K-sparsity violated')
            nnz_sum += int(counts.sum())
        else:
            nnz_sum += int(np.count_nonzero(x))
        patches += len(x)
        prediction, empty = search.predict(query, manifest['grid'], truth.shape)
        matrix += confusion(prediction, truth, classes=classes)
        empties += empty
        next_image = i+1
        if next_image % config.shard_images == 0 or next_image == manifest['counts']['validation']:
            atomic_json(folder/'progress.json', {'identity': identity, 'next_image': next_image,
                'confusion': matrix.tolist(), 'nnz_sum': nnz_sum, 'patches': patches, 'empty': empties,
                'seconds': prior_seconds+time.perf_counter()-started})
            hub.push(relative)
            print(f'[eval {dataset}/{encoder}/{method}] {next_image}/{manifest["counts"]["validation"]}', flush=True)
    if next_image != manifest['counts']['validation']:
        raise RuntimeError('incomplete validation split')
    result = {'complete': True, 'identity': identity, 'dataset': dataset, 'encoder': encoder, 'method': method,
              'k': config.k if sparse else None, 'seed': config.seed, 'query_images': next_image,
              **metrics(matrix, include_absent=dataset == 'ade20k'), 'mean_nnz': nnz_sum/patches,
              'gallery_mean_nnz': gallery_nnz, 'empty_patches': empties,
              'query_seconds': prior_seconds+time.perf_counter()-started,
              'timing_scope': 'query streaming, encoding, exact retrieval, upsampling, metrics and periodic backups',
              'search_workspace_bytes': search.workspace_bytes,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
              'supervision': 'training-mask labels in SAE loss' if method == 'csrv2' else 'no class targets in SAE loss',
              'confusion': matrix.tolist()}
    atomic_json(folder/'result.json', result)
    hub.push(relative)
    del search
    return result
