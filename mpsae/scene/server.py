"""Fresh-server verification and resumable checkpoint restoration orchestration."""
from pathlib import Path
import platform
import shutil
import time

from ..io import atomic_json, canonical_hash, read_json


def doctor(device='cuda:0', expected_cuda=None):
    """Exercise actual data decoding, native backbone, SAE and ranking operations."""
    import numpy as np
    import torch
    from torch import nn
    from torchvision.models import resnet50
    from ..server import verify_data_stack, validate_cuda_runtime
    from ..embed.mrl_ff2048 import apply_blurpool
    from ..model import SparseAutoencoder
    from ..evaluate.knn import fp32_matmul
    from .core import encode, predict, resnet_spatial

    try:
        versions = verify_data_stack()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError('Scene data-stack verification failed. Run bash setup_scene_server.sh '
                           'and use .venv-scene/bin/python.') from error
    target = torch.device(device)
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Check nvidia-smi and run bash setup_scene_server.sh')
        capability = torch.cuda.get_device_capability(target)
        validate_cuda_runtime(capability, torch.version.cuda, expected=expected_cuda)
        hardware = {'name': torch.cuda.get_device_name(target), 'capability': list(capability),
                    'vram_bytes': torch.cuda.get_device_properties(target).total_memory}
    elif target.type == 'cpu':
        if expected_cuda:
            raise ValueError('expected_cuda needs a CUDA device')
        hardware = {'name': 'CPU verification only'}
    else:
        raise ValueError('scene evaluation supports CUDA or CPU')
    network = resnet50(weights=None)
    apply_blurpool(network)
    network.fc = nn.Identity()
    network = network.eval().requires_grad_(False).to(target)
    with torch.inference_mode(), fp32_matmul(), torch.autocast(
            target.type, dtype=torch.float16 if target.type == 'cuda' else torch.bfloat16,
            enabled=target.type == 'cuda'):
        patches = resnet_spatial(network, torch.randn(1, 3, 64, 64, device=target))
        if patches.shape != (1, 2, 2, 2048) or not torch.isfinite(patches).all():
            raise RuntimeError('spatial ResNet GPU check failed')
    raw = patches.float().reshape(4, 2048).cpu().numpy()
    sae = SparseAutoencoder(2048, 32).to(target).eval().requires_grad_(False)
    with torch.inference_mode(), fp32_matmul():
        code = encode(sae, raw, 8, target, batch=2)
        labels = np.zeros((4, 21), dtype=np.float32)
        labels[:, 0] = 1
        prediction, _ = predict(code, labels, code, dim=32, grid=(2, 2), output_size=(4, 4),
                                 device=target, neighbors=2, gallery_chunk=2, query_block=2)
    if prediction.shape != (4, 4):
        raise RuntimeError('retrieval decoder check failed')
    del network, sae, patches
    if target.type == 'cuda':
        torch.cuda.empty_cache()
    return {'verified': True, 'device': str(target), 'hardware': hardware,
            'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'data_versions': versions, 'training': False}


def publish_results(path, repo):
    from huggingface_hub import HfApi
    path = Path(path)
    if not (read_json(path) or {}).get('complete'):
        raise ValueError('cannot publish an incomplete report')
    commit = HfApi().upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(path.parent),
        path_in_repo=f'scene-v1/reports/{path.parent.name}',
        allow_patterns=['*.json', '*.csv', '*.pdf', '*.png'], ignore_patterns=['per_image/**'],
        commit_message='Save frozen spatial SAE evaluation')
    return str(commit)


def bootstrap(config, workspace, *, repo, revision, prefix, device='cuda:0',
              refresh_source=False, publish=False):
    """Restore, verify and evaluate. Caller owns the workspace lock."""
    from .__main__ import restore
    from .run import run, source_records
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    selection = {'repo': repo, 'requested_revision': revision, 'prefix': prefix,
                 'cells': [cell.__dict__ for cell in config.cells()]}
    receipt_path = workspace / 'sources' / f'{canonical_hash(selection)[:20]}.json'
    status_path = workspace / 'status.json'
    state = {'complete': False, 'stage': 'verify_environment', 'updated': time.time(),
             'training_runs': 0, 'selection': selection, 'publish_requested': publish}

    def stage(name):
        state.update(stage=name, updated=time.time())
        atomic_json(status_path, state)
        print(f'[scene server] {name}', flush=True)

    try:
        config.plan()
        stage('verify_environment')
        state['environment'] = doctor(device)
        state['free_disk_bytes'] = shutil.disk_usage(workspace).free
        if state['free_disk_bytes'] < 8 * 2**30:
            print('[scene server] Less than 8 GiB free: new extraction/downloads may need more disk.', flush=True)
        stage('restore_checkpoints')
        receipt = None if refresh_source else read_json(receipt_path)
        if receipt and receipt.get('selection') != selection:
            raise RuntimeError('source receipt does not match the requested checkpoint selection')
        if receipt:
            try:
                source_records(config, receipt['source_root'])
            except FileNotFoundError:
                # A removed cache is repaired at the OLD pinned revision, never silently upgraded.
                pinned = receipt['revision']
                receipt = {**restore(config, repo=repo, revision=pinned, prefix=prefix,
                                      cache_dir=str(workspace / 'hf_checkpoints')), 'selection': selection}
                atomic_json(receipt_path, receipt)
        else:
            receipt = {**restore(config, repo=repo, revision=revision, prefix=prefix,
                                  cache_dir=str(workspace / 'hf_checkpoints')), 'selection': selection}
            atomic_json(receipt_path, receipt)
        state['source'] = receipt
        stage('evaluate')
        result = run(config, workspace, receipt['source_root'], device=device)
        state['report'] = result['report']
        atomic_json(workspace / 'latest.json', {'report': result['report'], 'source_revision': receipt['revision'],
                                               'evaluation_complete': True})
        if publish:
            stage('publish')
            state['upload_commit'] = publish_results(Path(result['report']) / 'results.json', repo)
        state['complete'] = True
        stage('complete')
        return {'complete': True, 'report': result['report'], 'source_revision': receipt['revision'],
                'status': str(status_path), 'published': bool(publish)}
    except BaseException as error:
        state.update(complete=False, error_type=type(error).__name__, updated=time.time())
        # Error values may contain URLs/auth details; persist only the type and failed stage.
        atomic_json(status_path, state)
        raise
