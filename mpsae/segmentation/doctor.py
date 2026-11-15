"""Environment check shared by the supported segmentation campaigns."""
import platform


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
        raise RuntimeError('Segmentation data-stack verification failed. Install requirements-patch-scene.txt.') from error
    target = torch.device(device)
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Check nvidia-smi and the installed CUDA-enabled PyTorch build')
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

