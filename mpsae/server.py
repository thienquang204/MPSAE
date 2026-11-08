"""Verify a native NVIDIA environment without credentials or real datasets.

Run ``python -m mpsae.server`` after sourcing activate_server.sh.
Failures propagate: importing a package is insufficient evidence that its
CUDA libraries, JIT compiler or FFCV preprocessing pipeline actually work.
"""
from __future__ import annotations

import argparse
import importlib
import io
import os
import platform
from pathlib import Path
import tempfile


def validate_cuda_runtime(capability, runtime, *, expected=None):
    """Torch-free architecture guard shared by tests and GPU verification."""
    required = {(7, 0): (12, 6), (7, 5): (12, 6), (8, 0): (12, 6),
                (8, 6): (12, 6), (8, 9): (12, 6), (9, 0): (12, 6),
                (10, 0): (12, 8), (12, 0): (12, 8), (10, 3): (12, 9)}
    capability = tuple(capability)
    if capability not in required:
        raise RuntimeError(f"Unsupported GPU compute capability: {capability}")
    try:
        version = tuple(map(int, str(runtime).split(".")[:2]))
    except ValueError as error:
        raise RuntimeError(f"Invalid PyTorch CUDA runtime: {runtime}") from error
    if version < required[capability]:
        raise RuntimeError(f"sm_{capability[0]}{capability[1]} needs this project's CUDA "
                           f"{'.'.join(map(str, required[capability]))}+ profile; PyTorch has {runtime}. "
                           "Run bash setup_server.sh on this machine.")
    if expected and str(runtime) != expected:
        raise RuntimeError(f"Selected profile expects CUDA {expected}; PyTorch has {runtime}. "
                           "Source activate_server.sh or rerun setup_server.sh.")


def verify_server(*, check_extension: bool = True) -> dict:
    import numpy as np
    import torch
    import cupy as cp
    from PIL import Image
    from ffcv.fields import IntField, RGBImageField
    from ffcv.writer import DatasetWriter

    from .config import BACKBONES
    from .embed.csrv2_timm import _loader

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use CUDA; check the NVIDIA driver and active .venv")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpus = []
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        validate_cuda_runtime(capability, torch.version.cuda, expected=os.environ.get("MPSAE_CUDA_VERSION"))
        gpus.append({"index": index, "name": torch.cuda.get_device_name(index), "capability": list(capability)})
        print(f"GPU {index}: {gpus[-1]['name']}; sm_{capability[0]}{capability[1]}; "
              f"torch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
        matrix = torch.randn(128, 128, device=f"cuda:{index}")
        with torch.no_grad():
            product = matrix @ matrix.T
            if not bool(torch.isfinite(product).all()):
                raise RuntimeError(f"nonfinite PyTorch GPU {index} matmul")
            with torch.autocast("cuda", dtype=torch.float16):
                product = matrix @ matrix.T
            if not bool(torch.isfinite(product).all()):
                raise RuntimeError(f"nonfinite PyTorch GPU {index} fp16 matmul")
        with cp.cuda.Device(index):
            values = cp.arange(16, dtype=cp.float32)
            # Forces NVRTC compilation rather than just testing allocation.
            kernel = cp.ElementwiseKernel("float32 x", "float32 y", "y = x * 2;", "mpsae_setup_double")
            observed = kernel(values).get()
            np.testing.assert_array_equal(observed, np.arange(16, dtype=np.float32) * 2)
        print(f"GPU {index} PyTorch fp32/fp16 + CuPy NVRTC kernel: OK", flush=True)

    class TinyImages:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            pixels = np.full((256, 256, 3), 32 + index * 32, dtype=np.uint8)
            return Image.fromarray(pixels), index

    with tempfile.TemporaryDirectory(prefix="mpsae-ffcv-check-") as temporary:
        path = Path(temporary) / "tiny.ffcv"
        writer = DatasetWriter(str(path), {
            "image": RGBImageField(write_mode="jpg", jpeg_quality=95),
            "label": IntField(),
        }, num_workers=1)
        writer.from_indexed_dataset(TinyImages())
        spec = BACKBONES["rn18"]
        mean, std = spec.normalize
        # Exercise both augmentation/order settings used for source extraction.
        for train in (False, True):
            loader = _loader(path, spec=spec, train=train, batch_size=2, workers=1,
                             seed=1227, device=device, mean=mean, std=std)
            seen = []
            for images, labels in loader:
                if images.shape[1:] != (3, spec.input_size, spec.input_size):
                    raise RuntimeError(f"unexpected FFCV image shape: {images.shape}")
                if images.device != device or not bool(torch.isfinite(images).all()):
                    raise RuntimeError("FFCV did not produce finite CUDA images")
                for image, label in zip(images, labels):
                    index = int(label.item())
                    expected = torch.tensor([(32 + index * 32) / 255 - m for m in mean], device=device)
                    expected /= torch.tensor(std, device=device)
                    torch.testing.assert_close(image.mean(dim=(1, 2)), expected, atol=0.025, rtol=0)
                    seen.append(index)
            if sorted(seen) != list(range(4)):
                raise RuntimeError(f"FFCV dropped/duplicated/misaligned rows: {seen}")
            del loader
    print("FFCV JPEG decode + train/validation GPU normalization: OK", flush=True)

    versions = {}
    for name in ("torchvision", "timm", "safetensors", "datasets", "pyarrow",
                 "huggingface_hub", "numpy", "numba", "cv2", "ffcv", "cupy", "yaml", "psutil"):
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "installed"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    try:
        for format_name in ("png", "pdf"):
            with io.BytesIO() as buffer:
                figure.savefig(buffer, format=format_name)
    finally:
        plt.close(figure)
    print("Project imports + headless PNG/PDF export: OK", flush=True)

    if check_extension:
        from .evaluate.timing import maybe_build_inverted_index

        extension = maybe_build_inverted_index()
        if extension is None:
            raise RuntimeError("retrieval CUDA extension did not build; verify CUDA_HOME, nvcc, g++ and ninja")
        indices = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        values = torch.tensor([[2., 3.]], device=device)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
        documents = torch.tensor([0, 1], dtype=torch.int32, device=device)
        postings = torch.tensor([4., 5.], device=device)
        scores = torch.zeros((1, 2), device=device)
        extension.accumulate(indices, values, offsets, documents, postings, scores)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(scores, torch.tensor([[8., 15.]], device=device))
        print("Compiled sparse-retrieval CUDA kernel: OK", flush=True)
    print("Native NVIDIA environment verification passed.", flush=True)
    return {"gpu": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
            "torch": str(torch.__version__), "versions": versions, "extension_checked": check_extension,
            "gpus": gpus, "cpu_architecture": platform.machine(),
            "cuda_home": os.environ.get("CUDA_HOME"), "compile_architectures": os.environ.get("TORCH_CUDA_ARCH_LIST")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-extension", action="store_true",
                        help="diagnose core dependencies without the secondary compiled timing backend")
    parser.add_argument("--profile-output", type=Path, help="save verified hardware and package provenance as JSON")
    args = parser.parse_args(argv)
    result = verify_server(check_extension=not args.skip_extension)
    if args.profile_output:
        from .io import atomic_json

        atomic_json(args.profile_output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
