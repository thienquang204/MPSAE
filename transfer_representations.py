"""Frozen parents, disk-backed extraction and resumable ImageNet-only compressors.

The legacy ImageNet objectives are imported unchanged. No work happens on import.
This module is executed only by transfer_experiments.py when the user launches it.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import __version__ as torchvision_version, models, transforms

from csr_vs_mmpot_imagenet import (
    TopKSAE, build_image_dataset, cosine_annealed_k,
    nonnegative_contrastive_loss, partial_matching_gap,
)

PROTOCOL = "frozen-transfer-v1"
MRL_REVISION = "72b128ce966ea468f85905a81602e46a60106e25"
MRL_FILES = {
    "resnet18": ("resnet-family/resnet18_mrl/final_weights.pt", "7941c08ebef2210d74703cce7e23737e5cd9ba3c8a6e78eb93d7c440745c8784"),
    "resnet50": ("mrl/Imagenet1k_R50_mrl/final_weights.pt", "6e7a4ad6d7af1991e7d6b093925b1d032dbe385f111beb99700a9f1fec306e83"),
}
DIMS = {"resnet18": 512, "resnet50": 2048}
SPARSE_METHODS = ("csr", "mpsae", "csrv2", "mpsaev2")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(root, stage, **fields):
    value = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **fields}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False, default=str) + "\n")
    atomic_json(root / "status.json", value)
    print(json.dumps(value, default=str), flush=True)


@contextlib.contextmanager
def run_lock(root):
    """An OS advisory lock releases after crashes; no stale-PID guessing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".transfer.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Another transfer process owns {root}") from exc
    try:
        yield
    finally:
        handle.close()


def disk_budget(directory, size):
    Path(directory).mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(directory).free
    if available < size + (512 << 20):
        raise RuntimeError(f"Insufficient disk at {directory}: need {size:,} bytes plus 512 MiB reserve, free {available:,}")


class BlurPoolConv2d(nn.Module):
    """FFCV/MRL pre-convolution blur, including strided projection shortcuts.

    Architecture reference: libffcv/ffcv-imagenet train_imagenet.py,
    BlurPoolConv2d and create_model_and_scaler. The hashed release weights
    and strict tensor-key checking determine compatibility, not model width.
    """
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        kernel = torch.tensor([1., 2., 1.])
        kernel = torch.outer(kernel, kernel).reshape(1, 1, 3, 3) / 16
        self.register_buffer("blur_filter", kernel.repeat(conv.in_channels, 1, 1, 1))

    def forward(self, x):
        return self.conv(F.conv2d(x, self.blur_filter, padding=1, groups=self.conv.in_channels))


def apply_blurpool(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and max(child.stride) > 1 and child.in_channels >= 16:
            setattr(module, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def load_parent(architecture, parent, args, device):
    """Download on demand, hash, and strictly load a native released parent."""
    if architecture not in DIMS or parent not in ("mrl", "timm"):
        raise ValueError(f"Unsupported frozen parent: {architecture}/{parent}")
    from huggingface_hub import hf_hub_download
    cache = Path(args.cache_dir) / "weights"
    cache.mkdir(parents=True, exist_ok=True)
    kwargs = {"cache_dir": str(cache), "local_files_only": not args.download}
    if parent == "mrl":
        filename, expected = MRL_FILES[architecture]
        path = hf_hub_download("aniketr/mrl-resnet50", filename, revision=MRL_REVISION, **kwargs)
        sha = file_hash(path)
        if sha != expected:
            raise RuntimeError(f"MRL {architecture} checkpoint hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        cleaned = {}
        for key, value in state.items():
            # Releases may have DDP and Sequential wrappers. Never discard
            # unknown feature tensors or use strict=False to hide mismatches.
            while key.split(".", 1)[0] in ("module", "model", "0") and "." in key:
                key = key.split(".", 1)[1]
            if key.startswith("fc."):
                continue
            if key in cleaned:
                raise RuntimeError(f"Duplicate MRL tensor after wrapper removal: {key}")
            cleaned[key] = value
        network = getattr(models, architecture)(weights=None)
        apply_blurpool(network)
        network.fc = nn.Identity()
        network.load_state_dict(cleaned, strict=True)
        provenance = {"repository": "aniketr/mrl-resnet50", "revision": MRL_REVISION, "file": filename, "sha256": sha, "blurpool": True,
                      "architecture_reference": "https://github.com/RAIVNLab/MRL/blob/main/utils.py", "torchvision": torchvision_version}
    else:
        import timm
        from safetensors.torch import load_file
        repo = f"timm/{architecture}.a1_in1k"
        resolution_file = cache / f"{architecture}.a1_in1k.resolved.json"
        if resolution_file.exists():
            resolution = json.loads(resolution_file.read_text(encoding="utf-8"))
            if resolution.get("repository") != repo:
                raise RuntimeError(f"Incorrect parent repository in {resolution_file}")
            revision = resolution["revision"]
        else:
            from huggingface_hub import HfApi
            if not args.download:
                raise RuntimeError(f"Missing pinned timm resolution {resolution_file}; enable downloads once")
            revision = HfApi().model_info(repo, revision="main").sha
            if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise RuntimeError(f"Could not pin an immutable parent revision for {repo}")
            atomic_json(resolution_file, {"repository": repo, "revision": revision})
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"Parent revision in {resolution_file} must be an immutable commit SHA")
        path = hf_hub_download(repo, "model.safetensors", revision=revision, **kwargs)
        config_path = hf_hub_download(repo, "config.json", revision=revision, **kwargs)
        network = timm.create_model(f"{architecture}.a1_in1k", pretrained=False)
        network.load_state_dict(load_file(path), strict=True)
        network.reset_classifier(0)
        provenance = {"repository": repo, "revision": revision, "file": "model.safetensors", "sha256": file_hash(path), "config_sha256": file_hash(config_path), "timm": timm.__version__}
    provenance.update({"architecture": architecture, "parent": parent, "dimension": DIMS[architecture], "precision": "fp32", "torch": str(torch.__version__),
                       "encoder_parameters": sum(p.numel() for p in network.parameters())})
    network.eval().requires_grad_(False).to(device)
    return network, provenance


class SeededImageNet(Dataset):
    """Per-index crop seed keeps view identity independent of worker/resume order."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            # Transforms consume CPU RNG; do not reseed CUDA when workers=0.
            torch.random.default_generator.manual_seed(42 + int(index))
            value = self.dataset[int(index)]
        return value[0]


@contextlib.contextmanager
def offline_huggingface(enabled):
    """Honor offline mode even when datasets/hub were imported earlier."""
    if not enabled:
        yield
        return
    from datasets import config as datasets_config
    from huggingface_hub import constants as hub_constants
    previous_env = {key: os.environ.get(key) for key in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")}
    previous_flags = (datasets_config.HF_DATASETS_OFFLINE, datasets_config.HF_HUB_OFFLINE,
                      hub_constants.HF_HUB_OFFLINE)
    try:
        for key in previous_env:
            os.environ[key] = "1"
        datasets_config.HF_DATASETS_OFFLINE = True
        datasets_config.HF_HUB_OFFLINE = True
        hub_constants.HF_HUB_OFFLINE = True
        yield
    finally:
        (datasets_config.HF_DATASETS_OFFLINE, datasets_config.HF_HUB_OFFLINE,
         hub_constants.HF_HUB_OFFLINE) = previous_flags
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def imagenet_dataset(args):
    root = Path(args.imagenet_root)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if args.imagenet_backend == "hf":
        manifest_path = root / "imagenet_download_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        training = manifest.get("splits", {}).get("train", {})
        if not (manifest.get("validation_passed") is True and manifest.get("dataset_id") == args.imagenet_dataset_id and manifest.get("revision") == args.imagenet_revision and training.get("fully_checked") is True and training.get("validation_status") == "passed"):
            if not args.download:
                raise RuntimeError("ImageNet train cache needs validation/download; use --download or run download_imagenet.py")
            from download_imagenet import main as download_main
            result = download_main(["--cache-dir", str(root), "--dataset-id", args.imagenet_dataset_id, "--revision", args.imagenet_revision, "--splits", "train", "--check-samples", "all"])
            if result:
                raise RuntimeError("ImageNet download/validation failed; check HF_TOKEN and dataset access")
        manifest = json.loads(manifest_path.read_text())
        training = manifest["splits"]["train"]
        identity = file_hash(manifest_path)
    else:
        identity = None
    # The downloader above performs the network access. Subsequent loading in
    # offline mode must not silently fetch missing shards when download is off.
    with offline_huggingface(not args.download and args.imagenet_backend == "hf"):
        dataset, _ = build_image_dataset("train", root, transform, args.imagenet_backend, args.imagenet_dataset_id, args.imagenet_revision, "HF_TOKEN")
    if args.imagenet_backend == "hf":
        loaded_fingerprint = getattr(dataset.split, "_fingerprint", None)
        if not loaded_fingerprint or loaded_fingerprint != training.get("fingerprint"):
            raise RuntimeError("Loaded ImageNet training shards differ from the fully validated manifest. Re-run download_imagenet.py with --splits train --check-samples all before fitting")
        identity = digest({"validation_manifest": identity, "dataset_fingerprint": loaded_fingerprint})
    if len(dataset) != 1_281_167:
        raise RuntimeError(f"Strict ImageNet fitting requires 1,281,167 training images, found {len(dataset)}")
    if args.imagenet_backend == "imagefolder":
        if len(dataset.classes) != 1000:
            raise RuntimeError("ImageNet ImageFolder must contain exactly 1000 classes")
        # Paths, lengths and mtimes invalidate stale local caches without
        # simultaneously decoding every image into memory. Extraction decodes
        # every image and fails on corrupt images before any compressor fit.
        h = hashlib.sha256()
        for name, label in dataset.samples:
            stat = Path(name).stat()
            h.update(f"{name}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        identity = h.hexdigest()
    return SeededImageNet(dataset), {"source": str(root), "fingerprint": identity, "view": "rrc224-bilinear-antialias-flip0.5-per-index-seed42-v1", "count": len(dataset)}


def extract_features(network, dataset, provenance, identity, path, args, device):
    """FP32 .npy memmap with durable batch cursor; partial rows are never read."""
    path = Path(path)
    meta_path = path.with_suffix(".json")
    partial = path.with_suffix(".partial.npy")
    signature = digest({"protocol": PROTOCOL, "source": args.source_hash, "parent": provenance, "data": identity, "rows": len(dataset)})
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if metadata and metadata.get("signature") != signature:
        raise RuntimeError(f"Incompatible feature cache {path}; use a new TRANSFER_CACHE directory")
    shape = (len(dataset), provenance["dimension"])
    if shape[0] == 0:
        raise ValueError(f"Cannot extract an empty crop/image dataset: {path}")
    if metadata.get("complete") and not path.exists():
        raise RuntimeError(f"Completed feature cache is missing its data file: {path}")
    if path.exists() and not metadata.get("complete"):
        # Publication order is flush data -> save full cursor -> rename data ->
        # save complete metadata. Recover precisely that final crash window.
        if metadata.get("rows_complete") == len(dataset) and not partial.exists():
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            if existing.shape != shape or existing.dtype != np.float32:
                raise RuntimeError(f"Malformed published feature cache: {path}")
            del existing
            metadata.update(complete=True, serialized_bytes=path.stat().st_size)
            atomic_json(meta_path, metadata)
        else:
            raise RuntimeError(f"Feature data exists without matching completed provenance: {path}")
    if metadata.get("complete") and path.exists():
        array = np.load(path, mmap_mode="r")
        if array.shape != shape or array.dtype != np.float32 or metadata.get("rows_complete") != len(dataset):
            raise RuntimeError(f"Malformed complete feature cache: {path}")
        return path
    cursor = int(metadata.get("rows_complete", 0))
    if not 0 <= cursor <= len(dataset):
        raise RuntimeError(f"Invalid feature-cache resume cursor: {meta_path}")
    if cursor and not partial.exists():
        raise RuntimeError(f"Partially written feature cache is missing: {partial}")
    if not partial.exists():
        disk_budget(path.parent, math.prod(shape) * 4)
        array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
    else:
        array = np.load(partial, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.float32:
            raise RuntimeError(f"Malformed partial cache: {partial}")
    loader = DataLoader(Subset(dataset, range(cursor, len(dataset))), batch_size=args.feature_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        **({"prefetch_factor": 2} if args.workers else {}))
    started = time.monotonic()
    prior_seconds = float(metadata.get("seconds", 0))
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            features = network(batch.to(device, non_blocking=True)).float()
            if features.shape != (len(batch), shape[1]) or not torch.isfinite(features).all():
                raise RuntimeError(f"Parent output shape/nonfinite failure: {provenance}")
            stop = cursor + len(batch)
            array[cursor:stop] = features.cpu().numpy()
            cursor = stop
            if batch_index % args.log_every == 0 or cursor == len(dataset):
                array.flush()
                metadata = {"signature": signature, "complete": False, "rows_complete": cursor, "rows": len(dataset), "parent": provenance, "data": identity, "seconds": prior_seconds + time.monotonic() - started, "microbatch": args.feature_batch_size}
                atomic_json(meta_path, metadata)
                event(args.output_dir, "extract", path=str(path), rows=cursor, total=len(dataset))
    array.flush()
    del array
    os.replace(partial, path)
    metadata.update(complete=True, rows_complete=cursor, serialized_bytes=path.stat().st_size)
    atomic_json(meta_path, metadata)
    return path


def load_complete_features(path):
    """Open a read-only memmap only after checking the publication manifest."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not metadata.get("complete") or not metadata.get("signature"):
        raise RuntimeError(f"Feature cache has not completed extraction: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (array.ndim != 2 or array.dtype != np.float32 or len(array) == 0
            or array.shape != (metadata.get("rows"), metadata.get("parent", {}).get("dimension"))
            or metadata.get("rows_complete") != len(array)):
        raise RuntimeError(f"Feature cache shape/dtype/cursor disagrees with its manifest: {path}")
    return array, metadata


def train_compressor(path, feature_path, parent, method, k, seed, args, device):
    """Same existing losses, table-9 batch and FP32, with atomic epoch resume.

    V2 extensions anneal to the labelled training endpoint k; each endpoint has
    its own fit. They are explicitly not called fixed-k Table 9 reproductions.
    """
    path = Path(path)
    if method not in SPARSE_METHODS or k not in (8, 16, 64, 256):
        raise ValueError(f"Unsupported transfer compressor/budget: {method}/{k}")
    data, feature_metadata = load_complete_features(feature_path)
    epochs = 10 + (args.mpsaev2_extra_epochs if method == "mpsaev2" else 0)
    recipe = {"protocol": PROTOCOL, "source": args.source_hash, "method": method, "parent": parent,
              "feature_signature": feature_metadata["signature"],
              "seed": seed, "k_train": k, "width": 8192, "epochs": epochs, "batch_size": 1024,
              "lr": 4e-5, "weight_decay": 1e-4, "epsilon": 6.25e-10, "aux_k": 512, "aux_beta": 1 / 32,
              "contrastive_gamma": 1.0, "temperature": 0.2, "drop_last": True, "precision": "fp32",
              "annealing": {"start": max(64, 4 * k), "fraction": 0.7} if method.endswith("v2") else None,
              "mmpot": {"weight": 2.0, "mass": 0.9, "eta": 0.2, "iterations": 100, "tolerance": 1e-4, "microbatch": args.ot_microbatch} if method.startswith("mpsae") else None}
    signature = digest(recipe)
    torch.manual_seed(seed)
    model = TopKSAE(data.shape[1], 8192, 30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5, weight_decay=1e-4, eps=6.25e-10, betas=(0.9, 0.999))
    start_epoch = 0
    history = []
    if path.exists():
        # Loading to CPU avoids holding a second full optimizer on the GPU.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise RuntimeError(f"Incompatible compressor checkpoint {path}; use a fresh output directory")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        if not isinstance(start_epoch, int) or not 0 <= start_epoch <= epochs or len(history) != start_epoch:
            raise RuntimeError(f"Invalid compressor checkpoint epoch/history: {path}")
        if [row.get("epoch") for row in history] != list(range(1, start_epoch + 1)):
            raise RuntimeError(f"Noncontiguous compressor checkpoint history: {path}")
        del checkpoint
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError(f"Nonfinite compressor checkpoint parameters: {path}")
    batches = len(data) // 1024
    if batches == 0:
        raise RuntimeError("No complete 1024-example compressor batch")
    for epoch in range(start_epoch, epochs):
        # Epoch-specific permutations make interruption replay exact and avoid
        # saving fragile NumPy/Python pickled RNG state.
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(data))
        started = time.monotonic()
        total = 0.0
        model.train()
        for batch in range(batches):
            x = torch.from_numpy(np.array(data[order[batch * 1024:(batch + 1) * 1024]], dtype=np.float32, copy=True)).to(device)
            step = epoch * batches + batch
            current_k = cosine_annealed_k(step, epochs * batches, max(64, 4 * k), k, 0.7) if method.endswith("v2") else k
            optimizer.zero_grad(set_to_none=True)
            if method in ("csr", "csrv2"):
                reconstruction, statistics, z = model.csr_losses(x, current_k, 512, 1.0, 0.125, 1 / 32)
                regularizer = nonnegative_contrastive_loss(z, 0.2)
                model.update_activity(z)
                loss = reconstruction + regularizer
            else:
                reconstruction, statistics, views = model.reconstruction_losses(x, current_k, 512, 1.0, 0.125, 1 / 32)
                regularizer, _ = partial_matching_gap(*views, 0.9, 0.2, 100, 1e-4, args.ot_microbatch)
                loss = reconstruction + 2.0 * regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite compressor loss in {path}, epoch {epoch}, batch {batch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite compressor gradient in {path}")
            optimizer.step()
            total += float(loss.detach())
            if batch % args.log_every == 0:
                event(args.output_dir, "compressor", method=method, seed=seed, k=k, epoch=epoch + 1, batch=batch, batches=batches, loss=float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": total / batches, "seconds": time.monotonic() - started, "samples": batches * 1024})
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(f"Nonfinite compressor parameters after epoch {epoch + 1}: {path}")
        atomic_torch(path, {"signature": signature, "recipe": recipe, "epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history})
        atomic_json(path.with_suffix(".json"), {"signature": signature, "recipe": recipe, "history": history, "complete": epoch + 1 == epochs})
    # Also repair the harmless crash window after a final checkpoint rename
    # but before its human-readable status manifest was written.
    atomic_json(path.with_suffix(".json"), {"signature": signature, "recipe": recipe, "history": history, "complete": True})
    model.eval().requires_grad_(False)
    return model, {"signature": signature, "checkpoint_sha256": file_hash(path), "recipe": recipe, "fit_seconds": sum(x["seconds"] for x in history)}


def encode_sparse(model, k, features_path, directory, compressor, args, device):
    """Bounded-memory indexed FP32 cache; account for padding and row counts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source, feature_metadata = load_complete_features(features_path)
    if not 0 < k <= model.decoder.shape[0]:
        raise ValueError("Sparse encoding budget must fit the compressor dictionary")
    signature = digest({"compressor": compressor["signature"], "k": k, "source": feature_metadata["signature"]})
    meta_path = directory / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta and meta.get("signature") != signature:
        raise RuntimeError(f"Incompatible sparse cache: {directory}")
    complete = bool(meta.get("complete"))
    start = int(meta.get("rows_complete", 0))
    if not 0 <= start <= len(source) or (complete and start != len(source)):
        raise RuntimeError(f"Invalid sparse cache resume cursor: {directory}")
    arrays = {}
    layout = (("values", np.float32, (len(source), k)), ("indices", np.uint32, (len(source), k)), ("nnz", np.uint32, (len(source),)))
    required = len(source) * (8 * k + 4)
    if not meta:
        disk_budget(directory, required)
    for name, dtype, shape in layout:
        path = directory / f"{name}.npy"
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r" if complete else "r+")
            if arrays[name].shape != shape or arrays[name].dtype != dtype:
                raise RuntimeError(f"Invalid sparse cache array {path}")
        elif complete or start:
            raise RuntimeError(f"Missing array in a partially/completely committed sparse cache: {path}")
        else:
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if complete:
        return directory
    started = time.monotonic()
    seconds_before = float(meta.get("seconds", 0))
    model.eval()
    with torch.inference_mode():
        for begin in range(start, len(source), args.encode_batch_size):
            end = min(begin + args.encode_batch_size, len(source))
            x = torch.from_numpy(np.array(source[begin:end], dtype=np.float32, copy=True)).to(device)
            preactivations = model.preactivations(x)
            if not torch.isfinite(preactivations).all():
                raise FloatingPointError("Nonfinite sparse crop preactivations")
            values, indices = torch.topk(preactivations, k, dim=1, sorted=True)
            values = values.relu()
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite sparse crop code")
            arrays["values"][begin:end] = values.cpu().numpy()
            arrays["indices"][begin:end] = indices.cpu().numpy().astype(np.uint32)
            arrays["nnz"][begin:end] = (values > 0).sum(1).cpu().numpy().astype(np.uint32)
            if (begin // args.encode_batch_size) % args.log_every == 0 or end == len(source):
                for array in arrays.values():
                    array.flush()
                meta = {"signature": signature, "complete": False, "rows_complete": end, "width": 8192, "k": k, "rows": len(source), "seconds": seconds_before + time.monotonic() - started,
                        "format": "padded-topk-fp32-values-uint32-indices-uint32-nnz",
                        "support_semantics": "first_nnz_entries_have_positive_values;zero_padding_is_not_support"}
                atomic_json(meta_path, meta)
                event(args.output_dir, "sparse_cache", path=str(directory), rows=end, total=len(source))
    nnz_sum = sum(int(arrays["nnz"][i:i + 65536].sum(dtype=np.uint64)) for i in range(0, len(source), 65536))
    meta.update(complete=True, mean_nnz=nnz_sum / len(source), nominal_nonzero_payload_bytes=nnz_sum * 8,
                serialized_array_bytes=sum((directory / f"{name}.npy").stat().st_size for name in arrays),
                bytes_per_region=required / len(source), padding_included=True)
    atomic_json(meta_path, meta)
    return directory
