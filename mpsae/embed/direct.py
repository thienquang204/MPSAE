"""Versioned ImageNet extraction from JPEGs; no on-disk decoded-image copy.

This is a new protocol, NOT a byte-equivalent replacement for released FFCV.
MRL uses a centre crop. timm train uses per-image seeded crop/flip, every row
once in sorted order, with no FFCV resize/recompression or dropped tail.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..io import atomic_json, canonical_hash, read_json, require_disk, sha256_file

RECIPE = "imagenet-direct-pil-v1"


def preflight_backbones(names, cache, device):
    """Load every selected weight and exercise its graph BEFORE dataset download."""
    import torch
    from PIL import Image
    from ..config import BACKBONES
    from ..benchmark.extract import _network, evaluation_transform
    from ..logs import get_logger

    target = torch.device(device)
    for name in names:
        spec = BACKBONES[name]
        model, pretrained, _ = _network(spec, Path(cache), target, os.environ.get("HF_TOKEN"))
        try:
            transform, _ = evaluation_transform(spec, pretrained)
            images = transform(Image.new("RGB", (280, 300), (96, 96, 96))).unsqueeze(0)
            with torch.inference_mode():
                values = forward_batches(model, images, spec, target)
            import numpy as np
            if not np.isfinite(values).all():
                raise RuntimeError(f"{name}: nonfinite pretrained forward pass")
            get_logger("preflight").info("%s: pinned weights, preprocessing and CUDA inference OK", name)
        finally:
            del model
            torch.cuda.empty_cache()


class SeededImages:
    """Per-row RNG makes augmentation independent of worker count and resume."""
    def __init__(self, dataset, transform, seed, start=0):
        self.dataset, self.transform, self.seed, self.start = dataset, transform, seed, start

    def __len__(self):
        return len(self.dataset) - self.start

    def __getitem__(self, index):
        import torch
        position = index + self.start
        image, label = self.dataset[position]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + position)
            image = self.transform(image)
        return image, label


def release_sink(sink, features, labels):
    """Only reclaim this completed writer, after its archive has been verified."""
    from ..io import exclusive_lock
    for array in (features, labels):
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    with exclusive_lock(sink.directory / "sink.lock"):
        for filename in ("features.npy", "labels.npy", "sink.json"):
            (sink.directory / filename).unlink(missing_ok=True)


def reclaim_verified_writer(cache, manifest):
    """Recover cleanup after a crash between archive publication and writer GC.

    Caller must verify the archive first. No incomplete writer is removed.
    """
    import re
    from types import SimpleNamespace
    signature = manifest.get("identity", "")
    if not isinstance(signature, str) or not re.fullmatch(r"[a-f0-9]{64}", signature):
        return
    directory = Path(cache) / "features" / signature
    receipt = read_json(directory / "sink.json")
    if (receipt and receipt.get("complete") and receipt.get("signature") == signature
            and receipt.get("rows") == manifest["rows"] and receipt.get("dim") == manifest["dim"]):
        release_sink(SimpleNamespace(directory=directory), None, None)


def extract_source(spec, image_root, output, cache, *, batch_size=256, workers=16,
                   device="cuda:0", seed=1227):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from .archive import FeatureSink, finalize_archive, library_versions, load_current_archive
    from ..benchmark.extract import _network, evaluation_transform
    from ..data.imagenet import imagenet_folder_stats
    from ..logs import Progress

    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production direct ImageNet extraction requires CUDA")
    output, cache, image_root = Path(output), Path(cache), Path(image_root)
    output.mkdir(parents=True, exist_ok=True)
    model, config, weights = _network(spec, cache, target, os.environ.get("HF_TOKEN"))
    evaluation, preprocessing = evaluation_transform(spec, config)
    normalize = {"mean": preprocessing["mean"], "std": preprocessing["std"]}
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    results = {}
    try:
        for split in ("train", "val"):
            directory = image_root / split
            stats = imagenet_folder_stats(directory)
            dataset = ImageFolder(str(directory))
            if stats["classes"] != 1000 or len(dataset) != stats["images"]:
                raise RuntimeError(f"{directory}: incomplete ImageNet class/sample inventory")
            augment = split == "train" and spec.recipe != "mrl"
            transform = transforms.Compose([
                transforms.RandomResizedCrop(spec.input_size,
                    interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.Normalize(normalize["mean"], normalize["std"]),
            ]) if augment else evaluation
            ids = [Path(path).relative_to(directory).as_posix() for path, _ in dataset.samples]
            identity = {"recipe": RECIPE, "backbone": spec.name, "split": split,
                        "dim": spec.dim, "input_size": spec.input_size, "weights": weights,
                        "source": {"fingerprint": stats["fingerprint"], "rows": len(dataset),
                                   "dataset_revision": (read_json(image_root / "_mpsae" /
                                       f"prepare_{'validation' if split == 'val' else split}.json") or {}).get("revision")},
                        "normalize": normalize, "preprocessing": preprocessing,
                        "augmentation": "per_image_seeded_random_resized_crop_hflip" if augment else "none",
                        "seed": seed, "order": "sorted_relative_path", "drop_last": False,
                        "ids_sha256": canonical_hash(ids), "precision": "autocast_fp16_stored_float32",
                        "tf32": False, "libraries": library_versions()}
            path = output / f"1K_{split}_{spec.name}.npz"
            existing = load_current_archive(path, identity=identity)
            ids_path = path.with_suffix(".ids.json")
            if existing is not None:
                if read_json(ids_path) != ids:
                    raise RuntimeError(f"{ids_path}: missing or inconsistent sample IDs")
                reclaim_verified_writer(cache, existing)
                results[split] = existing
                continue
            # Reserve writer + atomic archive. Completed writers are reclaimed.
            require_disk(output, len(dataset) * (spec.dim * 4 + 8) * 2,
                         reserve_bytes=10 << 30, what=f"direct {spec.name}/{split}")
            with FeatureSink(cache, rows=len(dataset), dim=spec.dim, identity=identity,
                             stage=f"direct/{spec.name}/{split}") as sink:
                if sink.remaining:
                    loader = DataLoader(SeededImages(dataset, transform, seed, sink.cursor),
                                        batch_size=batch_size, num_workers=workers,
                                        shuffle=False, drop_last=False, pin_memory=True)
                    progress = Progress(len(dataset), done=sink.cursor,
                                        stage=f"direct/{spec.name}/{split}", what="images")
                    with torch.inference_mode():
                        for images, labels in loader:
                            features = forward_batches(model, images, spec, target)
                            sink.write(features, labels.numpy())
                            progress.advance(len(labels))
                features, labels = sink.publish()
            atomic_json(ids_path, ids)
            manifest = finalize_archive(path, features, labels, manifest={
                **identity, "identity": canonical_hash(identity),
                "sample_ids_file": ids_path.name, "sample_ids_sha256": sha256_file(ids_path),
                "deviations": ["Direct PIL decoding: no FFCV max-resolution 500 / mixed JPEG recompression; "
                               "all rows retained; sorted order; per-image seeded timm train augmentation. "
                               "Not the historical FFCV reproduction protocol."],
            })
            if sha256_file(path) != manifest["sha256"]:
                raise RuntimeError(f"{path}: archive verification failed")
            release_sink(sink, features, labels)
            results[split] = manifest
    finally:
        del model
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
        torch.cuda.empty_cache()
    return results


def forward_batches(model, images, spec, target):
    """OOM retry only changes inference microbatching, never sample selection."""
    import numpy as np
    import torch

    size = len(images)
    while True:
        try:
            chunks = []
            for start in range(0, len(images), size):
                batch = images[start:start + size].to(target, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    value = model(batch) if spec.recipe == "mrl" else model.forward_head(
                        model.forward_features(batch), pre_logits=True)
                if value.ndim != 2 or value.shape[1] != spec.dim:
                    raise RuntimeError(f"{spec.name}: unexpected feature shape {tuple(value.shape)}")
                chunks.append(value.float().cpu().numpy())
                del value, batch
            return np.concatenate(chunks)
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            chunks.clear()
            # Remove the failed operation's tensors before retrying.
            if "value" in locals():
                del value
            if "batch" in locals():
                del batch
            torch.cuda.empty_cache()
            size = max(1, size // 2)
