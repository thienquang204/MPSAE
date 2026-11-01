"""Export ResNet-18 ImageNet embeddings once, in the FF2048 archive layout.

Every row uses the evaluation centre crop, so one archive serves any pool
selection or data seed. MMRSAE training reads only the archive and its JSON
receipt (mmrsae_data._rn18_split); raw images are needed here alone.

    python export_rn18_embeddings.py --split all --data-root data/imagenet1k --output-dir data/embeddings
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

from mmrsae_data import IMAGENET_REPO, _array_hash, _hub_file, _labels, _option, _resolved_release
from repair_data import atomic_json, canonical_hash, exclusive_lock, sha256_file


RECIPE = "rn18-imagenet-export-v1"
TRANSFORM = "center_crop224of256_resize256;mean_std0.5;fp32"
ARCHIVES = {"train": "1K_train_rn18.npz", "validation": "1K_val_rn18.npz"}


@contextmanager
def _offline_dataset(enabled):
    """No backend/network fallback when the user requests cached inputs only."""
    if not enabled:
        yield
        return
    from datasets import config as dataset_config
    from huggingface_hub import constants as hub_config
    old_environment = {name: os.environ.get(name) for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")}
    flags = [(dataset_config, "HF_DATASETS_OFFLINE"), (dataset_config, "HF_HUB_OFFLINE"), (hub_config, "HF_HUB_OFFLINE")]
    previous = [(module, key, getattr(module, key)) for module, key in flags if hasattr(module, key)]
    try:
        for name in old_environment:
            os.environ[name] = "1"
        for module, key, _ in previous:
            setattr(module, key, True)
        yield
    finally:
        for module, key, value in previous:
            setattr(module, key, value)
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _raw_split(split, args):
    import numpy as np
    backend = _option(args, "data_backend", "hf")
    if backend == "imagefolder":
        from torchvision.datasets import ImageFolder
        directory = Path(args.data_root) / ("train" if split == "train" else "val")
        dataset = ImageFolder(directory)
        if len(dataset.classes) != 1000 or any(not re.fullmatch(r"n[0-9]{8}", value) for value in dataset.classes):
            raise ValueError("ImageFolder must contain all 1,000 ImageNet synset directories (n########)")
        labels = _labels(np.asarray(dataset.targets, dtype=np.int64), len(dataset), split)
        digest = hashlib.sha256()
        for path, label in dataset.samples:
            stat = Path(path).stat()
            digest.update(f"{Path(path).relative_to(directory).as_posix()}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        identity = {"backend": backend, "path": str(directory.resolve()), "fingerprint": digest.hexdigest(),
                    "fingerprint_kind": "relative_paths_labels_sizes_mtimes", "classes": dataset.classes}
    elif backend == "hf":
        from datasets import DownloadConfig, load_dataset
        release = _resolved_release(IMAGENET_REPO, "dataset", _option(args, "imagenet_revision", "main"), args)
        # Giving split= alone can still prepare every split in a dataset
        # builder. Restrict the shard manifest itself before any download.
        shards = sorted(name for name in release["files"]
                        if name.endswith(".parquet") and re.match(rf"{split}(?:-|\.)", Path(name).name))
        if not shards:
            raise RuntimeError(f"Pinned ImageNet release has no recognized {split} Parquet shards")
        with _offline_dataset(not args.download):
            try:
                # The repo metadata lists train/validation/test, so the builder's
                # split check rejects a single-split manifest; _labels() below
                # verifies the exact row count and class vocabulary instead.
                dataset = load_dataset(IMAGENET_REPO, split=split, revision=release["revision"],
                                       data_files={split: shards}, verification_mode="no_checks",
                                       cache_dir=str(args.data_root), token=os.environ.get("HF_TOKEN") or True,
                                       download_config=DownloadConfig(max_retries=5, local_files_only=not args.download))
            except Exception as error:
                raise RuntimeError(
                    f"Cannot prepare ImageNet {split} ({type(error).__name__}: {error}). If this is an access error, "
                    f"accept https://huggingface.co/datasets/{IMAGENET_REPO} access conditions and authenticate using "
                    "HF_TOKEN or `hf auth login`; no alternative dataset is substituted."
                ) from error
        if not {"image", "label"}.issubset(dataset.column_names):
            raise ValueError("ImageNet source must contain image and label columns")
        labels = _labels(np.asarray(dataset.select_columns(["label"])["label"]), len(dataset), split)
        names = list(getattr(dataset.features["label"], "names", []))
        if len(names) != 1000:
            raise ValueError("ImageNet source must declare the 1,000-class label vocabulary")
        identity = {"backend": backend, "repository": IMAGENET_REPO, "revision": release["revision"],
                    "fingerprint": dataset._fingerprint, "classes": names,
                    "shards_sha256": canonical_hash({name: release["files"][name] for name in shards})}
    else:
        raise ValueError("data-backend must be hf or imagefolder")
    identity.update(split=split, rows=len(labels), labels_sha256=_array_hash(labels),
                    pixel_validation="every_selected_image_during_extraction")
    return dataset, labels, identity


def _parent_files(args):
    import timm
    import torch
    import torchvision
    release = _resolved_release("timm/resnet18.a1_in1k", "model", _option(args, "parent_revision", "main"), args)
    weights, expected = _hub_file(release, "model.safetensors", args, require_lfs=True)
    configuration, _ = _hub_file(release, "config.json", args)
    observed = sha256_file(weights)
    if observed != expected:
        raise RuntimeError("ResNet-18 weight checksum differs from its pinned release")
    identity = {"repository": release["repository"], "revision": release["revision"],
                "weights_sha256": observed, "config_sha256": sha256_file(configuration),
                "dimension": 512, "precision": "float32", "tf32": False, "timm": timm.__version__,
                "torchvision": torchvision.__version__, "torch": str(torch.__version__)}
    return weights, identity


class _EvaluationCrop:
    def __call__(self, image):
        from torchvision.transforms import functional as fn, InterpolationMode
        width, height = image.size
        side = max(1, int(min(width, height) * 224 / 256))
        image = fn.crop(image, (height - side) // 2, (width - side) // 2, side, side)
        return fn.resize(image, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)


class _Images:
    """Deterministic centre-crop views in source-row order."""
    def __init__(self, dataset, backend):
        from torchvision import transforms
        self.dataset, self.backend = dataset, backend
        self.transform = transforms.Compose([_EvaluationCrop(), transforms.ToTensor(),
                                             transforms.Normalize((0.5,) * 3, (0.5,) * 3)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, source_id):
        sample = self.dataset[source_id]
        original = sample["image"] if self.backend == "hf" else sample[0]
        image = None
        try:
            if original is None or min(original.size) < 1:
                raise ValueError(f"Invalid ImageNet image at source row {source_id}")
            image = original.convert("RGB")
            image.load()
            return self.transform(image)
        except Exception as error:
            raise RuntimeError(f"Failed decoding/preprocessing ImageNet row {source_id}") from error
        finally:
            if image is not None and image is not original:
                image.close()
            if original is not None and hasattr(original, "close"):
                original.close()


def _materialize(rows, dimension, identity, args, make_batches):
    """Resume FP32 writes from durable cursors; publish only complete maps."""
    import numpy as np
    signature = canonical_hash(identity)
    directory = Path(args.cache_dir) / "features" / signature
    directory.mkdir(parents=True, exist_ok=True)
    final, partial, receipt = directory / "features.npy", directory / "features.partial.npy", directory / "features.json"
    shape = (rows, dimension)
    with exclusive_lock(directory / "extract.lock"):
        metadata = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        if metadata and metadata.get("signature") != signature:
            raise RuntimeError(f"Feature cache identity mismatch: {receipt}")
        cursor = int(metadata.get("rows_complete", 0))
        if not 0 <= cursor <= rows:
            raise RuntimeError(f"Invalid durable extraction cursor: {receipt}")
        if final.exists() and not metadata.get("complete"):
            if cursor != rows or partial.exists():
                raise RuntimeError(f"Uncommitted feature cache: {final}")
            metadata.update(complete=True, bytes=final.stat().st_size, mtime_ns=final.stat().st_mtime_ns)
            atomic_json(receipt, metadata)
        if metadata.get("complete"):
            if not final.is_file() or final.stat().st_size != metadata["bytes"] or final.stat().st_mtime_ns != metadata["mtime_ns"]:
                raise RuntimeError(f"Completed feature cache is missing or changed: {final}")
            array = np.load(final, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32 or cursor != rows:
                raise RuntimeError(f"Malformed completed feature cache: {final}")
            callback = _option(args, "progress_callback")
            if callback:
                callback({"phase": "features", "rows_complete": rows, "rows": rows, "cache_hit": True})
            return array, metadata
        if cursor and not partial.exists():
            raise RuntimeError(f"Partial feature data missing at durable cursor {cursor}: {partial}")
        if not partial.exists():
            required = rows * dimension * 4 + (512 << 20)
            if shutil.disk_usage(directory).free < required:
                raise RuntimeError(f"Insufficient free disk for feature cache ({required:,} bytes including reserve)")
            array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
        else:
            array = np.load(partial, mmap_mode="r+", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise RuntimeError(f"Malformed partial feature cache: {partial}")
        metadata.update(signature=signature, identity=identity, complete=False, rows_complete=cursor, rows=rows)
        atomic_json(receipt, metadata)
        started, prior = time.monotonic(), float(metadata.get("seconds", 0))
        for batch_number, values in enumerate(make_batches(cursor)):
            values = np.asarray(values, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != dimension or cursor + len(values) > rows or not np.isfinite(values).all():
                raise RuntimeError("Parent produced malformed/nonfinite feature rows")
            if (np.square(values.astype(np.float64)).sum(axis=1) == 0).any():
                raise RuntimeError("Parent produced zero-norm feature rows; exact normalized retrieval would be invalid")
            stop = cursor + len(values)
            array[cursor:stop] = values
            cursor = stop
            if (batch_number + 1) % 25 == 0 or cursor == rows:
                array.flush()
                with partial.open("r+b") as handle:
                    os.fsync(handle.fileno())
                metadata.update(rows_complete=cursor, seconds=prior + time.monotonic() - started)
                atomic_json(receipt, metadata)
                print(f"ImageNet feature cache: {cursor:,}/{rows:,} rows", flush=True)
                callback = _option(args, "progress_callback")
                if callback:
                    callback({"phase": "features", "rows_complete": cursor, "rows": rows,
                              "seconds": metadata["seconds"], "cache_hit": False})
        if cursor != rows:
            raise RuntimeError(f"Extraction ended early: {cursor}/{rows}")
        del array
        os.replace(partial, final)
        metadata.update(complete=True, bytes=final.stat().st_size, mtime_ns=final.stat().st_mtime_ns)
        atomic_json(receipt, metadata)
        return np.load(final, mmap_mode="r", allow_pickle=False), metadata


def _extract(dataset, rows, source, weights, parent, args):
    identity = {"recipe": RECIPE, "source": source, "parent": parent, "transform": TRANSFORM}
    images = _Images(dataset, args.data_backend)

    def batches(cursor):
        import torch
        import timm
        from safetensors.torch import load_file
        from torch.utils.data import DataLoader, Subset
        network = timm.create_model("resnet18.a1_in1k", pretrained=False)
        network.load_state_dict(load_file(str(weights)), strict=True)
        network.reset_classifier(0)
        network.eval().requires_grad_(False).to(args.device)
        loader = DataLoader(Subset(images, range(cursor, rows)), shuffle=False,
                            batch_size=args.feature_batch_size, num_workers=args.workers, pin_memory=True,
                            **({"prefetch_factor": 2} if args.workers else {}))
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                for batch in loader:
                    output = network(batch.to(args.device, non_blocking=True)).float()
                    if output.shape != (len(batch), 512):
                        raise RuntimeError(f"Unexpected ResNet-18 feature shape: {tuple(output.shape)}")
                    yield output.cpu().numpy()
        finally:
            del network
    return _materialize(rows, 512, identity, args, batches)


def export_split(split, args):
    """Write <output>/<archive>.npz (data.npy, label.npy) and its .json receipt."""
    import numpy as np
    archive = args.output_dir / ARCHIVES[split]
    receipt = archive.with_suffix(".json")
    if archive.is_file() and receipt.is_file():
        manifest = json.loads(receipt.read_text(encoding="utf-8"))
        if manifest.get("sha256") != sha256_file(archive):
            raise RuntimeError(f"{archive} differs from its receipt; move both aside and export again")
        print(f"RN18 {split} embeddings already exported: {archive}", flush=True)
        return manifest
    dataset, labels, source = _raw_split(split, args)
    weights, parent = _parent_files(args)
    features, cache = _extract(dataset, len(labels), source, weights, parent, args)
    del dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, data=features, label=labels)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, archive)
    manifest = {"recipe": RECIPE, "split": split, "rows": len(labels), "dimension": 512,
                "transform": TRANSFORM, "parent": parent, "source": source,
                "labels_sha256": _array_hash(labels), "feature_cache": cache.get("signature"),
                "sha256": sha256_file(archive)}
    atomic_json(receipt, manifest)
    os.chmod(receipt, 0o644)  # Receipts travel with the archives; keep them copyable.
    print(f"RN18 {split} embeddings exported: {archive} ({len(labels):,} x 512)", flush=True)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data/imagenet1k"), help="Raw ImageNet (HF cache or ImageFolder)")
    parser.add_argument("--data-backend", choices=("hf", "imagefolder"), default="hf")
    parser.add_argument("--output-dir", type=Path, default=Path("data/embeddings"), help="Archive directory")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/mmrsae"), help="Release receipts, weights and resumable extraction")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imagenet-revision", default="main")
    parser.add_argument("--parent-revision", default="main")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.feature_batch_size < 1 or args.workers < 0:
        parser.error("--feature-batch-size must be positive and --workers nonnegative")
    return args


def main(argv=None):
    import torch
    args = parse_args(argv)
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("RN18 export requires a CUDA device")
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for split in (("train", "validation") if args.split == "all" else (args.split,)):
        export_split(split, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
