"""ImageNet inputs for manually tuned MMRSAE runs.

Only requested splits are opened. ResNet-50 reuses released FF2048 arrays;
ResNet-18 extracts the selected rows once into resumable FP32 memory maps.
Imports of dataset/model libraries are deferred until preparation is requested.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

from repair_data import atomic_json, canonical_hash, exclusive_lock, inspect_npz, open_feature_cache, sha256_file


FF2048_REPO = "W1nd-navigator/CSR-precompute-embeds"
FF2048_PARENT = "RAIVNLab/MRL:r50_mrl0_e0_ff2048.pt"
IMAGENET_REPO = "ILSVRC/imagenet-1k"
COUNTS = {"train": 1_281_167, "validation": 50_000}
DATA_RECIPE = "mmrsae-imagenet-features-v1"


@dataclass
class FeatureBundle:
    train: Any
    train_labels: Any
    query: Any
    query_labels: Any
    fit_ids: Any
    gallery_ids: Any
    query_ids: Any
    anchor_ids: Any
    provenance: dict


def _option(args, name, default=None):
    return getattr(args, name, default)


def _array_hash(values):
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _labels(values, rows, split):
    import numpy as np
    values = np.asarray(values)
    if values.shape != (rows,) or values.dtype.kind not in "iu":
        raise ValueError(f"{split}: expected one integer ImageNet label per row")
    if rows != COUNTS[split]:
        raise ValueError(f"{split}: expected {COUNTS[split]:,} source rows, found {rows:,}")
    if not np.array_equal(np.unique(values), np.arange(1000)):
        raise ValueError(f"{split}: expected all 1,000 class IDs in [0, 999]")
    return values.astype(np.int64, copy=False)


def _balanced_take(labels, available, count, rng):
    """Balanced allocation without scanning the entire label array per class."""
    import numpy as np
    if not 0 <= count <= len(available):
        raise ValueError(f"Requested {count:,} rows from a pool of {len(available):,}")
    if count == 0:
        return np.empty(0, dtype=np.int64), available
    randomized = rng.permutation(available)
    order = np.argsort(labels[randomized], kind="stable")
    grouped = randomized[order]
    sizes = np.bincount(labels[grouped], minlength=1000)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    allocation = np.zeros(1000, dtype=np.int64)
    remainder = count
    class_order = rng.permutation(1000)
    while remainder:
        eligible = class_order[allocation[class_order] < sizes[class_order]]
        share = max(1, remainder // len(eligible))
        for category in eligible:
            take = min(share, int(sizes[category] - allocation[category]), remainder)
            allocation[category] += take
            remainder -= take
            if not remainder:
                break
    selected = np.sort(np.concatenate([
        grouped[offsets[c]:offsets[c] + allocation[c]] for c in range(1000)
    ]))
    remaining = np.setdiff1d(available, selected, assume_unique=True)
    return selected, remaining


def _select_pools(labels, args):
    import numpy as np
    evaluation = _option(args, "evaluation", "dev")
    if evaluation not in ("dev", "official"):
        raise ValueError("evaluation must be dev or official")
    train_size = int(_option(args, "train_size", 128_000))
    dev_size = int(_option(args, "dev_size", 10_000)) if evaluation == "dev" else 0
    anchor_count = int(_option(args, "anchor_count", 1024))
    if train_size < 0 or dev_size < 0 or anchor_count < 2 or (evaluation == "dev" and not dev_size):
        raise ValueError("Require train-size >= 0, a positive dev-size, and anchor-count >= 2")
    rng = np.random.default_rng(np.random.SeedSequence([int(_option(args, "data_seed", 42)), 9127]))
    available = np.arange(len(labels), dtype=np.int64)
    query_ids, available = _balanced_take(labels, available, dev_size, rng)
    anchors, remainder = _balanced_take(labels, available, anchor_count, rng)
    fit_ids = available if train_size == 0 else _balanced_take(labels, remainder, train_size, rng)[0]
    if not len(fit_ids):
        raise ValueError("The fitting/gallery pool is empty")
    return {"fit": fit_ids, "query": query_ids, "anchors": anchors}


def _resolved_release(repo, repo_type, requested_revision, args):
    """Pin moving revisions once; subsequent manual runs reuse the receipt."""
    root = Path(args.cache_dir) / "releases"
    identity = {"repository": repo, "repo_type": repo_type, "requested_revision": requested_revision}
    receipt = root / (canonical_hash(identity) + ".json")
    with exclusive_lock(receipt.with_suffix(".lock")):
        if receipt.exists():
            result = json.loads(receipt.read_text(encoding="utf-8"))
            if any(result.get(k) != v for k, v in identity.items()):
                raise RuntimeError(f"Incompatible release receipt: {receipt}")
        else:
            if not args.download:
                raise RuntimeError(f"No pinned release receipt for {repo}; enable --download once")
            from huggingface_hub import HfApi
            try:
                info = HfApi().repo_info(repo, repo_type=repo_type, revision=requested_revision,
                                         files_metadata=True, token=os.environ.get("HF_TOKEN") or None)
            except Exception as error:
                raise RuntimeError(
                    f"Cannot resolve {repo}@{requested_revision}. For ImageNet, accept its access "
                    "conditions and authenticate with HF_TOKEN or `hf auth login`."
                ) from error
            files = {}
            for item in info.siblings or []:
                lfs = getattr(item, "lfs", None)
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                files[item.rfilename] = {"sha256": sha, "size": getattr(item, "size", None)}
            result = {**identity, "revision": info.sha, "files": files}
            if not re.fullmatch(r"[0-9a-f]{40}", str(result["revision"])):
                raise RuntimeError(f"Hub did not provide an immutable commit for {repo}")
            atomic_json(receipt, result)
    if not re.fullmatch(r"[0-9a-f]{40}", str(result.get("revision", ""))):
        raise RuntimeError(f"Invalid immutable revision in {receipt}")
    return result


def _hub_file(release, filename, args, *, require_lfs=False):
    from huggingface_hub import hf_hub_download
    entry = release["files"].get(filename)
    if entry is None:
        raise RuntimeError(f"{filename} is absent from the pinned release")
    expected = entry.get("sha256")
    if require_lfs and not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
        raise RuntimeError(f"No trusted Hub LFS SHA-256 for {filename}; refusing unverifiable release")
    path = Path(hf_hub_download(
        release["repository"], filename, repo_type=release["repo_type"],
        revision=release["revision"], cache_dir=str(Path(args.cache_dir) / "hub"),
        local_files_only=not args.download, token=os.environ.get("HF_TOKEN") or None,
    ))
    if entry.get("size") is not None and path.stat().st_size != entry["size"]:
        raise RuntimeError(f"Downloaded file length differs from release metadata: {path}")
    return path, expected


def _ff2048_split(split, args):
    name = "train" if split == "train" else "val"
    explicit = _option(args, "ff2048_" + name)
    filename = f"FF2048_RN50_Embeds/1K_{name}_ff2048.npz"
    candidate = Path(explicit) if explicit else Path(args.data_root) / filename
    expected = _option(args, "ff2048_" + name + "_sha256") or None
    if expected and not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Invalid --ff2048-{name}-sha256")
    expected = expected.lower() if expected else None
    if explicit and not candidate.is_file():
        raise FileNotFoundError(candidate)
    if candidate.is_file():
        revision = _option(args, "ff2048_revision", "main")
        # A local path/header is not evidence that this is the declared parent.
        # Offline use needs an explicit immutable revision plus a trusted hash,
        # or an existing release receipt from which to recover both values.
        if expected and re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            trust = "user_supplied_sha256_and_immutable_revision"
        else:
            release = _resolved_release(FF2048_REPO, "dataset", revision, args)
            entry = release["files"].get(filename)
            remote_hash = entry.get("sha256") if entry else None
            if not re.fullmatch(r"[0-9a-f]{64}", str(remote_hash)):
                raise RuntimeError(f"No trusted Hub LFS SHA-256 for {filename}; refusing unverifiable local archive")
            if expected and expected != remote_hash:
                raise ValueError("User archive hash differs from the pinned release metadata")
            expected = remote_hash
            revision = release["revision"]
            trust = "pinned_Hub_LFS_sha256"
        source = {"kind": "local_archive", "identity_verified": True,
                  "repository": FF2048_REPO, "revision": revision,
                  "filename": filename, "trust": trust}
    else:
        release = _resolved_release(FF2048_REPO, "dataset", _option(args, "ff2048_revision", "main"), args)
        candidate, remote_hash = _hub_file(release, filename, args, require_lfs=True)
        if expected and expected != remote_hash:
            raise ValueError("User archive hash differs from the pinned release metadata")
        expected = remote_hash
        source = {"kind": "released_archive", "identity_verified": True,
                  "repository": FF2048_REPO, "revision": release["revision"],
                  "filename": filename, "trust": "pinned_Hub_LFS_sha256"}
    # Reject a mistaken local archive from its small header/label members,
    # before streaming a potentially large feature payload onto disk.
    header, header_labels = inspect_npz(candidate, hash_archive=False)
    if header["feature_shape"][1] != 2048 or not str(header["feature_dtype"]).lstrip("<>=|").startswith("f"):
        raise ValueError("FF2048 archive requires floating-point features of width 2048")
    _labels(header_labels, header["feature_shape"][0], split)
    del header_labels
    features, labels, metadata = open_feature_cache(candidate, Path(args.cache_dir) / "ff2048", expected_sha256=expected)
    if features.shape[1] != 2048 or features.dtype.kind != "f":
        raise ValueError(f"FF2048 requires floating-point features of width 2048, got {features.shape}/{features.dtype}")
    labels = _labels(labels, len(features), split)
    return features, labels, {**source, **metadata, "parent": FF2048_PARENT}


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
                dataset = load_dataset(IMAGENET_REPO, split=split, revision=release["revision"],
                                       data_files={split: shards},
                                       cache_dir=str(args.data_root), token=os.environ.get("HF_TOKEN") or True,
                                       download_config=DownloadConfig(max_retries=5, local_files_only=not args.download))
            except Exception as error:
                raise RuntimeError(
                    f"Cannot prepare ImageNet {split}. Accept https://huggingface.co/datasets/{IMAGENET_REPO} "
                    "access conditions and authenticate using HF_TOKEN or `hf auth login`; no alternative dataset is substituted."
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


class _SelectedImages:
    """Worker-safe index-seeded transforms, independent of extraction/resume order."""
    def __init__(self, dataset, ids, query_ids, backend, data_seed):
        from torchvision import transforms
        self.dataset, self.ids, self.backend = dataset, ids, backend
        self.data_seed = int(data_seed)
        self.query_ids = set(int(value) for value in query_ids)
        normalization = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                                         interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), normalization,
        ])
        self.query_transform = transforms.Compose([_EvaluationCrop(), transforms.ToTensor(), normalization])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        import torch
        source_id = int(self.ids[index])
        sample = self.dataset[source_id]
        original = sample["image"] if self.backend == "hf" else sample[0]
        image = None
        try:
            if original is None or min(original.size) < 1:
                raise ValueError(f"Invalid ImageNet image at source row {source_id}")
            image = original.convert("RGB")
            image.load()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.data_seed + source_id)
                transform = self.query_transform if source_id in self.query_ids else self.train_transform
                return transform(image)
        except Exception as error:
            raise RuntimeError(f"Failed decoding/preprocessing ImageNet row {source_id}") from error
        finally:
            if image is not None and image is not original:
                image.close()
            if original is not None and hasattr(original, "close"):
                original.close()


def _collate_images(images):
    """Group the train224/query256 boundary without padding or changing views."""
    import torch
    groups = {}
    for index, image in enumerate(images):
        groups.setdefault(tuple(image.shape), []).append((index, image))
    return [(torch.tensor([row[0] for row in rows], dtype=torch.long),
             torch.stack([row[1] for row in rows])) for rows in groups.values()]


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


def _rn18_features(dataset, source_ids, query_ids, source, weights, parent, args, device):
    identity = {"recipe": DATA_RECIPE, "source": source, "parent": parent,
                "source_ids_sha256": _array_hash(source_ids), "query_ids_sha256": _array_hash(query_ids),
                "data_seed": int(_option(args, "data_seed", 42)),
                "transform": "rrc224_flip0.5_seedPlusSourceRow;query_center224of256_resize256;mean_std0.5;fp32"}
    selected = _SelectedImages(dataset, source_ids, query_ids, args.data_backend, _option(args, "data_seed", 42))

    def batches(cursor):
        import torch
        import timm
        from safetensors.torch import load_file
        from torch.utils.data import DataLoader, Subset
        # Cache hits and cache misses must not change the later SAE RNG stream.
        with torch.random.fork_rng(devices=[]):
            network = timm.create_model("resnet18.a1_in1k", pretrained=False)
        network.load_state_dict(load_file(str(weights)), strict=True)
        network.reset_classifier(0)
        network.eval().requires_grad_(False).to(device)
        loader = DataLoader(Subset(selected, range(cursor, len(selected))), shuffle=False,
                            batch_size=int(_option(args, "feature_batch_size", 256)),
                            num_workers=int(_option(args, "workers", 4)), pin_memory=True,
                            collate_fn=_collate_images,
                            generator=torch.Generator().manual_seed(int(_option(args, "data_seed", 42))),
                            **({"prefetch_factor": 2} if _option(args, "workers", 4) else {}))
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                for groups in loader:
                    values = torch.empty((sum(len(images) for _, images in groups), 512), dtype=torch.float32)
                    for positions, images in groups:
                        output = network(images.to(device, non_blocking=True)).float()
                        if output.shape != (len(images), 512):
                            raise RuntimeError(f"Unexpected ResNet-18 feature shape: {tuple(output.shape)}")
                        values[positions] = output.cpu()
                    yield values.numpy()
        finally:
            del network
    return _materialize(len(source_ids), 512, identity, args, batches)


def prepare_imagenet_features(args, backbone, device):
    """Prepare one backbone after the caller's CUDA retrieval preflight.

    In dev mode query_ids index train and query/query_labels are None. In
    official mode they index the separate validation feature map. Pool IDs are
    always local feature-row IDs; source-row identities are recorded separately.
    """
    import numpy as np
    import torch
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MMRSAE data preparation requires the selected CUDA device; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if backbone not in ("resnet18", "resnet50"):
        raise ValueError("MMRSAE supports resnet18 and resnet50")
    if int(_option(args, "feature_batch_size", 256)) < 1 or int(_option(args, "workers", 4)) < 0:
        raise ValueError("feature-batch-size must be positive and workers nonnegative")
    if int(_option(args, "max_val", 0)) < 0:
        raise ValueError("max-val must be nonnegative")
    query, query_labels, query_source, query_cache = None, None, None, None
    query_source_ids = np.empty(0, dtype=np.int64)
    if backbone == "resnet50":
        train, labels, source = _ff2048_split("train", args)
        pools = _select_pools(labels, args)
        selected = np.unique(np.concatenate(list(pools.values())))
        # The common released FP32 array is memory mapped once, shared across
        # budgets and seeds. Validate only rows this manual run can consume.
        for start in range(0, len(selected), 8192):
            values = np.asarray(train[selected[start:start + 8192]], dtype=np.float32)
            if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
                raise RuntimeError("Invalid/zero FF2048 features in the selected training pools")
        cache = {"path": str(train.filename), "dtype": str(train.dtype), "conversion": "FP32 batches when consumed"}
        source_ids = np.arange(len(train), dtype=np.int64)
        parent = {"identity": FF2048_PARENT, "dimension": 2048, "precision": str(train.dtype)}
    else:
        dataset, source_labels, source = _raw_split("train", args)
        pools = _select_pools(source_labels, args)
        source_ids = np.unique(np.concatenate(list(pools.values())))
        weights, parent = _parent_files(args)
        train, cache = _rn18_features(dataset, source_ids, pools["query"], source, weights, parent, args, device)
        labels = source_labels[source_ids]
        pools = {name: np.searchsorted(source_ids, ids).astype(np.int64) for name, ids in pools.items()}
        del dataset
    if args.evaluation == "official":
        maximum = int(_option(args, "max_val", 0))
        if backbone == "resnet50":
            query, query_labels, query_source = _ff2048_split("validation", args)
            query_cache = {"path": str(query.filename), "dtype": str(query.dtype)}
            ids = np.arange(len(query), dtype=np.int64)
            query_source_ids = ids.copy()
            if maximum:
                ids, _ = _balanced_take(query_labels, ids, min(maximum, len(ids)), np.random.default_rng(_option(args, "data_seed", 42) + 173))
            pools["query"] = ids
            for start in range(0, len(ids), 8192):
                values = np.asarray(query[ids[start:start + 8192]], dtype=np.float32)
                if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
                    raise RuntimeError("Invalid/zero FF2048 validation features")
        else:
            dataset, all_query_labels, query_source = _raw_split("validation", args)
            if query_source["classes"] != source["classes"]:
                raise ValueError("Training and validation class vocabularies/order differ")
            ids = np.arange(len(all_query_labels), dtype=np.int64)
            if maximum:
                ids, _ = _balanced_take(all_query_labels, ids, min(maximum, len(ids)), np.random.default_rng(_option(args, "data_seed", 42) + 173))
            query, query_cache = _rn18_features(dataset, ids, ids, query_source, weights, parent, args, device)
            query_labels = all_query_labels[ids]
            query_source_ids = ids
            pools["query"] = np.arange(len(ids), dtype=np.int64)
            query_source = {**query_source, "selected_source_ids_sha256": _array_hash(ids)}
            del dataset
    split_identity = {"recipe": DATA_RECIPE, "evaluation": args.evaluation, "data_seed": int(_option(args, "data_seed", 42)),
                      "train_size": int(args.train_size), "dev_size": int(args.dev_size),
                      "anchor_count": int(args.anchor_count), "source": source,
                      "official_query_source_ids_sha256": _array_hash(query_source_ids),
                      "source_ids_sha256": _array_hash(source_ids),
                      "pools": {name: {"rows": len(ids), "sha256": _array_hash(ids)} for name, ids in pools.items()}}
    split_directory = Path(args.cache_dir) / "splits" / canonical_hash(split_identity)
    split_directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(split_directory / "write.lock"):
        for name, values in {"source_ids": source_ids, "query_source_ids": query_source_ids, **pools}.items():
            path = split_directory / f"{name}.npy"
            if path.exists():
                if not np.array_equal(np.load(path, allow_pickle=False), values):
                    raise RuntimeError(f"Cached split rows differ: {path}")
            else:
                temporary = path.with_suffix(".partial.npy")
                with temporary.open("wb") as handle:
                    np.save(handle, values, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
        atomic_json(split_directory / "splits.json", split_identity)
    stable_source = {name: value for name, value in source.items() if name not in ("path", "archive_bytes")}
    stable_query = ({name: value for name, value in query_source.items() if name not in ("path", "archive_bytes")}
                    if query_source else None)
    identity = {"recipe": DATA_RECIPE, "backbone": backbone, "parent": parent,
                "train_source": stable_source, "query_source": stable_query,
                "data_seed": int(_option(args, "data_seed", 42)), "evaluation": args.evaluation,
                "source_ids_sha256": _array_hash(source_ids), "query_source_ids_sha256": _array_hash(query_source_ids),
                "pools": split_identity["pools"], "feature_dtype": str(train.dtype),
                "preprocessing": "released_FF2048" if backbone == "resnet50" else cache["identity"]["transform"]}
    provenance = {"identity": identity, "recipe": DATA_RECIPE, "backbone": backbone, "parent": parent,
                  "train_source": source, "query_source": query_source, "feature_cache": cache,
                  "query_cache": query_cache, "splits": split_identity, "split_directory": str(split_directory.resolve()),
                  "source_row_ids": str((split_directory / "source_ids.npy").resolve()),
                  "query_source_row_ids": str((split_directory / "query_source_ids.npy").resolve()),
                  "row_id_semantics": "fit/gallery/anchor IDs index train; dev query IDs index train; official query IDs index query",
                  "anchors_overlap_fit": bool(np.intersect1d(pools["fit"], pools["anchors"]).size),
                  "official_validation_opened": args.evaluation == "official"}
    return FeatureBundle(train, labels, query, query_labels, pools["fit"], pools["fit"].copy(),
                         pools["query"], pools["anchors"], provenance)
