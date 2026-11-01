"""ImageNet inputs for manually tuned MMRSAE runs.

Only requested splits are opened. Local NPZ/JSON pairs take precedence;
missing pairs are downloaded individually from the configured embeddings
repository. Raw ImageNet extraction is available only in explicit legacy mode.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from repair_data import atomic_json, canonical_hash, exclusive_lock, inspect_npz, open_feature_cache


FF2048_REPO = "W1nd-navigator/CSR-precompute-embeds"
EMBEDDINGS_REPO = "quangnguyenopt/NSFW"
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
                    f"Cannot resolve {repo}@{requested_revision}. Check repository access and "
                    "authenticate with HF_TOKEN or `hf auth login`. Raw ImageNet access "
                    "conditions apply only to the explicit legacy image workflow."
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


def _hub_file(release, filename, args, *, require_lfs=False, local_dir=None):
    from huggingface_hub import hf_hub_download
    entry = release["files"].get(filename)
    if entry is None:
        raise RuntimeError(f"{filename} is absent from the pinned release")
    expected = entry.get("sha256")
    if require_lfs and not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
        raise RuntimeError(f"No trusted Hub LFS SHA-256 for {filename}; refusing unverifiable release")
    location = ({"local_dir": str(local_dir)} if local_dir is not None else
                {"cache_dir": str(Path(args.cache_dir) / "hub")})
    path = Path(hf_hub_download(
        release["repository"], filename, repo_type=release["repo_type"],
        revision=release["revision"], **location,
        local_files_only=not args.download, token=os.environ.get("HF_TOKEN") or None,
    ))
    if entry.get("size") is not None and path.stat().st_size != entry["size"]:
        raise RuntimeError(f"Downloaded file length differs from release metadata: {path}")
    return path, expected


def _validate_receipt(manifest, backbone, split):
    """Keep the original parent/preprocessing identity when using a mirror."""
    if (not isinstance(manifest, dict) or manifest.get("split") != split
            or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256")))):
        raise ValueError(f"Invalid {backbone}/{split} embedding receipt: split or SHA-256")
    if backbone == "resnet50":
        name = "train" if split == "train" else "val"
        if (manifest.get("repository") != FF2048_REPO
                or manifest.get("filename") != f"FF2048_RN50_Embeds/1K_{name}_ff2048.npz"
                or manifest.get("parent") != FF2048_PARENT
                or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("revision")))):
            raise ValueError("FF2048 receipt does not identify the released ResNet-50 parent")
    else:
        parent = manifest.get("parent", {})
        if (manifest.get("recipe") != "rn18-imagenet-export-v1"
                or manifest.get("dimension") != 512 or manifest.get("rows") != COUNTS[split]
                or not manifest.get("transform") or not isinstance(manifest.get("source"), dict)
                or not isinstance(parent, dict) or parent.get("dimension") != 512
                or parent.get("repository") != "timm/resnet18.a1_in1k"):
            raise ValueError("RN18 receipt does not identify the exported ImageNet features and parent")
    return manifest["sha256"]


def _precomputed_split(split, args, backbone):
    """Local pair or two targeted Hub downloads; never download raw images."""
    name = "train" if split == "train" else "val"
    suffix = "ff2048" if backbone == "resnet50" else "rn18"
    explicit = _option(args, suffix + "_" + name)
    filename = f"1K_{name}_{suffix}.npz"
    candidate = Path(explicit) if explicit else Path(args.embeddings_dir) / filename
    receipt = candidate.with_suffix(".json")
    if explicit and not candidate.is_file():
        raise FileNotFoundError(candidate)
    # Preserve the older authenticated local FF2048 path without a sidecar.
    if explicit and backbone == "resnet50" and not receipt.is_file():
        return _ff2048_split(split, args)
    if not candidate.is_file() or not receipt.is_file():
        if not args.download:
            raise FileNotFoundError(f"Missing embedding pair {candidate} / {receipt.name}; "
                                    "copy the NPZ and JSON together or enable --download")
        repo = _option(args, "embeddings_repo", EMBEDDINGS_REPO)
        release = _resolved_release(repo, _option(args, "embeddings_repo_type", "model"),
                                    _option(args, "embeddings_revision", "main"), args)
        receipt_path, _ = _hub_file(release, Path(filename).with_suffix(".json").name,
                                    args, local_dir=args.embeddings_dir)
        manifest = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = _validate_receipt(manifest, backbone, split)
        entry = release["files"].get(filename, {})
        if entry.get("sha256") != expected:
            raise ValueError(f"Hub archive SHA-256 differs from {receipt_path.name}")
        if not candidate.is_file():
            candidate, _ = _hub_file(release, filename, args, require_lfs=True,
                                      local_dir=args.embeddings_dir)
        receipt = receipt_path
    manifest = json.loads(receipt.read_text(encoding="utf-8"))
    expected = _validate_receipt(manifest, backbone, split)
    override = _option(args, "ff2048_" + name + "_sha256") if backbone == "resnet50" else None
    if override and override.lower() != expected:
        raise ValueError("Explicit FF2048 SHA-256 differs from the embedding receipt")
    width = 2048 if backbone == "resnet50" else 512
    features, labels, metadata = _open_archive(candidate, expected, split, width,
                                               Path(args.cache_dir) / suffix)
    if backbone == "resnet18" and manifest.get("labels_sha256") != _array_hash(labels):
        raise ValueError("RN18 labels differ from their export receipt")
    return features, labels, {**metadata, "kind": "precomputed_archive", "identity_verified": True,
        "filename": filename, "trust": "archive_receipt_sha256", "receipt": manifest,
        "parent": manifest["parent"], "transform": manifest.get("transform", "released_FF2048")}


def _ff2048_split(split, args):
    name = "train" if split == "train" else "val"
    explicit = _option(args, "ff2048_" + name)
    filename = f"FF2048_RN50_Embeds/1K_{name}_ff2048.npz"
    candidate = Path(explicit) if explicit else Path(args.embeddings_dir) / Path(filename).name
    expected = _option(args, "ff2048_" + name + "_sha256") or None
    if expected and not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Invalid --ff2048-{name}-sha256")
    expected = expected.lower() if expected else None
    if explicit and not candidate.is_file():
        raise FileNotFoundError(candidate)
    receipt = candidate.with_suffix(".json")
    if candidate.is_file():
        revision = _option(args, "ff2048_revision", "main")
        # A local path/header is not evidence that this is the declared parent.
        # Offline use needs an immutable revision plus a trusted hash: from the
        # receipt beside the archive, explicit options, or a release receipt.
        if not expected and receipt.is_file():
            manifest = json.loads(receipt.read_text(encoding="utf-8"))
            if (manifest.get("repository") != FF2048_REPO or manifest.get("filename") != filename
                    or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256")))
                    or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("revision")))):
                raise ValueError(f"Invalid FF2048 receipt: {receipt}")
            expected, revision, trust = manifest["sha256"], manifest["revision"], "receipt_sha256_and_immutable_revision"
        elif expected and re.fullmatch(r"[0-9a-f]{40}", str(revision)):
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
    features, labels, metadata = _open_archive(candidate, expected, split, 2048, Path(args.cache_dir) / "ff2048")
    if source["kind"] == "local_archive" and not receipt.exists():
        # Verified against the Hub: record it so copies of the embeddings
        # folder verify offline, like the RN18 export receipts.
        try:
            atomic_json(receipt, {"repository": FF2048_REPO, "revision": source["revision"], "filename": filename,
                                  "split": split, "sha256": expected, "parent": FF2048_PARENT})
            os.chmod(receipt, 0o644)
        except OSError as error:
            print(f"Could not write FF2048 receipt {receipt}: {error}", flush=True)
    return features, labels, {**source, **metadata, "parent": FF2048_PARENT}


def _rn18_split(split, args):
    """Read an archive written by export_rn18_embeddings.py, trusted via its receipt hash."""
    name = "train" if split == "train" else "val"
    explicit = _option(args, "rn18_" + name)
    candidate = Path(explicit) if explicit else Path(args.embeddings_dir) / f"1K_{name}_rn18.npz"
    receipt = candidate.with_suffix(".json")
    if not candidate.is_file() or not receipt.is_file():
        if explicit:
            raise FileNotFoundError(f"Missing ResNet-18 embeddings {candidate} or its receipt {receipt.name}")
        # Legacy extraction now happens after the Python runner's GPU preflight
        # and honors CLI overrides as well as environment-derived settings.
        from export_rn18_embeddings import export_split
        import copy
        export_args = copy.copy(args)
        export_args.output_dir = Path(args.embeddings_dir)
        export_args.data_root = Path(_option(args, "data_root", "data/imagenet1k"))
        export_args.data_backend = _option(args, "data_backend", "hf")
        export_args.feature_batch_size = _option(args, "feature_batch_size", 256)
        export_args.workers = _option(args, "workers", 4)
        export_split(split, export_args)
    manifest = json.loads(receipt.read_text(encoding="utf-8"))
    expected = manifest.get("sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or manifest.get("split") != split:
        raise ValueError(f"Invalid ResNet-18 export receipt: {receipt}")
    features, labels, metadata = _open_archive(candidate, expected, split, 512, Path(args.cache_dir) / "rn18")
    source = {"kind": "exported_archive", "identity_verified": True, "filename": candidate.name,
              "trust": "export_receipt_sha256", "recipe": manifest["recipe"], "image_source": manifest["source"],
              "transform": manifest["transform"]}
    return features, labels, {**source, **metadata, "parent": manifest["parent"]}


def _open_archive(candidate, expected, split, width, cache_dir):
    # Reject a mistaken local archive from its small header/label members,
    # before streaming a potentially large feature payload onto disk.
    header, header_labels = inspect_npz(candidate, hash_archive=False)
    if header["feature_shape"][1] != width or not str(header["feature_dtype"]).lstrip("<>=|").startswith("f"):
        raise ValueError(f"{candidate.name} requires floating-point features of width {width}")
    _labels(header_labels, header["feature_shape"][0], split)
    del header_labels
    features, labels, metadata = open_feature_cache(candidate, cache_dir, expected_sha256=expected)
    if features.shape[1] != width or features.dtype.kind != "f":
        raise ValueError(f"{candidate.name} requires floating-point features of width {width}, got {features.shape}/{features.dtype}")
    return features, _labels(labels, len(features), split), metadata


def _check_rows(features, ids, what):
    import numpy as np
    batch_rows = max(1, (8 << 20) // (features.shape[1] * max(features.dtype.itemsize, 4)))
    for start in range(0, len(ids), batch_rows):
        values = np.asarray(features[ids[start:start + batch_rows]], dtype=np.float32)
        if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
            raise RuntimeError(f"Invalid/zero features in the selected {what}")


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
    if int(_option(args, "max_val", 0)) < 0:
        raise ValueError("max-val must be nonnegative")
    if _option(args, "embedding_source", "precomputed") == "precomputed":
        def read_split(split, settings):
            return _precomputed_split(split, settings, backbone)
    else:
        read_split = _ff2048_split if backbone == "resnet50" else _rn18_split
    query, query_labels, query_source, query_cache = None, None, None, None
    query_source_ids = np.empty(0, dtype=np.int64)
    train, labels, source = read_split("train", args)
    pools = _select_pools(labels, args)
    # The archive is memory mapped once, shared across budgets and seeds.
    # Validate only rows this manual run can consume.
    _check_rows(train, np.unique(np.concatenate(list(pools.values()))), "training pools")
    cache = {"path": str(train.filename), "dtype": str(train.dtype), "conversion": "FP32 batches when consumed"}
    source_ids = np.arange(len(train), dtype=np.int64)
    parent = ({"identity": FF2048_PARENT, "dimension": 2048, "precision": str(train.dtype)}
              if backbone == "resnet50" else source["parent"])
    if args.evaluation == "official":
        maximum = int(_option(args, "max_val", 0))
        query, query_labels, query_source = read_split("validation", args)
        query_cache = {"path": str(query.filename), "dtype": str(query.dtype)}
        ids = np.arange(len(query), dtype=np.int64)
        query_source_ids = ids.copy()
        if maximum:
            ids, _ = _balanced_take(query_labels, ids, min(maximum, len(ids)), np.random.default_rng(_option(args, "data_seed", 42) + 173))
        pools["query"] = ids
        _check_rows(query, ids, "validation queries")
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
                "preprocessing": "released_FF2048" if backbone == "resnet50" else source["transform"]}
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
