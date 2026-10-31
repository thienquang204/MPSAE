"""Create the immutable, class-balanced training-only repair pilot pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

from repair_data import PROTOCOL, atomic_json, canonical_hash, exclusive_lock, inspect_npz, load_splits, sha256_file


def _rng(source_sha256, purpose):
    import numpy as np
    material = f"{PROTOCOL}/splits/v1/{source_sha256}/{purpose}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:16], "little"))


def build_indices(labels, source_sha256, *, fit_per_class=128, cheap_per_class=2,
                  holdout_per_class=8, anchor_count=1024):
    import numpy as np

    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("Expected a one-dimensional integer label array")
    classes = np.unique(labels)
    if not len(classes) or min(fit_per_class, cheap_per_class, holdout_per_class) < 1:
        raise ValueError("Every class needs positive fit, cheap, and holdout allocations")
    if anchor_count < len(classes):
        raise ValueError("Anchor bank must contain at least one example per represented class")
    pools = {key: [] for key in ("fit_indices", "dev_cheap_indices", "dev_holdout_indices", "anchor_indices")}
    remainder = []
    per_class = fit_per_class + cheap_per_class + holdout_per_class
    # One stable sort avoids scanning 1.28M labels independently 1,000 times.
    order = np.argsort(labels, kind="stable")
    ordered_labels = labels[order]
    starts = np.searchsorted(ordered_labels, classes, side="left")
    ends = np.searchsorted(ordered_labels, classes, side="right")
    for class_id, start, end in zip(classes, starts, ends):
        members = order[start:end].copy()
        if len(members) < per_class + 1:
            raise ValueError(f"Class {class_id} has {len(members)} rows; needs at least {per_class + 1}")
        _rng(source_sha256, f"class/{int(class_id)}").shuffle(members)
        position = 0
        for key, count in (("fit_indices", fit_per_class), ("dev_cheap_indices", cheap_per_class), ("dev_holdout_indices", holdout_per_class), ("anchor_indices", 1)):
            pools[key].append(members[position:position + count])
            position += count
        remainder.append(members[position:])
    remaining = np.concatenate(remainder)
    extra = anchor_count - len(classes)
    if extra > len(remaining):
        raise ValueError("Not enough unused rows for the requested additional anchors")
    # The extra anchors are globally sampled once; no class label is a target.
    if extra:
        selection = _rng(source_sha256, "extra-anchors").choice(len(remaining), size=extra, replace=False)
        pools["anchor_indices"].append(remaining[selection])
    result = {}
    for name, chunks in pools.items():
        result[name] = np.concatenate(chunks).astype(np.int64, copy=False)
        _rng(source_sha256, f"pool-order/{name}").shuffle(result[name])
    result["gallery_indices"] = result["fit_indices"].copy()
    result["dev_full_indices"] = np.concatenate([result["dev_cheap_indices"], result["dev_holdout_indices"]])
    return result


def prepare_splits(train_npz, out_dir, *, fit_per_class=128, cheap_per_class=2,
                   holdout_per_class=8, anchor_count=1024, expected_source=None):
    import numpy as np

    # Configured research preparation authenticates the archive before writing
    # split files. The standalone synthetic-data utility remains available
    # without expected_source and keeps its explicit provenance warnings.
    if expected_source is not None:
        for key, length in (("revision", 40), ("train_sha256", 64)):
            value = expected_source.get(key)
            if (not isinstance(value, str) or len(value) != length or
                any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError(f"Pin source.{key} from trusted release metadata before preparing research splits")
        if (expected_source["repository"] != "W1nd-navigator/CSR-precompute-embeds" or
            expected_source["parent"] != "RAIVNLab/MRL:r50_mrl0_e0_ff2048.pt" or
            expected_source["train_file"] != "FF2048_RN50_Embeds/1K_train_ff2048.npz" or
            expected_source["expected_rows"] != 1281167 or expected_source["dimension"] != 2048 or
            expected_source["classes"] != 1000 or Path(train_npz).name != "1K_train_ff2048.npz"):
            raise ValueError("Configured research splits require the declared released FF2048 training source")
    source, labels = inspect_npz(train_npz)
    if expected_source is not None:
        if source["sha256"] != expected_source["train_sha256"]:
            raise ValueError("Training archive SHA-256 differs from the trusted configured release digest")
        if source["feature_shape"] != [1281167, 2048] or source["classes"] != 1000:
            raise ValueError("Released FF2048 training shape/class count mismatch")
    parameters = dict(fit_per_class=fit_per_class, cheap_per_class=cheap_per_class,
                      holdout_per_class=holdout_per_class, anchor_count=anchor_count)
    provenance_warnings = []
    if source["feature_shape"] != [1281167, 2048]:
        provenance_warnings.append("Nonreference feature shape: this split is suitable for a labeled synthetic/research check, not strict FF2048 training.")
    if source["classes"] != 1000:
        provenance_warnings.append("Nonreference class count: the reference pilot uses 1,000 classes.")
    for message in provenance_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    recipe = {"protocol": PROTOCOL, "source_sha256": source["sha256"], "parameters": parameters}
    recipe_hash = canonical_hash(recipe)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out_dir / "split.lock"):
        manifest_path, indices_path = out_dir / "split_manifest.json", out_dir / "split_indices.npz"
        if manifest_path.exists() or indices_path.exists():
            if not (manifest_path.exists() and indices_path.exists()):
                raise ValueError("Incomplete existing split; inspect it and choose a new output directory")
            _, manifest = load_splits(out_dir, expected_source_sha256=source["sha256"])
            if manifest.get("recipe_sha256") != recipe_hash:
                raise ValueError("Existing split has a different recipe; it will not be overwritten")
            return manifest
        indices = build_indices(labels, source["sha256"], **parameters)
        fd, temporary = tempfile.mkstemp(prefix="split_indices.", suffix=".tmp", dir=out_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, **indices)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, indices_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        manifest = {
            "protocol": PROTOCOL, "namespace": f"{PROTOCOL}/splits/v1", "source": source,
            "recipe_sha256": recipe_hash, "parameters": parameters,
            "indices_sha256": sha256_file(indices_path),
            "counts": {name: len(values) for name, values in indices.items()},
            "distinct_source_rows": sum(len(indices[name]) for name in ("fit_indices", "dev_full_indices", "anchor_indices")),
            "class_ids": np.unique(labels).tolist(), "warnings": provenance_warnings,
            "label_usage": "Class-balanced split allocation and evaluation only; never SAE or neighborhood targets.",
            "row_identity": "Row indices in the hashed training archive; no original image-name mapping is implied.",
            "official_validation_opened": False,
        }
        atomic_json(manifest_path, manifest)
    load_splits(out_dir, expected_source_sha256=source["sha256"])
    return manifest


def prepare_from_config(train_npz, out_dir, config):
    """Use the same locked pool sizes and source pins as the research runner."""
    from make_plan import validate_config

    validate_config(config)
    if config["model"]["hidden_dim"] != 8192:
        raise ValueError("The declared FF2048 research profile requires dictionary width 8192")
    parameters = {key: config["pilot"][key] for key in
                  ("fit_per_class", "cheap_per_class", "holdout_per_class", "anchor_count")}
    return prepare_splits(train_npz, out_dir, **parameters, expected_source=config["source"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Research JSON supplies pool sizes and required trusted source pins; conflicting explicit counts are rejected")
    parser.add_argument("--fit-per-class", type=int)
    parser.add_argument("--cheap-per-class", type=int)
    parser.add_argument("--holdout-per-class", type=int)
    parser.add_argument("--anchor-count", type=int)
    args = parser.parse_args(argv)
    defaults = dict(fit_per_class=128, cheap_per_class=2, holdout_per_class=8, anchor_count=1024)
    if args.config is not None:
        from make_plan import load_config
        config = load_config(args.config)
        for key in defaults:
            value = getattr(args, key)
            if value is not None and value != config["pilot"][key]:
                parser.error(f"--{key.replace('_', '-')} conflicts with the configured pilot pool size")
        manifest = prepare_from_config(args.train_npz, args.out_dir, config)
    else:
        parameters = {key: default if getattr(args, key) is None else getattr(args, key)
                      for key, default in defaults.items()}
        manifest = prepare_splits(args.train_npz, args.out_dir, **parameters)
    print(json.dumps({"protocol": manifest["protocol"], "source_sha256": manifest["source"]["sha256"], "counts": manifest["counts"], "warnings": manifest["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
