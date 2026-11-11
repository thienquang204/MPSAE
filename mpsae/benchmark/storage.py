"""Dataset lifecycle for direct campaigns. Deletion requires verified backup.

Only explicit owned directories can be collected. The legacy workspace is
read for ImageNet reuse; its fits/reports/features are never overwritten.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import shutil

from ..config import BACKBONES
from ..io import atomic_json, canonical_hash, exclusive_lock, open_npz_memmap, read_json, require_disk, sha256_file
from ..logs import get_logger


def remove_owned(path, owner):
    path, owner = Path(path).absolute(), Path(owner).resolve()
    target = path.resolve()
    if target == owner or not target.is_relative_to(owner) or target != path or path.is_symlink():
        raise ValueError(f"refusing cleanup outside owned directory: {path}")
    # HF snapshots use internal file symlinks. rmtree unlinks these rather
    # than following them; never permit a link resolving outside the owner.
    if path.is_dir():
        for directory, folders, files in os.walk(path, followlinks=False):
            for name in folders + files:
                entry = Path(directory) / name
                if not entry.resolve().is_relative_to(owner):
                    raise ValueError(f"refusing linked cleanup entry: {entry}")
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def backup_before_cleanup(observer, files):
    if observer is None or not hasattr(observer, "verify_backup"):
        raise RuntimeError("cleanup_raw requires verified HF backup: use all_in_one.sh or "
                           "python -m mpsae.remote; local-only runs must set cleanup_raw: false")
    return observer.verify_backup(files)


def reclaim_legacy_ffcv(workspace):
    """Rescue a full server using only stdlib, before CUDA/Python installation."""
    from ..data.imagenet import EXPECTED_ROWS, imagenet_folder_stats
    workspace = Path(workspace).resolve()
    removed = []
    for split, key in (("train", "train"), ("val", "validation")):
        candidates = [workspace / "ffcv" / f"{split}_500_0.50_90{suffix}"
                      for suffix in (".ffcv", ".ffcv.partial", ".ffcv.json")]
        if not any(path.exists() for path in candidates):
            continue
        directory = workspace / "imagenet" / split
        stats = imagenet_folder_stats(directory)
        if stats["images"] != EXPECTED_ROWS[key] or stats["classes"] != 1000:
            raise RuntimeError(f"{directory}: original images are incomplete; refusing to remove FFCV")
        for path in candidates:
            if path.exists():
                size = path.stat().st_size
                remove_owned(path, workspace)
                removed.append({"path": str(path), "bytes": size})
    return {"removed": removed, "reclaimed_bytes": sum(p["bytes"] for p in removed),
            "free_bytes": shutil.disk_usage(workspace).free}


def verified_source(root, name):
    """Validate archives AND row IDs before reusing data after image deletion."""
    from ..embed.direct import RECIPE, reclaim_verified_writer
    from ..data.imagenet import EXPECTED_ROWS
    files = []
    for split in ("train", "val"):
        path = root / "embeddings" / name / f"1K_{split}_{name}.npz"
        metadata = read_json(path.with_name(path.name + ".json"))
        if metadata is None or not path.is_file():
            return None
        if metadata.get("recipe") != RECIPE or metadata.get("backbone") != name:
            raise RuntimeError(f"{path}: wrong extraction protocol/backbone")
        if metadata.get("rows") != EXPECTED_ROWS["validation" if split == "val" else split]:
            raise RuntimeError(f"{path}: incomplete ImageNet row count")
        features, labels, info = open_npz_memmap(path, root / "cache", expected_sha256=metadata["sha256"])
        try:
            if info["rows"] != metadata["rows"] or info["dim"] != BACKBONES[name].dim:
                raise RuntimeError(f"{path}: shape mismatch")
            ids_path = path.with_suffix(".ids.json")
            ids = read_json(ids_path)
            if (not isinstance(ids, list) or len(ids) != len(labels)
                    or canonical_hash(ids) != metadata.get("ids_sha256")
                    or sha256_file(ids_path) != metadata.get("sample_ids_sha256")):
                raise RuntimeError(f"{ids_path}: sample identity verification failed")
        finally:
            features._mmap.close()
        reclaim_verified_writer(root / "cache", metadata)
        files.extend([path, path.with_name(path.name + ".json"), ids_path])
    return files


def validate_source_accuracy(root, name, device):
    """Apply source dense gates while raw images are still recoverable."""
    from ..features import load_bundle
    from ..runs import RunSettings, dense_reference
    from ..train import FitSettings
    spec = BACKBONES[name]
    source = load_bundle(spec, root / "embeddings", root / "cache")
    try:
        dense = dense_reference(spec, source, RunSettings(root=root / "runs", fit=FitSettings(device=device)))
        if spec.reference_dense_top1 is not None and abs(dense["top1"] - spec.reference_dense_top1) > 0.3:
            raise RuntimeError(f"{name}: dense reference gate failed; raw images retained")
        return dense
    finally:
        source.train.features._mmap.close()
        source.validation.features._mmap.close()


def prepare_sources(campaign, workspace, root, *, device, workers, batch_size, observer, task):
    from ..data.imagenet import EXPECTED_ROWS, imagenet_folder_stats, prepare_imagenet
    from ..embed.direct import extract_source, preflight_backbones

    image_root = Path(workspace).resolve() / "imagenet"
    state = root / "embeddings" / "imagenet-lifecycle.json"
    cleanup = campaign.execution["cleanup_raw"]
    if cleanup and (observer is None or not hasattr(observer, "verify_backup")):
        backup_before_cleanup(observer, [])  # fail BEFORE downloads
    with exclusive_lock(Path(workspace) / "state" / "direct-imagenet.lock", timeout=0):
        complete = {name: verified_source(root, name) for name in campaign.backbones}
        missing = [name for name, files in complete.items() if files is None]
        if missing and campaign.execution.get("precomputed_only"):
            raise FileNotFoundError(f"Missing precomputed ImageNet embeddings: {', '.join(missing)}; "
                                    "download and link them with mpsae.embeddings_hub first")
        retired = read_json(image_root / "direct-retired.json")
        if missing and retired:
            raise RuntimeError("ImageNet images were already retired, but selected embeddings are missing. "
                               "Restore verified embeddings from HF first; automatic redownload is disabled.")
        if missing:
            if not campaign.imagenet["prepare"]:
                raise RuntimeError("missing direct embeddings with imagenet.prepare=false")
            if cleanup:
                # Rescue an already full server BEFORE downloading any missing
                # split: discard a redundant FFCV only if its JPEG split is whole.
                reclaim_legacy_ffcv(workspace)
            # The downloader's per-shard receipts reuse previously materialised JPEGs.
            # Bypass the old outer stage receipt so missing images cannot be hidden.
            rows = sum(EXPECTED_ROWS.values())
            needed = sum(rows * (BACKBONES[n].dim * 4 + 8) for n in missing)
            needed += max(rows * (BACKBONES[n].dim * 4 + 8) for n in missing)
            preflight_backbones(missing, root / "cache", device)
            prepare_imagenet(image_root, token=os.environ.get("HF_TOKEN"), reserve_bytes=needed + (10 << 30))
            for split, key in (("train", "train"), ("val", "validation")):
                stats = imagenet_folder_stats(image_root / split)
                if stats["images"] != EXPECTED_ROWS[key] or stats["classes"] != 1000:
                    raise RuntimeError(f"{image_root / split}: incomplete source dataset")
            require_disk(root, needed, reserve_bytes=10 << 30, what="all missing ImageNet embeddings + one writer")
            atomic_json(state, {"state": "downloaded", "backbones": list(campaign.backbones)})
            for name in missing:
                result = task(f"imagenet/embed/{name}", lambda n=name: extract_source(
                    BACKBONES[n], image_root, root / "embeddings" / n, root / "cache",
                    device=device, workers=workers, batch_size=batch_size))
                if result is None:
                    raise RuntimeError(f"{name} extraction failed; retaining all raw ImageNet data")
        files = []
        for name in campaign.backbones:
            checked = verified_source(root, name)
            if not checked:
                raise RuntimeError(f"{name}: source verification incomplete")
            files.extend(checked)
        if not retired:
            for name in campaign.backbones:
                checked = task(f"imagenet/dense-check/{name}", lambda n=name:
                               validate_source_accuracy(root, n, device))
                if checked is None:
                    raise RuntimeError(f"{name}: dense verification failed; retaining raw ImageNet")
        atomic_json(state, {"state": "verified", "backbones": list(campaign.backbones)})
        if cleanup:
            backup = backup_before_cleanup(observer, files)
            # Marker BEFORE deletion makes crash/resume safe, also during rmtree.
            atomic_json(image_root / "direct-retired.json", {"state": "deleting", "backup": backup})
            for split in ("train", "val"):
                remove_owned(image_root / split, image_root)
            # Dataset-owned Hub shard cache only; retain tiny revision receipts.
            remove_owned(image_root / "_mpsae" / "hub", image_root)
            atomic_json(image_root / "direct-retired.json", {"state": "raw_deleted", "backup": backup})
            atomic_json(state, {"state": "raw_deleted", "backbones": list(campaign.backbones), "backup": backup})
            get_logger("storage").info("ImageNet: all selected embeddings verified/backed up; raw images reclaimed")


def save_bundle(bundle, path, root, request):
    splits = {}
    for split in (bundle.gallery, bundle.query):
        splits[split.name] = {"path": str(split.path.resolve().relative_to(root.resolve())),
                              "ids": split.ids, "manifest": split.manifest}
    atomic_json(path, {"request": request, "name": bundle.name, "protocol": bundle.protocol,
                      "backbone": bundle.backbone, "gallery": bundle.gallery.name,
                      "query": bundle.query.name, "exclude_self": bundle.exclude_self,
                      "identity": bundle.identity, "manifest": bundle.manifest, "splits": splits})


def load_bundle_receipt(path, root, request):
    from .extract import FeatureBundle, FeatureSplit
    payload = read_json(path)
    if payload is None or payload.get("request") != request:
        return None
    splits = {}
    for name, entry in payload["splits"].items():
        archive = root / entry["path"]
        if not archive.resolve().is_relative_to(root.resolve()):
            raise ValueError("feature receipt points outside workspace")
        if not archive.is_file():
            return None
        features, labels, _ = open_npz_memmap(archive, root / "cache",
                                            expected_sha256=entry["manifest"]["sha256"])
        if len(entry["ids"]) != len(labels) or canonical_hash(entry["ids"]) != entry["manifest"]["ids_sha256"]:
            features._mmap.close()
            raise RuntimeError(f"{path}: sample IDs disagree with verified archive")
        splits[name] = FeatureSplit(name, features, labels, entry["ids"], entry["manifest"], archive)
    return FeatureBundle(payload["name"], payload["protocol"], payload["backbone"],
                         splits[payload["gallery"]], splits[payload["query"]],
                         payload["exclude_self"], payload["identity"], payload["manifest"])


def run_targets(campaign, root, catalog, *, device, workers, batch_size, observer, task):
    from .data import load_dataset_bundle
    from .extract import extract_bundle
    from .run import checkpoint_record, evaluate_point, verify_feature_provenance
    from ..train import load_fitted
    from ..embed.archive import library_versions

    if campaign.execution["cleanup_raw"] and (observer is None or not hasattr(observer, "verify_backup")):
        backup_before_cleanup(observer, [])

    for name in campaign.downstream["datasets"]:
        # Each target owns its HF cache; no process-global cache deletion.
        raw = root / "cache" / "raw_datasets" / name
        directory = root / "benchmarks" / name
        bundles, requests, receipts, weights = {}, {}, {}, {}
        for backbone in campaign.target_backbones(name):
            candidates = [c for c in campaign.downstream_cells(name) if c.backbone == backbone]
            if not candidates:
                continue
            record = checkpoint_record(candidates[0], root, require_provenance=True)
            weights[backbone] = record["backbone_weights"]
            requests[backbone] = {"backbone": backbone, "weights": weights[backbone],
                                  "revision": campaign.downstream["revisions"].get(name, "main"),
                                  "batch_size": batch_size, "libraries": library_versions(),
                                  "recipe": "direct-target-lifecycle-v1"}
            receipts[backbone] = directory / f"{backbone}-bundle.json"
            bundles[backbone] = load_bundle_receipt(receipts[backbone], root, requests[backbone])
            if bundles[backbone] is None:
                # Features already built by mpsae.embeddings_hub with the same
                # backbone weights: use them instead of downloading the dataset.
                from ..embeddings_hub import load_campaign_hub_bundle
                hub_bundle = load_campaign_hub_bundle(root, name, backbone, weights[backbone], root / "cache",
                    revision=campaign.downstream["revisions"].get(name))
                if hub_bundle is not None:
                    save_bundle(hub_bundle, receipts[backbone], root, requests[backbone])
                    bundles[backbone] = hub_bundle
        missing = [n for n, bundle in bundles.items() if bundle is None]
        if missing:
            if campaign.execution.get("precomputed_only"):
                raise FileNotFoundError(
                    f"{name}: missing or incompatible precomputed embeddings for {', '.join(missing)}; "
                    "check the linked bundle's backbone weights and dataset revision. Extraction is disabled.")
            # Leave room for features and transient archive writer before downloading.
            from .data import DATASETS
            spec = DATASETS[name]
            rows = sum(spec.expected_rows[s] for s in {spec.gallery_split, spec.query_split})
            require_disk(root, 2 * rows * sum(BACKBONES[n].dim * 4 + 8 for n in missing),
                         reserve_bytes=10 << 30, what=f"{name} embeddings (raw download additionally required)")
            dataset = load_dataset_bundle(name, raw, token=os.environ.get("HF_TOKEN"),
                revision=campaign.downstream["revisions"].get(name, "main"),
                reserve_bytes=(10 << 30) + 2 * rows * sum(BACKBONES[n].dim * 4 + 8 for n in missing))
            for backbone in missing:
                bundle = task(f"features/{backbone}/{name}", lambda n=backbone: extract_bundle(
                    BACKBONES[n], dataset, root / "benchmarks", root / "cache", device=device,
                    batch_size=batch_size, workers=workers, token=os.environ.get("HF_TOKEN"),
                    expected_weights=weights[n], reclaim_sink=True))
                if bundle is None:
                    raise RuntimeError(f"{name}/{backbone}: extraction failed; retaining raw data")
                bundles[backbone] = bundle
                save_bundle(bundle, receipts[backbone], root, requests[backbone])
            del dataset
            gc.collect()
        if campaign.execution["cleanup_raw"]:
            files = list(receipts.values())
            for bundle in bundles.values():
                for split in (bundle.gallery, bundle.query):
                    files.extend([split.path, split.path.with_name(split.path.name + ".json")])
            backup = backup_before_cleanup(observer, list(dict.fromkeys(files)))
            for subdir in ("hf_datasets", "downloads", "hub"):
                remove_owned(raw / subdir, root)
            atomic_json(directory / "lifecycle.json", {"state": "raw_deleted", "backup": backup})
        for backbone, bundle in bundles.items():
            spec = BACKBONES[backbone]
            dense = task(f"dense/{backbone}/{name}", lambda: evaluate_point(
                bundle, spec, root, device=device, **campaign.evaluation_options(name)))
            if dense:
                catalog["evaluations"].append(dense)
            for cell in (c for c in campaign.downstream_cells(name) if c.backbone == backbone):
                checkpoint = checkpoint_record(cell, root, require_provenance=True)
                verify_feature_provenance(bundle, checkpoint)
                model = load_fitted(cell, root / "runs" / cell.slug, spec, device=device)
                model.eval().requires_grad_(False)
                try:
                    for budget in campaign.budgets("downstream", "eval_k", backbone):
                        record = task(f"{campaign.task_for(name)}/{name}/{cell.slug}/eval-k{budget}", lambda k=budget:
                            evaluate_point(bundle, spec, root, cell=cell, checkpoint=checkpoint,
                                           model=model, k=k, device=device,
                                           retain_codes=campaign.execution["retain_codes"],
                                           **campaign.evaluation_options(name)))
                        if record:
                            catalog["evaluations"].append(record)
                finally:
                    del model
            for split in {id(bundle.gallery): bundle.gallery, id(bundle.query): bundle.query}.values():
                mmap = getattr(split.features, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        del bundles
        gc.collect()


def main(argv=None):
    import argparse
    import json
    from ..cli import load_dotenv
    load_dotenv(".env")
    parser = argparse.ArgumentParser(description="Reclaim redundant legacy FFCV after checking original ImageNet inventory")
    parser.add_argument("--workspace", default=os.environ.get("MPSAE_WORKSPACE", "work"))
    args = parser.parse_args(argv)
    with exclusive_lock(Path(args.workspace) / "state" / "direct-imagenet.lock", timeout=0):
        print(json.dumps(reclaim_legacy_ffcv(args.workspace), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
