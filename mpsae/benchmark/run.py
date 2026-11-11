"""One sequential, resumable campaign over existing ImageNet and frozen transfer.

Only source ImageNet fits may update parameters. Target datasets are read by
extract/encode/search functions under no_grad; there is no adaptation path.
"""
from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from .. import __version__
from ..config import (BACKBONES, METHOD_EPOCHS, METHOD_VARIANTS, Cell, training_preset,
                      validate_checkpoint_ot_weight)
from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, sha256_file
from ..logs import get_logger, setup_logging
from .config import Campaign, DEFAULT_TASKS, knn_settings

EVALUATION_VERSION = "frozen-transfer-ranking-v1"


def campaign_workspace(workspace: str | Path, campaign: Campaign, smoke: bool) -> Path:
    root = Path(workspace)
    if smoke:
        return root / f"smoke-{campaign.name}"
    # Different pixels/row selection from historical FFCV: never reuse its fits.
    return root / "direct-v1" if campaign.execution.get("storage_mode") == "direct_v1" else root


def campaign_paths(campaign: Campaign, workspace: str | Path, *, smoke=False) -> dict[str, Path]:
    root = campaign_workspace(workspace, campaign, smoke)
    plan_id = campaign.plan()["plan_id"]
    return {"workspace": root, "state": root / "campaigns" / campaign.name / plan_id,
            "reports": root / "reports" / campaign.name / plan_id}


def campaign_status(campaign: Campaign, workspace: str | Path, *, smoke=False) -> dict[str, Any]:
    paths = campaign_paths(campaign, workspace, smoke=smoke)
    return {"plan": campaign.plan(), "workspace": str(paths["workspace"]),
            "state": read_json(paths["state"] / "status.json") or {"complete": False, "tasks": {}},
            "catalog": read_json(paths["state"] / "catalog.json") or {},
            "report_manifest": str(paths["reports"] / "manifest.json")}


def _remove_owned_cache(path: Path, workspace: Path) -> None:
    """Remove only a point-specific cache below this workspace's cache root."""
    root = (workspace / "cache" / "benchmark_codes").resolve()
    target = path.resolve()
    if (not root.is_relative_to(workspace.resolve()) or path.is_symlink()
            or target == root or root not in target.parents):
        raise ValueError(f"refusing unexpected transient cache target: {target}")
    if target.is_dir():
        gc.collect()
        shutil.rmtree(target)


def _weights(cell: Cell, root: Path) -> dict[str, Any]:
    directory = root / "runs" / cell.slug
    config = read_json(directory / "config.json") or {}
    candidates = [config.get("backbone_weights")]
    preprocessing = config.get("backbone_preprocessing")
    source_data = (config.get("identity") or {}).get("data", {})
    for split in ("train", "val"):
        name = f"1K_{split}_{cell.backbone}.npz"
        for folder in (root / "embeddings" / cell.backbone, root / "embeddings"):
            manifest = (read_json(folder / (name + ".json"))
                        or read_json((folder / name).with_suffix(".json")) or {})
            source_split = "validation" if split == "val" else "train"
            if not manifest or manifest.get("sha256") != (source_data.get(source_split) or {}).get("sha256"):
                continue
            candidates.append(manifest.get("weights"))
            candidate_preprocessing = {key: manifest.get(key) for key in
                                       ("input_size", "normalize", "preprocessing")}
            if preprocessing is None:
                preprocessing = candidate_preprocessing
    valid = [item for item in candidates if isinstance(item, dict)
             and all(item.get(key) for key in ("repository", "revision", "file", "sha256"))]
    if not valid:
        raise RuntimeError(f"{cell.slug}: no pinned backbone weight provenance. Keep the ImageNet "
                           "archive .npz.json sidecars or a fit config with backbone_weights; "
                           "do not guess the weights from the backbone name.")
    keys = ("repository", "revision", "file", "sha256")
    identity = {key: valid[0][key] for key in keys}
    if any({key: value[key] for key in keys} != identity for value in valid[1:]):
        raise RuntimeError(f"{cell.slug}: source backbone weight manifests disagree")
    return {"weights": identity, "preprocessing": preprocessing}


def checkpoint_record(cell: Cell, root: Path, *, smoke=False,
                      require_provenance=True) -> dict[str, Any]:
    """Validate source artifacts without loading source embeddings or changing them."""
    directory = root / "runs" / cell.slug
    summary = read_json(directory / "summary.json")
    config = read_json(directory / "config.json")
    path = directory / "checkpoint.pt"
    if not summary or not summary.get("complete") or not config or not path.is_file():
        raise FileNotFoundError(f"missing completed source fit: {directory}")
    identity = summary.get("identity")
    signature = summary.get("signature")
    if not isinstance(identity, dict) or canonical_hash(identity) != signature:
        raise RuntimeError(f"{directory}: invalid source training signature")
    if config.get("signature") != signature or canonical_hash(config.get("identity")) != signature:
        raise RuntimeError(f"{directory}: source config and summary disagree")
    expected = (cell.backbone, cell.method, cell.k, cell.seed)
    actual = tuple(identity.get(key) for key in ("backbone", "method", "target_k", "seed"))
    if expected != actual or summary.get("cell") != cell.__dict__:
        raise RuntimeError(f"{directory}: checkpoint does not identify the selected source cell")
    validate_checkpoint_ot_weight(cell.method, cell.k, identity, context=str(directory))
    variant = METHOD_VARIANTS.get(cell.method)
    if variant and (identity.get("objective") or {}).get("variant") != variant:
        raise RuntimeError(f"{directory}: checkpoint uses a different {cell.method} objective variant; "
                           f"expected {variant}")
    synthetic = any("synthetic" in str(value.get("recipe", "")).lower()
                    for value in identity.get("data", {}).values() if isinstance(value, dict))
    if synthetic != smoke:
        raise RuntimeError(f"{directory}: synthetic and scientific artifacts cannot be mixed")
    if cell.method in METHOD_EPOCHS:
        schedule = identity.get("schedule", {})
        epochs = METHOD_EPOCHS[cell.method]
        per_epoch, total = schedule.get("batches_per_epoch"), schedule.get("total_steps")
        if (schedule.get("epochs") != epochs or summary.get("steps") != total
                or not isinstance(per_epoch, int) or isinstance(per_epoch, bool) or per_epoch < 1
                or not isinstance(total, int) or isinstance(total, bool) or total < 1
                or not smoke and total != epochs * per_epoch):
            raise RuntimeError(f"{directory}: MPSAEv2 needs a completed {epochs}-epoch checkpoint")
    provenance = _weights(cell, root) if require_provenance and not smoke else {}
    return {"checkpoint_id": sha256_file(path), "path": str(path), "signature": signature,
            "cell": cell.__dict__, "scientific": not smoke,
            "training_identity": identity, "source_code_version": config.get("code_version"),
            "backbone_weights": provenance.get("weights"),
            "backbone_preprocessing": provenance.get("preprocessing")}


def verify_feature_provenance(features, checkpoint: dict, *, smoke=False) -> None:
    if smoke:
        return
    extracted = features.manifest["extraction"]
    keys = ("repository", "revision", "file", "sha256")
    if {key: extracted["weights"].get(key) for key in keys} != checkpoint["backbone_weights"]:
        raise RuntimeError("downstream features use different backbone weights from this source fit")
    original = checkpoint.get("backbone_preprocessing")
    if not original:
        raise RuntimeError("source preprocessing provenance missing; retain original archive sidecars")
    old = original.get("preprocessing") or {}
    normalization = original.get("normalize")
    if isinstance(normalization, dict):
        mean, std = normalization.get("mean"), normalization.get("std")
    elif isinstance(normalization, (list, tuple)) and len(normalization) == 2:
        mean, std = normalization
    else:
        mean, std = old.get("mean"), old.get("std")
    new = extracted["preprocessing"]
    size = original.get("input_size") or old.get("crop")
    if (not mean or not std or not size or list(mean) != new["mean"]
            or list(std) != new["std"] or int(size) != new["center_crop"]):
        raise RuntimeError("source and downstream normalization/input size disagree or are undocumented")


def classifier_protocol(features, knn=None) -> dict[str, Any]:
    """Pinned voting policy; the saved source split remains a separate identity."""
    from ..evaluate.classification import (PROTOCOL_VERSION, VOTE_RULE, CLASS_TIE_RULE,
                                           EMPTY_QUERY_POLICY, REFERENCE)
    from ..evaluate.ranking import TIE_RULE

    if (features.exclude_self or features.gallery.name != "train"
            or features.query.name == features.gallery.name or features.name == "sop"):
        raise ValueError("kNN classification requires train support and a separate query split; "
                         "SOP keeps its product retrieval protocol")
    return {"name": "dino_weighted_knn", **knn_settings(knn),
            "metric_version": PROTOCOL_VERSION, "vote_rule": VOTE_RULE,
            "tie_rule": TIE_RULE, "class_tie_rule": CLASS_TIE_RULE,
            "empty_query_policy": EMPTY_QUERY_POLICY, "normalization": "l2", "reference": REFERENCE}


def _split_identity(split) -> dict[str, Any]:
    labels = split.labels.tolist() if hasattr(split.labels, "tolist") else list(split.labels)
    return {"split": split.name, "ids_sha256": canonical_hash(list(split.ids)),
            "labels_sha256": canonical_hash(labels)}


def evaluation_identity(features, checkpoint: dict | None, *, k: int | None,
                        device: str, gallery_chunk: int, query_block: int,
                        task: str = "retrieval", knn=None) -> dict[str, Any]:
    from ..evaluate.ranking import PROTOCOL_VERSION

    identity = {"version": EVALUATION_VERSION, "metric_version": PROTOCOL_VERSION,
            "mode": "frozen", "features": features.identity,
            "checkpoint": checkpoint["checkpoint_id"] if checkpoint else "dense",
            "train_signature": checkpoint["signature"] if checkpoint else None,
            "eval_k": k, "encode_precision": "float32_cpu" if str(device).startswith("cpu")
            else "autocast_fp16_float32_storage", "search_precision": "float32_tf32_disabled",
            "ranks": [1, 10, 100], "exclude_self": features.exclude_self,
            "no_positive_policy": "error", "gallery_chunk": gallery_chunk,
            "query_block": query_block}
    if task == "knn_classification":
        classifier = classifier_protocol(features, knn)
        identity.update(task=task, classifier=classifier,
                        metric_version=classifier["metric_version"],
                        support=_split_identity(features.gallery), query=_split_identity(features.query))
        identity.pop("ranks")
        identity.pop("no_positive_policy")
    elif task != "retrieval":
        raise ValueError(f"unknown downstream task {task!r}")
    # Leave the existing retrieval identity intact so SOP results can be reused.
    return identity


def evaluate_point(features, spec, root: Path, *, cell: Cell | None = None,
                   checkpoint: dict | None = None, model=None, k: int | None = None,
                   device="cuda:0", retain_codes=False, gallery_chunk=4096,
                   query_block=512, smoke=False, task=None, knn=None) -> dict[str, Any]:
    """Cache a single immutable evaluation; it cannot train a target model."""
    from ..evaluate.ranking import evaluate_retrieval
    from ..runs import _close_code_arrays, _close_failed_cache_maps

    task = task or DEFAULT_TASKS.get(features.name, "retrieval")
    identity = evaluation_identity(features, checkpoint, k=k, device=device,
                                   gallery_chunk=gallery_chunk, query_block=query_block,
                                   task=task, knn=knn)
    evaluation_id = canonical_hash(identity)
    out = root / "evaluations" / evaluation_id / "result.json"
    with exclusive_lock(out.with_suffix(".lock")):
        previous = read_json(out)
        if previous and canonical_hash(previous.get("identity")) == evaluation_id and previous.get("complete"):
            if not retain_codes:
                _remove_owned_cache(root / "cache" / "benchmark_codes" / evaluation_id, root)
            return previous
        codes_root = root / "cache" / "benchmark_codes" / evaluation_id
        gallery = queries = None
        try:
            def score(gallery, queries, dim, *, normalize=False):
                arguments = {"dim": dim, "device": device,
                             "gallery_ids": features.gallery.ids, "query_ids": features.query.ids,
                             "gallery_chunk": gallery_chunk, "query_block": query_block,
                             "normalize": normalize}
                if task == "knn_classification":
                    from ..evaluate.classification import evaluate_knn

                    return evaluate_knn(gallery, features.gallery.labels, queries, features.query.labels,
                                        **arguments, **knn_settings(knn))
                return evaluate_retrieval(gallery, features.gallery.labels, queries, features.query.labels,
                                          exclude_self=features.exclude_self, **arguments)

            if cell is None:
                gallery, queries = features.gallery.features, features.query.features
                metrics = score(gallery, queries, spec.dim, normalize=True)
                import numpy as np

                metrics.update(nnz_mean=float(np.count_nonzero(gallery, axis=1).mean()),
                               code_bytes_per_image=spec.dim * 4)
            else:
                from ..evaluate.encode import encode_split

                if checkpoint is None or model is None or k is None:
                    raise ValueError("a sparse evaluation requires a frozen checkpoint, model and budget")
                model.eval().requires_grad_(False)
                gallery = encode_split(model, features.gallery.features, None, k, device=device,
                                       cache_dir=codes_root,
                                       identity={"evaluation": evaluation_id, "split": features.gallery.name})
                queries = gallery if features.query is features.gallery else encode_split(
                    model, features.query.features, None, k, device=device, cache_dir=codes_root,
                    identity={"evaluation": evaluation_id, "split": features.query.name})
                metrics = score(gallery, queries, spec.preset.hidden_dim)
                metrics.update(nnz_mean=gallery["stats"]["nnz_mean"],
                               query_nnz_mean=queries["stats"]["nnz_mean"],
                               code_bytes_per_image=k * 8,
                               encode_seconds=gallery["stats"]["seconds"]
                               + (0 if queries is gallery else queries["stats"]["seconds"]))
            comparison = {"features": features.identity, "ranking": EVALUATION_VERSION,
                          "precision": identity["encode_precision"]}
            if task == "knn_classification":
                comparison.update(task=task, classifier=identity["classifier"],
                                  support=identity["support"], query=identity["query"])
            record = {"evaluation_id": evaluation_id, "identity": identity,
                      "comparison_id": canonical_hash(comparison), "task": task,
                      "dataset": features.name, "protocol": features.protocol,
                      "backbone": spec.name, "mode": "frozen", "training_on_target": False,
                      "method": cell.method if cell else "dense", "train_k": cell.k if cell else None,
                      "eval_k": k, "seed": cell.seed if cell else None,
                      "checkpoint_id": checkpoint["checkpoint_id"] if checkpoint else None,
                      "signature": checkpoint["signature"] if checkpoint else None,
                      "complete": True, "scientific": not smoke,
                      "feature_manifest": features.manifest, "metrics": metrics}
            if task == "knn_classification":
                record.update(classifier=identity["classifier"],
                              uses_target_support_labels=True, parameter_updates_on_target=False)
            atomic_json(out, record)
            return record
        except BaseException as error:
            if not retain_codes:
                _close_failed_cache_maps(error, codes_root)
            raise
        finally:
            if not retain_codes:
                # Arrays are no longer needed by a scalar result; close mmap
                # references before removing this evaluation's private cache.
                _close_code_arrays(gallery)
                _close_code_arrays(queries)
                gallery = queries = None
                _remove_owned_cache(codes_root, root)


def _selected_imagenet_record(campaign, record):
    """Expose only current metrics; preserve richer source caches on disk."""
    backbone = record["backbone"]

    def point_view(point):
        result = dict(point)
        if not campaign.wants_probe(backbone, point["k"]):
            result = {key: value for key, value in result.items() if not key.startswith("probe_")}
        if not campaign.imagenet["timing"]:
            for key in ("timing", "timing_config", "timing_status", "timing_error",
                        "retrieval_ms", "index_bytes"):
                result.pop(key, None)
        return result

    result = dict(record)
    if result.get("e1"):
        result["e1"] = point_view(result["e1"]) if campaign.imagenet["e1"] else None
    if result.get("e2"):
        if campaign.imagenet["e2"]:
            budgets = campaign.budgets("imagenet", "eval_k", backbone)
            result["e2"] = {**result["e2"], "eval_k": list(budgets),
                "points": [point_view(p) for p in result["e2"].get("points", []) if p["k"] in budgets]}
            result["e2"].pop("cached_points", None)
        else:
            result["e2"] = None
    return result


def _report(campaign, paths, catalog, failures):
    from ..report.figures import build_all
    from .report import build_report

    output = paths["reports"]
    output.mkdir(parents=True, exist_ok=True)
    downstream = None
    if campaign.downstream["enabled"]:
        downstream = build_report(catalog["evaluations"], output / "downstream",
                                  expected_points=campaign.plan()["downstream_sparse_points"]
                                  + campaign.plan()["downstream_dense_points"],
                                  failures=failures, expected_seeds=campaign.seeds)
    imagenet = None
    if campaign.imagenet["enabled"]:
        imagenet = build_all(paths["workspace"] / "runs", output / "imagenet",
                             records=[_selected_imagenet_record(campaign, r) for r in catalog["imagenet"]],
                             dense_references=catalog["dense"],
                             seeds=campaign.seeds)
    selected = [c for c in campaign.source_cells() if campaign.imagenet["enabled"]
                and c.k in campaign.budgets("imagenet", "train_k", c.backbone)]
    expected = {"source_cells": len(campaign.source_cells()),
                "imagenet_e1": len(selected) if campaign.imagenet["e1"] else 0,
                "imagenet_e2": sum(c.k in BACKBONES[c.backbone].e2_anchors for c in selected)
                if campaign.imagenet["e2"] else 0}
    completed = {"source_cells": len(catalog["checkpoints"]),
                 "imagenet_e1": sum(bool(r.get("e1")) for r in catalog["imagenet"]),
                 "imagenet_e2": sum(bool(r.get("e2")) for r in catalog["imagenet"])}
    plot_issues = [reason for report in (downstream or {}, imagenet or {})
                   for reason in report.get("skipped", [])
                   if "Error:" in reason or reason.startswith(("figures:", "all figures:"))]
    manifest = {"name": campaign.name, "plan_id": campaign.plan()["plan_id"],
                "complete": not failures and not plot_issues and (downstream is None or bool(downstream["complete"]))
                and completed == expected,
                "expected": expected, "completed": completed, "failures": failures,
                "plot_issues": plot_issues,
                "imagenet": imagenet, "downstream": downstream,
                "catalog": str(paths["state"] / "catalog.json")}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output.parent / "latest.json", {"plan_id": campaign.plan()["plan_id"],
                                               "manifest": str(output / "manifest.json")})
    return manifest


def report_campaign(campaign: Campaign, workspace: str | Path, *, smoke=False):
    paths = campaign_paths(campaign, workspace, smoke=smoke)
    catalog = read_json(paths["state"] / "catalog.json")
    if not catalog:
        raise FileNotFoundError("no catalog for this exact campaign plan; run it first")
    status = read_json(paths["state"] / "status.json") or {}
    return _report(campaign, paths, catalog, status.get("failures", []))


def run_campaign(campaign: Campaign, workspace: str | Path, *, device="cuda:0",
                 workers=16, embed_batch_size=256, smoke=False, observer=None) -> dict[str, Any]:
    import os
    import torch

    from ..features import load_bundle
    from ..pipeline import (Paths, PipelineOptions, stage_embed, stage_ffcv,
                            stage_imagenet)
    from ..runs import RunSettings, dense_reference, run_cell
    from ..train import FitSettings, compatible_fit_identity, fit, fit_identity, load_fitted
    from .data import load_dataset_bundle
    from .extract import extract_bundle, make_synthetic_feature_bundle

    paths = campaign_paths(campaign, workspace, smoke=smoke)
    root = paths["workspace"]
    direct = campaign.execution.get("storage_mode") == "direct_v1" and not smoke
    Paths(root).create()
    paths["state"].mkdir(parents=True, exist_ok=True)
    setup_logging(root / "logs", name=campaign.name)
    log = get_logger("campaign")
    with exclusive_lock(paths["state"] / "campaign.lock", timeout=0), (
            exclusive_lock(root / "state" / "direct-workspace.lock", timeout=0) if direct else nullcontext()):
        plan = campaign.plan()
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, check=True).stdout.strip()
        except Exception:
            commit = None
        atomic_json(paths["state"] / "resolved.json", plan)
        atomic_json(paths["state"] / "runtime.json", {
            "device": device, "smoke": smoke, "code_version": __version__, "git_commit": commit,
            "torch": str(torch.__version__), "workers": workers, "embed_batch_size": embed_batch_size,
            "native_profile": read_json(Path(sys.prefix) / "server-profile.json")})
        catalog = {"plan_id": plan["plan_id"], "checkpoints": {}, "evaluations": [],
                   "imagenet": [], "dense": {}, "scientific": not smoke}
        state = {"plan_id": plan["plan_id"], "complete": False, "tasks": {}, "failures": []}

        def persist():
            atomic_json(paths["state"] / "status.json", state)
            atomic_json(paths["state"] / "catalog.json", catalog)

        def task(name, operation):
            state["tasks"][name] = {"state": "running"}
            persist()
            try:
                result = operation()
            except Exception as error:
                failure = {"task": name, "error": f"{type(error).__name__}: {error}"}
                state["tasks"][name] = {"state": "failed", **failure}
                state["failures"].append(failure)
                log.error("%s", failure)
                persist()
                if not campaign.execution["continue_on_error"]:
                    raise
                return None
            state["tasks"][name] = {"state": "done"}
            persist()
            if observer is not None:
                observer.on_task(name, result)
            return result

        options = PipelineOptions(paths=Paths(root), backbones=campaign.backbones,
                                  device=device, workers=workers, embed_batch_size=embed_batch_size,
                                  smoke=smoke)
        datasets = {}
        try:
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"requested CUDA device {device} is unavailable")
            missing_features = [name for name in campaign.backbones if any(
                not (root / "embeddings" / name / f"1K_{split}_{name}.npz").is_file()
                for split in ("train", "val"))]
            if (campaign.execution.get("precomputed_only") and not smoke and missing_features
                    and campaign.imagenet["enabled"]):
                raise FileNotFoundError(
                    f"Missing precomputed ImageNet embeddings: {', '.join(missing_features)}. "
                    "Run python -m mpsae.embeddings_hub sync --download-only --config <campaign.yaml> first.")
            if campaign.imagenet["enabled"] and missing_features and not smoke:
                if not str(device).startswith("cuda"):
                    raise RuntimeError("real ImageNet feature extraction requires CUDA; use existing source "
                                       "fits with imagenet.enabled=false for evaluation-only CPU work")
                if not campaign.imagenet["prepare"]:
                    raise FileNotFoundError("ImageNet feature archives missing with imagenet.prepare=false")
                if not direct and any(BACKBONES[name].recipe == "csrv2" for name in missing_features):
                    import importlib.util

                    missing_imports = [module for module in ("ffcv", "cupy")
                                       if importlib.util.find_spec(module) is None]
                    if missing_imports:
                        raise RuntimeError("ImageNet GPU extraction needs " + ", ".join(missing_imports)
                                           + f" in the active interpreter ({sys.executable}). "
                                           "Run 'bash setup_server.sh', then "
                                           "'source activate_server.sh' and retry. "
                                           "See docs/server.md; installing requirements.txt alone "
                                           "does not install the full server environment.")
                    # A package may be installed yet fail at import because
                    # its compiled JPEG/OpenCV/CUDA dependencies are broken.
                    import importlib

                    from ..ffcv_compat import prepare_ffcv

                    for module in ("ffcv", "cupy"):
                        try:
                            if module == "ffcv":
                                prepare_ffcv()
                            importlib.import_module(module)
                        except Exception as error:
                            raise RuntimeError(
                                f"{module} is installed but cannot import in {sys.executable}: {error}. "
                                "Run 'bash setup_server.sh', then "
                                "'source activate_server.sh'; see docs/server.md."
                            ) from error
            if not smoke and campaign.downstream["enabled"]:
                import importlib.util

                required = (("numpy",) if campaign.execution.get("precomputed_only") else
                            ("numpy", "PIL", "torchvision", "timm", "datasets", "huggingface_hub"))
                missing_imports = [module for module in required if importlib.util.find_spec(module) is None]
                if missing_imports:
                    raise RuntimeError(f"missing benchmark dependencies: {', '.join(missing_imports)}")
            if campaign.downstream["enabled"] and not smoke and not direct:
                for name in campaign.downstream["datasets"]:
                    datasets[name] = task(f"dataset/{name}", lambda n=name: load_dataset_bundle(
                        n, root / "cache", revision=campaign.downstream["revisions"].get(n, "main"),
                        token=os.environ.get("HF_TOKEN")))

            prepared = True
            if direct and campaign.imagenet["enabled"]:
                from .storage import prepare_sources
                prepare_sources(campaign, workspace, root, device=device, workers=workers,
                                batch_size=embed_batch_size, observer=observer, task=task)
                missing_features = []
            if campaign.imagenet["enabled"] and missing_features and not smoke:
                prepared = task("imagenet/prepare", lambda: stage_imagenet(options)) is not None
                if prepared and any(BACKBONES[n].recipe == "csrv2" for n in missing_features):
                    prepared = task("imagenet/ffcv", lambda: stage_ffcv(options)) is not None

            for name in campaign.backbones:
                spec = BACKBONES[name]
                source = None
                if campaign.imagenet["enabled"]:
                    if name in missing_features:
                        if not prepared:
                            continue
                        produced = task(f"imagenet/embed/{name}", lambda n=name: stage_embed(
                            replace(options, backbones=(n,))))
                        if produced is None:
                            continue
                    source = task(f"imagenet/load/{name}", lambda s=spec: load_bundle(
                        s, root / "embeddings", root / "cache"))
                    if source is None:
                        continue
                    dense = task(f"imagenet/dense/{name}", lambda: dense_reference(
                        spec, source, RunSettings(root=root / "runs", fit=FitSettings(device=device))))
                    if dense is None:
                        # Continue with other backbones after recording this
                        # failure, but never fit against an unverified source.
                        continue
                    if dense and not smoke and spec.reference_dense_top1 is not None and abs(
                            dense["top1"] - spec.reference_dense_top1) > 0.3:
                        raise RuntimeError(f"{name}: dense reference gate failed before source training")
                    if dense:
                        catalog["dense"][name] = dense

                feature_bundles = {}
                for cell in (c for c in campaign.source_cells() if c.backbone == name):
                    out = root / "runs" / cell.slug
                    maximum = campaign.execution["max_steps"] or (2 if smoke else 0)
                    fits = FitSettings(device=device, max_steps=maximum,
                                       log_every=campaign.execution["train_log_every"],
                                       checkpoint_every=campaign.execution["checkpoint_every"],
                                       residency=campaign.execution["feature_residency"])
                    settings = RunSettings(root=root / "runs", embeddings_dir=root / "embeddings",
                                           cache_dir=root / "cache", fit=fits,
                                           linear_probe=campaign.imagenet["linear_probe"],
                                           probe_k_values=campaign.probe_budgets(name),
                                           run_e2=campaign.imagenet["e2"],
                                           e2_k_values=campaign.budgets("imagenet", "eval_k", name),
                                           measure_timing=campaign.imagenet["timing"],
                                           timing_query_batch=campaign.imagenet["timing_query_batch"],
                                           timing_warmup=campaign.imagenet["timing_warmup"],
                                           timing_rounds=campaign.imagenet["timing_rounds"],
                                           timing_gallery_chunk=campaign.imagenet["timing_gallery_chunk"],
                                           timing_build_entry_cap=campaign.imagenet["timing_build_entry_cap"],
                                           timing_retry_skipped=campaign.imagenet["timing_retry_skipped"],
                                           retain_codes=campaign.execution["retain_codes"])
                    wants_imagenet = campaign.imagenet["enabled"] and cell.k in campaign.budgets(
                        "imagenet", "train_k", name)
                    wants_downstream = cell in campaign.downstream_cells()
                    previous = read_json(out / "summary.json") or {}
                    missing = not (previous.get("complete") and (out / "checkpoint.pt").is_file()
                                   and (out / "config.json").is_file())
                    if missing and (not campaign.imagenet["enabled"] or not campaign.execution["train_missing"]):
                        def missing_source():
                            raise FileNotFoundError(f"{cell.slug}: complete checkpoint required; source training disabled")
                        task(f"source/{cell.slug}", missing_source)
                        continue

                    def source_fit(c=cell):
                        if source is not None:
                            if wants_imagenet and campaign.imagenet["e1"]:
                                run_cell(c, source, settings, code_version=__version__)
                            else:
                                fit(c, source, out, fits, code_version=__version__, spec=spec)
                        record = checkpoint_record(c, root, smoke=smoke,
                                                   require_provenance=wants_downstream)
                        # Even evaluation-only imports must match the selected
                        # training recipe, not merely have compatible shapes.
                        from types import SimpleNamespace

                        saved_identity = record["training_identity"]
                        rows = saved_identity["data"]["train"]["rows"]
                        preset = training_preset(c.method, spec.preset, k=c.k)
                        per_epoch = -(-rows // preset.batch_size)
                        total = per_epoch * preset.epochs
                        if maximum:
                            total = min(total, maximum)
                        requested = fit_identity(c, SimpleNamespace(identity=saved_identity["data"]),
                                                 preset, total_steps=total, per_epoch=per_epoch,
                                                 code_version=__version__)
                        if not compatible_fit_identity(saved_identity, requested):
                            raise RuntimeError(f"{c.slug}: source checkpoint uses a different training recipe")
                        # load_fitted verifies the actual checkpoint signature and model shapes.
                        model = load_fitted(c, out, spec, device=device)
                        model.eval().requires_grad_(False)
                        return record, model

                    fitted = task(f"source/{cell.slug}", source_fit)
                    if fitted is None:
                        continue
                    checkpoint, model = fitted
                    catalog["checkpoints"][cell.slug] = checkpoint
                    if wants_imagenet:
                        e1 = read_json(out / "eval_e1.json") if campaign.imagenet["e1"] else None
                        e2 = (read_json(out / "eval_e2.json")
                              if campaign.imagenet["e2"] and cell.k in spec.e2_anchors else None)
                        catalog["imagenet"].append(_selected_imagenet_record(campaign, {
                            "backbone": name, "method": cell.method,
                            "k": cell.k, "seed": cell.seed, "path": str(out),
                            "config": read_json(out / "config.json"), "e1": e1, "e2": e2,
                            "complete": e1 is not None, "missing": [],
                            "has_train_log": (out / "train.jsonl").is_file(),
                            "code_version": checkpoint["source_code_version"]}))
                    if direct or not wants_downstream:
                        del model
                        continue
                    for dataset_name in campaign.downstream["datasets"]:
                        if name not in campaign.target_backbones(dataset_name):
                            continue
                        if dataset_name not in feature_bundles:
                            if smoke:
                                feature_bundles[dataset_name] = task(f"features/{name}/{dataset_name}",
                                    lambda n=dataset_name: make_synthetic_feature_bundle(
                                        spec, n, root / "benchmarks", root / "cache"))
                            elif datasets.get(dataset_name) is not None:
                                feature_bundles[dataset_name] = task(f"features/{name}/{dataset_name}",
                                    lambda n=dataset_name: extract_bundle(
                                        spec, datasets[n], root / "benchmarks", root / "cache",
                                        device=device, batch_size=embed_batch_size, workers=workers,
                                        token=os.environ.get("HF_TOKEN"),
                                        expected_weights=checkpoint["backbone_weights"]))
                            else:
                                feature_bundles[dataset_name] = None
                            features = feature_bundles[dataset_name]
                            if features is not None:
                                dense = task(f"dense/{name}/{dataset_name}", lambda f=features:
                                    evaluate_point(f, spec, root, device=device, smoke=smoke,
                                                   **campaign.evaluation_options(dataset_name)))
                                if dense:
                                    catalog["evaluations"].append(dense)
                        features = feature_bundles.get(dataset_name)
                        if features is None:
                            continue
                        if task(f"provenance/{dataset_name}/{cell.slug}", lambda f=features:
                                (verify_feature_provenance(f, checkpoint, smoke=smoke), True)[1]) is None:
                            continue
                        for budget in campaign.budgets("downstream", "eval_k", name):
                            record = task(f"{campaign.task_for(dataset_name)}/{dataset_name}/{cell.slug}/eval-k{budget}",
                                lambda k=budget, f=features: evaluate_point(
                                    f, spec, root, cell=cell, checkpoint=checkpoint, model=model, k=k,
                                    device=device, retain_codes=campaign.execution["retain_codes"], smoke=smoke,
                                    **campaign.evaluation_options(dataset_name)))
                            if record:
                                catalog["evaluations"].append(record)
                                persist()
                    del model
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                persist()
            if direct and campaign.downstream["enabled"]:
                from .storage import run_targets
                run_targets(campaign, root, catalog, device=device, workers=workers,
                            batch_size=embed_batch_size, observer=observer, task=task)
                persist()
        except BaseException as error:
            if not state["failures"]:
                state["failures"].append({"task": "campaign", "error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            persist()
            manifest = _report(campaign, paths, catalog, state["failures"])
            state["complete"] = manifest["complete"]
            state["report_manifest"] = str(paths["reports"] / "manifest.json")
            state["finished"] = datetime.now(timezone.utc).isoformat()
            persist()
        if not state["complete"]:
            raise RuntimeError(f"{len(state['failures'])} campaign task(s) failed; partial report: "
                               f"{paths['reports'] / 'manifest.json'}")
        return manifest
