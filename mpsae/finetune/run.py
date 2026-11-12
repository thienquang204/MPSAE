"""Run nine additional fits, with independent feature/evaluation identities."""
from copy import deepcopy
from dataclasses import replace
import csv
import math
import os
from pathlib import Path

from ..config import BACKBONES
from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, sha256_file
from ..logs import get_logger, setup_logging
from .data import close_bundle, extract_features, image_datasets, transforms


def output_root(config, workspace, smoke=False):
    return Path(workspace) / ("smoke-finetune-v1" if smoke else "finetune-v1") / config.name


def preflight_sources(config, source_root):
    """Fail before image downloads if one of the nine source fits is missing."""
    from ..benchmark.run import checkpoint_record
    from types import SimpleNamespace
    from .. import __version__
    from ..config import training_preset
    from ..train import load_fitted, fit_identity, compatible_fit_identity
    sources = {}
    for cell in config.cells():
        record = checkpoint_record(cell, Path(source_root))
        saved = record["training_identity"]
        rows = saved["data"]["train"]["rows"]
        if rows != 1_281_167:
            raise RuntimeError(f"{cell.slug}: expected a source fit on the full ImageNet training split")
        preset = training_preset(cell.method, BACKBONES[cell.backbone].preset, k=cell.k)
        per_epoch = -(-rows // preset.batch_size)
        requested = fit_identity(cell, SimpleNamespace(identity=saved["data"]), preset,
            total_steps=per_epoch * preset.epochs, per_epoch=per_epoch, code_version=__version__)
        if not compatible_fit_identity(saved, requested):
            raise RuntimeError(f"{cell.slug}: source checkpoint uses a different or capped training recipe")
        summary = read_json(Path(record["path"]).parent / "summary.json")
        if summary.get("steps") != per_epoch * preset.epochs:
            raise RuntimeError(f"{cell.slug}: source checkpoint has not completed its full training schedule")
        # Validate checkpoint tensors/signature too, without allocating source embeddings.
        model = load_fitted(cell, Path(record["path"]).parent, BACKBONES[cell.backbone], device="cpu")
        del model
        sources[cell.slug] = record
    for name in config.campaign.backbones:
        identities = {canonical_hash(r["backbone_weights"]) for r in sources.values()
                      if r["cell"]["backbone"] == name}
        if len(identities) != 1:
            raise RuntimeError(f"{name}: selected source SAEs use different backbone weights")
    return sources


def evaluate_model(cell, spec, sae, bundle, root, config, *, signature, summary,
                   device, baseline=False, smoke=False):
    from ..runs import RunSettings, evaluate_at, _cached_e1_complete
    from ..train import FitSettings
    campaign = config.campaign
    budgets = campaign.budgets("imagenet", "eval_k", cell.backbone)
    if smoke:
        budgets = (2, cell.k)
    settings = RunSettings(root=Path(root) / "runs", cache_dir=Path(root) / "cache",
        fit=FitSettings(device=device), linear_probe=campaign.imagenet["linear_probe"] and not baseline and not smoke,
        probe_k_values=campaign.probe_budgets(cell.backbone), measure_timing=campaign.imagenet["timing"] and not baseline and not smoke,
        timing_warmup=campaign.imagenet["timing_warmup"], timing_rounds=campaign.imagenet["timing_rounds"],
        timing_query_batch=campaign.imagenet["timing_query_batch"],
        timing_gallery_chunk=campaign.imagenet["timing_gallery_chunk"],
        timing_build_entry_cap=campaign.imagenet["timing_build_entry_cap"],
        retain_codes=False, query_block=128 if smoke else 8192)
    directory = settings.root / cell.slug
    path = directory / "evaluation.json"
    old = read_json(path) or {}
    done = {p["k"]: p for p in old.get("points", [])} if old.get("signature") == signature else {}
    sae.eval().requires_grad_(False)
    selected = sorted(set(budgets) | {cell.k})
    for k in selected:
        if _cached_e1_complete(done.get(k), settings.at_budget(k)):
            continue
        sae.to(device)
        def persist(point):
            done[k] = point
            atomic_json(path, {"signature": signature, "points": [done[b] for b in sorted(done)]})
        point = evaluate_at(cell, bundle, sae, spec, settings, k=k, signature=signature,
            want_probe=True, want_timing=True, previous=done.get(k), on_progress=persist)
        persist(point)
    def selected_point(k):
        point = dict(done[k])
        if not settings.at_budget(k).linear_probe:
            point = {key: value for key, value in point.items() if not key.startswith("probe_")}
        if not settings.measure_timing:
            for key in ("timing", "timing_config", "timing_status", "timing_error", "retrieval_ms", "index_bytes"):
                point.pop(key, None)
        return point

    e1 = {**selected_point(cell.k), "train_seconds": summary.get("train_seconds", 0),
          "gpu_hours": summary.get("gpu_hours", 0)}
    e2 = {"anchor_k": cell.k, "points": [selected_point(k) for k in budgets]}
    atomic_json(directory / "eval_e1.json", e1)
    atomic_json(directory / "eval_e2.json", e2)
    return {"backbone": cell.backbone, "method": cell.method, "k": cell.k, "seed": cell.seed,
        "e1": e1, "e2": e2, "config": read_json(directory / "config.json") or {},
        "path": str(directory), "complete": True, "missing": [], "has_train_log": not baseline,
        "regime": "frozen_same_images" if baseline else "supervised_joint_finetune"}


def write_report(config, root, catalog, failures):
    from ..report.figures import build_all
    output = root / "reports" / config.plan()["plan_id"]
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for regime, key in (("joint", "imagenet"), ("frozen", "frozen")):
        reports[regime] = build_all(root / ("baseline" if regime == "frozen" else "") / "runs",
            output / regime, records=catalog[key], seeds=config.campaign.seeds, dense_references={})
    if config.transfer["enabled"]:
        from .transfer import report as transfer_report
        reports["transfer"] = transfer_report(config, root, catalog.get("evaluations", []),
            smoke=not catalog["scientific"], failures=failures)
    def points(records):
        result = {}
        for record in records:
            for point in [record["e1"], *record["e2"]["points"]]:
                key = (record["backbone"], record["method"], record["k"], record["seed"], point["k"])
                result[key] = point["top1"]
        return result
    frozen, joint = points(catalog["frozen"]), points(catalog["imagenet"])
    with (output / "paired_1nn.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backbone", "method", "train_k", "seed", "inference_k", "frozen_top1",
                         "joint_top1", "delta_percentage_points"])
        for key in sorted(frozen.keys() & joint.keys()):
            writer.writerow([*key, frozen[key], joint[key], joint[key] - frozen[key]])
    expected = len(config.cells())
    plot_errors = [reason for report in reports.values() for reason in report.get("skipped", [])
                   if "Error:" in reason or reason.startswith(("figures:", "all figures:"))]
    manifest = {"regime": "supervised_joint_finetune", "plan": config.plan(),
        "complete": not failures and not plot_errors and len(catalog["imagenet"]) == expected
                    and len(catalog["frozen"]) == expected
                    and (not config.transfer["enabled"] or reports["transfer"]["complete"]),
        "scientific": catalog["scientific"], "failures": failures, "plot_errors": plot_errors,
        "reports": reports, "comparison": str(output / "paired_1nn.csv")}
    atomic_json(output / "manifest.json", manifest)
    (output / "README.txt").write_text(
        ("SYNTHETIC SMOKE TEST: not scientific results.\n" if not catalog["scientific"] else "")
        + "Supervised joint backbone + SAE fine-tuning with a frozen reconstruction teacher.\n"
          "paired_1nn.csv compares against frozen controls on identical images/preprocessing.\n"
          "This is an additional supervised experiment, not the original frozen-backbone protocol.\n",
        encoding="utf-8")
    atomic_json(root / "reports" / "latest.json", {"manifest": str(output / "manifest.json")})
    return manifest


def run(config, workspace, *, source_root=None, image_root=None, device="cuda:0", workers=8,
        extract_batch=64, download_images=False, smoke=False, backup=False,
        image_storage="folder", backup_interval=1800):
    config.ensure_enabled()
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset
    from ..benchmark.extract import _network
    from ..model import build_sae
    from ..train import load_fitted
    from .train import fit_joint, forward_features, autocast
    if workers < 0 or extract_batch < 1:
        raise ValueError("workers must be nonnegative; extract_batch must be positive")
    if not math.isfinite(backup_interval) or backup_interval <= 0:
        raise ValueError("backup_interval must be finite and positive")
    if image_storage not in ("folder", "packed"):
        raise ValueError("image_storage must be folder or packed")
    target = torch.device(device)
    if not smoke and (target.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("scientific backbone fine-tuning requires CUDA; use --smoke for CPU validation")
    if not smoke and config.training["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU does not support BF16; explicitly select float32")
    source_root = Path(source_root or Path(workspace) / "direct-v1").resolve()
    image_root = Path(image_root or Path(workspace) / "imagenet").resolve()
    root = output_root(config, workspace, smoke).resolve()
    if root == source_root or root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise ValueError("fine-tuning and frozen source roots must be separate")
    setup_logging(root / "logs", name="finetune")
    with exclusive_lock(root / "campaign.lock", timeout=0):
        catalog_path = root / "campaigns" / config.name / config.plan()["plan_id"] / "catalog.json"
        catalog = {"plan_id": config.plan()["plan_id"], "imagenet": [], "frozen": [], "evaluations": [],
                   "scientific": not smoke}
        failures = []
        transfer_datasets = {}
        session = None
        incremental = None
        if backup and not smoke:
            from ..remote import RemoteSession
            session = RemoteSession(config, root)
            session.remote_prefix = f"finetune-v1/{config.name}/"
            session.start()
            from .backup import IncrementalBackup
            incremental = IncrementalBackup(session, backup_interval)
        manifest = None
        prepared_networks = {}
        try:
            sources = {} if smoke else preflight_sources(config, source_root)
            if not smoke:
                # Exercise every exact pinned graph before a potentially large download.
                for name in config.campaign.backbones:
                    source = next(r for r in sources.values() if r["cell"]["backbone"] == name)
                    net, pretrained, weights = _network(BACKBONES[name], root / "cache" / "weights", target,
                        os.environ.get("HF_TOKEN"), source["backbone_weights"])
                    with torch.no_grad(), autocast(target, config.training["precision"]):
                        value = forward_features(net, torch.zeros(1, 3, 224, 224, device=target), BACKBONES[name])
                        if not torch.isfinite(value).all():
                            raise RuntimeError(f"{name}: nonfinite backbone preflight")
                    prepared_networks[name] = (net.to("cpu"), pretrained, weights)
                    del net, value
                    torch.cuda.empty_cache()
                if image_storage == "packed":
                    from .jpeg_store import prepare_packs
                    datasets, data_identity = prepare_packs(image_root,
                        token=os.environ.get("HF_TOKEN"), download=download_images)
                else:
                    if download_images:
                        from ..data.imagenet import prepare_imagenet
                        prepare_imagenet(image_root, token=os.environ.get("HF_TOKEN"))
                    datasets, data_identity = image_datasets(image_root)
            else:
                generator = torch.Generator().manual_seed(91)
                datasets = {split: TensorDataset(torch.randn(rows, 3, 8, 8, generator=generator),
                            torch.arange(rows) % 4) for split, rows in (("train", 32), ("val", 8))}
                data_identity = {"recipe": "synthetic-finetune-v1", "seed": 91}
            for name in config.campaign.backbones:
                spec = BACKBONES[name]
                if smoke:
                    spec = replace(spec, dim=16, recipe="mrl", k_list=(2, 8, 12, 16),
                        train_k_list=(8, 12), e2_anchors=(8, 12),
                        preset=replace(spec.preset, hidden_dim=64, k_aux=8, ot_microbatch=4))
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(101)
                        teacher = nn.Sequential(nn.Flatten(), nn.Linear(192, 16), nn.Tanh()).to(target)
                    train_transform = eval_transform = nn.Identity()
                    preprocessing = {"synthetic": True}
                    weights = {"synthetic": "fixed-teacher"}
                else:
                    teacher, pretrained, weights = prepared_networks.pop(name)
                    teacher.to(target)
                    train_transform, eval_transform, preprocessing = transforms(spec, pretrained)
                teacher.eval().requires_grad_(False)
                base_identity = {"data": data_identity, "backbone": name, "weights": weights,
                    "transform": preprocessing, "stage": "original_backbone"}
                baseline_bundle = None
                try:
                    for cell in (c for c in config.cells() if c.backbone == name):
                        directory = root / "runs" / cell.slug
                        options = config.training_options(cell)
                        if smoke:
                            options.update(precision="float32", batch_size=4, epochs=1, checkpoint_every=1)
                        source = sources.get(cell.slug, {"synthetic": True, "cell": cell.__dict__})
                        source_identity = {k: v for k, v in source.items() if k != "path"}
                        sae = (build_sae(spec.dim, spec.preset.hidden_dim, seed=cell.seed, device="cpu")
                               if smoke else load_fitted(cell, Path(source["path"]).parent, spec, device="cpu"))
                        # Keep the frozen control off CUDA during joint training.
                        frozen_sae = deepcopy(sae)
                        student = deepcopy(teacher)
                        summary = fit_joint(cell, spec, student, teacher, sae, datasets["train"],
                            train_transform, directory, options=options, source_identity=source_identity,
                            data_identity=data_identity, preprocessing=preprocessing, device=target,
                            workers=workers, classes=4 if smoke else 1000, max_steps=2 if smoke else 0,
                            on_checkpoint=incremental.checkpoint if incremental else None)
                        if session:
                            session.on_task(f"source/{cell.slug}", summary)
                        # Teacher features are shared across the three methods, never gradients.
                        if baseline_bundle is None:
                            baseline_bundle = extract_features(teacher, datasets, eval_transform, spec,
                                root / "cache" / "frozen_features", identity=base_identity,
                                batch_size=extract_batch, workers=workers, device=target,
                                precision=options["precision"])
                        # The teacher is only needed again at the next fit;
                        # fit_joint moves it back before its next forward pass.
                        teacher.to("cpu")
                        baseline_signature = canonical_hash({"features": baseline_bundle.identity, "source": source_identity})
                        baseline = evaluate_model(cell, spec, frozen_sae, baseline_bundle, root / "baseline", config,
                            signature=baseline_signature, summary={}, device=device, baseline=True, smoke=smoke)
                        catalog["frozen"].append(baseline)
                        del frozen_sae
                        atomic_json(catalog_path, catalog)
                        # The backbone changed; cached original embeddings must not be reused.
                        checkpoint_id = sha256_file(directory / "checkpoint.pt")
                        bundle = extract_features(student, datasets, eval_transform, spec,
                            root / "cache" / "joint_features", identity={**base_identity,
                                "stage": "joint", "checkpoint_sha256": checkpoint_id},
                            batch_size=extract_batch, workers=workers, device=target,
                            precision=options["precision"])
                        try:
                            if config.transfer["enabled"]:
                                from .transfer import evaluate_cell
                                def persist_transfer(record):
                                    catalog["evaluations"].append(record)
                                    atomic_json(catalog_path, catalog)
                                    atomic_json(catalog_path.parent / "transfer.json",
                                        {"records": catalog["evaluations"], "scientific": not smoke})
                                evaluate_cell(config, root, cell, spec, student, sae, eval_transform,
                                    preprocessing, summary, device=device, workers=workers,
                                    extract_batch=extract_batch, data_cache=transfer_datasets,
                                    smoke=smoke, on_result=persist_transfer)
                            # Search and probes consume cached features plus the SAE.
                            del student
                            signature = canonical_hash({"checkpoint": checkpoint_id, "features": bundle.identity})
                            record = evaluate_model(cell, spec, sae, bundle, root, config, signature=signature,
                                summary=summary, device=device, smoke=smoke)
                            catalog["imagenet"].append(record)
                            atomic_json(catalog_path, catalog)
                            if incremental:
                                incremental.results()
                        finally:
                            close_bundle(bundle)
                        del sae
                        if target.type == "cuda":
                            torch.cuda.empty_cache()
                finally:
                    if baseline_bundle is not None:
                        close_bundle(baseline_bundle)
                    del teacher
        except BaseException as error:
            failures.append({"error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            atomic_json(catalog_path, catalog)
            try:
                manifest = write_report(config, root, catalog, failures)
            finally:
                if session:
                    try:
                        synced = session.finish(scientific_success=bool(manifest and manifest["complete"]))
                        if not synced and not failures:
                            raise RuntimeError("HF backup incomplete; run python -m mpsae.finetune sync "
                                               "with the same --config and --workspace")
                    except Exception as error:
                        if not failures:
                            raise
                        get_logger("finetune").error("Final backup also failed: %s", type(error).__name__)
        if not manifest["complete"]:
            raise RuntimeError(f"incomplete fine-tuning report: {root / 'reports'}")
        return manifest
