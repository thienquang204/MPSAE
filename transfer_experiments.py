#!/usr/bin/env python3
"""Run the specified frozen crop-transfer suite; no experiments run on import."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import logging
import math
import os
import shutil
import statistics
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch

from transfer_representations import (
    DIMS, PROTOCOL, SPARSE_METHODS, atomic_json, digest, encode_sparse, event,
    extract_features, file_hash, imagenet_dataset, load_parent, run_lock, train_compressor,
)


def csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/transfer"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/transfer_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/frozen_transfer"))
    p.add_argument("--imagenet-root", type=Path, default=Path("data/imagenet1k"))
    p.add_argument("--imagenet-backend", choices=("hf", "imagefolder"), default="hf")
    p.add_argument("--imagenet-dataset-id", default="ILSVRC/imagenet-1k")
    p.add_argument("--imagenet-revision", default="main")
    p.add_argument("--tasks", type=csv_list, default=["detection", "mask", "pose"])
    p.add_argument("--architectures", type=csv_list, default=["resnet18", "resnet50"])
    p.add_argument("--methods", type=csv_list, default=["matryoshka", *SPARSE_METHODS])
    p.add_argument("--seeds", type=lambda x: [int(s) for s in csv_list(x)], default=[0, 1, 2])
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prepare-only", action="store_true", help="Download/validate downstream datasets and generate shared manifests, then stop")
    p.add_argument("--plan", action="store_true", help="Print configuration without downloads, extraction or training")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--feature-batch-size", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=1024)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--ot-microbatch", type=int, default=32)
    p.add_argument("--mpsaev2-extra-epochs", type=int, default=0)
    p.add_argument("--proposal-budget", type=int, choices=(128, 256, 512), default=256)
    p.add_argument("--edge-model", default="")
    p.add_argument("--edge-model-url", default=None)
    p.add_argument("--pascal3d-root", default="")
    p.add_argument("--pascal3d-url", default=None)
    p.add_argument("--voc2007-trainval-url", default=None)
    p.add_argument("--voc2007-test-url", default=None)
    p.add_argument("--voc2012-url", default=None)
    p.add_argument("--strict-dataset-counts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-verify-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-download-bytes", type=int, default=30 << 30)
    p.add_argument("--download-timeout", type=int, default=60)
    return p


class CombinedRecords:
    """Concatenate disk-backed manifests without copying JSON objects into RAM."""
    def __init__(self, fit, evaluation):
        self.parts = (fit, evaluation)
        self.fit_count = len(fit)
        self.fingerprint = digest([part.fingerprint for part in self.parts])
        # Preserve evaluation image/ground-truth metadata even when EdgeBoxes
        # produced zero proposals for an image. Such images remain in AP's
        # denominator. The fingerprint also binds every shared sampling plan.
        self.evaluation_metadata = getattr(evaluation, "metadata", {})
        self.metadata = {"fingerprint": self.fingerprint, "fit_count": self.fit_count}

    def __len__(self):
        return sum(len(part) for part in self.parts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.parts[0][index] if index < self.fit_count else self.parts[1][index - self.fit_count]


def validate(args):
    if not args.tasks or set(args.tasks) - {"detection", "mask", "pose"}:
        raise ValueError("Tasks must be a nonempty subset of detection,mask,pose")
    if not args.architectures or set(args.architectures) - set(DIMS):
        raise ValueError("Architectures must be resnet18,resnet50")
    if not args.methods or set(args.methods) - {"matryoshka", *SPARSE_METHODS}:
        raise ValueError("Methods must be matryoshka,csr,mpsae,csrv2,mpsaev2")
    if any(len(value) != len(set(value)) for value in (args.tasks, args.architectures, args.methods, args.seeds)):
        raise ValueError("Duplicate tasks/architectures/methods/seeds are not allowed")
    if set(args.seeds) != {0, 1, 2}:
        raise ValueError("This final protocol uses all three paired seeds 0,1,2")
    if set(args.architectures) != set(DIMS) and any(t != "detection" for t in args.tasks):
        raise ValueError("Mask/pose schedule selection requires all four parents across both architectures")
    if args.workers < 0 or min(args.feature_batch_size, args.encode_batch_size, args.log_every, args.download_timeout) <= 0:
        raise ValueError("Batch sizes, log interval and timeout must be positive; workers nonnegative")
    if args.max_download_bytes <= 0:
        raise ValueError("Download byte limit must be positive")
    if args.ot_microbatch < 2 or args.ot_microbatch > 1024 or args.mpsaev2_extra_epochs < 0:
        raise ValueError("Invalid MMPOT microbatch or extra epoch count")
    if not args.strict_dataset_counts:
        raise ValueError("Reduced datasets need a separately labelled pilot protocol; final suite requires official counts")


def release(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cache_path(args, task, phase, arch, parent):
    return args.cache_dir / "crops" / task / phase / f"{arch}_{parent}.npy"


def head_families(task):
    return ("affine", "mlp") if task == "detection" else ("mlp", "affine")


def compact_budgets(task, arch, method, family):
    if method == "matryoshka":
        if task == "detection" and family == "affine":
            return [2 ** exponent for exponent in range(3, int(math.log2(DIMS[arch])))]
        return [16, 32, 64, 128]
    return [8, 16, 64, 256] if task == "detection" and family == "affine" else [16, 64]


def fit_evaluate(args, task, phase, records, features, architecture, parent, provenance,
                 method, budget, seed, family, updates, device, compressor=None, checkpoint_step=None):
    from transfer_tasks import train_head, evaluate_head
    run_dir = args.output_dir / task / phase / architecture / method / f"budget_{budget}" / family / f"seed_{seed}"
    result_file = run_dir / ("result.json" if checkpoint_step is None else f"result_step_{checkpoint_step}.json")
    identity = {"source": args.source_hash, "records": records.fingerprint, "architecture": architecture,
                "parent": provenance, "method": method, "budget": budget, "seed": seed, "head_family": family,
                "updates": updates, "checkpoint_step": checkpoint_step, "compressor": compressor}
    signature = digest(identity)
    if result_file.exists():
        result = json.loads(result_file.read_text())
        if result.get("signature") != signature:
            raise RuntimeError(f"Stale result configuration: {result_file}")
        return result
    fit_ids = np.arange(records.fit_count, dtype=np.int64)
    eval_ids = np.arange(records.fit_count, len(records), dtype=np.int64)
    event(args.output_dir, "head", task=task, phase=phase, architecture=architecture, method=method,
          budget=budget, seed=seed, family=family)
    fit = train_head(task, features, records, fit_ids, run_dir, seed=seed, head_family=family,
                     updates=updates, device=device, sampling_dir=args.output_dir / task / phase / "sampling",
                     checkpoint_steps=(1000, 2500, 5000), log_every=args.log_every)
    metrics = evaluate_head(task, features, records, eval_ids, run_dir, device=device, checkpoint_step=checkpoint_step)
    sparse = method in SPARSE_METHODS
    result = {
        "signature": signature, "protocol_id": PROTOCOL, "task": task, "phase": phase,
        "architecture": architecture, "parent": parent, "parent_checkpoint_hash": provenance["sha256"],
        "method": method, "representation_fit": "ImageNet-global", "head_seed": seed,
        "compressor_seed": seed if sparse else None, "budget": budget,
        "k_train": budget if sparse else None, "k_eval": budget if sparse else None,
        "m": budget if method == "matryoshka" else None, "d": DIMS[architecture],
        "h": 8192 if sparse else None, "head_family": family, "proposal_budget": args.proposal_budget if task == "detection" else None,
        "crop_transform_id": "context1.2-floor-ceil-meanpad-rgb224-v1", "precision": "fp32",
        "compressor": compressor, "training": fit, "metrics": metrics,
        "representation_payload_bytes_per_region": 8 * budget + 4 if sparse else 4 * budget,
        "storage_convention": "padded-topk-with-uint32-count" if sparse else "dense-prefix-view-of-full-parent-cache",
        "source_fingerprint": args.source_hash, "seed_interpretation": "compressor+head" if sparse else "head-only",
    }
    if sparse:
        result["storage"] = json.loads((Path(features.path) / "metadata.json").read_text())
    parent_cache = cache_path(args, task, phase, architecture, parent).with_suffix(".json")
    extraction = json.loads(parent_cache.read_text())
    result.update(
        stored_bytes_per_region=fit["stored_bytes_per_region"],
        head_parameters=fit["head_parameters"],
        encoder_parameters=provenance["encoder_parameters"],
        compressor_parameters=(DIMS[architecture] * 8192 + 8192 + DIMS[architecture]) if sparse else 0,
        encoder_time=extraction["seconds"],
        compressor_time=features.metadata.get("seconds", 0.0) if sparse else 0.0,
        compressor_fit_time=compressor["fit_seconds"] if compressor else 0.0,
        head_time=fit["head_fit_time"],
        evaluation_time=metrics["evaluation_time"],
        runtime_memory={name: fit[name] for name in ("peak_gpu_memory_bytes", "peak_cpu_rss_bytes") if name in fit},
        cost_scope="Shared parent extraction/compression wall time includes cache I/O; do not sum repeated cache costs across readouts. Head/evaluation time is per fit. End-to-end deployment latency is not measured.",
    )
    atomic_json(result_file, result)
    return result


def flat_metrics(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flat_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def metric_delta_unit(name):
    """Only task scores have preservation deltas; counts/timing do not."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf in {"AP", "AP50", "AP75", "AP50_no_refinement", "AP75_no_refinement",
                "primary_macro_metric", "macro_IoU", "mean_IoU", "instance_weighted_IoU",
                "conditional_IoU_gain", "recall50", "recall75", "macro_Acc30", "macro_Acc15",
                "Acc30", "Acc15"}:
        return "percentage_points"
    if leaf in {"macro_MedErr", "MedErr", "pooled_MedErr"}:
        return "degrees"
    return None


def summarize(args, rows):
    from transfer_tasks import paired_bootstrap
    full = {(r["task"], r["architecture"], r["parent"], r["head_family"], r["head_seed"]): r
            for r in rows if r["method"].endswith("_full")}
    grouped = {}
    for row in rows:
        reference = full[(row["task"], row["architecture"], row["parent"], row["head_family"], row["head_seed"])]
        metrics, parent_metrics = flat_metrics(row["metrics"]), flat_metrics(reference["metrics"])
        row["deltas_from_own_parent"] = {
            name: value - parent_metrics[name] for name, value in metrics.items()
            if name in parent_metrics and metric_delta_unit(name) is not None}
        row["primary_delta_percentage_points"] = 100. * row["deltas_from_own_parent"]["primary_macro_metric"]
        key = (row["task"], row["architecture"], row["method"], row["budget"], row["head_family"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, group in grouped.items():
        summary = dict(zip(("task", "architecture", "method", "budget", "head_family"), key))
        summary["seeds"] = [r["head_seed"] for r in group]
        if set(summary["seeds"]) != {0, 1, 2}:
            raise RuntimeError(f"Incomplete final seed group: {key}")
        for name in ("stored_bytes_per_region", "head_parameters", "feature_cache_bytes_on_disk",
                     "measured_nnz", "unpadded_payload_bytes_per_region"):
            values = [r["training"].get(name) for r in group]
            if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                summary[name] = statistics.mean(values)
                summary[name + ".mean"] = summary[name]
                summary[name + ".sample_std"] = statistics.stdev(values)
        if "stored_bytes_per_region" not in summary or "head_parameters" not in summary:
            raise RuntimeError(f"Missing storage/parameter accounting for {key}")
        for name in sorted(set.intersection(*(set(flat_metrics(r["metrics"])) for r in group))):
            values = [flat_metrics(r["metrics"])[name] for r in group]
            summary[name + ".mean"] = statistics.mean(values)
            summary[name + ".sample_std"] = statistics.stdev(values)
            unit = metric_delta_unit(name)
            if unit is not None and all(name in r["deltas_from_own_parent"] for r in group):
                differences = [r["deltas_from_own_parent"][name] for r in group]
                summary[name + ".delta_from_parent"] = statistics.mean(differences)
                factor = 100. if unit == "percentage_points" else 1.
                suffix = ".delta_percentage_points" if unit == "percentage_points" else ".delta_degrees"
                summary[name + suffix] = factor * statistics.mean(differences)
                summary[name + suffix + "_sample_std"] = factor * statistics.stdev(differences)
        if key[0] in ("mask", "pose") and not key[2].endswith("_full"):
            parents = [full[(r["task"], r["architecture"], r["parent"], r["head_family"], r["head_seed"])] for r in group]
            compact_paths = [r["metrics"]["prediction_path"] for r in group]
            parent_paths = [r["metrics"]["prediction_path"] for r in parents]
            metrics = ("macro_IoU",) if key[0] == "mask" else ("macro_Acc30", "macro_MedErr")
            bootstrap_identity = digest({"source": args.source_hash, "group": key,
                                         "predictions": [(p, file_hash(p)) for p in compact_paths + parent_paths]})
            bootstrap_directory = args.output_dir / "uncertainty" / key[0] / key[1] / key[2] / f"budget_{key[3]}" / key[4]
            intervals = {}
            for condition in ("clean", "translated"):
                for metric in metrics:
                    interval_name = condition + "_" + metric
                    path = bootstrap_directory / (interval_name + ".json")
                    if path.exists():
                        cached = json.loads(path.read_text())
                        if cached.get("signature") != bootstrap_identity:
                            raise RuntimeError(f"Bootstrap predictions changed: {path}")
                        interval = cached["interval"]
                    else:
                        interval = paired_bootstrap(compact_paths, parent_paths, task=key[0], metric=metric,
                                                    condition=condition, replicates=1000, seed=2026)
                        atomic_json(path, {"signature": bootstrap_identity, "interval": interval})
                    intervals[interval_name] = interval
            summary["paired_image_bootstrap"] = intervals
        summaries.append(summary)
    atomic_json(args.output_dir / "summary.json", {"protocol_id": PROTOCOL, "metric_units": "accuracy/AP/IoU means and delta_from_parent are proportions; delta_percentage_points is explicitly multiplied by 100; rotation errors and delta_degrees are degrees",
                "metric_directions": {"AP50": "higher_is_better", "AP75": "higher_is_better", "IoU": "higher_is_better", "Acc30": "higher_is_better", "Acc15": "higher_is_better", "MedErr": "lower_is_better"}, "rows": rows, "summary": summaries,
                "limitations": ["MRL and sparse parents are different pretrained encoders", "V2 and MPSAE arms are labelled method extensions", "Timing is cached component timing, not measured end-to-end deployment latency", "Optional PCA, regional adaptation, keypoints, texture and descriptor tasks are not part of the primary three-task matrix"]})
    columns = sorted({key for row in summaries for key, value in row.items() if not isinstance(value, dict)})
    csv_path = args.output_dir / "summary.csv"
    temporary = csv_path.with_name("summary.csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, csv_path)
    return summaries


def package_results(args):
    path = args.output_dir / "transfer_results.zip"
    temporary = path.with_suffix(".tmp.zip")
    # Training checkpoints, caches and individual predictions stay on persistent
    # storage. The portable bundle contains tables, metrics and provenance.
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(args.output_dir.rglob("*.json")):
            if "sampling" not in item.parts and item.name != "RUN_COMPLETE.json":
                bundle.write(item, str(item.relative_to(args.output_dir)))
        bundle.write(args.output_dir / "summary.csv", "summary.csv")
        for item in sorted(args.output_dir.rglob("data_*.skipped.jsonl")):
            bundle.write(item, str(item.relative_to(args.output_dir)))
        for extension in ("*.png", "*.pdf"):
            for item in sorted((args.output_dir / "plots").glob(extension)):
                if ".tmp." not in item.name:
                    bundle.write(item, str(item.relative_to(args.output_dir)))
    os.replace(temporary, path)
    return path


def execute(args):
    from transfer_data import prepare_datasets, build_task_records, RegionCropDataset
    from transfer_tasks import FeatureStore, evaluate_controls, select_update_count, validate_readout_interface
    if "detection" in args.tasks:
        import cv2
        if not cv2.__version__.startswith("4.13.") or not hasattr(cv2, "ximgproc"):
            raise RuntimeError("Detection requires opencv-contrib-python-headless 4.13.x")
    if "pose" in args.tasks:
        from scipy.io import loadmat  # annotation dependency preflight before downloads
    args.source_hash = digest({name: file_hash(Path(__file__).with_name(name)) for name in (
        "transfer_experiments.py", "transfer_representations.py", "transfer_tasks.py", "transfer_data.py", "transfer_reporting.py", "csr_vs_mmpot_imagenet.py")})
    configuration = {key: value for key, value in vars(args).items() if key not in {"prepare_only", "plan", "download", "data_verify_images", "workers", "log_every", "feature_batch_size", "encode_batch_size"}}
    config_path = args.output_dir / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text())["signature"] != digest(configuration):
        raise RuntimeError("Output configuration changed; select a new TRANSFER_OUTPUT directory to preserve prior results")
    atomic_json(config_path, {"signature": digest(configuration), "configuration": configuration})
    complete_file = args.output_dir / "RUN_COMPLETE.json"
    if complete_file.exists() and not args.prepare_only:
        completion = json.loads(complete_file.read_text())
        if (completion.get("configuration_signature") == digest(configuration)
                and all((args.output_dir / name).is_file() for name in
                        ("transfer_results.zip", "summary.json", "summary.csv"))):
            event(args.output_dir, "already_complete", result=str(args.output_dir / "summary.json"))
            return 0
    datasets = prepare_datasets(args)
    if any(method in SPARSE_METHODS for method in args.methods):
        event(args.output_dir, "prepare_imagenet_train", root=str(args.imagenet_root))
        prepared_imagenet, imagenet_identity = imagenet_dataset(args)
        datasets["imagenet_train"] = imagenet_identity
        del prepared_imagenet
    atomic_json(args.output_dir / "datasets.json", datasets)
    records = {}
    for task in args.tasks:
        for phase in ("development", "final"):
            fitting = ("train" if phase == "development" else "trainval") if task == "detection" else ("fit" if phase == "development" else "train")
            evaluation = ("val" if phase == "development" else "test") if task == "detection" else ("dev" if phase == "development" else "val")
            records[task, phase] = CombinedRecords(build_task_records(task, fitting, args), build_task_records(task, evaluation, args))
            if records[task, phase].fit_count == 0 or len(records[task, phase]) == records[task, phase].fit_count:
                raise RuntimeError(f"Empty fitting or evaluation data for {task}/{phase}")
            for label, manifest in zip(("fit", "evaluation"), records[task, phase].parts):
                report_path = args.output_dir / task / phase / f"data_{label}.json"
                atomic_json(report_path, manifest.metadata)
                skipped = manifest.metadata.get("skipped_records")
                if skipped and Path(skipped).is_file():
                    destination = report_path.with_suffix(".skipped.jsonl")
                    temporary = destination.with_suffix(".tmp")
                    shutil.copyfile(skipped, temporary)
                    os.replace(temporary, destination)
    if args.prepare_only:
        event(args.output_dir, "data_ready", tasks=args.tasks)
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run the Docker launcher with NVIDIA GPU access")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    atomic_json(args.output_dir / "readout_interface_check.json", validate_readout_interface(device))
    atomic_json(args.output_dir / "runtime.json", {"torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "precision": "fp32", "tf32": False})
    # Priors use fitting data only. Save independently of method/architecture.
    for task in args.tasks:
        if task != "detection":
            for phase in ("development", "final"):
                data = records[task, phase]
                target = args.output_dir / task / phase / "controls"
                report = evaluate_controls(task, data, np.arange(data.fit_count), np.arange(data.fit_count, len(data)), target, device=device)
                atomic_json(target / "controls.json", report)
    # Full-parent development runs always precede compact-method fitting.
    development = {task: [] for task in args.tasks}
    for arch in args.architectures:
        for parent in ("mrl", "timm"):
            network, provenance = load_parent(arch, parent, args, device)
            for task in args.tasks:
                data = records[task, "development"]
                path = extract_features(network, RegionCropDataset(data), provenance, data.fingerprint,
                                        cache_path(args, task, "development", arch, parent), args, device)
                features = FeatureStore(path)
                family = "affine" if task == "detection" else "mlp"
                for step in ((5000,) if task == "detection" else (1000, 2500, 5000)):
                    result = fit_evaluate(args, task, "development", data, features, arch, parent, provenance,
                                          parent + "_full", DIMS[arch], 0, family, 5000, device, checkpoint_step=step)
                    development[task].append({"parent_id": arch + "/" + parent, "update_count": step,
                                              "primary_macro_metric": result["metrics"]["primary_macro_metric"]})
                del features
            del network
            release(device)
    schedules = {task: 5000 if task == "detection" else select_update_count(development[task]) for task in args.tasks}
    sensitivity = []
    for task in args.tasks:
        if task == "detection":
            continue
        controls = json.loads((args.output_dir / task / "development" / "controls" / "controls.json").read_text())
        prior_names = (("class_agnostic_mean_mask", "true_category_mean_mask_privileged") if task == "mask"
                       else ("known_category_training_rotation_medoid",))
        priors = {name: controls[name]["primary_macro_metric"] for name in prior_names}
        if any(value is None for value in priors.values()):
            raise RuntimeError(f"Missing development prior score for {task}")
        for parent_result in development[task]:
            if parent_result["update_count"] != schedules[task]:
                continue
            weak = parent_result["primary_macro_metric"] <= max(priors.values())
            item = {"task": task, **parent_result, "priors": priors, "weak_relative_to_prior": weak,
                    "decision": "retain_common_primary_protocol; report limited sensitivity" if weak else "retain_common_primary_protocol",
                    "diagnostics": {}}
            if weak:
                arch, parent = parent_result["parent_id"].split("/")
                path = cache_path(args, task, "development", arch, parent)
                provenance = json.loads(path.with_suffix(".json").read_text())["parent"]
                features = FeatureStore(path)
                # Predetermined full-parent diagnostics only. Their scores never
                # select a method-specific head or alter final compact rankings.
                for family in ("affine", "mlp256"):
                    check = fit_evaluate(args, task, "development", records[task, "development"], features,
                                         arch, parent, provenance, parent + "_full", DIMS[arch], 0, family, 5000, device)
                    item["diagnostics"][family] = check["metrics"]
                del features
            sensitivity.append(item)
    atomic_json(args.output_dir / "sensitivity.json", sensitivity)
    atomic_json(args.output_dir / "locked_schedules.json", {"updates": schedules, "development": development,
                "selection": "seed0 full parents only; compact rankings and held-out final labels excluded", "detection": "fixed5000", "controls": "See per-task controls.json for parent sensitivity; weak tasks remain reported"})
    final_rows = []
    for arch in args.architectures:
        for parent in ("mrl", "timm"):
            network, provenance = load_parent(arch, parent, args, device)
            paths = {}
            for task in args.tasks:
                data = records[task, "final"]
                paths[task] = extract_features(network, RegionCropDataset(data), provenance, data.fingerprint,
                                               cache_path(args, task, "final", arch, parent), args, device)
            imagenet_path = None
            if parent == "timm" and any(m in SPARSE_METHODS for m in args.methods):
                imagenet, identity = imagenet_dataset(args)
                imagenet_path = extract_features(network, imagenet, provenance, identity,
                                                  args.cache_dir / "imagenet" / f"{arch}_timm.npy", args, device)
                del imagenet
            del network
            release(device)
            for task in args.tasks:
                data = records[task, "final"]
                for family in head_families(task):
                    methods = [(parent + "_full", DIMS[arch])]
                    if parent == "mrl" and "matryoshka" in args.methods:
                        methods += [("matryoshka", m) for m in compact_budgets(task, arch, "matryoshka", family)]
                    for method, budget in methods:
                        features = FeatureStore(paths[task], width=budget)
                        for seed in args.seeds:
                            final_rows.append(fit_evaluate(args, task, "final", data, features, arch, parent, provenance,
                                                          method, budget, seed, family, schedules[task], device))
                        del features
            if parent == "timm":
                for method in [m for m in args.methods if m in SPARSE_METHODS]:
                    budgets = sorted({k for task in args.tasks for family in head_families(task) for k in compact_budgets(task, arch, method, family)})
                    for seed in args.seeds:
                        for k in budgets:
                            checkpoint = args.output_dir / "compressors" / arch / method / f"seed_{seed}" / f"k_{k}.pt"
                            model, compressor = train_compressor(checkpoint, imagenet_path, provenance, method, k, seed, args, device)
                            sparse_paths = {}
                            for task in args.tasks:
                                families = [f for f in head_families(task) if k in compact_budgets(task, arch, method, f)]
                                if not families:
                                    continue
                                sparse_paths[task] = encode_sparse(model, k, paths[task], args.cache_dir / "sparse" / task / arch / method / f"seed_{seed}" / f"k_{k}", compressor, args, device)
                            # Heads consume immutable cached codes. Release the
                            # compressor before fitting to bound GPU memory and
                            # keep head-memory reporting comparable to full parents.
                            del model
                            release(device)
                            for task, sparse_path in sparse_paths.items():
                                families = [f for f in head_families(task) if k in compact_budgets(task, arch, method, f)]
                                features = FeatureStore(sparse_path)
                                for family in families:
                                    final_rows.append(fit_evaluate(args, task, "final", records[task, "final"], features, arch, parent,
                                                                  provenance, method, k, seed, family, schedules[task], device, compressor))
                                del features
                            release(device)
    summaries = summarize(args, final_rows)
    from transfer_reporting import write_transfer_plots
    plot_report = write_transfer_plots(summaries, args.output_dir)
    atomic_json(args.output_dir / "plots.json", plot_report)
    bundle = package_results(args)
    atomic_json(complete_file, {"protocol_id": PROTOCOL, "configuration_signature": digest(configuration), "status": "complete", "fits": len(final_rows), "groups": len(summaries), "bundle": str(bundle)})
    event(args.output_dir, "complete", fits=len(final_rows), summary=str(args.output_dir / "summary.json"))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    if args.plan:
        print(json.dumps(vars(args), indent=2, default=str))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.prepare_only:
        device = torch.device(args.device)
        if device.type == "cuda" and (not torch.cuda.is_available() or
                                       (device.index is not None and device.index >= torch.cuda.device_count())):
            raise RuntimeError(f"Requested CUDA device {args.device} is unavailable")
    # Normalize before hashing so equivalent invocations share cache identities.
    for name in ("data_root", "cache_dir", "output_dir", "imagenet_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Lock failures must not overwrite the running owner's status/failure files.
    # Deduplicate identical roots and use a stable order for multiple processes.
    with contextlib.ExitStack() as locks:
        for root in sorted({args.output_dir, args.cache_dir, args.data_root}, key=str):
            locks.enter_context(run_lock(root))
        try:
            return execute(args)
        except BaseException as exc:
            event(args.output_dir, "failed", error=str(exc), exception=type(exc).__name__)
            (args.output_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise


if __name__ == "__main__":
    sys.exit(main())
