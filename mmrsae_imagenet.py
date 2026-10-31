"""Manually tunable, GPU-only ImageNet experiments for the MMRSAE family.

One selected variant and K are fitted per requested backbone. There is no
automatic ablation matrix, candidate selection, transfer task, or CPU fallback.
All downloads, GPU checks and fitting happen only when this entry point runs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

from repair_data import atomic_json, canonical_hash, exclusive_lock, sha256_file
from repair_runtime import (epoch_batches, load_checkpoint, named_parameters, pack_adam,
                            restore_adam, restore_rng, rng_state, save_checkpoint)
from wandb_tracking import (add_wandb_arguments, finish_wandb, init_wandb,
                            log_wandb_metrics, update_wandb_summary)


PROTOCOL = "mmrsae-imagenet-manual-v1"
VARIANTS = ("mmrsae", "mmrsae-rel", "mmrsae-wide", "mmrsae-annealed")
SOURCE_FILES = ("mmrsae_imagenet.py", "mmrsae.py", "mmrsae_data.py", "mmrsae_gpu.py",
                "mpsae_repair.py", "repair_data.py", "repair_runtime.py", "repair_benchmark.py",
                "sparse_gpu_retrieval.py", "retrieval_timing.py", "wandb_tracking.py",
                "csrc/sparse_retrieval.cpp", "csrc/sparse_retrieval.cu")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backbones", default="resnet50,resnet18")
    parser.add_argument("--variant", choices=VARIANTS, default="mmrsae")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--repair-rank", type=int, default=32)
    parser.add_argument("--mask", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--clean-probability", type=float, default=.2)
    parser.add_argument("--erase-probability", type=float, default=.25)
    parser.add_argument("--relation-weight", type=float)
    parser.add_argument("--ot-weight", type=float)
    parser.add_argument("--mixture-weight", type=float)
    parser.add_argument("--teacher-temperature", type=float, default=.1)
    parser.add_argument("--student-temperature", type=float, default=.1)
    parser.add_argument("--ot-mass", type=float, default=.9)
    parser.add_argument("--ot-eta", type=float, default=.2)
    parser.add_argument("--ot-iters", type=int, default=100)
    parser.add_argument("--ot-max-refine-iters", type=int, default=2000)
    parser.add_argument("--ot-tol", type=float, default=1e-4)
    parser.add_argument("--ot-microbatch", type=int, default=32)
    parser.add_argument("--k-aux", type=int, default=512)
    parser.add_argument("--dead-steps", type=int, default=30)
    parser.add_argument("--anchor-count", type=int, default=1024)
    parser.add_argument("--anchor-chunk-size", type=int, default=128)
    parser.add_argument("--bank-refresh", type=int, default=25)
    parser.add_argument("--alpha-power-iterations", type=int, default=50)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500, help="Per-backbone update cap; 0 runs all requested epochs")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-eps", type=float, default=6.25e-10)
    parser.add_argument("--seed", type=int, default=1227, help="Model/mask/data-order seed; feature selection uses data-seed")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10, help="Console/W&B update interval; local scalars saved every update")
    parser.add_argument("--eval-every", type=int, default=0, help="Optional completed-epoch evaluation interval; 0 evaluates only endpoint")

    parser.add_argument("--evaluation", choices=("dev", "official"), default="dev")
    parser.add_argument("--train-size", type=int, default=128000, help="Fitting/gallery rows; 0 uses all eligible training rows")
    parser.add_argument("--dev-size", type=int, default=10000, help="Training-derived development queries, disjoint from fitting and anchors")
    parser.add_argument("--max-val", type=int, default=0, help="Official validation limit; 0 uses the full official partition")
    parser.add_argument("--data-backend", choices=("hf", "imagefolder"), default="hf")
    parser.add_argument("--data-root", type=Path, default=Path("data/imagenet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/mmrsae"))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ff2048-train", type=Path)
    parser.add_argument("--ff2048-val", type=Path)
    parser.add_argument("--ff2048-revision", default="main")
    parser.add_argument("--ff2048-train-sha256")
    parser.add_argument("--ff2048-val-sha256")
    parser.add_argument("--imagenet-revision", default="main")
    parser.add_argument("--parent-revision", default="main")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--gallery-chunk-size", type=int, default=65536)
    parser.add_argument("--score-buffer-mib", type=int, default=256)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.8)
    parser.add_argument("--faiss-temp-memory-mib", type=int, default=512)
    parser.add_argument("--retrieval-warmup", type=int, default=3)
    parser.add_argument("--retrieval-repeats", type=int, default=5)
    parser.add_argument("--timed-retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dense-reference", action=argparse.BooleanOptionalAction, default=False,
                        help="Also measure the full frozen parent; requires its dense gallery to fit on the GPU")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mmrsae"))
    parser.add_argument("--run-name", help="One trial name; use a different name after changing learning settings")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true", help="Print settings only; no imports of GPU/training libraries or downloads")
    actions.add_argument("--preflight-only", action="store_true")
    actions.add_argument("--prepare-only", action="store_true")
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project=os.environ.get("WANDB_PROJECT", "MMRSAE"),
                        wandb_entity=os.environ.get("WANDB_ENTITY") or None)
    if not os.environ.get("WANDB_MODE") and not os.environ.get("WANDB_API_KEY"):
        parser.set_defaults(wandb_mode="offline")
    return parser


def validate_args(args):
    backbones = [name.strip().lower() for name in args.backbones.split(",") if name.strip()]
    if not backbones or len(set(backbones)) != len(backbones) or any(name not in ("resnet18", "resnet50") for name in backbones):
        raise ValueError("--backbones must be a unique comma-separated list of resnet18,resnet50")
    for name in ("k", "hidden_dim", "repair_rank", "anchor_count", "anchor_chunk_size", "bank_refresh",
                 "alpha_power_iterations", "epochs", "batch_size", "checkpoint_every", "log_every",
                 "feature_batch_size", "encode_batch_size", "query_batch_size", "gallery_chunk_size",
                 "score_buffer_mib", "retrieval_warmup", "retrieval_repeats", "ot_iters", "ot_max_refine_iters"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_steps", "seed", "data_seed", "train_size", "max_val", "workers", "eval_every",
                 "k_aux", "dead_steps", "faiss_temp_memory_mib"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.k > args.hidden_dim or args.ot_microbatch < 2 or args.batch_size < 2:
        raise ValueError("Require K <= hidden dimension and batch/OT group sizes >=2")
    if args.evaluation == "dev" and args.dev_size < 1:
        raise ValueError("--dev-size must be positive for development evaluation")
    for name in ("lr", "adam_eps", "teacher_temperature", "student_temperature", "ot_eta", "ot_tol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "relation_weight", "ot_weight"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("clean_probability", "erase_probability", "mixture_weight"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be in [0,1]")
    if not 0 < args.ot_mass < 1 or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("OT mass must be in (0,1) and GPU memory fraction in (0,1]")
    if args.ot_max_refine_iters < args.ot_iters:
        raise ValueError("OT refinement bound cannot be smaller than its initial iteration limit")
    if args.run_name and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name) or ".." in args.run_name):
        raise ValueError("--run-name must be a simple name containing letters, digits, underscores, dots, or dashes")
    for field in ("ff2048_train_sha256", "ff2048_val_sha256"):
        digest = getattr(args, field)
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return backbones


def retrieval_config(args):
    return {"query_batch_size": args.query_batch_size, "gallery_chunk_size": args.gallery_chunk_size,
            "score_buffer_mib": args.score_buffer_mib, "faiss_temp_memory_mib": args.faiss_temp_memory_mib,
            "gpu_memory_fraction": args.gpu_memory_fraction, "warmup": args.retrieval_warmup,
            "repeats": args.retrieval_repeats, "encode_batch_size": args.encode_batch_size}


def method_config(args, input_dim):
    from mmrsae import MMRSAEConfig
    return MMRSAEConfig(input_dim=input_dim, hidden_dim=args.hidden_dim, k=args.k, variant=args.variant,
        rank=args.repair_rank, seed=args.seed, mask=True if args.mask is None else args.mask,
        relation_weight=1. if args.relation_weight is None else args.relation_weight,
        ot_weight=args.ot_weight, mixture_weight=args.mixture_weight,
        teacher_temperature=args.teacher_temperature, student_temperature=args.student_temperature,
        mass=args.ot_mass, eta=args.ot_eta, max_iters=args.ot_iters, max_refine_iters=args.ot_max_refine_iters,
        tol=args.ot_tol, microbatch=args.ot_microbatch, k_aux=args.k_aux, dead_steps=args.dead_steps,
        clean_probability=args.clean_probability, erase_probability=args.erase_probability,
        anchor_chunk_size=args.anchor_chunk_size)


def serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if hasattr(value, "detach"):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def implementation_identity():
    directory = Path(__file__).resolve().parent
    return {name: sha256_file(directory / name) for name in SOURCE_FILES}


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing run/cache identity differs: {path}. Choose a new --run-name for a changed trial.")
    else:
        atomic_json(path, value)


def write_log(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(serializable(value), allow_nan=False) + "\n")
        stream.flush()


def tensor_rows(features, ids, device):
    import numpy as np
    import torch
    values = np.array(features[ids], dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise FloatingPointError("Parent feature batch contains nonfinite values")
    return torch.from_numpy(values).to(device)


def reconcile_log(path, next_step):
    """Keep evidence from a crash without presenting uncommitted updates twice."""
    if not path.exists():
        return
    committed, abandoned = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            target = committed if row["step"] < next_step else abandoned
        except (ValueError, KeyError):
            target = abandoned
        target.append(line)
    if abandoned:
        path.with_name(f"uncommitted_training_{time.time_ns()}.jsonl").write_text("\n".join(abandoned) + "\n", encoding="utf-8")
        path.write_text("\n".join(committed) + ("\n" if committed else ""), encoding="utf-8")


def evaluate(args, bundle, model, output, checkpoint_path, identity, step, *, dense=False, intermediate=False):
    from repair_benchmark import benchmark
    suffix = f"step_{step:07d}" if intermediate else "endpoint"
    directory = output / ("parent" if dense else "retrieval") / suffix
    config = {"protocol": PROTOCOL, "model": {"input_dim": model.config.input_dim,
              "hidden_dim": args.hidden_dim, "k": args.k}, "retrieval": retrieval_config(args)}
    benchmark_identity = {"experiment": identity, "checkpoint_sha256": identity["data_sha256"] if dense else sha256_file(checkpoint_path),
                          "query_source": canonical_hash(bundle.provenance["identity"]),
                          "method": "full_parent" if dense else args.variant, "pool": args.evaluation,
                          "step": 0 if dense else step, "seed": args.seed}
    result = benchmark(bundle.train, bundle.train_labels, bundle.gallery_ids, bundle.query_ids,
        None if dense else model.sae, None if dense else model.repair, config, directory, args.seed,
        identity=benchmark_identity, device=next(model.parameters()).device,
        query_features=bundle.query, query_labels=bundle.query_labels,
        timed=args.timed_retrieval and not intermediate)
    result.update({"family": "MMRSAE" if not dense else "full_parent", "variant": args.variant if not dense else None,
                   "evaluation_partition": args.evaluation, "step": step,
                   "interpretation": "manual development tuning" if args.evaluation == "dev" else
                   "official validation; any tuning or repeated inspection of these scores must be disclosed"})
    atomic_json(directory / "result.json", result)
    log_wandb_metrics("parent" if dense else "retrieval", result, step_metric="step")
    update_wandb_summary({"parent" if dense else "retrieval": result})
    return result


def diagnostic(args, bundle, model, bank, output, step, total):
    import torch
    if total < 1 or not 0 <= step <= total:
        raise ValueError("Diagnostic position must be in [0, total_updates]")
    target = output / f"diagnostic_{step:07d}.json"
    ids = bundle.fit_ids[:min(64, len(bundle.fit_ids))]
    before = model.sae.inactive_steps.clone()
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 9001)
    # Endpoint reports use completed-update counts, while the loss schedule
    # accepts zero-based update indices. Reuse the final update's budget.
    loss_step = min(step, total - 1)
    _, stats, views = model.loss(tensor_rows(bundle.train, ids, next(model.parameters()).device),
        bank=bank, row_ids=ids, mask_generator=generator, step=loss_step, total_steps=total,
        diagnostics=True, update_activity=False)
    if not torch.equal(before, model.sae.inactive_steps):
        raise RuntimeError("Diagnostic mutated SAE activity counters")
    report = {"step": step, **serializable(stats)}
    if "transported_marginals" in views:
        report["transported_marginals"] = serializable(views["transported_marginals"])
    if bank is not None:
        report["bank_refresh"] = serializable(bank.last_refresh_diagnostics)
    warnings = []
    if report.get("zero_overlap_fraction", 0.) >= .99:
        warnings.append("Almost all diagnostic codes have no landmark overlap; neighborhood learning may be ineffective")
    old, new = report.get("gradient_old_ot_norm", 0.), report.get("gradient_new_ot_norm", 0.)
    if old > 1e-12 and new < .01 * old and model.config.mixture_weight > 0 and model.config.ot_weight > 0:
        warnings.append("Neighborhood OT gradient is below 1% of old-cost gradient on this diagnostic batch")
    report["warnings"] = warnings
    atomic_json(target, report)
    log_wandb_metrics("diagnostics", report, step_metric="step")
    for message in warnings:
        print(f"MMRSAE diagnostic: {message}", flush=True)
    return report


def batch_count(rows, batch_size):
    if rows < 2:
        raise ValueError("MMRSAE fitting needs at least two examples")
    # Preserve every example without giving OT a singleton training minibatch.
    return math.ceil(rows / batch_size) - int(rows > batch_size and rows % batch_size == 1)


def training_batches(ids, seed, epoch, batch_size, offset=0):
    import numpy as np
    count = batch_count(len(ids), batch_size)
    if not 0 <= offset <= count:
        raise ValueError("Invalid resume minibatch offset")
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(ids))
    for ordinal in range(offset, count):
        first = ordinal * batch_size
        last = len(ids) if ordinal == count - 1 else first + batch_size
        yield ordinal, ids[order[first:last]]


def fit_and_evaluate(args, backbone, bundle, output, code, gpu, *, failure_context=None):
    import numpy as np
    import torch
    from mmrsae import MMRSAE, LandmarkBank

    cfg = method_config(args, int(bundle.train.shape[1]))
    device = torch.device(gpu["device"])
    per_epoch = batch_count(len(bundle.fit_ids), args.batch_size)
    total = per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    fit_recipe = {"protocol": PROTOCOL, "family": "MMRSAE", "backbone": backbone,
                  "method": cfg.to_dict(), "data_sha256": canonical_hash(bundle.provenance["identity"]),
                  "fit_ids_sha256": hashlib.sha256(np.asarray(bundle.fit_ids, dtype=np.int64).tobytes()).hexdigest(),
                  "data_seed": args.data_seed, "model_seed": args.seed, "batch_size": args.batch_size,
                  "batches_per_epoch": per_epoch, "total_updates": total,
                  "singleton_tail": "merged into previous batch; no discarded samples",
                  "optimizer": {"name": "Adam", "lr": args.lr, "betas": [.9, .999],
                                "eps": args.adam_eps, "weight_decay": args.weight_decay},
                  "bank_refresh": args.bank_refresh, "alpha_power_iterations": args.alpha_power_iterations,
                  "evaluation": args.evaluation, "code": code,
                  "precision": {"dtype": "float32", "amp": False, "tf32": False},
                  "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                              "device": gpu["device"], "device_name": gpu["device_name"]}}
    immutable_json(output / "fit_identity.json", fit_recipe)
    model = MMRSAE(cfg).to(device)
    named = named_parameters(model.sae, model.repair)
    optimizer = torch.optim.Adam(list(named.values()), lr=args.lr, betas=(.9, .999),
                                 eps=args.adam_eps, weight_decay=args.weight_decay)
    masks = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    bank = LandmarkBank(bundle.anchor_ids, tensor_rows(bundle.train, bundle.anchor_ids, device)) if model.needs_bank else None
    checkpoint = output / "last.pt"
    next_step, elapsed, calibration = 0, 0., None
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, expected_identity=fit_recipe)
        model.load_state_dict(saved["model"], strict=True)
        restore_adam(optimizer, named, saved["optimizer"])
        restore_rng(saved["rng"], mask_generator=masks)
        if bank is not None:
            if saved["bank"] is None:
                raise ValueError("Checkpoint is missing its landmark bank")
            bank.load_state_dict(saved["bank"])
        next_step = int(saved["next_step"])
        elapsed = float(saved["fit_seconds"])
        calibration = saved["calibration"]
        del saved
        print(f"Resuming {backbone}/{args.variant} at update {next_step}/{total}", flush=True)
    else:
        calibration = model.initialize_repair(tensor_rows(bundle.train, bundle.fit_ids[:32], device),
                                               iterations=args.alpha_power_iterations)
    if not 0 <= next_step <= total:
        raise ValueError("Checkpoint position is outside the requested training recipe")
    train_log = output / "training.jsonl"
    reconcile_log(train_log, next_step)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def save(position):
        torch.cuda.synchronize(device)
        save_checkpoint(checkpoint, {"format": "mmrsae-imagenet-v1", "identity": fit_recipe,
            "model": model.state_dict(), "optimizer": pack_adam(optimizer, named),
            "rng": rng_state(mask_generator=masks), "bank": None if bank is None else bank.state_dict(),
            "next_step": position, "next_epoch": position // per_epoch,
            "next_batch": position % per_epoch, "fit_seconds": elapsed + time.perf_counter() - started,
            "calibration": calibration})

    if bank is not None and bank.codes is None:
        bank.refresh(model.sae, model.repair, args.k, batch_size=args.encode_batch_size, step=next_step)
    if not checkpoint.exists():
        save(0)
    if next_step == 0 and not (output / "diagnostic_0000000.json").exists():
        diagnostic(args, bundle, model, bank, output, 0, total)
    # Baseline evaluation is optional, never an automatic backbone training job.
    dense = evaluate(args, bundle, model, output, checkpoint, fit_recipe, next_step, dense=True) if args.dense_reference else None
    epoch = next_step // per_epoch
    try:
        while next_step < total:
            for batch_number, ids in training_batches(bundle.fit_ids, args.seed, epoch, args.batch_size,
                                                       offset=next_step % per_epoch):
                if next_step >= total:
                    break
                features = tensor_rows(bundle.train, ids, device)
                optimizer.zero_grad(set_to_none=True)
                loss, stats, views = model.loss(features, bank=bank, row_ids=ids, mask_generator=masks,
                                                step=next_step, total_steps=total)
                loss.backward()
                for name, parameter in named.items():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError(f"Nonfinite gradient in {name}; update rejected")
                optimizer.step()
                row = {"step": next_step, "epoch": epoch, "batch_number": batch_number,
                       "examples": len(ids), "lr": optimizer.param_groups[0]["lr"],
                       "seconds": elapsed + time.perf_counter() - started, **serializable(stats)}
                write_log(train_log, row)
                next_step += 1
                del features, loss, stats, views
                if bank is not None and (next_step % args.bank_refresh == 0 or next_step == total):
                    refresh = bank.refresh(model.sae, model.repair, args.k,
                                           batch_size=args.encode_batch_size, step=next_step)
                    log_wandb_metrics("bank", {"step": next_step, **refresh}, step_metric="step")
                if next_step % args.log_every == 0 or next_step == total:
                    torch.cuda.synchronize(device)
                    row.update({"gpu_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
                                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20})
                    log_wandb_metrics("train", row, step_metric="step")
                    print(f"{backbone} {args.variant} K={args.k} update={next_step}/{total} "
                          f"loss={row['loss_total']:.6f} GPU={row['gpu_allocated_mib']:.0f}MiB", flush=True)
                at_epoch_end = next_step % per_epoch == 0
                if next_step % args.checkpoint_every == 0 or at_epoch_end or next_step == total:
                    save(next_step)
                if args.eval_every and at_epoch_end and (epoch + 1) % args.eval_every == 0 and next_step < total:
                    evaluate(args, bundle, model, output, checkpoint, fit_recipe, next_step, intermediate=True)
            epoch += 1
        # Full diagnostics and retrieval are endpoint work, not repeated each
        # update. They cannot pick an earlier training checkpoint automatically.
        final_diagnostic = diagnostic(args, bundle, model, bank, output, total, total)
        torch.cuda.synchronize(device)
        fit_seconds = elapsed + time.perf_counter() - started
        result = evaluate(args, bundle, model, output, checkpoint, fit_recipe, total)
        summary = {"protocol": PROTOCOL, "family": "MMRSAE", "variant": args.variant, "backbone": backbone,
                   "fit_identity": fit_recipe, "complete": True, "updates": next_step, "fit_seconds": fit_seconds,
                   "fit_time_scope": "training, key refresh, diagnostics, checkpoint I/O and any requested intermediate/parent evaluations; final retrieval reported separately",
                   "calibration": calibration, "retrieval": result, "full_parent": dense,
                   "diagnostics": final_diagnostic, "checkpoint_sha256": sha256_file(checkpoint),
                   "peak_torch_gpu_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                   "source": bundle.provenance, "initialization": "fresh MMRSAE; resumed only from this exact trial",
                   "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                   "precision": "float32; AMP and TF32 disabled", "automatic_model_selection": False}
        atomic_json(output / "summary.json", summary)
        update_wandb_summary({"complete": 1, "updates": next_step, "fit_seconds": fit_seconds,
                              "retrieval": result, "peak_torch_gpu_mib": summary["peak_torch_gpu_mib"]})
        return summary
    except BaseException as error:
        failure = {"step": next_step, "error": repr(error),
                   "traceback": traceback.format_exc(),
                   "resume_from": str(checkpoint) if checkpoint.exists() else None}
        if failure_context is not None:
            failure_context.update(failure)
        atomic_json(output / "failure.json", failure)
        update_wandb_summary({"failed": 1, "last_attempted_update": next_step})
        raise


def trial_settings(args):
    # Logging destinations and output paths do not change learned parameters.
    # They remain in configuration.json but may change when resuming locally.
    ignored = {"output_dir", "run_name", "backbones", "plan", "preflight_only", "prepare_only",
               "log_every", "checkpoint_every", "eval_every", "device"}
    return {key: serializable(value) for key, value in vars(args).items()
            if key not in ignored and not key.startswith("wandb_")}


def main(argv=None):
    args = build_parser().parse_args(argv)
    backbones = validate_args(args)
    if args.plan:
        rows = args.train_size or (1281167 - (args.dev_size if args.evaluation == "dev" else 0))
        batches = batch_count(rows, args.batch_size)
        updates = min(batches * args.epochs, args.max_steps) if args.max_steps else batches * args.epochs
        print(json.dumps({"protocol": PROTOCOL, "family": "MMRSAE", "backbones": backbones,
                          "variant": args.variant, "K": args.k, "fits": len(backbones),
                          "updates_per_fit": updates, "total_updates": updates * len(backbones),
                          "evaluation": args.evaluation, "settings": serializable(vars(args)),
                          "action": "plan only; no downloads, GPU initialization or training",
                          "automatic_ablation": False, "transfer_tasks": False}, indent=2))
        return 0

    import torch
    from mmrsae_gpu import preflight
    # First substantive operation: verify actual GPU retrieval before W&B
    # networking, ImageNet downloads, parent loading, extraction or fitting.
    for backbone in backbones:
        method_config(args, 2048 if backbone == "resnet50" else 512)
    largest_dim = 2048 if "resnet50" in backbones else 512
    expected_gallery = args.train_size or (1281167 - (args.dev_size if args.evaluation == "dev" else 0))
    gpu = preflight(args.device, {"model": {"hidden_dim": args.hidden_dim, "k": args.k, "rank": args.repair_rank},
                   "retrieval": retrieval_config(args), "dense_reference": args.dense_reference},
                    gallery_rows_estimate=expected_gallery, input_dim=largest_dim)
    if args.preflight_only:
        print(json.dumps(serializable(gpu), indent=2))
        return 0
    torch.cuda.set_device(torch.device(gpu["device"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmrsae_data import prepare_imagenet_features
    code = implementation_identity()
    trial = {"settings": trial_settings(args), "code": code}
    name = args.run_name or f"{args.variant}_k{args.k}_{canonical_hash(trial)[:12]}"
    suite = args.output_dir / name
    suite.mkdir(parents=True, exist_ok=True)
    summaries = {}
    with exclusive_lock(suite / "runner.lock"):
        immutable_json(suite / "trial.json", trial)
        atomic_json(suite / "configuration.json", serializable(vars(args)))
        atomic_json(suite / "gpu_preflight.json", serializable(gpu))
        for backbone in backbones:
            output = suite / backbone
            output.mkdir(parents=True, exist_ok=True)
            tracking = copy.copy(args)
            tracking.output_dir = output
            # A custom prefix remains identifiable when two backbones run.
            tracking.wandb_run_name = f"{args.wandb_run_name}/{backbone}" if args.wandb_run_name else f"{name}/{backbone}"
            tracking.wandb_group = args.wandb_group or name
            exit_code = 1
            # Share only this attempt's failure metadata with the fit handler.
            # Reading an existing failure.json could reuse a prior attempt's
            # training position when this attempt fails during preparation.
            failure_context = {}
            try:
                init_wandb(tracking, default_name=tracking.wandb_run_name, default_group=name,
                    tags=("MMRSAE", args.variant, backbone, "imagenet", args.evaluation),
                    extra_config={"family": "MMRSAE", "backbone": backbone, "gpu": gpu,
                                  "method": method_config(args, 2048 if backbone == "resnet50" else 512).to_dict(),
                                  "data_seed": args.data_seed, "automatic_ablation": False})
                log_wandb_metrics("status", {"preflight_passed": 1, "preparing_data": 1})
                data_args = copy.copy(args)
                data_args.progress_callback = lambda row: log_wandb_metrics("data", row)
                bundle = prepare_imagenet_features(data_args, backbone, torch.device(gpu["device"]))
                if "identity" not in bundle.provenance:
                    raise ValueError("Feature provider must expose a stable provenance identity")
                atomic_json(output / "data.json", serializable(bundle.provenance))
                if len(bundle.fit_ids) < 2 or len(bundle.gallery_ids) < 1 or len(bundle.query_ids) < 1:
                    raise ValueError("Fitting/gallery/query partitions must be nonempty")
                update_wandb_summary({"data": {"fitting_rows": len(bundle.fit_ids), "gallery_rows": len(bundle.gallery_ids),
                                              "query_rows": len(bundle.query_ids), "anchors": len(bundle.anchor_ids),
                                              "feature_dimension": bundle.train.shape[1]}})
                if args.prepare_only:
                    summaries[backbone] = {"status": "prepared", "source": bundle.provenance}
                else:
                    summaries[backbone] = fit_and_evaluate(args, backbone, bundle, output, code, gpu,
                                                           failure_context=failure_context)
                atomic_json(suite / "summary.json", {"family": "MMRSAE", "trial": name,
                            "evaluation": args.evaluation, "backbones": summaries,
                            "requested_backbones": backbones, "complete": len(summaries) == len(backbones)})
                exit_code = 0
                del bundle
            except BaseException as error:
                atomic_json(output / "failure.json", {**failure_context, "error": repr(error),
                            "traceback": traceback.format_exc()})
                raise
            finally:
                finish_wandb(exit_code=exit_code)
                torch.cuda.empty_cache()
        atomic_json(suite / "RUN_COMPLETE.json", {"protocol": PROTOCOL, "family": "MMRSAE", "backbones": backbones,
                    "variant": args.variant, "evaluation": args.evaluation, "prepare_only": args.prepare_only})
    print(f"MMRSAE ImageNet {'preparation' if args.prepare_only else 'experiment'} complete: {suite}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

