"""GPU-only ImageNet benchmark: CSR, CSRv2, DLMMSAE and DLMMSAEv2.

One selected variant and target K are fitted per requested backbone, then
the same checkpoint is evaluated at each requested inference K. There is no
automatic ablation matrix, candidate selection, transfer task, or CPU fallback.
All downloads, GPU checks and fitting happen only when this entry point runs.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
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
from mmrsae_methods import BENCHMARK_METHODS, canonical_method
from repair_runtime import (epoch_batches, load_checkpoint, named_parameters, pack_adam,
                            restore_adam, restore_rng, rng_state, save_checkpoint)
from wandb_tracking import (add_wandb_arguments, finish_wandb, init_wandb,
                            log_wandb_metrics, update_wandb_summary)


PROTOCOL = "mmrsae-imagenet-context-distillation-v4"
VARIANTS = BENCHMARK_METHODS
SOURCE_FILES = ("mmrsae_imagenet.py", "mmrsae_methods.py", "mmrsae.py", "mmrsae_dictionary.py", "dlmmsae.py", "mmrsae_probe.py", "mmrsae_data.py", "mmrsae_gpu.py",
                "mpsae_repair.py", "repair_data.py", "repair_runtime.py", "repair_benchmark.py",
                "sparse_gpu_retrieval.py", "retrieval_timing.py", "wandb_tracking.py",
                "csrc/sparse_retrieval.cpp", "csrc/sparse_retrieval.cu")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backbones", default="resnet50,resnet18")
    parser.add_argument("--variant", type=canonical_method, choices=VARIANTS, default="dlmmsae")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--eval-topk", default="auto",
                        help="Comma-separated inference budgets, or auto: 2,4,8,16,... up to parent dimension; training K is always included")
    parser.add_argument("--anneal-start-k", type=int,
                        help="CSRv2/DLMMSAEv2 initial training K; default min(hidden_dim,max(64,4*K))")
    parser.add_argument("--anneal-fraction", type=float, default=.7,
                        help="Fraction of annealed-method updates used for half-cosine training-K decay")
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--dictionary-rank", "--repair-rank", dest="repair_rank", type=int, default=32,
                        help="Rank of the landmark-query adapter; --repair-rank is a compatibility alias")
    parser.add_argument("--dictionary-temperature", type=float, default=.1)
    parser.add_argument("--dictionary-blend", type=float, default=.1,
                        help="Initial trainable landmark-context mixture, strictly between 0 and 1")
    parser.add_argument("--context-weight", type=float,
                        help="DLMMSAE raw-context relational KL weight; default 1 for DLMMSAE, 0 otherwise")
    parser.add_argument("--context-reference-count", type=int, default=1024,
                        help="Fixed training references Q, separate from synthesis landmarks A")
    parser.add_argument("--context-teacher-temperature", type=float, default=.1)
    parser.add_argument("--context-student-temperature", type=float, default=.1)
    parser.add_argument("--context-noise-std", type=float, default=.02,
                        help="DLMMSAE lookup-query Gaussian noise, relative to each input row RMS; 0 disables")
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
    parser.add_argument("--contrastive-weight", type=float, help="CSR NCL gamma; default 0.1 for ResNet-50, 1.0 for ResNet-18")
    parser.add_argument("--contrastive-temperature", type=float, default=.2)
    parser.add_argument("--allow-empty-codes", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep training/evaluating through all-zero codes as the original MPSAE runner does; "
                             "they are counted, and empty queries count as retrieval misses")
    parser.add_argument("--k-aux", type=int, default=512)
    parser.add_argument("--dead-steps", type=int, default=30)
    parser.add_argument("--anchor-count", type=int, default=1024)
    parser.add_argument("--anchor-chunk-size", type=int, default=128)
    parser.add_argument("--bank-refresh", type=int, default=25)

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
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"),
                        help="Local NPZ/JSON pairs; missing files are downloaded here individually")
    parser.add_argument("--embedding-source", choices=("precomputed", "legacy"), default="precomputed",
                        help="Use prepared embeddings for both backbones; legacy retains the original FF2048/RN18 export workflow")
    parser.add_argument("--embeddings-repo", default="quangnguyenopt/NSFW")
    parser.add_argument("--embeddings-repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--embeddings-revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/mmrsae"))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ff2048-train", type=Path)
    parser.add_argument("--ff2048-val", type=Path)
    parser.add_argument("--ff2048-revision", default="main")
    parser.add_argument("--ff2048-train-sha256")
    parser.add_argument("--ff2048-val-sha256")
    parser.add_argument("--rn18-train", type=Path, help="Default <embeddings-dir>/1K_train_rn18.npz")
    parser.add_argument("--rn18-val", type=Path, help="Default <embeddings-dir>/1K_val_rn18.npz")
    # Only the explicit legacy source needs images or a pretrained RN18 model.
    parser.add_argument("--data-root", type=Path, default=Path("data/imagenet1k"))
    parser.add_argument("--data-backend", choices=("hf", "imagefolder"), default="hf")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imagenet-revision", default="main")
    parser.add_argument("--parent-revision", default="main")

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
    parser.add_argument("--linear-probe", action=argparse.BooleanOptionalAction, default=True,
                        help="Fit a separate training-only linear classifier per inference K and a full-parent reference")
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--probe-lr", type=float, default=.1)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
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
                 "epochs", "batch_size", "checkpoint_every", "log_every",
                 "probe_epochs", "probe_batch_size",
                 "encode_batch_size", "query_batch_size", "gallery_chunk_size",
                 "score_buffer_mib", "retrieval_warmup", "retrieval_repeats", "ot_iters", "ot_max_refine_iters"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.feature_batch_size < 1 or args.workers < 0:
        raise ValueError("Legacy extraction requires a positive feature batch size and nonnegative workers")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.embeddings_repo) or not args.embeddings_revision.strip():
        raise ValueError("Embeddings repository must be owner/name, with a nonempty revision")
    for name in ("max_steps", "seed", "data_seed", "train_size", "max_val", "eval_every",
                 "k_aux", "dead_steps", "faiss_temp_memory_mib"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.k > args.hidden_dim or args.ot_microbatch < 2 or args.batch_size < 2:
        raise ValueError("Require K <= hidden dimension and batch/OT group sizes >=2")
    if args.context_reference_count < 3:
        raise ValueError("--context-reference-count must be at least 3 (two alternatives after self exclusion)")
    for backbone in backbones:
        evaluation_budgets(args, 2048 if backbone == "resnet50" else 512)
    if args.anneal_start_k is not None and not args.k <= args.anneal_start_k <= args.hidden_dim:
        raise ValueError("--anneal-start-k must be between K and hidden dimension")
    if not math.isfinite(args.anneal_fraction) or not 0 < args.anneal_fraction <= 1:
        raise ValueError("--anneal-fraction must be in (0,1]")
    if args.mixture_weight not in (None, 0):
        raise ValueError("MMRSAE uses angular OT only; --mixture-weight must be 0")
    if args.evaluation == "dev" and args.dev_size < 1:
        raise ValueError("--dev-size must be positive for development evaluation")
    for name in ("lr", "adam_eps", "teacher_temperature", "student_temperature", "dictionary_temperature", "ot_eta", "ot_tol",
                 "contrastive_temperature", "context_teacher_temperature", "context_student_temperature", "probe_lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.dictionary_blend) or not 0 < args.dictionary_blend < 1:
        raise ValueError("--dictionary-blend must be strictly between 0 and 1")
    for name in ("weight_decay", "relation_weight", "ot_weight", "contrastive_weight", "context_weight", "context_noise_std", "probe_weight_decay"):
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


def evaluation_budgets(args, input_dim):
    """Inference budgets for one trained checkpoint, capped only by SAE width."""
    if args.eval_topk == "auto":
        limit = min(input_dim, args.hidden_dim)
        budgets = [2 ** power for power in range(1, limit.bit_length()) if 2 ** power <= limit]
    else:
        try:
            budgets = [int(item.strip()) for item in args.eval_topk.split(",")]
        except ValueError as error:
            raise ValueError("--eval-topk must be auto or comma-separated positive integers") from error
        if not budgets or len(set(budgets)) != len(budgets):
            raise ValueError("--eval-topk requires unique budgets")
        if any(k < 1 or k > args.hidden_dim for k in budgets):
            raise ValueError("Evaluation K must be between 1 and hidden_dim")
    return sorted(set(budgets + [args.k]))


def retrieval_config(args):
    return {"query_batch_size": args.query_batch_size, "gallery_chunk_size": args.gallery_chunk_size,
            "score_buffer_mib": args.score_buffer_mib, "faiss_temp_memory_mib": args.faiss_temp_memory_mib,
            "gpu_memory_fraction": args.gpu_memory_fraction, "warmup": args.retrieval_warmup,
            "repeats": args.retrieval_repeats, "encode_batch_size": args.encode_batch_size,
            "allow_empty_codes": args.allow_empty_codes}


def method_config(args, input_dim):
    from mmrsae import MMRSAEConfig
    return MMRSAEConfig(input_dim=input_dim, hidden_dim=args.hidden_dim, k=args.k, variant=args.variant,
        rank=args.repair_rank, seed=args.seed, mask=args.mask, relation_weight=args.relation_weight,
        ot_weight=args.ot_weight, mixture_weight=args.mixture_weight,
        contrastive_weight=args.contrastive_weight, contrastive_temperature=args.contrastive_temperature,
        teacher_temperature=args.teacher_temperature, student_temperature=args.student_temperature,
        mass=args.ot_mass, eta=args.ot_eta, max_iters=args.ot_iters, max_refine_iters=args.ot_max_refine_iters,
        tol=args.ot_tol, microbatch=args.ot_microbatch, k_aux=args.k_aux, dead_steps=args.dead_steps,
        clean_probability=args.clean_probability, erase_probability=args.erase_probability,
        anchor_chunk_size=args.anchor_chunk_size, allow_empty_codes=args.allow_empty_codes,
        dictionary_temperature=args.dictionary_temperature, dictionary_blend=args.dictionary_blend,
        context_weight=args.context_weight, context_teacher_temperature=args.context_teacher_temperature,
        context_student_temperature=args.context_student_temperature, context_noise_std=args.context_noise_std,
        anneal_start_k=args.anneal_start_k, anneal_fraction=args.anneal_fraction)


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
    benchmark_identity = {"experiment": identity, "checkpoint_sha256": identity["data_sha256"] if dense else sha256_file(checkpoint_path),
                          "query_source": canonical_hash(bundle.provenance["identity"]),
                          "method": "full_parent" if dense else args.variant, "pool": args.evaluation,
                          "step": 0 if dense else step, "seed": args.seed}
    metadata = {"family": "MMRSAE" if not dense else "full_parent", "variant": args.variant if not dense else None,
                   "evaluation_partition": args.evaluation, "step": step,
                   "training_k": args.k, "ot_cost": "angular" if not dense and model.config.ot_weight else "none",
                   "interpretation": "manual development tuning" if args.evaluation == "dev" else
                   "official validation; any tuning or repeated inspection of these scores must be disclosed"}
    budgets = [args.k] if dense or intermediate else evaluation_budgets(args, model.config.input_dim)
    per_topk = {}
    for k in budgets:
        config = {"protocol": PROTOCOL, "model": {"input_dim": model.config.input_dim,
                  "hidden_dim": args.hidden_dim, "k": k}, "retrieval": retrieval_config(args)}
        target = directory if dense else directory / f"k_{k}"
        result = benchmark(bundle.train, bundle.train_labels, bundle.gallery_ids, bundle.query_ids,
            None if dense else model.sae, None if dense else model.repair, config, target, args.seed,
            identity=benchmark_identity, device=next(model.parameters()).device,
            query_features=bundle.query, query_labels=bundle.query_labels,
            timed=args.timed_retrieval and not intermediate)
        result.update(metadata)
        atomic_json(target / "result.json", result)
        if not dense:
            per_topk[str(k)] = result
            # Persist completed points even if a later budget fails or the job stops.
            atomic_json(directory / "result.json", {**metadata, "per_topk": per_topk,
                "evaluation_topk": budgets, "complete": False,
                "budget_evaluation": "single_checkpoint_multi_k"})
        log_wandb_metrics("parent" if dense else f"retrieval/k_{k}", result, step_metric="step")
    if not dense:
        # Keep endpoint-K fields for existing consumers, and expose every measured K.
        result = {**per_topk[str(args.k)], "per_topk": per_topk, "evaluation_topk": budgets,
                  "complete": True, "budget_evaluation": "single_checkpoint_multi_k"}
    atomic_json(directory / "result.json", result)
    log_wandb_metrics("parent" if dense else "retrieval", result, step_metric="step")
    update_wandb_summary({"parent" if dense else "retrieval": result})
    return result


def refresh_training_bank(args, model, bank, step, total, *, force=False, log=True):
    """Keep clean landmark keys at the budget used by the next loss.

    ``step`` is a completed-update count and may equal ``total`` for endpoint
    diagnostics. A changed training K requires fresh K/2K keys even between
    periodic refreshes; a resumed matching bank retains its saved age.
    """
    if total < 1 or not 0 <= step <= total:
        raise ValueError("Bank refresh position must be in [0, total_updates]")
    if bank is None:
        return None
    k = model.config.training_k(min(step, total - 1), total)
    if not force and bank.codes is not None and bank.k == k:
        return None
    refresh = bank.refresh(model.sae, model.repair, k,
                           batch_size=args.encode_batch_size, step=step)
    record = {"step": step, "training_k": k, **refresh}
    if log:
        log_wandb_metrics("bank", record, step_metric="step")
    return record


def diagnostic(args, bundle, model, bank, output, step, total, *, context_bank=None):
    import torch
    if total < 1 or not 0 <= step <= total:
        raise ValueError("Diagnostic position must be in [0, total_updates]")
    target = output / f"diagnostic_{step:07d}.json"
    ids = bundle.fit_ids[:min(64, len(bundle.fit_ids))]
    before = model.sae.inactive_steps.clone()
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 9001)
    context_generator = torch.Generator(device="cpu").manual_seed(args.seed + 9002)
    # Endpoint reports use completed-update counts, while the loss schedule
    # accepts zero-based update indices. Reuse the final update's budget.
    loss_step = min(step, total - 1)
    refresh_training_bank(args, model, bank, step, total)
    _, stats, views = model.loss(tensor_rows(bundle.train, ids, next(model.parameters()).device),
        bank=bank, row_ids=ids, mask_generator=generator, step=loss_step, total_steps=total,
        context_bank=context_bank, context_generator=context_generator,
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
    if report.get("final_zero_code_fraction", 0.) > 0:
        warnings.append(f"{report['final_zero_code_fraction']:.2%} of diagnostic codes are all zero (support collapse)")
    elif report.get("final_nnz_below_k_fraction", 0.) > 0:
        warnings.append(f"{report['final_nnz_below_k_fraction']:.2%} of diagnostic codes have fewer than K positive coefficients")
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


TRAINING_TIME_SCOPE = (
    "single GPU synchronized wall time: training batch transfer, forward/loss, backward, "
    "optimizer and training key refresh; excludes initialization, diagnostics, logging, "
    "checkpoint I/O, feature preparation and all evaluation"
)


class TrainingTimer:
    """Accumulate only explicit training regions, including completed CUDA work."""

    def __init__(self, device, seconds=0., *, synchronize=None, clock=None):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Training seconds must be finite and nonnegative")
        self.seconds = float(seconds)
        if synchronize is None:
            import torch
            synchronize = lambda: torch.cuda.synchronize(device)
        self.synchronize = synchronize
        self.clock = time.perf_counter if clock is None else clock

    @contextmanager
    def measure(self):
        self.synchronize()
        started = self.clock()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds += self.clock() - started


def context_reference_ids(args, bundle):
    """Deterministic training-only Q; its teacher distribution never sees dev/val."""
    import numpy as np
    anchors = np.asarray(bundle.anchor_ids, dtype=np.int64)
    for name in ("fit_ids", "gallery_ids"):
        if np.intersect1d(anchors, getattr(bundle, name)).size:
            raise ValueError(f"DLMMSAE synthesis landmarks must be disjoint from {name}")
    if args.evaluation == "dev" and np.intersect1d(anchors, bundle.query_ids).size:
        raise ValueError("DLMMSAE synthesis landmarks must be disjoint from development queries")
    candidates = np.asarray(bundle.fit_ids, dtype=np.int64)
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("DLMMSAE fitting IDs must be unique")
    count = args.context_reference_count
    if count < 3 or count > len(candidates):
        raise ValueError("DLMMSAE context reference count must be in [3, number of fitting rows]")
    rng = np.random.default_rng(np.random.SeedSequence([args.data_seed, 9131]))
    return np.sort(rng.choice(candidates, size=count, replace=False))


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
    reference_ids = context_reference_ids(args, bundle) if cfg.context_weight else None
    context_identity = None if reference_ids is None else {
        "source": "fixed_fitting_rows_disjoint_from_synthesis_landmarks",
        "count": len(reference_ids), "seed_stream": [args.data_seed, 9131],
        "ids_sha256": hashlib.sha256(reference_ids.astype("<i8").tobytes()).hexdigest(),
        "self_match_policy": "exclude each query's own ID from aligned teacher/student support",
        "keys": "fixed_original_embeddings_no_momentum_encoder",
    }
    fit_recipe = {"protocol": PROTOCOL, "family": "MMRSAE", "backbone": backbone,
                  "method": cfg.to_dict(), "data_sha256": canonical_hash(bundle.provenance["identity"]),
                  "fit_ids_sha256": hashlib.sha256(np.asarray(bundle.fit_ids, dtype=np.int64).tobytes()).hexdigest(),
                  "data_seed": args.data_seed, "model_seed": args.seed, "batch_size": args.batch_size,
                  "batches_per_epoch": per_epoch, "total_updates": total,
                  "singleton_tail": "merged into previous batch; no discarded samples",
                  "optimizer": {"name": "Adam", "lr": args.lr, "betas": [.9, .999],
                                "eps": args.adam_eps, "weight_decay": args.weight_decay},
                  "bank_refresh": args.bank_refresh,
                  "context_reference": context_identity,
                  "bank_budget_policy": "refresh before loss whenever training K changes",
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
    context_generator = torch.Generator(device="cpu").manual_seed(args.seed + 202)
    bank = (LandmarkBank(bundle.anchor_ids, tensor_rows(bundle.train, bundle.anchor_ids, device),
                         allow_empty_codes=cfg.allow_empty_codes) if model.needs_bank else None)
    context_bank = None
    if model.needs_context:
        from dlmmsae import ContextReferenceBank
        context_bank = ContextReferenceBank(reference_ids,
            tensor_rows(bundle.train, reference_ids, device), anchor_ids=bundle.anchor_ids)
    checkpoint = output / "last.pt"
    next_step, elapsed, dictionary_initialization = 0, 0., None
    training_timer = TrainingTimer(device)
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, expected_identity=fit_recipe)
        model.load_state_dict(saved["model"], strict=True)
        restore_adam(optimizer, named, saved["optimizer"])
        restore_rng(saved["rng"], mask_generator=masks)
        if context_bank is not None:
            if saved.get("context_bank") is None or saved.get("context_rng") is None:
                raise ValueError("Checkpoint is missing DLMMSAE context references or perturbation RNG")
            context_bank.load_state_dict(saved["context_bank"])
            context_generator.set_state(saved["context_rng"].cpu())
        if bank is not None:
            if saved["bank"] is None:
                raise ValueError("Checkpoint is missing its landmark bank")
            bank.load_state_dict(saved["bank"])
        next_step = int(saved["next_step"])
        elapsed = float(saved["fit_seconds"])
        training_timer = TrainingTimer(device, float(saved["training_seconds"]))
        dictionary_initialization = saved["dictionary_initialization"]
        del saved
        print(f"Resuming {backbone}/{args.variant} at update {next_step}/{total}", flush=True)
    else:
        dictionary_initialization = model.initialize_dictionary(
            tensor_rows(bundle.train, bundle.anchor_ids, device), anchor_ids=bundle.anchor_ids)
    if not 0 <= next_step <= total:
        raise ValueError("Checkpoint position is outside the requested training recipe")
    train_log = output / "training.jsonl"
    reconcile_log(train_log, next_step)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def save(position):
        torch.cuda.synchronize(device)
        save_checkpoint(checkpoint, {"format": "mmrsae-imagenet-v4", "identity": fit_recipe,
            "model": model.state_dict(), "optimizer": pack_adam(optimizer, named),
            "rng": rng_state(mask_generator=masks), "bank": None if bank is None else bank.state_dict(),
            "context_bank": None if context_bank is None else context_bank.state_dict(),
            "context_rng": None if context_bank is None else context_generator.get_state(),
            "training_seconds": training_timer.seconds, "training_time_scope": TRAINING_TIME_SCOPE,
            "next_step": position, "next_epoch": position // per_epoch,
            "next_batch": position % per_epoch, "fit_seconds": elapsed + time.perf_counter() - started,
            "dictionary_initialization": dictionary_initialization})

    with training_timer.measure():
        refresh_record = refresh_training_bank(args, model, bank, next_step, total, log=False)
    if refresh_record is not None:
        log_wandb_metrics("bank", refresh_record, step_metric="step")
    if not checkpoint.exists():
        save(0)
    if next_step == 0 and not (output / "diagnostic_0000000.json").exists():
        diagnostic(args, bundle, model, bank, output, 0, total, context_bank=context_bank)
    # Baseline evaluation is optional, never an automatic backbone training job.
    dense = evaluate(args, bundle, model, output, checkpoint, fit_recipe, next_step, dense=True) if args.dense_reference else None
    epoch = next_step // per_epoch
    try:
        while next_step < total:
            for batch_number, ids in training_batches(bundle.fit_ids, args.seed, epoch, args.batch_size,
                                                       offset=next_step % per_epoch):
                if next_step >= total:
                    break
                refresh_records = []
                with training_timer.measure():
                    refresh_records.append(refresh_training_bank(args, model, bank, next_step, total, log=False))
                    features = tensor_rows(bundle.train, ids, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, stats, views = model.loss(features, bank=bank, row_ids=ids, mask_generator=masks,
                        context_bank=context_bank, context_generator=context_generator,
                        step=next_step, total_steps=total)
                    loss.backward()
                    for name, parameter in named.items():
                        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                            raise FloatingPointError(f"Nonfinite gradient in {name}; update rejected")
                    optimizer.step()
                    if bank is not None and ((next_step + 1) % args.bank_refresh == 0 or next_step + 1 == total):
                        refresh_records.append(refresh_training_bank(args, model, bank, next_step + 1, total, force=True, log=False))
                for refresh_record in refresh_records:
                    if refresh_record is not None:
                        log_wandb_metrics("bank", refresh_record, step_metric="step")
                row = {"step": next_step, "epoch": epoch, "batch_number": batch_number,
                       "examples": len(ids), "lr": optimizer.param_groups[0]["lr"],
                       "training_seconds": training_timer.seconds,
                       "seconds": elapsed + time.perf_counter() - started, **serializable(stats)}
                write_log(train_log, row)
                next_step += 1
                del features, loss, stats, views
                if next_step % args.log_every == 0 or next_step == total:
                    torch.cuda.synchronize(device)
                    row.update({"gpu_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
                                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20})
                    log_wandb_metrics("train", row, step_metric="step")
                    print(f"{backbone} {args.variant} K={row['training_k']} target_K={args.k} update={next_step}/{total} "
                          f"loss={row['loss_total']:.6f} nnz_min={row['final_nnz_min']} "
                          f"zero_codes={row['zero_final_codes']} GPU={row['gpu_allocated_mib']:.0f}MiB", flush=True)
                at_epoch_end = next_step % per_epoch == 0
                if next_step % args.checkpoint_every == 0 or at_epoch_end or next_step == total:
                    save(next_step)
                if args.eval_every and at_epoch_end and (epoch + 1) % args.eval_every == 0 and next_step < total:
                    evaluate(args, bundle, model, output, checkpoint, fit_recipe, next_step, intermediate=True)
            epoch += 1
        # Full diagnostics and retrieval are endpoint work, not repeated each
        # update. They cannot pick an earlier training checkpoint automatically.
        final_diagnostic = diagnostic(args, bundle, model, bank, output, total, total, context_bank=context_bank)
        torch.cuda.synchronize(device)
        fit_seconds = elapsed + time.perf_counter() - started
        result = evaluate(args, bundle, model, output, checkpoint, fit_recipe, total)
        classification = None
        if args.linear_probe:
            from mmrsae_probe import evaluate_linear_probes
            classification = evaluate_linear_probes(args, bundle, model, output,
                checkpoint_identity=sha256_file(checkpoint))
        summary = {"protocol": PROTOCOL, "family": "MMRSAE", "variant": args.variant, "backbone": backbone,
                   "fit_identity": fit_recipe, "complete": True, "updates": next_step, "fit_seconds": fit_seconds,
                   "fit_time_scope": "training, key refresh, diagnostics, checkpoint I/O and any requested intermediate/parent evaluations; final retrieval reported separately",
                   "training_seconds": training_timer.seconds,
                   "training_gpu_hours": training_timer.seconds / 3600.,
                   "training_time_scope": TRAINING_TIME_SCOPE,
                   "training_device_name": gpu["device_name"], "training_gpu_count": 1,
                   "context_reference": context_identity,
                   "dictionary_initialization": dictionary_initialization,
                   "retrieval": result, "full_parent": dense,
                   "classification": classification,
                   "evaluation_topk": result["evaluation_topk"],
                   "budget_evaluation": "single_checkpoint_multi_k", "ot_cost": "angular" if cfg.ot_weight else "none",
                   "diagnostics": final_diagnostic, "checkpoint_sha256": sha256_file(checkpoint),
                   "peak_torch_gpu_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                   "source": bundle.provenance, "initialization": "fresh MMRSAE; resumed only from this exact trial",
                   "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                   "precision": "float32; AMP and TF32 disabled", "automatic_model_selection": False}
        atomic_json(output / "summary.json", summary)
        update_wandb_summary({"complete": 1, "updates": next_step, "fit_seconds": fit_seconds,
                              "training_gpu_hours": summary["training_gpu_hours"],
                              "classification": classification,
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
        rows = args.train_size or (1281167 - args.anchor_count - (args.dev_size if args.evaluation == "dev" else 0))
        batches = batch_count(rows, args.batch_size)
        updates = min(batches * args.epochs, args.max_steps) if args.max_steps else batches * args.epochs
        print(json.dumps({"protocol": PROTOCOL, "family": "MMRSAE", "backbones": backbones,
                          "variant": args.variant, "K": args.k, "fits": len(backbones),
                          "evaluation_topk": {name: evaluation_budgets(args, 2048 if name == "resnet50" else 512) for name in backbones},
                          "budget_evaluation": "single_checkpoint_multi_k",
                          "ot_cost": "none" if args.variant in ("csr", "csrv2", "mmrsae-rel") or args.ot_weight == 0 else "angular",
                          "context_distillation": args.variant.startswith("dlmmsae") and args.context_weight != 0,
                          "training_time_scope": TRAINING_TIME_SCOPE,
                          "linear_probe": {"enabled": args.linear_probe, "epochs": args.probe_epochs,
                              "one_classifier_per_inference_k": True, "full_parent_classifier": True,
                              "cost_scope": "separate from SAE training GPU hours"},
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
    expected_gallery = args.train_size or (1281167 - args.anchor_count - (args.dev_size if args.evaluation == "dev" else 0))
    max_eval_k = max(max(evaluation_budgets(args, 2048 if name == "resnet50" else 512)) for name in backbones)
    gpu = preflight(args.device, {"model": {"hidden_dim": args.hidden_dim, "k": max_eval_k,
                    "rank": args.repair_rank, "anchor_count": args.anchor_count,
                    "context_reference_count": args.context_reference_count if args.variant.startswith("dlmmsae") and args.context_weight != 0 else 0,
                    "landmark_dictionary": args.variant not in ("mpsae", "csr", "csrv2")},
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

