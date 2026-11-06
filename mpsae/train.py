"""One resumable fit: a single (backbone, method, K, seed) cell.

Optimisation follows the released CSR trainer: ``Adam`` over every parameter
(bias included, so the weight decay is coupled L2 and not AdamW's decoupled
form), ``eps = 6.25e-10``, a constant learning rate -- the release constructs
``StepLR(step_size=10, gamma=1)``, which never changes the rate -- no warmup,
no gradient clipping, and float32 throughout with TF32 disabled.

A fit is identified by a digest over the method description, the preset, the
embedding archive digests and the seed. Resuming checks that digest before
touching a checkpoint, so a changed setting can never be silently continued
from an incompatible state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from . import methods
from .config import BACKBONES, Backbone, Cell, TrainPreset
from .features import Bundle, BatchSource
from .io import atomic_json, canonical_hash, read_json, truncate_jsonl
from .logs import MetricWriter, Progress, format_duration, get_logger
from .model import build_sae, parameter_bytes

CHECKPOINT_FORMAT = "mpsae-fit-v2"


@dataclass
class FitSettings:
    """Knobs that do not change the objective."""

    device: str = "cuda:0"
    residency: str = "auto"
    checkpoint_every: int = 100
    log_every: int = 10
    max_steps: int = 0          # 0 = run the full schedule; >0 caps it (smoke runs)
    wandb_project: str | None = None
    compile_model: bool = False


def total_updates(bundle: Bundle, preset: TrainPreset, *, max_steps: int = 0) -> tuple[int, int]:
    """``(batches_per_epoch, total_steps)`` for the released epoch definition."""
    per_epoch = -(-bundle.train.rows // preset.batch_size)
    total = per_epoch * preset.epochs
    if max_steps:
        total = min(total, max_steps)
    return per_epoch, total


def fit_identity(cell: Cell, bundle: Bundle, preset: TrainPreset, *,
                 total_steps: int, per_epoch: int, code_version: str) -> dict[str, Any]:
    """Everything that must match for a checkpoint to be resumable."""
    return {
        "format": CHECKPOINT_FORMAT,
        "code_version": code_version,
        "backbone": cell.backbone,
        "method": cell.method,
        "target_k": cell.k,
        "seed": cell.seed,
        "objective": methods.describe(cell.method, cell.k, preset),
        "optimizer": {"name": "Adam", "lr": preset.lr, "betas": list(preset.adam_betas),
                      "eps": preset.adam_eps, "weight_decay": preset.weight_decay,
                      "schedule": "constant", "grad_clip": None,
                      "decay_applies_to": "all parameters (coupled L2, as released)"},
        "schedule": {"epochs": preset.epochs, "batch_size": preset.batch_size,
                     "batches_per_epoch": per_epoch, "total_steps": total_steps,
                     "drop_last": False},
        "precision": {"params": "float32", "autocast": False, "tf32": False},
        "data": bundle.identity,
    }


def _configure_backend() -> None:
    import torch

    # TF32 would silently change every matmul in the objective and the search.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _rng_state() -> dict[str, Any]:
    import torch

    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    import torch

    torch.set_rng_state(state["cpu"].cpu() if hasattr(state["cpu"], "cpu") else state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in state["cuda"]])


def fit(cell: Cell, bundle: Bundle, out_dir: str | Path, settings: FitSettings, *,
        code_version: str, spec: Backbone | None = None) -> dict[str, Any]:
    """Train one cell to completion, resuming any partial checkpoint.

    Returns a summary dict and leaves ``checkpoint.pt``, ``config.json``,
    ``train.jsonl`` and ``summary.json`` in ``out_dir``.
    """
    import torch

    spec = spec or BACKBONES[cell.backbone]
    preset = spec.preset
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"fit {cell.slug}")
    _configure_backend()
    device = torch.device(settings.device)

    per_epoch, total = total_updates(bundle, preset, max_steps=settings.max_steps)
    identity = fit_identity(cell, bundle, preset, total_steps=total,
                            per_epoch=per_epoch, code_version=code_version)
    signature = canonical_hash(identity)
    summary_path = out_dir / "summary.json"
    finished = read_json(summary_path)
    if finished and finished.get("identity") == signature and finished.get("complete"):
        log.info("already trained (%d steps, %s)", finished["steps"],
                 format_duration(finished["train_seconds"]))
        return finished
    atomic_json(out_dir / "config.json", {"identity": identity, "signature": signature,
                                          "cell": cell.__dict__, "code_version": code_version,
                                          "backbone": cell.backbone, "method": cell.method,
                                          "k": cell.k, "seed": cell.seed,
                                          "total_steps": total, "log_every": settings.log_every})

    model = build_sae(spec.dim, preset.hidden_dim, seed=cell.seed,
                      dead_threshold=preset.dead_threshold, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=preset.lr, betas=preset.adam_betas,
                                 eps=preset.adam_eps, weight_decay=preset.weight_decay)
    checkpoint_path = out_dir / "checkpoint.pt"
    next_step, elapsed = 0, 0.0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError(
                f"{checkpoint_path} belongs to a different recipe ({saved.get('signature')} "
                f"!= {signature}). Move it aside rather than resuming across a settings change.")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"])
        next_step, elapsed = int(saved["next_step"]), float(saved["train_seconds"])
        log.info("resuming at step %d/%d", next_step, total)
    else:
        # Seed the global streams once, as the release does before training.
        torch.manual_seed(cell.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cell.seed)

    truncate_jsonl(out_dir / "train.jsonl", next_step // max(1, settings.log_every))
    writer = MetricWriter(out_dir / "train.jsonl", stage=cell.slug)
    source = BatchSource(bundle.train.features, bundle.train.labels,
                         batch_size=preset.batch_size, seed=cell.seed, device=device,
                         residency=settings.residency,
                         with_labels=methods.requires_labels(cell.method))
    log.info("%s | d=%d h=%d | %d steps (%d/epoch x %d) | batch %d | %s params",
             cell.slug, spec.dim, preset.hidden_dim, total, per_epoch, preset.epochs,
             preset.batch_size, f"{parameter_bytes(model) // 4:,}")
    if cell.method in methods.ANNEALED:
        start = methods.initial_topk(cell.k, preset.hidden_dim)
        log.info("k-annealing: %d -> %d linearly over the first %d steps",
                 start, cell.k, int(total * preset.k_decay_ratio))

    progress = Progress(total, stage=cell.slug, what="steps", done=next_step)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    step = next_step
    last_metrics: dict[str, Any] = {}

    def save(position: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        payload = {"format": CHECKPOINT_FORMAT, "signature": signature, "identity": identity,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "rng": _rng_state(), "next_step": position,
                   "train_seconds": elapsed + time.perf_counter() - started}
        temporary = checkpoint_path.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)

    try:
        while step < total:
            epoch, within = divmod(step, per_epoch)
            for index, batch, labels in source.epoch(epoch, skip_batches=within):
                if step >= total:
                    break
                output = methods.compute_step(
                    cell.method, model, batch, k=cell.k, preset=preset,
                    step=step, total_steps=total, labels=labels)
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                optimizer.step()
                step += 1
                last_metrics = output.metrics
                if step % settings.log_every == 0 or step == total:
                    # ``elapsed`` is cumulative on purpose: logging is strided,
                    # so a per-record duration could not be summed into a total.
                    record = {**output.metrics, "epoch": epoch, "batch": index,
                              "lr": preset.lr, "rows": int(len(batch)),
                              "elapsed": elapsed + time.perf_counter() - started}
                    writer.log(record, step=step)
                    progress.advance(settings.log_every, extra=(
                        f"loss {record['loss_total']:.4f} k={record['training_k']} "
                        f"nnz {record['nnz_mean']:.1f} dead {record['dead_fraction']:.2f}"))
                if step % settings.checkpoint_every == 0:
                    save(step)
            if step < total and within == 0 and index + 1 < per_epoch:
                raise RuntimeError("epoch ended before its batches were exhausted")
        save(step)
    finally:
        writer.close()
        source.release()

    train_seconds = elapsed + time.perf_counter() - started
    summary = {
        "identity": identity, "signature": signature, "complete": True,
        "cell": cell.__dict__, "steps": step, "train_seconds": train_seconds,
        "gpu_hours": train_seconds / 3600.0,
        "steps_per_second": step / max(1e-9, train_seconds),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if torch.cuda.is_available() else None,
        "residency": source.residency,
        "final_metrics": last_metrics,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    atomic_json(summary_path, summary)
    log.info("trained %d steps in %s (%.1f steps/s)", step, format_duration(train_seconds),
             summary["steps_per_second"])
    return summary


def load_fitted(cell: Cell, out_dir: str | Path, spec: Backbone, *, device) -> Any:
    """Rebuild the autoencoder from a finished fit, ready for evaluation."""
    import torch

    out_dir = Path(out_dir)
    summary = read_json(out_dir / "summary.json")
    if not summary or not summary.get("complete"):
        raise FileNotFoundError(f"{out_dir} has no completed fit")
    saved = torch.load(out_dir / "checkpoint.pt", map_location=device, weights_only=False)
    if saved.get("signature") != summary["signature"]:
        raise RuntimeError(f"{out_dir}: checkpoint and summary disagree on the recipe")
    model = build_sae(spec.dim, spec.preset.hidden_dim, seed=cell.seed,
                      dead_threshold=spec.preset.dead_threshold, device=device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model
