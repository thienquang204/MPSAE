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
from copy import deepcopy
from pathlib import Path
import re
import time
from typing import Any

from . import methods
from .config import (BACKBONES, METHOD_EPOCHS, METHOD_VARIANTS, Backbone, Cell, TrainPreset,
                     training_preset, validate_checkpoint_ot_weight)
from .features import Bundle, BatchSource
from .io import atomic_json, canonical_hash, exclusive_lock, read_json, read_jsonl, truncate_jsonl
from .logs import MetricWriter, Progress, format_duration, get_logger
from .model import build_sae, parameter_bytes
from .timing_config import runtime_fingerprint

CHECKPOINT_FORMAT = "mpsae-fit-v2"
RUNTIME_IMPLEMENTATION = "packed-metrics-reused-topk-batched-ot-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


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
    #: Retrain even when a completed fit with the same recipe exists. Note this
    #: discards the checkpoint rather than resuming from it: ``--force train``
    #: means "do it again", not "do the rest of it".
    force: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint_every", "log_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        if self.residency not in ("auto", "cuda", "cpu", "mmap"):
            raise ValueError("residency must be auto, cuda, cpu or mmap")


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
    preset = training_preset(cell.method, preset, k=cell.k)
    identity = {
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
    # Historical v2 identities already imply these released defaults. Preserve
    # their signatures while distinguishing any explicitly changed objective
    # coefficient or refinement budget, previously omitted from the identity.
    if preset.nested_coef != 0.125:
        identity["objective"]["nested_coef"] = preset.nested_coef
    if cell.method in methods.OT_METHODS and preset.ot_max_refine_iters != 2000:
        identity["objective"]["ot"]["max_refine_iters"] = preset.ot_max_refine_iters
    return identity


def compatible_fit_identity(saved: dict[str, Any], requested: dict[str, Any]) -> bool:
    """Allow only provenance enrichment caused by the old sidecar-path bug.

    Earlier ``load_bundle`` calls looked for ``archive.json`` instead of the
    writer's ``archive.npz.json``. Archive digests and row counts were still
    measured, but recipe/source became None. Reading the sidecar correctly
    must not invalidate identical training data. This is not a settings-change
    escape hatch: SHA256, row count, code version, objective, precision, seed,
    optimizer and the complete remaining identity must all agree.
    """
    if canonical_hash(saved) == canonical_hash(requested):
        return True
    if not isinstance(saved.get("data"), dict) or not isinstance(requested.get("data"), dict):
        return False
    comparable = deepcopy(requested)
    for split in ("train", "validation"):
        old = saved["data"].get(split)
        new = comparable["data"].get(split)
        if not isinstance(old, dict) or not isinstance(new, dict):
            continue
        for key in ("recipe", "source"):
            if old.get(key) == new.get(key) and (key in old) == (key in new):
                continue
            if old.get(key) is not None:
                return False
            digest = old.get("sha256")
            rows = old.get("rows")
            if (not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                    or new.get("sha256") != digest
                    or isinstance(rows, bool) or not isinstance(rows, int) or rows < 1
                    or new.get("rows") != rows):
                return False
            if key in old:
                new[key] = None
            else:
                new.pop(key, None)
    return canonical_hash(saved) == canonical_hash(comparable)


def can_extend_mpsaev2(saved: dict[str, Any], requested: dict[str, Any]) -> bool:
    """Only extend the identical fixed-K, constant-LR recipe from 10 to 15 epochs."""
    if saved.get("method") != "mpsaev2" or requested.get("method") != "mpsaev2":
        return False
    old, new = saved.get("schedule", {}), requested.get("schedule", {})
    per_epoch = old.get("batches_per_epoch")
    if (old.get("epochs") != 10 or new.get("epochs") != 15
            or not isinstance(per_epoch, int) or isinstance(per_epoch, bool) or per_epoch < 1
            or old.get("total_steps") != 10 * per_epoch
            or new.get("total_steps") != 15 * per_epoch
            or saved.get("objective", {}).get("annealing") is not None
            or saved.get("optimizer", {}).get("schedule") != "constant"):
        return False
    extended = deepcopy(saved)
    extended["schedule"].update(epochs=15, total_steps=15 * per_epoch)
    return compatible_fit_identity(extended, requested)


def _existing_identity(record: dict[str, Any], path: Path,
                       requested: dict[str, Any], *, allow_extension: bool = False
                       ) -> tuple[dict[str, Any], str]:
    """Verify an existing artifact before any output is rewritten."""
    identity, signature = record.get("identity"), record.get("signature")
    if (not isinstance(identity, dict) or not isinstance(signature, str)
            or canonical_hash(identity) != signature):
        raise RuntimeError(f"{path} has an invalid training identity/signature")
    if not (compatible_fit_identity(identity, requested)
            or allow_extension and can_extend_mpsaev2(identity, requested)):
        raise RuntimeError(
            f"{path} belongs to a different recipe ({signature} != {canonical_hash(requested)}). "
            "Move it aside rather than resuming across a settings change.")
    return identity, signature


def _configure_backend() -> None:
    import torch

    # TF32 would silently change every matmul in the objective and the search.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _rng_state(device) -> dict[str, Any]:
    import torch

    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any], device) -> None:
    import torch

    torch.set_rng_state(state["cpu"].cpu() if hasattr(state["cpu"], "cpu") else state["cpu"])
    if "cuda" in state and device.type == "cuda":
        torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in state["cuda"]])


def fit(cell: Cell, bundle: Bundle, out_dir: str | Path, settings: FitSettings, *,
        code_version: str, spec: Backbone | None = None) -> dict[str, Any]:
    """Fit one cell, refusing concurrent writers even from different campaigns."""
    out_dir = Path(out_dir)
    with exclusive_lock(out_dir / "fit.lock", timeout=0):
        return _fit(cell, bundle, out_dir, settings, code_version=code_version, spec=spec)


def _fit(cell: Cell, bundle: Bundle, out_dir: str | Path, settings: FitSettings, *,
         code_version: str, spec: Backbone | None = None) -> dict[str, Any]:
    """Train one cell to completion, resuming any partial checkpoint.

    Returns a summary dict and leaves ``checkpoint.pt``, ``config.json``,
    ``train.jsonl`` and ``summary.json`` in ``out_dir``.
    """
    import torch

    spec = spec or BACKBONES[cell.backbone]
    preset = training_preset(cell.method, spec.preset, k=cell.k)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"fit {cell.slug}")
    _configure_backend()
    device = torch.device(settings.device)

    per_epoch, total = total_updates(bundle, preset, max_steps=settings.max_steps)
    if per_epoch < 1 or total < 1:
        raise ValueError("cannot fit an empty training split")
    identity = fit_identity(cell, bundle, preset, total_steps=total,
                            per_epoch=per_epoch, code_version=code_version)
    signature = canonical_hash(identity)
    summary_path = out_dir / "summary.json"
    checkpoint_path = out_dir / "checkpoint.pt"
    config_path = out_dir / "config.json"
    finished = read_json(summary_path)
    existing_config = read_json(config_path)
    # Check before writing config.json, deleting anything, or allocating the
    # fitting matrix. Rejected resumes must leave their metadata untouched.
    if existing_config is not None:
        _existing_identity(existing_config, config_path, identity, allow_extension=True)
    saved = None
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_identity, saved_signature = _existing_identity(saved, checkpoint_path, identity,
                                                            allow_extension=True)
        if (existing_config is not None and existing_config.get("signature") != saved_signature
                and not (can_extend_mpsaev2(saved_identity, identity)
                         or can_extend_mpsaev2(existing_config["identity"], identity))):
            raise RuntimeError(f"{out_dir}: config and checkpoint disagree on the recipe")
        position = saved.get("next_step")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position <= total:
            raise RuntimeError(f"{checkpoint_path}: invalid next_step {position!r}")
        if can_extend_mpsaev2(saved_identity, identity) and position > saved_identity["schedule"]["total_steps"]:
            raise RuntimeError(f"{checkpoint_path}: next_step exceeds the saved 10-epoch schedule")
    if finished is not None:
        _, finished_signature = _existing_identity(finished, summary_path, identity,
                                                    allow_extension=True)
        if (saved and saved["signature"] != finished_signature
                and not (can_extend_mpsaev2(saved_identity, identity)
                         or can_extend_mpsaev2(finished["identity"], identity))):
            raise RuntimeError(f"{out_dir}: checkpoint and summary disagree on the recipe")
    extending = any(can_extend_mpsaev2(record["identity"], identity)
                    for record in (saved, existing_config, finished) if record is not None)
    if finished and finished.get("complete"):
        if saved is None:
            raise FileNotFoundError(f"{out_dir}: completed summary has no checkpoint")
        finished_total = finished["identity"]["schedule"]["total_steps"] if extending else total
        if finished.get("steps") != finished_total or saved["next_step"] != finished_total:
            raise RuntimeError(f"{out_dir}: completed summary and checkpoint disagree on step count")
        if not settings.force and not extending:
            log.info("already trained (%d steps, %s)", finished["steps"],
                     format_duration(finished["train_seconds"]))
            return finished
        if settings.force:
            log.warning("retraining from scratch (--force); discarding the completed fit")
            for stale in (summary_path, checkpoint_path, out_dir / "train.jsonl"):
                stale.unlink(missing_ok=True)
            saved, existing_config = None, None
    if extending:
        # Keep weights, Adam, RNG and activity counters. Only the requested
        # stopping point and signature change. Mixed old/new metadata after an
        # interruption is accepted above only for this exact duration change.
        if finished is not None:
            atomic_json(summary_path, {**finished, "complete": False})
        if existing_config is not None:
            existing_config = {**existing_config, "identity": identity,
                               "signature": signature, "total_steps": total}
            atomic_json(config_path, existing_config)
        log.info("extending identical MPSAEv2 training from 10 to 15 epochs")
    elif saved is not None:
        # Preserve the original identity, including legacy missing provenance,
        # so resumed checkpoints and their existing evaluation caches keep the
        # same fit signature. The identical archive bytes were checked above.
        identity, signature = saved_identity, saved_signature
    elif existing_config is not None:
        identity, signature = existing_config["identity"], existing_config["signature"]
    if existing_config is None:
        atomic_json(config_path, {"identity": identity, "signature": signature,
                                          "cell": cell.__dict__, "code_version": code_version,
                                          "backbone": cell.backbone, "method": cell.method,
                                          "k": cell.k, "seed": cell.seed,
                                          # Provenance is outside the numerical identity.
                                          "backbone_weights": bundle.train.manifest.get("weights"),
                                          "backbone_preprocessing": {
                                              key: bundle.train.manifest.get(key)
                                              for key in ("input_size", "normalize", "preprocessing")},
                                          "total_steps": total, "log_every": settings.log_every})

    model = build_sae(spec.dim, preset.hidden_dim, seed=cell.seed,
                      dead_threshold=preset.dead_threshold, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=preset.lr, betas=preset.adam_betas,
                                 eps=preset.adam_eps, weight_decay=preset.weight_decay)
    next_step, elapsed = 0, 0.0
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"], device)
        next_step, elapsed = int(saved["next_step"]), float(saved["train_seconds"])
        log.info("resuming at step %d/%d", next_step, total)
    else:
        # Seed the global streams once, as the release does before training.
        torch.manual_seed(cell.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cell.seed)

    # Keep actual committed step positions, including a final off-stride log.
    # Counting next_step//log_every would discard that record on a completed
    # checkpoint recovered after a crash just before summary.json was written.
    prior_records = read_jsonl(out_dir / "train.jsonl")
    keep = sum(isinstance(record.get("step"), int) and record["step"] <= next_step
               for record in prior_records)
    truncate_jsonl(out_dir / "train.jsonl", keep)
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    step = next_step
    last_metrics: dict[str, Any] = dict((saved or {}).get("last_metrics") or {})
    if not last_metrics and keep:
        last_metrics = {key: value for key, value in prior_records[keep - 1].items()
                        if key not in {"wall", "step", "epoch", "batch", "lr", "rows", "elapsed"}}

    # Runtime provenance stays outside the scientific fit identity. Completed
    # fits are reused above; continuing an old checkpoint records the exact
    # step where the equivalent implementation changed.
    runtime_history = list((saved or {}).get("runtime_history", []))
    runtime = {"implementation": RUNTIME_IMPLEMENTATION, "torch": str(torch.__version__),
               "cuda": torch.version.cuda, "device": str(device),
               "hardware": runtime_fingerprint(device),
               "residency": source.residency, "log_every": settings.log_every,
               "checkpoint_every": settings.checkpoint_every}
    if not runtime_history or runtime_history[-1].get("runtime") != runtime:
        runtime_history.append({"from_step": next_step, "runtime": runtime})

    def save(position: int) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        payload = {"format": CHECKPOINT_FORMAT, "signature": signature, "identity": identity,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "rng": _rng_state(device), "next_step": position,
                   "last_metrics": last_metrics,
                   "runtime_history": runtime_history,
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
                optimizer.zero_grad(set_to_none=True)
                upcoming = step + 1
                collect_metrics = (upcoming % settings.log_every == 0
                                   or upcoming % settings.checkpoint_every == 0
                                   or upcoming == total)
                output = methods.compute_step(
                    cell.method, model, batch, k=cell.k, preset=preset,
                    step=step, total_steps=total, labels=labels,
                    collect_metrics=collect_metrics)
                output.loss.backward()
                optimizer.step()
                step += 1
                if collect_metrics:
                    last_metrics = output.metrics
                if step % settings.log_every == 0 or step == total:
                    # ``elapsed`` is cumulative on purpose: logging is strided,
                    # so a per-record duration could not be summed into a total.
                    record = {**output.metrics, "epoch": epoch, "batch": index,
                              "lr": preset.lr, "rows": int(len(batch)),
                              "elapsed": elapsed + time.perf_counter() - started}
                    writer.log(record, step=step)
                    progress.advance(step - progress.done, extra=(
                        f"loss {record['loss_total']:.4f} k={record['training_k']} "
                        f"nnz {record['nnz_mean']:.1f} dead {record['dead_fraction']:.2f}"))
                if step % settings.checkpoint_every == 0:
                    save(step)
            if step < total and within == 0 and index + 1 < per_epoch:
                raise RuntimeError("epoch ended before its batches were exhausted")
        if step > next_step or saved is None:
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
        if device.type == "cuda" else None,
        "residency": source.residency,
        "runtime_history": runtime_history,
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
    identity = saved.get("identity")
    if (not isinstance(identity, dict) or canonical_hash(identity) != saved.get("signature")
            or saved.get("signature") != summary.get("signature")
            or summary.get("identity") != identity):
        raise RuntimeError(f"{out_dir}: checkpoint and summary disagree on the recipe")
    expected = {"backbone": cell.backbone, "method": cell.method,
                "target_k": cell.k, "seed": cell.seed}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{out_dir}: checkpoint belongs to a different training cell")
    validate_checkpoint_ot_weight(cell.method, cell.k, identity, context=str(out_dir))
    variant = METHOD_VARIANTS.get(cell.method)
    if variant and (identity.get("objective") or {}).get("variant") != variant:
        raise RuntimeError(f"{out_dir}: checkpoint uses a different {cell.method} objective variant; "
                           f"expected {variant}")
    if saved.get("next_step") != summary.get("steps"):
        raise RuntimeError(f"{out_dir}: checkpoint and summary disagree on step count")
    if cell.method in METHOD_EPOCHS:
        schedule = identity.get("schedule", {})
        if (schedule.get("epochs") != METHOD_EPOCHS[cell.method]
                or saved["next_step"] != schedule.get("total_steps")):
            raise RuntimeError(f"{out_dir}: MPSAEv2 checkpoint has not completed the requested "
                               f"{METHOD_EPOCHS[cell.method]}-epoch schedule")
    model = build_sae(spec.dim, spec.preset.hidden_dim, seed=cell.seed,
                      dead_threshold=spec.preset.dead_threshold, device=device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model
