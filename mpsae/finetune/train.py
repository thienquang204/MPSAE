"""Joint supervised fine-tuning, with a fixed teacher reconstruction target."""
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .. import methods
from ..config import training_preset
from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, read_jsonl, truncate_jsonl
from ..logs import MetricWriter, Progress
from ..train import _rng_state, _restore_rng, _configure_backend
from ..timing_config import runtime_fingerprint
from .config import VERSION
from .data import loader


def autocast(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if (
        torch.device(device).type == "cuda" and precision == "bf16") else nullcontext()


def forward_features(network, images, spec):
    values = network(images) if spec.recipe == "mrl" else network.forward_head(
        network.forward_features(images), pre_logits=True)
    if values.ndim != 2 or values.shape[1] != spec.dim:
        raise RuntimeError(f"{spec.name}: unexpected backbone features {tuple(values.shape)}")
    return values


def train_mode(network, freeze_bn_stats):
    network.train().requires_grad_(True)
    if freeze_bn_stats:
        for module in network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()  # affine parameters train; small-batch running statistics do not


def joint_step(cell, spec, network, teacher, sae, head, images, labels, *, options, step, total,
               collect_metrics=True):
    with torch.no_grad(), autocast(images.device, options["precision"]):
        target = forward_features(teacher, images, spec).float()
    with autocast(images.device, options["precision"]):
        features = forward_features(network, images, spec).float()
    # SAE, teacher reconstruction, contrastive and MMPOT stay float32.
    result = methods.compute_step(cell.method, sae, features, k=cell.k,
        preset=training_preset(cell.method, spec.preset, k=cell.k), step=step, total_steps=total,
        labels=labels, reconstruction_target=target, fixed_budget=True, collect_metrics=collect_metrics)
    logits = head(F.normalize(result.latents, dim=-1))
    ce = F.cross_entropy(logits, labels)
    loss = options["ce_weight"] * ce + options["method_weight"] * result.loss
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite joint fine-tuning loss")
    if not collect_metrics:
        return loss, {}
    values = torch.stack([result.loss.detach(), ce.detach(), loss.detach(),
        100 * (logits.argmax(-1) == labels).float().mean()]).cpu().tolist()
    return loss, {**result.metrics, **dict(zip(
        ("loss_method", "loss_ce", "loss_total", "head_top1"), values))}


def fit_joint(cell, spec, network, teacher, sae, dataset, transform, directory, *,
              options, source_identity, data_identity, preprocessing, device, workers=0,
              classes=1000, max_steps=0):
    directory = Path(directory)
    with exclusive_lock(directory / "fit.lock", timeout=0):
        return _fit(cell, spec, network, teacher, sae, dataset, transform, directory,
                    options=options, source_identity=source_identity, data_identity=data_identity,
                    preprocessing=preprocessing, device=device, workers=workers,
                    classes=classes, max_steps=max_steps)


def _fit(cell, spec, network, teacher, sae, dataset, transform, directory, *, options,
         source_identity, data_identity, preprocessing, device, workers, classes, max_steps):
    device = torch.device(device)
    _configure_backend()
    per_epoch = (len(dataset) // options["batch_size"] if options["drop_last"]
                 else -(-len(dataset) // options["batch_size"]))
    if not per_epoch:
        raise ValueError("not enough images for one complete physical training batch")
    total = per_epoch * options["epochs"]
    if max_steps:
        total = min(total, max_steps)
    identity = {"format": VERSION, "cell": cell.__dict__, "source": source_identity,
        "data": data_identity, "preprocessing": preprocessing, "training": options,
        "preset": asdict(training_preset(cell.method, spec.preset, k=cell.k)),
        "schedule": {"total_steps": total, "per_epoch": per_epoch, "drop_last": options["drop_last"]},
        "classes": classes, "smoke": bool(max_steps)}
    signature = canonical_hash(identity)
    checkpoint_path = directory / "checkpoint.pt"
    for filename in ("config.json", "summary.json"):
        previous = read_json(directory / filename)
        if previous and (previous.get("signature") != signature or canonical_hash(previous.get("identity")) != signature):
            raise RuntimeError(f"{directory}: changed fine-tuning recipe/source; use a separate campaign name")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path.exists() else None
    if saved and (saved.get("format") != VERSION or saved.get("signature") != signature
                  or canonical_hash(saved.get("identity")) != signature):
        raise RuntimeError("fine-tuning checkpoint identity mismatch")
    if saved and (not isinstance(saved.get("next_step"), int) or not 0 <= saved["next_step"] <= total):
        raise RuntimeError("invalid fine-tuning resume position")
    summary = read_json(directory / "summary.json")
    if summary and summary.get("complete") and (not saved or saved["next_step"] != total):
        raise RuntimeError("completed summary has no matching completed checkpoint")
    network.to(device)
    teacher.to(device).eval().requires_grad_(False)
    sae.to(device).train().requires_grad_(True)
    train_mode(network, options["freeze_bn_stats"])
    if saved:
        network.load_state_dict(saved["backbone"], strict=True)
        sae.load_state_dict(saved["sae"], strict=True)
        if saved["next_step"] == total and summary and summary.get("complete"):
            # Evaluation needs only these weights, not the training head or
            # potentially gigabytes of CUDA Adam state.
            return summary
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cell.seed)
        head = nn.Linear(spec.preset.hidden_dim, classes).to(device)
    optimizer_class = torch.optim.Adam if options["optimizer"] == "adam" else torch.optim.AdamW
    optimizer = optimizer_class([
        {"params": list(network.parameters()), "lr": options["backbone_lr"]},
        {"params": list(sae.parameters()), "lr": options["sae_lr"]},
        {"params": list(head.parameters()), "lr": options["classifier_lr"]}],
        weight_decay=options["weight_decay"], eps=options["adam_eps"], betas=tuple(options["adam_betas"]))
    step, prior_seconds = 0, 0.0
    if saved:
        head.load_state_dict(saved["classifier"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"], device)
        step, prior_seconds = saved["next_step"], saved["train_seconds"]
    else:
        torch.manual_seed(cell.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cell.seed)
    del saved
    if step == total and summary and summary.get("complete"):
        return summary
    atomic_json(directory / "config.json", {"signature": signature, "identity": identity,
        "regime": "supervised_joint_finetune", "total_steps": total,
        "runtime": runtime_fingerprint(device)})
    records = read_jsonl(directory / "train.jsonl")
    truncate_jsonl(directory / "train.jsonl", sum(r.get("step", total + 1) <= step for r in records))
    writer = MetricWriter(directory / "train.jsonl", stage=f"finetune/{cell.slug}")
    progress = Progress(total, done=step, stage=f"finetune/{cell.slug}", what="steps")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    progress_detail = ""

    def save():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        payload = {"format": VERSION, "identity": identity, "signature": signature,
            "backbone": network.state_dict(), "sae": sae.state_dict(), "classifier": head.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": _rng_state(device), "next_step": step,
            "train_seconds": prior_seconds + time.perf_counter() - started}
        temporary = checkpoint_path.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)

    # Commit only completed optimizer updates; a failed in-flight step is replayed
    # from the last durable checkpoint, never saved with partially mutated state.
    while step < total:
        epoch, within = divmod(step, per_epoch)
        batches = loader(dataset, transform, seed=cell.seed, epoch=epoch,
            batch_size=options["batch_size"], workers=workers,
            skip=within * options["batch_size"], training=True, drop_last=options["drop_last"])
        for images, labels in batches:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            collect_metrics = (step + 1) % options["log_every"] == 0 or step + 1 == total
            loss, metrics = joint_step(cell, spec, network, teacher, sae, head, images, labels,
                                      options=options, step=step, total=total, collect_metrics=collect_metrics)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters,
                                                  options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % options["log_every"] == 0 or step == total:
                metrics.update(grad_norm=float(norm), elapsed=prior_seconds + time.perf_counter() - started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
                writer.log({"epoch": epoch, **metrics}, step=step)
                progress_detail = f"last logged loss {metrics['loss_total']:.4f} at step {step}"
            if step % options["checkpoint_every"] == 0 or step == total:
                save()
            progress.advance(extra=progress_detail)
            if step >= total:
                break
    optimizer.zero_grad(set_to_none=True)
    summary = {"identity": identity, "signature": signature, "cell": cell.__dict__,
        "complete": True, "steps": step, "train_seconds": prior_seconds + time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
    summary["gpu_hours"] = summary["train_seconds"] / 3600
    atomic_json(directory / "summary.json", summary)
    return summary
