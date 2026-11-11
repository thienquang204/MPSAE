"""Independent training microbenchmark; never writes campaign checkpoints.

Run ``python -m mpsae.profile_train --help`` for options. Without local frozen
embeddings the workload is explicitly synthetic, not an experiment result.
The optional profiler window is separate from the reported benchmark window.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any

from . import __version__
from .config import (BACKBONES, METHODS, METHOD_VARIANTS, training_preset,
                     validate_experiment_methods, validate_checkpoint_ot_weight)
from .io import atomic_json, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=tuple(BACKBONES), default="ff2048")
    parser.add_argument("--method", choices=METHODS, default="mpsae")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1227)
    parser.add_argument("--out", type=Path, default=Path("profile_train.json"))
    parser.add_argument("--trace", type=Path,
                        help="optional Chrome trace from a separate, short profiler window")
    parser.add_argument("--embeddings-dir", type=Path,
                        help="local frozen archives in the campaign embeddings layout")
    parser.add_argument("--cache-dir", type=Path, default=Path(".profile-cache"))
    parser.add_argument("--checkpoint", type=Path,
                        help="read model weights from an existing checkpoint; never overwrite it")
    parser.add_argument("--start-step", type=int, default=None,
                        help="annealing position; defaults to checkpoint next_step or zero")
    return parser


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"p50": statistics.median(ordered),
            "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "mean": statistics.fmean(ordered), "min": ordered[0], "max": ordered[-1]}


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark fresh optimizer updates on an independent model instance."""
    validate_experiment_methods((args.method,))
    import torch

    from . import methods
    from .features import BatchSource, load_bundle
    from .model import build_sae

    if args.steps < 1 or args.warmup < 0 or args.seed < 0:
        raise ValueError("steps must be positive; warmup and seed must be nonnegative")
    if args.start_step is not None and args.start_step < 0:
        raise ValueError("start-step must be nonnegative")
    spec = BACKBONES[args.backbone]
    preset = training_preset(args.method, spec.preset, k=args.k)
    if not 1 <= args.k <= preset.hidden_dim:
        raise ValueError(f"k must lie in 1..{preset.hidden_dim}")
    if args.trace is not None and args.trace.resolve() == args.out.resolve():
        raise ValueError("out and trace must use different files")
    outputs = {args.out.resolve()}
    if args.trace is not None:
        outputs.add(args.trace.resolve())
    if args.checkpoint is not None and args.checkpoint.resolve() in outputs:
        raise ValueError("profile output must not overwrite the input checkpoint")
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("device must be CUDA or CPU")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; --device cpu is only a diagnostic fallback")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    previous_precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = None
    try:
        model = build_sae(spec.dim, preset.hidden_dim, seed=args.seed,
                          dead_threshold=preset.dead_threshold, device=device)
        checkpoint = None
        checkpoint_position = 0
        if args.checkpoint is not None:
            # weights_only avoids executing arbitrary objects in a supplied file.
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
                raise ValueError("checkpoint must contain the campaign's model state dictionary")
            identity = saved.get("identity") or {}
            validate_checkpoint_ot_weight(args.method, args.k, identity, context=str(args.checkpoint))
            for key, value in (("backbone", args.backbone), ("method", args.method),
                               ("target_k", args.k)):
                if key in identity and identity[key] != value:
                    raise ValueError(f"checkpoint {key}={identity[key]!r} differs from {value!r}")
            variant = METHOD_VARIANTS.get(args.method)
            if variant and (identity.get("objective") or {}).get("variant") != variant:
                raise ValueError(f"checkpoint uses a different {args.method} objective variant; "
                                 f"expected {variant}")
            model.load_state_dict(saved["model"], strict=True)
            checkpoint_position = int(saved.get("next_step", 0))
            checkpoint = {"path": str(args.checkpoint.resolve()),
                          "sha256": sha256_file(args.checkpoint),
                          "next_step": checkpoint_position,
                          "optimizer": "fresh; checkpoint optimizer state is not resumed"}
            del saved
        optimizer = torch.optim.Adam(model.parameters(), lr=preset.lr, betas=preset.adam_betas,
                                     eps=preset.adam_eps, weight_decay=preset.weight_decay)

        if args.embeddings_dir is not None:
            bundle = load_bundle(spec, args.embeddings_dir, args.cache_dir, splits=("train",))
            source = BatchSource(bundle.train.features, bundle.train.labels,
                                 batch_size=preset.batch_size, seed=args.seed, device=device,
                                 with_labels=methods.requires_labels(args.method))
            data_info = {"kind": "local_frozen_embeddings", "rows": bundle.train.rows,
                         "identity": bundle.identity, "residency": source.residency,
                         "embeddings_dir": str(args.embeddings_dir.resolve())}
            schedule_rows = bundle.train.rows

            def batches():
                epoch = 0
                while True:
                    for _, batch, labels in source.epoch(epoch):
                        yield batch, labels
                    epoch += 1

            iterator = iter(batches())

            def next_batch():
                return next(iterator)
        else:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            synthetic = torch.randn(preset.batch_size, spec.dim, generator=generator).to(device)
            labels = (torch.arange(preset.batch_size, device=device) % 1000
                      if methods.requires_labels(args.method) else None)
            data_info = {"kind": "synthetic_resident_batch_repeated", "rows": preset.batch_size,
                         "residency": str(device),
                         "warning": "random features; measures runtime only, not model quality"}
            schedule_rows = 1_281_167

            def next_batch():
                return synthetic, labels

        schedule_steps = math.ceil(schedule_rows / preset.batch_size) * preset.epochs
        start_step = checkpoint_position if args.start_step is None else args.start_step
        if start_step < 0:
            raise ValueError("checkpoint next_step must be nonnegative")
        model.train()

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def update(offset: int, batch, labels):
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function("train/forward_loss"):
                output = methods.compute_step(args.method, model, batch, k=args.k, preset=preset,
                    step=min(start_step + offset, schedule_steps - 1), total_steps=schedule_steps,
                    labels=labels, collect_metrics=False)
            with torch.profiler.record_function("train/backward"):
                output.loss.backward()
            with torch.profiler.record_function("train/optimizer"):
                optimizer.step()
            return output.loss.detach(), output.training_k

        for index in range(args.warmup):
            batch, labels = next_batch()
            update(index, batch, labels)
        sync()
        if device.type == "cuda":
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(args.steps)]
            # Initialize lazy events before the measured window.
            for start, stop in events:
                start.record(torch.cuda.current_stream(device))
                stop.record(torch.cuda.current_stream(device))
            sync()
            torch.cuda.reset_peak_memory_stats(device)
        else:
            events = []
        wall_samples, device_samples, rows_seen, budgets = [], [], 0, []
        benchmark_started = time.perf_counter()
        for index in range(args.steps):
            began = time.perf_counter()
            with torch.profiler.record_function("train/data"):
                batch, labels = next_batch()
            if events:
                start, stop = events[index]
                start.record(torch.cuda.current_stream(device))
            last_loss, budget = update(args.warmup + index, batch, labels)
            if events:
                stop.record(torch.cuda.current_stream(device))
                stop.synchronize()
                device_samples.append(start.elapsed_time(stop) / 1000.0)
            wall_samples.append(time.perf_counter() - began)
            rows_seen += len(batch)
            budgets.append(budget)
        benchmark_seconds = time.perf_counter() - benchmark_started
        # Snapshot the benchmark's memory before profiler instrumentation runs.
        memory = ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                   "allocated_bytes": torch.cuda.memory_allocated(device),
                   "reserved_bytes": torch.cuda.memory_reserved(device),
                   "free_bytes": torch.cuda.mem_get_info(device)[0],
                   "total_bytes": torch.cuda.mem_get_info(device)[1]}
                  if device.type == "cuda" else None)
        result = {
            "format": "mpsae-training-profile-v1", "scientific_result": False,
            "backbone": args.backbone, "method": args.method, "target_k": args.k,
            "seed": args.seed, "preset": asdict(preset), "data": data_info,
            "objective": methods.describe(args.method, args.k, preset),
            "checkpoint": checkpoint, "optimizer_state": "fresh",
            "precision": {"dtype": "float32", "autocast": False, "tf32": False},
            "schedule": {"total_steps": schedule_steps, "start_step": start_step,
                         "position": "start_step + update offset, capped at final schedule step",
                         "measured_k_min": min(budgets), "measured_k_max": max(budgets)},
            "warmup_steps": args.warmup, "measured_steps": args.steps,
            "samples": rows_seen, "benchmark_seconds": benchmark_seconds,
            "samples_per_second": rows_seen / max(benchmark_seconds, 1e-12),
            "wall_step_seconds": _summary(wall_samples),
            "cuda_compute_seconds": _summary(device_samples) if device_samples else None,
            "timing_scope": {
                "wall": "batch_fetch_and_train_update_with_per_step_completion_sync",
                "cuda": "forward_backward_optimizer_excludes_batch_fetch",
                "excluded": "initialization_loading_warmup_profiler_checkpoint_io_metric_logging",
                "note": "synchronized microbenchmark; not end-to-end campaign throughput"},
            "last_loss": float(last_loss), "memory": memory,
            "runtime": {"device": str(device), "device_name": torch.cuda.get_device_name(device)
                        if device.type == "cuda" else platform.processor() or "CPU",
                        "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "python": platform.python_version(), "code_version": __version__},
            "trace": None,
        }
        atomic_json(args.out, result)
        if args.trace is not None:
            trace_steps = min(5, args.steps)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True,
                                         profile_memory=True) as profiler:
                for index in range(trace_steps):
                    with torch.profiler.record_function("train/data"):
                        batch, labels = next_batch()
                    update(args.warmup + args.steps + index, batch, labels)
                    profiler.step()
                sync()
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(args.trace))
            result["trace"] = {"path": str(args.trace.resolve()), "steps": trace_steps,
                               "timed_in_benchmark": False, "window": "after_benchmark"}
            atomic_json(args.out, result)
        return result
    finally:
        if source is not None:
            source.release()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_precision


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_profile(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"{result['data']['kind']}: {result['samples_per_second']:.1f} samples/s; "
          f"step p50 {result['wall_step_seconds']['p50'] * 1000:.2f} ms; {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
