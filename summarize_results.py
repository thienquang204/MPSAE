#!/usr/bin/env python3
"""Print the MRL vs MMPOT result tables for a finished run.

Standard library only, so it runs inside the container and on the host without
installing torch. Point it at a run directory (or the /output root) and it
finds every ``summary.json`` beneath it.

    python summarize_results.py runs/imagenet_rn50_b256_e5_lr0475
    python summarize_results.py runs --metric top1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

METHODS = ("mrl", "mmpot")
BENCHMARKS = ("head", "knn", "linear")
# Higher is better for accuracy, lower is better for loss.
LOWER_IS_BETTER = {"loss"}
METRIC_ORDER = ("top1", "top5", "loss")


def find_summaries(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("summary.json"))


def metric_order_key(metric: str) -> Tuple[int, str]:
    try:
        return (METRIC_ORDER.index(metric), "")
    except ValueError:
        return (len(METRIC_ORDER), metric)


def collect_metrics(summary: Dict[str, Any], benchmark: str) -> Tuple[List[int], List[str]]:
    """Return the dimensions and metric names both methods report."""
    per_method = {}
    for method in METHODS:
        block = summary.get("methods", {}).get(method, {}).get(benchmark)
        if not isinstance(block, dict):
            return [], []
        per_method[method] = block.get("per_dim", {})

    shared_dims = set(per_method["mrl"]) & set(per_method["mmpot"])
    if not shared_dims:
        return [], []
    dims = sorted(int(dim) for dim in shared_dims)

    metrics: Optional[set] = None
    for method in METHODS:
        for dim in dims:
            entry = per_method[method].get(str(dim), {})
            names = {
                key for key, value in entry.items() if isinstance(value, (int, float))
            }
            metrics = names if metrics is None else (metrics & names)
    return dims, sorted(metrics or set(), key=metric_order_key)


def format_delta(delta: float, metric: str) -> str:
    better = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
    mark = "+" if delta > 0 else ""
    flag = "  <-- MMPOT wins" if better and abs(delta) >= 1e-9 else ""
    return f"{mark}{delta:.3f}{flag}"


def print_benchmark_table(summary: Dict[str, Any], benchmark: str, wanted: Sequence[str]) -> bool:
    dims, metrics = collect_metrics(summary, benchmark)
    if not dims or not metrics:
        return False
    if wanted:
        metrics = [m for m in metrics if m in wanted]
    if not metrics:
        return False

    methods = summary.get("methods", {})
    for metric in metrics:
        unit = "" if metric == "loss" else " (%)"
        print(f"\n  [{benchmark}] {metric}{unit}")
        print(f"    {'dim':>6}  {'MRL':>9}  {'MMPOT':>9}  {'delta':>9}")
        print(f"    {'-' * 6}  {'-' * 9}  {'-' * 9}  {'-' * 9}")
        totals = {method: 0.0 for method in METHODS}
        for dim in dims:
            values = {
                method: float(
                    methods[method][benchmark]["per_dim"][str(dim)][metric]
                )
                for method in METHODS
            }
            for method in METHODS:
                totals[method] += values[method]
            delta = values["mmpot"] - values["mrl"]
            print(
                f"    {dim:>6}  {values['mrl']:>9.3f}  {values['mmpot']:>9.3f}  "
                f"{format_delta(delta, metric)}"
            )
        means = {method: totals[method] / len(dims) for method in METHODS}
        mean_delta = means["mmpot"] - means["mrl"]
        print(f"    {'-' * 6}  {'-' * 9}  {'-' * 9}  {'-' * 9}")
        print(
            f"    {'mean':>6}  {means['mrl']:>9.3f}  {means['mmpot']:>9.3f}  "
            f"{format_delta(mean_delta, metric)}"
        )
    return True


def print_summary(path: Path, wanted: Sequence[str]) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"!! could not read {path}: {exc}", file=sys.stderr)
        return False

    config = summary.get("config", {})
    dataset = summary.get("dataset", {})
    print()
    print("=" * 72)
    print(f"RESULTS  {path.parent}")
    print("=" * 72)
    print(
        f"  dataset={dataset.get('name')} train={dataset.get('train_samples')} "
        f"val={dataset.get('val_samples')} classes={dataset.get('num_classes')}"
    )
    print(
        f"  arch={config.get('architecture')} optimizer={config.get('optimizer')} "
        f"lr={config.get('lr')} epochs={config.get('epochs')} batch={config.get('batch_size')}"
    )
    print(
        f"  ot_marginals={summary.get('ot_marginals', config.get('ot_marginals'))} "
        f"ot_lambda={config.get('ot_lambda')} ot_mass={config.get('ot_mass')} "
        f"fidelity={summary.get('mmpot_fidelity')}"
    )

    printed = False
    for benchmark in BENCHMARKS:
        printed |= print_benchmark_table(summary, benchmark, wanted)
    if not printed:
        # A single-method run has nothing to compare; show what is there.
        for method in METHODS:
            block = summary.get("methods", {}).get(method)
            if not block:
                continue
            for benchmark in BENCHMARKS:
                per_dim = (block.get(benchmark) or {}).get("per_dim")
                if not per_dim:
                    continue
                printed = True
                print(f"\n  [{benchmark}] {method} only")
                for dim in sorted(per_dim, key=int):
                    entries = ", ".join(
                        f"{k}={v:.3f}"
                        for k, v in sorted(per_dim[dim].items(), key=lambda kv: metric_order_key(kv[0]))
                        if isinstance(v, (int, float))
                    )
                    print(f"    dim {int(dim):>6}: {entries}")
    if not printed:
        print("  (no per-dimension benchmark results in this summary)")
    return printed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("runs"))
    parser.add_argument(
        "--metric",
        default="top1,top5",
        help="comma-separated metrics to print, or 'all' (default: top1,top5)",
    )
    args = parser.parse_args(argv)

    root = args.root.expanduser()
    if not root.exists():
        print(f"No results: {root} does not exist", file=sys.stderr)
        return 1

    wanted: List[str] = []
    if args.metric.strip().lower() != "all":
        wanted = [m.strip() for m in args.metric.split(",") if m.strip()]

    summaries = find_summaries(root)
    if not summaries:
        print(f"No summary.json found under {root}", file=sys.stderr)
        return 1

    for path in summaries:
        print_summary(path, wanted)
    print()
    print(f"{len(summaries)} run(s) summarized from {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
