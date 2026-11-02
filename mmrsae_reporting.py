#!/usr/bin/env python3
"""Tables and figures for CSR, CSRv2, DLMMSAE and DLMMSAEv2.

Trials are grouped by everything that defines a comparable experiment except
the method (settings without ``variant``, plus the implementation identity), so
a figure never mixes training K, data sizes, evaluation pools or code versions. Each
group gets ``reports/<group>/`` with results.csv/results.md and PNG+PDF figures:
top-1 versus inference K, endpoint top-1 by method, training loss, and support
health. Other ablation methods are filtered out, and the legacy DLMMSAE annealed
name is displayed as DLMMSAEv2. Failed or partial fits are reported from whatever they wrote; nothing here is trained,
evaluated or downloaded, and it runs on CPU after any launcher invocation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
import sys

from repair_data import atomic_json, canonical_hash
from mmrsae_methods import BENCHMARK_METHODS, METHOD_LABELS, canonical_method

# Fixed method order and color slots: color follows the method, never its rank.
METHODS = BENCHMARK_METHODS
COLORS = ("#008300", "#7953a8", "#ad284f", "#176b76")
MARKERS = ("P", "X", "*", "h")
OTHER_COLOR = "#8c8b84"
BACKBONES = ("resnet50", "resnet18")
INK, INK_MUTED, GRID, SURFACE = "#1f1e1b", "#5f5e58", "#e4e3de", "#fcfcfb"


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _label(method):
    method = canonical_method(method)
    return METHOD_LABELS.get(method, method.upper())


def _backbone_label(backbone):
    return backbone.replace("resnet", "ResNet-")


def _hardware(gpu):
    """Stable hardware/software identity; excludes probe timings and free memory."""
    return {key: gpu.get(key) for key in ("device_name", "device_total_bytes", "compute_capability",
                                         "torch_version", "cuda_version") if gpu.get(key) is not None}


def _timing_fields(result):
    """Only repeated, warm CUDA measurements qualify for a latency curve."""
    timing = result.get("timing") or {}
    milliseconds = result.get("mean_retrieval_milliseconds_per_batch")
    valid = (result.get("timed") is True and timing.get("version") == 2
             and timing.get("primary_statistic") == "median_of_interleaved_repeat_totals"
             and timing.get("cache_policy") == "warm_repeated_search_of_each_resident_query_batch"
             and _finite(milliseconds) and milliseconds > 0)
    cohort = None
    if valid:
        identity = {key: result.get(key) for key in (
            "device_name", "torch_version", "cuda_version", "dtype", "query_batch_size", "gallery_count",
            "query_count", "gallery_ids_sha256", "query_ids_sha256")}
        identity["timer"] = {key: timing.get(key) for key in
                             ("version", "scope", "primary_statistic", "warmup_per_shape", "repeats_per_batch")}
        cohort = canonical_hash(identity)[:10]
    return {"retrieval_batch_ms": milliseconds if valid else None,
            "timing_cohort": cohort, "device_name": result.get("device_name"),
            "query_batch_size": result.get("query_batch_size"),
            "retrieval_timing_scope": timing.get("scope") if valid else None}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_log(path):
    rows = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # An interrupted append leaves at most one partial line.
    return rows


def _style(method):
    if method in METHODS:
        slot = METHODS.index(method)
        return COLORS[slot], MARKERS[slot]
    return OTHER_COLOR, "x"


def _method_order(methods):
    return sorted(methods, key=lambda name: (METHODS.index(name) if name in METHODS else len(METHODS), name))


def collect(output_dir):
    """Return {group_id: {"label", "settings", "trials": {method: {...}}}}."""
    groups = {}
    for trial_path in sorted(Path(output_dir).glob("*/trial.json")):
        trial = _read_json(trial_path)
        if not isinstance(trial, dict) or "settings" not in trial:
            continue
        settings = dict(trial["settings"])
        method = settings.pop("variant", trial_path.parent.name.split("_k")[0])
        method = canonical_method(method)
        if method not in METHODS:
            continue
        hardware = _hardware(_read_json(trial_path.parent / "gpu_preflight.json") or {})
        key = canonical_hash({"settings": settings, "code": trial.get("code"), "hardware": hardware})
        label = f"{settings.get('evaluation', 'dev')}_k{settings.get('k', '?')}_{key[:8]}"
        group = groups.setdefault(label, {"label": label, "settings": settings,
                                          "hardware": hardware, "trials": {}})
        backbones = {}
        for backbone in BACKBONES:
            directory = trial_path.parent / backbone
            if not directory.is_dir():
                continue
            summary = _read_json(directory / "summary.json")
            failure = _read_json(directory / "failure.json")
            complete = isinstance(summary, dict) and bool(summary.get("complete"))
            backbones[backbone] = {
                "status": "complete" if complete else ("failed" if failure else "incomplete"),
                "summary": summary if complete else None,
                "failure": failure,
                "log": _read_log(directory / "training.jsonl"),
                "retrieval": (summary.get("retrieval") if complete else None)
                             or _read_json(directory / "retrieval" / "endpoint" / "result.json") or {},
                "full_parent": (summary.get("full_parent") if complete else None)
                               or _read_json(directory / "parent" / "endpoint" / "result.json") or {},
                "classification": (summary.get("classification") if complete else None)
                                  or _read_json(directory / "classification" / "endpoint" / "result.json") or {},
            }
        if method in group["trials"]:
            # Same method and settings in two directories (explicit --run-name):
            # keep the one with more completed backbones, then the newer one.
            old = group["trials"][method]
            score = lambda item: (sum(b["status"] == "complete" for b in item["backbones"].values()), item["mtime"])
            candidate = {"directory": str(trial_path.parent), "backbones": backbones,
                         "mtime": trial_path.stat().st_mtime}
            if score(candidate) <= score(old):
                continue
        group["trials"][method] = {"directory": str(trial_path.parent), "backbones": backbones,
                                   "mtime": trial_path.stat().st_mtime}
    return groups


def result_rows(group):
    rows = []
    for method in _method_order(group["trials"]):
        trial = group["trials"][method]
        for backbone in BACKBONES:
            entry = trial["backbones"].get(backbone)
            if entry is None:
                continue
            summary = entry["summary"] or {}
            retrieval = entry.get("retrieval") or summary.get("retrieval") or {}
            parent = entry.get("full_parent") or summary.get("full_parent") or {}
            classification = entry.get("classification") or summary.get("classification") or {}
            class_parent = classification.get("full_parent") or {}
            parent_timing = _timing_fields(parent)
            failure = entry["failure"] or {}
            rows.append({
                "method": method, "backbone": backbone, "status": entry["status"],
                "training_k": group["settings"].get("k"),
                "K": retrieval.get("K", group["settings"].get("k")),
                "top1": retrieval.get("top1"), "correct": retrieval.get("correct"),
                "queries": retrieval.get("total"), "actual_nnz_mean": retrieval.get("actual_nnz_mean"),
                "empty_query_codes": retrieval.get("empty_query_codes"),
                "empty_query_fraction": retrieval.get("empty_query_fraction"),
                "empty_gallery_codes": retrieval.get("empty_gallery_codes"),
                "full_parent_top1": parent.get("top1"),
                "full_parent_batch_ms": parent_timing["retrieval_batch_ms"],
                "full_parent_timing_cohort": parent_timing["timing_cohort"],
                "classification_top1": None, "classification_actual_nnz_mean": None,
                "classification_protocol": classification.get("protocol"),
                "full_parent_classification_top1": class_parent.get("top1"),
                "updates": summary.get("updates", len(entry["log"])),
                "fit_seconds": summary.get("fit_seconds"),
                "training_gpu_hours": summary.get("training_gpu_hours"),
                "training_seconds": summary.get("training_seconds"),
                "training_time_scope": summary.get("training_time_scope"),
                "training_device_name": summary.get("training_device_name", group.get("hardware", {}).get("device_name")),
                "train_max_zero_codes": max((row.get("zero_final_codes", 0) for row in entry["log"]), default=None),
                "train_min_nnz": min((row.get("final_nnz_min") for row in entry["log"]
                                      if row.get("final_nnz_min") is not None), default=None),
                "failure": failure.get("error"), "directory": trial["directory"],
                **_timing_fields(retrieval),
            })
    return rows


def budget_rows(group):
    """Measured rows per inference K; legacy summaries supply only their actual K."""
    rows = []
    for base in result_rows(group):
        entry = group["trials"][base["method"]]["backbones"][base["backbone"]]
        retrieval = entry.get("retrieval") or (entry.get("summary") or {}).get("retrieval") or {}
        classification = entry.get("classification") or (entry.get("summary") or {}).get("classification") or {}
        class_points = classification.get("per_topk") or {}
        per_topk = retrieval.get("per_topk")
        if not isinstance(per_topk, dict):
            per_topk = {str(base["K"]): retrieval} if base["K"] is not None else {}
        for key in sorted(per_topk, key=int):
            point = per_topk[key]
            value = point.get("top1")
            if not _finite(value):
                continue
            row = dict(base, K=int(key), top1=value,
                       budget_evaluation=retrieval.get("budget_evaluation", "single_k"))
            for field in ("correct", "actual_nnz_mean", "empty_query_codes", "empty_query_fraction", "empty_gallery_codes"):
                row[field] = point.get(field)
            row["queries"] = point.get("total")
            row.update(_timing_fields(point))
            class_point = class_points.get(str(key)) or {}
            row["classification_top1"] = class_point.get("top1")
            row["classification_actual_nnz_mean"] = class_point.get("query_actual_nnz_mean")
            if row["classification_actual_nnz_mean"] is None and class_point.get("actual_nnz_scope") == "query":
                row["classification_actual_nnz_mean"] = class_point.get("actual_nnz_mean")
            rows.append(row)
    return rows


def _format(value, digits=2):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_tables(rows, target, settings):
    fields = list(rows[0]) if rows else ["method", "backbone", "status"]
    with (target / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    columns = [("method", "Method"), ("backbone", "Backbone"), ("training_k", "Train K"),
               ("K", "Inference K"), ("status", "Status"), ("top1", "1-NN (%)"),
               ("classification_top1", "Linear-probe Top-1 (%)"),
               ("actual_nnz_mean", "Gallery nnz"), ("empty_query_codes", "Empty queries"),
               ("empty_gallery_codes", "Empty gallery"), ("train_max_zero_codes", "Max zero codes/batch"),
               ("full_parent_top1", "Full parent 1-NN"), ("retrieval_batch_ms", "Search ms/batch"),
               ("training_gpu_hours", "Train GPU-hours"), ("updates", "Updates"), ("fit_seconds", "Fit (s)")]
    lines = [f"# MMRSAE results: {target.name}", "",
             f"K={settings.get('k')}, evaluation={settings.get('evaluation')}, "
             f"train_size={settings.get('train_size')}, dev_size={settings.get('dev_size')}, "
             f"max_steps={settings.get('max_steps')}, epochs={settings.get('epochs')}, seed={settings.get('seed')}.",
             "Empty (all-zero) queries are counted as retrieval misses.", "",
             "Each curve evaluates one trained checkpoint at multiple inference budgets; these are not independent per-K fits.", "",
             "1-NN retrieval and supervised linear-probe Top-1 classification are separate measurements. "
             "Blank entries mean not measured. Active dimensions are measured mean positive coefficients, not the requested K.", "",
             "Search time is mean CUDA batch time derived from the median of interleaved warm repeat totals. "
             "It excludes embedding encoding, normalization, transfers, index build and feature extraction; "
             "the final partial batch is included. Timing cohorts separate hardware, query batch sizes, row pools and timer settings.", "",
             "Training GPU-hours use the explicitly measured training interval. fit_seconds includes evaluation and other work "
             "and is never substituted. GPU-hours exclude frozen backbone pretraining, extraction and linear-probe fitting. "
             "Mean-accuracy compute plots require every backbone in the stated set at the same K; their x-axis sums GPU-hours across those fits.", "",
             "| " + " | ".join(title for _, title in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[name], 0 if name == "fit_seconds" else 2) for name, _ in columns) + " |")
    failures = [row for row in rows if row["failure"]]
    if failures:
        lines += ["", "Failures:"] + [f"- {row['method']}/{row['backbone']}: {row['failure']}" for row in failures]
    (target / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": .6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.6,
    })
    return plt


def _save(plt, fig, target, stem):
    names = []
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = target / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **options)
        names.append(path.name)
    plt.close(fig)
    return names


def plot_top1(plt, rows, target, title):
    methods = _method_order({row["method"] for row in rows})
    backbones = [name for name in BACKBONES if any(row["backbone"] == name for row in rows)]
    if not methods or not backbones:
        return []
    lookup = {(row["method"], row["backbone"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(max(6., 4.2 * len(backbones)), 3.8))
    width = .8 / len(methods)
    top = max([row[key] for row in rows for key in ("top1", "full_parent_top1")
               if _finite(row.get(key))] + [1.])
    for slot, method in enumerate(methods):
        color, _ = _style(method)
        for position, backbone in enumerate(backbones):
            row = lookup.get((method, backbone))
            x = position - .4 + width * (slot + .5)
            if row is None:
                continue
            if row["top1"] is None:
                ax.text(x, top * .02, row["status"], rotation=90, ha="center", va="bottom",
                        fontsize=7, color=INK_MUTED)
                continue
            # 2px surface gap between adjacent bars; value labelled above each bar.
            ax.bar(x, row["top1"], width, color=color, edgecolor=SURFACE, linewidth=1.5,
                   label=_label(method) if position == 0 else None)
            note = f"{row['top1']:.1f}"
            if row.get("empty_query_fraction"):
                note += f"\n{row['empty_query_fraction']:.0%} empty"
            ax.text(x, row["top1"] + top * .015, note, ha="center", va="bottom", fontsize=6.5, color=INK)
    for position, backbone in enumerate(backbones):
        parents = [row["full_parent_top1"] for row in rows
                   if row["backbone"] == backbone and row["full_parent_top1"] is not None]
        if parents:
            ax.hlines(parents[0], position - .45, position + .45, colors=INK_MUTED, linestyles="--", linewidth=1)
            ax.text(position + .45, parents[0], " full parent", va="center", fontsize=6.5, color=INK_MUTED)
    ax.set_xticks(range(len(backbones)), [name.replace("resnet", "ResNet-") for name in backbones])
    ax.set_ylabel("Top-1 1-NN accuracy (%)")
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    ax.legend(ncol=min(len(methods), 6), loc="upper center", bbox_to_anchor=(.5, -.1), frameon=False)
    return _save(plt, fig, target, "top1_by_method")


def _series(log, key):
    points = [(row["step"] + 1, row[key]) for row in log
              if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
    return [x for x, _ in points], [y for _, y in points]


def plot_accuracy_by_k(plt, rows, target, title):
    """Plot every measured budget, with gaps wherever a method lacks a point."""
    backbones = [name for name in BACKBONES if any(row["backbone"] == name for row in rows)]
    if not backbones:
        return []
    fig, axes = plt.subplots(1, len(backbones), squeeze=False,
                             figsize=(5.2 * len(backbones), 3.8))
    handles, labels = [], []
    for column, backbone in enumerate(backbones):
        ax = axes[0][column]
        selected = [row for row in rows if row["backbone"] == backbone]
        grid = sorted({row["K"] for row in selected})
        for method in _method_order({row["method"] for row in selected}):
            points = {row["K"]: row["top1"] for row in selected if row["method"] == method}
            color, marker = _style(method)
            ax.plot(grid, [points.get(k, math.nan) for k in grid], color=color,
                    marker=marker, markersize=4, label=_label(method))
        _parent_line(ax, selected)
        ax.set_xscale("log", base=2)
        ax.set_xticks(grid, [str(k) for k in grid])
        ax.minorticks_off()
        ax.set_xlabel("Inference budget K")
        ax.set_ylabel("Top-1 1-NN accuracy (%)")
        ax.set_title(backbone.replace("resnet", "ResNet-"), loc="left")
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle(title, x=.01, ha="left", fontsize=10)
    fig.legend(handles, labels, ncol=min(len(labels), 4), loc="lower center",
               bbox_to_anchor=(.5, -.08), frameon=False)
    fig.tight_layout(rect=(0, .07, 1, .94))
    return _save(plt, fig, target, "top1_by_k")


def _parent_line(ax, rows, field="full_parent_top1", *, color=INK_MUTED, label="Full representation"):
    values = sorted({row[field] for row in rows if _finite(row.get(field))})
    if len(values) == 1:
        ax.axhline(values[0], color=color, linestyle="--", linewidth=1, label=label)
    elif values:
        # Preserve evidence of inconsistent baselines instead of silently choosing one.
        ax.axhspan(values[0], values[-1], color=color, alpha=.12, label=f"{label} (measured range)")


def _figure_legend(fig, axes):
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, ncol=min(4, len(labels)), loc="lower center",
                   bbox_to_anchor=(.5, -.04), frameon=False)


def _grid(plt, count, *, width=4., height=3.):
    columns = min(3, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(width * columns, height * rows + .6))
    flattened = list(axes.flat)
    for ax in flattened[count:]:
        ax.set_visible(False)
    return fig, flattened[:count]


def _positive_axis(ax, values):
    from matplotlib.ticker import FuncFormatter
    if values and all(value > 0 for value in values):
        ax.set_xscale("log", base=2)
    elif values:
        ax.set_xscale("symlog", base=2, linthresh=1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def plot_backbone_curves(plt, rows, target):
    """One method per panel, with a separate full-representation line per backbone."""
    methods = _method_order({row["method"] for row in rows})
    if not methods:
        return []
    fig, axes = _grid(plt, len(methods))
    for ax, method in zip(axes, methods):
        selected = [row for row in rows if row["method"] == method]
        grid = sorted({row["K"] for row in selected})
        for slot, backbone in enumerate(BACKBONES):
            points = [row for row in selected if row["backbone"] == backbone]
            if not points:
                continue
            lookup = {row["K"]: row["top1"] for row in points}
            color = ("#2a78d6", "#ad284f")[slot]
            ax.plot(grid, [lookup.get(k, math.nan) for k in grid], color=color,
                    marker=("o", "s")[slot], markersize=4, label=_backbone_label(backbone))
            _parent_line(ax, points, color=color, label=f"Full {_backbone_label(backbone)}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(grid, [str(k) for k in grid], rotation=35 if len(grid) > 7 else 0)
        ax.minorticks_off()
        ax.set_xlabel("Inference budget K")
        ax.set_ylabel("1-NN accuracy (%)")
        ax.set_title(_label(method), loc="left")
    fig.suptitle("Backbone comparison at every measured K", x=.01, ha="left", fontsize=10)
    _figure_legend(fig, axes)
    fig.tight_layout(rect=(0, .06, 1, .95))
    return _save(plt, fig, target, "top1_by_k_backbones")


def plot_active_dimensions(plt, rows, target):
    """Classification and retrieval are distinct panels, never interchangeable."""
    backbones = [name for name in BACKBONES if any(row["backbone"] == name for row in rows)]
    if not backbones:
        return []
    fig, axes = plt.subplots(len(backbones), 2, squeeze=False, figsize=(10., 3.5 * len(backbones) + .5))
    for row_index, backbone in enumerate(backbones):
        selected = [row for row in rows if row["backbone"] == backbone]
        for column, (metric, dimension, parent, ylabel) in enumerate((
                ("classification_top1", "classification_actual_nnz_mean", "full_parent_classification_top1",
                 "Linear-probe Top-1 accuracy (%)"),
                ("top1", "actual_nnz_mean", "full_parent_top1", "1-NN accuracy (%)"))):
            ax = axes[row_index][column]
            dimensions = []
            for method in _method_order({row["method"] for row in selected}):
                points = sorted((row for row in selected if row["method"] == method), key=lambda row: row["K"])
                color, marker = _style(method)
                x = [row.get(dimension) if _finite(row.get(dimension)) else math.nan for row in points]
                y = [row.get(metric) if _finite(row.get(metric)) else math.nan for row in points]
                measured = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
                if measured:
                    dimensions.extend(a for a, _ in measured)
                    ax.plot(x, y, color=color, marker=marker, markersize=4, label=_label(method))
            if not dimensions:
                ax.text(.5, .45, "Classification not measured" if column == 0 else "Active dimensions not measured",
                        transform=ax.transAxes, ha="center", color=INK_MUTED)
            _parent_line(ax, selected, parent)
            _positive_axis(ax, dimensions)
            ax.set_xlabel("Mean active dimensions (query)" if column == 0 else "Mean active dimensions (gallery)")
            ax.set_ylabel(ylabel)
            ax.set_title(_backbone_label(backbone), loc="left")
    fig.suptitle("Measured active dimensions; curves connect increasing inference K", x=.01, ha="left", fontsize=10)
    _figure_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, .07, 1, .95))
    return _save(plt, fig, target, "accuracy_by_active_dim")


def plot_classification_by_k(plt, rows, target):
    measured = [row for row in rows if _finite(row.get("classification_top1"))]
    backbones = [name for name in BACKBONES if any(row["backbone"] == name for row in measured)]
    if not backbones:
        return []
    fig, axes = _grid(plt, len(backbones), width=5., height=3.5)
    for ax, backbone in zip(axes, backbones):
        selected = [row for row in rows if row["backbone"] == backbone]
        grid = sorted({row["K"] for row in selected})
        for method in _method_order({row["method"] for row in selected}):
            points = {row["K"]: row.get("classification_top1") for row in selected if row["method"] == method}
            color, marker = _style(method)
            ax.plot(grid, [points[k] if _finite(points.get(k)) else math.nan for k in grid],
                    color=color, marker=marker, markersize=4, label=_label(method))
        _parent_line(ax, selected, "full_parent_classification_top1")
        ax.set_xscale("log", base=2)
        ax.set_xticks(grid, [str(k) for k in grid])
        ax.minorticks_off()
        ax.set_xlabel("Inference budget K")
        ax.set_ylabel("Linear-probe Top-1 accuracy (%)")
        ax.set_title(_backbone_label(backbone), loc="left")
    fig.suptitle("Supervised classification on frozen representations", x=.01, ha="left", fontsize=10)
    _figure_legend(fig, axes)
    fig.tight_layout(rect=(0, .10, 1, .94))
    return _save(plt, fig, target, "classification_top1_by_k")


def plot_retrieval_tradeoff(plt, rows, target):
    cohorts = sorted({(row["backbone"], row["timing_cohort"]) for row in rows
                      if row.get("timing_cohort") and _finite(row.get("retrieval_batch_ms"))},
                     key=lambda item: (BACKBONES.index(item[0]), item[1]))
    if not cohorts:
        return []
    fig, axes = _grid(plt, len(cohorts), width=5., height=3.6)
    for ax, (backbone, cohort) in zip(axes, cohorts):
        selected = [row for row in rows if row["backbone"] == backbone and row.get("timing_cohort") == cohort]
        for method in _method_order({row["method"] for row in selected}):
            points = sorted((row for row in selected if row["method"] == method), key=lambda row: row["K"])
            color, marker = _style(method)
            ax.plot([row["retrieval_batch_ms"] for row in points], [row["top1"] for row in points],
                    color=color, marker=marker, markersize=4, label=_label(method))
            # K is recoverable at each measured marker; no unmeasured speedup is asserted.
            for point in points:
                ax.annotate(str(point["K"]), (point["retrieval_batch_ms"], point["top1"]),
                            xytext=(3, 4), textcoords="offset points", fontsize=5.5, color=color)
        _parent_line(ax, selected)
        parents = {(row["full_parent_batch_ms"], row["full_parent_top1"]) for row in selected
                   if row.get("full_parent_timing_cohort") == cohort
                   and _finite(row.get("full_parent_batch_ms")) and _finite(row.get("full_parent_top1"))}
        for index, (milliseconds, top1) in enumerate(sorted(parents)):
            ax.plot(milliseconds, top1, "X", color=INK_MUTED, markersize=8,
                    label="Full representation (measured search)" if index == 0 else None)
        ax.set_xscale("log")
        ax.set_xlabel("Warm resident search time per batch (ms)")
        ax.set_ylabel("1-NN accuracy (%)")
        first = selected[0]
        ax.set_title(f"{_backbone_label(backbone)} | batch {first.get('query_batch_size')}\n"
                     f"{first.get('device_name') or 'GPU unspecified'} | cohort {cohort}", loc="left", fontsize=8.5)
    fig.suptitle("Retrieval accuracy / search time (marker labels: K; excludes encoding)", x=.01, ha="left", fontsize=10)
    _figure_legend(fig, axes)
    fig.tight_layout(rect=(0, .10, 1, .92))
    return _save(plt, fig, target, "top1_vs_retrieval_ms")


def compute_rows(rows):
    """Average only identical backbone sets at a fixed inference K."""
    measured = [row for row in rows if _finite(row.get("training_gpu_hours"))
                and row["training_gpu_hours"] > 0 and _finite(row.get("top1"))]
    result = [{"method": row["method"], "backbone_set": row["backbone"], "K": row["K"],
               "top1": row["top1"], "training_gpu_hours": row["training_gpu_hours"],
               "aggregation": "single_backbone", "backbone_count": 1,
               "training_time_scope": row.get("training_time_scope"),
               "training_device_name": row.get("training_device_name")}
              for row in measured]
    backbones = tuple(name for name in BACKBONES if any(row["backbone"] == name for row in rows))
    if len(backbones) < 2:
        return result
    groups = defaultdict(dict)
    for row in measured:
        groups[(row["method"], row["K"])][row["backbone"]] = row
    for (method, k), points in sorted(groups.items()):
        if set(points) != set(backbones):
            continue
        scopes = {points[name].get("training_time_scope") for name in backbones}
        devices = {points[name].get("training_device_name") for name in backbones}
        if len(scopes) != 1 or len(devices) != 1:
            continue
        result.append({"method": method, "backbone_set": "+".join(backbones), "K": k,
                       "top1": sum(points[name]["top1"] for name in backbones) / len(backbones),
                       "training_gpu_hours": sum(points[name]["training_gpu_hours"] for name in backbones),
                       "aggregation": "macro_accuracy_sum_compute", "backbone_count": len(backbones),
                       "training_time_scope": next(iter(scopes)), "training_device_name": next(iter(devices))})
    return result


def plot_compute_tradeoff(plt, rows, target):
    figures = []
    for backbone_set in sorted({row["backbone_set"] for row in rows}):
        selected = [row for row in rows if row["backbone_set"] == backbone_set]
        budgets = sorted({row["K"] for row in selected})
        fig, axes = _grid(plt, len(budgets))
        average = "+" in backbone_set
        for ax, k in zip(axes, budgets):
            points = [row for row in selected if row["K"] == k]
            for point in points:
                color, marker = _style(point["method"])
                ax.scatter(point["training_gpu_hours"], point["top1"], color=color, marker=marker,
                           s=55, edgecolor=SURFACE, linewidth=.5, label=_label(point["method"]))
            ax.set_xlabel("Sum of training GPU-hours" if average else "Training GPU-hours")
            ax.set_ylabel("Mean 1-NN accuracy (%)" if average else "1-NN accuracy (%)")
            ax.set_xlim(left=0)
            ax.set_title(f"Inference K = {k}", loc="left")
        fig.suptitle(f"{_backbone_label(backbone_set)} | one fit per backbone, evaluated at every K",
                     x=.01, ha="left", fontsize=10)
        _figure_legend(fig, axes)
        fig.tight_layout(rect=(0, .07, 1, .94))
        figures += _save(plt, fig, target, f"top1_vs_gpu_hours_{backbone_set.replace('+', '_and_')}")
    return figures


def plot_metric_panels(plt, group, target, stem, panels, title):
    """panels: [(log key, y label, transform or None, (ymin, ymax) or None)].

    One row per panel, one column per backbone. Methods whose curves coincide
    exactly (for example a support that never shrinks) are named in the panel,
    since the last-drawn line would otherwise hide them.
    """
    backbones = [name for name in BACKBONES if any(name in trial["backbones"] for trial in group["trials"].values())]
    methods = _method_order(group["trials"])
    available = [panel for panel in panels if any(_series(trial["backbones"][b]["log"], panel[0])[0]
                 for trial in group["trials"].values() for b in trial["backbones"])]
    if not backbones or not available:
        return []
    fig, axes = plt.subplots(len(available), len(backbones), squeeze=False,
                             figsize=(3.6 * len(backbones), 2.4 * len(available)), sharex="col")
    for row_index, (key, ylabel, transform, limits) in enumerate(available):
        for column, backbone in enumerate(backbones):
            ax = axes[row_index][column]
            drawn = {}
            for method in methods:
                entry = group["trials"][method]["backbones"].get(backbone)
                if entry is None:
                    continue
                x, y = _series(entry["log"], key)
                if not x:
                    continue
                if transform:
                    y = [transform(value) for value in y]
                drawn.setdefault((tuple(x), tuple(y)), []).append(method)
                color, marker = _style(method)
                # Sparse markers are the secondary (non-color) encoding of identity.
                ax.plot(x, y, color=color, marker=marker, markevery=max(1, len(x) // 8), markersize=4,
                        markeredgecolor=SURFACE, markeredgewidth=.8, label=_label(method))
                if entry["status"] == "failed":
                    ax.plot(x[-1], y[-1], marker="x", color=INK, markersize=6)
            overlaps = [names for names in drawn.values() if len(names) > 1]
            if overlaps:
                plotted = sum(len(names) for names in drawn.values())
                notes = ["all methods identical" if len(names) == plotted else
                         textwrap.fill(f"identical ({len(names)}): " + ", ".join(_label(name) for name in names), 34)
                         for names in overlaps]
                ax.text(.03, .45, "\n".join(notes), transform=ax.transAxes, fontsize=6.5,
                        color=INK_MUTED, va="center", linespacing=1.3)
            if limits:
                low, high = limits
                current_low, current_high = ax.get_ylim()
                ax.set_ylim(current_low if low is None else low,
                            max(current_high, high) if high is not None else current_high)
            if row_index == 0:
                ax.set_title(backbone.replace("resnet", "ResNet-"), loc="left")
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == len(available) - 1:
                ax.set_xlabel("Update")
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes.flat:
        for handle, name in zip(*ax.get_legend_handles_labels()):
            if name not in labels:
                handles.append(handle)
                labels.append(name)
    fig.suptitle(title, x=.01, ha="left", fontsize=10)
    fig.legend(handles, labels, ncol=min(len(labels), 6), loc="lower center", bbox_to_anchor=(.5, -.04), frameon=False)
    fig.tight_layout(rect=(0, .04, 1, .97))
    return _save(plt, fig, target, stem)


def report(output_dir, reports_dir=None):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir) if reports_dir else output_dir / "reports"
    groups = collect(output_dir)
    if not groups:
        print(f"MMRSAE report: no trials under {output_dir}", flush=True)
        return {}
    plt = _pyplot()
    index = {}
    for label, group in groups.items():
        target = reports_dir / label
        target.mkdir(parents=True, exist_ok=True)
        rows = result_rows(group)
        per_budget = budget_rows(group)
        represented = {(row["method"], row["backbone"]) for row in per_budget}
        table_rows = per_budget + [dict(row, budget_evaluation="single_k") for row in rows
                                  if (row["method"], row["backbone"]) not in represented]
        write_tables(table_rows, target, group["settings"])
        compute = compute_rows(per_budget)
        compute_fields = list(compute[0]) if compute else ["method", "backbone_set", "K", "top1", "training_gpu_hours"]
        with (target / "compute_tradeoff.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=compute_fields)
            writer.writeheader()
            writer.writerows(compute)
        settings = group["settings"]
        subtitle = (f"K={settings.get('k')} · {settings.get('evaluation')} · "
                    f"train {settings.get('train_size') or 'all'} rows")
        k_limit = (settings.get("k") or 8) + .5
        figures = plot_top1(plt, rows, target, f"Endpoint retrieval · {subtitle}")
        figures += plot_accuracy_by_k(plt, per_budget, target,
            f"Accuracy across inference budgets (one checkpoint per method; train K={settings.get('k')})")
        figures += plot_backbone_curves(plt, per_budget, target)
        figures += plot_classification_by_k(plt, per_budget, target)
        figures += plot_active_dimensions(plt, per_budget, target)
        figures += plot_retrieval_tradeoff(plt, per_budget, target)
        figures += plot_compute_tradeoff(plt, compute, target)
        figures += plot_metric_panels(plt, group, target, "training_loss",
            [("loss_total", "Total loss (method-specific objective)", None, None),
             ("mse_final_K", "Reconstruction MSE (K)", None, None),
             ("loss_context_unweighted", "Context relation KL (unweighted)", None, None),
             ("weighted_context", "Weighted context loss", None, None)],
            f"Training objective · {subtitle}")
        figures += plot_metric_panels(plt, group, target, "dictionary_and_schedule",
            [("dictionary_blend", "Dictionary blend", None, (0, 1)),
             ("training_k", "Training budget K", None, (0, k_limit))],
            f"Dictionary contribution and K schedule · {subtitle}")
        figures += plot_metric_panels(plt, group, target, "support_health",
            [("actual_nnz_mean", "Mean positive coefficients", None, (0, k_limit)),
             ("final_nnz_min", "Min positive coefficients", None, (0, k_limit)),
             ("final_zero_code_fraction", "All-zero codes (% of batch)", lambda value: 100 * value, (0, 1))],
            f"Sparse support health · {subtitle}")
        coverage = {
            "retrieval_accuracy_points": len(per_budget),
            "classification_accuracy_points": sum(_finite(row.get("classification_top1")) for row in per_budget),
            "retrieval_active_dimension_points": sum(_finite(row.get("actual_nnz_mean")) for row in per_budget),
            "classification_active_dimension_points": sum(_finite(row.get("classification_actual_nnz_mean"))
                                                          and _finite(row.get("classification_top1")) for row in per_budget),
            "warm_search_timing_points": sum(_finite(row.get("retrieval_batch_ms")) for row in per_budget),
            "training_compute_points": sum(row["aggregation"] == "single_backbone" for row in compute),
            "macro_average_compute_points": sum(row["aggregation"] == "macro_accuracy_sum_compute" for row in compute),
            "hardware": group.get("hardware", {}),
            "missing_data_policy": "missing measurements are omitted, never filled with zero or paper values",
            "classification_metric": "supervised linear-probe Top-1; independent from 1-NN retrieval",
            "active_dimensions": "mean positive coordinates: classification queries / retrieval gallery",
            "timing_scope": "warm resident CUDA search only; mean batch ms from median repeat totals; excludes encoding",
            "compute_scope": "measured SAE training GPU-hours, excluding backbone pretraining, extraction, probes and retrieval",
            "macro_average_policy": "same inference K and entire observed backbone set; macro accuracy, sum of GPU-hours",
        }
        atomic_json(target / "plot_coverage.json", coverage)
        overview = ["# Plot overview", "", "All points come from local measurements; missing measurements remain missing.", "",
                    "| Measurement | Available points |", "|---|---:|"]
        overview += [f"| {name.replace('_', ' ')} | {value} |" for name, value in coverage.items() if name.endswith("_points")]
        overview += ["", "The K curves reuse one checkpoint per method and backbone. K is an inference budget, "
                     "while active dimensions are measured nonzero counts. The linear probe is trained separately at each K.", "",
                     "Training cost excludes the optional supervised probe. Averaged compute plots use identical backbone sets "
                     "and identical K, with summed GPU-hours. Warm search figures exclude feature encoding and split incompatible timing cohorts.", "",
                     "See [all measurements](results.csv), [compute aggregation](compute_tradeoff.csv), "
                     "and [coverage and definitions](plot_coverage.json).", "", "## Figures", ""]
        overview += [f"- [{name}]({name})" for name in figures if name.endswith(".png")]
        (target / "overview.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
        index[label] = {"directory": str(target), "methods": _method_order(group["trials"]),
                        "settings": settings, "evaluation_topk": sorted({row["K"] for row in per_budget}),
                        "tables": ["results.csv", "results.md", "compute_tradeoff.csv"],
                        "coverage": coverage, "overview": "overview.md", "figures": figures}
        print(f"MMRSAE report: {target} ({len(group['trials'])} methods, {len(figures)} figure files)", flush=True)
    atomic_json(reports_dir / "index.json", index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "mmrsae",
                        help="MMRSAE output directory containing <trial>/trial.json")
    parser.add_argument("--reports-dir", type=Path, help="Default: <output-dir>/reports")
    args = parser.parse_args(argv)
    report(args.output_dir, args.reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
