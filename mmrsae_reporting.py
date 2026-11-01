#!/usr/bin/env python3
"""Tables and figures for every MMRSAE ImageNet output directory.

Trials are grouped by everything that defines a comparable experiment except
the method (settings without ``variant``, plus the implementation identity), so
a figure never mixes K, data sizes, evaluation pools or code versions. Each
group gets ``reports/<group>/`` with results.csv/results.md and PNG+PDF figures:
endpoint top-1 by method, training loss, and sparse-support health. Failed or
partial fits are reported from whatever they wrote; nothing here is trained,
evaluated or downloaded, and it runs on CPU after any launcher invocation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
import sys

from repair_data import atomic_json, canonical_hash

# Fixed method order and color slots: color follows the method, never its rank.
METHODS = ("mmrsae", "mmrsae-rel", "mmrsae-wide", "mmrsae-annealed", "mpsae", "csr")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
MARKERS = ("o", "s", "^", "D", "v", "P")
OTHER_COLOR = "#8c8b84"
BACKBONES = ("resnet50", "resnet18")
INK, INK_MUTED, GRID, SURFACE = "#1f1e1b", "#5f5e58", "#e4e3de", "#fcfcfb"


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
        key = canonical_hash({"settings": settings, "code": trial.get("code")})
        label = f"{settings.get('evaluation', 'dev')}_k{settings.get('k', '?')}_{key[:8]}"
        group = groups.setdefault(label, {"label": label, "settings": settings, "trials": {}})
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
            retrieval = summary.get("retrieval") or {}
            parent = summary.get("full_parent") or {}
            failure = entry["failure"] or {}
            rows.append({
                "method": method, "backbone": backbone, "status": entry["status"],
                "top1": retrieval.get("top1"), "correct": retrieval.get("correct"),
                "queries": retrieval.get("total"), "actual_nnz_mean": retrieval.get("actual_nnz_mean"),
                "empty_query_codes": retrieval.get("empty_query_codes"),
                "empty_query_fraction": retrieval.get("empty_query_fraction"),
                "empty_gallery_codes": retrieval.get("empty_gallery_codes"),
                "full_parent_top1": parent.get("top1"),
                "updates": summary.get("updates", len(entry["log"])),
                "fit_seconds": summary.get("fit_seconds"),
                "train_max_zero_codes": max((row.get("zero_final_codes", 0) for row in entry["log"]), default=None),
                "train_min_nnz": min((row.get("final_nnz_min") for row in entry["log"]
                                      if row.get("final_nnz_min") is not None), default=None),
                "failure": failure.get("error"), "directory": trial["directory"],
            })
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
    columns = [("method", "Method"), ("backbone", "Backbone"), ("status", "Status"), ("top1", "Top-1 (%)"),
               ("actual_nnz_mean", "Gallery nnz"), ("empty_query_codes", "Empty queries"),
               ("empty_gallery_codes", "Empty gallery"), ("train_max_zero_codes", "Max zero codes/batch"),
               ("full_parent_top1", "Full parent top-1"), ("updates", "Updates"), ("fit_seconds", "Fit (s)")]
    lines = [f"# MMRSAE results: {target.name}", "",
             f"K={settings.get('k')}, evaluation={settings.get('evaluation')}, "
             f"train_size={settings.get('train_size')}, dev_size={settings.get('dev_size')}, "
             f"max_steps={settings.get('max_steps')}, epochs={settings.get('epochs')}, seed={settings.get('seed')}.",
             "Empty (all-zero) queries are counted as retrieval misses.", "",
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
    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * len(methods) * len(backbones)), 3.4))
    width = .8 / len(methods)
    top = max([row["top1"] for row in rows if row["top1"] is not None] + [1.])
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
                   label=method if position == 0 else None)
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
                        markeredgecolor=SURFACE, markeredgewidth=.8, label=method)
                if entry["status"] == "failed":
                    ax.plot(x[-1], y[-1], marker="x", color=INK, markersize=6)
            overlaps = [names for names in drawn.values() if len(names) > 1]
            if overlaps:
                plotted = sum(len(names) for names in drawn.values())
                notes = ["all methods identical" if len(names) == plotted else
                         textwrap.fill(f"identical ({len(names)}): " + ", ".join(names), 34)
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
        write_tables(rows, target, group["settings"])
        settings = group["settings"]
        subtitle = (f"K={settings.get('k')} · {settings.get('evaluation')} · "
                    f"train {settings.get('train_size') or 'all'} rows")
        k_limit = (settings.get("k") or 8) + .5
        figures = plot_top1(plt, rows, target, f"Endpoint retrieval · {subtitle}")
        figures += plot_metric_panels(plt, group, target, "training_loss",
            [("loss_total", "Total loss (method-specific objective)", None, None),
             ("mse_final_K", "Reconstruction MSE (K)", None, None)],
            f"Training objective · {subtitle}")
        figures += plot_metric_panels(plt, group, target, "support_health",
            [("actual_nnz_mean", "Mean positive coefficients", None, (0, k_limit)),
             ("final_nnz_min", "Min positive coefficients", None, (0, k_limit)),
             ("final_zero_code_fraction", "All-zero codes (% of batch)", lambda value: 100 * value, (0, 1))],
            f"Sparse support health · {subtitle}")
        index[label] = {"directory": str(target), "methods": _method_order(group["trials"]),
                        "settings": settings, "tables": ["results.csv", "results.md"], "figures": figures}
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
