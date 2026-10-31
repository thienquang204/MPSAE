"""Task-specific storage/accuracy plots generated during an explicit run only.

Matplotlib is imported lazily and uses its noninteractive backend. Plot inputs
are completed three-seed summary rows; this module never reruns any model.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


METHODS = {
    "matryoshka": ("MRL prefixes", "#0072B2", "o"),
    "csr": ("CSR", "#D55E00", "s"),
    "mpsae": ("MPSAE", "#009E73", "^"),
    "csrv2": ("CSRv2", "#CC79A7", "D"),
    "mpsaev2": ("MPSAEv2", "#E69F00", "v"),
}
PARENTS = {
    "mrl_full": ("MRL full parent", "#0072B2", "--"),
    "timm_full": ("timm full parent", "#555555", ":"),
}


def _metric_specs(task):
    if task == "detection":
        return [
            ("AP50", "AP at IoU 0.50 (%)", 100., "primary", "higher_is_better"),
            ("AP75", "AP at IoU 0.75 (%)", 100., "primary", "higher_is_better"),
        ]
    if task == "mask":
        return [(condition + ".macro_IoU", "Macro foreground IoU (%)", 100., condition,
                 "higher_is_better") for condition in ("clean", "translated")]
    if task == "pose":
        return [(condition + "." + metric, label, scale, condition, direction)
                for condition in ("clean", "translated")
                for metric, label, scale, direction in (
                    ("macro_Acc30", "Macro accuracy within 30 degrees (%)", 100., "higher_is_better"),
                    ("macro_MedErr", "Mean of category median errors (degrees)", 1., "lower_is_better"))]
    raise ValueError(f"Unsupported plotting task: {task}")


def _number(row, key, positive=False):
    value = float(row[key])
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid plot input {key} for {row['method']}: {value}")
    return value


def write_transfer_plots(summaries, output_dir):
    """Write atomic PNG/PDF figures under plots/ and return their provenance."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for row in summaries:
        key = (row["task"], row["architecture"], row["head_family"])
        groups.setdefault(key, []).append(row)
    artifacts = []
    for (task, architecture, family), rows in sorted(groups.items()):
        full = {row["method"]: row for row in rows if row["method"] in PARENTS}
        if set(full) != set(PARENTS):
            raise ValueError(f"Both full-parent references are required for {task}/{architecture}/{family}")
        for metric, ylabel, scale, condition, direction in _metric_specs(task):
            figure, axis = plt.subplots(figsize=(8.6, 6.2))
            try:
                for method, (label, color, marker) in METHODS.items():
                    points = sorted((row for row in rows if row["method"] == method),
                                    key=lambda row: _number(row, "stored_bytes_per_region", positive=True))
                    if not points:
                        continue
                    x = [_number(row, "stored_bytes_per_region", positive=True) for row in points]
                    y = [_number(row, metric + ".mean") * scale for row in points]
                    errors = [_number(row, metric + ".sample_std") * scale for row in points]
                    if any(error < 0 for error in errors):
                        raise ValueError("Plot standard deviations must be nonnegative")
                    axis.errorbar(x, y, yerr=errors, label=label, color=color, marker=marker,
                                  linewidth=1.6, markersize=5, capsize=3, elinewidth=1.0)
                for method, (label, color, style) in PARENTS.items():
                    row = full[method]
                    x = _number(row, "stored_bytes_per_region", positive=True)
                    y = _number(row, metric + ".mean") * scale
                    error = _number(row, metric + ".sample_std") * scale
                    if error < 0:
                        raise ValueError("Parent standard deviations must be nonnegative")
                    axis.axhline(y, color=color, linestyle=style, linewidth=1.2,
                                 label=f"{label} ({x:g} B)")
                    axis.errorbar([x], [y], yerr=[error], color=color, marker="X", markersize=7,
                                  linestyle="none", capsize=3, label="_nolegend_")
                axis.set_xscale("log", base=2)
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                axis.set_xlabel("Representation bytes per region (log scale)")
                axis.set_ylabel(ylabel)
                task_title = {"detection": "VOC2007 detection", "mask": "VOC2012 supplied-box masks",
                              "pose": "PASCAL3D+ known-box, known-category viewpoint"}[task]
                condition_label = {"primary": "refined boxes", "clean": "clean crop",
                                   "translated": "mean of four translated crops"}[condition]
                family_label = "affine head" if family == "affine" else "width-128 MLP"
                title = f"{architecture.upper()} | {task_title}\n{family_label}; {condition_label}"
                if task == "pose":
                    title += "\nPrimary subset: non-occluded, non-truncated objects"
                axis.set_title(title, fontsize=10)
                axis.grid(True, which="major", alpha=.25)
                axis.legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncol=3,
                            fontsize=8, frameon=False)
                figure.text(.5, .025,
                            "Mean +/- sample SD across three paired seeds. "
                            + ("Lower error is better.\n" if direction == "lower_is_better" else "Higher is better.\n")
                            + "Sparse bytes include padding, row counts and cache metadata; "
                            "dense bytes describe the FP32 prefix.\n"
                            + "Extraction-cache disk usage and head parameter counts are reported separately.",
                            ha="center", va="bottom", fontsize=7)
                figure.subplots_adjust(left=.11, right=.98, top=.88, bottom=.32)
                stem = f"{task}_{architecture}_{family}_{metric.replace('.', '_')}_by_bytes"
                files = []
                for extension in ("png", "pdf"):
                    destination = plot_dir / f"{stem}.{extension}"
                    temporary = destination.with_name(stem + ".tmp." + extension)
                    figure.savefig(temporary, dpi=180, format=extension)
                    os.replace(temporary, destination)
                    files.append(str(destination.relative_to(output_dir)))
                artifacts.append({"task": task, "architecture": architecture, "head_family": family,
                                  "metric": metric, "condition": condition, "metric_direction": direction,
                                  "evaluation_subset": "non_occluded_non_truncated" if task == "pose" else "all_eligible",
                                  "display_units": "degrees" if scale == 1. else "percent",
                                  "files": files})
            finally:
                plt.close(figure)
    return {"plots": artifacts, "error_bars": "sample_standard_deviation_over_three_paired_seeds",
            "storage_axis": "dense_fp32_prefix_payload_or_measured_padded_sparse_cache_bytes_per_region",
            "parent_references": "separate_native_MRL_and_timm_full_vectors",
            "plotting_version": matplotlib.__version__}
