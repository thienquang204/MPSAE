#!/usr/bin/env python3
"""Shared, headless reporting for the Matryoshka experiment runners.

The training scripts keep their hot paths focused on PyTorch.  This module is
called after benchmarking to turn their append-safe JSONL histories into
stable CSV/JSON records and publication-ready plots.  "Impact" below means a
loss component's measured contribution to the optimized objective; it is not
presented as a causal ablation result.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


METHODS = ("mrl", "mmpot")


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_history(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL and keep the last record for each epoch after a resume."""
    if not path.is_file():
        return []
    by_epoch: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                epoch = int(record["epoch"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid history record {path}:{line_number}: {exc}") from exc
            by_epoch[epoch] = record
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _history_row(method: str, record: Mapping[str, Any], ot_weight: float) -> Dict[str, Any]:
    train = record.get("train", {})
    head = record.get("head", {})
    ot_loss = _finite(train.get("ot_loss"))
    return {
        "method": method,
        "epoch": record.get("epoch"),
        "learning_rate": record.get("lr"),
        "objective_loss": train.get("loss"),
        "mrl_classification_loss": train.get("mrl_loss"),
        "mmpot_loss_unweighted": train.get("ot_loss"),
        "mmpot_loss_weight": ot_weight if method == "mmpot" else 0.0,
        "mmpot_loss_weighted": (ot_weight * ot_loss) if method == "mmpot" and ot_loss is not None else 0.0,
        "ot_transport_mass": train.get("ot_mass"),
        "ot_capacity_violation": train.get("ot_cap_violation"),
        "training_samples": train.get("samples"),
        "epoch_seconds": train.get("seconds"),
        "samples_per_second": train.get("samples_per_second"),
        "validation_mean_top1": head.get("mean_top1"),
        "validation_full_top1": head.get("full_top1"),
        "selection_metric": record.get("selection_metric"),
        "selection_value": record.get("selected_value"),
        "best_selection_value": record.get("best_metric"),
    }


def _component_impact(
    method: str,
    component: str,
    values: Sequence[float],
    objectives: Sequence[float],
) -> Dict[str, Any]:
    initial, final = values[0], values[-1]
    decrease = initial - final
    contributions = [
        100.0 * value / objective
        for value, objective in zip(values, objectives)
        if abs(objective) > 1e-12
    ]
    return {
        "method": method,
        "loss_component": component,
        "epochs_recorded": len(values),
        "initial_value": initial,
        "final_value": final,
        "minimum_value": min(values),
        "maximum_value": max(values),
        "absolute_decrease": decrease,
        "percent_decrease_from_initial": 100.0 * decrease / abs(initial) if abs(initial) > 1e-12 else None,
        "decreased_from_initial": final < initial,
        "final_objective_contribution_percent": contributions[-1] if contributions else None,
        "mean_objective_contribution_percent": sum(contributions) / len(contributions) if contributions else None,
    }


def _configure_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: Any, output_dir: Path, stem: str) -> List[str]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=400, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return [path.name for path in paths]


def _plot_training(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    prefix: str,
    ot_weight: float,
) -> List[str]:
    if not any(histories.values()):
        return []
    _configure_plot_style()
    import matplotlib.pyplot as plt

    colors = {"mrl": "#0072B2", "mmpot": "#D55E00"}
    artifacts: List[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True)
    for method, history in histories.items():
        if not history:
            continue
        epochs = [int(record["epoch"]) for record in history]
        objective = [float(record["train"]["loss"]) for record in history]
        mrl = [float(record["train"]["mrl_loss"]) for record in history]
        axes[0].plot(epochs, objective, marker="o", color=colors[method], label=f"{method.upper()} objective")
        axes[1].plot(epochs, mrl, marker="o", color=colors[method], label=f"{method.upper()} classification")
        if method == "mmpot":
            weighted_ot = [ot_weight * float(record["train"]["ot_loss"]) for record in history]
            axes[1].plot(epochs, weighted_ot, marker="s", linestyle="--", color="#CC79A7", label=f"{ot_weight:g} x MMPOT")
    axes[0].set_title("(a) Optimized objective", loc="left", fontweight="bold")
    axes[1].set_title("(b) Objective components", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean training loss")
        axis.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
        axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    artifacts.extend(_save_figure(fig, output_dir, f"{prefix}_training_loss_curves"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), constrained_layout=True)
    for method, history in histories.items():
        if not history:
            continue
        epochs = [int(record["epoch"]) for record in history]
        color = colors[method]
        axes[0, 0].plot(epochs, [record["train"]["loss"] for record in history], marker="o", color=color, label=method.upper())
        axes[0, 1].plot(epochs, [record["head"]["mean_top1"] for record in history], marker="o", color=color, label=f"{method.upper()} mean")
        axes[0, 1].plot(epochs, [record["head"]["full_top1"] for record in history], linestyle="--", color=color, label=f"{method.upper()} full")
        axes[1, 0].plot(epochs, [record["lr"] for record in history], marker="o", color=color, label=method.upper())
        if method == "mmpot":
            axes[1, 1].plot(epochs, [record["train"]["ot_mass"] for record in history], marker="o", color="#009E73", label="Transport mass")
            axes[1, 1].plot(epochs, [record["train"]["ot_cap_violation"] for record in history], marker="s", color="#CC79A7", label="Capacity violation")
    titles = (
        "(a) Training objective",
        "(b) Validation accuracy",
        "(c) Learning-rate schedule",
        "(d) OT constraint diagnostics",
    )
    ylabels = ("Mean loss", "Top-1 accuracy (%)", "Learning rate", "Value")
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
        if axis.lines:
            axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    artifacts.extend(_save_figure(fig, output_dir, f"{prefix}_training_procedure_overview"))
    plt.close(fig)
    return artifacts


def _plot_benchmarks(summary: Mapping[str, Any], output_dir: Path, prefix: str) -> List[str]:
    rows = summary.get("comparison", [])
    top1 = [row for row in rows if row.get("metric") == "top1"]
    if not top1:
        return []
    _configure_plot_style()
    import matplotlib.pyplot as plt

    benchmarks = [name for name in ("head", "knn", "linear") if any(row.get("benchmark") == name for row in top1)]
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(4.1 * len(benchmarks), 3.0), constrained_layout=True, squeeze=False)
    for axis, benchmark in zip(axes[0], benchmarks):
        selected = sorted((row for row in top1 if row["benchmark"] == benchmark), key=lambda row: int(row["dimension"]))
        dims = [int(row["dimension"]) for row in selected]
        axis.plot(dims, [row["mrl"] for row in selected], marker="o", color="#0072B2", label="MRL")
        axis.plot(dims, [row["mmpot"] for row in selected], marker="s", color="#D55E00", label="MMPOT")
        axis.set_xscale("log", base=2)
        axis.set_xticks(dims, [str(dim) for dim in dims], rotation=45)
        axis.set_title(f"{benchmark.capitalize()} top-1", fontweight="bold")
        axis.set_xlabel("Nested dimension")
        axis.set_ylabel("Accuracy (%)")
        axis.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
        axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    artifacts = _save_figure(fig, output_dir, f"{prefix}_benchmark_accuracy_comparison")
    plt.close(fig)
    return artifacts


def generate_matryoshka_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    filename_prefix: str,
) -> Dict[str, Any]:
    """Create loss records and plots for either Matryoshka-MMPOT runner."""
    output_dir = output_dir.expanduser().resolve()
    ot_weight = float(summary.get("config", {}).get("ot_lambda", 0.0))
    histories = {
        method: _load_history(output_dir / method / "history.jsonl")
        for method in METHODS
        if (output_dir / method).is_dir()
    }
    combined_rows: List[Dict[str, Any]] = []
    for method, history in histories.items():
        method_rows = [_history_row(method, record, ot_weight) for record in history]
        combined_rows.extend(method_rows)
        _write_csv(method_rows, output_dir / method / f"{method}_training_history.csv")

    history_path = output_dir / f"{filename_prefix}_training_history.csv"
    _write_csv(combined_rows, history_path)

    probe_rows: List[Dict[str, Any]] = []
    for method in METHODS:
        probe_history = (
            summary.get("methods", {})
            .get(method, {})
            .get("linear", {})
            .get("training_history", [])
        )
        probe_rows.extend(
            {
                "method": method,
                "epoch": row.get("epoch"),
                "linear_probe_loss": row.get("loss"),
                "learning_rate": row.get("learning_rate"),
            }
            for row in probe_history
        )
    probe_history_path = output_dir / f"{filename_prefix}_linear_probe_history.csv"
    _write_csv(probe_rows, probe_history_path)

    impact_rows: List[Dict[str, Any]] = []
    for method, history in histories.items():
        if not history:
            continue
        objectives = [float(record["train"]["loss"]) for record in history]
        components = {
            "optimized_objective": objectives,
            "mrl_classification": [float(record["train"]["mrl_loss"]) for record in history],
        }
        if method == "mmpot":
            components["mmpot_weighted_objective_contribution"] = [
                ot_weight * float(record["train"]["ot_loss"]) for record in history
            ]
        impact_rows.extend(
            _component_impact(method, component, values, objectives)
            for component, values in components.items()
        )
    for method in METHODS:
        values = [
            float(row["linear_probe_loss"])
            for row in probe_rows
            if row["method"] == method
        ]
        if values:
            impact_rows.append(
                _component_impact(
                    f"{method}_linear_probe",
                    "cross_entropy",
                    values,
                    values,
                )
            )

    impact_csv = output_dir / f"{filename_prefix}_loss_component_impact.csv"
    impact_json = output_dir / f"{filename_prefix}_loss_component_impact.json"
    _write_csv(impact_rows, impact_csv)
    _atomic_json(
        {
            "impact_definition": "Measured contribution to the optimized training objective; not a causal ablation estimate.",
            "mmpot_loss_weight": ot_weight,
            "components": impact_rows,
        },
        impact_json,
    )

    plot_files = _plot_training(histories, output_dir, filename_prefix, ot_weight)
    plot_files.extend(_plot_benchmarks(summary, output_dir, filename_prefix))
    artifacts = [
        path.name
        for path in (history_path, probe_history_path, impact_csv, impact_json)
        if path.is_file()
    ] + plot_files
    report = {
        "history_epochs": {method: len(history) for method, history in histories.items()},
        "loss_impact_definition": "Measured contribution to the optimized training objective; not a causal ablation estimate.",
        "artifacts": artifacts,
    }
    _atomic_json(report, output_dir / f"{filename_prefix}_report_manifest.json")
    report["artifacts"].append(f"{filename_prefix}_report_manifest.json")
    return report
