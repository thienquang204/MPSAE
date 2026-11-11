"""Tables and figures for explicitly selected frozen-transfer evaluations.

The campaign supplies the records: this module never searches a workspace for
results. A table row is one dataset/protocol/backbone/mode/training-budget/
inference-budget/method condition, averaged over training seeds. Dataset or
feature revisions can be isolated with ``comparison_id`` (or ``protocol_id``).
Metrics are fractions in JSON/CSV and percentages only on figure axes. DINO
kNN classification and ranked retrieval remain separate tasks throughout.

Only the standard library is imported until figures are requested. The JSON
and CSV artefacts remain available when matplotlib cannot be imported.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..config import IMPLEMENTED_METHODS, METHODS, SEEDS
from ..io import atomic_json, canonical_hash
from ..logs import get_logger

FORMAT = "mpsae-frozen-transfer-report-v2"
ACCURACY_METRICS = (
    "recall_at_1", "recall_at_10", "recall_at_100", "precision_at_10", "map_at_r",
    "macro_recall_at_1", "macro_recall_at_10", "macro_recall_at_100",
    "macro_precision_at_10", "macro_map_at_r",
)
MEASUREMENT_METRICS = (
    "nnz_mean", "query_nnz_mean", "gallery_nnz_mean", "empty_queries",
    "query_count", "gallery_count", "search_seconds", "encode_seconds",
    "code_bytes_per_image", "index_bytes", "search_ms_per_query", "voting_seconds",
)
METRICS = ACCURACY_METRICS + MEASUREMENT_METRICS
PAIRS = (("mpsae", "csr"), ("mpsae", "csrv2"))
_SCOPE = ("dataset", "protocol", "comparison_id", "backbone", "mode", "train_k", "eval_k",
          "task", "classifier_signature")
_ALIASES = {"queries": "query_count", "gallery_rows": "gallery_count",
            "seconds": "search_seconds"}
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")
_KNN_METRIC = re.compile(r"knn_([1-9][0-9]*)_(top1|top5|balanced_accuracy|macro_f1)$")


def _classification_metadata(record: dict[str, Any]) -> None:
    """Make task and voting settings explicit before grouping any results."""
    task = record.setdefault("task", "retrieval")
    if task not in ("retrieval", "knn_classification"):
        raise ValueError(f"unknown benchmark task {task!r}")
    classifier = record.get("classifier")
    if task == "retrieval":
        if classifier:
            raise ValueError("retrieval record must not carry a classifier")
        record["classifier_signature"] = ""
        return
    if not isinstance(classifier, Mapping) or classifier.get("name") != "dino_weighted_knn":
        raise ValueError("classification record needs a dino_weighted_knn classifier")
    classifier = dict(classifier)
    neighbours = classifier.get("n_neighbors")
    if (not isinstance(neighbours, (tuple, list)) or not neighbours
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in neighbours)
            or len(set(neighbours)) != len(neighbours)):
        raise ValueError("classifier.n_neighbors must contain distinct positive integers")
    temperature = classifier.get("temperature")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("classifier.temperature must be finite and positive")
    classifier.update(n_neighbors=sorted(neighbours), temperature=float(temperature))
    record["classifier"] = classifier
    # Never trust a caller's comparison_id to identify the voting protocol.
    record["classifier_signature"] = canonical_hash(classifier)


def _metric_names(records: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    dynamic = {name for record in records for name in record.get("metrics", {})
               if _KNN_METRIC.fullmatch(name)}
    return METRICS + tuple(sorted(dynamic))


def _metric_values(record: Mapping[str, Any]) -> dict[str, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("benchmark record needs a metrics mapping")
    values = dict(metrics)
    for old, new in _ALIASES.items():
        if new not in values and old in values:
            values[new] = values[old]
    clean: dict[str, float] = {}
    classification = record.get("task") == "knn_classification"
    neighbours = record.get("classifier", {}).get("n_neighbors", []) if classification else []
    if classification:
        classifier = record["classifier"]
        for header_key, metric_key in (("n_neighbors", "n_neighbors"),
                                       ("temperature", "temperature"),
                                       ("metric_version", "protocol_version"),
                                       ("vote_rule", "vote_rule"), ("tie_rule", "tie_rule"),
                                       ("class_tie_rule", "class_tie_rule"),
                                       ("empty_query_policy", "empty_query_policy")):
            if (header_key in classifier and metric_key in values
                    and classifier[header_key] != values[metric_key]):
                raise ValueError(f"classifier.{header_key} disagrees with measured metric policy")
    for name in values:
        if name.startswith("knn_"):
            match = _KNN_METRIC.fullmatch(name)
            if not classification or match is None or int(match[1]) not in neighbours:
                raise ValueError(f"metric {name!r} does not match the declared classification task")
    if classification and any(name in values for name in ACCURACY_METRICS):
        raise ValueError("classification results cannot contain retrieval accuracy metrics")
    for name in _metric_names([{"metrics": values}]):
        value = values.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        if (name in ACCURACY_METRICS or _KNN_METRIC.fullmatch(name)) and value > 1:
            raise ValueError(f"{name} must be a fraction in [0, 1], not a percentage")
        clean[name] = value
    if classification:
        if any(f"knn_{n}_top1" not in clean for n in neighbours):
            raise ValueError("classification result needs top1 for every declared neighbor count")
    elif not any(name in clean for name in ACCURACY_METRICS):
        raise ValueError("benchmark record contains no supported retrieval accuracy metric")
    if ("search_ms_per_query" not in clean and "search_seconds" in clean
            and clean.get("query_count", 0) > 0):
        clean["search_ms_per_query"] = 1000 * clean["search_seconds"] / clean["query_count"]
    return clean


def _normalise(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate conditions and reject two checkpoints for the same seed/cell.

    Different seeds necessarily have different checkpoint IDs and are the
    repetitions we average. Two different evaluations *within* the same seed
    and condition are ambiguous and must not silently become extra samples.
    """
    by_id: dict[str, dict[str, Any]] = {}
    conditions: dict[tuple, str] = {}
    sweep_checkpoints: dict[tuple, str] = {}
    incomplete: list[str] = []
    for item in records:
        record = dict(item)
        # Historical experiments may remain in a mixed result collection.
        # Exclude them before validating or reading their retired payloads.
        method = record.get("method")
        if method in IMPLEMENTED_METHODS and method not in METHODS:
            continue
        evaluation_id = record.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise ValueError("each benchmark record needs an evaluation_id")
        if not record.get("complete", True):
            incomplete.append(evaluation_id)
            continue
        for name in ("dataset", "protocol", "backbone", "method"):
            if not isinstance(record.get(name), str) or not record[name]:
                raise ValueError(f"benchmark record {evaluation_id} needs {name}")
        if record["method"] not in (*METHODS, "dense"):
            raise ValueError(f"unknown benchmark method {record['method']!r}")
        record["mode"] = record.get("mode", "frozen")
        if not isinstance(record["mode"], str) or not record["mode"]:
            raise ValueError("mode must be a nonempty protocol name")
        comparison_id = record.get("comparison_id", record.get("protocol_id", ""))
        if not isinstance(comparison_id, str):
            raise TypeError("comparison_id/protocol_id must be a string")
        record["comparison_id"] = comparison_id
        _classification_metadata(record)
        dense = record["method"] == "dense"
        if dense:
            if any(record.get(name) is not None for name in ("seed", "train_k", "eval_k")):
                raise ValueError("dense references have no seed, train_k or eval_k")
            record.update(seed=None, train_k=None, eval_k=None)
        else:
            for name in ("seed", "train_k", "eval_k"):
                value = record.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
                    raise ValueError(f"sparse record needs an integer {name}")
        # Preserve protocol diagnostics and per-class payloads in the raw
        # evaluation table; only supported scalar fields enter aggregates.
        scalar_metrics = _metric_values(record)
        record["metrics"] = {**record["metrics"], **scalar_metrics}
        condition = tuple(record.get(name) for name in _SCOPE) + (record["method"], record["seed"])
        previous_id = conditions.get(condition)
        if previous_id is not None and previous_id != evaluation_id:
            raise ValueError("multiple evaluation/checkpoint IDs for one benchmark condition and seed: "
                             f"{previous_id}, {evaluation_id}")
        if not dense and record.get("checkpoint_id"):
            sweep = tuple(record.get(name) for name in _SCOPE if name != "eval_k") + (
                record["method"], record["seed"])
            previous_checkpoint = sweep_checkpoints.setdefault(sweep, record["checkpoint_id"])
            if previous_checkpoint != record["checkpoint_id"]:
                raise ValueError("inference-K sweep mixes different checkpoints for one training condition")
        if evaluation_id in by_id and canonical_hash(by_id[evaluation_id]) != canonical_hash(record):
            raise ValueError(f"conflicting records for evaluation_id {evaluation_id}")
        conditions[condition] = evaluation_id
        by_id[evaluation_id] = record
    return list(by_id.values()), sorted(set(incomplete))


def _stats(values: Sequence[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    return (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None,
            len(values))


def _sort_key(row: Mapping[str, Any]) -> tuple:
    method_order = {"dense": -1, **{method: index for index, method in enumerate(METHODS)}}
    return tuple(str(row.get(name, "")) for name in _SCOPE[:5]) + (
        int(row.get("train_k") or 0), int(row.get("eval_k") or 0),
        str(row.get("task", "retrieval")), str(row.get("classifier_signature", "")),
        method_order.get(str(row.get("method")), 99), str(row.get("method", "")))


def aggregate_records(records: Sequence[Mapping[str, Any]], *,
                      expected_seeds: Sequence[int] = SEEDS) -> list[dict[str, Any]]:
    """Mean/sample-std/n per scientific condition; dense references appear once."""
    clean, _ = _normalise(records)
    expected = tuple(dict.fromkeys(int(seed) for seed in expected_seeds))
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for record in clean:
        key = tuple(record.get(name) for name in _SCOPE) + (record["method"],)
        groups.setdefault(key, []).append(record)
    rows = []
    for key, members in groups.items():
        row = dict(zip(_SCOPE + ("method",), key))
        if row["task"] == "knn_classification":
            row["classifier"] = dict(members[0]["classifier"])
        dense = row["method"] == "dense"
        present = sorted(member["seed"] for member in members if member["seed"] is not None)
        missing = [] if dense else sorted(set(expected) - set(present))
        row.update(n=len(members), seeds_present=present, seeds_missing=missing,
                   complete=not missing,
                   evaluation_ids=sorted(member["evaluation_id"] for member in members),
                   checkpoint_ids=sorted({str(member.get("checkpoint_id") or member.get("signature"))
                                          for member in members
                                          if member.get("checkpoint_id") or member.get("signature")}))
        for name in _metric_names(members):
            values = [member["metrics"][name] for member in members if name in member["metrics"]]
            if values:
                mean, std, count = _stats(values)
                row.update({f"{name}_mean": mean, f"{name}_std": std, f"{name}_n": count})
        rows.append(row)
    return sorted(rows, key=_sort_key)


def paired_differences(records: Sequence[Mapping[str, Any]], *,
                       expected_seeds: Sequence[int] = SEEDS) -> list[dict[str, Any]]:
    """Per-seed treatment-control differences, never unpaired curve subtraction."""
    clean, _ = _normalise(records)
    expected = set(int(seed) for seed in expected_seeds)
    index: dict[tuple, dict[str, dict[int, Mapping[str, Any]]]] = {}
    for record in clean:
        if record["method"] == "dense":
            continue
        scope = tuple(record[name] for name in _SCOPE)
        index.setdefault(scope, {}).setdefault(record["method"], {})[record["seed"]] = record
    rows = []
    for scope, methods in index.items():
        for treatment, control in PAIRS:
            lhs, rhs = methods.get(treatment, {}), methods.get(control, {})
            paired = sorted(set(lhs) & set(rhs))
            if not paired:
                continue
            row = dict(zip(_SCOPE, scope))
            if row["task"] == "knn_classification":
                row["classifier"] = dict(lhs[paired[0]]["classifier"])
            row.update(treatment=treatment, control=control, seeds_paired=paired,
                       seeds_missing=sorted(expected - set(paired)),
                       seeds_unpaired=sorted(set(lhs) ^ set(rhs)), n=len(paired),
                       evaluation_pairs=[{"seed": seed,
                                          "treatment": lhs[seed]["evaluation_id"],
                                          "control": rhs[seed]["evaluation_id"]} for seed in paired])
            for name in _metric_names([*lhs.values(), *rhs.values()]):
                values = [lhs[seed]["metrics"][name] - rhs[seed]["metrics"][name]
                          for seed in paired
                          if name in lhs[seed]["metrics"] and name in rhs[seed]["metrics"]]
                if values:
                    mean, std, count = _stats(values)
                    row.update({f"delta_{name}_mean": mean, f"delta_{name}_std": std,
                                f"delta_{name}_n": count})
            rows.append(row)
    return sorted(rows, key=lambda row: _sort_key(row) + (row["treatment"], row["control"]))


def _write_table(rows: Sequence[Mapping[str, Any]], directory: Path, name: str) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{name}.json"
    csv_path = directory / f"{name}.csv"
    atomic_json(json_path, list(rows))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict))
                                 else value for key, value in row.items()})
    return {"json": str(json_path), "csv": str(csv_path)}


def _panel_scope(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (row["dataset"], row["protocol"], row["comparison_id"],
            row.get("task", "retrieval"), row.get("classifier_signature", ""))


def _scope_label(scope: tuple[str, ...]) -> str:
    dataset = scope[0]
    return {"eurosat": "EuroSAT", "sop": "SOP", "flowers102": "Flowers102"}.get(dataset, dataset)


def _scope_caption(scopes) -> str:
    return "; ".join(f"{i + 1}: " + ", ".join(scope) for i, scope in enumerate(scopes))


def _curve(ax, rows: Sequence[Mapping[str, Any]], method: str, metric: str, *,
           x: str = "eval_k", delta: bool = False, label: str | None = None,
           scale: float = 100, xscale: float = 1) -> bool:
    from ..report.style import BAND_ALPHA, method_style

    key = f"delta_{metric}" if delta else metric
    points = [(row.get(x) if x == "eval_k" else row.get(f"{x}_mean"),
               row.get(f"{key}_mean"), row.get(f"{key}_std"), row.get(f"{key}_n", 0),
               row.get("eval_k", 0)) for row in rows]
    points = [point for point in points if point[0] is not None and point[1] is not None]
    points.sort(key=lambda point: point[4])
    if not points:
        return False
    style = method_style(method)
    xs = [float(point[0]) * xscale for point in points]
    means = [scale * float(point[1]) for point in points]
    stds = [scale * float(point[2]) if point[2] is not None else None for point in points]
    ax.plot(xs, means, color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], label=label or style["label"])
    if any(std is not None for std in stds):
        ax.fill_between(xs, [value - (std or 0) for value, std in zip(means, stds)],
                        [value + (std or 0) for value, std in zip(means, stds)],
                        where=[std is not None for std in stds],
                        color=style["color"], alpha=BAND_ALPHA, linewidth=0)
    return True


def _draw_quality(rows: Sequence[Mapping[str, Any]], dense: Sequence[Mapping[str, Any]],
                  out_dir: Path, name: str, title: str, *, x: str = "eval_k") -> list[Path]:
    import matplotlib.pyplot as plt
    from ..report.style import PAGE_WIDTH, save_figure, seed_caption
    from ..report.figures import _dedup_legend

    scopes = sorted({_panel_scope(row) for row in rows})
    if not scopes or not any(row.get(x if x == "eval_k" else f"{x}_mean") is not None
                             for row in rows):
        return []
    fig, axes = plt.subplots(len(scopes), 2, figsize=(PAGE_WIDTH, 2.45 * len(scopes) + 0.5),
                             squeeze=False, layout="constrained")
    try:
        for index, scope in enumerate(scopes):
            subset = [row for row in rows if _panel_scope(row) == scope]
            for ax, metric, pretty in zip(axes[index], ("recall_at_1", "map_at_r"),
                                          ("R@1 (%)", "mAP@R (%)")):
                for method in METHODS:
                    _curve(ax, [row for row in subset if row["method"] == method], method, metric, x=x,
                           xscale=1 / 1024 if x == "code_bytes_per_image" else 1)
                ax.set_ylabel(pretty)
                ax.set_title(_scope_label(scope) + (f" ({index + 1})" if
                             sum(item[0] == scope[0] for item in scopes) > 1 else ""), fontsize=9)
                if x == "eval_k":
                    budgets = sorted({int(row["eval_k"]) for row in subset})
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(budgets, labels=[str(k) for k in budgets])
                    ax.set_xlabel("Inference $K$")
                elif x == "nnz_mean":
                    ax.set_xscale("symlog", base=2, linthresh=1)
                    ax.set_xlabel("Mean nonzeros (gallery)")
                else:
                    ax.set_xscale("symlog", base=2, linthresh=1 / 1024)
                    ax.set_xlabel("Code size (KiB/image)")
        _dedup_legend(fig, list(axes.flat), ncol=3 if name.startswith("paired_") else 6)
        fig.suptitle(title)
        caption = (
            f"{title}. Rows keep datasets and retrieval protocols separate. "
            f"{seed_caption([seed for row in rows for seed in row['seeds_present']])} "
            "Per-point n is in the tables. R@1 is the share of queries with a relevant first "
            "neighbour; mAP@R ranks to each query's full relevant-image count. Accuracies are percentages. "
            "Every inference budget uses the same SAE checkpoint at the training K in the title. "
            "Protocols/revisions in row order: " + _scope_caption(scopes)
        )
        return save_figure(fig, out_dir, name, caption=caption)
    finally:
        plt.close(fig)


def _draw_paired(rows: Sequence[Mapping[str, Any]], out_dir: Path,
                 name: str, title: str) -> list[Path]:
    import matplotlib.pyplot as plt
    from ..report.style import PAGE_WIDTH, label_for, save_figure, seed_caption
    from ..report.figures import _dedup_legend

    scopes = sorted({_panel_scope(row) for row in rows})
    if not scopes:
        return []
    fig, axes = plt.subplots(len(scopes), 2, figsize=(PAGE_WIDTH, 2.45 * len(scopes) + 0.5),
                             squeeze=False, layout="constrained")
    try:
        for index, scope in enumerate(scopes):
            subset = [row for row in rows if _panel_scope(row) == scope]
            for ax, metric, pretty in zip(axes[index], ("recall_at_1", "map_at_r"),
                                          ("Δ R@1 (pp)", "Δ mAP@R (pp)")):
                for treatment, control in PAIRS:
                    _curve(ax, [row for row in subset if row["treatment"] == treatment
                                and row["control"] == control], treatment, metric, delta=True,
                           label=f"{label_for(treatment)} − {label_for(control)}")
                ax.axhline(0, color="#444444", linewidth=0.7)
                budgets = sorted({int(row["eval_k"]) for row in subset})
                ax.set_xscale("log", base=2)
                ax.set_xticks(budgets, labels=[str(k) for k in budgets])
                ax.set_xlabel("Inference $K$")
                ax.set_ylabel(pretty)
                ax.set_title(_scope_label(scope) + (f" ({index + 1})" if
                             sum(item[0] == scope[0] for item in scopes) > 1 else ""), fontsize=9)
        _dedup_legend(fig, list(axes.flat), ncol=3 if name.startswith("paired_") else 6)
        fig.suptitle(title)
        return save_figure(fig, out_dir, name, caption=(
            f"{title}. Each difference is formed within the same dataset, protocol, backbone, "
            "training budget, inference budget and training seed before aggregation. Differences "
            "are in percentage points. "
            f"{seed_caption([seed for row in rows for seed in row['seeds_paired']])} Unmatched seeds are excluded "
            "and listed explicitly in the paired table. Protocols/revisions in row order: "
            + _scope_caption(scopes)))
    finally:
        plt.close(fig)


def _draw_costs(rows, out_dir, name, title):
    """Cost vs inference K for the same fixed-training-budget records."""
    import matplotlib.pyplot as plt
    from ..report.style import PAGE_WIDTH, COLUMN_WIDTH, save_figure, seed_caption
    from ..report.figures import _dedup_legend, _log2_k_axis

    scopes = sorted({_panel_scope(row) for row in rows})
    fields = [(field, label, scale) for field, label, scale in
              (("search_ms_per_query", "Search (ms/query)", 1),
               ("code_bytes_per_image", "Code size (KiB/image)", 1 / 1024))
              if any(r.get(f"{field}_mean") is not None for r in rows)]
    if not scopes or not fields:
        return []
    fig, axes = plt.subplots(len(scopes), len(fields), squeeze=False,
                             figsize=(PAGE_WIDTH if len(fields) > 1 else COLUMN_WIDTH,
                                      2.45 * len(scopes) + 0.5), layout="constrained")
    for i, scope in enumerate(scopes):
        subset = [r for r in rows if _panel_scope(r) == scope]
        for ax, (field, label, scale) in zip(axes[i], fields):
            for method in METHODS:
                _curve(ax, [r for r in subset if r["method"] == method], method, field, scale=scale)
            _log2_k_axis(ax, [r["eval_k"] for r in subset], label="Inference $K$")
            ax.set_ylabel(label)
            ax.set_title(_scope_label(scope) + (f" ({i + 1})" if
                         sum(item[0] == scope[0] for item in scopes) > 1 else ""), fontsize=9)
    _dedup_legend(fig, list(axes.flat), ncol=6)
    fig.suptitle(title)
    return save_figure(fig, out_dir, name, caption=(
        "Measured neighbor-search cost at each inference K; the SAE checkpoint is fixed. Search "
        "ms/query is the shared search wall time divided by query count, excluding encoding and voting; "
        "it is an amortised cost, not single-query latency. Code size includes the stored "
        "sparse representation per image. "
        f"{seed_caption([seed for r in rows for seed in r['seeds_present']])} "
        "Protocols/revisions in row order: " + _scope_caption(scopes)))


def _draw_classification(rows, out_dir, name, title, *, kind="accuracy", x="eval_k"):
    """DINO kNN panels keep neighbor counts separate from embedding budgets."""
    import matplotlib.pyplot as plt
    from ..report.style import PAGE_WIDTH, COLUMN_WIDTH, label_for, save_figure, seed_caption
    from ..report.figures import _dedup_legend, _log2_k_axis

    delta = kind == "paired"
    fields = (("balanced_accuracy", "Balanced acc. (%)"), ("macro_f1", "Macro-F1 (%)")) \
        if kind == "balance" else (("top1", "Accuracy (%)" if not delta else "Delta accuracy (pp)"),)
    scopes = sorted({_panel_scope(row) for row in rows})
    panels = []
    for scope in scopes:
        subset = [row for row in rows if _panel_scope(row) == scope]
        classifier = subset[0]["classifier"]
        for neighbours in classifier["n_neighbors"]:
            if any(row.get(f"{'delta_' if delta else ''}knn_{neighbours}_{metric}_mean") is not None
                   for row in subset for metric, _ in fields):
                panels.append((scope, neighbours, classifier, subset))
    if not panels:
        return []
    per_page = 4 // len(fields)
    pages = [panels[start:start + per_page] for start in range(0, len(panels), per_page)]
    paths = []
    for page_index, page in enumerate(pages, start=1):
        count = len(page) * len(fields)
        columns = min(2, count)
        height = (count + columns - 1) // columns
        fig, axes = plt.subplots(height, columns, squeeze=False, layout="constrained",
                                 figsize=(PAGE_WIDTH if columns == 2 else COLUMN_WIDTH,
                                          2.45 * height + 0.5))
        try:
            axis_index = 0
            for scope, neighbours, classifier, subset in page:
                for field, label in fields:
                    ax = axes.flat[axis_index]
                    axis_index += 1
                    metric = f"knn_{neighbours}_{field}"
                    if delta:
                        for treatment, control in PAIRS:
                            _curve(ax, [row for row in subset if row["treatment"] == treatment
                                       and row["control"] == control], treatment, metric, delta=True,
                                   label=f"{label_for(treatment)} - {label_for(control)}")
                        ax.axhline(0, color="#444444", linewidth=0.7)
                    else:
                        for method in METHODS:
                            _curve(ax, [row for row in subset if row["method"] == method], method,
                                   metric, x=x, xscale=1 / 1024 if x == "code_bytes_per_image" else 1)
                    if x == "eval_k":
                        _log2_k_axis(ax, [row["eval_k"] for row in subset], label="Inference $K$")
                    elif x == "nnz_mean":
                        ax.set_xscale("symlog", base=2, linthresh=1)
                        ax.set_xlabel("Mean nonzeros (support)")
                    else:
                        ax.set_xscale("symlog", base=2, linthresh=1 / 1024)
                        ax.set_xlabel("Code size (KiB/image)")
                    ax.set_ylabel(label)
                    ax.set_title(f"{_scope_label(scope)} · {neighbours}-NN · "
                                 f"T={classifier['temperature']:g}", fontsize=9)
            for ax in list(axes.flat)[axis_index:]:
                ax.set_visible(False)
            _dedup_legend(fig, list(axes.flat)[:axis_index], ncol=3)
            fig.suptitle(title)
            seeds_key = "seeds_paired" if delta else "seeds_present"
            caption = (
                (f"Part {page_index} of {len(pages)}. " if len(pages) > 1 else "")
                + f"{title}. DINO weighted kNN classification on frozen sparse codes. "
                "The neighbor count in each panel is independent of the embedding's inference K. "
                "Votes sum exp(cosine similarity / temperature) by support class. "
                "All inference budgets use the same checkpoint at the training K in the title. "
                + ("Differences are paired within training seed and identical classifier settings. "
                   if delta else "")
                + seed_caption([seed for row in rows for seed in row[seeds_key]])
                + " Per-point n and top-5 results are in the tables. Panel protocols: "
                + "; ".join(f"{_scope_label(scope)}, {neighbours}-NN, "
                            f"{json.dumps(classifier, sort_keys=True)}, protocol={scope[1]}, "
                            f"comparison={scope[2]}"
                            for scope, neighbours, classifier, _ in page)
            )
            page_name = name if page_index == 1 else f"{name}_part{page_index}"
            paths.extend(save_figure(fig, out_dir, page_name, caption=caption))
        finally:
            plt.close(fig)
    return paths


def _render_figures(rows: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]],
                     out_dir: Path) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    from ..report.style import apply_style, backbone_label

    fonts = apply_style()
    figures: dict[str, list[str]] = {}
    skipped: list[str] = []
    dense = [row for row in rows if row["method"] == "dense"]
    sparse = [row for row in rows if row["method"] in METHODS]
    scopes = sorted({(row["backbone"], row["mode"], row["train_k"]) for row in sparse})
    for backbone, mode, train_k in scopes:
        subset = [row for row in sparse if (row["backbone"], row["mode"], row["train_k"])
                  == (backbone, mode, train_k)]
        references = [row for row in dense if (row["backbone"], row["mode"]) == (backbone, mode)]
        differences = [row for row in paired if (row["backbone"], row["mode"], row["train_k"])
                       == (backbone, mode, train_k)]
        suffix = _SAFE_NAME.sub("_", f"{backbone}_{mode}_train-k{train_k}")
        title = f"{backbone_label(backbone)} · Train $K$={train_k}"
        retrieval = [row for row in subset if row.get("task", "retrieval") == "retrieval"]
        retrieval_pairs = [row for row in differences if row.get("task", "retrieval") == "retrieval"]
        classification = [row for row in subset if row.get("task") == "knn_classification"]
        classification_pairs = [row for row in differences if row.get("task") == "knn_classification"]
        jobs = []
        if retrieval:
            jobs.extend([
                (f"quality_{suffix}", lambda: _draw_quality(retrieval, references, out_dir,
                                                           f"quality_{suffix}", title)),
                (f"nonzeros_{suffix}", lambda: _draw_quality(retrieval, references, out_dir,
                                                            f"nonzeros_{suffix}", title, x="nnz_mean")),
                (f"storage_{suffix}", lambda: _draw_quality(retrieval, references, out_dir,
                                                           f"storage_{suffix}", title,
                                                           x="code_bytes_per_image")),
                (f"paired_{suffix}", lambda: _draw_paired(retrieval_pairs, out_dir,
                                                         f"paired_{suffix}", title)),
                (f"cost_{suffix}", lambda: _draw_costs(retrieval, out_dir, f"cost_{suffix}", title)),
            ])
        if classification:
            jobs.extend([
                (f"classification_{suffix}", lambda: _draw_classification(
                    classification, out_dir, f"classification_{suffix}", title)),
                (f"classification_balance_{suffix}", lambda: _draw_classification(
                    classification, out_dir, f"classification_balance_{suffix}", title, kind="balance")),
                (f"classification_nonzeros_{suffix}", lambda: _draw_classification(
                    classification, out_dir, f"classification_nonzeros_{suffix}", title, x="nnz_mean")),
                (f"classification_storage_{suffix}", lambda: _draw_classification(
                    classification, out_dir, f"classification_storage_{suffix}", title, x="code_bytes_per_image")),
                (f"classification_paired_{suffix}", lambda: _draw_classification(
                    classification_pairs, out_dir, f"classification_paired_{suffix}", title, kind="paired")),
                (f"classification_cost_{suffix}", lambda: _draw_costs(
                    classification, out_dir, f"classification_cost_{suffix}", title)),
            ])
        for name, draw in jobs:
            try:
                paths = draw()
            except Exception as error:
                skipped.append(f"{name}: {type(error).__name__}: {error}")
            else:
                if paths:
                    figures[name] = [str(path) for path in paths]
                else:
                    skipped.append(f"{name}: required measurements are absent")
    return figures, skipped, fonts


def build_report(records: Sequence[Mapping[str, Any]], out_dir: str | Path, *,
                  expected_points: int | None = None, failures: Sequence[Any] | None = None,
                  expected_seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    """Write a reproducible campaign report from its explicit result list.

    ``expected_points`` counts evaluation records, including dense references
    if they are part of the campaign. A report without an expected count leaves
    ``complete`` null rather than claiming an unknown campaign is finished.
    Failed/incomplete points never enter a mean and remain visible in the
    manifest. A plotting failure does not discard correctly written tables.
    """
    if expected_points is not None and (isinstance(expected_points, bool)
                                        or not isinstance(expected_points, int)
                                        or expected_points < 0):
        raise ValueError("expected_points must be a nonnegative integer or None")
    expected_seeds = tuple(dict.fromkeys(int(seed) for seed in expected_seeds))
    clean, incomplete = _normalise(records)
    if expected_points is not None and len(clean) > expected_points:
        raise ValueError("more supplied evaluations than the campaign expected")
    rows = aggregate_records(clean, expected_seeds=expected_seeds)
    paired = paired_differences(clean, expected_seeds=expected_seeds)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = list(failures or ())
    missing = None if expected_points is None else expected_points - len(clean)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "out_dir": str(out_dir),
        "complete": (not missing and not failures and not incomplete)
        if expected_points is not None else None,
        "counts": {"provided_records": len(records), "completed_points": len(clean),
                   "expected_points": expected_points, "missing_points": missing,
                   "aggregate_rows": len(rows), "paired_rows": len(paired)},
        "expected_seeds": list(expected_seeds),
        "evaluation_ids": sorted(record["evaluation_id"] for record in clean),
        "incomplete_evaluation_ids": incomplete,
        "failures": failures,
        "metric_units": "fractions in JSON/CSV; percent and percentage points in figures",
        "spread": "sample standard deviation across training seeds (ddof=1)",
        "tables": {}, "figures": {}, "skipped": [], "fonts": None,
    }
    table_dir = out_dir / "tables"
    retrieval_rows = [row for row in rows if row["task"] == "retrieval"]
    classification_rows = [row for row in rows if row["task"] == "knn_classification"]
    for name, values in (("retrieval", retrieval_rows), ("classification", classification_rows),
                         ("paired", paired), ("evaluations", clean)):
        manifest["tables"][name] = _write_table(values, table_dir, name)
    try:
        figures, skipped, fonts = _render_figures(rows, paired, out_dir / "figures")
        manifest.update(figures=figures, skipped=skipped, fonts=fonts)
    except Exception as error:
        manifest["skipped"].append(f"figures: {type(error).__name__}: {error}")
    manifest["plot_failures"] = [reason for reason in manifest["skipped"]
                                  if "Error:" in reason or reason.startswith("figures:")]
    if manifest["complete"] is not None:
        manifest["complete"] = bool(manifest["complete"] and not manifest["plot_failures"]
                                    and all(row["complete"] for row in rows))
    atomic_json(out_dir / "manifest.json", manifest)
    get_logger("benchmark:report").info(
        "%d/%s evaluations, %d aggregate rows, %d figures -> %s", len(clean),
        expected_points if expected_points is not None else "?", len(rows),
        len(manifest["figures"]), out_dir)
    return manifest
