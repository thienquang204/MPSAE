"""Turn a tree of per-run JSON into tables, honestly.

Three rules drive every function here.

*Partial grids still report.* 175 fits do not finish at the same time, and a
report that refuses to render until the last seed lands is a report nobody
reads. Every loader tolerates a missing file; every aggregate carries ``n`` and
the list of seeds it is actually missing, so an unfinished cell announces itself
instead of masquerading as a finished one.

*Sample standard deviation, not population.* The default uses one seed, so
standard deviation is undefined and returned as None. For explicitly selected
multi-seed campaigns, the sample estimator uses ``ddof=1``.

*Differences are paired.* A seed fixes both the initialisation and the data
order, so the same seed under two methods is a matched pair. We report the mean
and sample std of the per-seed difference, never a difference of means. We do
not report p-values or infer significance from a single seed.

Pure standard library. ``statistics`` covers every estimator we need,
so a report renders on a machine with neither numpy nor torch installed.
"""
from __future__ import annotations

import csv
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..config import BACKBONES, METHODS, METHOD_VARIANTS, SEEDS, Cell, grid, method_from_directory
from ..io import read_json, read_jsonl
from ..logs import get_logger

_K_DIR = re.compile(r"^k(\d+)$")
_SEED_DIR = re.compile(r"^seed(\d+)$")

#: Per-run E1 fields we average over seeds, as ``field -> column base``.
_E1_METRICS = {
    "top1": "top1",
    "probe_top1": "probe_top1",
    "probe_top5": "probe_top5",
    "nnz_mean": "nnz",
    "empty_queries": "empty_queries",
    "seconds": "search_seconds",
    "train_seconds": "train_seconds",
    "gpu_hours": "gpu_hours",
    "index_bytes": "index_bytes",
}

#: Per-point E2 fields we average over seeds.
_E2_METRICS = {
    "top1": "top1",
    "nnz_mean": "nnz",
    "probe_top1": "probe_top1",
    "empty_queries": "empty_queries",
    "seconds": "search_seconds",
    "index_bytes": "index_bytes",
    "retrieval_ms": "retrieval_ms",
}


# --------------------------------------------------------------------- loading

def runs_dir(root: str | Path) -> Path:
    """Accept either the project root or the ``runs/`` directory itself."""
    root = Path(root)
    return root if root.name == "runs" else root / "runs"


def load_runs(root: str | Path, *, backbones: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Every run directory under ``runs/``, parsed and annotated.

    One record per ``runs/<backbone>/<method>/k<K>/seed<SEED>/`` directory::

        {"backbone", "method", "k", "seed", "path",
         "config": dict | None, "e1": dict | None, "e2": dict | None,
         "has_train_log": bool, "complete": bool, "missing": [str, ...]}

    ``complete`` means E1 evaluated; E2 only exists for the anchor K values, so
    its absence is not a defect and is not counted against a cell.
    """
    log = get_logger("report")
    base = runs_dir(root)
    records: list[dict[str, Any]] = []
    if not base.is_dir():
        log.warning("no runs directory at %s; nothing to report", base)
        return records

    wanted = set(backbones) if backbones else None
    for backbone_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        backbone = backbone_dir.name
        if wanted is not None and backbone not in wanted:
            continue
        if backbone not in BACKBONES:
            log.warning("skipping unknown backbone directory %s", backbone_dir)
            continue
        for method_dir in sorted(p for p in backbone_dir.iterdir() if p.is_dir()):
            method = method_from_directory(method_dir.name)
            if method is None:
                log.warning("skipping historical or unknown method directory %s", method_dir)
                continue
            for k_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                match = _K_DIR.match(k_dir.name)
                if match is None:
                    continue
                for seed_dir in sorted(p for p in k_dir.iterdir() if p.is_dir()):
                    seed_match = _SEED_DIR.match(seed_dir.name)
                    if seed_match is None:
                        continue
                    record = _load_one(seed_dir, backbone, method,
                                       int(match.group(1)), int(seed_match.group(1)))
                    validate_result_variant(record)
                    records.append(record)

    expected = {(c.backbone, c.method, c.k, c.seed) for c in grid()
                if wanted is None or c.backbone in wanted}
    found = {(r["backbone"], r["method"], r["k"], r["seed"]) for r in records}
    done = sum(1 for r in records if r["complete"])
    unexpected = sorted(found - expected)
    log.info("runs: %d directories, %d with E1 results, %d expected cells (%.1f%% complete)",
             len(records), done, len(expected), 100.0 * done / max(1, len(expected)))
    if unexpected:
        log.warning("%d run(s) outside the configured grid, e.g. %s",
                    len(unexpected), "/".join(map(str, unexpected[0])))
    for record in records:
        if record["missing"]:
            log.debug("incomplete %s: missing %s", record["path"], ", ".join(record["missing"]))
    return records


def _load_one(path: Path, backbone: str, method: str, k: int, seed: int) -> dict[str, Any]:
    config = read_json(path / "config.json")
    e1 = read_json(path / "eval_e1.json")
    e2 = read_json(path / "eval_e2.json")
    missing = [name for name, value in (("config.json", config), ("eval_e1.json", e1))
               if value is None]
    return {"backbone": backbone, "method": method, "k": k, "seed": seed,
            "path": str(path), "config": config, "e1": e1, "e2": e2,
            "has_train_log": (path / "train.jsonl").is_file(),
            "code_version": (config or {}).get("code_version"),
            "complete": e1 is not None, "missing": missing}


def validate_result_variant(record: Mapping[str, Any]) -> None:
    """A directory name cannot establish the objective behind measurements.

    Current partial runs may not have written their config yet. Once a run has
    E1 results or E2 points, require its saved objective to identify the current
    variant, including when records were supplied directly to the report API.
    """
    expected = METHOD_VARIANTS.get(record.get("method"))
    if not expected:
        return
    e2 = record.get("e2")
    has_results = (record.get("e1") is not None
                   or isinstance(e2, Mapping) and bool(e2.get("points")))
    if not has_results:
        return
    config = record.get("config") or {}
    identity = config.get("identity") or {}
    objective = identity.get("objective") or {}
    if objective.get("variant") != expected:
        location = record.get("path", (record.get("backbone"), record.get("k"),
                                       record.get("seed")))
        raise ValueError(f"{location}: {record['method']} results have a missing or different "
                         f"objective variant; expected {expected}. Historical measurements "
                         "cannot be reported as the current method.")


def load_dense_references(root: str | Path) -> dict[str, dict[str, Any]]:
    """Our own measured dense 1-NN per backbone (``runs/<backbone>/dense_reference.json``).

    Kept separate from ``reference.dense_reference`` so a figure cannot confuse
    the number we measured with the number CSR published.
    """
    base = runs_dir(root)
    out: dict[str, dict[str, Any]] = {}
    if not base.is_dir():
        return out
    for name in sorted(BACKBONES):
        payload = read_json(base / name / "dense_reference.json")
        if isinstance(payload, Mapping) and payload.get("top1") is not None:
            out[name] = dict(payload)
    return out


# ------------------------------------------------------------------ estimators

def _stats(values: Sequence[float]) -> tuple[float | None, float | None, int]:
    """``(mean, sample std, n)``. std is ``None`` below two observations."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None, 0
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else None
    return mean, std, len(clean)


def _metric_columns(samples: Mapping[str, list[tuple[int, float]]],
                    metrics: Mapping[str, str]) -> dict[str, Any]:
    """Expand ``field -> [(seed, value)]`` into ``<base>_{mean,std,n}`` columns."""
    out: dict[str, Any] = {}
    for field, base in metrics.items():
        pairs = samples.get(field, [])
        mean, std, count = _stats([value for _, value in pairs])
        out[f"{base}_mean"], out[f"{base}_std"], out[f"{base}_n"] = mean, std, count
    return out


def _collect(records: Iterable[Mapping[str, Any]], payload_key: str,
             metrics: Mapping[str, str]) -> dict[str, list[tuple[int, float]]]:
    samples: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        payload = record[payload_key]
        for field in metrics:
            value = payload.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                samples.setdefault(field, []).append((record["seed"], float(value)))
    return samples


def _seed_note(seeds: Sequence[int], expected: Sequence[int]) -> tuple[str, str]:
    """``(seeds present, seeds missing)`` as compact strings for the tables."""
    present = sorted(set(seeds))
    missing = [s for s in expected if s not in set(present)]
    return ("+".join(map(str, present)), "+".join(map(str, missing)))


# -------------------------------------------------------------------------- E1

def aggregate_e1(records: Iterable[Mapping[str, Any]], *,
                 seeds: Sequence[int] = SEEDS) -> list[dict[str, Any]]:
    """Mean / sample-std / n over seeds, per (backbone, method, K).

    Rows are sorted backbone, then method in ``config.METHODS`` order, then K,
    which is also the order the figures want.
    """
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("e1") is None:
            continue
        groups.setdefault((record["backbone"], record["method"], record["k"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for (backbone, method, k), members in groups.items():
        samples = _collect(members, "e1", _E1_METRICS)
        present, missing = _seed_note([m["seed"] for m in members], seeds)
        versions = sorted({m.get("code_version") for m in members if m.get("code_version")})
        row: dict[str, Any] = {"backbone": backbone, "method": method, "k": k}
        row.update(_metric_columns(samples, _E1_METRICS))
        row.update(seeds_present=present, seeds_missing=missing,
                   complete=not missing,
                   code_versions="+".join(map(str, versions)) if versions else "")
        rows.append(row)
    return _sorted_rows(rows, ("backbone", "method", "k"))


def paired_difference(records: Iterable[Mapping[str, Any]], *, treatment: str,
                      control: str, seeds: Sequence[int] = SEEDS) -> list[dict[str, Any]]:
    """Per-seed difference ``treatment - control``, summarised per (backbone, K).

    Pairing is the point: a seed fixes initialisation *and* data order, so the
    two methods under one seed share their nuisance variation and the difference
    removes it. A seed present for only one of the two methods is excluded from
    the pair and named in ``seeds_missing`` -- never silently dropped.

    No p-values. Three paired observations cannot support an inference claim;
    the mean and sample std of the differences are what the data can say.
    """
    if treatment == control:
        raise ValueError("treatment and control must differ")
    log = get_logger("report")
    index: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    for record in records:
        if record.get("e1") is not None:
            key = (record["backbone"], record["method"], record["k"], record["seed"])
            index[key] = record["e1"]

    cells = sorted({(backbone, k) for (backbone, method, k, _) in index
                    if method in (treatment, control)})
    rows: list[dict[str, Any]] = []
    for backbone, k in cells:
        deltas: dict[str, list[tuple[int, float]]] = {}
        paired_seeds: list[int] = []
        unpaired: list[int] = []
        treat_top1: list[float] = []
        control_top1: list[float] = []
        for seed in seeds:
            lhs = index.get((backbone, treatment, k, seed))
            rhs = index.get((backbone, control, k, seed))
            if lhs is None or rhs is None:
                if lhs is not None or rhs is not None:
                    unpaired.append(seed)
                continue
            paired_seeds.append(seed)
            if isinstance(lhs.get("top1"), (int, float)):
                treat_top1.append(float(lhs["top1"]))
            if isinstance(rhs.get("top1"), (int, float)):
                control_top1.append(float(rhs["top1"]))
            for field in ("top1", "probe_top1", "nnz_mean"):
                a, b = lhs.get(field), rhs.get(field)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    deltas.setdefault(field, []).append((seed, float(a) - float(b)))
        if not paired_seeds:
            continue
        row: dict[str, Any] = {"backbone": backbone, "k": k,
                               "treatment": treatment, "control": control}
        row.update(_metric_columns(deltas, {"top1": "delta_top1",
                                            "probe_top1": "delta_probe_top1",
                                            "nnz_mean": "delta_nnz"}))
        # Plain names, not ``_mean``: these are context for the delta, and the
        # table formatter must not pair them with a standard deviation.
        row.update(treatment_top1=_stats(treat_top1)[0],
                   control_top1=_stats(control_top1)[0],
                   seeds_paired="+".join(map(str, paired_seeds)),
                   seeds_missing="+".join(map(str, sorted(set(seeds) - set(paired_seeds)))),
                   seeds_unpaired="+".join(map(str, sorted(set(unpaired)))))
        rows.append(row)
        if unpaired:
            log.warning("%s k=%d: seeds %s have only one of %s/%s and are excluded from the pair",
                        backbone, k, row["seeds_unpaired"], treatment, control)
    return _sorted_rows(rows, ("backbone", "k"))


# -------------------------------------------------------------------------- E2

def aggregate_e2(records: Iterable[Mapping[str, Any]], *,
                 seeds: Sequence[int] = SEEDS) -> list[dict[str, Any]]:
    """Mean / sample-std / n per (backbone, method, anchor K, inference K).

    ``nnz`` rides along so a figure can plot accuracy against the sparsity the
    code actually realised instead of the nominal budget: after the ReLU a code
    can hold far fewer than K nonzeros, and the gap grows with K.
    """
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for record in records:
        payload = record.get("e2")
        if not isinstance(payload, Mapping):
            continue
        anchor = payload.get("anchor_k", record["k"])
        points = payload.get("points")
        if not isinstance(points, Sequence):
            continue
        for point in points:
            if not isinstance(point, Mapping) or point.get("k") is None:
                continue
            key = (record["backbone"], record["method"], int(anchor), int(point["k"]))
            groups.setdefault(key, []).append({"seed": record["seed"], "e2": point})

    rows: list[dict[str, Any]] = []
    for (backbone, method, anchor, k), members in groups.items():
        samples = _collect(members, "e2", _E2_METRICS)
        present, missing = _seed_note([m["seed"] for m in members], seeds)
        row: dict[str, Any] = {"backbone": backbone, "method": method,
                               "anchor_k": anchor, "k": k,
                               "at_anchor": k == anchor}
        row.update(_metric_columns(samples, _E2_METRICS))
        row.update(seeds_present=present, seeds_missing=missing, complete=not missing)
        rows.append(row)
    return _sorted_rows(rows, ("backbone", "method", "anchor_k", "k"))


# -------------------------------------------------------------------- training

def training_summary(root: str | Path, *, max_curve_points: int = 400,
                     backbones: Sequence[str] | None = None,
                     selected_cells: Iterable[tuple[str, str, int, int]] | None = None
                     ) -> list[dict[str, Any]]:
    """One row per run, read from ``train.jsonl``.

    Besides the scalar summary (final losses, mean step time, GPU hours) each
    row carries a down-sampled ``curve``: ``{field: [value, ...]}`` aligned with
    ``curve["step"]``. Keeping the curve here rather than in the figure module
    means the diagnostic plot has no file I/O of its own and the same numbers
    back both the table and the panels. ``selected_cells`` restricts reading to
    explicit (backbone, method, K, seed) cells before any log is parsed; a
    campaign therefore cannot accidentally read a stale or malformed run.
    """
    log = get_logger("report")
    base = runs_dir(root)
    rows: list[dict[str, Any]] = []
    if not base.is_dir():
        return rows
    wanted = set(backbones) if backbones else None
    if selected_cells is None:
        paths = sorted(base.glob("*/*/k*/seed*/train.jsonl"))
    else:
        cells = set(selected_cells)
        for backbone, method, k, seed in cells:
            if (backbone not in BACKBONES or method not in METHODS
                    or isinstance(k, bool) or not isinstance(k, int) or k < 1
                    or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
                raise ValueError(f"invalid selected training cell: {(backbone, method, k, seed)}")
        paths = sorted(base / Cell(backbone, method, k, seed).slug / "train.jsonl"
                       for backbone, method, k, seed in cells)

    for path in paths:
        if not path.is_file():
            continue
        seed_dir = path.parent
        k_match, seed_match = _K_DIR.match(seed_dir.parent.name), _SEED_DIR.match(seed_dir.name)
        if k_match is None or seed_match is None:
            continue
        backbone = seed_dir.parents[2].name
        method = method_from_directory(seed_dir.parents[1].name)
        if method is None:
            continue
        if wanted is not None and backbone not in wanted:
            continue
        log_records = read_jsonl(path)
        if not log_records:
            log.debug("empty train log %s", path)
            continue
        config = read_json(seed_dir / "config.json") or {}
        final = log_records[-1]
        # ``elapsed`` is cumulative wall-clock since the fit began, so the total
        # is its last value -- summing strided records would multiply it by the
        # number of records. Prefer the fit summary, which survives log striding.
        summary = read_json(seed_dir / "summary.json") or {}
        elapsed = [float(r["elapsed"]) for r in log_records
                   if isinstance(r.get("elapsed"), (int, float))]
        train_seconds = summary.get("train_seconds") or (max(elapsed) if elapsed else None)
        total_steps = config.get("total_steps")
        last_step = final.get("step")
        row: dict[str, Any] = {
            "backbone": backbone, "method": method,
            "k": int(k_match.group(1)), "seed": int(seed_match.group(1)),
            "logged_records": len(log_records),
            "last_step": last_step, "total_steps": total_steps,
            "finished": (isinstance(total_steps, int) and isinstance(last_step, int)
                         and last_step >= total_steps - 1),
            "train_seconds": train_seconds,
            "gpu_hours": train_seconds / 3600.0 if train_seconds else None,
            "step_seconds_mean": (train_seconds / last_step
                                  if train_seconds and isinstance(last_step, int) and last_step
                                  else None),
            "residency": summary.get("residency"),
        }
        for field in ("loss_total", "loss_mse_k", "loss_mse_4k", "loss_aux",
                      "loss_contrastive", "loss_ot", "nnz_mean", "dead_fraction",
                      "empty_fraction", "active_units", "training_k", "lr"):
            value = final.get(field)
            if value is not None:
                row[f"final_{field}"] = value
        row["k_schedule"] = _k_trajectory(log_records)
        row["curve"] = _downsample(log_records, max_curve_points)
        rows.append(row)
    log.info("training logs: %d run(s), %d finished",
             len(rows), sum(1 for r in rows if r["finished"]))
    return _sorted_rows(rows, ("backbone", "method", "k", "seed"))


def _k_trajectory(log_records: Sequence[Mapping[str, Any]]) -> str:
    """Realised k-annealing as ``step:k`` change points, compact enough for a table."""
    points: list[str] = []
    previous = None
    for record in log_records:
        current = record.get("training_k")
        if current is None or current == previous:
            continue
        points.append(f"{record.get('step', len(points))}:{current}")
        previous = current
    return ",".join(points)


def _downsample(log_records: Sequence[Mapping[str, Any]], limit: int) -> dict[str, list[Any]]:
    """Evenly thin a log to at most ``limit`` points, always keeping the last."""
    if limit < 2:
        raise ValueError("need at least two curve points")
    indices = list(range(len(log_records)))
    if len(indices) > limit:
        stride = max(1, len(indices) // (limit - 1))
        indices = indices[::stride]
        if indices[-1] != len(log_records) - 1:
            indices.append(len(log_records) - 1)
    fields = ("step", "loss_total", "loss_mse_k", "loss_mse_4k", "loss_aux",
              "loss_contrastive", "loss_ot", "training_k", "dead_fraction",
              "nnz_mean", "empty_fraction")
    curve: dict[str, list[Any]] = {}
    for field in fields:
        column = [log_records[i].get(field) for i in indices]
        if any(value is not None for value in column):
            curve[field] = column
    curve.setdefault("step", list(indices))
    return curve


# ---------------------------------------------------------------------- tables

def _sorted_rows(rows: list[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    order = {method: i for i, method in enumerate(METHODS)}

    def sort_key(row: Mapping[str, Any]) -> tuple:
        out: list[Any] = []
        for key in keys:
            value = row.get(key)
            if key == "method":
                out.append(order.get(value, 99))
            else:
                out.append(value if value is not None else -1)
        return tuple(out)

    return sorted(rows, key=sort_key)


#: Display precision by column-name substring, most specific first. Accuracies
#: get two decimals because that is the precision the papers report.
_DIGITS: tuple[tuple[str, int], ...] = (
    ("index_bytes", 0), ("bytes", 0), ("fraction", 4), ("empty", 1),
    ("active_units", 1), ("top1", 2), ("top5", 2), ("nnz", 2), ("gpu_hours", 3),
    ("seconds", 3), ("loss", 5), ("lr", 6),
)

_TEXT_COLUMNS = frozenset({"backbone", "method", "treatment", "control", "seeds_present",
                           "seeds_missing", "seeds_paired", "seeds_unpaired",
                           "code_versions", "k_schedule", "path", "complete",
                           "at_anchor", "finished"})


def _digits(column: str) -> int:
    for needle, digits in _DIGITS:
        if needle in column:
            return digits
    return 4


def _columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Union of keys in first-seen order, so column order follows construction.

    A metric nobody measured (``probe_top5`` when probes were not run) is
    dropped entirely rather than rendered as a column of em-dashes: an absent
    column says "not measured" at least as clearly and keeps the table legible.
    """
    seen: set[str] = set()
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen and key != "curve":  # the curve is for figures only
                seen.add(key)
                columns.append(key)
    empty = {column[: -len("_mean")] for column in columns if column.endswith("_mean")
             and all(row.get(column) is None for row in rows)}
    return [column for column in columns
            if not any(column == f"{base}{suffix}"
                       for base in empty for suffix in ("_mean", "_std", "_n"))]


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Machine-readable: enough precision to recompute, and never exponential
        # for a byte count that happens to average to a round number.
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "+".join(map(str, value))
    return str(value)


def _display(value: Any, column: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.{_digits(column)}f}"
    if isinstance(value, (list, tuple)):
        return "+".join(map(str, value)) or "—"
    return str(value)


def _markdown(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], name: str) -> str:
    """GitHub-flavoured table: ``mean ± std`` fused, ``n`` kept as its own column."""
    bases = {c[: -len("_mean")] for c in columns if c.endswith("_mean")}
    n_columns = [f"{base}_n" for base in bases if f"{base}_n" in columns]
    # When every metric in a row rests on the same number of seeds -- the usual
    # case -- one ``n`` column says it. Per-metric ``n`` only appears when the
    # metrics genuinely disagree, which is precisely when the reader needs it.
    shared_n = bool(n_columns) and all(
        len({row.get(column) for column in n_columns
             if row.get(f"{column[: -len('_n')]}_mean") is not None}) <= 1
        for row in rows)

    header: list[str] = []
    plan: list[tuple[str, str]] = []  # (kind, key)
    emitted_shared_n = False
    for column in columns:
        base = column.rsplit("_", 1)[0]
        if column.endswith("_std") and base in bases:
            continue
        if column.endswith("_mean") and base in bases:
            header.append(base)
            plan.append(("pm", base))
        elif column.endswith("_n") and base in bases:
            if not shared_n:
                header.append(f"n ({base})")
                plan.append(("plain", column))
            elif not emitted_shared_n:
                emitted_shared_n = True
                header.append("n")
                plan.append(("n", "|".join(n_columns)))
        else:
            header.append(column)
            plan.append(("plain", column))

    align = ["---" if (kind == "plain" and key in _TEXT_COLUMNS) else "---:"
             for kind, key in plan]
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join(align)} |"]
    for row in rows:
        cells: list[str] = []
        for kind, key in plan:
            if kind == "pm":
                mean, std = row.get(f"{key}_mean"), row.get(f"{key}_std")
                cells.append(f"{_display(mean, key)} ± {_display(std, key)}")
            elif kind == "n":
                values = {row.get(column) for column in key.split("|")
                          if row.get(f"{column[: -len('_n')]}_mean") is not None}
                cells.append(_display(values.pop() if values else None, "n"))
            else:
                cells.append(_display(row.get(key), key))
        lines.append(f"| {' | '.join(cells)} |")
    note = ("Mean ± sample standard deviation (ddof=1) over seeds; `n` is the number of seeds "
            "actually present and `—` marks a quantity we did not measure. No p-values: three "
            "seeds cannot support an inference claim.")
    return f"### {name}\n\n" + "\n".join(lines) + f"\n\n{note}\n"


def write_tables(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                 name: str) -> dict[str, Path]:
    """Write ``<name>.csv`` (machine-readable) and ``<name>.md`` (for the paper).

    The CSV keeps ``mean``/``std``/``n`` in separate columns at full precision so
    a later script can recompute; the Markdown fuses mean and std into one
    two-decimal cell and keeps ``n`` beside it, because a mean without its ``n``
    is not a reportable number.
    """
    log = get_logger("report")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / f"{name}.csv", out_dir / f"{name}.md"
    columns = _columns(rows)
    if not columns:
        csv_path.write_text("", encoding="utf-8")
        md_path.write_text(f"### {name}\n\nNo rows: no completed runs matched this table.\n",
                           encoding="utf-8")
        log.warning("table %s is empty (no completed runs)", name)
        return {"csv": csv_path, "md": md_path}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_cell(row.get(column)) for column in columns])
    md_path.write_text(_markdown(rows, columns, name), encoding="utf-8")
    incomplete = sum(1 for row in rows if row.get("seeds_missing"))
    log.info("table %s: %d rows -> %s, %s%s", name, len(rows), csv_path.name, md_path.name,
             f" ({incomplete} row(s) missing at least one seed)" if incomplete else "")
    return {"csv": csv_path, "md": md_path}
