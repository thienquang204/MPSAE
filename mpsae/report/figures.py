"""Publication figures for experiments E1 and E2.

Each function consumes already-aggregated rows -- the figures never touch the
run tree, so what a panel shows and what the corresponding table says are the
same numbers by construction. Each returns the paths it wrote, or an empty list
when the data it needs is absent: a half-finished grid should still produce
every figure it can, and ``build_all`` must not die because one panel is empty.

E1 compares matched training/inference K. E2 holds the SAE checkpoint fixed
and measures accuracy, realised sparsity and costs over inference K. Only
experiment measurements enter these figures; training diagnostics are separate.

Captions are written beside every figure (``<name>.caption.txt``) so the claim
travels with the image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import BACKBONE_ORDER, BACKBONES, BASELINES, METHODS, OURS, SEEDS
from ..logs import get_logger
from .aggregate import (aggregate_e1, aggregate_e2, load_dense_references, load_runs,
                        paired_difference, training_summary, validate_result_variant, write_tables)
from .style import (BAND_ALPHA, COLUMN_WIDTH, PAGE_WIDTH,
                    apply_style, backbone_label, label_for,
                    method_style, save_figure, seed_caption)


def _pyplot() -> Any:
    """Import pyplot behind the Agg backend.

    The backend is selected inside the function, not at module import, so that
    importing ``mpsae.report`` on a headless machine (or one without matplotlib
    at all) stays free and never tries to open a display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------- row helpers

def _for_backbone(rows: Sequence[Mapping[str, Any]], backbone: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("backbone") == backbone and row.get("method") in METHODS]


def _methods_present(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Methods with data, in the canonical order so the legend never shuffles."""
    found = {row["method"] for row in rows if row.get("method")}
    return [method for method in METHODS if method in found]


def _series(rows: Sequence[Mapping[str, Any]], *, method: str, x: str = "k",
            y: str = "top1") -> tuple[list[float], list[float], list[float | None], list[int]]:
    """``(x, mean, std, n)`` for one method, sorted by ``x``, means only.

    A point with no mean is dropped rather than interpolated: we would rather
    show a gap than invent a value.
    """
    points: list[tuple[float, float, float | None, int]] = []
    for row in rows:
        if row.get("method") != method:
            continue
        abscissa = row.get(x) if x in row else row.get(f"{x}_mean")
        mean = row.get(f"{y}_mean")
        if abscissa is None or mean is None:
            continue
        points.append((float(abscissa), float(mean), row.get(f"{y}_std"),
                       int(row.get(f"{y}_n") or 0)))
    points.sort(key=lambda item: item[0])
    return ([p[0] for p in points], [p[1] for p in points],
            [p[2] for p in points], [p[3] for p in points])


def _defined(xs: Sequence[Any], ys: Sequence[Any]) -> tuple[list[float], list[float]]:
    """Drop ``(x, y)`` pairs where either side is missing.

    A train log can start recording a field late (the OT diagnostics only exist
    once the first matching runs), and matplotlib cannot plot ``None``. Dropping
    the pair leaves a gap, which is the truthful rendering of "not logged".
    """
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        out_x.append(float(x))
        out_y.append(float(y))
    return out_x, out_y


def _log2_k_axis(ax: Any, ks: Sequence[float], *, label: str = "Train $K$",
                 compact: bool = False) -> None:
    """Log2 x-axis ticked at exactly the trained K values, nothing between.

    Four-digit budgets run into each other at the right-hand end of a
    column-width panel, so once the labels are wide enough to touch they are
    rotated rather than thinned: dropping ticks would hide which budgets were
    actually trained, which is the one thing this axis exists to show.
    """
    from matplotlib import ticker

    unique = sorted({float(k) for k in ks})
    if not unique:
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks(unique)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{int(round(value))}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    crowded = compact or (len(unique) >= 9 and max(unique) >= 1024)
    if crowded:
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
            tick.set_rotation_mode("anchor")
    ax.set_xlabel(label)


def _plot_method(ax: Any, xs, means, stds, counts, *, method: str,
                 band: bool = True, label: str | None = None, **overrides) -> None:
    """One method's mean line plus its sample-std band."""
    style = method_style(method)
    kwargs = {"color": style["color"], "marker": style["marker"],
              "linestyle": style["linestyle"], "zorder": style["zorder"]}
    kwargs.update(overrides)
    text = style["label"] if label is None else label
    ax.plot(xs, means, label=text, **kwargs)
    if band and any(std is not None for std in stds):
        lower = [mean - (std or 0.0) for mean, std in zip(means, stds)]
        upper = [mean + (std or 0.0) for mean, std in zip(means, stds)]
        ax.fill_between(xs, lower, upper, where=[std is not None for std in stds],
                        color=style["color"], alpha=BAND_ALPHA, linewidth=0.0,
                        zorder=style["zorder"] - 0.5)


def _dedup_legend(fig: Any, axes: Sequence[Any], *, ncol: int, y: float = -0.01) -> None:
    """Shared short labels; per-point seed counts remain in the tables."""
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if not handles:
        return
    ncol = min(ncol, 3 if fig.get_figwidth() < 5 else 6, len(handles))
    try:
        # Constrained layout reserves room for an "outside" legend, so it
        # cannot land on top of the x tick labels the way a bbox offset can.
        fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False)
    except (ValueError, TypeError):
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
                   ncol=ncol, frameon=False)


# --------------------------------------------------------------- E1: accuracy

def figure_e1_curves(rows: Sequence[Mapping[str, Any]], out_dir: str | Path, backbone: str, *,
                     measured_dense: Mapping[str, Any] | None = None,
                     seeds: Sequence[int] = SEEDS) -> list[Path]:
    """Accuracy vs K for every method on one backbone. ``rows`` from ``aggregate_e1``.

    The std band is the sample std over seeds, not a confidence interval, and
    the caption says so. All plotted results come from the selected experiment.
    ``measured_dense`` is accepted for older callers; references are table-only.
    """
    log = get_logger("report")
    subset = _for_backbone(rows, backbone)
    if not subset:
        log.info("skip figure_e1_curves[%s]: no aggregated E1 rows", backbone)
        return []
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8), layout="constrained")
    ks: list[float] = []
    for method in _methods_present(subset):
        xs, means, stds, counts = _series(subset, method=method)
        if not xs:
            continue
        ks.extend(xs)
        _plot_method(ax, xs, means, stds, counts, method=method)

    _log2_k_axis(ax, ks)
    ax.set_ylabel("1-NN (%)")
    ax.set_title(backbone_label(backbone))
    _dedup_legend(fig, [ax], ncol=3)
    caption = (
        f"ImageNet 1-NN accuracy for {backbone_label(backbone)}. Each point uses a model "
        f"trained and evaluated at the same K. {seed_caption(seeds)}"
    )
    return save_figure(fig, out_dir, f"e1_curves_{backbone}", caption=caption)


def figure_e1_delta(records: Sequence[Mapping[str, Any]], out_dir: str | Path, backbone: str, *,
                    control: str = "csr", treatments: Sequence[str] = OURS,
                    seeds: Sequence[int] = SEEDS) -> list[Path]:
    """Paired per-seed improvement over ``control`` vs K. The claim figure.

    ``records`` are the *raw* run records from ``load_runs``, not aggregated
    rows: pairing has to happen seed by seed, and a difference of two means
    would throw away exactly the nuisance variation the pairing removes.
    """
    log = get_logger("report")
    if records and "top1_mean" in records[0]:
        raise ValueError("figure_e1_delta needs raw run records from load_runs(), "
                         "not aggregated rows: the difference must be formed per seed")
    plt = _pyplot()
    series: list[tuple[str, list[Mapping[str, Any]]]] = []
    for treatment in treatments:
        if treatment == control:
            continue
        rows = [row for row in paired_difference(records, treatment=treatment, control=control,
                                                 seeds=seeds)
                if row["backbone"] == backbone]
        if rows:
            series.append((treatment, rows))
    if not series:
        log.info("skip figure_e1_delta[%s vs %s]: no seed-paired cells", backbone, control)
        return []

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.7), layout="constrained")
    ks: list[float] = []
    # A small horizontal offset per method keeps overlapping error bars legible
    # on a log2 axis; it is multiplicative so it looks uniform at every K.
    spread = 0.035
    centre = (len(series) - 1) / 2.0
    for index, (treatment, rows) in enumerate(series):
        style = method_style(treatment)
        points = sorted(((row["k"], row["delta_top1_mean"], row["delta_top1_std"],
                          row["delta_top1_n"]) for row in rows
                         if row.get("delta_top1_mean") is not None),
                        key=lambda item: item[0])
        if not points:
            continue
        shift = 2.0 ** (spread * (index - centre))
        xs = [k * shift for k, _, _, _ in points]
        ks.extend(k for k, _, _, _ in points)
        ax.errorbar(xs, [m for _, m, _, _ in points],
                    yerr=([s if s is not None else 0.0 for _, _, s, _ in points]
                          if any(s is not None for _, _, s, _ in points) else None),
                    color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                    linewidth=1.1, markersize=3.8, capsize=2.0, elinewidth=0.8,
                    zorder=style["zorder"],
                    label=style["label"])

    ax.axhline(0.0, color="#444444", linewidth=0.7, linestyle="-", zorder=1.0)
    _log2_k_axis(ax, ks)
    ax.set_ylabel("Δ 1-NN (pp)")
    ax.set_title(f"{backbone_label(backbone)} · vs {label_for(control)}")
    _dedup_legend(fig, [ax], ncol=3)
    caption = (
        f"Per-seed improvement in 1-NN top-1 over {label_for(control)} on "
        f"{backbone_label(backbone)}, "
        f"in percentage points. {seed_caption(seeds)} Differences are paired within each seed; "
        "the seed fixes both initialisation and data "
        "order, so a shared seed is a matched pair and the difference cancels that nuisance "
        "variation. These are descriptive differences; no p-values are reported. Markers "
        "are offset slightly in K so overlapping bars stay readable."
    )
    return save_figure(fig, out_dir, f"e1_delta_{backbone}_vs_{control}", caption=caption)


# ------------------------------------------------------- E2: one checkpoint, many K

def _inference_figure(rows, out_dir, name, *, x="k", seeds=SEEDS, dense=None):
    """Export at most two backbone/training-K groups per publication figure."""
    rows = [row for row in rows if row.get("method") in METHODS]
    groups = sorted({(row["backbone"], int(row["anchor_k"])) for row in rows})
    if len(groups) <= 2:
        return _inference_page(rows, out_dir, name, x=x, seeds=seeds)
    pages = (len(groups) + 1) // 2
    written = []
    for page, first in enumerate(range(0, len(groups), 2), start=1):
        selected = set(groups[first:first + 2])
        panel_rows = [row for row in rows
                      if (row["backbone"], int(row["anchor_k"])) in selected]
        page_name = name if page == 1 else f"{name}_part{page}"
        written.extend(_inference_page(panel_rows, out_dir, page_name,
                                       x=x, seeds=seeds, page_note=f"Part {page} of {pages}. "))
    return written


def _inference_page(rows, out_dir, name, *, x="k", seeds=SEEDS, page_note=""):
    """One row per backbone/training K, with a shared method legend."""
    groups = sorted({(row["backbone"], int(row["anchor_k"])) for row in rows})
    metrics = [(field, label) for field, label in
               (("probe_top1", "Linear probe (%)"), ("top1", "1-NN (%)"))
               if any(row.get(f"{field}_mean") is not None for row in rows)]
    if not groups or not metrics or not any(
            row.get(x if x == "k" else f"{x}_mean") is not None for row in rows):
        return []
    plt = _pyplot()
    width = PAGE_WIDTH if len(metrics) > 1 else COLUMN_WIDTH
    fig, axes = plt.subplots(len(groups), len(metrics), squeeze=False,
                             figsize=(width, 2.45 * len(groups) + 0.4), layout="constrained")
    for i, (backbone, anchor) in enumerate(groups):
        panel = [row for row in rows if row["backbone"] == backbone and row["anchor_k"] == anchor]
        for ax, (field, label) in zip(axes[i], metrics):
            for method in _methods_present(panel):
                # Follow inference K, including on the measured-nnz axis where
                # increasing K need not increase the realised number of nonzeros.
                ordered = sorted((r for r in panel if r["method"] == method), key=lambda r: r["k"])
                points = [(r, r.get(x if x == "k" else f"{x}_mean")) for r in ordered]
                points = [(r, value) for r, value in points if value is not None
                          and r.get(f"{field}_mean") is not None]
                if not points:
                    continue
                _plot_method(ax, [value for r, value in points],
                             [r[f"{field}_mean"] for r, value in points],
                             [r.get(f"{field}_std") for r, value in points],
                             [r.get(f"{field}_n", 0) for r, value in points], method=method)
                for r, value in points:
                    if r["k"] == anchor:
                        ax.scatter([value], [r[f"{field}_mean"]], s=40, facecolors="none",
                                   edgecolors=method_style(method)["color"], linewidths=0.8,
                                   zorder=5, label="_nolegend_")
            if x == "k":
                ax.axvline(anchor, color="#999999", linewidth=0.65, linestyle=":", zorder=0)
                _log2_k_axis(ax, [r["k"] for r in panel], label="Inference $K$")
            else:
                ax.set_xscale("symlog", base=2, linthresh=1)
                ax.set_xlabel("Mean nonzeros (gallery)")
            ax.set_ylabel(label)
            ax.set_title(f"{backbone_label(backbone)} · Train $K$={anchor}", fontsize=9)
            if not any(r.get(f"{field}_mean") is not None for r in panel):
                ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
    _dedup_legend(fig, list(axes.flat), ncol=6)
    caption = (
        f"{page_note}ImageNet accuracy. Each method/seed uses the same frozen SAE checkpoint throughout "
        "each inference-K curve; rows separate backbones and training K. Linear probes are fitted "
        "separately on training codes at selected inference K values and evaluated on validation codes. "
        "The probe classifier is refitted; the SAE is not. Circles outline the training budget. "
        "Curves connect increasing inference K. Mean nonzeros, when used, count measured gallery "
        "entries and can be smaller than K. "
        f"{seed_caption(seeds)}"
    )
    return save_figure(fig, out_dir, name, caption=caption)


def figure_e2_generalisation(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                             backbone: str, *, e1_rows=None, seeds=SEEDS,
                             measured_dense=None) -> list[Path]:
    return _inference_figure(_for_backbone(rows, backbone), out_dir,
                             f"e2_generalisation_{backbone}", seeds=seeds,
                             dense={backbone: measured_dense or {}})


def figure_e2_nnz(rows: Sequence[Mapping[str, Any]], out_dir: str | Path, backbone: str, *,
                  e1_rows=None, seeds=SEEDS) -> list[Path]:
    return _inference_figure(_for_backbone(rows, backbone), out_dir,
                             f"e2_nnz_{backbone}", x="nnz", seeds=seeds)


# ------------------------------------------------------------------ diagnostic

def figure_training(rows: Sequence[Mapping[str, Any]], out_dir: str | Path, backbone: str, *,
                    k: int | None = None) -> list[Path]:
    """2x2 training diagnostic. ``rows`` from ``training_summary``.

    Not a paper figure: optimised for spotting a fit that diverged, a dead
    dictionary, or a k-schedule that did not anneal the way the config claims.
    One K at a time (``k``, default the median trained K) with every seed drawn,
    because disagreement between seeds is exactly what we want to see here.
    """
    log = get_logger("report")
    subset = [row for row in _for_backbone(rows, backbone) if row.get("curve")]
    if not subset:
        log.info("skip figure_training[%s]: no training logs", backbone)
        return []
    ks = sorted({int(row["k"]) for row in subset})
    chosen = int(k) if k is not None else ks[len(ks) // 2]
    subset = [row for row in subset if int(row["k"]) == chosen]
    if not subset:
        log.info("skip figure_training[%s]: no training logs at K=%d", backbone, chosen)
        return []

    plt = _pyplot()
    fig, grid_axes = plt.subplots(2, 2, figsize=(PAGE_WIDTH, 4.1), squeeze=False,
                                  layout="constrained")
    (ax_total, ax_parts), (ax_k, ax_dead) = grid_axes
    components = (("loss_mse_k", "-", "MSE$_K$"), ("loss_mse_4k", "--", "MSE$_{4K}$"),
                  ("loss_aux", ":", "AuxK"), ("loss_contrastive", "-.", "contrastive"),
                  ("loss_ot", (0, (3, 1, 1, 1)), "OT gap"))
    seen_components: set[str] = set()
    for row in subset:
        curve = row["curve"]
        steps = curve.get("step") or []
        style = method_style(row["method"])
        base = {"color": style["color"], "linewidth": 0.9, "alpha": 0.85}
        for ax, field in ((ax_total, "loss_total"), (ax_k, "training_k"),
                          (ax_dead, "dead_fraction")):
            xs, ys = _defined(steps, curve.get(field) or [])
            if xs:
                ax.plot(xs, ys, label=style["label"], **base)
        for field, dashes, pretty in components:
            xs, ys = _defined(steps, curve.get(field) or [])
            if not xs:
                continue
            label = pretty if pretty not in seen_components else "_nolegend_"
            seen_components.add(pretty)
            ax_parts.plot(xs, ys, linestyle=dashes, label=label, **base)

    for ax, title, ylabel, logscale in (
            (ax_total, "Total loss", "loss", True),
            (ax_parts, "Loss terms", "loss", True),
            (ax_k, "Training budget", "Train $K$", False),
            (ax_dead, "Dead latents", "fraction of $h$", False)):
        ax.set_title(title, fontsize=8.6)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if logscale:
            ax.set_yscale("log")
    for ax in (ax_total, ax_parts, ax_k, ax_dead):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=6.4, loc="best", ncol=2)
    fig.suptitle(f"{backbone_label(backbone)} · Train $K$={chosen}")
    caption = (
        f"Training diagnostics for {backbone_label(backbone)} at K={chosen}; one line per method "
        "per seed, no averaging, because seed disagreement is the signal here. Internal "
        "diagnostic, not a paper figure."
    )
    return save_figure(fig, out_dir, f"training_{backbone}_k{chosen}", caption=caption)


# -------------------------------------------------------------------- headline

def figure_backbone_summary(rows: Sequence[Mapping[str, Any]], out_dir: str | Path, *,
                            dense: Mapping[str, Mapping[str, Any]] | None = None,
                            share_y_span: float = 25.0,
                            seeds: Sequence[int] = SEEDS) -> list[Path]:
    """One panel per backbone, accuracy vs K, shared legend. ``rows`` from ``aggregate_e1``.

    The y-axis is shared only when every backbone's accuracies fall inside
    ``share_y_span`` percentage points. Forcing a shared axis across backbones
    whose measured accuracies differ by tens of points would compress the
    interesting panels into a flat line, so the rule is applied to the data and
    the outcome is stated in the caption.
    """
    log = get_logger("report")
    rows = [row for row in rows if row.get("method") in METHODS]
    present = [name for name in BACKBONE_ORDER
               if any(row.get("backbone") == name for row in rows)]
    present += sorted({row["backbone"] for row in rows} - set(present))
    if not present:
        log.info("skip figure_backbone_summary: no aggregated E1 rows")
        return []

    means = [row["top1_mean"] for row in rows if row.get("top1_mean") is not None]
    shared = bool(means) and (max(means) - min(means)) <= share_y_span
    plt = _pyplot()
    ncol = min(3, len(present))
    nrow = (len(present) + ncol - 1) // ncol
    fig, grid_axes = plt.subplots(nrow, ncol, figsize=(PAGE_WIDTH, 2.15 * nrow + 0.45),
                                 squeeze=False, sharey=shared, layout="constrained")
    axes = [grid_axes[row][col] for row in range(nrow) for col in range(ncol)]

    for ax, backbone in zip(axes, present):
        subset = _for_backbone(rows, backbone)
        ks: list[float] = []
        for method in _methods_present(subset):
            xs, mean_values, stds, counts = _series(subset, method=method)
            if not xs:
                continue
            ks.extend(xs)
            _plot_method(ax, xs, mean_values, stds, counts, method=method, markersize=2.8)
        _log2_k_axis(ax, ks)
        ax.set_title(backbone_label(backbone), fontsize=8.4)
    for ax in axes[len(present):]:
        ax.set_visible(False)
    for row in range(nrow):
        grid_axes[row][0].set_ylabel("1-NN (%)")

    # Seed counts can differ between backbones, so they are dropped here rather
    # than splitting one method into two legend entries; the tables carry n.
    _dedup_legend(fig, axes[:len(present)], ncol=min(5, len(METHODS)), y=-0.02)
    caption = (
        "1-NN top-1 on ImageNet-1K against training K, for every backbone; inference K equals training K. "
        f"{seed_caption(seeds)} Per-series n is in "
        "the accompanying table, since it can differ between panels. "
        + ("The y-axis is shared across panels."
           if shared else
           "The y-axis is per-panel: the backbones' measured accuracies differ by more than "
           f"{share_y_span:.0f} points, and a shared axis would flatten the smaller panels.")
    )
    return save_figure(fig, out_dir, "backbone_summary", caption=caption)


# ------------------------------------------------------------------ efficiency

def figure_efficiency(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                      backbone: str, *, seeds=SEEDS) -> list[Path]:
    """Measured costs of inference sweeps; never connect distinct training K."""
    subset = _for_backbone(rows, backbone)
    if not subset:
        return []
    anchors = sorted({row.get("anchor_k", row["k"]) for row in subset})
    fields = [(field, label, scale) for field, label, scale in
              (("search_seconds", "Search time (s)", 1),
               ("index_bytes", "Index (MiB)", 2**20))
              if any(row.get(f"{field}_mean") is not None for row in subset)]
    if not fields:
        return []
    plt = _pyplot()
    fig, axes = plt.subplots(len(anchors), len(fields), squeeze=False,
                             figsize=(COLUMN_WIDTH * len(fields), 2.45 * len(anchors) + 0.4),
                             layout="constrained")
    for i, anchor in enumerate(anchors):
        panel = [r for r in subset if r.get("anchor_k", r["k"]) == anchor]
        for ax, (field, label, scale) in zip(axes[i], fields):
            for method in _methods_present(panel):
                points = sorted((r for r in panel if r["method"] == method and
                                 r.get(f"{field}_mean") is not None and r.get("top1_mean") is not None),
                                key=lambda r: r["k"])
                if points:
                    _plot_method(ax, [r[f"{field}_mean"] / scale for r in points],
                                 [r["top1_mean"] for r in points], [None] * len(points),
                                 [], method=method, band=False)
            ax.set_xscale("symlog", linthresh=0.001)
            ax.set_xlabel(label)
            ax.set_ylabel("1-NN (%)")
            ax.set_title(f"{backbone_label(backbone)} · Train $K$={anchor}", fontsize=9)
    _dedup_legend(fig, list(axes.flat), ncol=6)
    return save_figure(fig, out_dir, f"efficiency_{backbone}", caption=(
        "ImageNet 1-NN accuracy versus measured search time for the full validation set and "
        "index memory. Curves connect increasing inference K from the same training checkpoint. "
        "Search time excludes encoding. It is wall time, not isolated per-query GPU latency. "
        f"Missing costs are omitted. {seed_caption(seeds)}"))


# ----------------------------------------------------------------------- driver

def figure_e2_cost(rows, out_dir, backbone, *, seeds=SEEDS):
    subset = _for_backbone(rows, backbone)
    anchors = sorted({r["anchor_k"] for r in subset})
    fields = [(field, label, scale) for field, label, scale in
              (("search_seconds", "Search time (s)", 1),
               ("retrieval_ms", "Top-1 search (ms/batch)", 1),
               ("index_bytes", "Index (MiB)", 1 / 2**20))
              if any(r.get(f"{field}_mean") is not None for r in subset)]
    if not anchors or not fields:
        return []
    plt = _pyplot()
    fig, axes = plt.subplots(len(anchors), len(fields), squeeze=False,
                             figsize=(PAGE_WIDTH if len(fields) > 1 else COLUMN_WIDTH,
                                      2.45 * len(anchors) + 0.4), layout="constrained")
    for i, anchor in enumerate(anchors):
        panel = [r for r in subset if r["anchor_k"] == anchor]
        for ax, (field, label, scale) in zip(axes[i], fields):
            for method in _methods_present(panel):
                xs, means, stds, counts = _series(panel, method=method, y=field)
                if xs:
                    _plot_method(ax, xs, [v * scale for v in means],
                                 [v * scale if v is not None else None for v in stds],
                                 counts, method=method)
            _log2_k_axis(ax, [r["k"] for r in panel], label="Inference $K$")
            ax.set_ylabel(label)
            ax.set_title(f"Train $K$={anchor}", fontsize=9)
    _dedup_legend(fig, list(axes.flat), ncol=6)
    fig.suptitle(backbone_label(backbone))
    return save_figure(fig, out_dir, f"e2_cost_{backbone}", caption=(
        "Costs from the same fixed SAE checkpoint at each inference K. Search time covers "
        "the full ImageNet validation query set, excluding encoding. Top-1 search times sparse "
        "scoring, max/argmax and merging across gallery shards per configured query batch "
        "(512 by default), not per-query "
        "latency or end-to-end retrieval time. Index memory is from that benchmark. Timing "
        f"settings and skipped measurements remain in eval_e2.json. {seed_caption(seeds)}"))


def build_all(root: str | Path, out_dir: str | Path, *,
              backbones: Sequence[str] | None = None,
              records: Sequence[Mapping[str, Any]] | None = None,
              dense_references: Mapping[str, Mapping[str, Any]] | None = None,
              seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    """Load, aggregate, and write every table and figure. Returns a manifest.

    Nothing here raises on missing data. A partially finished grid is the normal
    state of this project, so each figure either renders what exists or records
    why it could not, and one empty panel never takes the report down with it.

    Without ``records`` this preserves the original workspace discovery. A
    campaign passes explicit raw records and dense references, so old runs and
    logs outside its plan cannot enter the report. An explicit empty list means
    no selected runs, never "fall back to every run on disk".
    """
    log = get_logger("report")
    out_dir = Path(out_dir)
    explicit = records is not None
    seeds = tuple(seeds)
    if explicit:
        wanted_backbones = set(backbones) if backbones is not None else None
        records = [dict(record) for record in records
                   if record.get("method") in METHODS and record["seed"] in seeds
                   and (wanted_backbones is None or record["backbone"] in wanted_backbones)]
        selected_cells = {(record["backbone"], record["method"], record["k"], record["seed"])
                          for record in records}
        if len(selected_cells) != len(records):
            raise ValueError("explicit report selection contains duplicate training cells")
        for record in records:
            validate_result_variant(record)
    else:
        records = load_runs(root, backbones=backbones)
        selected_cells = None
    e1 = aggregate_e1(records, seeds=seeds)
    e2 = aggregate_e2(records, seeds=seeds)
    training = training_summary(root, backbones=backbones, selected_cells=selected_cells)
    if dense_references is not None:
        dense = {name: dict(payload) for name, payload in dense_references.items()
                 if backbones is None or name in backbones}
    else:
        dense = {} if explicit else load_dense_references(root)

    manifest: dict[str, Any] = {
        "root": str(root), "out_dir": str(out_dir), "fonts": None,
        "selection": "explicit" if explicit else "workspace", "seeds": list(seeds),
        "counts": {"runs": len(records), "runs_with_e1": sum(1 for r in records if r["complete"]),
                   "e1_cells": len(e1), "e2_points": len(e2), "training_logs": len(training)},
        "tables": {}, "figures": {}, "skipped": [],
    }

    table_dir = out_dir / "tables"
    for name, table_rows in (("e1_accuracy", e1), ("e2_generalisation", e2),
                             ("training_summary", training)):
        manifest["tables"][name] = {key: str(path) for key, path
                                    in write_tables(table_rows, table_dir, name).items()}
    pairs = [(treatment, control) for control in BASELINES
             for treatment in OURS if treatment != control]
    for treatment, control in pairs:
        rows = paired_difference(records, treatment=treatment, control=control, seeds=seeds)
        if not rows:
            manifest["skipped"].append(
                f"table paired_{treatment}_vs_{control}: no paired seeds")
            continue
        name = f"paired_{treatment}_vs_{control}"
        manifest["tables"][name] = {key: str(path) for key, path
                                    in write_tables(rows, table_dir, name).items()}
    if dense:
        rows = [{"backbone": name, "dim": BACKBONES[name].dim if name in BACKBONES else None,
                 "top1_mean": payload.get("top1"), "top1_std": None, "top1_n": 1,
                 "probe_top1_mean": payload.get("probe_top1"), "probe_top1_std": None,
                 "probe_top1_n": 1 if payload.get("probe_top1") is not None else 0}
                for name, payload in dense.items()]
        manifest["tables"]["dense_reference"] = {
            key: str(path) for key, path
            in write_tables(rows, table_dir, "dense_reference").items()}

    # Tables are written before the style is installed, so a machine without
    # matplotlib still produces every CSV and Markdown table it can.
    try:
        manifest["fonts"] = apply_style()
    except Exception as error:
        log.warning("matplotlib unavailable (%s: %s); tables written, figures skipped",
                    type(error).__name__, error)
        manifest["skipped"].append(f"all figures: {type(error).__name__}: {error}")
        return manifest

    figure_dir = out_dir / "figures"
    measured_backbones = {row["backbone"] for row in e1 + e2 + training}
    present = [name for name in BACKBONE_ORDER if name in measured_backbones]
    present += sorted(measured_backbones - set(present))

    jobs: list[tuple[str, Any]] = [
        ("inference_summary", lambda: _inference_figure(e2, figure_dir, "inference_summary",
                                                       seeds=seeds, dense=dense)),
        ("backbone_summary", lambda: figure_backbone_summary(e1, figure_dir, dense=dense,
                                                             seeds=seeds)),
    ]
    for backbone in present:
        jobs.extend([
            (f"e1_curves_{backbone}",
             lambda b=backbone: figure_e1_curves(e1, figure_dir, b, measured_dense=dense.get(b),
                                                seeds=seeds)),
            (f"e1_delta_{backbone}_vs_csr",
             lambda b=backbone: figure_e1_delta(records, figure_dir, b, control="csr", seeds=seeds)),
            (f"e1_delta_{backbone}_vs_csrv2",
             lambda b=backbone: figure_e1_delta(records, figure_dir, b, control="csrv2", seeds=seeds)),
            (f"e2_generalisation_{backbone}",
             lambda b=backbone: figure_e2_generalisation(e2, figure_dir, b, seeds=seeds, measured_dense=dense.get(b))),
            (f"e2_nnz_{backbone}",
             lambda b=backbone: figure_e2_nnz(e2, figure_dir, b, seeds=seeds)),
            (f"e2_cost_{backbone}", lambda b=backbone: figure_e2_cost(e2, figure_dir, b, seeds=seeds)),
            (f"efficiency_{backbone}", lambda b=backbone: figure_efficiency(e2 or e1, figure_dir, b, seeds=seeds)),
            (f"training_{backbone}", lambda b=backbone: figure_training(training, figure_dir, b)),
        ])

    for name, job in jobs:
        try:
            paths = job()
        except Exception as error:  # one bad panel must not lose the whole report
            log.warning("figure %s failed (%s: %s)", name, type(error).__name__, error,
                        exc_info=True)
            manifest["skipped"].append(f"figure {name}: {type(error).__name__}: {error}")
            continue
        if paths:
            manifest["figures"][name] = [str(path) for path in paths]
        else:
            manifest["skipped"].append(f"figure {name}: no data")

    log.info("report: %d table(s), %d figure(s), %d skipped -> %s",
             len(manifest["tables"]), len(manifest["figures"]), len(manifest["skipped"]), out_dir)
    for note in manifest["skipped"]:
        log.info("skipped %s", note)
    return manifest
