"""Publication figures for experiments E1 and E2.

Each function consumes already-aggregated rows -- the figures never touch the
run tree, so what a panel shows and what the corresponding table says are the
same numbers by construction. Each returns the paths it wrote, or an empty list
when the data it needs is absent: a half-finished grid should still produce
every figure it can, and ``build_all`` must not die because one panel is empty.

Which figure answers which claim:

* ``figure_e1_curves``   -- does our objective beat CSR/CSRv2 at matched K?
* ``figure_e1_delta``    -- the same claim, paired per seed. This is the figure
  the headline sentence of the paper points at.
* ``figure_e2_generalisation`` -- CSRv2 §3 asserts that CSR's multi-TopK loss
  "ensures CSR could generalize to different k at test time". Neither paper
  measures it. This panel grid does.
* ``figure_e2_nnz``      -- where a larger nominal K stops buying real sparsity,
  because after the ReLU a code can hold far fewer than K nonzeros.
* ``figure_training``    -- our diagnostic, not a paper figure.
* ``figure_backbone_summary`` -- the headline overview across all five backbones.
* ``figure_efficiency``  -- the accuracy/latency/footprint trade-off CSR argues
  for, in the two units a practitioner actually pays in.

Captions are written beside every figure (``<name>.caption.txt``) so the claim
travels with the image.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..config import BACKBONE_ORDER, BACKBONES, BASELINES, METHODS, OURS, SEEDS
from ..logs import get_logger
from . import reference
from .aggregate import (aggregate_e1, aggregate_e2, load_dense_references, load_runs,
                        paired_difference, training_summary, validate_result_variant, write_tables)
from .style import (BACKDROP_STYLE, BAND_ALPHA, COLUMN_WIDTH, DENSE_STYLE, PAGE_WIDTH,
                    PUBLISHED_STYLE, apply_style, backbone_label, label_for,
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
    return [row for row in rows if row.get("backbone") == backbone]


def _methods_present(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Methods with data, in the canonical order so the legend never shuffles."""
    found = {row["method"] for row in rows if row.get("method")}
    ordered = [method for method in METHODS if method in found]
    return ordered + sorted(found - set(ordered))


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


def _n_note(counts: Sequence[int]) -> str:
    """``" (n=3)"`` or ``" (n=2-3)"``. A mean is never shown without its n."""
    values = sorted({count for count in counts if count})
    if not values:
        return ""
    if len(values) == 1:
        return f" (n={values[0]})"
    return f" (n={values[0]}–{values[-1]})"


def _log2_k_axis(ax: Any, ks: Sequence[float], *, label: str = "Active dimension $K$",
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
    ax.plot(xs, means, label=f"{text}{_n_note(counts)}", **kwargs)
    if band and any(std is not None for std in stds):
        lower = [mean - (std or 0.0) for mean, std in zip(means, stds)]
        upper = [mean + (std or 0.0) for mean, std in zip(means, stds)]
        ax.fill_between(xs, lower, upper, where=[std is not None for std in stds],
                        color=style["color"], alpha=BAND_ALPHA, linewidth=0.0,
                        zorder=style["zorder"] - 0.5)


def _overlay_published(ax: Any, backbone: str) -> list[str]:
    """Draw the published reference curves. Returns their provenance strings."""
    provenance: list[str] = []
    for entry in reference.overlays(backbone):
        style = dict(PUBLISHED_STYLE.get(str(entry["series"]), {}))
        curve = entry["curve"]
        if not isinstance(curve, Mapping) or not curve:
            continue
        ks = sorted(curve)
        ax.plot(ks, [curve[k] for k in ks], color=style.get("color", "#9E9E9E"),
                marker=style.get("marker", "x"), linestyle=style.get("linestyle", "--"),
                linewidth=0.9, markersize=4.0, markeredgewidth=0.9,
                zorder=style.get("zorder", 1.0), label=str(entry["label"]))
        provenance.append(f"{entry['label']}: {entry['provenance']}")
    return provenance


def _annotate_rule(ax: Any, value: float, text: str, *, color: str | None = None,
                   side: str = "right", below: bool = False) -> None:
    """Horizontal upper-reference rule with its label pinned to one edge.

    The published dense accuracy and our measured one are usually within a
    fraction of a point of each other -- that closeness is the result, not a
    problem -- so their labels would overprint if both sat in the same corner.
    Callers place one left and one right.
    """
    kwargs = dict(DENSE_STYLE)
    if color is not None:
        kwargs["color"] = color
    ax.axhline(value, **kwargs)
    x = 1.0 if side == "right" else 0.0
    offset = (-3, -3.5) if below else (-3, 3.0)
    if side != "right":
        offset = (3, offset[1])
    ax.annotate(text, xy=(x, value), xycoords=("axes fraction", "data"),
                xytext=offset, textcoords="offset points",
                ha=side, va="top" if below else "bottom",
                fontsize=6.8, color=kwargs["color"])


#: Trailing " (n=3)" / " (n=2–3)" added by ``_n_note``.
_N_SUFFIX = re.compile(r"\s*\(n=[0-9]+(?:–[0-9]+)?\)$")


def _dedup_legend(fig: Any, axes: Sequence[Any], *, ncol: int, y: float = -0.01,
                  strip_counts: bool = False) -> None:
    """One shared legend below the panels, each label once, in first-seen order.

    ``strip_counts`` drops the ``(n=...)`` suffix before de-duplicating. Across
    panels the seed count can legitimately differ -- one backbone may have a
    missing seed -- and without this the same method would appear twice in the
    shared legend. The per-panel counts stay in the tables, which is where a
    reader checks them.
    """
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if strip_counts:
                label = _N_SUFFIX.sub("", label)
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if not handles:
        return
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
    the caption says so. Published curves are overlaid only on ``ff2048``,
    where they are measured on the same backbone, and only in grey dashes.
    """
    log = get_logger("report")
    subset = _for_backbone(rows, backbone)
    if not subset:
        log.info("skip figure_e1_curves[%s]: no aggregated E1 rows", backbone)
        return []
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.55))
    ks: list[float] = []
    for method in _methods_present(subset):
        xs, means, stds, counts = _series(subset, method=method)
        if not xs:
            continue
        ks.extend(xs)
        _plot_method(ax, xs, means, stds, counts, method=method)

    provenance = _overlay_published(ax, backbone)
    published_dense = reference.dense_reference(backbone)
    if published_dense is not None:
        _annotate_rule(ax, float(published_dense["top1"]),
                       f"{published_dense['label']} {published_dense['top1']:.2f}", side="right")
        provenance.append(f"{published_dense['label']}: {published_dense['provenance']}")
    if measured_dense and measured_dense.get("top1") is not None:
        # Left edge, and below the rule: it normally sits within ~0.05 of the
        # published value, so sharing the right corner would overprint it.
        _annotate_rule(ax, float(measured_dense["top1"]),
                       f"dense, ours {float(measured_dense['top1']):.2f}",
                       color="#111111", side="left", below=True)

    _log2_k_axis(ax, ks)
    ax.set_ylabel("ImageNet-1K 1-NN top-1 (%)")
    ax.set_title(backbone_label(backbone))
    ax.legend(loc="lower right", ncol=1, fontsize=6.9)
    caption = (
        "1-NN top-1 on ImageNet-1K as a function of the active dimension K, "
        f"{backbone_label(backbone)}. "
        f"{seed_caption(seeds)} "
        "Grey dashed series are values transcribed from the published papers, not measured "
        "by us"
        + (" (" + "; ".join(provenance) + ")." if provenance else ".")
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

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
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
                    label=f"{style['label']} − {label_for(control)}"
                          f"{_n_note([n for _, _, _, n in points])}")

    ax.axhline(0.0, color="#444444", linewidth=0.7, linestyle="-", zorder=1.0)
    _log2_k_axis(ax, ks)
    ax.set_ylabel(f"Δ top-1 vs {label_for(control)} (pp)")
    ax.set_title(f"{backbone_label(backbone)} — paired over seeds")
    ax.legend(loc="best", fontsize=6.9)
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

def figure_e2_generalisation(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                             backbone: str, *,
                             e1_rows: Sequence[Mapping[str, Any]] | None = None,
                             seeds: Sequence[int] = SEEDS) -> list[Path]:
    """Small multiples, one panel per E2 anchor K. ``rows`` from ``aggregate_e2``.

    ``e1_rows`` (from ``aggregate_e1``) is drawn faintly behind each panel as
    the "trained at this K" reference, rather than a guaranteed upper bound.
    """
    log = get_logger("report")
    subset = _for_backbone(rows, backbone)
    if not subset:
        log.info("skip figure_e2_generalisation[%s]: no aggregated E2 rows", backbone)
        return []
    anchors = sorted({int(row["anchor_k"]) for row in subset})
    methods = _methods_present(subset)
    plt = _pyplot()
    ncol = min(3, len(anchors))
    nrow = (len(anchors) + ncol - 1) // ncol
    fig, grid_axes = plt.subplots(nrow, ncol, figsize=(PAGE_WIDTH, 1.95 * nrow + 0.5),
                                 squeeze=False, sharey=True, layout="constrained")
    axes = [grid_axes[row][col] for row in range(nrow) for col in range(ncol)]
    e1_subset = _for_backbone(e1_rows or [], backbone)

    for ax, anchor in zip(axes, anchors):
        panel = [row for row in subset if int(row["anchor_k"]) == anchor]
        ks: list[float] = []
        for method in methods:
            xs, means, stds, counts = _series(panel, method=method)
            if not xs:
                continue
            ks.extend(xs)
            _plot_method(ax, xs, means, stds, counts, method=method, markersize=2.8)
            at_anchor = [(x, y) for x, y in zip(xs, means) if x == anchor]
            if at_anchor:
                ax.scatter(*zip(*at_anchor), s=30, facecolors="none",
                           edgecolors=method_style(method)["color"], linewidths=0.8,
                           zorder=5, label="_nolegend_")
            if e1_subset:
                bx, bmeans, _, _ = _series(e1_subset, method=method)
                if bx:
                    ax.plot(bx, bmeans, color=method_style(method)["color"],
                            linestyle="-", marker="", label="_nolegend_", **BACKDROP_STYLE)
        ax.axvline(anchor, color="#444444", linewidth=0.7, linestyle=(0, (2, 2)), zorder=0.9)
        # Flip the label to the left of the rule once the anchor is past the
        # midpoint of the budget range, otherwise it runs off the panel.
        budgets = sorted({float(value) for value in ks}) or [float(anchor)]
        past_middle = anchor >= budgets[len(budgets) // 2]
        ax.annotate(f"trained at $K^\\ast$={anchor}", xy=(anchor, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(-3 if past_middle else 3, -8), textcoords="offset points",
                    fontsize=6.6, color="#444444", ha="right" if past_middle else "left")
        _log2_k_axis(ax, ks, label="Inference $K$")
        ax.set_title(f"$K^\\ast$ = {anchor}", fontsize=8.4)
    for ax in axes[len(anchors):]:
        ax.set_visible(False)
    for row in range(nrow):
        grid_axes[row][0].set_ylabel("1-NN top-1 (%)")

    _dedup_legend(fig, axes[:len(anchors)], ncol=min(5, len(methods)), y=-0.03)
    fig.suptitle(f"One checkpoint, every inference K — {backbone_label(backbone)}")
    caption = (
        "A single checkpoint trained at K* (dashed vertical) evaluated at every K in the grid, "
        f"{backbone_label(backbone)}. CSRv2 §3 states that CSR's multi-TopK loss \"ensures CSR "
        "could generalize to different k at test time\"; neither paper measures it, and this is "
        "that measurement. The faint line behind each panel is the per-K E1 curve of the same "
        "method — models trained at each K — and is a per-K reference, not a guaranteed "
        f"upper bound. Hollow rings mark inference K equal to training K*. {seed_caption(seeds)}"
    )
    return save_figure(fig, out_dir, f"e2_generalisation_{backbone}", caption=caption)


def figure_e2_nnz(rows: Sequence[Mapping[str, Any]], out_dir: str | Path, backbone: str, *,
                  e1_rows: Sequence[Mapping[str, Any]] | None = None,
                  seeds: Sequence[int] = SEEDS) -> list[Path]:
    """Accuracy vs the *measured* mean nnz. ``rows`` from ``aggregate_e2``.

    Nominal K is an upper bound on the number of nonzeros, not the count: the
    Top-K gate is followed by a ReLU, so negative selected pre-activations are
    zeroed and a code can be far sparser than its budget. Plotting against the
    measured nnz shows where raising K stops buying anything real.
    """
    log = get_logger("report")
    subset = _for_backbone(rows, backbone)
    e1_subset = _for_backbone(e1_rows or [], backbone)
    if not subset and not e1_subset:
        log.info("skip figure_e2_nnz[%s]: no rows carrying nnz", backbone)
        return []
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.55))
    drew = False
    for method in _methods_present(list(e1_subset) + list(subset)):
        xs, means, stds, counts = _series(e1_subset, method=method, x="nnz")
        if xs:
            drew = True
            _plot_method(ax, xs, means, stds, counts, method=method,
                         label=f"{label_for(method)}, trained at each $K$")
        # Each anchor checkpoint sweeps its own nnz range; thin and unlabelled so
        # the trained-at-each-K curves stay the subject of the panel.
        for anchor in sorted({int(row["anchor_k"]) for row in subset
                              if row["method"] == method}):
            panel = [row for row in subset
                     if row["method"] == method and int(row["anchor_k"]) == anchor]
            ax_x, ax_y, _, _ = _series(panel, method=method, x="nnz")
            if ax_x:
                drew = True
                ax.plot(ax_x, ax_y, color=method_style(method)["color"], marker="",
                        linestyle="-", label="_nolegend_", **BACKDROP_STYLE)
    if not drew:
        log.info("skip figure_e2_nnz[%s]: nnz_mean missing from every row", backbone)
        plt.close(fig)
        return []

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Measured mean nonzeros per code")
    ax.set_ylabel("1-NN top-1 (%)")
    ax.set_title(f"Accuracy vs realised sparsity — {backbone_label(backbone)}")
    ax.legend(loc="lower right", fontsize=6.6)
    caption = (
        f"1-NN top-1 against the mean number of nonzeros actually present in the code, "
        f"{backbone_label(backbone)}. The Top-K gate is followed by a ReLU, so a code can hold "
        "fewer than K nonzeros and the nominal budget overstates the cost; the point at which a "
        "curve flattens is the point at which raising K stops buying real sparsity. Bold curves "
        "are models trained at each K (E1); faint curves are single E2 checkpoints swept across "
        f"inference K. {seed_caption(seeds)}"
    )
    return save_figure(fig, out_dir, f"e2_nnz_{backbone}", caption=caption)


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
    subset = [row for row in rows if row.get("backbone") == backbone and row.get("curve")]
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
    # The contrastive and OT terms are mutually exclusive by method, so one
    # "coupling" panel can show both without ever overlaying them.
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
            (ax_parts, "Loss components (line style)", "loss", True),
            (ax_k, "Realised k-annealing", "training $k_t$", False),
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
    fig.suptitle(f"Training diagnostics — {backbone_label(backbone)}, K={chosen}, all seeds")
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
    whose dense accuracies differ by tens of points would compress the
    interesting panels into a flat line, so the rule is applied to the data and
    the outcome is stated in the caption.
    """
    log = get_logger("report")
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
        measured = (dense or {}).get(backbone)
        if measured and measured.get("top1") is not None:
            _annotate_rule(ax, float(measured["top1"]),
                           f"dense {float(measured['top1']):.2f}", color="#111111")
        _log2_k_axis(ax, ks)
        ax.set_title(backbone_label(backbone), fontsize=8.4)
    for ax in axes[len(present):]:
        ax.set_visible(False)
    for row in range(nrow):
        grid_axes[row][0].set_ylabel("1-NN top-1 (%)")

    # Seed counts can differ between backbones, so they are dropped here rather
    # than splitting one method into two legend entries; the tables carry n.
    _dedup_legend(fig, axes[:len(present)], ncol=min(5, len(METHODS)), y=-0.02,
                  strip_counts=True)
    caption = (
        "1-NN top-1 on ImageNet-1K against the active dimension K, for every backbone. "
        f"{seed_caption(seeds)} Per-series n is in "
        "the accompanying table, since it can differ between panels. The black dashed rule in each "
        "panel is our own dense (un-compressed) 1-NN accuracy for that backbone. "
        + ("The y-axis is shared across panels."
           if shared else
           "The y-axis is per-panel: the backbones' dense accuracies differ by more than "
           f"{share_y_span:.0f} points, and a shared axis would flatten the smaller panels.")
    )
    return save_figure(fig, out_dir, "backbone_summary", caption=caption)


# ------------------------------------------------------------------ efficiency

def figure_efficiency(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                      backbone: str) -> list[Path]:
    """Accuracy vs retrieval time and vs index size. ``rows`` from ``aggregate_e1``.

    This is the trade-off CSR argues for, in the two units a deployment pays in:
    query latency and bytes on disk. K increases along each curve, so a curve
    that bends left-and-up is strictly better at equal cost.
    """
    log = get_logger("report")
    subset = _for_backbone(rows, backbone)
    panels = [("search_seconds", "Retrieval wall-clock for 50K queries (s)"),
              ("index_bytes", "Index size (bytes)")]
    usable = [(field, label) for field, label in panels
              if any(row.get(f"{field}_mean") is not None for row in subset)]
    if not usable:
        log.info("skip figure_efficiency[%s]: neither timing nor index size recorded", backbone)
        return []

    plt = _pyplot()
    fig, grid_axes = plt.subplots(1, len(usable), figsize=(COLUMN_WIDTH * len(usable), 2.5),
                                 squeeze=False, sharey=True, layout="constrained")
    axes = [grid_axes[0][index] for index in range(len(usable))]
    for ax, (field, xlabel) in zip(axes, usable):
        for method in _methods_present(subset):
            xs, means, stds, counts = _series(subset, method=method, x=field)
            if not xs:
                continue
            _plot_method(ax, xs, means, stds, counts, method=method, band=False,
                         markersize=3.2)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("1-NN top-1 (%)")
    _dedup_legend(fig, axes, ncol=min(5, len(METHODS)), y=-0.05)
    fig.suptitle(f"Fidelity vs cost — {backbone_label(backbone)}")
    caption = (
        f"1-NN top-1 against retrieval cost on {backbone_label(backbone)}; K increases along each "
        "curve, so up-and-left is strictly better. Markers are means over seeds; the cost axes "
        "show no band because wall-clock time is hardware- and load-dependent and its spread is "
        "not a property of the method. Both axes are logarithmic."
    )
    return save_figure(fig, out_dir, f"efficiency_{backbone}", caption=caption)


# ----------------------------------------------------------------------- driver

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
                   if record["seed"] in seeds
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
             for treatment in tuple(OURS) + ("csr_multitopk",) if treatment != control]
    pairs.append(("mpsaev2", "mpsae"))  # fixed-K OT + CL versus fixed-K OT
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
                 "probe_top1_n": 1 if payload.get("probe_top1") is not None else 0,
                 "published_top1": (reference.dense_reference(name) or {}).get("top1")}
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
             lambda b=backbone: figure_e2_generalisation(e2, figure_dir, b, e1_rows=e1, seeds=seeds)),
            (f"e2_nnz_{backbone}",
             lambda b=backbone: figure_e2_nnz(e2, figure_dir, b, e1_rows=e1, seeds=seeds)),
            (f"efficiency_{backbone}", lambda b=backbone: figure_efficiency(e1, figure_dir, b)),
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
