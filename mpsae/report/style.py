"""The one place that owns how a figure looks.

Design decisions, because they are arguments and not taste:

*Colour carries meaning.* Baselines are cool and desaturated; our two methods
are a warm, saturated accent; the ablation is grey; published references are
neutral grey dashes. A reader scanning the paper should see at a glance which
curve is the subject of the figure. The hues are from the Okabe-Ito
colour-blind-safe set (vermillion, amber, blue, sky) adjusted in lightness, so
no pair of series relies on a red/green distinction.

*Every series is also distinguishable without colour.* Each entry fixes a
marker and a line style as well as a colour, and the luminances within each
semantic group are separated, so the figures survive greyscale printing and
photocopying -- which is how reviewers often read them.

*Type is embedded and editable.* ``pdf.fonttype = 42`` / ``ps.fonttype = 42``
emit Type-42 (TrueType) rather than Type-3, so the vector PDFs keep selectable,
searchable text and pass the camera-ready font checks.

Matplotlib only; no seaborn, which would reset rcParams behind our back.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..logs import get_logger

#: Preferred faces, most preferred first. Resolved against the installed fonts
#: at ``apply_style()`` time; the last entry ships with matplotlib itself, so
#: the list can never fail to resolve.
SERIF_CANDIDATES = ("Libertinus Serif", "Linux Libertine O", "STIX Two Text",
                    "Times New Roman", "Nimbus Roman", "DejaVu Serif")
SANS_CANDIDATES = ("Inter", "Helvetica Neue", "Helvetica", "Arial",
                   "Nimbus Sans", "DejaVu Sans")

BASE_FONT_SIZE = 9.0
TICK_FONT_SIZE = 8.0
TITLE_FONT_SIZE = 10.0


# ----------------------------------------------------------------- semantics

#: Warm saturated accent: ours, the subject of every figure.
_OURS_PRIMARY = "#D55E00"     # Okabe-Ito vermillion, dark (L* ~ 50)
_OURS_SECONDARY = "#E69F00"   # Okabe-Ito amber, light (L* ~ 72) -- separated in
                              # luminance from vermillion so the two ours-curves
                              # stay distinct in greyscale.
#: Cool desaturated: published-algorithm baselines, deliberately recessive.
_BASE_PRIMARY = "#5B7FA6"     # muted steel blue (desaturated Okabe-Ito blue)
_BASE_SECONDARY = "#3C6E71"   # dark teal; darker than the steel blue
#: Neutral: the ablation is a control, not a competitor.
_ABLATION = "#8A8A8A"
#: Neutral, lighter still: numbers we did not measure.
_PUBLISHED = "#9E9E9E"
#: Our own dense upper reference.
_DENSE = "#4D4D4D"

#: Per-method drawing instructions. ``zorder`` encodes the reading order we
#: want: published references behind, baselines next, ours on top.
METHOD_STYLE: Mapping[str, Mapping[str, Any]] = {
    "csr": {"color": _BASE_PRIMARY, "marker": "o", "linestyle": "--",
            "label": "CSR", "zorder": 2.0},
    "csrv2": {"color": _BASE_SECONDARY, "marker": "s", "linestyle": ":",
              "label": "CSRv2", "zorder": 2.1},
    "csr_multitopk": {"color": _ABLATION, "marker": "^", "linestyle": "-.",
                      "label": "CSR + multi-TopK (ablation)", "zorder": 1.9},
    "mpsae": {"color": _OURS_PRIMARY, "marker": "o", "linestyle": "-",
              "label": "MPSAE (ours)", "zorder": 3.0},
    "mpsaev2": {"color": _OURS_SECONDARY, "marker": "D", "linestyle": "-",
                "label": "MPSAEv2 (ours)", "zorder": 3.1},
}

#: Published overlays: one neutral grey dashed language, markers only to tell
#: the series apart, so no reader mistakes them for a measurement of ours.
PUBLISHED_STYLE: Mapping[str, Mapping[str, Any]] = {
    "csr": {"color": _PUBLISHED, "marker": "x", "linestyle": "--",
            "label": "CSR (published)", "zorder": 1.2},
    "mrl": {"color": _PUBLISHED, "marker": "+", "linestyle": "--",
            "label": "MRL (published)", "zorder": 1.1},
    "csrv2_linear": {"color": _PUBLISHED, "marker": "1", "linestyle": "--",
                     "label": "CSRv2-linear (published)", "zorder": 1.3},
}

#: Style for a horizontal "upper reference" rule (dense, un-compressed).
DENSE_STYLE: Mapping[str, Any] = {"color": _DENSE, "linestyle": (0, (6, 3)),
                                  "linewidth": 0.9, "zorder": 1.0}

#: Style for the faint "trained at this K" backdrop in the E2 panels.
BACKDROP_STYLE: Mapping[str, Any] = {"linewidth": 0.8, "alpha": 0.25, "zorder": 0.8}

LABELS: Mapping[str, str] = {name: str(entry["label"]) for name, entry in METHOD_STYLE.items()}

BACKBONE_LABELS: Mapping[str, str] = {
    "ff2048": "FF2048 (ResNet-50, d=2048)",
    "rn18": "ResNet-18 (d=512)",
    "convnext_tiny": "ConvNeXt-T (d=768)",
    "vit_b16": "ViT-B/16 (d=768)",
    "dinov2_vits14": "DINOv2 ViT-S/14 (d=384)",
}

#: Opacity of the mean ± std band. Low enough that overlapping bands from five
#: methods stay readable, high enough to survive a greyscale print.
BAND_ALPHA = 0.16

#: Single-column and double-column widths in inches for a two-column template.
COLUMN_WIDTH = 3.35
PAGE_WIDTH = 6.9


def method_style(method: str) -> dict[str, Any]:
    """Drawing kwargs for a method, with a visible fallback for unknown names.

    An unregistered method must still plot -- a new ablation should not crash a
    report -- but it is rendered in a plain neutral style so nobody mistakes it
    for one of the five configured configurations.
    """
    entry = METHOD_STYLE.get(method)
    if entry is not None:
        return dict(entry)
    return {"color": "#000000", "marker": ".", "linestyle": "-",
            "label": f"{method} (unregistered)", "zorder": 2.5}


def label_for(method: str) -> str:
    return LABELS.get(method, method)


def backbone_label(backbone: str) -> str:
    return BACKBONE_LABELS.get(backbone, backbone)


# --------------------------------------------------------------------- rcParams

def _first_available(candidates: Sequence[str]) -> list[str]:
    """Candidates that are actually installed, in preference order.

    Resolving here instead of trusting matplotlib's fallback chain keeps the
    figure from silently rendering in a face we did not choose, and keeps a
    missing font from raising at draw time. ``DejaVu`` ships with matplotlib, so
    the returned list is never empty.
    """
    try:
        from matplotlib import font_manager

        installed = {font.name for font in font_manager.fontManager.ttflist}
    except Exception as error:  # a broken font cache must not break the report
        get_logger("report").warning("font enumeration failed (%s: %s); using matplotlib defaults",
                                     type(error).__name__, error)
        return list(candidates)
    found = [name for name in candidates if name in installed]
    missing = [name for name in candidates[:-1] if name not in installed]
    if missing and found:
        get_logger("report").debug("fonts unavailable: %s; using %s",
                                   ", ".join(missing), found[0])
    return found or list(candidates)


def apply_style(*, family: str = "serif") -> dict[str, Any]:
    """Install the paper rcParams globally and return what was chosen.

    ``family`` picks the serif (default, matches most CV templates' body text)
    or the neutral sans. This is the only global state the report package owns;
    everything else is passed as arguments.
    """
    import matplotlib as mpl

    if family not in ("serif", "sans-serif"):
        raise ValueError("family must be 'serif' or 'sans-serif'")
    serif = _first_available(SERIF_CANDIDATES)
    sans = _first_available(SANS_CANDIDATES)
    mpl.rcParams.update({
        "font.family": family,
        "font.serif": serif,
        "font.sans-serif": sans,
        "mathtext.fontset": "dejavuserif" if family == "serif" else "dejavusans",
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "figure.titlesize": TITLE_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": TICK_FONT_SIZE,
        # Screen-legible while iterating, press-ready on save.
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        # Thin frame, no box: the data should be the heaviest ink on the page.
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        # Horizontal-only guides: we read values off the y-axis, never the x.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#B0B0B0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "lines.markeredgewidth": 0.0,
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.2,
        # Type-42 so the PDF carries editable, searchable, embeddable text.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
    })
    chosen = {"family": family, "serif": serif[0] if serif else None,
              "sans-serif": sans[0] if sans else None}
    get_logger("report").debug("style: %s", chosen)
    return chosen


# ----------------------------------------------------------------------- saving

def save_figure(fig: Any, out_dir: str | Path, name: str, *,
                caption: str | None = None) -> list[Path]:
    """Write ``<name>.png`` and ``<name>.pdf`` with identical geometry, then close.

    Both formats come from the same figure object and the same ``bbox_inches``,
    so the raster preview and the vector version a paper imports cannot drift
    apart. A ``caption`` is written beside them as ``<name>.caption.txt`` so the
    claim a figure supports lives with the figure instead of being re-invented
    in LaTeX. The figure is closed because ``build_all`` draws dozens and
    matplotlib keeps every open figure alive.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        for suffix in (".png", ".pdf"):
            path = out_dir / f"{name}{suffix}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
            written.append(path)
        if caption:
            path = out_dir / f"{name}.caption.txt"
            path.write_text(caption.strip() + "\n", encoding="utf-8")
            written.append(path)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
    get_logger("report").info("figure %s -> %s", name,
                              ", ".join(path.name for path in written))
    return written
