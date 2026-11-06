"""Tables and publication figures for experiments E1 and E2.

``aggregate`` is pure standard library, so a table renders anywhere; ``style``
and ``figures`` import matplotlib lazily, inside the functions that draw, so
importing this package on a machine without matplotlib still works and still
produces CSV/Markdown. ``reference`` holds numbers transcribed from the papers
and is never mixed into a measurement of ours.
"""
from __future__ import annotations

from . import reference
from .aggregate import (aggregate_e1, aggregate_e2, load_dense_references, load_runs,
                        paired_difference, runs_dir, training_summary, write_tables)
from .figures import (build_all, figure_backbone_summary, figure_e1_curves, figure_e1_delta,
                      figure_e2_generalisation, figure_e2_nnz, figure_efficiency,
                      figure_training)
from .style import LABELS, METHOD_STYLE, apply_style, save_figure

__all__ = [
    "reference",
    # aggregation
    "load_runs", "load_dense_references", "runs_dir",
    "aggregate_e1", "aggregate_e2", "paired_difference", "training_summary", "write_tables",
    # style
    "apply_style", "save_figure", "METHOD_STYLE", "LABELS",
    # figures
    "build_all", "figure_e1_curves", "figure_e1_delta", "figure_e2_generalisation",
    "figure_e2_nnz", "figure_training", "figure_backbone_summary", "figure_efficiency",
]
