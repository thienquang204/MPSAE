"""CSRv2's k-annealing curriculum.

Transcribed from ``get_current_k`` in ``CSRv2/image/main_visual.py`` (identical
to ``CSRv2/text/CSR_training.py``). The schedule is **linear** over the first
``k_decay_ratio`` of the optimiser steps; a cosine shape is explicitly reported
as worse in the paper's own sensitivity study (App. G.2), so using one would be
a different algorithm, not a different setting.

The step counter is per batch and is passed *before* being incremented, so the
first batch of training sees ``step = 0`` and therefore ``k = k_init``.
"""
from __future__ import annotations

from .config import initial_topk  # re-exported: the k_init rule lives with the presets

__all__ = ["current_k", "initial_topk", "annealing_table"]


def current_k(step: int, initial_k: int, final_k: int, total_steps: int,
              k_decay_ratio: float = 0.7) -> int:
    """Linearly interpolate K from ``initial_k`` to ``final_k``.

    ``k_decay_steps = int(total_steps * k_decay_ratio)``; once ``step`` reaches
    it, K stays at ``final_k``. The truncation to ``int`` and Python's
    round-half-to-even are both part of the released behaviour, and a
    ``k_decay_steps`` of zero short-circuits before any division.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a nonnegative integer, got {step!r}")
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if final_k < 1 or initial_k < final_k:
        raise ValueError(f"require 1 <= final_k <= initial_k, got {final_k} and {initial_k}")
    if not 0 < k_decay_ratio <= 1:
        raise ValueError("k_decay_ratio must be in (0, 1]")
    decay_steps = int(total_steps * k_decay_ratio)
    if step >= decay_steps:
        return final_k
    progress = step / decay_steps
    return round(initial_k * (1 - progress) + final_k * progress)


def annealing_table(initial_k: int, final_k: int, total_steps: int,
                    k_decay_ratio: float = 0.7, *, points: int = 12
                    ) -> list[tuple[int, int]]:
    """``(step, k)`` samples across training, for logs and figures."""
    if points < 2:
        raise ValueError("points must be at least two")
    steps = sorted({min(total_steps - 1, round(index * (total_steps - 1) / (points - 1)))
                    for index in range(points)})
    return [(step, current_k(step, initial_k, final_k, total_steps, k_decay_ratio))
            for step in steps]
