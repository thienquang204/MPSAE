"""The CSRv2 k-annealing curriculum, against the released function.

``get_current_k`` is pure Python, so it is transcribed inline here rather than
imported from ``tests/reference`` (which pulls in torch). Source:
CSRv2 @ 3b2d4d2, ``image/main_visual.py``.
"""
from __future__ import annotations

import pytest

from mpsae.config import BACKBONES
from mpsae.schedules import annealing_table, current_k, initial_topk


def released_get_current_k(global_step, initial_k, final_k, total_steps, k_decay_ratio=0.7):
    k_decay_steps = int(total_steps * k_decay_ratio)
    if global_step >= k_decay_steps:
        return final_k
    progress = global_step / k_decay_steps
    return round(initial_k * (1 - progress) + final_k * progress)


@pytest.mark.parametrize("total_steps", [1, 10, 313, 3130, 12510])
@pytest.mark.parametrize("final_k", [2, 3, 8, 96, 2048])
def test_matches_the_released_function_at_every_step(total_steps, final_k):
    start = initial_topk(final_k, 8192)
    for step in range(total_steps):
        assert current_k(step, start, final_k, total_steps) == \
            released_get_current_k(step, start, final_k, total_steps)


def test_shape_is_linear_not_cosine():
    """A cosine shape is a different algorithm: CSRv2 App. G.2 reports it as worse."""
    start, final, total = 64, 8, 1000
    decay = int(total * 0.7)
    midpoint = current_k(decay // 2, start, final, total)
    linear = round(start + (final - start) * 0.5)
    assert midpoint == pytest.approx(linear, abs=1)
    # The cosine midpoint would sit at the arithmetic mean of a half-period,
    # which for this configuration is the same point -- so check a quarter too,
    # where the two shapes genuinely separate.
    quarter = current_k(decay // 4, start, final, total)
    assert quarter == pytest.approx(round(start + (final - start) * 0.25), abs=1)


def test_first_step_uses_k_init_and_the_tail_is_flat():
    """The release passes ``global_step`` before incrementing it."""
    start, final, total = 64, 8, 3130
    assert current_k(0, start, final, total) == start
    decay = int(total * 0.7)
    # Rounding reaches final_k slightly before the cutoff, so the last strictly
    # larger value sits a little earlier; the cutoff itself and the tail are flat.
    assert current_k(decay // 2, start, final, total) > final
    assert current_k(decay, start, final, total) == final
    assert current_k(total - 1, start, final, total) == final


def test_monotone_non_increasing():
    start, final, total = 2048, 8, 3130
    values = [current_k(step, start, final, total) for step in range(total)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
    assert values[0] == start and values[-1] == final


def test_k_init_rule_including_the_released_clamps():
    """Paper: 64 below 64, else 4k. Code then clamps to h//4 and up to k."""
    assert initial_topk(2, 8192) == 64
    assert initial_topk(63, 8192) == 64
    assert initial_topk(64, 8192) == 256
    assert initial_topk(96, 8192) == 384
    assert initial_topk(512, 8192) == 2048
    # 4k would exceed the dictionary, so the clamp to h//4 takes over.
    assert initial_topk(768, 8192) == 2048
    assert initial_topk(1024, 8192) == 2048
    # At the widest budget there is nothing left to anneal from.
    assert initial_topk(2048, 8192) == 2048


def test_no_backbone_anneals_past_its_dictionary():
    for name, spec in BACKBONES.items():
        for k in spec.k_list:
            start = initial_topk(k, spec.preset.hidden_dim)
            assert 4 * start <= spec.preset.hidden_dim, (name, k)


def test_rejects_nonsense():
    with pytest.raises(ValueError):
        current_k(-1, 64, 8, 100)
    with pytest.raises(ValueError):
        current_k(0, 8, 64, 100)       # initial below final
    with pytest.raises(ValueError):
        current_k(0, 64, 8, 0)
    with pytest.raises(ValueError):
        current_k(0, 64, 8, 100, 0.0)


def test_annealing_table_is_sampled_and_ends_flat():
    table = annealing_table(64, 8, 3130, points=8)
    assert table[0] == (0, 64)
    assert table[-1][1] == 8
    assert [step for step, _ in table] == sorted({step for step, _ in table})
