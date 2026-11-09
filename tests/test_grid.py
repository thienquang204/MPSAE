"""Invariants of the experiment grid. Runs without torch."""
from __future__ import annotations

import pytest

from mpsae.config import (BACKBONES, BRANCH1, BRANCH2, METHODS, SEEDS, effective_runs,
                          grid, initial_topk)


def test_every_budget_fits_the_embedding():
    """A budget above the backbone's own width is not a compression."""
    for name, spec in BACKBONES.items():
        assert max(spec.k_list) <= spec.dim, name
        assert sorted(spec.k_list) == list(spec.k_list), name
        assert len(set(spec.k_list)) == len(spec.k_list), name


def test_four_times_the_budget_fits_the_dictionary():
    """``L(4k)`` must be expressible, including at the annealing start."""
    for name, spec in BACKBONES.items():
        hidden = spec.preset.hidden_dim
        for k in spec.k_list:
            assert 4 * min(4 * k, hidden) >= 0
            start = initial_topk(k, hidden)
            assert k <= start <= hidden, (name, k)
            assert 4 * start <= hidden, (name, k, start)


def test_e2_anchors_are_trained_budgets():
    for name, spec in BACKBONES.items():
        assert set(spec.e2_anchors) <= set(spec.training_budgets), name


def test_grid_size_matches_the_plan():
    counts = effective_runs()
    assert counts == {"ff2048": 55, "rn18": 45, "convnext_tiny": 25,
                      "vit_b16": 25, "dinov2_vits14": 25, "total": 175}
    assert len(grid()) == 175
    assert {cell.seed for cell in grid()} == {1227}


def test_branches_partition_the_backbones():
    assert set(BRANCH1) | set(BRANCH2) == set(BACKBONES)
    assert not set(BRANCH1) & set(BRANCH2)


def test_presets_follow_the_published_tables():
    """CSR Table 3 for FF2048, Table 9 for the timm backbones."""
    ff2048 = BACKBONES["ff2048"].preset
    assert (ff2048.lr, ff2048.weight_decay, ff2048.batch_size, ff2048.epochs) == \
        (4e-5, 1e-4, 4096, 10)
    assert (ff2048.k_aux, ff2048.aux_coef, ff2048.cl_coef) == (512, 1 / 32, 0.1)
    assert ff2048.adam_eps == 6.25e-10
    rn18 = BACKBONES["rn18"].preset
    assert (rn18.batch_size, rn18.cl_coef) == (1024, 1.0)
    # The two 768-d backbones are treated like FF2048, the 384-d one like RN18.
    assert BACKBONES["convnext_tiny"].preset is ff2048
    assert BACKBONES["vit_b16"].preset is ff2048
    assert BACKBONES["dinov2_vits14"].preset is rn18


def test_grid_is_deterministic_and_ordered():
    first, second = grid(), grid()
    assert first == second
    backbone_order = [cell.backbone for cell in first]
    assert backbone_order == sorted(backbone_order, key=backbone_order.index)


@pytest.mark.parametrize("method", METHODS)
def test_each_method_appears_for_every_seed(method):
    cells = [cell for cell in grid(("ff2048",), (method,), SEEDS) if cell.method == method]
    assert len(cells) == len(BACKBONES["ff2048"].k_list) * len(SEEDS)
    assert {cell.seed for cell in cells} == set(SEEDS)
