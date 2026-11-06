"""The OT matching gap: mathematical contracts, plus the pinned legacy numerics.

The contracts hold always. The golden comparison runs only when
``tests/golden/ot_reference.json`` exists -- produce it once with
``python scripts/capture_golden.py`` on a machine with torch, which reads the
pre-rewrite implementation out of git. Until then these tests still check that
the solver is feasible, the gap is nonnegative, and the gradient is the
envelope gradient.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Must precede the mpsae import: ``mpsae.ot`` imports torch at module scope, so
# a plain skip mark would still fail during collection.
pytest.importorskip("torch")

from mpsae import ot  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "ot_reference.json"
MASS, ETA, ITERS, TOL = 0.9, 0.2, 100, 1e-4


def _views(torch, *, group=8, hidden=64, k=4, seed=20261106):
    generator = torch.Generator().manual_seed(seed)
    pre = torch.randn(group, hidden, generator=generator)
    views = []
    for budget in (k, 2 * k, 4 * k):
        selected = torch.topk(pre, k=min(budget, hidden), dim=-1)
        dense = torch.zeros_like(pre).scatter_(-1, selected.indices, selected.values)
        views.append(torch.relu(dense))
    return pre, views


# ------------------------------------------------------------------ contracts

def test_angular_cost_is_in_the_unit_interval(torch):
    _, (z1, z2, z3) = _views(torch)
    cost = ot.angular_cost(z1, z2, z3)
    assert cost.shape == (len(z1),) * 3
    assert float(cost.min()) >= 0.0 and float(cost.max()) <= 1.0


def test_identical_views_have_a_zero_diagonal(torch):
    _, (z1, _, _) = _views(torch)
    cost = ot.angular_cost(z1, z1, z1)
    diagonal = torch.arange(len(z1))
    assert torch.allclose(cost[diagonal, diagonal, diagonal], torch.zeros(len(z1)), atol=1e-6)


def test_an_all_zero_code_costs_one_not_nan(torch):
    """Training tolerates empty codes; the cost must stay finite."""
    _, (z1, z2, z3) = _views(torch)
    z2 = z2.clone()
    z2[0] = 0
    cost = ot.angular_cost(z1, z2, z3)
    assert bool(torch.isfinite(cost).all())


def test_solver_returns_a_feasible_plan(torch):
    _, (z1, z2, z3) = _views(torch)
    cost = ot.angular_cost(z1, z2, z3)
    plan, slacks, stats = ot.greenkhorn(cost, MASS, ETA, ITERS, TOL)
    n = len(cost)
    assert stats["mass_error"] <= TOL
    assert stats["cap_violation"] <= TOL
    assert float(plan.min()) >= 0.0
    for axis, slack in zip(((1, 2), (0, 2), (0, 1)), slacks):
        marginal = plan.sum(axis)
        assert float(marginal.max()) <= 1 / n + TOL
        assert torch.allclose(marginal + slack, torch.full((n,), 1 / n), atol=10 * TOL)


def test_gap_is_nonnegative_and_finite(torch):
    _, (z1, z2, z3) = _views(torch)
    gap, stats = ot.matching_gap(z1, z2, z3, mass=MASS, eta=ETA, max_iters=ITERS,
                                 tol=TOL, microbatch=len(z1))
    assert bool(torch.isfinite(gap))
    assert float(gap) >= -max(TOL, 1e-6)
    assert not stats["negative_gap"]


def test_gap_vanishes_when_the_views_agree(torch):
    """Perfectly consistent budgets have nothing to transport away."""
    _, (z1, _, _) = _views(torch)
    aligned, _ = ot.matching_gap(z1, z1, z1, mass=MASS, eta=ETA, max_iters=ITERS,
                                 tol=TOL, microbatch=len(z1))
    _, (a, b, c) = _views(torch, seed=99)
    disagreeing, _ = ot.matching_gap(a, b, c, mass=MASS, eta=ETA, max_iters=ITERS,
                                     tol=TOL, microbatch=len(a))
    assert float(aligned) <= float(disagreeing) + 1e-6


def test_gradient_is_the_envelope_gradient(torch):
    """``d gap / d C = s*J - P*``: the plan is held fixed by the envelope theorem."""
    _, (z1, z2, z3) = _views(torch)
    cost = ot.angular_cost(z1, z2, z3).detach().requires_grad_(True)
    gap, _, delta = ot.matching_gap_from_cost(cost, mass=MASS, eta=ETA, max_iters=ITERS, tol=TOL)
    grad, = torch.autograd.grad(gap, cost)
    assert torch.allclose(grad, delta, atol=1e-6)
    n = len(cost)
    diagonal = torch.arange(n)
    # The reference plan puts mass/n on each diagonal cell and nothing else.
    assert torch.allclose(delta[diagonal, diagonal, diagonal]
                          + ot.greenkhorn(cost.detach(), MASS, ETA, ITERS, TOL)[0]
                          [diagonal, diagonal, diagonal],
                          torch.full((n,), MASS / n), atol=1e-5)


def test_gradient_reaches_all_three_views(torch):
    _, (z1, z2, z3) = _views(torch)
    views = [z.detach().requires_grad_(True) for z in (z1, z2, z3)]
    gap, _ = ot.matching_gap(*views, mass=MASS, eta=ETA, max_iters=ITERS, tol=TOL,
                             microbatch=len(z1))
    grads = torch.autograd.grad(gap, views, allow_unused=True)
    for index, grad in enumerate(grads):
        assert grad is not None, index
        assert bool(torch.isfinite(grad).all())


def test_batched_solver_agrees_with_the_single_group_solver(torch):
    costs = []
    for seed in (1, 2, 3):
        _, (z1, z2, z3) = _views(torch, seed=seed)
        costs.append(ot.angular_cost(z1, z2, z3).detach())
    stacked = torch.stack(costs)
    plans, slacks, records = ot.batched_greenkhorn(stacked, MASS, ETA, ITERS, TOL)
    for index, cost in enumerate(costs):
        plan, slack, stats = ot.greenkhorn(cost, MASS, ETA, ITERS, TOL)
        assert torch.allclose(plans[index], plan, atol=1e-5), index
        assert records[index]["iterations"] == stats["iterations"], index
        for one, other in zip(slacks[index].unbind(0), slack):
            assert torch.allclose(one, other, atol=1e-5)


def test_groups_do_not_couple(torch):
    """Changing one group must not move another group's contribution."""
    _, (z1, z2, z3) = _views(torch, group=8)
    first, _ = ot.matching_gap(z1[:4], z2[:4], z3[:4], microbatch=4, max_iters=ITERS)
    perturbed = [z.clone() for z in (z1, z2, z3)]
    perturbed[0][4:] += 5.0
    whole, _ = ot.matching_gap(*perturbed, microbatch=4, max_iters=ITERS)
    second, _ = ot.matching_gap(perturbed[0][4:], perturbed[1][4:], perturbed[2][4:],
                                microbatch=4, max_iters=ITERS)
    assert float(whole) == pytest.approx(float((first + second) / 2), abs=1e-5)


def test_rejects_invalid_problems(torch):
    cost = torch.rand(4, 4, 4)
    with pytest.raises(ValueError):
        ot.greenkhorn(cost, 1.5, ETA, ITERS, TOL)          # mass outside (0,1)
    with pytest.raises(ValueError):
        ot.greenkhorn(cost, MASS, 0.0, ITERS, TOL)         # eta must be positive
    with pytest.raises(ValueError):
        ot.greenkhorn(-cost, MASS, ETA, ITERS, TOL)        # negative cost
    with pytest.raises(ValueError):
        ot.greenkhorn(cost * float("nan"), MASS, ETA, ITERS, TOL)
    with pytest.raises(ValueError):
        ot.matching_gap(torch.rand(4, 8), torch.rand(4, 8), torch.rand(4, 8), microbatch=1)


# -------------------------------------------------------------- golden values

@pytest.mark.skipif(not GOLDEN.is_file(),
                    reason="run scripts/capture_golden.py on a torch machine first")
def test_matches_the_pinned_legacy_numerics(torch):
    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        pre = torch.tensor(case["pre"]["values"]).reshape(case["pre"]["shape"])
        views = []
        for budget in (case["k"], 2 * case["k"], 4 * case["k"]):
            selected = torch.topk(pre, k=min(budget, case["hidden"]), dim=-1)
            dense = torch.zeros_like(pre).scatter_(-1, selected.indices, selected.values)
            views.append(torch.relu(dense))
        cost = ot.angular_cost(*views)
        expected = torch.tensor(case["cost"]["values"]).reshape(case["cost"]["shape"])
        assert torch.allclose(cost, expected.to(cost.dtype), atol=1e-6), case["name"]

        plan, _, stats = ot.greenkhorn(cost.detach(), MASS, ETA, ITERS, TOL)
        expected_plan = torch.tensor(case["plan"]["values"]).reshape(case["plan"]["shape"])
        assert torch.allclose(plan, expected_plan.to(plan.dtype), atol=1e-6), case["name"]
        assert stats["iterations"] == case["solver_stats"]["iterations"], case["name"]

        differentiable = ot.angular_cost(*[z.detach().requires_grad_(True) for z in views])
        gap, _, _ = ot.matching_gap_from_cost(differentiable, mass=MASS, eta=ETA,
                                              max_iters=ITERS, tol=TOL)
        assert float(gap) == pytest.approx(case["gap"], abs=1e-6), case["name"]
