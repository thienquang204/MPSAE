"""CPU oracles for the vectorized CUDA OT path, including mandatory guards."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from mpsae import ot


@pytest.fixture(autouse=True, scope="module")
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def views(rows, *, zero=False):
    generator = torch.Generator().manual_seed(281)
    pre = torch.randn(rows, 24, generator=generator)
    result = []
    for k in (2, 4, 8):
        chosen = pre.topk(k, dim=1)
        z = torch.zeros_like(pre).scatter(1, chosen.indices, chosen.values).relu()
        if zero:
            z[::3] = 0
        result.append(z)
    return result


@pytest.mark.parametrize("zero", [False, True])
def test_batched_cost_matches_independent_costs_and_gradients(zero):
    z = [v.reshape(3, 4, 24).requires_grad_() for v in views(12, zero=zero)]
    expected = torch.stack([ot.angular_cost(*(v[i] for v in z)) for i in range(3)])
    actual = ot.batched_angular_cost(*z)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    weights = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual) / actual.numel()
    first = torch.autograd.grad((actual * weights).sum(), z, retain_graph=True)
    second = torch.autograd.grad((expected * weights).sum(), z)
    for a, b in zip(first, second):
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("rows", [8, 9, 10, 11])
@pytest.mark.parametrize("collect_metrics", [False, True])
def test_vectorized_gap_matches_reference_with_tail_and_gradient(rows, collect_metrics):
    z = [v.requires_grad_() for v in views(rows)]
    expected, _ = ot.matching_gap(*z, microbatch=4, collect_metrics=collect_metrics)
    actual, stats = ot._matching_gap_batched(*z, microbatch=4, collect_metrics=collect_metrics)
    assert torch.allclose(actual, expected, atol=3e-6, rtol=1e-5)
    first = torch.autograd.grad(actual, z, retain_graph=True)
    second = torch.autograd.grad(expected, z)
    for a, b in zip(first, second):
        assert torch.allclose(a, b, atol=3e-6, rtol=1e-4)
        if rows == 9:
            assert torch.count_nonzero(a[-1]) == 0
    if collect_metrics:
        assert stats["groups"] == (2 if rows == 9 or rows == 8 else 3)
        assert stats["mass_error_max"] <= 1e-4
        assert stats["cap_violation_max"] <= 1e-4
    else:
        assert stats == {}


def test_fixed_shape_entropy_and_all_zero_views_are_finite():
    x = torch.tensor([[0., .1, .9, 0.], [0., 0., 0., 0.]])
    expected = torch.stack([ot.entropy_term(row) for row in x])
    assert torch.equal(ot._group_entropy(x), expected)
    z = [torch.zeros(8, 24, requires_grad=True) for _ in range(3)]
    actual, stats = ot._matching_gap_batched(*z, microbatch=4)
    expected, _ = ot.matching_gap(*z, microbatch=4)
    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected, atol=3e-6)
    assert stats["mass_error_max"] <= 1e-4
    for grad in torch.autograd.grad(actual, z):
        assert torch.isfinite(grad).all()


def test_batched_cost_gradient_is_the_envelope_gradient():
    cost = ot.batched_angular_cost(*(z.reshape(2, 4, 24) for z in views(8)))
    cost = cost.detach().requires_grad_()
    loss, stats = ot._batched_matching_gap_from_costs(
        cost, mass=.9, eta=.2, max_iters=100, tol=1e-4,
        require_feasible=True, max_refine_iters=2000)
    grad, = torch.autograd.grad(loss.sum(), cost)
    diagonal = torch.arange(4)
    plan = -grad.clone()
    plan[:, diagonal, diagonal, diagonal] += .9 / 4
    assert plan.min() >= -1e-7
    assert torch.allclose(plan.flatten(1).sum(1), torch.full((2,), .9), atol=1e-4)
    for axes in ((2, 3), (1, 3), (1, 2)):
        assert plan.sum(axes).max() <= .25 + 1e-4


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware required"))])
def test_production_group_size_matches_independent_groups(device):
    z = [v.to(device).requires_grad_() for v in views(66)]
    actual, stats = ot._matching_gap_batched(*z, microbatch=32)
    expected = torch.stack([ot.matching_gap_from_cost(
        ot.angular_cost(*(v[start:stop] for v in z)))[0]
        for start, stop in ((0, 32), (32, 64), (64, 66))]).mean()
    assert torch.allclose(actual, expected, atol=5e-6, rtol=1e-5)
    first = torch.autograd.grad(actual, z, retain_graph=True)
    second = torch.autograd.grad(expected, z)
    for a, b in zip(first, second):
        assert torch.allclose(a, b, atol=3e-6, rtol=1e-4)
    assert stats["groups"] == 3
    assert stats["mass_error_max"] <= 1e-4
    assert stats["cap_violation_max"] <= 1e-4


def _bad_feasible_solve(costs, mass, eta, max_iters, tol, **kwargs):
    groups, n = costs.shape[:2]
    plan = torch.zeros_like(costs)
    ids = torch.arange(n)
    plan[:, ids, (ids + 1) % n, (ids + 1) % n] = mass / n
    slack = costs.new_full((groups, 3, n), (1 - mass) / n)
    stats = [dict(mass=mass, mass_error=0., cap_violation=0., constraint_error=0., iterations=1)
             for _ in range(groups)]
    return plan, slack, stats


def test_negative_gaps_are_refined_together(monkeypatch):
    original = ot.batched_greenkhorn
    calls = []

    def solve(costs, *args, **kwargs):
        calls.append((len(costs), kwargs.get("stable_kl", False)))
        return (_bad_feasible_solve if len(calls) == 1 else original)(costs, *args, **kwargs)

    monkeypatch.setattr(ot, "batched_greenkhorn", solve)
    cost = torch.ones(2, 3, 3, 3)
    ids = torch.arange(3)
    cost[:, ids, ids, ids] = 0
    losses, records = ot._batched_matching_gap_from_costs(
        cost, mass=.9, eta=.01, max_iters=100, tol=1e-4,
        require_feasible=True, max_refine_iters=2000)
    assert calls == [(2, False), (2, True)]
    assert losses.min() >= -1e-4
    assert all(r["objective_refinement"] == 1 for r in records)


@pytest.mark.parametrize("collect_metrics", [False, True])
def test_negative_gap_guard_survives_disabling_metrics(monkeypatch, collect_metrics):
    monkeypatch.setattr(ot, "batched_greenkhorn", _bad_feasible_solve)
    cost = torch.ones(2, 3, 3, 3)
    ids = torch.arange(3)
    cost[:, ids, ids, ids] = 0
    with pytest.raises(RuntimeError, match="materially negative"):
        ot._batched_matching_gap_from_costs(
            cost, mass=.9, eta=.01, max_iters=100, tol=1e-4,
            require_feasible=True, max_refine_iters=2000, collect_metrics=collect_metrics)


def test_nonfinite_gap_guard_survives_disabling_metrics(monkeypatch):
    monkeypatch.setattr(ot, "_batched_gap_values",
                        lambda costs, *args: costs.new_full((len(costs),), float("nan")))
    cost = torch.rand(2, 3, 3, 3)
    with pytest.raises(FloatingPointError, match="nonfinite OT gap"):
        ot._batched_matching_gap_from_costs(
            cost, mass=.9, eta=.2, max_iters=100, tol=1e-4,
            require_feasible=True, max_refine_iters=2000, collect_metrics=False)


@pytest.mark.parametrize("collect_metrics", [False, True])
def test_infeasible_plan_is_rejected_after_bounded_refinement(monkeypatch, collect_metrics):
    def solve(*args, **kwargs):
        plan, slacks, records = _bad_feasible_solve(*args, **kwargs)
        for record in records:
            record["mass_error"] = .5
        return plan, slacks, records

    monkeypatch.setattr(ot, "batched_greenkhorn", solve)
    with pytest.raises(RuntimeError, match="infeasible after bounded refinement"):
        ot._batched_matching_gap_from_costs(
            torch.ones(2, 3, 3, 3), mass=.9, eta=.2, max_iters=100, tol=1e-4,
            require_feasible=True, max_refine_iters=2000, collect_metrics=collect_metrics)
