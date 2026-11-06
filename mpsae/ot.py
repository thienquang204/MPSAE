"""Entropic multi-marginal partial-OT matching gap -- the MPSAE objective.

Where CSR regularises its sparse codes with a contrastive term, MPSAE couples
the codes of *one* example across three sparsity budgets (K, 2K, 4K) and
penalises how far their optimal partial matching is from the identity
matching. Prior art: the matching-gap formulation of Piran et al. (ICML'24) and
multi-marginal partial OT; the combination with a nested Top-K autoencoder is
ours.

The gap, for a batch group of ``n`` examples with angular cost ``C``:

    gap = F_eta(C, s*J) - min_{P in U_s} F_eta(C, P)

``J`` is the identity matching scaled to total mass ``s``; ``U_s`` is the set of
nonnegative three-dimensional plans with total mass ``s`` and every marginal
bounded by ``1/n``; ``F_eta`` adds ``eta`` times the negative entropy of the
plan and of its three slack marginals. The minimiser is found with a greedy
(Greenkhorn-style) solver on slack variables, under ``no_grad``; the gradient
flows only through the cost, using the envelope theorem, which is why
``plan_gradient`` returns ``s*J - P*``.

Numerics are preserved verbatim from the previous implementation
(``mpsae_repair.py`` in commit 89806f5): the association order of the four
additions in the scaling factor, the ``clamp(-60, 0)`` exponent guard, the
``1e-12`` floors and the two-stage refinement all change float32 results and
are therefore part of the method, not cosmetics.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

#: Groups are solved independently; 32**3 plan entries keep kernels small.
DEFAULT_MICROBATCH = 32
SYNC_EVERY = 10


def angular_cost(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
    """``C[i,j,l]`` in ``[0, 1]``: mean pairwise cosine distance of a triple.

    ``(2/9) * [(1 - a_i.b_j) + (1 - a_i.c_l) + (1 - b_j.c_l)]``. The scale puts
    three pairwise distances, each at most ``2``, into the unit interval so one
    ``eta`` is meaningful across budgets. An all-zero code normalises to zero
    and therefore contributes distance ``1`` rather than a NaN.
    """
    for name, z in (("z1", z1), ("z2", z2), ("z3", z3)):
        if z.ndim != 2 or len(z) != len(z1):
            raise ValueError(f"{name} must be a [group, hidden] matrix aligned with z1")
    a, b, c = (F.normalize(z.float(), dim=1, eps=1e-8) for z in (z1, z2, z3))
    return ((2 / 9) * ((1 - a @ b.T)[:, :, None]
                       + (1 - a @ c.T)[:, None, :]
                       + (1 - b @ c.T)[None, :, :])).clamp(0.0, 1.0)


def entropy_term(x: torch.Tensor) -> torch.Tensor:
    """``sum x (log x - 1)`` over the strictly positive entries."""
    positive = x > 0
    return (x[positive] * (x[positive].log() - 1)).sum()


def _validate(cost: torch.Tensor, mass: float, eta: float, max_iters: int, tol: float,
              *, batched: bool = False) -> None:
    """Reject invalid problems before an exponential kernel is formed.

    The solvers implement a nonnegative cost with a bounded exponent; they are
    not a general signed-cost solver. NaNs must be rejected here because a NaN
    residual silently fails every ``<= tol`` comparison later.
    """
    rank = 4 if batched else 3
    sides = cost.shape[1:] if batched else cost.shape
    if (cost.ndim != rank or len(set(sides)) != 1 or not sides or sides[0] < 2
            or (batched and cost.shape[0] < 1)):
        raise ValueError("expected [groups,n,n,n] costs" if batched else
                         "expected a cubic cost with side >= 2")
    if not cost.is_floating_point():
        raise TypeError("transport costs must be floating point")
    if not bool((torch.isfinite(cost) & cost.ge(0)).all()):
        if not bool(torch.isfinite(cost).all()):
            raise ValueError("nonfinite transport cost")
        raise ValueError("negative transport cost; the angular cost must be nonnegative")
    if (not math.isfinite(mass) or not 0 < mass < 1 or not math.isfinite(eta) or eta <= 0
            or not math.isfinite(tol) or tol <= 0
            or isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1):
        raise ValueError("require finite 0 < mass < 1, positive eta and tol, "
                         "and a positive integer max_iters")


def _kl(target: torch.Tensor, value: torch.Tensor, *, stable: bool) -> torch.Tensor:
    if stable:
        relative = (value.clamp_min(1e-12) - target) / target
        return (target * (relative - torch.log1p(relative))).clamp_min(0).sum()
    return (value - target
            + target.clamp_min(1e-12) * (target.clamp_min(1e-12) / value.clamp_min(1e-12)).log()).sum()


@torch.no_grad()
def greenkhorn(cost: torch.Tensor, mass: float, eta: float, max_iters: int, tol: float,
               *, solver_dtype: torch.dtype = torch.float32, stable_kl: bool = False
               ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    """Greedy slack-variable solver for one group. Returns ``(plan, slacks, stats)``.

    Each iteration updates whichever constraint is currently most violated, in
    KL, among the three marginal caps and the total mass. ``float64`` exists
    for finite-difference reference checks only; training uses ``float32``.
    """
    _validate(cost, mass, eta, max_iters, tol)
    if solver_dtype not in (torch.float32, torch.float64):
        raise ValueError("solver_dtype must be float32 or float64")
    n = len(cost)
    cap = torch.full((n,), 1 / n, device=cost.device, dtype=solver_dtype)
    kernel = torch.exp((-cost.to(solver_dtype) / eta).clamp(-60.0, 0.0))
    scales = [torch.ones_like(cap) for _ in range(3)]
    weight = cost.new_ones((), dtype=solver_dtype)
    target = cost.new_tensor(mass, dtype=solver_dtype)
    total = mass + 3 * (1 - mass)
    error, iteration = float("inf"), 0

    for iteration in range(max_iters):
        v1, v2, v3 = scales
        product = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
        # Keep this association: regrouping the four additions changes which
        # constraint float32 declares worst near convergence.
        factor = total / (weight * product.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(1e-12)
        estimates = [factor * (weight * product.sum(axis) + v)
                     for axis, v in zip(((1, 2), (0, 2), (0, 1)), scales)]
        current = factor * weight * product.sum()
        errors = torch.stack([_kl(cap, v, stable=stable_kl) for v in estimates]
                             + [_kl(target[None], current[None], stable=stable_kl)])
        error = float(errors.max())
        if error <= tol:
            break
        worst = int(errors.argmax())
        if worst < 3:
            scales[worst].mul_(cap / estimates[worst].clamp_min(1e-12)).clamp_(1e-12, 1e12)
        else:
            weight.mul_(target / current.clamp_min(1e-12)).clamp_(1e-12, 1e12)

    v1, v2, v3 = scales
    product = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
    factor = total / (weight * product.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(1e-12)
    plan = factor * weight * product
    marginals = [plan.sum(axis) for axis in ((1, 2), (0, 2), (0, 1))]
    slacks = [(cap - v).clamp_min(0) for v in marginals]
    achieved = float(plan.sum())
    return plan, slacks, {
        "mass": achieved,
        "mass_error": abs(achieved - mass),
        "cap_violation": max(float((v - cap).clamp_min(0).max()) for v in marginals),
        "constraint_error": error,
        "iterations": iteration + 1,
    }


@torch.no_grad()
def batched_greenkhorn(costs: torch.Tensor, mass: float, eta: float, max_iters: int,
                       tol: float, *, stable_kl: bool = False
                       ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """The same solve for many equally sized groups at once.

    Groups keep independent scales, convergence decisions and worst-constraint
    choices; this batches kernel launches and synchronises only every
    ``SYNC_EVERY`` iterations. It does not couple examples across groups.
    """
    _validate(costs, mass, eta, max_iters, tol, batched=True)
    groups, n = costs.shape[:2]
    cap = costs.new_full((groups, n), 1 / n, dtype=torch.float32)
    kernel = torch.exp((-costs.float() / eta).clamp(-60.0, 0.0))
    scales = costs.new_ones((groups, 3, n), dtype=torch.float32)
    weight = costs.new_ones((groups,), dtype=torch.float32)
    active = torch.ones(groups, dtype=torch.bool, device=costs.device)
    stopped = torch.full((groups,), max_iters, dtype=torch.long, device=costs.device)
    target = weight.new_full((groups,), mass)
    total = mass + 3 * (1 - mass)
    axes = torch.arange(3, device=costs.device)[None, :, None]
    worst_error = weight.new_full((groups,), float("inf"))

    for iteration in range(max_iters):
        product = (kernel * scales[:, 0, :, None, None] * scales[:, 1, None, :, None]
                   * scales[:, 2, None, None, :])
        subtotal = product.sum((1, 2, 3))
        factor = total / (weight * subtotal + scales[:, 0].sum(1) + scales[:, 1].sum(1)
                          + scales[:, 2].sum(1)).clamp_min(1e-12)
        marginals = torch.stack((product.sum((2, 3)), product.sum((1, 3)), product.sum((1, 2))), 1)
        estimates = factor[:, None, None] * (weight[:, None, None] * marginals + scales)
        current = factor * weight * subtotal
        if stable_kl:
            relative = (estimates.clamp_min(1e-12) - cap[:, None]) / cap[:, None]
            residuals = (cap[:, None] * (relative - torch.log1p(relative))).clamp_min(0).sum(2)
            relative_mass = (current.clamp_min(1e-12) - target) / target
            mass_error = (target * (relative_mass - torch.log1p(relative_mass))).clamp_min(0)
        else:
            residuals = (estimates - cap[:, None]
                         + cap[:, None] * (cap[:, None] / estimates.clamp_min(1e-12)).log()).sum(2)
            mass_error = current - target + target * (target / current.clamp_min(1e-12)).log()
        errors = torch.cat((residuals, mass_error[:, None]), 1)
        worst_error, worst = errors.max(1)
        converged = active & worst_error.le(tol)
        stopped = torch.where(converged, stopped.new_full((), iteration + 1), stopped)
        active &= ~converged
        ratios = cap[:, None] / estimates.clamp_min(1e-12)
        use_axis = active[:, None, None] & (worst[:, None, None] == axes)
        scales = torch.where(use_axis, scales * ratios, scales).clamp(1e-12, 1e12)
        weight = torch.where(active & worst.eq(3),
                             weight * (target / current.clamp_min(1e-12)), weight).clamp(1e-12, 1e12)
        if (iteration + 1) % SYNC_EVERY == 0 and not bool(active.any()):
            break

    product = (kernel * scales[:, 0, :, None, None] * scales[:, 1, None, :, None]
               * scales[:, 2, None, None, :])
    factor = total / (weight * product.sum((1, 2, 3)) + scales[:, 0].sum(1)
                      + scales[:, 1].sum(1) + scales[:, 2].sum(1)).clamp_min(1e-12)
    plans = factor[:, None, None, None] * weight[:, None, None, None] * product
    marginals = torch.stack((plans.sum((2, 3)), plans.sum((1, 3)), plans.sum((1, 2))), 1)
    slacks = (cap[:, None] - marginals).clamp_min(0)
    achieved = plans.sum((1, 2, 3))
    caps = (marginals - cap[:, None]).clamp_min(0).flatten(1).max(1).values
    # One host transfer for every group's scalar diagnostics.
    packed = torch.stack((achieved, (achieved - mass).abs(), caps, worst_error,
                          stopped.float()), 1).cpu().tolist()
    fields = ("mass", "mass_error", "cap_violation", "constraint_error", "iterations")
    return plans, slacks, [dict(zip(fields, row)) for row in packed]


def matching_gap_from_cost(cost: torch.Tensor, *, mass: float = 0.9, eta: float = 0.2,
                           max_iters: int = 100, tol: float = 1e-4,
                           require_feasible: bool = True, max_refine_iters: int = 2000,
                           solved=None, solver_dtype: torch.dtype = torch.float32,
                           stable_kl: bool = False
                           ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Differentiable gap for one group, plus the envelope gradient ``s*J - P*``.

    Two guards, both of which refine the solve rather than hide the symptom:
    an infeasible plan (marginal cap or mass residual above ``tol``), and a
    materially negative gap, which can only mean the "optimal" plan was not
    optimal. Neither is clipped away.
    """
    _validate(cost, mass, eta, max_iters, tol)
    if isinstance(max_refine_iters, bool) or not isinstance(max_refine_iters, int) \
            or max_refine_iters < max_iters:
        raise ValueError("max_refine_iters must be an integer >= max_iters")
    plan, slacks, stats = (greenkhorn(cost.detach(), mass, eta, max_iters, tol,
                                     solver_dtype=solver_dtype, stable_kl=stable_kl)
                           if solved is None else solved)
    stats.setdefault("refinement_iterations", 0)
    stats.setdefault("objective_refinement", 0)

    def residual(record: dict[str, float]) -> float:
        values = [float(record[name]) for name in ("mass_error", "cap_violation")]
        return max(values) if all(math.isfinite(v) and v >= 0 for v in values) else float("inf")

    def refine(record: dict[str, float], reason: str):
        refined_plan, refined_slacks, refined = greenkhorn(
            cost.detach(), mass, eta, max_refine_iters, min(1e-10, tol * tol * 0.01),
            solver_dtype=solver_dtype, stable_kl=True)
        refined["refinement_iterations"] = refined["iterations"]
        refined["iterations"] += int(record.get("iterations", 0))
        refined["objective_refinement"] = int(reason == "negative_gap")
        return refined_plan, refined_slacks, refined

    if require_feasible and not stats["refinement_iterations"] and residual(stats) > tol:
        plan, slacks, stats = refine(stats, "infeasible")
    if require_feasible and residual(stats) > tol:
        raise RuntimeError(f"partial OT infeasible after bounded refinement: {stats}")

    diagonal = torch.arange(len(cost), device=cost.device)
    reference_plan = cost.new_full((len(cost),), mass / len(cost))
    reference_slack = cost.new_full((len(cost),), (1 - mass) / len(cost))

    def gap_value(current_plan: torch.Tensor, current_slacks) -> torch.Tensor:
        linear = mass * cost[diagonal, diagonal, diagonal].mean() - (cost * current_plan.detach()).sum()
        entropy = entropy_term(reference_plan) + 3 * entropy_term(reference_slack)
        entropy = entropy - entropy_term(current_plan) - sum(entropy_term(q) for q in current_slacks)
        value = linear + eta * entropy.detach()
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("nonfinite OT gap")
        return value

    loss = gap_value(plan, slacks)
    gap_tolerance = max(tol, 1e-6)
    if require_feasible and float(loss.detach()) < -gap_tolerance and not stats["refinement_iterations"]:
        plan, slacks, stats = refine(stats, "negative_gap")
        if residual(stats) > tol:
            raise RuntimeError(f"partial OT infeasible after bounded refinement: {stats}")
        loss = gap_value(plan, slacks)
    stats["negative_gap"] = float(loss.detach()) < -gap_tolerance
    stats["diagonal_plan_mass"] = float(plan[diagonal, diagonal, diagonal].sum())
    if require_feasible and stats["negative_gap"]:
        raise RuntimeError(f"materially negative OT gap: {float(loss.detach())}")

    delta = -plan.detach().clone()
    delta[diagonal, diagonal, diagonal] += mass / len(cost)
    return loss, stats, delta


def matching_gap(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, *,
                 mass: float = 0.9, eta: float = 0.2, max_iters: int = 100,
                 tol: float = 1e-4, microbatch: int = DEFAULT_MICROBATCH,
                 require_feasible: bool = True, max_refine_iters: int = 2000,
                 ) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean matching gap of a batch, solved in independent groups.

    A trailing group of fewer than two rows has no meaningful matching and is
    dropped; the remaining groups are averaged. Equal-sized groups on CUDA go
    through the batched solver.
    """
    if microbatch < 2:
        raise ValueError("an OT group needs at least two examples")
    for name, z in (("z1", z1), ("z2", z2), ("z3", z3)):
        if z.ndim != 2 or len(z) != len(z1):
            raise ValueError(f"{name} must align with z1 as [batch, hidden]")
        if not bool(torch.isfinite(z).all()):
            raise ValueError(f"nonfinite code in OT view {name}")

    costs = []
    for start in range(0, len(z1), microbatch):
        stop = min(start + microbatch, len(z1))
        if stop - start < 2:
            continue
        costs.append(angular_cost(z1[start:stop], z2[start:stop], z3[start:stop]))
    if not costs:
        raise ValueError(f"batch of {len(z1)} is too small for OT groups of {microbatch}")

    solved: dict[int, tuple] = {}
    for side in sorted({len(cost) for cost in costs}):
        positions = [index for index, cost in enumerate(costs) if len(cost) == side]
        if not (z1.is_cuda and len(positions) > 1):
            continue
        stacked = torch.stack([costs[index].detach() for index in positions])
        plans, slacks, records = batched_greenkhorn(stacked, mass, eta, max_iters, tol)
        failing = [slot for slot, record in enumerate(records)
                   if require_feasible
                   and max(record["mass_error"], record["cap_violation"]) > tol]
        if failing:
            better_plans, better_slacks, better = batched_greenkhorn(
                stacked[failing], mass, eta, max_refine_iters,
                min(1e-10, tol * tol * 0.01), stable_kl=True)
            for slot, index in enumerate(failing):
                first_pass = records[index]["iterations"]
                plans[index], slacks[index] = better_plans[slot], better_slacks[slot]
                records[index] = better[slot]
                records[index]["refinement_iterations"] = better[slot]["iterations"]
                records[index]["iterations"] += first_pass
        for slot, position in enumerate(positions):
            solved[position] = (plans[slot], list(slacks[slot].unbind(0)), records[slot])

    losses, records = [], []
    for position, cost in enumerate(costs):
        loss, record, _ = matching_gap_from_cost(
            cost, mass=mass, eta=eta, max_iters=max_iters, tol=tol,
            require_feasible=require_feasible, max_refine_iters=max_refine_iters,
            solved=solved.get(position))
        detached = cost.detach()
        record.update(cost_mean=float(detached.mean()), cost_min=float(detached.min()),
                      cost_max=float(detached.max()),
                      cost_std_over_eta=float(detached.std(unbiased=False)) / eta)
        losses.append(loss)
        records.append(record)

    stats = {key: sum(float(record[key]) for record in records) / len(records)
             for key in records[0]}
    stats["cost_min"] = min(record["cost_min"] for record in records)
    stats["cost_max"] = max(record["cost_max"] for record in records)
    stats["mass_error_max"] = max(record["mass_error"] for record in records)
    stats["cap_violation_max"] = max(record["cap_violation"] for record in records)
    stats["groups"] = len(records)
    return torch.stack(losses).mean(), stats
