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


def batched_angular_cost(z1: torch.Tensor, z2: torch.Tensor,
                         z3: torch.Tensor) -> torch.Tensor:
    """The same angular cost for independent ``[groups,n,hidden]`` views.

    Batched matmuls avoid three small launches per group. Group membership,
    normalization epsilon and the order of the three pairwise additions stay
    identical to :func:`angular_cost`; no cross-group similarities are formed.
    """
    if z1.ndim != 3 or any(z.shape != z1.shape for z in (z2, z3)):
        raise ValueError("batched views must share [groups,n,hidden] shape")
    a, b, c = (F.normalize(z.float(), dim=2, eps=1e-8) for z in (z1, z2, z3))
    return ((2 / 9) * ((1 - a @ b.transpose(1, 2))[:, :, :, None]
                       + (1 - a @ c.transpose(1, 2))[:, :, None, :]
                       + (1 - b @ c.transpose(1, 2))[:, None, :, :])).clamp(0.0, 1.0)


def _group_entropy(x: torch.Tensor) -> torch.Tensor:
    """Zero-safe entropy per first dimension without dynamic-size indexing."""
    positive = x > 0
    safe = torch.where(positive, x, torch.ones_like(x))
    terms = torch.where(positive, x * (safe.log() - 1), torch.zeros_like(x))
    return terms.flatten(1).sum(1)


def _batched_gap_values(costs: torch.Tensor, plans: torch.Tensor, slacks: torch.Tensor,
                        mass: float, eta: float) -> torch.Tensor:
    """One differentiable gap per group, with detached solver outputs."""
    n = costs.shape[1]
    diagonal = torch.arange(n, device=costs.device)
    linear = mass * costs[:, diagonal, diagonal, diagonal].mean(1)
    linear = linear - (costs * plans.detach()).flatten(1).sum(1)
    with torch.no_grad():
        reference = _group_entropy(costs.new_full((1, n), mass / n))
        reference = reference + 3 * _group_entropy(costs.new_full((1, n), (1 - mass) / n))
        # Preserve the three slack reductions and their addition order.
        slack_entropy = sum(_group_entropy(slacks[:, axis]) for axis in range(3))
        entropy = reference - _group_entropy(plans) - slack_entropy
    return linear + eta * entropy


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


def _batched_matching_gap_from_costs(costs: torch.Tensor, *, mass: float, eta: float,
                                     max_iters: int, tol: float,
                                     require_feasible: bool, max_refine_iters: int,
                                     collect_metrics: bool = True
                                     ) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Solve and validate many independent groups with packed host checks.

    The single-cost API remains the numerical reference. This path preserves
    the same bounded refinement policy while avoiding per-group scalar reads
    and the unused envelope-delta clone in the production training loop.
    """
    if (isinstance(max_refine_iters, bool) or not isinstance(max_refine_iters, int)
            or max_refine_iters < max_iters):
        raise ValueError("max_refine_iters must be an integer >= max_iters")
    plans, slacks, records = batched_greenkhorn(costs.detach(), mass, eta, max_iters, tol)
    for record in records:
        record.update(refinement_iterations=0, objective_refinement=0)

    def residual(record):
        values = [record[key] for key in ("mass_error", "cap_violation")]
        return max(values) if all(math.isfinite(v) and v >= 0 for v in values) else float("inf")

    def refine(positions, reason):
        if not positions:
            return
        better_plans, better_slacks, better = batched_greenkhorn(
            costs.detach()[positions], mass, eta, max_refine_iters,
            min(1e-10, tol * tol * 0.01), stable_kl=True)
        for slot, position in enumerate(positions):
            first_pass = records[position]["iterations"]
            records[position] = better[slot]
            records[position]["refinement_iterations"] = better[slot]["iterations"]
            records[position]["iterations"] += first_pass
            records[position]["objective_refinement"] = int(reason == "negative_gap")
        # The solver is detached; replacing plans must not sever the cost graph.
        plans[positions] = better_plans
        slacks[positions] = better_slacks

    if require_feasible:
        refine([i for i, record in enumerate(records) if residual(record) > tol], "infeasible")
        if any(residual(record) > tol for record in records):
            raise RuntimeError(f"partial OT infeasible after bounded refinement: {records}")

    losses = _batched_gap_values(costs, plans, slacks, mass, eta)
    gap_tolerance = max(tol, 1e-6)

    def checked_flags(values):
        # One transfer validates every group; these checks are mandatory even
        # when optional metrics are disabled on non-logging steps.
        flags = torch.stack((torch.isfinite(values), values.detach().lt(-gap_tolerance)), 1)
        flags = flags.detach().cpu().tolist()
        if not all(row[0] for row in flags):
            raise FloatingPointError("nonfinite OT gap")
        return [row[1] for row in flags]

    negative = checked_flags(losses)
    if require_feasible:
        positions = [i for i, flag in enumerate(negative)
                     if flag and not records[i]["refinement_iterations"]]
        if positions:
            refine(positions, "negative_gap")
            if any(residual(record) > tol for record in records):
                raise RuntimeError(f"partial OT infeasible after bounded refinement: {records}")
            losses = _batched_gap_values(costs, plans, slacks, mass, eta)
            negative = checked_flags(losses)
        if any(negative):
            raise RuntimeError("materially negative OT gap after bounded refinement")

    if not collect_metrics:
        return losses, []
    n = costs.shape[1]
    diagonal = torch.arange(n, device=costs.device)
    detached = costs.detach().flatten(1)
    packed = torch.stack((plans[:, diagonal, diagonal, diagonal].sum(1),
                          detached.mean(1), detached.min(1).values, detached.max(1).values,
                          detached.std(1, unbiased=False) / eta), 1).cpu().tolist()
    names = ("diagonal_plan_mass", "cost_mean", "cost_min", "cost_max", "cost_std_over_eta")
    for record, flag, values in zip(records, negative, packed):
        record["negative_gap"] = flag
        record.update(zip(names, values))
    return losses, records


def matching_gap_from_cost(cost: torch.Tensor, *, mass: float = 0.9, eta: float = 0.2,
                           max_iters: int = 100, tol: float = 1e-4,
                           require_feasible: bool = True, max_refine_iters: int = 2000,
                           solved=None, solver_dtype: torch.dtype = torch.float32,
                           stable_kl: bool = False, collect_metrics: bool = True,
                           return_delta: bool = True
                           ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
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
    loss_value = float(loss.detach())
    if require_feasible and loss_value < -gap_tolerance and not stats["refinement_iterations"]:
        plan, slacks, stats = refine(stats, "negative_gap")
        if residual(stats) > tol:
            raise RuntimeError(f"partial OT infeasible after bounded refinement: {stats}")
        loss = gap_value(plan, slacks)
        loss_value = float(loss.detach())
    stats["negative_gap"] = loss_value < -gap_tolerance
    if collect_metrics:
        stats["diagonal_plan_mass"] = float(plan[diagonal, diagonal, diagonal].sum())
    if require_feasible and stats["negative_gap"]:
        raise RuntimeError(f"materially negative OT gap: {float(loss.detach())}")

    delta = None
    if return_delta:
        delta = -plan.detach().clone()
        delta[diagonal, diagonal, diagonal] += mass / len(cost)
    return loss, stats, delta


def _aggregate_records(records: list[dict[str, float]]) -> dict[str, float]:
    stats = {key: sum(float(record[key]) for record in records) / len(records)
             for key in records[0]}
    stats["cost_min"] = min(record["cost_min"] for record in records)
    stats["cost_max"] = max(record["cost_max"] for record in records)
    stats["mass_error_max"] = max(record["mass_error"] for record in records)
    stats["cap_violation_max"] = max(record["cap_violation"] for record in records)
    stats["groups"] = len(records)
    return stats


def _matching_gap_batched(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, *,
                          mass: float = 0.9, eta: float = 0.2, max_iters: int = 100,
                          tol: float = 1e-4, microbatch: int = DEFAULT_MICROBATCH,
                          require_feasible: bool = True, max_refine_iters: int = 2000,
                          collect_metrics: bool = True
                          ) -> tuple[torch.Tensor, dict[str, float]]:
    """Tensorized CUDA training path; callable on CPU for parity tests.

    Full groups are processed together and a smaller tail uses the reference
    single-group path. A singleton tail is skipped, and all included groups
    retain equal weight regardless of their size.
    """
    groups = len(z1) // microbatch
    if groups < 1:
        raise ValueError("batched matching needs at least one full group")
    end = groups * microbatch
    costs = batched_angular_cost(*(z[:end].reshape(groups, microbatch, z.shape[1])
                                  for z in (z1, z2, z3)))
    losses, records = _batched_matching_gap_from_costs(
        costs, mass=mass, eta=eta, max_iters=max_iters, tol=tol,
        require_feasible=require_feasible, max_refine_iters=max_refine_iters,
        collect_metrics=collect_metrics)
    if len(z1) - end >= 2:
        tail_cost = angular_cost(z1[end:], z2[end:], z3[end:])
        tail_loss, tail_stats, _ = matching_gap_from_cost(
            tail_cost, mass=mass, eta=eta, max_iters=max_iters, tol=tol,
            require_feasible=require_feasible, max_refine_iters=max_refine_iters,
            collect_metrics=collect_metrics, return_delta=False)
        losses = torch.cat((losses, tail_loss[None]))
        if collect_metrics:
            flat = tail_cost.detach().flatten()
            packed = torch.stack((flat.mean(), flat.min(), flat.max(),
                                  flat.std(unbiased=False) / eta)).cpu().tolist()
            tail_stats.update(zip(("cost_mean", "cost_min", "cost_max", "cost_std_over_eta"), packed))
            records.append(tail_stats)
    return losses.mean(), _aggregate_records(records) if collect_metrics else {}


def matching_gap(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, *,
                 mass: float = 0.9, eta: float = 0.2, max_iters: int = 100,
                 tol: float = 1e-4, microbatch: int = DEFAULT_MICROBATCH,
                 require_feasible: bool = True, max_refine_iters: int = 2000,
                 collect_metrics: bool = True,
                 ) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean matching gap of a batch, solved in independent groups.

    A trailing group of fewer than two rows has no meaningful matching and is
    dropped; the remaining groups are averaged. Equal-sized groups on CUDA go
    through vectorized cost, solve and gap calculation. ``collect_metrics``
    controls optional telemetry, never the finite/feasibility/gap guards.
    """
    if microbatch < 2:
        raise ValueError("an OT group needs at least two examples")
    for name, z in (("z1", z1), ("z2", z2), ("z3", z3)):
        if z.ndim != 2 or len(z) != len(z1):
            raise ValueError(f"{name} must align with z1 as [batch, hidden]")
        if not bool(torch.isfinite(z).all()):
            raise ValueError(f"nonfinite code in OT view {name}")

    if z1.is_cuda and len(z1) // microbatch > 1:
        return _matching_gap_batched(
            z1, z2, z3, mass=mass, eta=eta, max_iters=max_iters, tol=tol,
            microbatch=microbatch, require_feasible=require_feasible,
            max_refine_iters=max_refine_iters, collect_metrics=collect_metrics)

    costs = []
    for start in range(0, len(z1), microbatch):
        stop = min(start + microbatch, len(z1))
        if stop - start < 2:
            continue
        costs.append(angular_cost(z1[start:stop], z2[start:stop], z3[start:stop]))
    if not costs:
        raise ValueError(f"batch of {len(z1)} is too small for OT groups of {microbatch}")

    # CPU and a single full CUDA group retain the scalar reference path.
    losses, records = [], []
    for cost in costs:
        loss, record, _ = matching_gap_from_cost(
            cost, mass=mass, eta=eta, max_iters=max_iters, tol=tol,
            require_feasible=require_feasible, max_refine_iters=max_refine_iters,
            collect_metrics=collect_metrics, return_delta=False)
        if collect_metrics:
            detached = cost.detach()
            record.update(cost_mean=float(detached.mean()), cost_min=float(detached.min()),
                          cost_max=float(detached.max()),
                          cost_std_over_eta=float(detached.std(unbiased=False)) / eta)
        losses.append(loss)
        records.append(record)

    stats = _aggregate_records(records) if collect_metrics else {}
    return torch.stack(losses).mean(), stats
