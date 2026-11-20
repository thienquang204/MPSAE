"""Balanced three-marginal entropic OT, for the exact s=1 endpoint.

Cyclic log-domain scaling solves the joint tensor, not independent pairwise
problems. The reference-minus-optimum gap uses the same entropy convention as
partial OT. Solver iterations are detached; d gap/d C = J - P*. See Piran et
al. (2024), variational Definition 3.1 and Proposition 3.2. Updates below retain
the old potential when expressed in terms of the current full log coupling.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def _solve(costs, eta, max_iters, tol):
    groups, n = costs.shape[:2]
    kernel = -costs / eta
    u = costs.new_zeros((groups, n))
    v, w = u.clone(), u.clone()
    done = torch.zeros(groups, dtype=torch.bool, device=costs.device)
    iterations = torch.zeros(groups, dtype=torch.int64, device=costs.device)
    # A conservative inner accuracy protects the envelope gradient. Float64
    # bounded refinement handles tolerances below float32 resolution.
    target = min(tol, 1e-6)
    for step in range(1, max_iters + 1):
        active = ~done[:, None]
        u = torch.where(active, -math.log(n) - torch.logsumexp(
            kernel + v[:, None, :, None] + w[:, None, None, :], (2, 3)), u)
        v = torch.where(active, -math.log(n) - torch.logsumexp(
            kernel + u[:, :, None, None] + w[:, None, None, :], (1, 3)), v)
        w = torch.where(active, -math.log(n) - torch.logsumexp(
            kernel + u[:, :, None, None] + v[:, None, :, None], (1, 2)), w)
        if step % 5 and step != max_iters:
            continue
        log_plan = kernel + u[:, :, None, None] + v[:, None, :, None] + w[:, None, None, :]
        plan = log_plan.exp()
        margins = torch.stack((plan.sum((2, 3)), plan.sum((1, 3)), plan.sum((1, 2))), 1)
        residual = (margins - 1/n).abs().sum((1, 2))
        primal = (costs * plan + eta * plan * (log_plan - 1)).sum((1, 2, 3))
        dual = eta * ((u + v + w).mean(1) - plan.sum((1, 2, 3)))
        objective_error = (primal - dual).abs()
        good = (torch.isfinite(plan).flatten(1).all(1) & torch.isfinite(primal)
                & torch.isfinite(dual) & (residual <= target)
                & (objective_error <= max(tol, 32 * torch.finfo(costs.dtype).eps)
                   * primal.abs().clamp_min(1)))
        iterations = torch.where(~done, step, iterations)
        done |= good
        # One synchronization per five cycles, independently frozen groups.
        if bool(done.all()):
            break
    stats = torch.stack((plan.sum((1, 2, 3)), residual,
                         (margins-1/n).abs().amax((1, 2)),
                         (margins-1/n).clamp_min(0).amax((1, 2)), objective_error,
                         iterations, done), 1).cpu().tolist()
    records = []
    for mass, l1, linf, cap, error, count, converged in stats:
        records.append(dict(mass=mass, mass_error=abs(mass-1), cap_violation=cap,
                            marginal_l1=l1, marginal_linf=linf, constraint_error=l1,
                            primal_dual_error=error, iterations=int(count),
                            converged=bool(converged), refinement_iterations=0,
                            objective_refinement=0, solver_float64=costs.dtype == torch.float64))
    return plan, records


def matching_gaps(costs, *, eta, max_iters, tol, max_refine_iters,
                  require_feasible=True, collect_metrics=True, solver_dtype=None):
    """Return differentiable gaps, diagnostics and detached plans for [g,n,n,n]."""
    from .ot import _validate, _batched_gap_values

    _validate(costs, 1.0, eta, max_iters, tol, batched=True, allow_balanced=True)
    if (isinstance(max_refine_iters, bool) or not isinstance(max_refine_iters, int)
            or max_refine_iters < max_iters):
        raise ValueError("max_refine_iters must be an integer >= max_iters")
    dtype = solver_dtype or costs.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("balanced solver requires float32 or float64")
    detached = costs.detach().to(dtype)
    plans, records = _solve(detached, eta, max_iters, tol)
    bad = [i for i, r in enumerate(records) if not r['converged']]
    if bad and require_feasible:
        refined, improved = _solve(detached[bad].double(), eta, max_refine_iters, tol)
        plans = plans.double()  # retain the refinement accuracy in the loss
        plans[bad] = refined
        for i, record in zip(bad, improved):
            record['refinement_iterations'] = record['iterations']
            record['iterations'] += records[i]['iterations']
            records[i] = record
        if any(not r['converged'] for r in records):
            raise RuntimeError(f"balanced MMOT did not converge after bounded refinement: {records}")
    # At full mass the three slack marginals are exactly zero.
    slacks = plans.new_zeros((costs.shape[0], 3, costs.shape[1]))
    losses = _batched_gap_values(costs, plans, slacks, 1.0, eta)
    values = losses.detach().cpu().tolist()
    if any(not math.isfinite(x) for x in values):
        raise FloatingPointError("nonfinite balanced matching gap")
    if require_feasible and any(x < -max(tol, 1e-6) for x in values):
        raise RuntimeError("materially negative balanced matching gap")
    if not collect_metrics:
        return losses, [], plans
    n = costs.shape[1]
    d = torch.arange(n, device=costs.device)
    flat = costs.detach().flatten(1)
    packed = torch.stack((plans[:, d, d, d].sum(1), flat.mean(1), flat.amin(1),
                          flat.amax(1), flat.std(1, unbiased=False)/eta), 1).cpu().tolist()
    for record, value, row in zip(records, values, packed):
        record.update(zip(('diagonal_plan_mass', 'cost_mean', 'cost_min', 'cost_max',
                           'cost_std_over_eta'), row))
        record['negative_gap'] = value < -max(tol, 1e-6)
    return losses, records, plans
