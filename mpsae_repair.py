"""Isolated research kernels for mpsae-repair-neighborhood-v1.

The historical paper runner remains unchanged.  This module requires only
PyTorch, and deliberately keeps labels, data ordering and selection gates out
of the objective. Bank encoding and inference are pure (no activity updates).
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def nonnegative_topk(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Signed Top-K then ReLU, with the lowest coordinate winning exact ties.

    A linear tie scan avoids sorting all 8192 coordinates. Nonselected values
    have zero derivative; there is no straight-through support estimator.
    """
    if pre.ndim != 2 or k < 1:
        raise ValueError("Top-K requires a matrix and positive K")
    k = min(int(k), pre.shape[1])
    threshold = pre.topk(k, dim=1, sorted=False).values.min(1, keepdim=True).values
    greater = pre > threshold
    equal = pre == threshold
    remaining = k - greater.sum(1, keepdim=True)
    chosen = greater | (equal & (equal.cumsum(1, dtype=torch.int32) <= remaining))
    return F.relu(pre) * chosen.to(pre.dtype)


class ResearchSAE(nn.Module):
    """Baseline-compatible tied SAE, with deterministic pure budget gates."""

    def __init__(self, input_dim: int, hidden_dim: int = 8192, dead_steps: int = 30):
        super().__init__()
        decoder = torch.empty(hidden_dim, input_dim)
        nn.init.kaiming_uniform_(decoder, a=math.sqrt(5))
        self.decoder = nn.Parameter(decoder)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("inactive_steps", torch.zeros(hidden_dim, dtype=torch.long))
        self.dead_steps = int(dead_steps)

    keep_topk = staticmethod(nonnegative_topk)

    def preactivations(self, x):
        return (x - self.pre_bias) @ self.decoder.T + self.encoder_bias

    def encode(self, x, k):
        return self.keep_topk(self.preactivations(x), k)

    def decode(self, z):
        return z @ self.decoder + self.pre_bias

    @torch.no_grad()
    def update_activity(self, z):
        self.inactive_steps.mul_((~z.gt(1e-5).any(0)).to(self.inactive_steps.dtype))
        self.inactive_steps.add_(1)

    @staticmethod
    def normalized_auxiliary_mse(reconstruction, target):
        baseline = target.mean(0, keepdim=True).expand_as(target)
        return (F.mse_loss(reconstruction, target) / F.mse_loss(baseline, target)).nan_to_num(0.0)


class RepairModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rank: int = 32,
                 alpha: float = 1.0, seed: int = 1227):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("rank and fixed alpha must be positive")
        # A private CPU generator makes adding repair independent of SAE/data RNG.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        u = torch.empty(input_dim, rank)
        # U is used on the right of row vectors; initialize as Linear(d,rank).
        nn.init.kaiming_uniform_(u.T, a=math.sqrt(5), generator=generator)
        self.U = nn.Parameter(u)
        self.V = nn.Parameter(torch.zeros(rank, hidden_dim))
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, sae, x, start, k):
        # The starting code has at most k nonzeros. COO SpMM avoids a dense
        # B-by-h decoder multiply and a B-by-k-by-d gathered dictionary tensor.
        residual = x - indexed_decode(start, sae.decoder, k) - sae.pre_bias
        corrected = start + self.alpha * (residual @ sae.decoder.T)
        corrected = corrected + F.relu(residual @ self.U) @ self.V
        return nonnegative_topk(corrected, k)


def indexed_decode(z, dictionary, k):
    values, columns = z.topk(min(k, z.shape[1]), dim=1, sorted=False)
    rows = torch.arange(len(z), device=z.device)[:, None].expand_as(columns)
    indices = torch.stack((rows.reshape(-1), columns.reshape(-1)))
    sparse = torch.sparse_coo_tensor(indices, values.reshape(-1), z.shape,
                                    dtype=z.dtype, device=z.device)
    return torch.sparse.mm(sparse, dictionary)


def encode_repaired(sae, repair, x, k):
    initial = sae.encode(x, k)
    return initial if repair is None else repair(sae, x, initial, k)


@torch.no_grad()
def estimate_alpha(dictionary, iterations: int = 40, seed: int = 1337) -> float:
    """0.9/spectral-norm-estimate^2; a scale heuristic, not a certified bound."""
    if iterations < 1:
        raise ValueError("power iterations must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(dictionary.shape[1], generator=generator).to(dictionary)
    vector = F.normalize(vector, dim=0)
    for _ in range(iterations):
        vector = F.normalize(dictionary.T @ (dictionary @ vector), dim=0)
    squared = (dictionary @ vector).square().sum().item()
    if not math.isfinite(squared) or squared <= 0:
        raise ValueError("cannot estimate alpha from a degenerate dictionary")
    return .9 / squared


@torch.no_grad()
def backtrack_alpha(sae, x, k, alpha, max_halvings: int = 30):
    """Require per-example fixed-IHT descent on the declared training slice."""
    initial = sae.encode(x, k)
    residual = x - sae.decode(initial)
    initial_errors = residual.square().mean(1)
    for halvings in range(max_halvings + 1):
        repaired = nonnegative_topk(initial + alpha * (residual @ sae.decoder.T), k)
        errors = (x - sae.decode(repaired)).square().mean(1)
        if bool(torch.all(errors <= initial_errors + 1e-7 * initial_errors.clamp_min(1))):
            return float(alpha), {"alpha_halvings": halvings,
                                  "fixed_iht_max_mse_increase": float((errors - initial_errors).max())}
        alpha *= .5
    raise RuntimeError("fixed-IHT alpha backtracking did not establish measured descent")


def mask_start(z, k, generator, clean_probability=.2, erase_probability=.25):
    if generator is None:
        raise ValueError("masking requires a dedicated reproducible generator")
    if not 0 <= clean_probability <= 1 or not 0 <= erase_probability <= 1:
        raise ValueError("invalid masking probabilities")
    values, indices = z.topk(min(k, z.shape[1]), dim=1, sorted=False)
    # Draw K values per row, rather than B*8192 random numbers.
    clean = torch.rand((len(z), 1), generator=generator, device=generator.device).to(z.device) < clean_probability
    erase = torch.rand(values.shape, generator=generator, device=generator.device).to(z.device) < erase_probability
    erased = (~clean) & erase & values.gt(0)
    result = torch.zeros_like(z).scatter(1, indices, values * (~erased))
    stats = {"forced_clean_row_fraction": clean.float().mean(),
             "unchanged_start_fraction": (~erased.any(1)).float().mean(),
             "mask_erase_fraction": erased.sum() / values.gt(0).sum().clamp_min(1),
             "mask_erased_coefficients": erased.sum(),
             "mask_positive_coefficients": values.gt(0).sum()}
    return result, stats


def auxiliary_budget(k, hidden_dim, step, total_steps, mode="baseline"):
    end = min(4 * k, hidden_dim)
    start = min(hidden_dim, max(64, 8 * k))
    if mode in ("baseline", "4k", "fixed"):
        return end
    if mode in ("wide", "fixed_wide"):
        return start
    if mode not in ("anneal", "annealed"):
        raise ValueError(f"unknown auxiliary budget mode {mode!r}")
    duration = math.floor(.7 * total_steps)
    fraction = max(0., 1. - step / duration) if duration else 0.
    return round(end + (start - end) * fraction)


def reconstruction_terms(sae, x, k, aux_budget=None, *, repair=None,
                         mask_generator=None, mask=False, update_activity=True,
                         k_aux=512, clean_probability=.2, erase_probability=.25):
    pre = sae.preactivations(x)
    raw = {name: sae.keep_topk(pre, budget) for name, budget in
           (("raw_k", k), ("raw_2k", 2*k), ("raw_4k", 4*k))}
    aux_budget = min(aux_budget or 4*k, sae.decoder.shape[0])
    wide = raw["raw_4k"] if aux_budget == min(4*k, sae.decoder.shape[0]) else sae.keep_topk(pre, aux_budget)
    stats = {}
    starts = [raw["raw_k"], raw["raw_2k"]]
    if mask:
        if repair is None:
            raise ValueError("masked starts require repair")
        masked = [mask_start(z, budget, mask_generator, clean_probability, erase_probability)
                  for z, budget in zip(starts, (k, 2*k))]
        starts = [item[0] for item in masked]
        stats.update({name: sum(item[1][name] for item in masked)/2
                      for name in ("forced_clean_row_fraction", "unchanged_start_fraction")})
        erased = sum(item[1]["mask_erased_coefficients"] for item in masked)
        positive = sum(item[1]["mask_positive_coefficients"] for item in masked)
        stats.update({"mask_erased_coefficients": erased,
                      "mask_positive_coefficients": positive,
                      "mask_erase_fraction": erased/positive.clamp_min(1)})
    else:
        stats.update({"forced_clean_row_fraction": x.new_tensor(1.),
                      "unchanged_start_fraction": x.new_tensor(1.),
                      "mask_erase_fraction": x.new_tensor(0.)})
    small, medium = [z if repair is None else repair(sae, x, z, budget)
                     for z, budget in zip(starts, (k, 2*k))]
    reconstruction = indexed_decode(small, sae.decoder, k) + sae.pre_bias
    main = F.mse_loss(reconstruction, x)
    nested = F.mse_loss(indexed_decode(wide, sae.decoder, aux_budget) + sae.pre_bias, x)
    # The dead mask is read immediately after RAW K update, as in MPSAE.
    if update_activity:
        sae.update_activity(raw["raw_k"])
        dead = sae.inactive_steps > sae.dead_steps
    else:
        # Simulate the same read point on a copy. A diagnostic that raises
        # midway through its objective must not leave real counters modified.
        simulated = sae.inactive_steps.clone()
        active = raw["raw_k"].gt(1e-5).any(0)
        simulated.mul_((~active).to(simulated.dtype)).add_(1)
        dead = simulated > sae.dead_steps
    if k_aux > 0:
        aux_z = sae.keep_topk(pre * dead[None].to(pre.dtype), min(k_aux, pre.shape[1]))
        target = x - reconstruction.detach() + sae.pre_bias.detach()
        dead_loss = sae.normalized_auxiliary_mse(
            indexed_decode(aux_z, sae.decoder, min(k_aux, pre.shape[1])) + sae.pre_bias, target)
    else:
        dead_loss = main.new_zeros(())
    if update_activity:
        sae.update_activity(raw["raw_4k"])
        sae.update_activity(raw["raw_2k"])
    total = main + nested / 8 + dead_loss / 32
    final_nnz = small.gt(0).sum(1)
    # Support-shrinkage monitors only; none of these enter the objective. The
    # raw views are nested, so ||raw_jK||^2 - ||raw_K||^2 is their tail energy.
    with torch.no_grad():
        head = raw["raw_k"].square().sum(1)
        safe_head = head.clamp_min(torch.finfo(head.dtype).tiny)
        tail_2k = ((raw["raw_2k"].square().sum(1) - head).clamp_min(0) / safe_head).sqrt()
        tail_4k = ((raw["raw_4k"].square().sum(1) - head).clamp_min(0) / safe_head).sqrt()
        alive = head.gt(0)
    stats.update({"mse_final_K": main, "mse_raw_aux": nested, "loss_dead": dead_loss,
                  "dead_fraction": dead.float().mean(), "auxiliary_budget": aux_budget,
                  "actual_nnz_mean": final_nnz.float().mean(),
                  "zero_final_codes": final_nnz.eq(0).sum(),
                  "final_nnz_min": final_nnz.min(),
                  "final_nnz_below_k_fraction": final_nnz.lt(k).float().mean(),
                  "final_zero_code_fraction": final_nnz.eq(0).float().mean(),
                  "raw_kth_preactivation_nonpositive_fraction": raw["raw_k"].gt(0).sum(1).lt(k).float().mean(),
                  "raw_zero_code_fraction": (~alive).float().mean(),
                  "tail_ratio_2k_mean": tail_2k[alive].mean() if bool(alive.any()) else head.new_zeros(()),
                  "tail_ratio_4k_mean": tail_4k[alive].mean() if bool(alive.any()) else head.new_zeros(()),
                  "encoder_bias_mean": sae.encoder_bias.detach().mean()})
    raw.update({"final_k": small, "final_2k": medium, "raw_auxiliary": wide})
    return total, stats, raw


def compact_codes(z, k, *, allow_empty=False):
    """Compact L2-normalized codes. ``allow_empty`` keeps all-zero rows zero,
    as the original MPSAE training path does, instead of failing."""
    values, indices = z.topk(min(k, z.shape[1]), dim=1, sorted=False)
    norms = values.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(values).all()) or (not allow_empty and bool(norms.le(0).any())):
        raise ValueError("nonfinite or zero-norm sparse code")
    return indices, values / norms.clamp_min(torch.finfo(values.dtype).tiny) if allow_empty else values / norms


def indexed_cosines(query, k, normalized_bank_transpose, anchor_chunk=128, *, allow_empty=False):
    """Sparse selected-value derivatives; intermediates <=B*K*anchor_chunk.
    An allowed empty query has zero cosine to every landmark."""
    if anchor_chunk < 1:
        raise ValueError("anchor chunk size must be positive")
    indices, values = compact_codes(query, k, allow_empty=allow_empty)
    key = normalized_bank_transpose.detach()
    similarities, overlaps = [], []
    for start in range(0, key.shape[1], anchor_chunk):
        selected = key[:, start:start + anchor_chunk][indices]
        similarities.append((selected * values[..., None]).sum(1))
        overlaps.append((selected.gt(0) & values[..., None].gt(0)).any(1))
    return torch.cat(similarities, 1), torch.cat(overlaps, 1)


class LandmarkBank:
    def __init__(self, anchor_ids, teacher_features, *, allow_empty_codes=False):
        self.anchor_ids = torch.as_tensor(anchor_ids, dtype=torch.long, device=teacher_features.device)
        if len(self.anchor_ids) != len(teacher_features) or len(self.anchor_ids.unique()) != len(self.anchor_ids):
            raise ValueError("landmark identities must be distinct and match teacher rows")
        self.teacher_features = teacher_features.detach().clone()
        if not bool(torch.isfinite(self.teacher_features).all()) or bool(self.teacher_features.norm(dim=1).eq(0).any()):
            raise ValueError("invalid teacher bank features")
        self.teacher_normalized = F.normalize(self.teacher_features, dim=1)
        # Empty student keys stay zero (zero cosine) instead of aborting refresh.
        self.allow_empty_codes = bool(allow_empty_codes)
        self.codes = None
        self.refresh_count = 0
        self.last_refresh_step = None
        self.refresh_seconds = 0.
        self.last_refresh_diagnostics = {}

    @torch.no_grad()
    def refresh(self, sae, repair, k, batch_size=256, step=0):
        if self.teacher_features.is_cuda:
            torch.cuda.synchronize(self.teacher_features.device)
        started = time.perf_counter()
        old = self.codes
        all_codes = [[], []]
        for batch in self.teacher_features.split(batch_size):
            pre = sae.preactivations(batch)
            for bucket, budget in zip(all_codes, (k, 2*k)):
                raw = sae.keep_topk(pre, budget)
                z = raw if repair is None else repair(sae, batch, raw, budget)
                # Fail on invalid keys; zeros are kept zero only when explicitly allowed.
                compact_codes(z, budget, allow_empty=self.allow_empty_codes)
                bucket.append(z.detach())
        self.codes = [torch.cat(bucket).detach() for bucket in all_codes]
        self.k = int(k)
        self.normalized_transposes = [F.normalize(z, dim=1).T.contiguous() for z in self.codes]
        drift = {}
        if old is not None:
            for label, previous, current in zip(("k", "2k"), old, self.codes):
                intersection = (previous.gt(0) & current.gt(0)).sum(1)
                union = (previous.gt(0) | current.gt(0)).sum(1)
                drift[f"bank_{label}_support_jaccard"] = float((intersection / union.clamp_min(1)).mean())
                drift[f"bank_{label}_cosine"] = float(F.cosine_similarity(previous, current, dim=1).mean())
        if self.teacher_features.is_cuda:
            torch.cuda.synchronize(self.teacher_features.device)
        elapsed = time.perf_counter() - started
        self.refresh_count += 1
        self.last_refresh_step = int(step)
        self.refresh_seconds += elapsed
        self.last_refresh_diagnostics = {**drift, "bank_refresh_seconds": elapsed,
                                         "bank_refresh_count": self.refresh_count}
        return self.last_refresh_diagnostics

    def profiles(self, x, row_ids, z_k, z_2k, *, k, teacher_temperature=.1,
                 student_temperature=.1, anchor_chunk=128):
        if self.codes is None:
            raise RuntimeError("refresh the clean student bank before its first use")
        if min(teacher_temperature, student_temperature) <= 0:
            raise ValueError("profile temperatures must be positive")
        ids = torch.as_tensor(row_ids, device=self.anchor_ids.device, dtype=torch.long)
        # One support for the complete minibatch, including all its OT groups.
        allowed = ~torch.isin(self.anchor_ids, ids)
        if not bool(allowed.any()):
            raise ValueError("minibatch union excludes every landmark")
        with torch.no_grad():
            logits = (F.normalize(x.detach(), dim=1) @ self.teacher_normalized.T) / teacher_temperature
            teacher_log = logits[:, allowed].log_softmax(1)
            teacher = teacher_log.exp()
        student_logs, overlap = [], []
        for z, budget, bank in zip((z_k, z_2k), (k, 2*k), self.normalized_transposes):
            cos, hits = indexed_cosines(z, budget, bank, anchor_chunk, allow_empty=self.allow_empty_codes)
            student_logs.append((cos[:, allowed] / student_temperature).log_softmax(1))
            overlap.append(hits[:, allowed].any(1))
        denominator = math.log(max(2, int(allowed.sum())))
        stats = {"zero_overlap_fraction": (~torch.stack(overlap).any(0)).float().mean(),
                 "zero_overlap_k_fraction": (~overlap[0]).float().mean(),
                 "zero_overlap_2k_fraction": (~overlap[1]).float().mean(),
                 "teacher_entropy_normalized": -(teacher * teacher_log).sum(1).mean() / denominator,
                 "student_k_entropy_normalized": -(student_logs[0].exp() * student_logs[0]).sum(1).mean() / denominator,
                 "student_2k_entropy_normalized": -(student_logs[1].exp() * student_logs[1]).sum(1).mean() / denominator,
                 "bank_allowed_count": int(allowed.sum()),
                 "teacher_neighbor_preservation": ((student_logs[0].argmax(1) == teacher.argmax(1)).float().mean()
                                                    + (student_logs[1].argmax(1) == teacher.argmax(1)).float().mean()) / 2}
        return {"teacher": teacher, "teacher_log": teacher_log,
                "student_logs": student_logs, "allowed": allowed, "stats": stats}

    def state_dict(self):
        # Persist compact keys rather than two mostly-zero M-by-8192 arrays.
        compact = None
        if self.codes is not None:
            compact = []
            for z, budget in zip(self.codes, (self.k, 2*self.k)):
                values, indices = z.topk(min(budget, z.shape[1]), dim=1, sorted=False)
                compact.append({"values": values.cpu(), "indices": indices.to(torch.int32).cpu(),
                                "width": z.shape[1]})
        return {"anchor_ids": self.anchor_ids.detach().cpu(),
                "compact_codes": compact, "k": getattr(self, "k", None),
                "refresh_count": self.refresh_count, "last_refresh_step": self.last_refresh_step,
                "refresh_seconds": self.refresh_seconds,
                "last_refresh_diagnostics": self.last_refresh_diagnostics}

    def load_state_dict(self, state):
        if not torch.equal(state["anchor_ids"].cpu(), self.anchor_ids.cpu()):
            raise ValueError("checkpoint landmark order differs")
        self.k = state["k"]
        compact = state["compact_codes"]
        self.codes = None if compact is None else [self.teacher_features.new_zeros(
            (len(self.anchor_ids), item["width"])).scatter(1,
            item["indices"].to(device=self.teacher_features.device, dtype=torch.long),
            item["values"].to(self.teacher_features)) for item in compact]
        if self.codes is not None:
            self.normalized_transposes = [F.normalize(z, dim=1).T.contiguous() for z in self.codes]
        for field in ("refresh_count", "last_refresh_step", "refresh_seconds", "last_refresh_diagnostics"):
            setattr(self, field, state[field])


def relational_kl(profiles):
    teacher = profiles["teacher"].detach()
    teacher_log = profiles["teacher_log"].detach()
    return sum((teacher * (teacher_log - p)).sum(1).mean()
               for p in profiles["student_logs"]) / 2


def circular_variance_cost(z1, z2, z3):
    a, b, c = [F.normalize(z.float(), dim=1, eps=1e-8) for z in (z1, z2, z3)]
    return ((2/9) * ((1-a@b.T)[:, :, None] + (1-a@c.T)[:, None, :]
                     + (1-b@c.T)[None, :, :])).clamp(0., 1.)


def neighborhood_cost(log_p, log_q, teacher_log):
    """Bounded three-way Hellinger affinity distance over landmark profiles.

    C[i,j,k] = 1 - sum_m (p[i,m] q[j,m] teacher[k,m])**(1/3).
    Holder's inequality bounds the affinity by one for normalized profiles.
    Renormalizing in log space makes that contract independent of accumulated
    log-softmax roundoff and arbitrary row offsets. Exact zero probabilities
    may be represented by -inf, but empty supports, NaNs and +inf are invalid.
    The teacher is constant; both student profiles remain differentiable.
    """
    profiles = (log_p, log_q, teacher_log)
    if any(not isinstance(value, torch.Tensor) for value in profiles):
        raise TypeError("Neighborhood profiles must be floating-point tensors")
    if any(value.ndim != 2 or not value.shape[0] or not value.shape[1] for value in profiles):
        raise ValueError("Neighborhood profiles must be nonempty [rows, landmarks] matrices")
    if any(not value.is_floating_point() for value in profiles):
        raise TypeError("Neighborhood log-probabilities must be floating-point")
    if any(value.device != log_p.device for value in profiles):
        raise ValueError("Neighborhood profiles must share one device")
    if any(value.shape[1] != log_p.shape[1] for value in profiles):
        raise ValueError("Neighborhood profiles must share the same landmark columns")
    # Keep double precision for reference/gradient checks; never accumulate a
    # 1,024-landmark affinity in float16/bfloat16 or under autocast.
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in profiles) else torch.float32
    with torch.autocast(device_type=log_p.device.type, enabled=False):
        values = [value.to(dtype=dtype) for value in (log_p, log_q, teacher_log.detach())]
        valid = torch.stack([(~(torch.isnan(value) | torch.isposinf(value)).any())
                             & torch.isfinite(value).any(1).all() for value in values]).all()
        if not bool(valid):
            raise ValueError("Each neighborhood profile needs finite support and no NaN/+inf entries")
        roots = []
        for value in values:
            shifted = value - value.max(dim=1, keepdim=True).values
            normalized = shifted - torch.logsumexp(shifted, dim=1, keepdim=True)
            roots.append((normalized / 3).exp())
        a, b, c = roots
        # Largest intermediate is rows_p*rows_q*M, never n*n*n*M.
        pairs = (a[:, None, :] * b[None, :, :]).flatten(0, 1)
        affinity = (pairs @ c.T).reshape(len(a), len(b), len(c))
        if not bool(torch.isfinite(affinity).all()):
            raise FloatingPointError("Nonfinite neighborhood affinity")
        # GEMM reduction can overshoot one by a few float32 ulps even when
        # profiles are identical. Project the affinity to its mathematical
        # range before subtraction, so an OT kernel never sees negative cost.
        return 1 - affinity.clamp(0., 1.)


def entropy_term(x):
    positive = x > 0
    return (x[positive] * (x[positive].log() - 1)).sum()


def _validate_transport_problem(cost, mass, eta, max_iters, tol, *, batched=False):
    """Reject invalid transport inputs before forming an exponential kernel.

    These solvers implement the declared nonnegative angular/neighborhood
    costs. Their exponent clipping is not a solver for arbitrary signed costs.
    In particular, NaN residuals must not slip through comparisons with tol.
    """
    rank = 4 if batched else 3
    sides = cost.shape[1:] if batched else cost.shape
    if cost.ndim != rank or len(set(sides)) != 1 or not sides or sides[0] < 2 or (batched and cost.shape[0] < 1):
        raise ValueError("expected [groups,n,n,n] costs" if batched else "expected cubic cost of side >=2")
    if not cost.is_floating_point():
        raise TypeError("transport costs must use a floating-point dtype")
    if not bool((torch.isfinite(cost) & cost.ge(0)).all()):
        if not bool(torch.isfinite(cost).all()):
            raise ValueError("nonfinite transport cost")
        raise ValueError("negative transport cost; angular/neighborhood costs must be nonnegative")
    if (not math.isfinite(mass) or not 0 < mass < 1 or
            not math.isfinite(eta) or eta <= 0 or not math.isfinite(tol) or tol <= 0 or
            isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1):
        raise ValueError("partial solver requires finite 0<mass<1, positive finite eta/tol, and positive integer max_iters")


@torch.no_grad()
def greenkhorn_mmpot(cost, mass, eta, max_iters, tol, *, solver_dtype=torch.float32,
                    stable_kl=False):
    """Historical slack-variable solver, factored out without image imports.

    Default arithmetic and stopping criterion match csr_vs_mmpot_imagenet.py.
    Refinement may request a cancellation-resistant KL; float64 is reserved
    for tiny finite-difference reference calculations, never the training default.
    Feasibility is independently checked by matching_gap_from_cost.
    """
    _validate_transport_problem(cost, mass, eta, max_iters, tol)
    if solver_dtype not in (torch.float32, torch.float64):
        raise ValueError("solver_dtype must be float32 or float64")
    n = len(cost)
    p = torch.full((n,), 1/n, device=cost.device, dtype=solver_dtype)
    kernel = torch.exp((-cost.to(solver_dtype)/eta).clamp(-60., 0.))
    scales = [torch.ones_like(p) for _ in range(3)]
    w = cost.new_ones((), dtype=solver_dtype)
    target = cost.new_tensor(mass, dtype=solver_dtype)
    total = mass + 3*(1-mass)
    error = float("inf")

    def kl(t, v):
        if stable_kl:
            relative = (v.clamp_min(1e-12)-t)/t
            return (t*(relative-torch.log1p(relative))).clamp_min(0).sum()
        return (v-t+t.clamp_min(1e-12)*(t.clamp_min(1e-12)/v.clamp_min(1e-12)).log()).sum()

    for iteration in range(max_iters):
        v1, v2, v3 = scales
        product = kernel*v1[:, None, None]*v2[None, :, None]*v3[None, None, :]
        # Keep the historical association: regrouping the four additions can
        # change a float32 near-convergence greedy constraint choice.
        factor = total/(w*product.sum()+v1.sum()+v2.sum()+v3.sum()).clamp_min(1e-12)
        estimates = [factor*(w*product.sum(axis)+v) for axis, v in
                     zip(((1, 2), (0, 2), (0, 1)), scales)]
        current = factor*w*product.sum()
        errors = torch.stack([kl(p, v) for v in estimates]+[kl(target[None], current[None])])
        error = float(errors.max())
        if error <= tol:
            break
        worst = int(errors.argmax())
        if worst < 3:
            scales[worst].mul_(p/estimates[worst].clamp_min(1e-12)).clamp_(1e-12, 1e12)
        else:
            w.mul_(target/current.clamp_min(1e-12)).clamp_(1e-12, 1e12)
    v1, v2, v3 = scales
    product = kernel*v1[:, None, None]*v2[None, :, None]*v3[None, None, :]
    factor = total/(w*product.sum()+v1.sum()+v2.sum()+v3.sum()).clamp_min(1e-12)
    plan = factor*w*product
    marginals = [plan.sum(axis) for axis in ((1, 2), (0, 2), (0, 1))]
    slacks = [(p-v).clamp_min(0) for v in marginals]
    return plan, slacks, {"mass": float(plan.sum()), "mass_error": abs(float(plan.sum())-mass),
                          "cap_violation": max(float((v-p).clamp_min(0).max()) for v in marginals),
                          "constraint_error": error, "iterations": iteration+1}


@torch.no_grad()
def batched_greenkhorn_mmpot(costs, mass, eta, max_iters, tol, *, stable_kl=False):
    """Equivalent independent solves with GPU synchronization only every 10 steps.

    Groups retain separate scales, convergence decisions, and worst constraints.
    This batches the solver work; it does not couple examples across groups.
    """
    _validate_transport_problem(costs, mass, eta, max_iters, tol, batched=True)
    groups, n = costs.shape[:2]
    p = costs.new_full((groups, n), 1/n, dtype=torch.float32)
    kernel = torch.exp((-costs.float()/eta).clamp(-60., 0.))
    scales = costs.new_ones((groups, 3, n), dtype=torch.float32)
    w = costs.new_ones((groups,), dtype=torch.float32)
    active = torch.ones(groups, dtype=torch.bool, device=costs.device)
    stopped = torch.full((groups,), max_iters, dtype=torch.long, device=costs.device)
    target = w.new_full((groups,), mass)
    total = mass+3*(1-mass)
    axes = torch.arange(3, device=costs.device)[None, :, None]
    for iteration in range(max_iters):
        product = kernel*scales[:, 0, :, None, None]*scales[:, 1, None, :, None]*scales[:, 2, None, None, :]
        subtotal = product.sum((1,2,3))
        factor = total/(w*subtotal+scales[:, 0].sum(1)+scales[:, 1].sum(1)+scales[:, 2].sum(1)).clamp_min(1e-12)
        marginals = torch.stack((product.sum((2,3)), product.sum((1,3)), product.sum((1,2))), 1)
        estimates = factor[:, None, None]*(w[:, None, None]*marginals+scales)
        current = factor*w*subtotal
        if stable_kl:
            relative = (estimates.clamp_min(1e-12)-p[:, None])/p[:, None]
            residuals = (p[:, None]*(relative-torch.log1p(relative))).clamp_min(0).sum(2)
            relative_mass = (current.clamp_min(1e-12)-target)/target
            mass_error = (target*(relative_mass-torch.log1p(relative_mass))).clamp_min(0)
        else:
            residuals = (estimates-p[:, None]+p[:, None]*(p[:, None]/estimates.clamp_min(1e-12)).log()).sum(2)
            mass_error = current-target+target*(target/current.clamp_min(1e-12)).log()
        errors = torch.cat((residuals, mass_error[:, None]), 1)
        worst_error, worst = errors.max(1)
        converged = active & worst_error.le(tol)
        stopped = torch.where(converged, stopped.new_full((), iteration+1), stopped)
        active &= ~converged
        ratios = p[:, None]/estimates.clamp_min(1e-12)
        use_axis = active[:, None, None] & (worst[:, None, None] == axes)
        scales = torch.where(use_axis, scales*ratios, scales).clamp(1e-12, 1e12)
        w = torch.where(active & worst.eq(3), w*(target/current.clamp_min(1e-12)), w).clamp(1e-12, 1e12)
        if (iteration+1) % 10 == 0 and not bool(active.any()):
            break
    product = kernel*scales[:, 0, :, None, None]*scales[:, 1, None, :, None]*scales[:, 2, None, None, :]
    factor = total/(w*product.sum((1,2,3))+scales[:, 0].sum(1)+scales[:, 1].sum(1)+scales[:, 2].sum(1)).clamp_min(1e-12)
    plans = factor[:, None, None, None]*w[:, None, None, None]*product
    marginals = torch.stack((plans.sum((2,3)), plans.sum((1,3)), plans.sum((1,2))), 1)
    slacks = (p[:, None]-marginals).clamp_min(0)
    mass_values = plans.sum((1,2,3))
    caps = (marginals-p[:, None]).clamp_min(0).flatten(1).max(1).values
    # One transfer for every group's scalar diagnostics.
    packed = torch.stack((mass_values, (mass_values-mass).abs(), caps, worst_error,
                          stopped.float()), 1).cpu().tolist()
    records = [dict(zip(("mass", "mass_error", "cap_violation", "constraint_error", "iterations"), row))
               for row in packed]
    return plans, slacks, records


def matching_gap_from_cost(cost, *, mass=.9, eta=.2, max_iters=100, tol=1e-4,
                           require_feasible=True, max_refine_iters=2000, solved=None,
                           solver_dtype=torch.float32, stable_kl=False):
    _validate_transport_problem(cost, mass, eta, max_iters, tol)
    if isinstance(max_refine_iters, bool) or not isinstance(max_refine_iters, int) or max_refine_iters < 1:
        raise ValueError("max_refine_iters must be a positive integer")
    plan, slacks, stats = (greenkhorn_mmpot(cost.detach(), mass, eta, max_iters, tol,
                                          solver_dtype=solver_dtype, stable_kl=stable_kl)
                           if solved is None else solved)
    stats.setdefault("refinement_iterations", 0)
    stats.setdefault("objective_refinement", 0)

    def residual_error(record):
        residuals = [float(record[name]) for name in ("mass_error", "cap_violation")]
        return max(residuals) if all(math.isfinite(value) and value >= 0 for value in residuals) else float("inf")

    def refine(record, reason):
        plan, slacks, refined = greenkhorn_mmpot(cost.detach(), mass, eta, max_refine_iters,
                                                min(1e-10, tol*tol*.01),
                                                solver_dtype=solver_dtype, stable_kl=True)
        refined["refinement_iterations"] = refined["iterations"]
        refined["iterations"] += int(record.get("iterations", 0))
        refined["objective_refinement"] = int(reason == "negative_matching_gap")
        return plan, slacks, refined

    if require_feasible and not stats["refinement_iterations"] and residual_error(stats) > tol:
        plan, slacks, stats = refine(stats, "marginal_feasibility")
    if require_feasible and residual_error(stats) > tol:
        raise RuntimeError(f"partial OT is infeasible after bounded refinement: {stats}")
    diagonal = torch.arange(len(cost), device=cost.device)
    ref_plan = cost.new_full((len(cost),), mass/len(cost))
    ref_slack = cost.new_full((len(cost),), (1-mass)/len(cost))

    def gap_value(current_plan, current_slacks):
        gradient_gap = mass*cost[diagonal, diagonal, diagonal].mean() - (cost*current_plan.detach()).sum()
        entropy = entropy_term(ref_plan)+3*entropy_term(ref_slack)
        entropy = entropy - entropy_term(current_plan) - sum(entropy_term(q) for q in current_slacks)
        loss = gradient_gap + eta*entropy.detach()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite estimated OT gap")
        return loss

    loss = gap_value(plan, slacks)
    gap_tolerance = max(tol, 1e-6)
    # Marginal feasibility alone does not certify an accurate optimizer. A
    # feasible but suboptimal plan can make the estimated reference gap negative.
    # Retry the actual objective before declaring failure; never hide the issue
    # by clipping a materially negative gap or changing its envelope gradient.
    if require_feasible and float(loss.detach()) < -gap_tolerance and not stats["refinement_iterations"]:
        plan, slacks, stats = refine(stats, "negative_matching_gap")
        if residual_error(stats) > tol:
            raise RuntimeError(f"partial OT is infeasible after bounded refinement: {stats}")
        loss = gap_value(plan, slacks)
    stats["negative_gap"] = float(loss.detach()) < -gap_tolerance
    stats["diagonal_plan_mass"] = float(plan[diagonal, diagonal, diagonal].sum())
    if require_feasible and stats["negative_gap"]:
        raise RuntimeError(f"materially negative estimated OT gap: {float(loss.detach())}")
    delta = -plan.detach().clone()
    delta[diagonal, diagonal, diagonal] += mass/len(cost)
    return loss, stats, delta


def mixed_matching_gap(z1, z2, z3, *, profiles=None, mixture_weight=0., mass=.9,
                       eta=.2, max_iters=100, tol=1e-4, microbatch=32,
                       require_feasible=True, max_refine_iters=2000, diagnostics=False):
    if not 0 <= mixture_weight <= 1 or microbatch < 2:
        raise ValueError("invalid mixture weight or OT microbatch")
    if mixture_weight and profiles is None:
        raise ValueError("neighborhood OT requires profiles")
    losses, records, old_terms, new_terms = [], [], [], []
    marginals = []
    prepared = []
    for start in range(0, len(z1), microbatch):
        stop = min(start+microbatch, len(z1))
        if stop-start < 2:
            continue
        old = circular_variance_cost(z1[start:stop], z2[start:stop], z3[start:stop])
        new = None
        if mixture_weight:
            new = neighborhood_cost(profiles["student_logs"][0][start:stop],
                                    profiles["student_logs"][1][start:stop],
                                    profiles["teacher_log"][start:stop])
        cost = old if new is None else (1-mixture_weight)*old+mixture_weight*new
        prepared.append((old, new, cost))
    solved_groups = {}
    # Group costs have at most 32^3 entries each. Batched plans substantially
    # reduce tiny launches and host synchronizations at the 4096-row recipe.
    for side in sorted({len(item[2]) for item in prepared}):
        positions = [i for i, item in enumerate(prepared) if len(item[2]) == side]
        if z1.is_cuda and len(positions) > 1:
            costs = torch.stack([prepared[i][2].detach() for i in positions])
            plans, slacks, solver_records = batched_greenkhorn_mmpot(costs, mass, eta, max_iters, tol)
            failing = [j for j, rec in enumerate(solver_records)
                       if require_feasible and max(rec["mass_error"], rec["cap_violation"]) > tol]
            if failing:
                refined_plans, refined_slacks, refined_records = batched_greenkhorn_mmpot(
                    costs[failing], mass, eta, max_refine_iters, min(1e-10, tol*tol*.01), stable_kl=True)
                for slot, j in enumerate(failing):
                    initial_iters = solver_records[j]["iterations"]
                    plans[j], slacks[j] = refined_plans[slot], refined_slacks[slot]
                    solver_records[j] = refined_records[slot]
                    solver_records[j]["refinement_iterations"] = refined_records[slot]["iterations"]
                    solver_records[j]["iterations"] += initial_iters
            for j, position in enumerate(positions):
                solved_groups[position] = (plans[j], list(slacks[j].unbind(0)), solver_records[j])
    for position, (old, new, cost) in enumerate(prepared):
        loss, record, delta = matching_gap_from_cost(cost, mass=mass, eta=eta, max_iters=max_iters,
                          tol=tol, require_feasible=require_feasible, max_refine_iters=max_refine_iters,
                          solved=solved_groups.get(position))
        losses.append(loss)
        record.update({"cost0_mean": float(old.detach().mean()),
                       "cost0_min": float(old.detach().min()), "cost0_max": float(old.detach().max()),
                       "costmix_min": float(cost.detach().min()), "costmix_max": float(cost.detach().max()),
                       "costmix_std": float(cost.detach().std(unbiased=False)),
                       "cost0_std_over_eta": float(old.detach().std(unbiased=False))/eta,
                       "costmix_std_over_eta": float(cost.detach().std(unbiased=False))/eta})
        if new is not None:
            record["costnbr_mean"] = float(new.detach().mean())
            record["costnbr_min"] = float(new.detach().min())
            record["costnbr_max"] = float(new.detach().max())
            record["costnbr_std_over_eta"] = float(new.detach().std(unbiased=False))/eta
            ids = torch.arange(len(new), device=new.device)
            record["diagonal_affinity"] = float((1-new[ids, ids, ids]).detach().mean())
            record["shuffled_affinity"] = float((1-new[ids, ids.roll(1), ids.roll(2)]).detach().mean())
        records.append(record)
        if diagnostics:
            old_terms.append(((1-mixture_weight)*old*delta).sum())
            new_terms.append(old.new_zeros(()) if new is None else (mixture_weight*new*delta).sum())
            plan = -delta.clone()
            ids = torch.arange(len(cost), device=cost.device)
            plan[ids, ids, ids] += mass/len(cost)
            marginals.append([plan.sum(axis).detach().cpu().tolist() for axis in ((1,2),(0,2),(0,1))])
    if not losses:
        raise ValueError("no valid OT groups")
    stats = {key: sum(float(item[key]) for item in records)/len(records) for key in records[0]}
    for key in records[0]:
        if key.startswith("cost") and key.endswith("_min"):
            stats[key] = min(item[key] for item in records)
        elif key.startswith("cost") and key.endswith("_max"):
            stats[key] = max(item[key] for item in records)
    stats["mass_error_max"] = max(item["mass_error"] for item in records)
    stats["cap_violation_max"] = max(item["cap_violation"] for item in records)
    context = {}
    if diagnostics:
        context = {"old_ot": torch.stack(old_terms).mean(), "new_ot": torch.stack(new_terms).mean(),
                   "transported_marginals": marginals}
    return torch.stack(losses).mean(), stats, context


def research_loss(sae, x, k, *, repair=None, mask=False, mask_generator=None,
                  auxiliary_budget=None, bank=None, row_ids=None, relation_weight=0.,
                  ot_weight=2., mixture_weight=0., teacher_temperature=.1,
                  student_temperature=.1, mass=.9, eta=.2, max_iters=100, tol=1e-4,
                  microbatch=32, require_feasible=True, max_refine_iters=2000,
                  diagnostics=False, update_activity=True, k_aux=512,
                  clean_probability=.2, erase_probability=.25, anchor_chunk_size=128,
                  allow_empty_codes=False):
    """``allow_empty_codes`` keeps training through all-zero final codes, as
    the original MPSAE runner does; their count is reported, the objective is
    unchanged (an all-zero view has zero normalized direction in the OT cost)."""
    reconstruction, stats, views = reconstruction_terms(sae, x, k, auxiliary_budget,
        repair=repair, mask=mask, mask_generator=mask_generator, update_activity=update_activity,
        k_aux=k_aux, clean_probability=clean_probability, erase_probability=erase_probability)
    for name in ("final_k", "final_2k", "raw_4k"):
        z = views[name]
        if not bool(torch.isfinite(z).all()):
            raise ValueError(f"invalid final code in {name}")
        if not allow_empty_codes and bool(z.gt(0).sum(1).eq(0).any()):
            raise ValueError(f"invalid final code in {name}")
    profiles = None
    relation = x.new_zeros(())
    gap = x.new_zeros(())
    # Skip disabled mechanisms entirely; C does not even need a bank.
    if relation_weight or (ot_weight and mixture_weight):
        if bank is None or row_ids is None:
            raise ValueError("relational arms require a refreshed bank and row IDs")
        profiles = bank.profiles(x, row_ids, views["final_k"], views["final_2k"], k=k,
              teacher_temperature=teacher_temperature, student_temperature=student_temperature,
              anchor_chunk=anchor_chunk_size)
        stats.update(profiles["stats"])
        if relation_weight:
            relation = relational_kl(profiles)
    components = {"reconstruction": reconstruction, "relation": relation_weight*relation}
    if ot_weight:
        gap, solver_stats, context = mixed_matching_gap(views["final_k"], views["final_2k"],
              views["raw_4k"], profiles=profiles, mixture_weight=mixture_weight, mass=mass,
              eta=eta, max_iters=max_iters, tol=tol, microbatch=microbatch,
              require_feasible=require_feasible, max_refine_iters=max_refine_iters, diagnostics=diagnostics)
        cost_fields = {"cost0_mean", "costnbr_mean", "costmix_std", "cost0_min", "cost0_max",
                       "costnbr_min", "costnbr_max", "costmix_min", "costmix_max"}
        stats.update({"solver_"+key: value for key, value in solver_stats.items()
                      if key not in cost_fields})
        stats.update({key: value for key, value in solver_stats.items()
                      if key in cost_fields})
        stats["solver_capacity_violation"] = solver_stats["cap_violation"]
        if diagnostics:
            components.update({key: ot_weight*context[key] for key in ("old_ot", "new_ot")})
            views["transported_marginals"] = context["transported_marginals"]
    total = reconstruction + relation_weight*relation + ot_weight*gap
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite research objective")
    stats.update({"loss_total": total, "loss_rel_unweighted": relation, "gap_value": gap,
                  "weighted_rel": relation_weight*relation, "weighted_ot": ot_weight*gap})
    views["component_losses"] = components
    views["profiles"] = profiles
    if diagnostics:
        stats.update(gradient_diagnostics(components, views))
        raw_support = views["raw_k"].gt(0)
        final_support = views["final_k"].gt(0)
        stats["support_change_rate"] = (~raw_support.eq(final_support).all(1)).float().mean()
        stats["active_unit_coverage"] = final_support.any(0).float().mean()
    return total, stats, views


def gradient_diagnostics(components, views, threshold=1e-10):
    """Common tensor space includes raw 4K, whose new-cost derivative is zero."""
    tensors = [views[name] for name in ("final_k", "final_2k", "raw_4k")]
    selected = [tensor.detach().gt(0) for tensor in tensors]
    records = {}
    for name, loss in components.items():
        grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True) if loss.requires_grad else (None,)*3
        grads = [torch.zeros_like(tensor) if grad is None else grad for grad, tensor in zip(grads, tensors)]
        records[f"gradient_{name}_norm"] = sum(grad.square().sum() for grad in grads).sqrt()
        selected_count = sum(mask.sum() for mask in selected).clamp_min(1)
        small_count = sum((grad.abs().le(threshold) & mask).sum() for grad, mask in zip(grads, selected))
        records[f"gradient_{name}_selected_near_zero_fraction"] = small_count/selected_count
        for budget_name, grad, mask in zip(("k", "2k", "raw_4k"), grads, selected):
            records[f"gradient_{name}_{budget_name}_norm"] = grad.norm()
            records[f"gradient_{name}_{budget_name}_selected_near_zero_fraction"] = (
                (grad.abs().le(threshold) & mask).sum()/mask.sum().clamp_min(1))
    return records
