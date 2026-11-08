"""One training objective per method, sharing a single autoencoder.

    method          K during training   gates per step      regulariser
    -------------   -----------------   -----------------   ------------------------
    csr             fixed K             K, 4K, K            gamma * inbatch_cl
    csrv2           annealed k(t)       k, 4k, k            gamma * label_based_cl
    mpsae           fixed K             K, 4K, 2K           2 * OT matching gap
    mpsaev2         annealed k(t)       k, 4k, 2k           2 * OT matching gap
    csr_multitopk   fixed K             K, 4K, 2K           gamma * inbatch_cl  (+ L(2K)/8)

Every method gates exactly three times per step, so the dead-latent counter --
and therefore ``dead_threshold`` -- behaves identically across the comparison.
For the baselines that is the released sequence verbatim: CSR runs three
forward passes, the third only to produce a duplicate contrastive view. Our
methods spend the third gate on the 2K view they actually consume. The
``csr_multitopk`` ablation deliberately matches our gate sequence, so the only
difference from ``csr`` is the extra ``L(2K)/8`` reconstruction term and the
objective differences from ``mpsae`` are the extra reconstruction and OT
versus contrastive. MPSAE uses batches of 1024; baseline batch sizes follow
the backbone preset, so update counts can also differ.

Shared objective (released CSR):

    L = L(k) + 0.125 * L(4k) + (1/32) * NMSE_aux + gamma * L_reg

``mpsaev2`` is the "variant B" the user selected: the *main* budget anneals, so
it is CSRv2 with the contrastive term replaced by the OT gap, and a comparison
against ``csrv2`` also includes the MPSAE batch-size setting. The alternative
(fixed main budget, annealed OT views only) is recorded in
``docs/notes/mpsaev2_legacy_A.md`` and is not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from . import losses, ot
from .config import METHODS, TrainPreset, initial_topk
from .model import SparseAutoencoder
from .schedules import current_k

#: Methods whose main Top-K budget follows the CSRv2 annealing curriculum.
ANNEALED = ("csrv2", "mpsaev2")
#: Methods regularised by the OT matching gap instead of a contrastive loss.
OT_METHODS = ("mpsae", "mpsaev2")
#: Methods using a released contrastive loss.
CL_METHODS = ("csr", "csrv2", "csr_multitopk")
#: Methods with the additional L(2K)/8 reconstruction term.
MULTITOPK = ("csr_multitopk",)


@dataclass
class StepOutput:
    """Everything one training step produces."""

    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    metrics: dict[str, float] = field(default_factory=dict)
    training_k: int = 0


def requires_labels(method: str) -> bool:
    """``csrv2`` is the only method whose objective reads the label column."""
    _check(method)
    return method == "csrv2"


def _check(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def plan_budget(method: str, k: int, preset: TrainPreset, *, step: int, total_steps: int) -> int:
    """The main Top-K budget this step, after any annealing."""
    _check(method)
    if method not in ANNEALED:
        return k
    start = initial_topk(k, preset.hidden_dim)
    return current_k(step, start, k, total_steps, preset.k_decay_ratio)


def describe(method: str, k: int, preset: TrainPreset) -> dict[str, Any]:
    """Serialisable description of the objective, stored in the run identity."""
    _check(method)
    described: dict[str, Any] = {
        "method": method,
        "target_k": k,
        "hidden_dim": preset.hidden_dim,
        "reconstruction": ["L(k)", "L(4k)/8"] + (["L(2k)/8"] if method in MULTITOPK else []),
        "aux": {"k_aux": preset.k_aux, "coef": preset.aux_coef,
                "dead_threshold": preset.dead_threshold, "gates_per_step": 3},
        "gate_sequence": ["k", "4k", "k" if method in CL_METHODS and method not in MULTITOPK else "2k"],
    }
    if method in ANNEALED:
        described["annealing"] = {"shape": "linear", "k_init": initial_topk(k, preset.hidden_dim),
                                  "k_final": k, "decay_ratio": preset.k_decay_ratio,
                                  "reference": "CSRv2 get_current_k"}
    if method in CL_METHODS:
        described["contrastive"] = {
            "loss": "label_based_cl" if method == "csrv2" else "inbatch_cl",
            "weight": preset.cl_coef, "temperature": preset.cl_temperature,
            "views": "two identical deterministic forward passes",
        }
    if method in OT_METHODS:
        described["ot"] = {"weight": preset.ot_weight, "cost": "angular", "mass": preset.ot_mass,
                           "eta": preset.ot_eta, "microbatch": preset.ot_microbatch,
                           "max_iters": preset.ot_max_iters, "tol": preset.ot_tol,
                           "views": ["k", "2k", "4k"]}
    return described


def compute_step(method: str, sae: SparseAutoencoder, x: torch.Tensor, *,
                 k: int, preset: TrainPreset, step: int, total_steps: int,
                 labels: torch.Tensor | None = None,
                 update_activity: bool = True) -> StepOutput:
    """Evaluate one method's objective on one batch.

    Pre-activations are computed once and shared by every view. That is
    value- and gradient-identical to the release's three forward passes -- the
    passes are deterministic and the encoder is linear, so gradients simply
    accumulate through one node instead of three -- but it costs a third of
    the encoder work. The *gate* count is not collapsed, because it drives the
    dead-latent counter.
    """
    _check(method)
    if x.ndim != 2 or not len(x):
        raise ValueError("expected a nonempty [batch, dim] float batch")
    if x.dtype != torch.float32:
        raise TypeError(f"training runs in float32; got {x.dtype}")
    if not 0 <= step < total_steps:
        raise ValueError(f"step {step} outside [0, {total_steps})")
    if requires_labels(method):
        if labels is None:
            raise ValueError("csrv2's label_based_cl needs the label column; "
                             "the embedding archive must carry real 'labels'")
        if labels.ndim != 1 or len(labels) != len(x):
            raise ValueError("labels must be one integer per row")

    hidden = sae.hidden_dim
    budget = plan_budget(method, k, preset, step=step, total_steps=total_steps)
    wide = min(4 * budget, hidden)
    middle = min(2 * budget, hidden)

    pre = sae.pre_activations(x)
    z_main, _ = sae.gate(pre, budget)
    # Gate 1 of 3. The dead mask must be read after this update, as released.
    if update_activity:
        sae.count_activity(pre, budget)
    z_aux = sae.auxiliary_latents(pre, preset.k_aux)
    reconstruction = sae.decode(z_main)

    z_wide, _ = sae.gate(pre, wide)
    if update_activity:
        sae.count_activity(pre, wide)             # gate 2 of 3

    needs_middle = method in OT_METHODS or method in MULTITOPK
    z_middle = sae.gate(pre, middle)[0] if needs_middle else None
    if update_activity:
        # Gate 3 of 3: the released duplicate K view for the contrastive
        # methods, the 2K view for ours.
        sae.count_activity(pre, middle if needs_middle else budget)

    main_term = losses.mse(reconstruction, x)
    wide_term = losses.mse(sae.decode(z_wide), x)
    aux_term = losses.normalized_mse(
        sae.decode(z_aux), losses.auxk_target(x, reconstruction, sae.pre_bias))

    components: dict[str, torch.Tensor] = {
        "mse_k": main_term,
        "mse_4k": preset.nested_coef * wide_term,
        "aux": preset.aux_coef * aux_term,
    }
    metrics: dict[str, float] = {
        "training_k": budget, "view_2k": middle, "view_4k": wide,
        "loss_mse_k": float(main_term.detach()),
        "loss_mse_4k": float(wide_term.detach()),
        "loss_aux": float(aux_term.detach()),
    }

    if method in MULTITOPK:
        middle_term = losses.mse(sae.decode(z_middle), x)
        components["mse_2k"] = preset.nested_coef * middle_term
        metrics["loss_mse_2k"] = float(middle_term.detach())

    if method in CL_METHODS:
        # The release feeds two identical deterministic passes; passing the
        # same tensor twice is exactly that, without a redundant gate.
        if method == "csrv2":
            regulariser = losses.label_based_cl(z_main, z_main, labels, preset.cl_temperature)
        else:
            regulariser = losses.inbatch_cl(z_main, z_main, preset.cl_temperature)
        components["contrastive"] = preset.cl_coef * regulariser
        metrics["loss_contrastive"] = float(regulariser.detach())
    else:
        gap, gap_stats = ot.matching_gap(
            z_main, z_middle, z_wide, mass=preset.ot_mass, eta=preset.ot_eta,
            max_iters=preset.ot_max_iters, tol=preset.ot_tol,
            microbatch=preset.ot_microbatch, max_refine_iters=preset.ot_max_refine_iters)
        components["ot"] = preset.ot_weight * gap
        metrics["loss_ot"] = float(gap.detach())
        metrics.update({f"ot_{name}": value for name, value in gap_stats.items()})

    total = sum(components.values())
    if not bool(torch.isfinite(total)):
        detail = {name: float(value.detach()) for name, value in components.items()}
        raise FloatingPointError(f"nonfinite {method} objective: {detail}")
    metrics["loss_total"] = float(total.detach())
    metrics.update(sae.sparsity_report(z_main, budget))
    return StepOutput(loss=total, components=components, metrics=metrics, training_k=budget)
