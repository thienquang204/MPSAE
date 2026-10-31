"""Public MMRSAE family: a tied sparse encoder with one residual repair step.

The mathematical kernels remain shared with ``mpsae_repair``. This interface
selects one manually configured method; it does not create an ablation grid,
download data, run diagnostics, or start training on import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn

from mpsae_repair import (
    LandmarkBank,
    RepairModule,
    ResearchSAE,
    auxiliary_budget,
    backtrack_alpha,
    compact_codes,
    encode_repaired,
    estimate_alpha,
    research_loss,
)


VARIANTS = ("mmrsae", "mmrsae-rel", "mmrsae-wide", "mmrsae-annealed")


@dataclass(frozen=True)
class MMRSAEConfig:
    """Resolved, serializable settings for exactly one MMRSAE fit.

    ``None`` means use the selected variant's default, rather than ignoring an
    explicit zero. Set ``alpha`` to a positive value to fix it manually; otherwise
    call :meth:`MMRSAE.initialize_repair` once on a training-only feature batch.
    The default reconstruction weights are shared with the original method.
    """

    input_dim: int
    hidden_dim: int = 8192
    k: int = 8
    variant: str = "mmrsae"
    rank: int = 32
    seed: int = 1227
    alpha: float | None = None
    mask: bool = True
    relation_weight: float = 1.0
    ot_weight: float | None = None
    mixture_weight: float | None = None
    auxiliary_mode: str | None = None
    teacher_temperature: float = 0.1
    student_temperature: float = 0.1
    mass: float = 0.9
    eta: float = 0.2
    max_iters: int = 100
    max_refine_iters: int = 2000
    tol: float = 1e-4
    microbatch: int = 32
    k_aux: int = 512
    dead_steps: int = 30
    clean_probability: float = 0.2
    erase_probability: float = 0.25
    anchor_chunk_size: int = 128

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown MMRSAE variant {self.variant!r}; choose {VARIANTS}")
        defaults = {
            "ot_weight": 0.0 if self.variant == "mmrsae-rel" else 2.0,
            "mixture_weight": 0.0 if self.variant == "mmrsae-rel" else 0.5,
            "auxiliary_mode": {"mmrsae-wide": "wide", "mmrsae-annealed": "anneal"}.get(
                self.variant, "baseline"),
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        for name in ("input_dim", "hidden_dim", "k", "rank", "max_iters", "max_refine_iters",
                     "microbatch", "anchor_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dead_steps", "k_aux", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.k > self.hidden_dim:
            raise ValueError("k must not exceed hidden_dim")
        if self.microbatch < 2:
            raise ValueError("OT microbatch must contain at least two examples")
        if self.hidden_dim > torch.iinfo(torch.int32).max:
            raise ValueError("hidden_dim exceeds the retrieval index's int32 coordinate range")
        if self.max_refine_iters < self.max_iters:
            raise ValueError("max_refine_iters must be at least max_iters")
        if not isinstance(self.mask, bool):
            raise ValueError("mask must be a boolean")
        for name in ("relation_weight", "ot_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("mixture_weight", "clean_probability", "erase_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("teacher_temperature", "student_temperature", "eta", "tol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.mass) or not 0 < self.mass < 1:
            raise ValueError("The partial-transport solver requires 0 < mass < 1")
        if self.alpha is not None and (not math.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("alpha must be finite and positive when provided")
        if self.auxiliary_mode not in ("baseline", "wide", "anneal"):
            raise ValueError("auxiliary_mode must be baseline, wide, or anneal")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MMRSAE(nn.Module):
    """One reusable MMRSAE encoder, objective and checkpoint state.

    All variants serve a repaired nonnegative Top-K code. Training may erase
    initial coefficients, but inference and landmark refresh always use clean
    initial codes. Transport and landmark profiles are training-only.

    ``sae`` and ``repair`` remain accessible to the shared GPU benchmark and
    landmark-bank implementation. Save this module's complete ``state_dict``:
    it includes activity counters, the fixed alpha and its calibration status.
    """

    def __init__(self, config: MMRSAEConfig):
        super().__init__()
        self.config = config
        # ResearchSAE initializes on CPU. Preserve the caller's CPU RNG and do
        # not seed or initialize CUDA generators as a constructor side effect.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.sae = ResearchSAE(config.input_dim, config.hidden_dim, config.dead_steps)
        self.repair = RepairModule(config.input_dim, config.hidden_dim, config.rank,
                                   alpha=config.alpha if config.alpha is not None else 1.0,
                                   seed=config.seed)
        self.register_buffer("repair_initialized", torch.tensor(config.alpha is not None))
        # A Python flag avoids a device synchronization on every encode/loss.
        # It is restored explicitly with the registered buffer when loading.
        self._repair_ready = config.alpha is not None

    @property
    def needs_bank(self) -> bool:
        return bool(self.config.relation_weight or
                    (self.config.ot_weight and self.config.mixture_weight))

    def _require_ready(self):
        if not self._repair_ready:
            raise RuntimeError("Call initialize_repair(training_features) after moving MMRSAE "
                               "to its training device, or load a calibrated checkpoint")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._repair_ready = bool(self.repair_initialized.item())

    @torch.no_grad()
    def initialize_repair(self, training_features: torch.Tensor, *, iterations: int = 40,
                          max_halvings: int = 30) -> dict[str, float | int | str]:
        """Set alpha once using the initial dictionary and training-only rows.

        This checks local reconstruction descent for fixed IHT, not convergence
        of the learned repair or accuracy. Never recalibrate after resuming:
        alpha is part of the saved model and changing it changes the method.
        """
        if self._repair_ready:
            return {"alpha": float(self.repair.alpha), "alpha_source": "configured_or_restored"}
        if training_features.ndim != 2 or training_features.shape[1] != self.config.input_dim:
            raise ValueError("Alpha calibration requires [examples, input_dim] training features")
        if len(training_features) < 1 or training_features.device != self.sae.decoder.device:
            raise ValueError("Alpha calibration requires nonempty features on the model device")
        if training_features.dtype != torch.float32 or not bool(torch.isfinite(training_features).all()):
            raise ValueError("Alpha calibration requires finite float32 features")
        if max_halvings < 0:
            raise ValueError("max_halvings must be nonnegative")
        alpha = estimate_alpha(self.sae.decoder, iterations=iterations, seed=self.config.seed + 1)
        alpha, diagnostics = backtrack_alpha(self.sae, training_features, self.config.k,
                                             alpha, max_halvings=max_halvings)
        self.repair.alpha.fill_(alpha)
        self.repair_initialized.fill_(True)
        self._repair_ready = True
        return {"alpha": alpha, "alpha_source": "training_only_power_estimate_and_backtracking",
                **diagnostics}

    def encode(self, x: torch.Tensor, k: int | None = None, *, compact: bool = False):
        """Clean repaired code without activity or RNG side effects.

        The normal result is the unnormalized [B,h] code. ``compact=True``
        returns (int32 coordinates, float32 L2-normalized values), each [B,K],
        directly suitable for CUDA retrieval. Zero/nonfinite compact rows fail.
        """
        self._require_ready()
        k = self.config.k if k is None else k
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self.config.hidden_dim:
            raise ValueError("Requested K must be an integer in [1, hidden_dim]")
        if x.ndim != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError("MMRSAE expects [examples, input_dim] features")
        if x.dtype != torch.float32 or self.sae.decoder.dtype != torch.float32:
            raise TypeError("MMRSAE's declared implementation uses float32 inputs and parameters")
        z = encode_repaired(self.sae, self.repair, x, k)
        if compact:
            indices, values = compact_codes(z, k)
            return indices.to(torch.int32).contiguous(), values.contiguous()
        return z

    def forward(self, x: torch.Tensor, k: int | None = None):
        return self.encode(x, k)

    def loss(self, x: torch.Tensor, *, bank: LandmarkBank | None = None,
             row_ids=None, mask_generator: torch.Generator | None = None,
             step: int = 0, total_steps: int = 1, diagnostics: bool = False,
             update_activity: bool = True):
        """Return (scalar loss, statistics, views) for one training batch.

        Expensive component-gradient diagnostics are opt-in. The bank is
        refreshed by the caller at a declared interval and must be checkpointed
        separately together with the mask RNG and optimizer.
        """
        self._require_ready()
        if step < 0 or total_steps < 1 or step >= total_steps:
            raise ValueError("step must be in [0, total_steps)")
        if x.ndim != 2 or x.shape[1] != self.config.input_dim or not len(x):
            raise ValueError("MMRSAE loss expects nonempty [examples, input_dim] features")
        if x.dtype != torch.float32 or self.sae.decoder.dtype != torch.float32:
            raise TypeError("MMRSAE training requires float32 inputs and parameters")
        if self.config.ot_weight and len(x) < 2:
            raise ValueError("An OT-enabled training batch requires at least two examples")
        if self.needs_bank and (bank is None or row_ids is None):
            raise ValueError("This MMRSAE configuration requires landmark profiles and row IDs")
        if row_ids is not None and len(row_ids) != len(x):
            raise ValueError("Training row IDs must align with feature rows")
        cfg = self.config
        budget = auxiliary_budget(cfg.k, cfg.hidden_dim, step, total_steps, cfg.auxiliary_mode)
        return research_loss(
            self.sae, x, cfg.k, repair=self.repair, mask=cfg.mask,
            mask_generator=mask_generator, auxiliary_budget=budget, bank=bank, row_ids=row_ids,
            relation_weight=cfg.relation_weight, ot_weight=cfg.ot_weight,
            mixture_weight=cfg.mixture_weight, teacher_temperature=cfg.teacher_temperature,
            student_temperature=cfg.student_temperature, mass=cfg.mass, eta=cfg.eta,
            max_iters=cfg.max_iters, max_refine_iters=cfg.max_refine_iters, tol=cfg.tol,
            microbatch=cfg.microbatch, require_feasible=True, diagnostics=diagnostics,
            update_activity=update_activity, k_aux=cfg.k_aux,
            clean_probability=cfg.clean_probability, erase_probability=cfg.erase_probability,
            anchor_chunk_size=cfg.anchor_chunk_size,
        )
