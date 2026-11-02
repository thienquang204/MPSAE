"""Public MMRSAE family: landmark synthesis followed by a tied sparse encoder.

The mathematical kernels remain shared with ``mpsae_repair``. This interface
selects one manually configured method; it does not create an ablation grid,
download data, run diagnostics, or start training on import.

The ``mpsae`` and ``csr`` baselines reuse the same tied SAE, initialization,
reconstruction terms and serving path without repair, masking or landmark KL,
so they can be fitted and evaluated under the identical MMRSAE protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from mmrsae_dictionary import LandmarkConditionedSAE
from mmrsae_methods import canonical_method

from mpsae_repair import (
    LandmarkBank,
    ResearchSAE,
    auxiliary_budget,
    compact_codes,
    encode_repaired,
    gradient_diagnostics,
    reconstruction_terms,
    research_loss,
)


VARIANTS = ("mmrsae", "mmrsae-rel", "mmrsae-wide", "mmrsae-annealed", "dlmmsae", "dlmmsaev2")
# Historical ablations remain available to the library; the benchmark CLI is
# restricted to mmrsae_methods.BENCHMARK_METHODS.
# Comparison methods without repair: MPSAE, fixed-K CSR and cosine-annealed CSRv2.
BASELINES = ("mpsae", "csr", "csrv2")
METHODS = VARIANTS + BASELINES


def cosine_annealed_k(step, total_steps, start_k, target_k, anneal_fraction=.7):
    """CSRv2's exact half-cosine formula, shared by CSRv2 and DLMMSAEv2."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    anneal_steps = max(1, math.ceil(total_steps * anneal_fraction))
    if anneal_steps == 1 or step >= anneal_steps - 1:
        return target_k
    progress = step / (anneal_steps - 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    value = target_k + (start_k - target_k) * cosine_decay
    return min(start_k, max(target_k, int(round(value))))


def nonnegative_contrastive_loss(codes: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Released CSR two-view NCL, including L2 normalization and temperature.

    Matches ``csr_vs_mmpot_imagenet.nonnegative_contrastive_loss``: both views
    are the same deterministic code, as in the released CSR training loop.
    """
    batch = codes.shape[0]
    identities = torch.arange(batch, device=codes.device).repeat(2)
    normalized = F.normalize(torch.cat((codes.float(), codes.float()), dim=0), dim=-1)
    logits = (normalized @ normalized.T) / temperature
    same_sample = identities[:, None].eq(identities[None, :])
    same_sample.fill_diagonal_(False)
    negatives = ~identities[:, None].eq(identities[None, :])
    positive_logits = logits.masked_fill(~same_sample, -torch.inf)
    denominator_logits = logits.masked_fill(~(same_sample | negatives), -torch.inf)
    return -(torch.logsumexp(positive_logits, dim=1)
             - torch.logsumexp(denominator_logits, dim=1)).mean()


@dataclass(frozen=True)
class MMRSAEConfig:
    """Resolved, serializable settings for exactly one MMRSAE fit.

    ``None`` means use the selected variant's default, rather than ignoring an
    explicit zero. Baselines (``mpsae``, ``csr``, ``csrv2``) have no repair,
    masking or landmark KL. All enabled OT objectives use angular cost only.
    ``allow_empty_codes`` (default) follows the original MPSAE runner: an
    all-zero final code does not stop training; it is counted and reported.
    Initialize the landmark dictionary once using training-only anchors.
    The default reconstruction weights are shared with the original method.
    """

    input_dim: int
    hidden_dim: int = 8192
    k: int = 8
    variant: str = "mmrsae"
    rank: int = 32
    seed: int = 1227
    alpha: float | None = None
    mask: bool | None = None
    relation_weight: float | None = None
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
    contrastive_weight: float | None = None
    contrastive_temperature: float = 0.2
    k_aux: int = 512
    dead_steps: int = 30
    clean_probability: float = 0.2
    erase_probability: float = 0.25
    anchor_chunk_size: int = 128
    allow_empty_codes: bool = True
    anneal_start_k: int | None = None
    anneal_fraction: float = .7
    dictionary_temperature: float = .1
    dictionary_blend: float = .1
    context_weight: float | None = None
    context_teacher_temperature: float = .1
    context_student_temperature: float = .1
    context_noise_std: float = .02
    architecture: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "variant", canonical_method(self.variant))
        if self.variant not in METHODS:
            raise ValueError(f"Unknown MMRSAE variant {self.variant!r}; choose {METHODS}")
        baseline = self.variant in BASELINES
        object.__setattr__(self, "architecture", "plain-topk-v1" if baseline else
                           ("landmark-context-distilled-sae-v1" if self.is_dlmmsae else "landmark-synthesis-sae-v1"))
        defaults = {
            "mask": not baseline,
            "relation_weight": 0.0 if baseline else 1.0,
            "ot_weight": 0.0 if self.variant in ("mmrsae-rel", "csr", "csrv2") else 2.0,
            "mixture_weight": 0.0,
            "auxiliary_mode": "wide" if self.variant == "mmrsae-wide" else "baseline",
            # CSR gamma: 0.1 for FF2048 ResNet-50, 1.0 for the ResNet-18 study.
            "contrastive_weight": (0.1 if self.input_dim == 2048 else 1.0) if self.variant in ("csr", "csrv2") else 0.0,
            "anneal_start_k": min(self.hidden_dim, max(64, 4 * self.k)),
            "context_weight": 1.0 if self.is_dlmmsae else 0.0,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        if baseline and (self.mask or self.relation_weight or self.mixture_weight
                         or self.auxiliary_mode != "baseline"):
            raise ValueError(f"Baseline {self.variant!r} has no repair, masking, landmark KL, "
                             "neighborhood OT or auxiliary-budget schedule")
        if self.mixture_weight != 0:
            raise ValueError("MMRSAE objectives use angular OT only; mixture_weight must be 0")
        if not self.is_dlmmsae and self.context_weight:
            raise ValueError("Only DLMMSAE variants use dictionary context distillation")
        if self.variant not in ("csr", "csrv2") and self.contrastive_weight:
            raise ValueError("Only csr/csrv2 baselines use the contrastive objective")
        if self.variant in ("csr", "csrv2") and self.ot_weight:
            raise ValueError("CSR/CSRv2 use NCL without OT")
        for name in ("input_dim", "hidden_dim", "k", "rank", "max_iters", "max_refine_iters",
                     "microbatch", "anchor_chunk_size", "anneal_start_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dead_steps", "k_aux", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.k > self.hidden_dim:
            raise ValueError("k must not exceed hidden_dim")
        if not self.k <= self.anneal_start_k <= self.hidden_dim:
            raise ValueError("anneal_start_k must be between k and hidden_dim")
        if not math.isfinite(self.anneal_fraction) or not 0 < self.anneal_fraction <= 1:
            raise ValueError("anneal_fraction must be in (0,1]")
        if self.microbatch < 2:
            raise ValueError("OT microbatch must contain at least two examples")
        if self.hidden_dim > torch.iinfo(torch.int32).max:
            raise ValueError("hidden_dim exceeds the retrieval index's int32 coordinate range")
        if self.max_refine_iters < self.max_iters:
            raise ValueError("max_refine_iters must be at least max_iters")
        if not isinstance(self.mask, bool):
            raise ValueError("mask must be a boolean")
        if not isinstance(self.allow_empty_codes, bool):
            raise ValueError("allow_empty_codes must be a boolean")
        for name in ("relation_weight", "ot_weight", "contrastive_weight", "context_weight", "context_noise_std"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("mixture_weight", "clean_probability", "erase_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("teacher_temperature", "student_temperature", "eta", "tol", "contrastive_temperature", "dictionary_temperature",
                     "context_teacher_temperature", "context_student_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.mass) or not 0 < self.mass < 1:
            raise ValueError("The partial-transport solver requires 0 < mass < 1")
        if self.alpha is not None:
            raise ValueError("alpha belongs to the old residual-repair architecture; landmark synthesis has no alpha")
        if not math.isfinite(self.dictionary_blend) or not 0 < self.dictionary_blend < 1:
            raise ValueError("dictionary_blend must be strictly between 0 and 1")
        if self.auxiliary_mode not in ("baseline", "wide"):
            raise ValueError("auxiliary_mode must be baseline or wide; MMRSAE anneals training K")
        if self.variant in ("mmrsae-annealed", "dlmmsaev2") and self.auxiliary_mode != "baseline":
            raise ValueError("Annealed dictionary SAE uses auxiliary budget 4 * training K")

    @property
    def is_dlmmsae(self) -> bool:
        return self.variant in ("dlmmsae", "dlmmsaev2")

    @property
    def has_dictionary(self) -> bool:
        return self.variant not in BASELINES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def training_k(self, step, total_steps):
        if self.variant in ("csrv2", "mmrsae-annealed", "dlmmsaev2"):
            return cosine_annealed_k(step, total_steps, self.anneal_start_k,
                                     self.k, self.anneal_fraction)
        return self.k


class MMRSAE(nn.Module):
    """One reusable MMRSAE encoder, objective and checkpoint state.

    MMRSAE first synthesizes a dense embedding from fixed training landmarks,
    then applies the SAE and its sole output Top-K gate. Baselines use the plain
    SAE. There is no post-gate repair. The synthesis dictionary is part of
    inference; the separate KL bank and transport are training-only.

    ``sae`` and ``repair`` remain accessible to the shared GPU benchmark and
    landmark-bank implementation. Save this module's complete ``state_dict``:
    it includes activity counters and the immutable synthesis dictionary.
    """

    def __init__(self, config: MMRSAEConfig):
        super().__init__()
        self.config = config
        # ResearchSAE initializes on CPU. Preserve the caller's CPU RNG and do
        # not seed or initialize CUDA generators as a constructor side effect.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            if config.has_dictionary:
                self.sae = LandmarkConditionedSAE(config.input_dim, config.hidden_dim,
                    config.dead_steps, rank=config.rank, dictionary_temperature=config.dictionary_temperature,
                    dictionary_blend=config.dictionary_blend, seed=config.seed)
            else:
                self.sae = ResearchSAE(config.input_dim, config.hidden_dim, config.dead_steps)
        # Shared retrieval still accepts this slot for the historical repair study.
        self.repair = None

    @property
    def needs_dictionary(self) -> bool:
        return self.config.has_dictionary

    @property
    def needs_bank(self) -> bool:
        return bool(self.config.relation_weight)

    @property
    def needs_context(self) -> bool:
        return bool(self.config.context_weight)

    def _require_ready(self):
        if self.needs_dictionary and not self.sae.dictionary.initialized:
            raise RuntimeError("Call initialize_dictionary(training_anchors) after moving MMRSAE "
                               "to its training device, or load an initialized checkpoint")

    @torch.no_grad()
    def initialize_dictionary(self, training_features: torch.Tensor, *, anchor_ids=None):
        """Persist fixed training-only landmarks, independent of the KL bank."""
        if not self.needs_dictionary:
            return {"source": "not_applicable_without_landmark_synthesis"}
        self.sae.dictionary.initialize(training_features, anchor_ids=anchor_ids)
        return {"source": "fixed_training_landmarks", "anchor_count": len(training_features),
                "dictionary_temperature": self.config.dictionary_temperature,
                "initial_blend": self.config.dictionary_blend}

    def encode(self, x: torch.Tensor, k: int | None = None, *, compact: bool = False):
        """Clean dictionary-first code without activity or RNG side effects.

        The normal result is the unnormalized [B,h] code. ``compact=True``
        returns (int32 coordinates, float32 L2-normalized values), each [B,K],
        directly suitable for CUDA retrieval. Nonfinite rows fail; all-zero
        rows stay zero when ``allow_empty_codes`` is set, otherwise they fail.
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
            indices, values = compact_codes(z, k, allow_empty=self.config.allow_empty_codes)
            return indices.to(torch.int32).contiguous(), values.contiguous()
        return z

    def forward(self, x: torch.Tensor, k: int | None = None):
        return self.encode(x, k)

    def loss(self, x: torch.Tensor, *, bank: LandmarkBank | None = None,
             row_ids=None, mask_generator: torch.Generator | None = None,
             context_bank=None, context_generator: torch.Generator | None = None,
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
        if self.needs_context and (context_bank is None or row_ids is None):
            raise ValueError("DLMMSAE requires a context reference bank and training row IDs")
        if row_ids is not None and len(row_ids) != len(x):
            raise ValueError("Training row IDs must align with feature rows")
        cfg = self.config
        k = cfg.training_k(step, total_steps)
        budget = auxiliary_budget(k, cfg.hidden_dim, step, total_steps, cfg.auxiliary_mode)
        if cfg.variant in ("csr", "csrv2"):
            result = self._csr_loss(x, budget, k=k, diagnostics=diagnostics, update_activity=update_activity)
            result[1]["training_k"] = k
            return result
        result = research_loss(
            self.sae, x, k, repair=self.repair, mask=cfg.mask,
            mask_generator=mask_generator, auxiliary_budget=budget, bank=bank, row_ids=row_ids,
            relation_weight=cfg.relation_weight, ot_weight=cfg.ot_weight,
            mixture_weight=cfg.mixture_weight, teacher_temperature=cfg.teacher_temperature,
            student_temperature=cfg.student_temperature, mass=cfg.mass, eta=cfg.eta,
            max_iters=cfg.max_iters, max_refine_iters=cfg.max_refine_iters, tol=cfg.tol,
            microbatch=cfg.microbatch, require_feasible=True, diagnostics=diagnostics,
            update_activity=update_activity, k_aux=cfg.k_aux,
            clean_probability=cfg.clean_probability, erase_probability=cfg.erase_probability,
            anchor_chunk_size=cfg.anchor_chunk_size, allow_empty_codes=cfg.allow_empty_codes,
        )
        result[1]["training_k"] = k
        if self.needs_dictionary:
            result[1]["dictionary_blend"] = self.sae.dictionary.blend_logit.sigmoid().detach()
        if self.needs_context:
            from dlmmsae import context_distillation_loss
            context, context_stats = context_distillation_loss(self.sae.dictionary, x,
                context_bank, row_ids=row_ids, generator=context_generator,
                noise_std=cfg.context_noise_std, teacher_temperature=cfg.context_teacher_temperature,
                student_temperature=cfg.context_student_temperature)
            loss, stats, views = result
            weighted = cfg.context_weight * context
            total = loss + weighted
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Nonfinite DLMMSAE objective")
            stats.update(context_stats)
            stats.update(loss_total=total, loss_context_unweighted=context, weighted_context=weighted)
            views["component_losses"]["context"] = weighted
            if diagnostics:
                # Context acts upstream of sparse views: inspect its actual
                # adapter gradients, not a misleading zero w.r.t. Top-K codes.
                grads = torch.autograd.grad(weighted, (self.sae.dictionary.U, self.sae.dictionary.V),
                                            retain_graph=True, allow_unused=True)
                stats["gradient_context_adapter_norm"] = sum(g.square().sum() for g in grads if g is not None).sqrt()
            return total, stats, views
        return result

    def _csr_loss(self, x, budget, *, k, diagnostics, update_activity):
        """CSR: shared reconstruction terms plus gamma * NCL, without transport.

        Activity follows the released CSR order (K, 4K, then K again for the
        duplicate NCL view); the dead mask is read after the first K update.
        """
        cfg = self.config
        reconstruction, stats, views = reconstruction_terms(
            self.sae, x, k, budget, update_activity=False, k_aux=cfg.k_aux)
        z = views["final_k"]
        if not bool(torch.isfinite(z).all()) or (
                not cfg.allow_empty_codes and bool(z.gt(0).sum(1).eq(0).any())):
            raise ValueError("invalid final code in final_k")
        if update_activity:
            for view in ("raw_k", "raw_4k", "raw_k"):
                self.sae.update_activity(views[view])
        contrastive = nonnegative_contrastive_loss(z, cfg.contrastive_temperature)
        total = reconstruction + cfg.contrastive_weight * contrastive
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("nonfinite CSR objective")
        components = {"reconstruction": reconstruction,
                      "contrastive": cfg.contrastive_weight * contrastive}
        stats.update({"loss_total": total, "loss_contrastive_unweighted": contrastive,
                      "weighted_contrastive": cfg.contrastive_weight * contrastive})
        views["component_losses"] = components
        views["profiles"] = None
        if diagnostics:
            stats.update(gradient_diagnostics(components, views))
            stats["active_unit_coverage"] = z.gt(0).any(0).float().mean()
        return total, stats, views
