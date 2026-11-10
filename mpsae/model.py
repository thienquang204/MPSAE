"""Tied Top-K sparse autoencoder with the released CSR semantics.

The parameterisation, initialisation, gating and dead-latent bookkeeping follow
``CSR/Vision/model_zoo.py`` (and the identical ``CSRv2/image/model_zoo.py``).
Every method in this project shares this module, so a difference in results can
only come from the objective, never from the architecture.

Two deliberate restructurings, both value- and gradient-identical to the
release, are marked in the code below: pre-activations are computed once per
step instead of three times, and the auxiliary dead-latent mask is applied with
a straight-through expression instead of an in-place ``.data`` mutation. The
activity counter is *not* restructured: callers drive it explicitly, because the
number and order of counter updates per step is part of the algorithm.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

#: A latent counts as active when its selected pre-activation exceeds this.
#: Released value, used verbatim (``model_zoo.py``: ``topk.values > 1e-5``).
ACTIVE_EPSILON = 1e-5


class SparseAutoencoder(nn.Module):
    """``z = ReLU(TopK(W(x - b_pre) + b_lat))``, ``x̂ = Wᵀz + b_pre``.

    Parameters are named after the release so checkpoints stay recognisable:
    ``pre_bias``, ``encoder.weight``, ``latent_bias``. The decoder is the
    transpose of the encoder (tied) and is never renormalised.
    """

    def __init__(self, input_dim: int, hidden_dim: int, *, dead_threshold: int = 30) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 4:
            raise ValueError("input_dim and hidden_dim must be positive, hidden_dim >= 4")
        if hidden_dim > torch.iinfo(torch.int32).max:
            raise ValueError("hidden_dim exceeds the int32 coordinate range used by sparse retrieval")
        self.input_dim, self.hidden_dim = int(input_dim), int(hidden_dim)
        self.dead_threshold = int(dead_threshold)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        # bias=False plus a separate latent_bias, exactly as released; the
        # default nn.Linear init (kaiming_uniform, a=sqrt(5)) is part of it.
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer("stats_last_nonzero", torch.zeros(hidden_dim, dtype=torch.long))

    # ------------------------------------------------------------ geometry

    @property
    def dictionary(self) -> torch.Tensor:
        """``[hidden_dim, input_dim]`` decoder rows (the tied encoder weight)."""
        return self.encoder.weight

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
                f"dead_threshold={self.dead_threshold}")

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected [batch, {self.input_dim}] features, got {tuple(x.shape)}")
        return F.linear(x - self.pre_bias, self.encoder.weight, self.latent_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.encoder.weight.t(), None) + self.pre_bias

    # ------------------------------------------------------------- gating

    def gate(self, pre: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense ``ReLU(TopK(pre))`` plus the selected indices.

        Ties are resolved by ``torch.topk``, i.e. the released behaviour; no
        deterministic tie rule is imposed, because imposing one would change
        the baselines.
        """
        dense, indices, _ = self.gate_with_selection(pre, k)
        return dense, indices

    def gate_with_selection(self, pre: torch.Tensor, k: int
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate once and retain its indices/values for activity bookkeeping.

        Each requested K still runs its own TopK: taking prefixes of a wider
        selection could change ties. Reusing this exact selection avoids a
        second TopK solely to update the dead-latent counter.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"K must be a positive integer, got {k!r}")
        if k > pre.shape[-1]:
            raise ValueError(f"K={k} exceeds hidden_dim={pre.shape[-1]}")
        selected = torch.topk(pre, k=k, dim=-1)
        dense = torch.zeros_like(pre).scatter_(-1, selected.indices, selected.values)
        return F.relu(dense), selected.indices, selected.values

    @torch.no_grad()
    def count_activity(self, pre: torch.Tensor, k: int) -> None:
        """Advance the dead-latent counter for one Top-K selection.

        Released arithmetic: latents whose selected value exceeds 1e-5 are
        reset to zero, then every counter is incremented. Because the training
        step gates three times (K, 4K, and the duplicate contrastive view), the
        counter advances three times per optimiser step and ``dead_threshold=30``
        therefore corresponds to roughly ten steps. That is the release's
        behaviour and callers must reproduce the same number of calls.
        """
        selected = torch.topk(pre.detach(), k=k, dim=-1)
        self.count_selected_activity(selected.indices, selected.values)

    @torch.no_grad()
    def count_selected_activity(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        """Apply one released counter update from an already computed gate."""
        hit = torch.zeros_like(self.stats_last_nonzero)
        hit.scatter_add_(0, indices.reshape(-1),
                         values.gt(ACTIVE_EPSILON).to(hit.dtype).reshape(-1))
        self.stats_last_nonzero *= 1 - hit.clamp(max=1)
        self.stats_last_nonzero += 1

    def dead_mask(self) -> torch.Tensor:
        """Latents that have not fired for more than ``dead_threshold`` gates."""
        return self.stats_last_nonzero > self.dead_threshold

    def auxiliary_latents(self, pre: torch.Tensor, k_aux: int) -> torch.Tensor:
        """Top-``k_aux`` over dead latents only, for the AuxK revival term.

        The release writes ``x.data *= dead_mask``, an untracked in-place write
        on the pre-activations. We cannot do that: the release recomputes the
        pre-activations for each of its three forward passes, whereas we share
        one tensor across all the views, and mutating it would corrupt them.

        Multiplying instead is equivalent, including in the gradient. A masked
        position holds exactly zero, so ReLU's derivative there is zero and no
        gradient would reach it either way -- which also covers the first ten
        or so steps, when nothing has yet crossed the threshold and the mask is
        entirely zero. ``tests/test_against_release.py`` pins both regimes
        against the released module.
        """
        if k_aux < 1:
            raise ValueError("k_aux must be positive; disable the term by weighting it zero")
        masked = pre * self.dead_mask().to(pre.dtype)
        latents, _ = self.gate(masked, min(k_aux, pre.shape[-1]))
        return latents

    # ------------------------------------------------------------ serving

    @torch.no_grad()
    def encode(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Inference code. No counter updates, no autograd, no side effects."""
        return self.gate(self.pre_activations(x), k)[0]

    @torch.no_grad()
    def encode_compact(self, x: torch.Tensor, k: int, *, normalize: bool = True
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(int32 indices, float32 values)`` of shape ``[batch, k]``.

        Values are L2-normalised over the full code by default, which is what
        exact cosine retrieval consumes. Rows whose code is entirely zero stay
        zero rather than producing NaNs; the retrieval layer counts them and
        scores them as misses, matching how an empty code behaves in FAISS.
        """
        dense, _ = self.gate(self.pre_activations(x), k)
        values, indices = dense.topk(k, dim=-1, sorted=False)
        values = values.to(torch.float32)
        if normalize:
            norm = values.norm(dim=-1, keepdim=True)
            values = torch.where(norm > 0, values / norm.clamp_min(torch.finfo(torch.float32).tiny),
                                 torch.zeros_like(values))
        return indices.to(torch.int32).contiguous(), values.contiguous()

    # --------------------------------------------------------- diagnostics

    @torch.no_grad()
    def sparsity_tensors(self, z: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
        """Device-side diagnostics, allowing one packed transfer with losses."""
        nnz = z.gt(0).sum(1)
        return {"nnz_mean": nnz.float().mean(),
                "nnz_min": nnz.min(),
                "below_k_fraction": nnz.lt(k).float().mean(),
                "empty_fraction": nnz.eq(0).float().mean(),
                "dead_fraction": self.dead_mask().float().mean(),
                "active_units": z.gt(0).any(0).sum()}

    @torch.no_grad()
    def sparsity_report(self, z: torch.Tensor, k: int) -> dict[str, float]:
        tensors = self.sparsity_tensors(z, k)
        values = torch.stack([value.float() for value in tensors.values()]).cpu().tolist()
        result = dict(zip(tensors, values))
        for name in ("nnz_min", "active_units"):
            result[name] = int(result[name])
        return result


def build_sae(input_dim: int, hidden_dim: int, *, seed: int,
              dead_threshold: int = 30, device=None) -> SparseAutoencoder:
    """Construct an autoencoder with a reproducible, CPU-seeded initialisation.

    The release seeds the global generator before building the model. Forking
    the RNG here keeps that determinism without letting construction perturb
    the data-order or masking streams of the caller.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = SparseAutoencoder(input_dim, hidden_dim, dead_threshold=dead_threshold)
    if device is not None:
        model = model.to(device)
    return model


def parameter_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def spectral_norm_estimate(matrix: torch.Tensor, *, iterations: int = 40, seed: int = 1337) -> float:
    """Power-iteration estimate of ``||matrix||_2``, for reporting only."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = F.normalize(torch.randn(matrix.shape[1], generator=generator).to(matrix), dim=0)
    for _ in range(iterations):
        vector = F.normalize(matrix.t() @ (matrix @ vector), dim=0)
    value = (matrix @ vector).norm().item()
    if not math.isfinite(value):
        raise ValueError("degenerate dictionary: spectral norm estimate is not finite")
    return value
