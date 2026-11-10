"""Reconstruction, AuxK and the two released contrastive objectives.

Formulas follow ``CSR/Vision/main_visual.py`` and ``CSRv2/image/main_visual.py``.
Both contrastive losses are offered twice: a literal transcription of the
release and a log-sum-exp form that is algebraically identical but does not
build an ``exp`` matrix. The literal versions exist so the tests can prove the
stable ones agree, and so a reviewer can diff against the paper's code.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

#: Released temperature for both contrastive losses (hard-coded upstream).
CL_TEMPERATURE = 0.2


# ----------------------------------------------------------- reconstruction

def mse(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``nn.MSELoss()`` -- a mean over elements, not the summed squared norm.

    The papers write ``L(k) = ||x - x̂||²``; the released code uses the
    elementwise mean. The mean is what reproduces the published numbers, since
    every loss weight was tuned against it.
    """
    return F.mse_loss(reconstruction, target)


def normalized_mse(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """AuxK's variance-normalised error, divided by a batch-mean predictor.

    ``normalized_mse`` in the release, including the ``nan_to_num(0)`` that
    absorbs a constant target (zero denominator) in the first steps.
    """
    baseline = target.mean(dim=0, keepdim=True).broadcast_to(target.shape)
    return (mse(reconstruction, target) / mse(baseline, target)).nan_to_num(0.0)


def auxk_target(x: torch.Tensor, reconstruction: torch.Tensor,
                pre_bias: torch.Tensor) -> torch.Tensor:
    """``x - x̂.detach() + b_pre.detach()``: the residual AuxK must explain.

    ``b_pre`` is added back because the auxiliary reconstruction also adds it,
    so the term measures the dictionary's residual fit and not the bias.
    """
    return x - reconstruction.detach() + pre_bias.detach()


# ------------------------------------------------------------- contrastive

def _paired_similarity(z1: torch.Tensor, z2: torch.Tensor, temperature: float
                       ) -> torch.Tensor:
    """L2-normalised cosine logits over the stacked ``[2B, h]`` views."""
    if z1.shape != z2.shape or z1.ndim != 2 or not len(z1):
        raise ValueError("contrastive views must be equal, nonempty [batch, hidden] matrices")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    stacked = F.normalize(torch.cat((z1, z2), dim=0).float(), dim=-1)
    return stacked @ stacked.t() / temperature


def _masked_log_ratio(logits: torch.Tensor, positive: torch.Tensor,
                      negative: torch.Tensor) -> torch.Tensor:
    """``-mean log(sum_pos exp / (sum_pos exp + sum_neg exp))`` via log-sum-exp.

    Rows with no positive contribute ``0`` to the released loss through its
    ``clamp(min=1e-8)``; here they are dropped from the mean instead, which is
    the same limit without manufacturing a ``-log(1e-8/...)`` spike. With real
    labels every row has at least its duplicate copy as a positive, so this
    branch only guards degenerate batches.
    """
    neutral = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive, neutral)
    union_logits = logits.masked_fill(~(positive | negative), neutral)
    has_positive = positive.any(dim=1)
    if not bool(has_positive.any()):
        return logits.new_zeros(())
    ratio = (torch.logsumexp(positive_logits, dim=1)
             - torch.logsumexp(union_logits, dim=1))
    return -ratio[has_positive].mean()


def inbatch_cl(z1: torch.Tensor, z2: torch.Tensor,
               temperature: float = CL_TEMPERATURE) -> torch.Tensor:
    """CSR's non-negative contrastive loss (``inbatch_cl``).

    The positive of row ``i`` is the other copy of the same sample; negatives
    are every other sample, in both copies. The release feeds two identical
    deterministic forward passes, so this acts as a uniformity term on the
    L2-normalised sparse codes -- there is no augmentation anywhere in the
    released pipeline, despite what the CSRv2 paper says about CSR.
    """
    logits = _paired_similarity(z1, z2, temperature)
    batch = len(z1)
    identity = torch.arange(batch, device=logits.device).repeat(2)
    same = identity[:, None].eq(identity[None, :])
    positive = same.clone().fill_diagonal_(False)
    return _masked_log_ratio(logits, positive, ~same)


def label_based_cl(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor,
                   temperature: float = CL_TEMPERATURE) -> torch.Tensor:
    """CSRv2's supervised contrastive loss (``label_based_cl``).

    Positives are every same-label sample across both copies, excluding only
    the diagonal self-comparison; negatives are every different-label sample.

    Requires real labels. The released image and text loaders read
    ``train_data.get('labels')`` while every released producer writes
    ``label``, so with the published archives all labels collapse to ``0``,
    the loss collapses to ~0, and the advertised supervision never happens.
    Our embedding archives store ``labels``, so the term is live; the
    degenerate release behaviour is reproducible by passing a constant vector.
    """
    if labels.ndim != 1 or len(labels) != len(z1):
        raise ValueError("labels must be one integer per row of z1")
    logits = _paired_similarity(z1, z2, temperature)
    doubled = torch.cat((labels, labels), dim=0)
    same = doubled[:, None].eq(doubled[None, :])
    positive = same.clone().fill_diagonal_(False)
    return _masked_log_ratio(logits, positive, ~same)


def duplicate_view_cl(z: torch.Tensor, labels: torch.Tensor | None = None,
                      temperature: float = CL_TEMPERATURE) -> torch.Tensor:
    """The existing contrastive objective when both views are the same tensor.

    Every other sample occurs twice, while the anchor's own duplicate occurs
    once (only the self-comparison is excluded). Folding that multiplicity
    into off-diagonal logits uses B x B storage instead of 2B x 2B. Passing
    labels selects the CSRv2 positive set; otherwise only the duplicate is
    positive. This matches the stable losses above, including their zero-code
    behaviour, without a scalar host check for the guaranteed-positive rows.
    """
    if z.ndim != 2 or not len(z):
        raise ValueError("contrastive codes must be a nonempty [batch, hidden] matrix")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if labels is not None and (labels.ndim != 1 or len(labels) != len(z)):
        raise ValueError("labels must be one integer per row of z")
    normalized = F.normalize(z.float(), dim=-1)
    similarity = normalized @ normalized.t() / temperature
    # Never replace the diagonal by 1/temperature: zero and near-zero rows
    # need its actual value and gradient through normalization.
    diagonal = similarity.diagonal()
    logits = similarity + math.log(2.0)
    logits.diagonal().copy_(diagonal)
    denominator = torch.logsumexp(logits, dim=1)
    if labels is None:
        numerator = diagonal
    else:
        same = labels[:, None].eq(labels[None, :])
        numerator = torch.logsumexp(
            logits.masked_fill(~same, torch.finfo(logits.dtype).min), dim=1)
    return (denominator - numerator).mean()


# ------------------------------- literal transcriptions, used only by tests


def inbatch_cl_released(z1: torch.Tensor, z2: torch.Tensor,
                        temperature: float = CL_TEMPERATURE) -> torch.Tensor:
    """Literal ``inbatch_cl`` from the release (single process, gather = identity)."""
    indexes = torch.arange(z1.shape[0], device=z1.device).repeat(2)
    z = F.normalize(torch.cat((z1, z2), dim=0), dim=-1)
    sim = torch.exp(torch.einsum("if, jf -> ij", z, z) / temperature)
    positive = indexes[:, None].eq(indexes[None, :]).clone()
    positive.fill_diagonal_(False)
    negative = indexes[:, None].ne(indexes[None, :])
    pos = (sim * positive).sum(1)
    neg = (sim * negative).sum(1)
    return -(torch.log(pos / (pos + neg))).mean()


def label_based_cl_released(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor,
                            temperature: float = CL_TEMPERATURE) -> torch.Tensor:
    """Literal ``label_based_cl`` from the release, including its clamps."""
    z = F.normalize(torch.cat((z1, z2), dim=0), dim=-1)
    doubled = torch.cat((labels, labels), dim=0)
    sim = torch.exp(torch.einsum("if, jf -> ij", z, z) / temperature)
    positive = doubled[:, None].eq(doubled[None, :]).float()
    positive = positive - torch.eye(positive.shape[0], positive.shape[1], device=positive.device)
    negative = doubled[:, None].ne(doubled[None, :]).float()
    pos = (sim * positive).sum(1).clamp(min=1e-8)
    neg = (sim * negative).sum(1).clamp(min=1e-8)
    return -(torch.log(pos / (pos + neg))).mean()
