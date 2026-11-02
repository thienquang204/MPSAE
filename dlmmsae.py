"""Dictionary-level relational distillation for DLMMSAE.

A fixed synthesis bank A constructs context. A separate fixed training bank Q
tests its geometry against the clean input, before residual blending and Top-K.
These immutable backbone features need neither an EMA encoder nor a FIFO queue.
Q is training-only and checkpointed separately from the serving model.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


class ContextReferenceBank:
    """Frozen, ordered training references, disjoint from synthesis anchors."""

    def __init__(self, reference_ids, features, *, anchor_ids=None):
        if (features.ndim != 2 or len(features) < 2 or features.shape[1] < 1
                or features.dtype != torch.float32):
            raise ValueError("Context references require at least two float32 feature rows")
        if not bool(torch.isfinite(features).all()) or bool(features.norm(dim=1).eq(0).any()):
            raise ValueError("Context reference features must be finite and nonzero")
        ids = torch.as_tensor(reference_ids, dtype=torch.long, device=features.device)
        if ids.shape != (len(features),) or len(ids.unique()) != len(ids):
            raise ValueError("Context reference IDs must be distinct and align with features")
        if anchor_ids is not None and bool(torch.isin(ids, torch.as_tensor(
                anchor_ids, dtype=torch.long, device=ids.device)).any()):
            raise ValueError("Context references must be disjoint from synthesis anchors")
        self.ids = ids.detach().clone()
        self.features = features.detach().clone()
        self.normalized_keys = F.normalize(self.features, dim=1)

    def state_dict(self):
        return {"version": 1, "ids": self.ids.detach().cpu().clone(),
                "features": self.features.detach().cpu().clone()}

    def load_state_dict(self, state):
        # Reconstructed from the pinned data identity before restore. Refuse
        # a changed order or source rather than silently changing KL targets.
        if (state.get("version") != 1 or not isinstance(state.get("ids"), torch.Tensor)
                or not isinstance(state.get("features"), torch.Tensor)
                or not torch.equal(self.ids.cpu(), state.get("ids"))
                or not torch.equal(self.features.cpu(), state.get("features"))):
            raise ValueError("Context checkpoint references differ from the fixed training bank")


def context_distillation_loss(dictionary, x, bank, *, row_ids, generator=None,
                              noise_std=.02, teacher_temperature=.1, student_temperature=.1):
    """KL(clean-input profile || raw-context profile) on identical Q support.

    Noise is Gaussian feature perturbation scaled by each row's RMS, applied
    only to the student's lookup query. It is not image augmentation. Neither
    the clean teacher nor the immutable keys receive gradients. A private CPU
    generator makes this auxiliary view reproducible independently of masking.
    """
    for name, value in (("teacher_temperature", teacher_temperature),
                        ("student_temperature", student_temperature)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    if (x.ndim != 2 or not len(x) or x.dtype != torch.float32
            or x.shape[1] != bank.features.shape[1] or x.device != bank.features.device):
        raise ValueError("Context inputs and references must have matching float32 shape/device")
    if not bool(torch.isfinite(x).all()) or bool(x.norm(dim=1).eq(0).any()):
        raise ValueError("Context inputs must be finite and nonzero")
    if row_ids is None:
        raise ValueError("Context distillation requires training row IDs")
    ids = torch.as_tensor(row_ids, dtype=torch.long, device=x.device)
    if ids.shape != (len(x),):
        raise ValueError("Context row IDs must align with inputs")
    if bool(torch.isin(bank.ids, dictionary.anchor_ids).any()):
        raise ValueError("Context references must be disjoint from synthesis anchors")
    valid = ids[:, None].ne(bank.ids[None, :])
    if bool(valid.sum(dim=1).lt(2).any()):
        raise ValueError("Context distillation needs at least two eligible references after self exclusion")
    clean = x.detach()
    with torch.no_grad():
        teacher_logits = (F.normalize(clean, dim=1) @ bank.normalized_keys.T) / teacher_temperature
        teacher_log = teacher_logits.masked_fill(~valid, -torch.inf).log_softmax(dim=1)
        target = teacher_log.exp()
        teacher_log = teacher_log.masked_fill(~valid, 0.)
    student_input = clean
    if noise_std:
        if generator is None or generator.device.type != "cpu":
            raise ValueError("Context perturbation requires a dedicated CPU generator")
        noise = torch.randn(clean.shape, dtype=torch.float32, device="cpu", generator=generator).to(x.device)
        student_input = clean + noise_std * clean.square().mean(dim=1, keepdim=True).sqrt() * noise
    context = dictionary.synthesize(student_input, row_ids=ids, exclude_self=True)
    student_logits = (F.normalize(context, dim=1) @ bank.normalized_keys.T) / student_temperature
    student_log = student_logits.masked_fill(~valid, -torch.inf).log_softmax(dim=1)
    student_log = student_log.masked_fill(~valid, 0.)
    loss = (target * (teacher_log - student_log)).sum(dim=1).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite dictionary context objective")
    stats = {"context_reference_count": len(bank.ids),
             "context_valid_references_min": valid.sum(dim=1).min().detach(),
             "context_self_excluded_rows": (~valid).any(dim=1).sum().detach(),
             "context_teacher_entropy": -(target * teacher_log).sum(dim=1).mean().detach(),
             "context_cosine_to_input": F.cosine_similarity(context, clean, dim=1).mean().detach(),
             "context_norm_mean": context.norm(dim=1).mean().detach(),
             "context_noise_std": noise_std}
    return loss, stats
