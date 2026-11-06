"""Verbatim transcription of the released CSR / CSRv2 training code.

Source, read at these commits:
  CSR    github.com/neilwen987/CSR_Adaptive_Rep @ b284898
         Vision/model_zoo.py, Vision/main_visual.py
  CSRv2  github.com/Y-Research-SBU/CSRv2 @ 3b2d4d2
         image/model_zoo.py, image/main_visual.py

This file exists **only** as a test oracle. Nothing in ``mpsae`` imports it.
It is kept deliberately close to the original -- including the in-place
``x.data *= dead_mask`` and the raw ``exp`` contrastive losses -- so that
``tests/test_against_release.py`` compares our implementation against what the
authors actually published, rather than against our reading of it.

Distributed helpers are reduced to the identity, which is what the released
single-process path does (``gather`` returns its argument when world size is
one, and ``get_rank()`` is zero).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()


class CSR(nn.Module):
    """``CSR`` from ``model_zoo.py``, minus the timm backbone plumbing."""

    def __init__(self, n_latents: int, topk: int, auxk: int, normalize: bool,
                 n_inputs: int, dead_threshold: int,
                 pre_trained_backbone: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.pre_trained_backbone = pre_trained_backbone
        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.decoder: TiedTranspose = TiedTranspose(self.encoder)
        self.topk = topk
        self.auxk = auxk
        self.normalize = normalize
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return F.linear(x, self.encoder.weight, self.latent_bias)

    def preprocess(self, x: torch.Tensor):
        if not self.normalize:
            return x, dict()
        raise NotImplementedError("the released image pipeline always sets normalize=False")

    def top_k(self, x: torch.Tensor, topk=None):
        if topk is None:
            topk = self.topk
        topk = torch.topk(x, k=topk, dim=-1)
        z_topk = torch.zeros_like(x)
        z_topk.scatter_(-1, topk.indices, topk.values)
        latents_k = F.relu(z_topk)
        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(
            0,
            topk.indices.reshape(-1),
            (topk.values > 1e-5).to(tmp.dtype).reshape(-1),
        )
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1
        latents_auxk = None
        if self.auxk:
            aux_topk = torch.topk(input=self.auxk_mask_fn(x), k=self.auxk)
            z_auxk = torch.zeros_like(x)
            z_auxk.scatter_(-1, aux_topk.indices, aux_topk.values)
            latents_auxk = F.relu(z_auxk)
        return latents_k, latents_auxk

    def decode(self, latents: torch.Tensor, info=None) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

    def csr_forward(self, x: torch.Tensor, topk):
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents_k, latents_auxk = self.top_k(latents_pre_act, topk)
        recons = self.decode(latents_k, info)
        recons_aux = self.decode(latents_auxk, info)
        return latents_pre_act, latents_k, recons, recons_aux

    def forward(self, x: torch.Tensor, topk=None):
        latents_pre_act, latents_k, recons, recons_aux = self.csr_forward(x, topk=topk)
        return x, latents_pre_act, latents_k, recons, recons_aux


def normalized_mse(recon: torch.Tensor, xs: torch.Tensor, criterion) -> torch.Tensor:
    xs_mu = xs.mean(dim=0)
    return criterion(recon, xs) / criterion(xs_mu[None, :].broadcast_to(xs.shape), xs)


def inbatch_cl(latents_k_1: torch.Tensor, latents_k_2: torch.Tensor) -> torch.Tensor:
    n_repeats = 2
    temperature = 0.2
    indexes = torch.arange(latents_k_1.shape[0], device=latents_k_1.device)
    indexes = indexes.repeat(n_repeats)

    z = torch.concatenate((latents_k_1, latents_k_2), dim=0)
    z = F.normalize(z, dim=-1)
    gathered_z = z

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)
    gathered_indexes = indexes

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, 0:].fill_diagonal_(0)
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    return -(torch.mean(torch.log(pos / (pos + neg))))


def label_based_cl(latents_k_1: torch.Tensor, latents_k_2: torch.Tensor,
                   labels: torch.Tensor) -> torch.Tensor:
    temperature = 0.2
    z = torch.concatenate((latents_k_1, latents_k_2), dim=0)
    z = F.normalize(z, dim=-1)
    gathered_z = z

    labels_concat = torch.concatenate((labels, labels), dim=0)
    gathered_labels = labels_concat

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    labels_concat = labels_concat.unsqueeze(1)
    gathered_labels = gathered_labels.unsqueeze(0)

    pos_mask = (labels_concat == gathered_labels).float()
    eye_mask = torch.eye(pos_mask.size(0), pos_mask.size(1), device=pos_mask.device)
    pos_mask = pos_mask - eye_mask
    neg_mask = (labels_concat != gathered_labels).float()

    pos = torch.sum(sim * pos_mask, dim=1)
    neg = torch.sum(sim * neg_mask, dim=1)
    pos = torch.clamp(pos, min=1e-8)
    neg = torch.clamp(neg, min=1e-8)
    return -(torch.mean(torch.log(pos / (pos + neg))))


def get_current_k(global_step: int, initial_k: int, final_k: int, total_steps: int,
                  k_decay_ratio: float = 0.7) -> int:
    k_decay_steps = int(total_steps * k_decay_ratio)
    if global_step >= k_decay_steps:
        return final_k
    progress = global_step / k_decay_steps
    return round(initial_k * (1 - progress) + final_k * progress)


def released_step(model: CSR, images: torch.Tensor, *, topk: int, auxk_coef: float,
                  cl_coef: float, use_label_cl: bool = False,
                  target: Optional[torch.Tensor] = None):
    """One optimiser step's objective, in the released order of operations.

    Three forward passes: K, then 4K, then K again purely to obtain the
    duplicate contrastive view. Each pass calls ``top_k`` and therefore
    advances ``stats_last_nonzero``, which is why the counter moves three times
    per step in the release.
    """
    criterion = nn.MSELoss()
    x_1, _, latents_k_1, recons_1, recons_aux_1 = model(images, topk=topk)
    x_3, _, _, recons_3, _ = model(images, topk=4 * topk)

    loss_k = criterion(x_1, recons_1)
    loss_4k = criterion(x_3, recons_3)
    loss_auxk = normalized_mse(
        recons_aux_1, x_1 - recons_1.detach() + model.pre_bias.detach(), criterion).nan_to_num(0)

    _, _, latents_k_2, _, _ = model(images, topk=topk)
    if use_label_cl:
        loss_cl = label_based_cl(latents_k_1, latents_k_2, target)
    else:
        loss_cl = inbatch_cl(latents_k_1, latents_k_2)

    loss = loss_k + 0.125 * loss_4k + auxk_coef * loss_auxk + cl_coef * loss_cl
    return loss, {"loss_k": loss_k, "loss_4k": loss_4k, "loss_auxk": loss_auxk,
                  "loss_cl": loss_cl, "latents_k": latents_k_1}
