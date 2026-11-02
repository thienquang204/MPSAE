"""Training-landmark synthesis followed by the shared tied sparse autoencoder."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from mpsae_repair import ResearchSAE


class LandmarkDictionary(nn.Module):
    """Attention over fixed training embeddings, with a direct input path.

    This dictionary is part of inference. It is distinct from the periodically
    refreshed student-code bank used only for relational KL. Every lookup uses
    the same anchors, including a self anchor when present, in train and eval.
    """

    def __init__(self, input_dim, rank=32, temperature=.1, blend=.1, seed=1227):
        super().__init__()
        if input_dim < 1 or rank < 1:
            raise ValueError("dictionary dimensions must be positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("dictionary temperature must be positive")
        if not math.isfinite(blend) or not 0 < blend < 1:
            raise ValueError("dictionary blend must be strictly between 0 and 1")
        self.input_dim = input_dim
        generator = torch.Generator(device="cpu").manual_seed(seed)
        u = torch.empty(input_dim, rank)
        nn.init.kaiming_uniform_(u.T, a=math.sqrt(5), generator=generator)
        self.U = nn.Parameter(u)
        self.V = nn.Parameter(torch.zeros(rank, input_dim))
        self.blend_logit = nn.Parameter(torch.tensor(math.log(blend / (1 - blend))))
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("features", torch.empty(0, input_dim))
        self.register_buffer("anchor_ids", torch.empty(0, dtype=torch.long))
        self.register_buffer("architecture_version", torch.tensor(1, dtype=torch.int32))
        self.register_buffer("normalized_keys", torch.empty(0, input_dim), persistent=False)
        self.initialized = False

    @torch.no_grad()
    def initialize(self, features, anchor_ids=None):
        if self.initialized:
            raise RuntimeError("Landmark dictionary is fixed; initialize it only once")
        if features.ndim != 2 or features.shape[1] != self.input_dim or len(features) < 1:
            raise ValueError("Dictionary requires nonempty [anchors, input_dim] training features")
        if features.dtype != torch.float32 or features.device != self.U.device:
            raise ValueError("Dictionary features must be float32 on the model device")
        if not bool(torch.isfinite(features).all()) or bool(features.norm(dim=1).eq(0).any()):
            raise ValueError("Dictionary anchors must be finite and nonzero")
        ids = (torch.arange(len(features), device=features.device) if anchor_ids is None
               else torch.as_tensor(anchor_ids, device=features.device, dtype=torch.long))
        if ids.ndim != 1 or len(ids) != len(features) or len(ids.unique()) != len(ids):
            raise ValueError("Dictionary anchor IDs must be distinct and aligned with features")
        self.features = features.detach().clone()
        self.anchor_ids = ids.detach().clone()
        self.normalized_keys = F.normalize(self.features, dim=1)
        self.initialized = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + "architecture_version")
        features = state_dict.get(prefix + "features")
        ids = state_dict.get(prefix + "anchor_ids")
        valid = (version is not None and version.numel() == 1 and int(version.item()) == 1
                 and features is not None and features.ndim == 2
                 and features.shape[1] == self.input_dim and len(features) > 0
                 and ids is not None and ids.ndim == 1 and len(ids) == len(features)
                 and len(ids.unique()) == len(ids)
                 and bool(torch.isfinite(features).all())
                 and not bool(features.norm(dim=1).eq(0).any()))
        if not valid:
            error_msgs.append("Landmark dictionary architecture requires an initialized matching "
                              "checkpoint; old sparse-start repair checkpoints cannot be loaded")
        else:
            self.features = torch.empty(features.shape, device=self.U.device, dtype=self.U.dtype)
            self.anchor_ids = torch.empty(ids.shape, device=self.U.device, dtype=torch.long)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if valid:
            self.normalized_keys = F.normalize(self.features, dim=1)
            self.initialized = True

    def synthesize(self, x, *, row_ids=None, exclude_self=False):
        """Raw context before the residual blend or SAE gate.

        Auxiliary distillation may exclude a known training row's own anchor.
        Serving keeps the original policy; the experiment runner holds anchors
        out of fitting, gallery and query pools for every method.
        """
        if not self.initialized:
            raise RuntimeError("Initialize the landmark dictionary from training anchors before encoding")
        query = x + F.relu(x @ self.U) @ self.V
        logits = (F.normalize(query, dim=1) @ self.normalized_keys.T) / self.temperature
        if exclude_self:
            if row_ids is None:
                raise ValueError("Self-excluded synthesis requires training row IDs")
            ids = torch.as_tensor(row_ids, dtype=torch.long, device=x.device)
            if ids.shape != (len(x),):
                raise ValueError("Synthesis row IDs must align with inputs")
            valid = ids[:, None].ne(self.anchor_ids[None, :])
            if not bool(valid.any(dim=1).all()):
                raise ValueError("Self exclusion leaves no synthesis anchors")
            logits = logits.masked_fill(~valid, -torch.inf)
        return logits.softmax(dim=1) @ self.features

    def forward(self, x):
        context = self.synthesize(x)
        return x + self.blend_logit.sigmoid() * (context - x)


class LandmarkConditionedSAE(ResearchSAE):
    """Synthesize an embedding from landmarks before any SAE Top-K gate."""
    dictionary_first = True

    def __init__(self, input_dim, hidden_dim=8192, dead_steps=30, *, rank=32,
                 dictionary_temperature=.1, dictionary_blend=.1, seed=1227):
        super().__init__(input_dim, hidden_dim, dead_steps)
        self.dictionary = LandmarkDictionary(input_dim, rank, dictionary_temperature,
                                             dictionary_blend, seed)

    def preactivations(self, x):
        return super().preactivations(self.dictionary(x))
