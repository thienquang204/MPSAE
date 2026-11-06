"""Shared fixtures.

The grid, schedule and reporting logic is deliberately torch-free so it can be
checked on a laptop; everything numerical needs the GPU box. Modules that need
torch call ``pytest.importorskip("torch")`` *before* importing anything from
``mpsae``, because a skip mark would still fail during collection.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, Path(__file__).resolve().parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


@pytest.fixture(scope="session")
def torch():
    pytest.importorskip("torch")
    import torch as module

    module.backends.cuda.matmul.allow_tf32 = False
    module.backends.cudnn.allow_tf32 = False
    return module


@pytest.fixture
def small_preset():
    """A preset small enough to train on a CPU in a test."""
    from dataclasses import replace

    from mpsae.config import FF2048_PRESET

    return replace(FF2048_PRESET, hidden_dim=64, k_aux=8, batch_size=16, epochs=1,
                   ot_microbatch=4, ot_max_iters=30, ot_max_refine_iters=200)


@pytest.fixture
def batch(torch):
    generator = torch.Generator().manual_seed(7)
    return torch.randn(16, 12, generator=generator)


@pytest.fixture
def labels(torch):
    # Four classes, so csrv2's supervised positives are non-trivial.
    return torch.arange(16) % 4
