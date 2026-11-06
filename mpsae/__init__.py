"""MPSAE: adaptive sparse embeddings compared against CSR and CSRv2.

Layout
------
``config``    backbones, paper presets, the experiment grid
``io``        atomic JSON, digests, locks, memory-mapped NPZ archives
``logs``      console/file logging, JSONL metric streams, stage receipts
``model``     tied Top-K sparse autoencoder (released CSR semantics)
``losses``    reconstruction, AuxK, the two released contrastive objectives
``schedules`` CSRv2 linear k-annealing
``ot``        entropic multi-marginal partial-OT matching gap (ours)
``methods``   one objective per method: csr, csrv2, mpsae, mpsaev2, csr_multitopk
``train``     a single resumable fit
``runs``      the run matrix, resume and skip logic
``data``      ImageNet acquisition and FFCV serialisation
``embed``     frozen-backbone embedding extraction (MRL and CSRv2 recipes)
``evaluate``  encoding, exact 1-NN, linear probes, retrieval timing
``report``    tables and figures for experiments E1 and E2

Settings follow the published papers; algorithms follow the released code.
Deviations are recorded in the artefact manifests, never silently applied.
"""
from __future__ import annotations

__version__ = "2.0.0"

from . import config, io, logs  # noqa: F401  (cheap, torch-free)

__all__ = ["config", "io", "logs", "__version__"]
