"""Frozen embeddings: one module per published recipe, one shared output layer.

``extract_ff2048`` reproduces the MRL retrieval recipe, ``extract_timm`` the
CSRv2 FFCV recipe; both end in ``finalize_archive``, so every archive carries a
manifest naming its weights, its preprocessing and its deviations. torch, timm
and ffcv are imported inside the functions.
"""
from __future__ import annotations

from .archive import (FeatureSink, LABEL_KEY_DEVIATION, PRECISION, archive_manifest_path,
                      finalize_archive, library_versions, load_current_archive, validate_features)
from .csrv2_timm import extract_timm
from .mrl_ff2048 import apply_blurpool, extract_ff2048

__all__ = ["FeatureSink", "LABEL_KEY_DEVIATION", "PRECISION", "apply_blurpool",
           "archive_manifest_path", "extract_ff2048", "extract_timm", "finalize_archive",
           "library_versions", "load_current_archive", "validate_features"]
