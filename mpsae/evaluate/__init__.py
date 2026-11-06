"""Evaluation: encoding, exact 1-NN retrieval, linear probes, retrieval timing.

The three measurements the plan reports, and where each is defined:

``knn``      exact 1-NN Top-1, the primary metric (plan 4.3, CSR Table 4). No
             FAISS: blocked FP32 GEMM over L2-normalised codes, which is
             equivalent to ``IndexFlatL2`` rather than an approximation of it.
``probe``    linear probe on the full hidden-dimensional latent (CSR issue #18).
``timing``   retrieval cost under the papers' efficiency protocol (CSR App. E.3,
             CSRv2 App. G.1). Secondary; its CUDA extension is optional.

``encode`` owns the precision contract that the other three inherit: encode
under autocast float16, store float32, search float32 with TF32 disabled.

Importing this package does not import torch; the submodules are resolved on
first attribute access, so the planner and the report can import ``mpsae`` on a
machine without CUDA.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CodeSource": "encode",
    "encode_split": "encode",
    "encode_dense": "encode",
    "exact_top1": "knn",
    "top1_accuracy": "knn",
    "dense_top1": "knn",
    "verify_against_bruteforce": "knn",
    "fp32_matmul": "knn",
    "require_vram": "knn",
    "DENSE_GEMM_GALLERY_CHUNK": "knn",
    "DENSE_GEMM_QUERY_BLOCK": "knn",
    "linear_probe": "probe",
    "build_csr_gallery": "timing",
    "time_sparse_matmul": "timing",
    "time_dense_gemm": "timing",
    "speedup": "timing",
    "maybe_build_inverted_index": "timing",
}

__all__ = [*sorted(_EXPORTS), "encode", "knn", "probe", "timing"]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    if name in ("encode", "knn", "probe", "timing"):
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
