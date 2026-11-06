"""Exact 1-NN retrieval: the primary metric, reproduced without FAISS.

The released protocol (``CSRv2/image/faiss_nn.py``, ``compute_metrics.py``) is:
cast the stored float32 latents to contiguous float32, ``faiss.normalize_L2``
both sides, build an ``IndexFlatL2`` over the 1.28M train codes, search the 50k
validation codes, and report the share of queries whose *first* neighbour has
the query's label. Everything else in those scripts -- the 2048-long shortlist,
the CSV round trip, the mAP helper -- is unused by the reported Top-1.

We do not use FAISS. The ``faiss-gpu`` wheels trail new hardware (no sm_90 on
H100, no sm_120 on Blackwell), and the project must run on both. On
L2-normalised vectors

    ||q - g||^2 = 2 - 2 q.g,

so minimising L2 distance is *identical* to maximising the inner product, and a
blocked FP32 GEMM argmax returns the same neighbour as ``IndexFlatL2`` -- not an
approximation of it. ``faiss-cpu`` may be installed as a test oracle; it is
never a runtime dependency.

Precision: the search is float32 with TF32 **off**. TF32 keeps only 10 mantissa
bits in the multiplier, and over a 1.28M-row gallery of near-unit cosine scores
that reorders near-ties and changes the top-1 label for a measurable number of
queries. fp16/bf16 is worse still. The encoding upstream is fp16 because the
release encodes in fp16; the search is fp32 because the release searches in
fp32. Both halves of that contract matter.

Design ported from ``repair_benchmark.py:_dense_gemm_search`` (commit 89806f5),
including its chunk constants, which are tuned so that the score block, the
gallery chunk and the resident queries all stay well inside a 24 GiB card.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time
from typing import Any, Iterator

import numpy as np
import torch

from ..io import human_bytes
from ..logs import Progress, format_duration, get_logger
from .encode import CodeSource

#: Gallery rows densified per step and queries scored per step. Peak score
#: block is 8192 x 32768 float32 = 1.0 GiB; the densified chunk at h=8192 is
#: another 1.0 GiB. Both constants are the released-benchmark values.
DENSE_GEMM_GALLERY_CHUNK = 32768
DENSE_GEMM_QUERY_BLOCK = 8192

BACKEND = "blocked_fp32_gemm"

#: VRAM left free beyond the computed working set: cuBLAS workspace, the
#: allocator's fragmentation and whatever else shares the device.
VRAM_RESERVE_BYTES = 1 << 30


# --------------------------------------------------------------- precision

@contextmanager
def fp32_matmul() -> Iterator[None]:
    """Disable TF32 for the duration of a search, then restore it.

    Training sets its own policy; a context manager keeps the search's
    requirement from leaking into the caller's process state either way.
    """
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous[0]
        torch.backends.cudnn.allow_tf32 = previous[1]


def require_vram(device: torch.device, needed_bytes: int, *, what: str, knobs: str,
                 reserve_bytes: int = VRAM_RESERVE_BYTES) -> None:
    """Refuse to start a search that cannot finish, naming the knobs to lower."""
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(device)
    if free < needed_bytes + reserve_bytes:
        raise RuntimeError(
            f"{what} needs {human_bytes(needed_bytes)} plus {human_bytes(reserve_bytes)} "
            f"reserve on {device} ({human_bytes(free)} free of {human_bytes(total)}). "
            f"Lower {knobs}.")


# ------------------------------------------------------------------- search

def exact_top1(gallery: Any, queries: Any, *, dim: int, device: torch.device | str,
               gallery_chunk: int = DENSE_GEMM_GALLERY_CHUNK,
               query_block: int = DENSE_GEMM_QUERY_BLOCK,
               normalize: bool = False, shortlist: int = 1,
               progress_stage: str = "knn") -> tuple[torch.Tensor, torch.Tensor, float]:
    """Exact maximum-inner-product top-1 by blocked FP32 GEMM.

    ``gallery`` and ``queries`` are anything :class:`CodeSource` accepts: a
    dense float32 ``[n, dim]`` matrix or a compact ``(indices, values)`` pair.
    Both sides must already be L2-normalised, or ``normalize=True`` must be set
    so this routine does it while expanding rows.

    Queries stay dense and resident; the gallery is read once in row order and
    densified one chunk at a time on the GPU. The cost is therefore independent
    of K -- unlike posting-list accumulation, whose work grows with K^2 -- and
    identical for the sparse codes (expanded to ``dim = hidden_dim``) and the
    dense reference (``dim = d``).

    Ties keep the lowest gallery position: ``torch.max`` returns the first
    maximal column within a chunk, and chunks are merged with a strict ``>``.
    An all-zero query scores 0.0 everywhere and so lands on gallery row 0,
    which is what ``IndexFlatL2`` does with a zero query; callers count those
    rows separately and score them as misses.

    ``shortlist`` must be 1 today. The parameter exists because CSR Table 4
    reports only Top-1, while recall@k would need the same loop with ``topk``
    and a merge of partial lists; keeping the name reserved makes that a local
    change rather than a signature change.

    Returns ``(best_scores float32[n_query], positions int64[n_query], seconds)``
    on ``device``. ``seconds`` covers the densify-and-multiply loop between two
    device synchronisations, excluding the resident query build.
    """
    device = torch.device(device)
    if shortlist != 1:
        raise NotImplementedError("only exact top-1 is implemented; see the docstring")
    if gallery_chunk < 1 or query_block < 1:
        raise ValueError("gallery_chunk and query_block must be positive")
    gallery_source = CodeSource(gallery, dim=dim, device=device, normalize=normalize)
    query_source = CodeSource(queries, dim=dim, device=device, normalize=normalize)
    n_query, n_gallery = query_source.rows, gallery_source.rows
    if not n_query or not n_gallery:
        raise ValueError("both the gallery and the query set must be nonempty")
    gallery_chunk = min(gallery_chunk, n_gallery)
    query_block = min(query_block, n_query)
    require_vram(device,
                 n_query * dim * 4                    # resident dense queries
                 + gallery_chunk * dim * 4            # one densified gallery chunk
                 + query_block * gallery_chunk * 4    # one score block
                 + n_query * 12,                      # best scores and positions
                 what=f"exact top-1 over {n_gallery:,} x {n_query:,} at dim={dim}",
                 knobs="gallery_chunk, query_block, or the query split size")

    log = get_logger(progress_stage)
    dense_queries = torch.empty((n_query, dim), dtype=torch.float32, device=device)
    for first in range(0, n_query, query_block):
        last = min(first + query_block, n_query)
        dense_queries[first:last] = query_source.dense(first, last)
    best = torch.full((n_query,), -math.inf, dtype=torch.float32, device=device)
    positions = torch.zeros(n_query, dtype=torch.int64, device=device)

    progress = Progress(n_gallery, stage=progress_stage, what="gallery rows")
    with fp32_matmul(), torch.no_grad():
        _synchronize(device)
        started = time.perf_counter()
        for start in range(0, n_gallery, gallery_chunk):
            stop = min(start + gallery_chunk, n_gallery)
            chunk = gallery_source.dense(start, stop)
            for first in range(0, n_query, query_block):
                last = min(first + query_block, n_query)
                value, position = (dense_queries[first:last] @ chunk.T).max(dim=1)
                better = value > best[first:last]
                best[first:last] = torch.where(better, value, best[first:last])
                positions[first:last] = torch.where(better, position + start,
                                                    positions[first:last])
            progress.advance(stop - start)
        _synchronize(device)
        seconds = time.perf_counter() - started
    log.info("searched %s x %s at dim=%d in %s", f"{n_query:,}", f"{n_gallery:,}", dim,
             format_duration(seconds))
    return best, positions, seconds


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ----------------------------------------------------------------- protocol

def top1_accuracy(gallery_codes: Any, gallery_labels: Any, query_codes: Any,
                  query_labels: Any, *, dim: int, device: torch.device | str,
                  gallery_chunk: int = DENSE_GEMM_GALLERY_CHUNK,
                  query_block: int = DENSE_GEMM_QUERY_BLOCK,
                  normalize: bool = False,
                  progress_stage: str = "knn") -> dict[str, Any]:
    """The reported retrieval metric: sparse-code 1-NN Top-1 in percent.

    Gallery = every encoded train row, queries = the 50k encoded validation
    rows, both L2-normalised (``encode_compact`` already did it). ``dim`` is the
    autoencoder's ``hidden_dim``, the width the codes are expanded to.

    Empty query codes are counted and scored as misses. An empty code carries no
    information, so crediting it with whatever neighbour the tie-break happens
    to return would inflate the number of exactly the methods whose codes
    collapse -- the failure mode worth seeing.
    """
    gallery_labels = _labels(gallery_labels, "gallery")
    query_labels = _labels(query_labels, "query")
    device = torch.device(device)
    scores, positions, seconds = exact_top1(
        gallery_codes, query_codes, dim=dim, device=device, gallery_chunk=gallery_chunk,
        query_block=query_block, normalize=normalize, progress_stage=progress_stage)
    n_query = len(positions)
    if len(query_labels) != n_query:
        raise ValueError(f"{n_query:,} queries but {len(query_labels):,} query labels")
    gallery_source = CodeSource(gallery_codes, dim=dim, device=device)
    if len(gallery_labels) != gallery_source.rows:
        raise ValueError(f"{gallery_source.rows:,} gallery rows but "
                         f"{len(gallery_labels):,} gallery labels")
    predicted = torch.as_tensor(gallery_labels, device=device)[positions]
    truth = torch.as_tensor(query_labels, device=device)
    hit = predicted.eq(truth)
    # Only the all-zero codes are forced to miss. A nonempty code that happens
    # to share no latent with any gallery row also scores 0.0 and lands on row 0
    # by the tie rule, exactly as in FAISS, and is left to that outcome.
    empty_mask = CodeSource(query_codes, dim=dim, device=device).empty_mask()
    empty = int(empty_mask.sum())
    if empty:
        hit = hit & empty_mask.logical_not()
        get_logger(progress_stage).warning(
            "%s of %s queries have an empty code and are scored as misses",
            f"{empty:,}", f"{n_query:,}")
    correct = int(hit.sum())
    return {"top1": 100.0 * correct / n_query, "correct": correct, "queries": n_query,
            "empty_queries": empty, "seconds": seconds, "backend": BACKEND,
            # Mean cosine to the retrieved neighbour: a collapsing code shows up
            # here as a score near 1 with falling accuracy.
            "mean_neighbour_score": float(scores.mean()),
            "gallery_rows": gallery_source.rows, "dim": int(dim),
            "search_precision": "float32_tf32_disabled",
            "tie_rule": "lowest_gallery_position",
            "empty_query_policy": "counted_and_scored_as_miss"}


def dense_top1(gallery: Any, gallery_labels: Any, queries: Any, query_labels: Any, *,
               device: torch.device | str, dim: int | None = None,
               gallery_chunk: int = DENSE_GEMM_GALLERY_CHUNK,
               query_block: int = DENSE_GEMM_QUERY_BLOCK,
               progress_stage: str = "knn") -> dict[str, Any]:
    """1-NN Top-1 of the raw backbone embedding -- CSR Table 4 "Full Rep.".

    This is the pipeline's correctness gate. FF2048's published "Full Rep."
    is 71.19; the embed stage warns beyond 0.05 and fails beyond 0.3
    (plan gate G1). A deviation means the embedding recipe is wrong, and every
    sparse number downstream would be measured against the wrong backbone, so
    the caller stops the pipeline rather than training on it.

    Rows arrive un-normalised from :func:`~mpsae.evaluate.encode.encode_dense`
    and are L2-normalised here, mirroring ``faiss.normalize_L2`` being applied
    to the loaded arrays rather than at extraction time.
    """
    if dim is None:
        if getattr(gallery, "ndim", 0) != 2:
            raise TypeError("dense_top1 takes a dense [rows, d] gallery; pass dim explicitly "
                            "only if that matrix is wrapped")
        dim = int(gallery.shape[1])
    result = top1_accuracy(gallery, gallery_labels, queries, query_labels, dim=dim,
                           device=device, gallery_chunk=gallery_chunk,
                           query_block=query_block, normalize=True,
                           progress_stage=progress_stage)
    result["representation"] = "dense_backbone_embedding"
    return result


# -------------------------------------------------------------------- oracle

def verify_against_bruteforce(gallery: Any, queries: Any, *, dim: int,
                              device: torch.device | str = "cpu",
                              gallery_chunk: int = 97, query_block: int = 13
                              ) -> dict[str, Any]:
    """Compare :func:`exact_top1` with a plain NumPy argmax. Tests only.

    The deliberately awkward default chunk sizes exercise the merge path: with
    both constants coprime to the input sizes, no block boundary lines up with a
    tie, so a wrong merge rule shows up as a position mismatch rather than
    passing by luck.
    """
    scores, positions, _ = exact_top1(gallery, queries, dim=dim, device=device,
                                      gallery_chunk=gallery_chunk, query_block=query_block)
    reference_scores, reference_positions = _bruteforce(gallery, queries, dim=dim)
    got_scores = scores.detach().cpu().numpy()
    got_positions = positions.detach().cpu().numpy()
    score_gap = float(np.max(np.abs(got_scores - reference_scores))) if len(got_scores) else 0.0
    mismatched = np.flatnonzero(got_positions != reference_positions)
    return {"max_score_difference": score_gap,
            "position_mismatches": int(mismatched.size),
            "mismatched_queries": mismatched[:16].tolist(),
            "queries": len(got_positions)}


def _bruteforce(gallery: Any, queries: Any, *, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense float32 NumPy top-1 with the same lowest-position tie rule."""
    dense_gallery = _to_numpy_dense(gallery, dim)
    dense_queries = _to_numpy_dense(queries, dim)
    scores = dense_queries @ dense_gallery.T
    positions = np.argmax(scores, axis=1)  # NumPy also returns the first maximum
    return scores[np.arange(len(positions)), positions].astype(np.float32), positions


def _to_numpy_dense(codes: Any, dim: int) -> np.ndarray:
    source = CodeSource(codes, dim=dim, device="cpu")
    return source.dense(0, source.rows).numpy()


def _labels(labels: Any, what: str) -> np.ndarray:
    array = np.asarray(labels).reshape(-1)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{what} labels must be integer class ids, got {array.dtype}")
    if not len(array):
        raise ValueError(f"{what} labels are empty")
    return array.astype(np.int64, copy=False)
