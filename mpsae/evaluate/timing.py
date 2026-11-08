"""Retrieval-cost measurement: the efficiency figure, not a quality metric.

Protocol from the papers' own efficiency experiments -- CSR App. E.3 and CSRv2
App. G.1: queries are issued in batches of **512**, a batch is searched **100**
times untimed to warm caches and autotuners, then **2000** timed rounds of the
same batch are recorded. Repeating one resident batch is what the papers
measure, so these numbers are throughput of a warm search, not cold latency and
not single-query latency.

Two paths are timed so the paper's speedup ratio can be reproduced:

* ``time_sparse_matmul`` -- the sparse code against a ``torch.sparse_csr``
  gallery, which is the representation the papers claim the saving for;
* ``time_dense_gemm`` -- the same harness over the dense backbone embedding,
  the denominator of that ratio.

``maybe_build_inverted_index`` additionally offers this repo's hand-written CUDA
inverted index (``csrc/``). It is a strictly optional third data point: if it
does not compile, the function warns and returns ``None``. Nothing else in the
pipeline may depend on it, because the compiler toolchain on a rented GPU box
is not something a correctness gate should hinge on.

Each measurement reports CUDA-event time *and* a wall-clock cross-check. The
events bound the search interval on the stream; the wall clock also contains
launch and completion overhead. When they disagree badly the measurement is
dominated by dispatch, and the summary carries both so that is visible.
"""
from __future__ import annotations

import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from ..io import human_bytes
from ..logs import get_logger
from .encode import CodeSource
from .knn import fp32_matmul, require_vram

#: CSR App. E.3 / CSRv2 App. G.1.
PAPER_QUERY_BATCH = 512
PAPER_WARMUP = 100
PAPER_ROUNDS = 2000

#: Architectures the extension is asked to target: whatever the running device
#: is, plus H100 (9.0) and Blackwell (12.0) so a compiled cache is portable
#: across the machines this project rents.
EXTENSION_ARCHES = ("9.0", "12.0")

_EXTENSION: Any = None
_EXTENSION_FAILED = False


# ------------------------------------------------------------------ summary

def _summarise(samples: list[float]) -> dict[str, float]:
    """Median first: the mean of 2000 rounds is skewed by occasional preemption."""
    ordered = sorted(samples)
    return {"median": statistics.median(ordered), "mean": statistics.fmean(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
            "min": ordered[0], "max": ordered[-1], "rounds": len(ordered)}


def _measure(search, *, device: torch.device, warmup: int, rounds: int) -> dict[str, Any]:
    """Run ``search`` ``warmup`` times untimed, then ``rounds`` times timed."""
    if warmup < 1 or rounds < 1:
        raise ValueError("warmup and rounds must be positive")
    if device.type != "cuda":
        raise RuntimeError("retrieval timing requires CUDA; a CPU number would not be comparable")
    stream = torch.cuda.current_stream(device)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Events initialise lazily on first record; pay that before the timed loop.
    start.record(stream)
    stop.record(stream)
    stop.synchronize()
    for _ in range(warmup):
        search()
    torch.cuda.synchronize(device)
    cuda_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(rounds):
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        start.record(stream)
        search()
        stop.record(stream)
        stop.synchronize()
        wall_samples.append(time.perf_counter() - began)
        cuda_samples.append(start.elapsed_time(stop) / 1000.0)
    return {"cuda_seconds": _summarise(cuda_samples),
            "wall_seconds": _summarise(wall_samples),
            "warmup": warmup, "rounds": rounds}


def _per_query(summary: dict[str, Any], queries: int) -> dict[str, Any]:
    median = summary["cuda_seconds"]["median"]
    summary["queries_per_batch"] = queries
    summary["milliseconds_per_batch"] = 1000.0 * median
    summary["microseconds_per_query"] = 1e6 * median / queries
    summary["queries_per_second"] = queries / median if median > 0 else float("inf")
    summary["timing_scope"] = ("warm_resident_batch_search_repeated;"
                               "excludes_encoding_normalization_transfers_and_index_build")
    return summary


# ----------------------------------------------------------- sparse CSR path

def build_csr_gallery(gallery_compact: Any, *, dim: int, device: torch.device | str,
                      rows: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Assemble the gallery as one ``torch.sparse_csr_tensor``.

    Zero-valued slots are dropped: a code with fewer than K active latents pads
    its Top-K with zeros, and leaving those in the posting lists would inflate
    both the index size and the multiply count that the efficiency claim is
    about. Column indices are sorted within each row, which CSR expects and
    which cuSPARSE's row kernels rely on for coalesced reads.
    """
    device = torch.device(device)
    source = CodeSource(gallery_compact, dim=dim, device=device)
    if not source.is_compact:
        raise TypeError("the sparse path needs compact (indices, values) codes")
    limit = source.rows if rows is None else min(rows, source.rows)
    columns: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    block = 65536
    for start in range(0, limit, block):
        stop = min(start + block, limit)
        chunk_indices, chunk_values = source.compact(start, stop)
        keep = chunk_values > 0
        order = torch.argsort(torch.where(keep, chunk_indices.to(torch.int64),
                                          torch.full_like(chunk_indices, dim, dtype=torch.int64)),
                              dim=1)
        sorted_indices = chunk_indices.to(torch.int64).gather(1, order)
        sorted_values = chunk_values.gather(1, order)
        sorted_keep = keep.gather(1, order)
        counts.append(sorted_keep.sum(dim=1))
        columns.append(sorted_indices[sorted_keep])
        values.append(sorted_values[sorted_keep])
    row_counts = torch.cat(counts)
    crow = torch.zeros(limit + 1, dtype=torch.int64, device=device)
    crow[1:] = torch.cumsum(row_counts, dim=0)
    column_indices = torch.cat(columns)
    value_vector = torch.cat(values)
    matrix = torch.sparse_csr_tensor(crow, column_indices, value_vector,
                                     size=(limit, dim), device=device)
    index_bytes = sum(t.numel() * t.element_size() for t in (crow, column_indices, value_vector))
    return matrix, {"rows": limit, "dim": int(dim), "nnz": int(value_vector.numel()),
                    "nnz_per_row": float(value_vector.numel()) / limit,
                    "index_bytes": index_bytes, "index_bytes_human": human_bytes(index_bytes),
                    "layout": "torch_sparse_csr"}


def time_sparse_matmul(gallery_compact: Any, queries_compact: Any, *, dim: int,
                       device: torch.device | str, query_batch: int = PAPER_QUERY_BATCH,
                       warmup: int = PAPER_WARMUP, rounds: int = PAPER_ROUNDS,
                       gallery_rows: int | None = None,
                       batch_index: int = 0) -> dict[str, Any]:
    """Time ``queries @ gallery.T`` with a sparse CSR gallery.

    Implemented as ``gallery @ queries.T`` -- the transpose of the requested
    product, elementwise identical -- because cuSPARSE provides sparse-times-
    dense and PyTorch has no efficient dense-times-sparse-transpose kernel. The
    score matrix is ``[gallery_rows, query_batch]`` instead of its transpose;
    the top-1 reduction then runs down dim 0. No argmax is timed beyond that, so
    the number is comparable with the dense path, which does the same work.

    Multiplies run in float32 with TF32 off, as in the search itself: a timing
    taken at a precision we never retrieve at would not describe this pipeline.
    """
    device = torch.device(device)
    matrix, index_info = build_csr_gallery(gallery_compact, dim=dim, device=device,
                                           rows=gallery_rows)
    queries = CodeSource(queries_compact, dim=dim, device=device)
    first = batch_index * query_batch
    if first >= queries.rows:
        raise ValueError(f"batch_index={batch_index} starts past {queries.rows:,} queries")
    last = min(first + query_batch, queries.rows)
    require_vram(device, index_info["rows"] * (last - first) * 4 + (last - first) * dim * 4,
                 what="sparse retrieval score buffer",
                 knobs="query_batch or gallery_rows")
    dense_queries = queries.dense(first, last).contiguous()

    def search() -> None:
        scores = torch.matmul(matrix, dense_queries.T)
        torch.max(scores, dim=0)

    log = get_logger("timing")
    log.info("sparse CSR gallery: %s rows, %s nnz, index %s", f"{index_info['rows']:,}",
             f"{index_info['nnz']:,}", index_info["index_bytes_human"])
    with fp32_matmul(), torch.no_grad():
        summary = _measure(search, device=device, warmup=warmup, rounds=rounds)
    summary.update(backend="torch_sparse_csr_matmul", index=index_info,
                   protocol="CSR App. E.3 / CSRv2 App. G.1",
                   precision="float32_tf32_disabled")
    return _per_query(summary, last - first)


# ------------------------------------------------------------- dense GEMM path

def time_dense_gemm(gallery: Any, queries: Any, *, dim: int, device: torch.device | str,
                    query_batch: int = PAPER_QUERY_BATCH, warmup: int = PAPER_WARMUP,
                    rounds: int = PAPER_ROUNDS, gallery_rows: int | None = None,
                    batch_index: int = 0, normalize: bool = False) -> dict[str, Any]:
    """The same harness over a GPU-resident dense gallery.

    ``gallery`` is the dense backbone embedding (``dim = d``), the baseline the
    papers compare sparse retrieval against. Densifying the *sparse* gallery is
    not an option here: at h=8192 the 1.28M-row ImageNet gallery would be 42 TiB,
    which is also precisely why the search in ``knn.py`` streams it in chunks.
    Use ``gallery_rows`` to subsample when the dense gallery does not fit, and
    pass the same value to :func:`time_sparse_matmul` so the ratio is paired.
    """
    device = torch.device(device)
    source = CodeSource(gallery, dim=dim, device=device, normalize=normalize)
    limit = source.rows if gallery_rows is None else min(gallery_rows, source.rows)
    require_vram(device, limit * dim * 4 + limit * query_batch * 4 + query_batch * dim * 4,
                 what=f"dense gallery of {limit:,} x {dim}",
                 knobs="gallery_rows or query_batch")
    resident = torch.empty((limit, dim), dtype=torch.float32, device=device)
    for start in range(0, limit, 65536):
        stop = min(start + 65536, limit)
        resident[start:stop] = source.dense(start, stop)
    query_source = CodeSource(queries, dim=dim, device=device, normalize=normalize)
    first = batch_index * query_batch
    if first >= query_source.rows:
        raise ValueError(f"batch_index={batch_index} starts past {query_source.rows:,} queries")
    last = min(first + query_batch, query_source.rows)
    dense_queries = query_source.dense(first, last).contiguous()

    def search() -> None:
        scores = dense_queries @ resident.T
        torch.max(scores, dim=1)

    index_bytes = resident.numel() * resident.element_size()
    with fp32_matmul(), torch.no_grad():
        summary = _measure(search, device=device, warmup=warmup, rounds=rounds)
    summary.update(backend="dense_fp32_gemm",
                   index={"rows": limit, "dim": int(dim), "index_bytes": index_bytes,
                          "index_bytes_human": human_bytes(index_bytes),
                          "layout": "dense_row_major"},
                   protocol="CSR App. E.3 / CSRv2 App. G.1",
                   precision="float32_tf32_disabled")
    return _per_query(summary, last - first)


def speedup(sparse: dict[str, Any], dense: dict[str, Any]) -> dict[str, Any]:
    """Paired ratio of the two measurements above, with the pairing checked."""
    if sparse["queries_per_batch"] != dense["queries_per_batch"]:
        raise ValueError("speedup needs the same query batch on both sides")
    if sparse["index"]["rows"] != dense["index"]["rows"]:
        raise ValueError("speedup needs the same gallery row count on both sides")
    sparse_median = sparse["cuda_seconds"]["median"]
    dense_median = dense["cuda_seconds"]["median"]
    ratio = dense_median / sparse_median if sparse_median > 0 else float("inf")
    return {"dense_over_sparse": ratio,
            "sparse_milliseconds_per_batch": 1000.0 * sparse_median,
            "dense_milliseconds_per_batch": 1000.0 * dense_median,
            "index_bytes_ratio": sparse["index"]["index_bytes"] / dense["index"]["index_bytes"],
            "gallery_rows": sparse["index"]["rows"],
            "queries_per_batch": sparse["queries_per_batch"]}


# --------------------------------------------------- optional CUDA extension

def maybe_build_inverted_index(*, source_dir: str | Path | None = None,
                               verbose: bool = False) -> Any:
    """Compile and return the ``csrc/`` inverted-index extension, or ``None``.

    Respect TORCH_CUDA_ARCH_LIST generated from the selected GPU capabilities.
    Otherwise try this device plus H100 and Blackwell, then fall back to the
    running device alone: older nvcc rejects ``12.0`` outright. A second
    failure is logged and swallowed -- this is a secondary number and must never
    fail a run.
    """
    global _EXTENSION, _EXTENSION_FAILED
    if _EXTENSION is not None or _EXTENSION_FAILED:
        return _EXTENSION
    log = get_logger("timing")
    if not torch.cuda.is_available():
        log.warning("no CUDA device: skipping the sparse inverted-index extension")
        _EXTENSION_FAILED = True
        return None
    root = Path(source_dir) if source_dir is not None else \
        Path(__file__).resolve().parents[2] / "csrc"
    sources = [str(root / "sparse_retrieval.cpp"), str(root / "sparse_retrieval.cu")]
    missing = [path for path in sources if not Path(path).is_file()]
    if missing:
        log.warning("inverted-index sources missing (%s): skipping", ", ".join(missing))
        _EXTENSION_FAILED = True
        return None
    device_arch = ".".join(map(str, torch.cuda.get_device_capability()))
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    configured = previous.replace(" ", ";").split(";") if previous else []
    primary = [arch for arch in configured if arch] or sorted({device_arch, *EXTENSION_ARCHES})
    # Honor the installer's detected architectures rather than first compiling
    # optional architectures that the selected toolkit may not support.
    attempts = (primary,) if primary == [device_arch] else (primary, [device_arch])
    try:
        for arches in attempts:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arches)
            try:
                from torch.utils.cpp_extension import load

                _EXTENSION = load(name="mpsae_sparse_retrieval", sources=sources,
                                  extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
                                  extra_cuda_cflags=["-O3"], with_cuda=True, verbose=verbose)
                log.info("inverted-index extension built for sm %s", ", ".join(arches))
                return _EXTENSION
            except Exception as error:
                log.warning("inverted-index build for sm %s failed (%s: %s)",
                            ", ".join(arches), type(error).__name__, error)
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous
    log.warning("continuing without the inverted-index metric")
    _EXTENSION_FAILED = True
    return None
