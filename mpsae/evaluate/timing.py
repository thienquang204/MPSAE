"""Retrieval-cost measurement: the efficiency figure, not a quality metric.

Queries are issued in batches of **512**, with **5** untimed warmup searches
and **5** timed rounds by default. The papers' longer 100/2000 schedule can be
requested explicitly. The short schedule gives a quick estimate with fewer
samples; it does not reproduce that longer timing protocol. Repeating one
resident batch measures warm search throughput, not cold or single-query latency.

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
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from ..io import human_bytes
from ..logs import get_logger
from ..timing_config import (DEFAULT_BUILD_ENTRY_CAP, DEFAULT_GALLERY_CHUNK,
                             DEFAULT_ROUNDS, DEFAULT_WARMUP, PAPER_QUERY_BATCH,
                             SPARSE_BACKEND, TIMING_SCOPE, TIMING_VERSION,
                             runtime_fingerprint, timing_config)
from .encode import CodeSource
from .knn import fp32_matmul, require_vram

#: Architectures the extension is asked to target: whatever the running device
#: is, plus H100 (9.0) and Blackwell (12.0) so a compiled cache is portable
#: across the machines this project rents.
EXTENSION_ARCHES = ("9.0", "12.0")

_EXTENSION: Any = None
_EXTENSION_FAILED = False


# ------------------------------------------------------------------ summary

def _summarise(samples: list[float]) -> dict[str, float]:
    """Median first: occasional preemption can skew the mean."""
    ordered = sorted(samples)
    return {"median": statistics.median(ordered), "mean": statistics.fmean(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
            "min": ordered[0], "max": ordered[-1], "rounds": len(ordered)}


def _measure(search, *, device: torch.device, warmup: int, rounds: int,
             progress_seconds: float = 10.0) -> dict[str, Any]:
    """Run ``search`` ``warmup`` times untimed, then ``rounds`` times timed."""
    if (isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1
            or isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1):
        raise ValueError("warmup and rounds must be positive")
    if not math.isfinite(progress_seconds) or progress_seconds < 0:
        raise ValueError("progress_seconds must be finite and nonnegative")
    if device.type != "cuda":
        raise RuntimeError("retrieval timing requires CUDA; a CPU number would not be comparable")
    stream = torch.cuda.current_stream(device)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Events initialise lazily on first record; pay that before the timed loop.
    start.record(stream)
    stop.record(stream)
    stop.synchronize()
    log = get_logger("timing")
    phase_started = last_log = time.perf_counter()
    log.info("warm-up: 0/%d rounds", warmup)
    for index in range(warmup):
        search()
        now = time.perf_counter()
        if progress_seconds and now - last_log >= progress_seconds:
            # Warm-up is untimed; wait for completed work before estimating ETA.
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            log.info("warm-up: %d/%d rounds | elapsed %.1fs | ETA %.1fs",
                     index + 1, warmup, now - phase_started,
                     (now - phase_started) / (index + 1) * (warmup - index - 1))
            last_log = time.perf_counter()
    torch.cuda.synchronize(device)
    log.info("warm-up complete: %d rounds in %.1fs; measuring %d rounds",
             warmup, time.perf_counter() - phase_started, rounds)
    phase_started = last_log = time.perf_counter()
    cuda_samples: list[float] = []
    wall_samples: list[float] = []
    for index in range(rounds):
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        start.record(stream)
        search()
        stop.record(stream)
        stop.synchronize()
        wall_samples.append(time.perf_counter() - began)
        cuda_samples.append(start.elapsed_time(stop) / 1000.0)
        # Logging happens after both timing samples have been captured, never
        # inside the CUDA-event or wall-clock search interval.
        now = time.perf_counter()
        if progress_seconds and now - last_log >= progress_seconds:
            log.info("timing: %d/%d rounds | elapsed %.1fs | ETA %.1fs | last %.2fms",
                     index + 1, rounds, now - phase_started,
                     (now - phase_started) / (index + 1) * (rounds - index - 1),
                     1000 * cuda_samples[-1])
            last_log = time.perf_counter()
    log.info("timing complete: %d rounds | median %.2fms/batch",
             rounds, 1000 * statistics.median(cuda_samples))
    return {"cuda_seconds": _summarise(cuda_samples),
            "wall_seconds": _summarise(wall_samples),
            "warmup": warmup, "rounds": rounds}


def _per_query(summary: dict[str, Any], queries: int) -> dict[str, Any]:
    median = summary["cuda_seconds"]["median"]
    summary["queries_per_batch"] = queries
    summary["milliseconds_per_batch"] = 1000.0 * median
    summary["microseconds_per_query"] = 1e6 * median / queries
    summary["queries_per_second"] = queries / median if median > 0 else float("inf")
    summary["timing_scope"] = TIMING_SCOPE
    return summary


# ----------------------------------------------------------- sparse CSR path

def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shard_rows(k: int, requested: int, *, int32: bool = True) -> int:
    """Bound *local* CSR offsets; the whole gallery may exceed 2**31 nnz."""
    _positive_integer("k", k)
    _positive_integer("gallery_chunk", requested)
    if int32:
        maximum = torch.iinfo(torch.int32).max
        if k > maximum:
            raise ValueError("one row exceeds the int32 CSR entry range")
        return min(requested, maximum // k)
    return requested


def _scan_counts(source: CodeSource, limit: int, block: int) -> torch.Tensor:
    """Count actual nnz before resident allocations; keep row counts on CPU."""
    counts = torch.empty(limit, dtype=torch.int64, device="cpu")
    log = get_logger("timing")
    last_log = time.perf_counter()
    for first in range(0, limit, block):
        last = min(first + block, limit)
        indices, values = source.compact(first, last)
        keep = values > 0
        valid = (torch.isfinite(values).all() & values.ge(0).all()
                 & ((indices.ge(0) & indices.lt(source.dim)) | ~keep).all())
        if not bool(valid):
            raise ValueError("CSR codes need finite nonnegative values and valid active coordinates")
        counts[first:last] = keep.sum(1).cpu()
        if time.perf_counter() - last_log >= 10.0:
            log.info("counting sparse gallery: %s/%s rows", f"{last:,}", f"{limit:,}")
            last_log = time.perf_counter()
    return counts


@torch.no_grad()
def _build_csr_shards(gallery_compact: Any, *, dim: int, device: torch.device | str,
                      rows: int | None, gallery_chunk: int, build_entry_cap: int,
                      query_batch: int, index_dtype: torch.dtype
                      ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Two-pass, bounded-workspace build with no concatenation of GPU chunks.

    Each shard's final column/value buffers are allocated exactly once and
    filled in bounded entry blocks. All shards remain resident, so search
    timing excludes transfers just like the original full-resident harness.
    """
    _positive_integer("dim", dim)
    _positive_integer("build_entry_cap", build_entry_cap)
    _positive_integer("query_batch", query_batch)
    device = torch.device(device)
    source = CodeSource(gallery_compact, dim=dim, device=device)
    if not source.is_compact:
        raise TypeError("the sparse path needs compact (indices, values) codes")
    if rows is not None:
        _positive_integer("gallery_rows", rows)
    limit = source.rows if rows is None else min(rows, source.rows)
    _positive_integer("gallery rows", limit)
    if index_dtype not in (torch.int32, torch.int64):
        raise ValueError("CSR indices must be int32 or int64")
    if dim > torch.iinfo(index_dtype).max:
        raise ValueError("latent dimension exceeds the selected CSR index range")
    shard_rows = _shard_rows(source.k, gallery_chunk, int32=index_dtype == torch.int32)
    if build_entry_cap < source.k:
        raise ValueError("build_entry_cap cannot hold one compact gallery row")
    block = min(shard_rows, build_entry_cap // source.k, limit)
    counts = _scan_counts(source, limit, block)
    nnz = int(counts.sum())
    shard_count = math.ceil(limit / shard_rows)
    index_width = torch.iinfo(index_dtype).bits // 8
    index_bytes = nnz * (4 + index_width) + (limit + shard_count) * index_width
    # Conservative bound for sorting/masks/gather/conversion allocations.
    # cuSPARSE/allocator workspace is covered by require_vram's reserve too.
    build_workspace = block * source.k * 96 + block * 16
    score_workspace = min(shard_rows, limit) * query_batch * 4
    query_workspace = query_batch * dim * 4
    required = index_bytes + max(build_workspace, score_workspace + query_workspace)
    log = get_logger("timing")
    log.info("sparse gallery preflight: %s rows, %s actual nnz, %d shards | index %s | "
             "build workspace <= %s | score tile %s", f"{limit:,}", f"{nnz:,}", shard_count,
             human_bytes(index_bytes), human_bytes(build_workspace), human_bytes(score_workspace))
    if device.type == "cuda":
        # A preceding linear probe may have released a large resident code
        # cache. Release unused allocator blocks once, not in the search loop.
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    require_vram(device, required, what="resident CSR shards and build/search workspace",
                 knobs="gallery_chunk/build_entry_cap, or explicitly gallery_rows")

    shards = []
    last_log = time.perf_counter()
    for first in range(0, limit, shard_rows):
        last = min(first + shard_rows, limit)
        local_counts = counts[first:last]
        local_nnz = int(local_counts.sum())
        if local_nnz > torch.iinfo(index_dtype).max:
            raise ValueError("CSR shard nnz exceeds its offset dtype; reduce gallery_chunk")
        crow_cpu = torch.zeros(last - first + 1, dtype=torch.int64)
        crow_cpu[1:] = local_counts.cumsum(0)
        crow = crow_cpu.to(device=device, dtype=index_dtype)
        columns = torch.empty(local_nnz, device=device, dtype=index_dtype)
        values = torch.empty(local_nnz, device=device, dtype=torch.float32)
        for begin in range(first, last, block):
            end = min(begin + block, last)
            chunk_indices, chunk_values = source.compact(begin, end)
            keep = chunk_values > 0
            # Sorting workspace scales with the entry cap, not the full shard.
            sort_keys = torch.where(keep, chunk_indices.to(torch.int64), dim)
            order = torch.argsort(sort_keys, dim=1)
            sorted_indices = chunk_indices.gather(1, order)
            sorted_values = chunk_values.gather(1, order)
            sorted_keep = keep.gather(1, order)
            offset = int(crow_cpu[begin - first])
            stop = int(crow_cpu[end - first])
            packed_columns = sorted_indices[sorted_keep]
            packed_values = sorted_values[sorted_keep]
            if len(packed_values) != stop - offset:
                raise RuntimeError("gallery changed between counting and CSR construction")
            columns[offset:stop].copy_(packed_columns)
            values[offset:stop].copy_(packed_values)
            del chunk_indices, chunk_values, keep, sort_keys, order, sorted_indices
            del sorted_values, sorted_keep, packed_columns, packed_values
        shards.append(torch.sparse_csr_tensor(crow, columns, values,
                                               size=(last - first, dim), device=device))
        if time.perf_counter() - last_log >= 10.0 or last == limit:
            log.info("building sparse gallery: %s/%s rows | shards %d/%d",
                     f"{last:,}", f"{limit:,}", len(shards), shard_count)
            last_log = time.perf_counter()
    return tuple(shards), {
        "rows": limit, "dim": dim, "nnz": nnz, "nnz_per_row": nnz / limit,
        "index_bytes": index_bytes, "index_bytes_human": human_bytes(index_bytes),
        "layout": "torch_sparse_csr_shards", "index_dtype": str(index_dtype),
        "shards": shard_count, "max_shard_rows": min(shard_rows, limit),
        "build_block_rows": block, "build_entry_cap": build_entry_cap,
        "estimated_build_workspace_bytes": build_workspace,
        "score_workspace_bytes": score_workspace, "query_workspace_bytes": query_workspace,
        "estimated_additional_peak_bytes": required, "residency": "entire_gallery_on_device",
    }


def build_csr_shards(gallery_compact: Any, *, dim: int, device: torch.device | str,
                     rows: int | None = None, gallery_chunk: int = DEFAULT_GALLERY_CHUNK,
                     build_entry_cap: int = DEFAULT_BUILD_ENTRY_CAP,
                     query_batch: int = PAPER_QUERY_BATCH
                     ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Build resident CSR shards with int32 local offsets and bounded scratch."""
    return _build_csr_shards(gallery_compact, dim=dim, device=device, rows=rows,
                             gallery_chunk=gallery_chunk, build_entry_cap=build_entry_cap,
                             query_batch=query_batch, index_dtype=torch.int32)


def build_csr_gallery(gallery_compact: Any, *, dim: int, device: torch.device | str,
                      rows: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compatibility/reference single CSR build, with bounded two-pass scratch.

    The timed production path uses build_csr_shards. This helper keeps int64
    offsets for callers explicitly requesting one matrix; preflight rejects
    insufficient VRAM rather than doubling the index through concatenation.
    """
    source = CodeSource(gallery_compact, dim=dim, device=device)
    limit = source.rows if rows is None else min(rows, source.rows)
    shards, info = _build_csr_shards(gallery_compact, dim=dim, device=device, rows=rows,
                                    gallery_chunk=max(1, limit),
                                    build_entry_cap=DEFAULT_BUILD_ENTRY_CAP,
                                    query_batch=1, index_dtype=torch.int64)
    info["layout"] = "torch_sparse_csr"
    return shards[0], info


def sparse_sharded_top1(shards: tuple[torch.Tensor, ...], queries: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact max/argmax, preserving the first gallery index on equal scores.

    A zero query scores zero against every row and returns row zero here; the
    accuracy layer owns its explicit empty-query-as-miss policy. This helper
    only supplies the same numerical max/argmax used by the timing harness.
    """
    if not shards or queries.ndim != 2 or len(queries) < 1:
        raise ValueError("search needs nonempty shards and [queries, dim] codes")
    if queries.dtype != torch.float32:
        raise TypeError("timed search requires float32 query codes")
    best = queries.new_full((len(queries),), -float("inf"))
    winners = torch.zeros(len(queries), device=queries.device, dtype=torch.int64)
    offset = 0
    transposed = queries.T
    for shard in shards:
        scores = torch.matmul(shard, transposed)
        local, row = torch.max(scores, dim=0)
        improve = local > best
        best = torch.where(improve, local, best)
        winners = torch.where(improve, row + offset, winners)
        offset += shard.shape[0]
        del scores
    return best, winners


def time_sparse_matmul(gallery_compact: Any, queries_compact: Any, *, dim: int,
                       device: torch.device | str, query_batch: int = PAPER_QUERY_BATCH,
                       warmup: int = DEFAULT_WARMUP, rounds: int = DEFAULT_ROUNDS,
                       gallery_rows: int | None = None,
                       batch_index: int = 0,
                       gallery_chunk: int = DEFAULT_GALLERY_CHUNK,
                       build_entry_cap: int = DEFAULT_BUILD_ENTRY_CAP,
                       progress_seconds: float = 10.0) -> dict[str, Any]:
    """Time ``queries @ gallery.T`` with a sparse CSR gallery.

    Implemented as ``shard @ queries.T`` -- the transpose of the requested
    product, elementwise identical -- because cuSPARSE provides sparse-times-
    dense and PyTorch has no efficient dense-times-sparse-transpose kernel. The
    score tile is ``[shard_rows, query_batch]`` instead of its transpose;
    top-1 is reduced locally and merged across every resident shard. The
    dense path also tiles and merges; neither changes gallery membership.

    Multiplies run in float32 with TF32 off, as in the search itself: a timing
    taken at a precision we never retrieve at would not describe this pipeline.
    """
    config = timing_config(query_batch=query_batch, warmup=warmup, rounds=rounds,
                           gallery_chunk=gallery_chunk, build_entry_cap=build_entry_cap)
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("retrieval timing requires CUDA; use build_csr_shards for CPU checks")
    queries = CodeSource(queries_compact, dim=dim, device=device)
    first = batch_index * query_batch
    if first >= queries.rows:
        raise ValueError(f"batch_index={batch_index} starts past {queries.rows:,} queries")
    last = min(first + query_batch, queries.rows)
    shards, index_info = build_csr_shards(
        gallery_compact, dim=dim, device=device, rows=gallery_rows,
        gallery_chunk=gallery_chunk, build_entry_cap=build_entry_cap, query_batch=last - first)
    dense_queries = queries.dense(first, last).contiguous()

    def search() -> None:
        sparse_sharded_top1(shards, dense_queries)

    log = get_logger("timing")
    log.info("sparse CSR gallery: %s rows, %s nnz, index %s, %d resident shards",
             f"{index_info['rows']:,}", f"{index_info['nnz']:,}",
             index_info["index_bytes_human"], index_info["shards"])
    with fp32_matmul(), torch.no_grad():
        summary = _measure(search, device=device, warmup=warmup, rounds=rounds,
                           progress_seconds=progress_seconds)
    summary.update(backend=SPARSE_BACKEND, version=TIMING_VERSION,
                   runtime=runtime_fingerprint(device),
                   config=config, index=index_info,
                   protocol="CSR App. E.3 / CSRv2 App. G.1",
                   precision="float32_tf32_disabled")
    return _per_query(summary, last - first)


# ------------------------------------------------------------- dense GEMM path

def time_dense_gemm(gallery: Any, queries: Any, *, dim: int, device: torch.device | str,
                    query_batch: int = PAPER_QUERY_BATCH, warmup: int = DEFAULT_WARMUP,
                    rounds: int = DEFAULT_ROUNDS, gallery_rows: int | None = None,
                    batch_index: int = 0, normalize: bool = False,
                    gallery_chunk: int = DEFAULT_GALLERY_CHUNK,
                    progress_seconds: float = 10.0) -> dict[str, Any]:
    """The same harness over a GPU-resident dense gallery.

    ``gallery`` is the dense backbone embedding (``dim = d``), the baseline the
    papers compare sparse retrieval against. Densifying the *sparse* gallery is
    not an option here: at h=8192 the 1.28M-row ImageNet gallery is about 39 GiB,
    which is also precisely why the search in ``knn.py`` streams it in chunks.
    Use ``gallery_rows`` to subsample when the dense gallery does not fit, and
    pass the same value to :func:`time_sparse_matmul` so the ratio is paired.
    """
    config = timing_config(query_batch=query_batch, warmup=warmup, rounds=rounds,
                           gallery_chunk=gallery_chunk)
    config["backend"] = "dense_fp32_tiled_gemm"
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    if gallery_rows is not None:
        _positive_integer("gallery_rows", gallery_rows)
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("retrieval timing requires CUDA")
    source = CodeSource(gallery, dim=dim, device=device, normalize=normalize)
    limit = source.rows if gallery_rows is None else min(gallery_rows, source.rows)
    _positive_integer("gallery rows", limit)
    block = min(gallery_chunk, limit)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    require_vram(device, limit * dim * 4 + max(block * query_batch * 4 + query_batch * dim * 4,
                                              block * dim * 16),
                 what=f"dense gallery of {limit:,} x {dim}",
                 knobs="gallery_chunk, query_batch, or explicitly gallery_rows")
    resident = torch.empty((limit, dim), dtype=torch.float32, device=device)
    for start in range(0, limit, block):
        stop = min(start + block, limit)
        resident[start:stop] = source.dense(start, stop)
    query_source = CodeSource(queries, dim=dim, device=device, normalize=normalize)
    first = batch_index * query_batch
    if first >= query_source.rows:
        raise ValueError(f"batch_index={batch_index} starts past {query_source.rows:,} queries")
    last = min(first + query_batch, query_source.rows)
    dense_queries = query_source.dense(first, last).contiguous()

    def search() -> None:
        dense_tiled_top1(resident, dense_queries, gallery_chunk=gallery_chunk)

    index_bytes = resident.numel() * resident.element_size()
    with fp32_matmul(), torch.no_grad():
        summary = _measure(search, device=device, warmup=warmup, rounds=rounds,
                           progress_seconds=progress_seconds)
    summary.update(backend="dense_fp32_tiled_gemm", version=TIMING_VERSION, config=config,
                   runtime=runtime_fingerprint(device),
                   index={"rows": limit, "dim": int(dim), "index_bytes": index_bytes,
                          "index_bytes_human": human_bytes(index_bytes),
                          "layout": "dense_row_major", "max_shard_rows": block,
                          "score_workspace_bytes": block * (last - first) * 4},
                   protocol="CSR App. E.3 / CSRv2 App. G.1",
                   precision="float32_tf32_disabled")
    return _per_query(summary, last - first)


def dense_tiled_top1(gallery: torch.Tensor, queries: torch.Tensor, *,
                     gallery_chunk: int = DEFAULT_GALLERY_CHUNK
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense counterpart of sparse_sharded_top1 with identical tie handling."""
    _positive_integer("gallery_chunk", gallery_chunk)
    if gallery.ndim != 2 or queries.ndim != 2 or not len(gallery) or not len(queries):
        raise ValueError("search needs nonempty [rows, dim] tensors")
    if gallery.dtype != torch.float32 or queries.dtype != torch.float32:
        raise TypeError("timed search requires float32 codes")
    best = queries.new_full((len(queries),), -float("inf"))
    winners = torch.zeros(len(queries), device=queries.device, dtype=torch.int64)
    for first in range(0, len(gallery), gallery_chunk):
        scores = queries @ gallery[first:first + gallery_chunk].T
        local, row = torch.max(scores, dim=1)
        improve = local > best
        best = torch.where(improve, local, best)
        winners = torch.where(improve, row + first, winners)
        del scores
    return best, winners


def speedup(sparse: dict[str, Any], dense: dict[str, Any]) -> dict[str, Any]:
    """Paired ratio of the two measurements above, with the pairing checked."""
    if not isinstance(sparse.get("runtime"), dict) or not sparse["runtime"]:
        raise ValueError("speedup requires measured runtime/device metadata")
    if sparse["runtime"] != dense.get("runtime"):
        raise ValueError("speedup needs the same runtime/device on both sides")
    if sparse["queries_per_batch"] != dense["queries_per_batch"]:
        raise ValueError("speedup needs the same query batch on both sides")
    if sparse["index"]["rows"] != dense["index"]["rows"]:
        raise ValueError("speedup needs the same gallery row count on both sides")
    for key in ("precision", "timing_scope", "warmup", "rounds", "version"):
        if sparse.get(key) != dense.get(key):
            raise ValueError(f"speedup needs matching {key}")
    if (sparse.get("config") or {}).get("gallery_chunk") != (dense.get("config") or {}).get("gallery_chunk"):
        raise ValueError("speedup needs matching gallery tiling")
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
