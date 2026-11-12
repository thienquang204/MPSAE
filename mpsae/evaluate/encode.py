"""Turn a frozen embedding split into the sparse codes that retrieval consumes.

The precision contract is the whole point of this module and is taken from the
released code, not from the papers:

* the encoder forward runs under **autocast float16** -- ``CSRv2/image/utils.py``
  wraps ``generate_retrieval_data`` in ``with autocast():`` (line 81), as does
  the backbone extraction at line 107;
* the latents are **stored as float32** -- ``dump_feature_vector_array_lists``
  writes ``np.asarray(fwd_pass_x_list, dtype=np.float32)``;
* the search then runs in float32 (see ``knn.py``).

So a code produced here is an fp16 computation rounded to fp32, exactly as in
the baseline. Encoding in fp32 would be a *different* experiment: Top-K over
fp16 pre-activations selects a different latent whenever two pre-activations
fall inside one fp16 ulp, and that is part of the published numbers.

``CodeSource`` is the shared row-access layer for everything downstream
(``knn``, ``probe``, ``timing``): it hides whether a code matrix is dense or
compact, on the host or already on the GPU, so no consumer has to reimplement
densification and none of them can disagree about it.
"""
from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..io import (array_hash, atomic_json, canonical_hash, exclusive_lock, human_bytes,
                  read_json, require_disk)
from ..logs import Progress, format_duration, get_logger

#: Rows per host->device transfer when gathering from a memory-mapped matrix.
#: 64 MiB of float32 at d=2048 -- large enough to amortise the copy, small
#: enough that a stalled read is visible in the progress log.
DEFAULT_ENCODE_BATCH = 1024

#: Free-VRAM headroom demanded before pinning a whole code matrix on the GPU.
RESIDENT_RESERVE_BYTES = 2 << 30


# ---------------------------------------------------------------- row access

def _host_rows(array: Any, start: int, stop: int, *, dtype: torch.dtype,
               device: torch.device) -> torch.Tensor:
    """Copy ``array[start:stop]`` to ``device``.

    The copy is deliberate: ``np.memmap`` hands out read-only views and
    ``torch.as_tensor`` on one of those produces a tensor that warns on every
    later write. Copying once per block costs nothing next to the GEMM.
    """
    block = np.array(array[start:stop], copy=True)
    return torch.as_tensor(block, dtype=dtype, device=device).contiguous()


class CodeSource:
    """Uniform access to one code matrix, dense or compact.

    Accepts a dense ``[rows, dim]`` float32 array/tensor, a ``(indices, values)``
    pair, or the ``{"indices": ..., "values": ...}`` mapping that
    :func:`encode_split` returns. Rows are served as dense float32 ``device``
    tensors; compact rows are expanded with ``scatter_``, which is valid because
    ``torch.topk`` never repeats an index within a row.

    ``normalize`` applies L2 normalisation at expansion time, matching
    ``faiss.normalize_L2`` in ``CSRv2/image/faiss_nn.py``: a zero row stays zero
    instead of becoming NaN, and the retrieval layer counts those rows.
    """

    def __init__(self, codes: Any, *, dim: int, device: torch.device | str,
                 normalize: bool = False, prefer_resident: bool = False) -> None:
        self.dim = int(dim)
        if self.dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.device = torch.device(device)
        self.normalize = bool(normalize)
        indices, values = _split_codes(codes)
        self.indices, self.values = indices, values
        self.is_compact = indices is not None
        if self.is_compact:
            if indices.shape != values.shape or len(indices.shape) != 2:
                raise ValueError(f"compact indices {tuple(indices.shape)} and values "
                                 f"{tuple(values.shape)} must share a [rows, k] shape")
            if not 0 < indices.shape[1] <= self.dim:
                raise ValueError(f"compact k={indices.shape[1]} outside 1..{self.dim}")
        elif getattr(values, "ndim", 0) != 2:
            raise ValueError("dense codes must be a [rows, dim] matrix")
        elif values.shape[1] != self.dim:
            raise ValueError(f"dense codes have width {values.shape[1]}, expected {self.dim}")
        self.rows, self.k = int(values.shape[0]), int(values.shape[1])
        if prefer_resident:
            self._make_resident()

    # ---- construction helpers

    def _tensors(self) -> tuple[Any, ...]:
        return tuple(t for t in (self.indices, self.values) if t is not None)

    def _bytes(self) -> int:
        total = 0
        for array in self._tensors():
            if isinstance(array, torch.Tensor):
                total += array.numel() * array.element_size()
            else:
                total += int(array.size) * array.dtype.itemsize
        return total

    def _make_resident(self) -> None:
        """Pin the whole matrix on the GPU when it fits beside a reserve.

        Only the linear probe needs this: it reads rows in a shuffled order, and
        random reads from a 1.28M-row memory map are IO-bound. This is a
        performance choice with no effect on values or batch composition, so
        falling back to the mapped path is legitimate; it is logged either way.
        """
        if self.device.type != "cuda" or all(isinstance(t, torch.Tensor) for t in self._tensors()):
            return
        need = self._bytes()
        free, _ = torch.cuda.mem_get_info(self.device)
        log = get_logger("encode")
        if need + RESIDENT_RESERVE_BYTES > free:
            log.info("codes stay memory-mapped (%s needed, %s free)",
                     human_bytes(need), human_bytes(free))
            return
        if self.indices is not None:
            self.indices = _upload(self.indices, self.device)
        self.values = _upload(self.values, self.device)
        log.info("codes resident on %s (%s)", self.device, human_bytes(need))

    # ---- row serving

    def dense(self, start: int, stop: int) -> torch.Tensor:
        """``[stop - start, dim]`` float32 rows on ``self.device``."""
        if self.is_compact:
            values = self._slice(self.values, start, stop, torch.float32)
            indices = self._slice(self.indices, start, stop, torch.int64)
            dense = values.new_zeros((len(values), self.dim)).scatter_(1, indices, values)
        else:
            dense = self._slice(self.values, start, stop, torch.float32)
        return _l2_normalize(dense) if self.normalize else dense

    def dense_gather(self, positions: np.ndarray) -> torch.Tensor:
        """Dense rows at arbitrary ``positions`` (used by the shuffled probe)."""
        if self.is_compact:
            values = self._gather(self.values, positions, torch.float32)
            indices = self._gather(self.indices, positions, torch.int64)
            dense = values.new_zeros((len(values), self.dim)).scatter_(1, indices, values)
        else:
            dense = self._gather(self.values, positions, torch.float32)
        return _l2_normalize(dense) if self.normalize else dense

    def compact(self, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(int32 indices, float32 values)`` rows, for the sparse-index paths."""
        if not self.is_compact:
            raise TypeError("dense codes have no compact representation")
        values = self._slice(self.values, start, stop, torch.float32)
        if self.normalize:
            values = _l2_normalize(values)
        return self._slice(self.indices, start, stop, torch.int32), values

    def _slice(self, array: Any, start: int, stop: int, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array[start:stop].to(device=self.device, dtype=dtype).contiguous()
        return _host_rows(array, start, stop, dtype=dtype, device=self.device)

    def _gather(self, array: Any, positions: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            where = torch.as_tensor(positions, device=array.device, dtype=torch.int64)
            return array.index_select(0, where).to(device=self.device, dtype=dtype).contiguous()
        block = np.ascontiguousarray(array[positions])
        return torch.as_tensor(block, dtype=dtype, device=self.device).contiguous()

    # ---- diagnostics

    def empty_mask(self, *, block: int = 65536) -> torch.Tensor:
        """Boolean ``[rows]``: which codes are entirely zero.

        Such a row scores zero against every gallery vector, so ``IndexFlatL2``
        returns an arbitrary (in practice the first) neighbour for it. The
        retrieval layer needs the mask, not just a count, to score exactly those
        queries as misses and leave every other query alone.
        """
        mask = torch.zeros(self.rows, dtype=torch.bool, device=self.device)
        for start in range(0, self.rows, block):
            stop = min(start + block, self.rows)
            values = self._slice(self.values, start, stop, torch.float32)
            mask[start:stop] = values.ne(0).any(dim=1).logical_not()
        return mask

    def count_empty(self, *, block: int = 65536) -> int:
        return int(self.empty_mask(block=block).sum())


def _all_rows(features: Any) -> np.ndarray:
    """``row_ids=None`` means every row, in order.

    That is what the retrieval protocol asks for -- the whole train split is
    the gallery and the whole validation split the query set -- so the callers
    say ``None`` rather than each materialising an identical arange.
    """
    return np.arange(len(features), dtype=np.int64)


def _split_codes(codes: Any) -> tuple[Any, Any]:
    if isinstance(codes, CodeSource):
        raise TypeError("pass the underlying codes, not another CodeSource")
    if isinstance(codes, dict):
        if "values" not in codes:
            raise ValueError(f"code mapping needs a 'values' member; got {sorted(codes)}")
        return codes.get("indices"), codes["values"]
    if isinstance(codes, tuple):
        if len(codes) != 2:
            raise ValueError("a compact code tuple must be (indices, values)")
        return codes
    if getattr(codes, "ndim", 0) != 2:
        raise TypeError("expected a 2-D dense matrix or a compact (indices, values) pair, "
                        f"got {type(codes).__name__}")
    return None, codes


def _upload(array: Any, device: torch.device, *, block: int = 65536) -> torch.Tensor:
    """Block-wise host->device copy, so a 20 GiB matrix needs no 20 GiB staging."""
    dtype = torch.from_numpy(np.empty(0, dtype=array.dtype)).dtype
    target = torch.empty(tuple(array.shape), dtype=dtype, device=device)
    for start in range(0, len(array), block):
        stop = min(start + block, len(array))
        target[start:stop].copy_(torch.from_numpy(np.array(array[start:stop], copy=True)))
    return target


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=1, keepdim=True)
    return torch.where(norm > 0, x / norm.clamp_min(torch.finfo(x.dtype).tiny),
                       torch.zeros_like(x))


# -------------------------------------------------------------------- encode

def encode_split(sae, features: Any, row_ids: np.ndarray | None, k: int, *,
                 device: torch.device | str, batch_size: int = DEFAULT_ENCODE_BATCH,
                 cache_dir: str | Path | None = None,
                 identity: dict[str, Any] | None = None,
                 progress_stage: str = "encode") -> dict[str, Any]:
    """Encode ``features[row_ids]`` into compact Top-K codes.

    ``features`` is a memory-mapped float32 ``[N, d]`` matrix and ``row_ids`` a
    sorted int64 selection; streaming it in row order keeps the read sequential,
    which matters because the FF2048 gallery is 10.5 GiB.

    Returns ``{"indices": int32[n, k], "values": float32[n, k], "stats": {...}}``
    with values already L2-normalised by ``SparseAutoencoder.encode_compact``.

    With ``cache_dir`` and ``identity`` the result is memoised under
    ``canonical_hash(identity)``. Experiment E2 encodes one checkpoint at every
    K in the grid and the linear probe re-reads the same codes, so without the
    cache the gallery would be encoded several times per run. The returned
    arrays are then memory maps, not RAM: at K=2048 the FF2048 gallery is 21 GiB
    of values plus 10 GiB of indices.
    """
    device = torch.device(device)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"K must be a positive integer, got {k!r}")
    if k > sae.hidden_dim:
        raise ValueError(f"K={k} exceeds hidden_dim={sae.hidden_dim}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    row_ids = _all_rows(features) if row_ids is None else row_ids
    row_ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    if not len(row_ids):
        raise ValueError("refusing to encode an empty row selection")
    if row_ids[0] < 0 or row_ids[-1] >= len(features):
        raise ValueError(f"row_ids span [{row_ids[0]}, {row_ids[-1]}] outside "
                         f"{len(features):,} feature rows")
    if np.any(np.diff(row_ids) <= 0):
        raise ValueError("row_ids must be strictly increasing; the caller owns row order")
    if features.shape[1] != sae.input_dim:
        raise ValueError(f"features are {features.shape[1]}-dimensional, "
                         f"the encoder expects {sae.input_dim}")
    if (cache_dir is None) != (identity is None):
        raise ValueError("cache_dir and identity must be given together")

    log = get_logger(progress_stage)
    target = None
    if cache_dir is not None:
        # The key pins the caller's identity (checkpoint, method, data manifest)
        # plus everything about this call that changes the output: K, the width
        # codes are expressed in, and the exact row selection.
        target = Path(cache_dir) / canonical_hash({"identity": identity, "k": k,
                                                   "rows": len(row_ids),
                                                   "row_ids": array_hash(row_ids),
                                                   "precision": ("cuda_autocast_fp16" if device.type == "cuda"
                                                                 else "cpu_float32"),
                                                   "hidden_dim": sae.hidden_dim})
        with exclusive_lock(target.with_suffix(".lock")):
            cached = _read_cache(target, rows=len(row_ids), k=k)
            if cached is not None:
                log.info("reuse cached codes K=%d (%s rows) from %s",
                         k, f"{len(row_ids):,}", target)
                return cached
            return _encode_to_cache(sae, features, row_ids, k, device=device,
                                    batch_size=batch_size, target=target,
                                    identity=identity, progress_stage=progress_stage)
    return _encode(sae, features, row_ids, k, device=device, batch_size=batch_size,
                   writer=None, progress_stage=progress_stage)


def _encode(sae, features: Any, row_ids: np.ndarray, k: int, *, device: torch.device,
            batch_size: int, writer: Any, progress_stage: str) -> dict[str, Any]:
    """Stream the split through ``encode_compact`` under autocast fp16.

    ``writer``, when given, is a ``(indices, values)`` pair of writable arrays
    (memory maps) filled in place; otherwise both are allocated in RAM.
    """
    rows = len(row_ids)
    if writer is None:
        indices = np.empty((rows, k), dtype=np.int32)
        values = np.empty((rows, k), dtype=np.float32)
    else:
        indices, values = writer
    # The cast contract: fp16 compute, fp32 storage (CSRv2 utils.py:81 and
    # dump_feature_vector_array_lists). On CPU -- tests only -- fp16 matmul is
    # unimplemented, so autocast is skipped and the dtype is already fp32.
    autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda" else nullcontext())
    nnz_sum, nnz_min, empty, below_k = 0, k, 0, 0
    was_training = sae.training
    sae.eval()
    progress = Progress(rows, stage=progress_stage, what=f"rows (K={k})")
    started = time.perf_counter()
    try:
        with torch.no_grad(), autocast:
            for start in range(0, rows, batch_size):
                stop = min(start + batch_size, rows)
                batch = _feature_batch(features, row_ids, start, stop, device)
                code_indices, code_values = sae.encode_compact(batch, k)
                # The values must reach the host for storage anyway. Count
                # support there instead of synchronizing CUDA four extra times.
                host_values = code_values.cpu().numpy()
                nnz = np.count_nonzero(host_values > 0, axis=1)
                nnz_sum += int(nnz.sum())
                nnz_min = min(nnz_min, int(nnz.min()))
                empty += int(np.count_nonzero(nnz == 0))
                below_k += int(np.count_nonzero(nnz < k))
                indices[start:stop] = code_indices.cpu().numpy()
                values[start:stop] = host_values
                progress.advance(stop - start)
    finally:
        sae.train(was_training)
    seconds = time.perf_counter() - started
    stats = {"rows": rows, "k": k, "hidden_dim": int(sae.hidden_dim),
             "input_dim": int(sae.input_dim),
             "nnz_mean": nnz_sum / rows, "nnz_min": nnz_min,
             # Reported next to every accuracy number: experiment E2 compares
             # nominal K against the nnz actually realised (plan 4.2).
             "below_k_fraction": below_k / rows,
             "empty_codes": empty, "empty_fraction": empty / rows,
             "seconds": seconds, "rows_per_second": rows / max(seconds, 1e-9),
             "precision": ("autocast_float16_compute_float32_storage" if device.type == "cuda"
                           else "cpu_float32_compute_float32_storage"),
             "normalized": True}
    get_logger(progress_stage).info(
        "encoded %s rows at K=%d in %s | nnz mean %.2f min %d | %s empty",
        f"{rows:,}", k, format_duration(seconds), stats["nnz_mean"], nnz_min, f"{empty:,}")
    return {"indices": indices, "values": values, "stats": stats}


def _feature_batch(features: Any, row_ids: np.ndarray, start: int, stop: int,
                   device: torch.device) -> torch.Tensor:
    """Gather one feature batch, slicing when the selection is contiguous."""
    wanted = row_ids[start:stop]
    if wanted[-1] - wanted[0] == len(wanted) - 1:
        block = np.array(features[wanted[0]:wanted[-1] + 1], copy=True)
    else:
        block = np.ascontiguousarray(features[wanted])
    return torch.as_tensor(block, dtype=torch.float32, device=device)


def _read_cache(target: Path, *, rows: int, k: int) -> dict[str, Any] | None:
    manifest = read_json(target / "manifest.json")
    if manifest is None or not manifest.get("complete"):
        return None
    if manifest.get("rows") != rows or manifest.get("k") != k:
        raise RuntimeError(f"cached codes at {target} hold {manifest.get('rows')} rows at "
                           f"K={manifest.get('k')}, this call wants {rows} rows at K={k}; "
                           f"the identity hash collided -- delete the directory")
    indices = np.load(target / "indices.npy", mmap_mode="r", allow_pickle=False)
    values = np.load(target / "values.npy", mmap_mode="r", allow_pickle=False)
    if indices.shape != (rows, k) or values.shape != (rows, k):
        raise RuntimeError(f"cached code arrays at {target} have shapes {indices.shape} / "
                           f"{values.shape}, expected {(rows, k)}")
    return {"indices": indices, "values": values,
            "stats": {**manifest["stats"], "cached": True}}


def _encode_to_cache(sae, features: Any, row_ids: np.ndarray, k: int, *,
                     device: torch.device, batch_size: int, target: Path,
                     identity: dict[str, Any], progress_stage: str) -> dict[str, Any]:
    """Compute into ``.partial`` maps, fsync, then publish the manifest.

    The manifest is written last and is the only completion marker, so an
    interrupted encode is re-done rather than half-read. No content digest is
    stored: hashing 31 GiB of codes costs more than recomputing them, and the
    manifest already pins the identity, shape and row count.
    """
    rows = len(row_ids)
    target.mkdir(parents=True, exist_ok=True)
    require_disk(target, rows * k * 8, what=f"K={k} code cache")
    partial_indices = target / "indices.npy.partial"
    partial_values = target / "values.npy.partial"
    indices = np.lib.format.open_memmap(partial_indices, mode="w+",
                                        dtype=np.int32, shape=(rows, k))
    values = np.lib.format.open_memmap(partial_values, mode="w+",
                                       dtype=np.float32, shape=(rows, k))
    result = _encode(sae, features, row_ids, k, device=device, batch_size=batch_size,
                     writer=(indices, values), progress_stage=progress_stage)
    indices.flush()
    values.flush()
    # _encode returns the same writable maps. Close every reference before
    # renaming; Windows otherwise refuses publication of a successful encode.
    result.pop("indices")
    result.pop("values")
    indices._mmap.close()
    values._mmap.close()
    del indices, values
    for path in (partial_indices, partial_values):
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    os.replace(partial_indices, target / "indices.npy")
    os.replace(partial_values, target / "values.npy")
    atomic_json(target / "manifest.json",
                {"complete": True, "identity": identity, "k": k, "rows": rows,
                 "dtypes": {"indices": "int32", "values": "float32"},
                 "stats": result["stats"]})
    return {"indices": np.load(target / "indices.npy", mmap_mode="r", allow_pickle=False),
            "values": np.load(target / "values.npy", mmap_mode="r", allow_pickle=False),
            "stats": result["stats"]}


def encode_dense(features: Any, row_ids: np.ndarray | None, *, device: torch.device | str,
                 batch_size: int = 8192, to_device: bool = False) -> Any:
    """Gather the raw backbone rows for the dense reference measurement.

    No autoencoder is involved: this is the "Full Rep." row of CSR Table 4. The
    rows are returned un-normalised -- :func:`~mpsae.evaluate.knn.dense_top1`
    normalises them, so the normalisation happens in exactly one place.

    By default the result is a host float32 array, because the FF2048 gallery is
    10.5 GiB and must not silently occupy VRAM that the search needs. Pass
    ``to_device=True`` to keep it on the GPU; that raises if it does not fit.
    """
    device = torch.device(device)
    row_ids = _all_rows(features) if row_ids is None else row_ids
    row_ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    if not len(row_ids):
        raise ValueError("refusing to gather an empty row selection")
    dim = int(features.shape[1])
    rows = len(row_ids)
    if to_device:
        if device.type != "cuda":
            raise ValueError("to_device requires a CUDA device")
        free, _ = torch.cuda.mem_get_info(device)
        need = rows * dim * 4
        if need + RESIDENT_RESERVE_BYTES > free:
            raise RuntimeError(
                f"dense gather needs {human_bytes(need)} on {device} with "
                f"{human_bytes(free)} free; call with to_device=False (the search "
                f"streams host rows) or reduce the row selection")
        out: Any = torch.empty((rows, dim), dtype=torch.float32, device=device)
    else:
        out = np.empty((rows, dim), dtype=np.float32)
    progress = Progress(rows, stage="encode", what="dense rows")
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        wanted = row_ids[start:stop]
        if wanted[-1] - wanted[0] == len(wanted) - 1:
            block = np.array(features[wanted[0]:wanted[-1] + 1], copy=True)
        else:
            block = np.ascontiguousarray(features[wanted])
        if isinstance(out, torch.Tensor):
            out[start:stop].copy_(torch.from_numpy(block.astype(np.float32, copy=False)))
        else:
            out[start:stop] = block
        progress.advance(stop - start)
    return out
