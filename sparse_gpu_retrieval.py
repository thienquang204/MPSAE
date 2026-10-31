"""Exhaustive cosine retrieval over CUDA-resident nonnegative sparse codes.

The gallery stores feature posting lists, not dense dictionary-width vectors.
Only products between an active query feature and its gallery postings are
computed. A bounded, dense *score* workspace is necessary for exact top-1
selection, including gallery rows having no features in common with a query.

Construction, validation, extension compilation, normalization, and allocation
belong outside retrieval timing. ``search`` does no host synchronization and
uses the current PyTorch CUDA stream. The caller must validate and normalize
queries before timing. Atomic float32 sums are exhaustive but are not bitwise
deterministic; floating-point roundoff can affect almost-equal scores.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


_EXTENSION: Optional[Any] = None


def preload_sparse_retrieval(device: torch.device | str) -> None:
    """Validate the CUDA toolchain before a caller starts expensive training.

    Import alone remains side-effect free. Pipeline preflight may explicitly
    call this function; compilation/load is cached for later index preparation.
    """
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Sparse retrieval preflight requires an available CUDA device.")
    with torch.cuda.device(target):
        _load_extension()


def check_cuda_memory_budget(
    device: torch.device,
    additional_bytes: int,
    memory_fraction: float,
    operation: str,
) -> None:
    """Best-effort capacity guard, including this process's reusable cache.

    This is setup-only: allocation by another process can invalidate the
    estimate. It neither evicts tensors nor switches the retrieval backend.
    """
    if additional_bytes < 0 or not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError("Memory requests must be nonnegative and memory_fraction in (0, 1].")
    with torch.cuda.device(device):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if getattr(torch.cuda.get_device_properties(device), "is_integrated", False):
            # Unified-memory GPUs (e.g. GB10) report only unused host RAM as
            # free; reclaimable page cache is still available to allocations.
            free_bytes = max(free_bytes, _host_available_bytes())
        cached_bytes = max(
            0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        )
    headroom = math.ceil(total_bytes * (1.0 - memory_fraction))
    usable_bytes = max(0, free_bytes + cached_bytes - headroom)
    if additional_bytes > usable_bytes:
        raise RuntimeError(
            f"Insufficient resident GPU memory for {operation}: estimated "
            f"additional {additional_bytes / 2**20:.1f} MiB, available within "
            f"memory_fraction={memory_fraction:g}: {usable_bytes / 2**20:.1f} MiB. "
            "Reduce batch/chunk sizes for temporary buffers or use a GPU with "
            "enough memory for the full index. No retrieval backend was changed."
        )


def _host_available_bytes() -> int:
    """Linux MemAvailable (free plus reclaimable cache), or 0 if unreadable."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _load_extension() -> Any:
    """Compile/load once, explicitly before a timed retrieval invocation."""
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        source_directory = Path(__file__).resolve().parent / "csrc"
        try:
            _EXTENSION = load(
                name="graduate_thesis_sparse_retrieval",
                sources=[
                    str(source_directory / "sparse_retrieval.cpp"),
                    str(source_directory / "sparse_retrieval.cu"),
                ],
                extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
                extra_cuda_cflags=["-O3"],
                with_cuda=True,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "GPU sparse retrieval requires the bundled CUDA extension. "
                "Use a CUDA development image with nvcc, a C++ compiler, ninja, "
                "and a CUDA-compatible PyTorch installation. No dense or CPU "
                "fallback is used. Compilation happens before retrieval timing."
            ) from exc
    return _EXTENSION


@dataclass(frozen=True)
class SparseGalleryChunk:
    """Compressed posting lists for a consecutive interval of gallery rows."""

    row_count: int
    row_offset: int
    feature_offsets: torch.Tensor
    doc_indices: torch.Tensor
    values: torch.Tensor

    @property
    def nnz(self) -> int:
        return self.values.numel()

    @property
    def payload_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.feature_offsets, self.doc_indices, self.values)
        )


class CudaSparseIndex:
    """Feature-inverted exact top-1 index with the entire gallery on one GPU.

    ``add`` accepts unique feature indices within each row, as produced by
    ``torch.topk``. Values must be finite, nonnegative, and L2-normalized by the
    caller (zero rows stay zero). Zero-valued padding is removed from postings.
    Gallery row IDs are assigned in insertion order.

    ``memory_fraction`` leaves that fraction's complement of *total* GPU memory
    as headroom, accounting for other processes and reusable PyTorch cache.
    It is a guard, not a guarantee against concurrent allocations by other
    processes. Index construction never streams postings from CPU memory.
    """

    storage = "cuda_feature_inverted_index"

    def __init__(
        self,
        hidden_dim: int,
        device: torch.device | str,
        max_score_buffer_bytes: int = 256 * 1024 * 1024,
        memory_fraction: float = 0.90,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("Sparse GPU retrieval requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; sparse retrieval has no CPU fallback.")
        self.device = torch.device(
            "cuda", target.index if target.index is not None else torch.cuda.current_device()
        )
        self.max_score_buffer_bytes = int(max_score_buffer_bytes)
        self.memory_fraction = float(memory_fraction)
        if not 0 < self.hidden_dim <= torch.iinfo(torch.int32).max:
            raise ValueError("hidden_dim must fit a positive int32 feature index.")
        if self.max_score_buffer_bytes < 4:
            raise ValueError("max_score_buffer_bytes must hold at least one float32 score.")
        if not math.isfinite(self.memory_fraction) or not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1].")
        self.chunks: list[SparseGalleryChunk] = []
        self.row_count = 0
        self.nnz = 0
        self._check_capacity(self.hidden_dim * 8, "aggregate feature posting counts")
        self.feature_counts = torch.zeros(
            self.hidden_dim, device=self.device, dtype=torch.int64
        )
        self.payload_bytes = self.feature_counts.numel() * self.feature_counts.element_size()
        self.max_chunk_rows = 0
        self.workspace_bytes = 0
        self._score_workspace: Optional[torch.Tensor] = None
        self._prepared_batch_size = 0
        self._extension: Optional[Any] = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def _check_capacity(self, additional_bytes: int, operation: str) -> None:
        check_cuda_memory_budget(
            self.device, additional_bytes, self.memory_fraction, operation
        )

    def _check_compact_layout(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        if indices.ndim != 2 or values.ndim != 2 or indices.shape != values.shape:
            raise ValueError("Compact indices and values must have the same [rows, K] shape.")
        if indices.device != self.device or values.device != self.device:
            raise ValueError("Compact indices and values must reside on the index CUDA device.")
        if indices.dtype != torch.int32 or values.dtype != torch.float32:
            raise TypeError("Compact feature indices must be int32 and values must be float32.")
        if not indices.is_contiguous() or not values.is_contiguous():
            raise ValueError("Compact indices and values must be contiguous before retrieval.")
        if not 0 < indices.shape[1] <= self.hidden_dim:
            raise ValueError("Compact K must be between one and hidden_dim.")

    @torch.no_grad()
    def add(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        """Build a GPU posting-list chunk; all synchronization is setup-only.

        Construction uses O(rows*K + hidden_dim) temporary storage. The
        conservative workspace guard includes sorting scratch; its estimate is
        deliberately larger than the final 8 bytes per nonzero posting.
        """
        if self._score_workspace is not None:
            raise RuntimeError("Add all gallery chunks before calling prepare().")
        self._check_compact_layout(indices, values)
        rows, width = indices.shape
        if rows == 0:
            return
        if rows > torch.iinfo(torch.int32).max:
            raise ValueError("Each gallery chunk must fit int32 local document IDs.")
        if rows * 4 > self.max_score_buffer_bytes:
            raise ValueError("A gallery chunk exceeds the score buffer cap even for one query.")
        potential_entries = rows * width
        self._check_capacity(
            96 * potential_entries + 24 * (self.hidden_dim + 1),
            "sparse gallery construction (including temporary sorting storage)",
        )
        # .item() synchronizations are intentionally restricted to construction.
        if not bool(torch.isfinite(values).all().item()) or bool((values < 0).any().item()):
            raise ValueError("Sparse cosine codes must contain finite, nonnegative values.")
        if bool(((indices < 0) | (indices >= self.hidden_dim)).any().item()):
            raise ValueError("A compact gallery feature index is outside the dictionary.")
        try:
            flat_values = values.reshape(-1)
            keep = flat_values != 0
            features = indices.reshape(-1)[keep].to(dtype=torch.int64)
            documents = (
                torch.arange(rows, device=self.device, dtype=torch.int32)
                .unsqueeze(1)
                .expand(rows, width)
                .reshape(-1)[keep]
            )
            kept_values = flat_values[keep]
            # A feature-major, document-minor order makes accesses to each
            # posting list and its score destinations monotonic. The int32
            # feature/row limits guarantee these int64 keys cannot overflow.
            order = torch.argsort(features * rows + documents)
            posting_documents = documents.index_select(0, order).contiguous()
            posting_values = kept_values.index_select(0, order).contiguous()
            counts = torch.bincount(features, minlength=self.hidden_dim)
            offsets = torch.empty(self.hidden_dim + 1, device=self.device, dtype=torch.int64)
            offsets[0] = 0
            torch.cumsum(counts, dim=0, out=offsets[1:])
            chunk = SparseGalleryChunk(
                row_count=rows,
                row_offset=self.row_count,
                feature_offsets=offsets,
                doc_indices=posting_documents,
                values=posting_values,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "GPU memory was exhausted while building resident sparse postings. "
                "Reduce the gallery/index-build chunk size; the complete final "
                "index must still fit on the GPU. No fallback was performed."
            ) from exc
        self.chunks.append(chunk)
        self.feature_counts.add_(counts)
        self.row_count += rows
        self.nnz += chunk.nnz
        self.payload_bytes += chunk.payload_bytes
        self.max_chunk_rows = max(self.max_chunk_rows, rows)

    @torch.no_grad()
    def query_product_count(
        self, query_indices: torch.Tensor, query_values: torch.Tensor
    ) -> int:
        """Count actual posting products for a batch, strictly outside timing.

        Each positive query coordinate visits its feature's entire posting
        list, across all resident gallery chunks. This int64 sum measures
        executed query-value/posting-value multiplications, rather than a
        dense rows*hidden_dim proxy. ``.item()`` synchronizes intentionally;
        callers must invoke this during query preparation, never search.
        Queries must already satisfy the same data contract as ``search``.
        """
        self._check_compact_layout(query_indices, query_values)
        counts = self.feature_counts[query_indices.to(dtype=torch.int64)]
        return int(
            torch.where(query_values > 0, counts, 0).sum(dtype=torch.int64).item()
        )

    @torch.no_grad()
    def prepare(self, batch_size: int) -> None:
        """Compile the CUDA extension and allocate the score workspace outside timing."""
        if self.row_count == 0:
            raise ValueError("Cannot retrieve from an empty gallery.")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Prepared query batch_size must be positive.")
        required_bytes = batch_size * self.max_chunk_rows * 4
        if required_bytes > self.max_score_buffer_bytes:
            raise ValueError(
                f"Query batch {batch_size} and gallery chunk {self.max_chunk_rows} "
                f"need {required_bytes} score-buffer bytes, exceeding the configured "
                f"cap of {self.max_score_buffer_bytes}. Reduce either chunk or batch size."
            )
        with torch.cuda.device(self.device):
            self._extension = _load_extension()
        # Small outputs and reduction scratch also need room. The reduction
        # implementation belongs to PyTorch and can vary across versions.
        self._check_capacity(required_bytes + batch_size * 256 + 2**20, "retrieval workspace")
        try:
            self._score_workspace = torch.empty(
                (batch_size, self.max_chunk_rows), device=self.device, dtype=torch.float32
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "GPU score workspace allocation failed. Reduce query batch size "
                "or gallery chunk size; the index remains GPU-resident."
            ) from exc
        self._prepared_batch_size = batch_size
        self.workspace_bytes = required_bytes

    @torch.no_grad()
    def search(
        self, query_indices: torch.Tensor, query_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float32 cosine scores and int64 global gallery IDs on CUDA.

        Queries must have unique in-range indices, finite nonnegative values,
        and precomputed L2 normalization. Data validation must precede timing;
        this method only checks tensor metadata, without a host synchronization.
        The same index cannot be searched concurrently because it owns a shared
        score buffer. Keep calls on one stream (or synchronize between streams).
        """
        self._check_compact_layout(query_indices, query_values)
        if self._score_workspace is None or self._extension is None:
            raise RuntimeError("Call prepare(batch_size) before retrieval timing.")
        batch_size = query_indices.shape[0]
        if batch_size > self._prepared_batch_size:
            raise ValueError("Query batch exceeds the prepared score workspace.")
        if batch_size == 0:
            return (
                torch.empty(0, device=self.device, dtype=torch.float32),
                torch.empty(0, device=self.device, dtype=torch.int64),
            )
        best_scores: Optional[torch.Tensor] = None
        best_ids: Optional[torch.Tensor] = None
        for chunk in self.chunks:
            scores = self._score_workspace[:batch_size, : chunk.row_count]
            scores.zero_()
            self._extension.accumulate(
                query_indices,
                query_values,
                chunk.feature_offsets,
                chunk.doc_indices,
                chunk.values,
                scores,
            )
            chunk_scores, local_ids = torch.max(scores, dim=1)
            global_ids = local_ids + chunk.row_offset
            if best_scores is None:
                best_scores, best_ids = chunk_scores, global_ids
            else:
                # torch.max chooses the first column on ties. Strict comparison
                # keeps the earlier chunk, yielding the smallest global row ID.
                # For zero query/zero overlap, every score is 0 and row 0 wins.
                improves = chunk_scores > best_scores
                best_scores = torch.where(improves, chunk_scores, best_scores)
                best_ids = torch.where(improves, global_ids, best_ids)
        assert best_scores is not None and best_ids is not None
        return best_scores, best_ids
