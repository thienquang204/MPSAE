"""Search-only CUDA measurements. Importing this module never compiles kernels.

Each resident query batch is measured repeatedly before advancing to the next
batch. Repeat totals are paired sums of those interleaved measurements, not
independent end-to-end passes. CUDA events measure the search stream interval
(including GPU idle gaps between launches); wall time also includes dispatch
and completion waiting. Neither metric is single-query latency when B > 1.
"""

from __future__ import annotations

import statistics
import math
import time
from typing import Any, Callable, Hashable

import torch


TIMING_SCOPE = (
    "resident_gpu_search_including_score_reset_products_top1_and_chunk_merge;"
    "excludes_encoding_normalization_transfers_index_build_compilation_and_accuracy"
)


class CudaSearchTimer:
    def __init__(self, device: torch.device, warmup: int, repeats: int) -> None:
        if device.type != "cuda" or warmup < 1 or repeats < 1:
            raise ValueError("CUDA timing requires a GPU, positive warmup and repeats")
        self.device = device
        self.warmup = warmup
        self.repeats = repeats
        self.stream = torch.cuda.current_stream(device)
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        # Events lazily initialize on first record: pay that cost before timing.
        self.start.record(self.stream)
        self.stop.record(self.stream)
        self.stop.synchronize()
        self.warmed_shapes: set[Hashable] = set()
        self.records: list[dict[str, Any]] = []
        self.gpu_totals = [0.0] * repeats
        self.wall_totals = [0.0] * repeats

    def measure(
        self, search: Callable[[], Any], shape: Hashable, query_count: int
    ) -> Any:
        """Return a separate untimed accuracy result, never a selected fast repeat.

        The callback must enqueue all search work on self.stream and must not
        transfer data to/from the host or synchronize. Tensor metadata checks
        are allowed. Data validation and query transfer precede this method.
        """
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("Retrieval stream changed after timer construction")
        if shape not in self.warmed_shapes:
            for _ in range(self.warmup):
                search()
            torch.cuda.synchronize(self.device)
            self.warmed_shapes.add(shape)
        gpu_samples: list[float] = []
        wall_samples: list[float] = []
        for repeat in range(self.repeats):
            # Drain previous work before the wall clock starts. Wait on the end
            # event inside the interval: asynchronous enqueue time is not latency.
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            self.start.record(self.stream)
            output = search()
            self.stop.record(self.stream)
            self.stop.synchronize()
            wall_seconds = time.perf_counter() - started
            gpu_seconds = self.start.elapsed_time(self.stop) / 1000.0
            gpu_samples.append(gpu_seconds)
            wall_samples.append(wall_seconds)
            self.gpu_totals[repeat] += gpu_seconds
            self.wall_totals[repeat] += wall_seconds
            del output
        self.records.append({
            "query_count": query_count,
            "cuda_seconds": gpu_samples,
            "wall_seconds": wall_samples,
        })
        # Accuracy is evaluated exactly once per batch, separately from timing.
        # Atomic FP32 summation can still perturb essentially tied neighbors.
        return search()

    def summary(self, query_count: int) -> dict[str, Any]:
        if query_count <= 0 or not self.records:
            raise ValueError("Cannot summarize an empty query set")
        if sum(row["query_count"] for row in self.records) != query_count:
            raise ValueError("Timed query count disagrees with evaluated query count")
        gpu_seconds = statistics.median(self.gpu_totals)
        wall_seconds = statistics.median(self.wall_totals)
        if not all(math.isfinite(value) and value > 0 for value in (gpu_seconds, wall_seconds)):
            raise RuntimeError("Search timer returned a non-positive elapsed time")
        batches = len(self.records)
        return {
            # Preserve established result names, with the timer version explicit.
            "retrieval_search_seconds": gpu_seconds,
            "retrieval_wall_seconds": wall_seconds,
            "retrieval_query_batches": batches,
            "mean_retrieval_seconds_per_query": gpu_seconds / query_count,
            "mean_retrieval_milliseconds_per_query": 1000 * gpu_seconds / query_count,
            "mean_retrieval_wall_milliseconds_per_query": 1000 * wall_seconds / query_count,
            "mean_retrieval_milliseconds_per_batch": 1000 * gpu_seconds / batches,
            "retrieval_queries_per_second": query_count / gpu_seconds,
            "retrieval_wall_queries_per_second": query_count / wall_seconds,
            "timing": {
                "version": 2,
                "scope": TIMING_SCOPE,
                "primary_statistic": "median_of_interleaved_repeat_totals",
                "query_time_semantics": "amortized_batch_time_not_single_query_latency",
                "cuda_timer": "events_on_search_stream_including_launch_gaps",
                "wall_timer": "perf_counter_through_end_event_synchronization",
                "cache_policy": "warm_repeated_search_of_each_resident_query_batch",
                "warmup_per_shape": self.warmup,
                "warmup_shapes": [list(shape) for shape in sorted(self.warmed_shapes)],
                "repeats_per_batch": self.repeats,
                "timed_search_calls": self.repeats * batches,
                "accuracy_search_calls": batches,
                "cuda_repeat_total_seconds": self.gpu_totals,
                "wall_repeat_total_seconds": self.wall_totals,
                "cuda_repeat_total_min_seconds": min(self.gpu_totals),
                "cuda_repeat_total_max_seconds": max(self.gpu_totals),
                "wall_repeat_total_min_seconds": min(self.wall_totals),
                "wall_repeat_total_max_seconds": max(self.wall_totals),
                "batches": self.records,
            },
        }
