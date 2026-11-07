"""Exact ranked retrieval for frozen downstream representations.

This module leaves the ImageNet ``knn`` protocol unchanged.  New metrics use
fractions in [0, 1], with ``units="fraction"`` in every result.  Recall@r is
the metric-learning hit rate: at least one relevant result in the first r.
AP@R divides by *all* R relevant gallery images and retrieves at least R
neighbors; it is never approximated by AP over an arbitrary top-100 list.

Imports of NumPy/PyTorch are local so the metric definition can be inspected
and tested without a GPU or the numerical dependencies installed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from numbers import Integral
import time
from typing import Any, Sequence

PROTOCOL_VERSION = "frozen-ranked-retrieval-v1"
BACKEND = "blocked_fp32_ranked_gemm"
TIE_RULE = "score_desc_gallery_position_asc"


def _values(values: Any, *, name: str) -> list[Any]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence")
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _labels(values: Any, *, name: str) -> list[str | int]:
    result = _values(values, name=name)
    clean: list[str | int] = []
    for value in result:
        if isinstance(value, bool):
            raise TypeError(f"{name} must contain integer or string relevance IDs")
        if isinstance(value, Integral):
            clean.append(int(value))
        elif isinstance(value, str) and value.strip():
            clean.append(value)
        else:
            raise TypeError(f"{name} must contain nonempty string or integer relevance IDs")
    return clean


def _ids(values: Any, rows: int, *, name: str) -> list[str] | None:
    if values is None:
        return None
    supplied = _labels(values, name=name)
    result = [str(value) for value in supplied]
    if len(result) != rows:
        raise ValueError(f"{name}: {len(result)} IDs for {rows} rows")
    if len(set(result)) != rows:
        raise ValueError(f"{name} must be unique within the split")
    return result


def _ranks(ranks: Sequence[int]) -> tuple[int, ...]:
    supplied = tuple(ranks)
    if not supplied or any(isinstance(r, bool) or not isinstance(r, Integral) or r < 1
                           for r in supplied):
        raise ValueError("ranks must contain positive integers")
    return tuple(sorted({int(r) for r in supplied}))


def _protocol(gallery_labels, query_labels, *, gallery_ids=None, query_ids=None,
              exclude_self: bool = False):
    gallery = _labels(gallery_labels, name="gallery_labels")
    queries = _labels(query_labels, name="query_labels")
    gids = _ids(gallery_ids, len(gallery), name="gallery_ids")
    qids = _ids(query_ids, len(queries), name="query_ids")
    if exclude_self and (gids is None or qids is None):
        raise ValueError("self exclusion requires both gallery_ids and query_ids")
    lookup = {value: index for index, value in enumerate(gids or [])}
    self_positions = [lookup.get(value, -1) for value in qids] if exclude_self else [-1] * len(queries)
    counts = Counter(gallery)
    positives, available = [], []
    for label, position in zip(queries, self_positions):
        if position >= 0 and gallery[position] != label:
            raise ValueError("the same sample ID has different query and gallery relevance labels")
        positives.append(counts[label] - int(position >= 0))
        available.append(len(gallery) - int(position >= 0))
    return gallery, queries, gids, qids, self_positions, positives, available


def summarize_ranking(positions: Any, gallery_labels: Any, query_labels: Any, *,
                      ranks: Sequence[int] = (1, 10, 100), gallery_ids=None,
                      query_ids=None, exclude_self: bool = False,
                      empty_queries: Sequence[bool] | None = None,
                      no_positive_policy: str = "error") -> dict[str, Any]:
    """Score a complete-enough ranking, without importing torch or NumPy.

    ``positions`` is [queries, depth] in score order; -1 may pad missing
    candidates.  Precision@10 divides by min(10, available gallery), documented
    explicitly for tiny smoke datasets.  The default rejects queries without
    positives.  ``no_positive_policy="exclude"`` is an explicit alternative:
    excluded counts are reported and the denominator is ``scored_queries``.
    Empty codes remain in that denominator and receive zero on every metric.
    """
    ranks = _ranks(ranks)
    if no_positive_policy not in ("error", "exclude"):
        raise ValueError("no_positive_policy must be 'error' or 'exclude'")
    gallery, queries, _, _, self_positions, positives, available = _protocol(
        gallery_labels, query_labels, gallery_ids=gallery_ids, query_ids=query_ids,
        exclude_self=exclude_self)
    if len(positions) != len(queries):
        raise ValueError("one ranking is required per query")
    empty = [False] * len(queries) if empty_queries is None else list(empty_queries)
    if len(empty) != len(queries):
        raise ValueError("empty_queries must contain one flag per query")
    no_positive = sum(count == 0 for count in positives)
    if no_positive and no_positive_policy == "error":
        raise ValueError(f"{no_positive} query/queries have no relevant gallery image after self exclusion")

    scores: dict[str, list[float]] = defaultdict(list)
    by_class: dict[str | int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for query, label in enumerate(queries):
        if positives[query] == 0:
            continue
        ranked: list[int] = []
        for raw_position in positions[query]:
            if isinstance(raw_position, bool) or not isinstance(raw_position, Integral):
                raise TypeError("ranked gallery positions must be integers")
            position = int(raw_position)
            if position == -1:
                continue
            if not 0 <= position < len(gallery):
                raise ValueError("ranked position outside gallery")
            if position == self_positions[query]:
                raise ValueError("ranking still contains the query's own sample ID")
            ranked.append(position)
        if len(set(ranked)) != len(ranked):
            raise ValueError("a ranking contains duplicate gallery positions")
        needed = min(max(10, max(ranks), positives[query]), available[query])
        if len(ranked) < needed:
            raise ValueError(f"ranking depth {len(ranked)} is too short; need {needed} for exact metrics")
        relevant = [gallery[position] == label for position in ranked]
        values: dict[str, float] = {}
        for rank in ranks:
            values[f"recall_at_{rank}"] = float(any(relevant[:rank]))
        precision_depth = min(10, available[query])
        values["precision_at_10"] = sum(relevant[:precision_depth]) / precision_depth
        hits, precision_sum = 0, 0.0
        for rank, hit in enumerate(relevant[:positives[query]], 1):
            hits += int(hit)
            if hit:
                precision_sum += hits / rank
        values["map_at_r"] = precision_sum / positives[query]
        for field, value in values.items():
            value = 0.0 if empty[query] else value
            scores[field].append(value)
            by_class[label][field].append(value)
    scored = len(queries) - no_positive
    if not scored:
        raise ValueError("no query has a relevant gallery image")
    result: dict[str, Any] = {field: sum(values) / scored for field, values in scores.items()}
    for field in scores:
        result[f"macro_{field}"] = sum(sum(group[field]) / len(group[field])
                                           for group in by_class.values()) / len(by_class)
    result.update(units="fraction", queries=len(queries), scored_queries=scored,
                  gallery_rows=len(gallery), relevance_classes=len(by_class),
                  no_positive_queries=no_positive, no_positive_policy=no_positive_policy,
                  empty_queries=sum(bool(flag) for flag in empty),
                  empty_scored_queries=sum(bool(flag) and count > 0 for flag, count in zip(empty, positives)),
                  empty_query_policy="counted_and_scored_as_miss",
                  exclude_self=bool(exclude_self),
                  precision_at_10_denominator="min(10, available_gallery_for_query)",
                  relevant_per_query_min=min(positives), relevant_per_query_max=max(positives))
    return result


def exact_topk(gallery: Any, queries: Any, *, dim: int, device,
               depth: int, gallery_ids=None, query_ids=None, exclude_self: bool = False,
               gallery_chunk: int = 4096, query_block: int = 512,
               normalize: bool = False, progress_stage: str = "ranked retrieval"):
    """Return (scores, positions, seconds) as CPU NumPy arrays and a float.

    Gallery positions break equal-score ties, including ties at the truncation
    boundary. Stable sorts are intentional: plain ``topk`` can choose arbitrary
    members of a large zero-score tie and change sparse retrieval metrics.
    Queries and the retained top-depth candidates are resident; full pairwise
    scores are never allocated. Only the gallery streaming/search loop is timed,
    matching the existing ImageNet search scope.
    """
    import numpy as np
    import torch

    from .encode import CodeSource
    from .knn import fp32_matmul, require_vram
    from ..logs import Progress

    if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if gallery_chunk < 1 or query_block < 1:
        raise ValueError("gallery_chunk and query_block must be positive")
    device = torch.device(device)
    gs = CodeSource(gallery, dim=dim, device=device, normalize=normalize)
    qs = CodeSource(queries, dim=dim, device=device, normalize=normalize)
    if not gs.rows or not qs.rows:
        raise ValueError("gallery and query splits must be nonempty")
    gids = _ids(gallery_ids, gs.rows, name="gallery_ids")
    qids = _ids(query_ids, qs.rows, name="query_ids")
    if exclude_self and (gids is None or qids is None):
        raise ValueError("self exclusion requires both gallery_ids and query_ids")
    lookup = {value: index for index, value in enumerate(gids or [])}
    self_positions = np.array([lookup.get(value, -1) for value in qids], dtype=np.int64) \
        if exclude_self else np.full(qs.rows, -1, dtype=np.int64)
    depth = min(int(depth), gs.rows)
    gallery_chunk, query_block = min(gallery_chunk, gs.rows), min(query_block, qs.rows)
    require_vram(device,
                 qs.rows * dim * 4 + gallery_chunk * dim * 4
                 + qs.rows * depth * 12 + query_block * gallery_chunk * 24
                 + query_block * depth * 48,
                 what="exact ranked retrieval working set",
                 knobs="gallery_chunk, query_block, or ranking depth")
    dense_queries = torch.empty((qs.rows, dim), dtype=torch.float32, device=device)
    with torch.no_grad():
        for first in range(0, qs.rows, query_block):
            last = min(first + query_block, qs.rows)
            block = qs.dense(first, last)
            if not bool(torch.isfinite(block).all()):
                raise ValueError("nonfinite query representation")
            dense_queries[first:last] = block
    best = torch.full((qs.rows, depth), -torch.inf, dtype=torch.float32, device=device)
    positions = torch.full((qs.rows, depth), -1, dtype=torch.int64, device=device)
    excluded = torch.as_tensor(self_positions, device=device)
    progress = Progress(gs.rows, stage=progress_stage, what="gallery rows")
    with torch.no_grad(), fp32_matmul(), torch.autocast(device_type=device.type, enabled=False):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for start in range(0, gs.rows, gallery_chunk):
            stop = min(start + gallery_chunk, gs.rows)
            chunk = gs.dense(start, stop)
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("nonfinite gallery representation")
            for first in range(0, qs.rows, query_block):
                last = min(first + query_block, qs.rows)
                score = dense_queries[first:last] @ chunk.T
                if not bool(torch.isfinite(score).all()):
                    raise FloatingPointError("nonfinite retrieval score")
                if exclude_self:
                    self_rows = excluded[first:last]
                    mask = (self_rows >= start) & (self_rows < stop)
                    rows = torch.arange(last - first, device=device)[mask]
                    score[rows, self_rows[mask] - start] = -torch.inf
                order = torch.argsort(score, dim=1, descending=True, stable=True)[:, :depth]
                values = score.gather(1, order)
                local_positions = order + start
                # Every retained prior position precedes every position in this
                # chunk, so stable score sorting also preserves global tie order.
                candidates = torch.cat((best[first:last], values), dim=1)
                candidate_positions = torch.cat((positions[first:last], local_positions), dim=1)
                keep = torch.argsort(candidates, dim=1, descending=True, stable=True)[:, :depth]
                best[first:last] = candidates.gather(1, keep)
                positions[first:last] = candidate_positions.gather(1, keep)
            progress.advance(stop - start)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
    positions.masked_fill_(~torch.isfinite(best), -1)
    return best.cpu().numpy(), positions.cpu().numpy(), seconds


def evaluate_retrieval(gallery: Any, gallery_labels: Any, queries: Any,
                       query_labels: Any, *, dim: int, device, gallery_ids=None,
                       query_ids=None, exclude_self: bool = False,
                       ranks: Sequence[int] = (1, 10, 100), gallery_chunk: int = 4096,
                       query_block: int = 512, normalize: bool = False,
                       no_positive_policy: str = "error") -> dict[str, Any]:
    """Exact frozen retrieval with sample-ID exclusion and complete AP@R depth."""
    import torch

    from .encode import CodeSource

    ranks = _ranks(ranks)
    labels, truth, gids, qids, _, positives, _ = _protocol(
        gallery_labels, query_labels, gallery_ids=gallery_ids, query_ids=query_ids,
        exclude_self=exclude_self)
    if no_positive_policy not in ("error", "exclude"):
        raise ValueError("no_positive_policy must be 'error' or 'exclude'")
    if no_positive_policy == "error" and any(count == 0 for count in positives):
        raise ValueError("query split has no-positive queries after self exclusion")
    if not any(count > 0 for count in positives):
        raise ValueError("no query has a relevant gallery image")
    gs = CodeSource(gallery, dim=dim, device=device)
    qs = CodeSource(queries, dim=dim, device=device)
    if gs.rows != len(labels) or qs.rows != len(truth):
        raise ValueError("representation row counts disagree with relevance labels")
    depth = min(len(labels), max(10, max(ranks), max(positives)))
    _, positions, seconds = exact_topk(
        gallery, queries, dim=dim, device=device, depth=depth, gallery_ids=gids,
        query_ids=qids, exclude_self=exclude_self, gallery_chunk=gallery_chunk,
        query_block=query_block, normalize=normalize)
    empty: list[bool] = []
    with torch.no_grad():
        for first in range(0, qs.rows, query_block):
            last = min(first + query_block, qs.rows)
            values = qs.compact(first, last)[1] if qs.is_compact else qs.dense(first, last)
            # Dense backbone features may be signed; all-negative is not empty.
            empty.extend(values.ne(0).any(dim=1).logical_not().cpu().tolist())
    result = summarize_ranking(
        positions, labels, truth, ranks=ranks, gallery_ids=gids, query_ids=qids,
        exclude_self=exclude_self, empty_queries=empty,
        no_positive_policy=no_positive_policy)
    result.update(protocol_version=PROTOCOL_VERSION, backend=BACKEND, seconds=seconds,
                  depth=depth, search_precision="float32_tf32_disabled", tie_rule=TIE_RULE,
                  normalization="l2_during_search" if normalize else "as_supplied",
                  timing_scope="gallery_streaming_and_ranking;excludes_query_upload_and_metrics")
    return result
