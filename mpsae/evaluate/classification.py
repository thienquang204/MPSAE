"""DINO weighted kNN on frozen (optionally compact sparse) representations.

Voting follows facebookresearch/dino/eval_knn.py: sum exp(cosine / T) for
each neighbour's class, then rank classes. One search supplies all requested
neighbour counts. Subtracting each query's largest similarity before exp is
a common positive rescaling of every class score and avoids overflow.

Explicit sparse-evaluation conventions extend the original script:
equal neighbour scores use gallery position, equal class votes use ascending
label, and an empty query code abstains and is scored as incorrect. These
rules prevent arbitrary zero-code predictions from becoming apparent signal.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
import time
from typing import Any, Sequence

from .ranking import BACKEND, TIE_RULE, _ids, _labels, exact_topk

PROTOCOL_VERSION = "dino-weighted-knn-v1"
VOTE_RULE = "sum_exp_cosine_over_temperature"
CLASS_TIE_RULE = "vote_desc_label_asc"
EMPTY_QUERY_POLICY = "abstain_counted_and_scored_as_miss"
REFERENCE = "https://github.com/facebookresearch/dino/blob/main/eval_knn.py"


def validate_knn(n_neighbors: Sequence[int], temperature: float) -> tuple[tuple[int, ...], float]:
    """Validate protocol knobs without importing the numerical stack."""
    if not isinstance(n_neighbors, (list, tuple)) or not n_neighbors:
        raise ValueError("n_neighbors must be a nonempty list or tuple")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in n_neighbors):
        raise ValueError("n_neighbors must contain positive integers")
    if len(set(n_neighbors)) != len(n_neighbors):
        raise ValueError("n_neighbors must not contain duplicates")
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature)) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    return tuple(sorted(int(n) for n in n_neighbors)), float(temperature)


def evaluate_knn(gallery: Any, gallery_labels: Any, queries: Any, query_labels: Any, *,
                 dim: int, device, gallery_ids=None, query_ids=None,
                 n_neighbors: Sequence[int] = (1, 20), temperature: float = 0.07,
                 gallery_chunk: int = 4096, query_block: int = 512,
                 normalize: bool = False) -> dict[str, Any]:
    """Classify queries using a disjoint, labelled support set.

    ``normalize=False`` expects already L2-normalized codes, as returned by
    ``encode_split``. Dense backbone embeddings use ``normalize=True``.
    Labels are used only for voting/scoring, never representation fitting.
    Metrics are fractions; top5 follows DINO's min(5, n_neighbors) convention,
    additionally capped by the available class vocabulary.
    """
    import numpy as np
    import torch

    from .encode import CodeSource

    neighbors, temperature = validate_knn(n_neighbors, temperature)
    labels = _labels(gallery_labels, name="gallery_labels")
    truth = _labels(query_labels, name="query_labels")
    if len({type(label) for label in labels + truth}) != 1:
        raise TypeError("support and query labels must all be integers or all be strings")
    classes = sorted(set(labels))
    lookup = {label: index for index, label in enumerate(classes)}
    if not set(truth) <= set(classes):
        raise ValueError("query labels must be covered by support classes")
    if neighbors[-1] > len(labels):
        raise ValueError("n_neighbors exceeds the number of support samples")
    gids = _ids(gallery_ids, len(labels), name="gallery_ids")
    qids = _ids(query_ids, len(truth), name="query_ids")
    if (gids is None) != (qids is None):
        raise ValueError("supply both support and query IDs, or neither")
    if gids is not None and set(gids) & set(qids):
        raise ValueError("support and query sample IDs overlap; classification needs disjoint splits")
    device = torch.device(device)
    gs = CodeSource(gallery, dim=dim, device=device, normalize=normalize)
    qs = CodeSource(queries, dim=dim, device=device, normalize=normalize)
    if gs.rows != len(labels) or qs.rows != len(truth):
        raise ValueError("representation row counts disagree with class labels")
    if (isinstance(query_block, bool) or not isinstance(query_block, Integral) or query_block < 1
            or isinstance(gallery_chunk, bool) or not isinstance(gallery_chunk, Integral)
            or gallery_chunk < 1):
        raise ValueError("gallery_chunk and query_block must be positive integers")
    scores, positions, seconds = exact_topk(
        gallery, queries, dim=dim, device=device, depth=neighbors[-1],
        gallery_ids=gids, query_ids=qids, gallery_chunk=gallery_chunk,
        query_block=query_block, normalize=normalize, progress_stage="DINO kNN")
    if np.any(positions < 0):
        raise RuntimeError("kNN search returned fewer support neighbours than requested")

    support_labels = torch.tensor([lookup[label] for label in labels], dtype=torch.int64, device=device)
    query_labels = torch.tensor([lookup[label] for label in truth], dtype=torch.int64, device=device)
    count = len(classes)
    true_counts = torch.bincount(query_labels, minlength=count)
    totals = {n: {"top1": torch.zeros((), dtype=torch.int64, device=device),
                  "top5": torch.zeros((), dtype=torch.int64, device=device),
                  "tp": torch.zeros(count, dtype=torch.int64, device=device),
                  "predicted": torch.zeros(count, dtype=torch.int64, device=device)} for n in neighbors}
    empty_count = torch.zeros((), dtype=torch.int64, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for first in range(0, qs.rows, query_block):
            last = min(first + query_block, qs.rows)
            similarities = torch.as_tensor(scores[first:last], dtype=torch.float32, device=device)
            indices = torch.as_tensor(positions[first:last], dtype=torch.int64, device=device)
            nearest_labels = support_labels[indices]
            # This rescaling is shared by all neighbours/classes of one query.
            weights = ((similarities - similarities[:, :1]) / temperature).exp()
            values = qs.compact(first, last)[1] if qs.is_compact else qs.dense(first, last)
            valid = values.ne(0).any(dim=1)
            empty_count += (~valid).sum()
            targets = query_labels[first:last]
            for n in neighbors:
                votes = torch.zeros((last - first, count), dtype=torch.float32, device=device)
                votes.scatter_add_(1, nearest_labels[:, :n], weights[:, :n])
                predictions = torch.argsort(votes, dim=1, descending=True, stable=True)
                predicted = predictions[:, 0]
                correct = predicted.eq(targets) & valid
                total = totals[n]
                total["top1"] += correct.sum()
                width = min(5, n, count)
                total["top5"] += (predictions[:, :width].eq(targets[:, None]).any(dim=1) & valid).sum()
                total["tp"] += torch.bincount(targets[correct], minlength=count)
                total["predicted"] += torch.bincount(predicted[valid], minlength=count)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    voting_seconds = time.perf_counter() - started

    result: dict[str, Any] = {}
    for n in neighbors:
        total = totals[n]
        tp = total["tp"].double()
        predicted_counts = total["predicted"]
        observed = true_counts > 0
        f1_denominator = true_counts + predicted_counts
        f1_classes = f1_denominator > 0
        result.update({f"knn_{n}_top1": int(total["top1"]) / qs.rows,
                       f"knn_{n}_top5": int(total["top5"]) / qs.rows,
                       f"knn_{n}_balanced_accuracy": float((tp[observed] / true_counts[observed]).mean()),
                       f"knn_{n}_macro_f1": float((2 * tp[f1_classes] / f1_denominator[f1_classes]).mean())})
    result.update(units="fraction", protocol_version=PROTOCOL_VERSION, backend=BACKEND,
                  reference=REFERENCE, n_neighbors=list(neighbors), temperature=temperature,
                  vote_rule=VOTE_RULE, tie_rule=TIE_RULE, class_tie_rule=CLASS_TIE_RULE,
                  empty_query_policy=EMPTY_QUERY_POLICY, empty_queries=int(empty_count),
                  empty_scored_queries=int(empty_count), queries=qs.rows, scored_queries=qs.rows,
                  gallery_rows=gs.rows, support_classes=count, query_classes=len(set(truth)),
                  balanced_accuracy_classes="classes_present_in_query_truth",
                  macro_f1_classes="classes_present_in_query_truth_or_predictions_excluding_abstention",
                  top5_width={str(n): min(5, n, count) for n in neighbors},
                  depth=neighbors[-1], seconds=seconds, voting_seconds=voting_seconds,
                  search_precision="float32_tf32_disabled", vote_precision="float32",
                  normalization="l2_during_search" if normalize else "as_supplied_l2_codes",
                  timing_scope="gallery_streaming_and_ranking;excludes_query_upload_and_voting",
                  voting_timing_scope="neighbour_upload_empty_detection_class_votes_and_counts")
    return result
