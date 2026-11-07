"""Downstream retrieval contracts, including a torch-free metric oracle."""
from __future__ import annotations

import pytest

from mpsae.evaluate.ranking import summarize_ranking


def test_metrics_match_hand_calculation():
    # The first query gets a positive at rank 1, then a negative. R=2, so AP@R=.5.
    # The second query gets no positive until rank 3, so AP@R=0.
    result = summarize_ranking([[0, 1, 2, 3], [0, 2, 1, 3]],
                               ["a", "b", "a", "b"], ["a", "b"], ranks=(1, 2))
    assert result["units"] == "fraction"
    assert result["recall_at_1"] == 0.5
    assert result["map_at_r"] == 0.25
    assert result["precision_at_10"] == 0.5
    assert result["macro_map_at_r"] == 0.25


def test_macro_average_does_not_weight_larger_query_classes_more():
    result = summarize_ranking([[0, 1], [0, 1], [0, 1]], ["a", "b"],
                               ["a", "a", "b"], ranks=(1,))
    assert result["recall_at_1"] == pytest.approx(2 / 3)
    assert result["macro_recall_at_1"] == 0.5


def test_no_positive_policy_is_explicit_and_empty_codes_are_misses():
    with pytest.raises(ValueError, match="no relevant"):
        summarize_ranking([[0, 1]], ["a", "a"], ["missing"], ranks=(1,))
    result = summarize_ranking([[0, 1], [0, 1]], ["a", "a"], ["a", "missing"],
                               ranks=(1,), empty_queries=[True, False],
                               no_positive_policy="exclude")
    assert result["scored_queries"] == 1 and result["no_positive_queries"] == 1
    assert result["empty_queries"] == 1
    assert result["recall_at_1"] == result["map_at_r"] == 0


def test_incomplete_ap_r_ranking_is_rejected():
    with pytest.raises(ValueError, match="too short"):
        summarize_ranking([[0, 1]], ["a"] * 12, ["a"], ranks=(1,))


def test_self_exclusion_uses_ids_and_validates_integrity():
    result = summarize_ranking([[1, 2, -1]], ["a", "a", "b"], ["a"],
                               gallery_ids=["same", "other", "third"], query_ids=["same"],
                               exclude_self=True, ranks=(1,))
    assert result["recall_at_1"] == result["map_at_r"] == 1
    with pytest.raises(ValueError, match="own sample"):
        summarize_ranking([[0, 1]], ["a", "a"], ["a"],
                           gallery_ids=["same", "other"], query_ids=["same"],
                           exclude_self=True, ranks=(1,))
    with pytest.raises(ValueError, match="unique"):
        summarize_ranking([[0, 1]], ["a", "a"], ["a"], gallery_ids=["x", "x"], ranks=(1,))


def test_blocked_ranking_matches_bruteforce_and_legacy_top1():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from mpsae.evaluate.knn import exact_top1
    from mpsae.evaluate.ranking import exact_topk

    generator = np.random.default_rng(17)
    gallery = generator.normal(size=(29, 7)).astype(np.float32)
    queries = generator.normal(size=(11, 7)).astype(np.float32)
    scores, positions, _ = exact_topk(gallery, queries, dim=7, device="cpu", depth=13,
                                     gallery_chunk=5, query_block=3)
    brute = queries @ gallery.T
    expected = np.argsort(-brute, axis=1, kind="stable")[:, :13]
    assert np.array_equal(positions, expected)
    assert np.allclose(scores, np.take_along_axis(brute, expected, axis=1), atol=2e-6)
    old_scores, old_positions, _ = exact_top1(gallery, queries, dim=7, device="cpu",
                                             gallery_chunk=5, query_block=3)
    assert np.array_equal(positions[:, 0], old_positions.numpy())
    assert np.allclose(scores[:, 0], old_scores.numpy(), atol=2e-6)


def test_ties_and_permuted_self_ids_across_chunks():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from mpsae.evaluate.ranking import exact_topk

    gallery = np.ones((7, 2), dtype=np.float32)
    queries = np.ones((2, 2), dtype=np.float32)
    _, positions, _ = exact_topk(gallery, queries, dim=2, device="cpu", depth=7,
                                 gallery_ids=["a", "b", "c", "d", "e", "f", "g"],
                                 query_ids=["f", "b"], exclude_self=True,
                                 gallery_chunk=2, query_block=1)
    assert positions.tolist() == [[0, 1, 2, 3, 4, 6, -1], [0, 2, 3, 4, 5, 6, -1]]


def test_evaluator_retrieves_beyond_top100_for_map_r():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from mpsae.evaluate.ranking import evaluate_retrieval

    gallery = np.tile([[1.0, 0.0]], (121, 1)).astype(np.float32)
    result = evaluate_retrieval(gallery, ["b"] * 16 + ["a"] * 105,
                                np.array([[1.0, 0.0]], dtype=np.float32), ["a"],
                                dim=2, device="cpu", gallery_chunk=17)
    expected_ap = sum((rank - 16) / rank for rank in range(17, 106)) / 105
    assert result["depth"] == 105
    assert result["map_at_r"] == pytest.approx(expected_ap)
    assert result["recall_at_1"] == 0 and result["recall_at_100"] == 1


def test_signed_dense_features_and_zero_codes():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from mpsae.evaluate.ranking import evaluate_retrieval

    gallery = np.array([[-1.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    queries = np.array([[-1.0, -1.0], [0.0, 0.0]], dtype=np.float32)
    result = evaluate_retrieval(gallery, ["a", "b"], queries, ["a", "a"],
                                dim=2, device="cpu", normalize=True, ranks=(1,))
    assert result["empty_queries"] == 1
    assert result["recall_at_1"] == result["map_at_r"] == 0.5
    assert result["precision_at_10"] == 0.25
