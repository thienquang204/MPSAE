"""DINO voting parity and sparse-transfer protocol boundaries."""
from collections import Counter

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mpsae.evaluate import classification
from mpsae.evaluate.ranking import evaluate_retrieval


def _dino_reference(gallery, labels, queries, truth, neighbors, temperature):
    """Independent dense one-hot DINO rule; no production ranking/voting helpers.

    The upstream eval_knn.py uses exp(similarity/T), multiplies by one-hot
    neighbour labels, sums over neighbours, and sorts class totals.
    """
    similarity = queries @ gallery.T
    scores, indices = torch.topk(similarity, neighbors, dim=1, largest=True, sorted=True)
    one_hot = torch.nn.functional.one_hot(labels[indices], num_classes=int(labels.max()) + 1)
    votes = (one_hot * torch.exp(scores / temperature)[..., None]).sum(dim=1)
    order = votes.argsort(dim=1, descending=True)
    prediction = order[:, 0]
    top5 = order[:, :min(5, neighbors, votes.shape[1])].eq(truth[:, None]).any(dim=1)
    tp = Counter(int(y) for p, y in zip(prediction, truth) if p == y)
    predicted, actual = Counter(prediction.tolist()), Counter(truth.tolist())
    all_labels = set(predicted) | set(actual)
    return {"top1": float(prediction.eq(truth).double().mean()),
            "top5": float(top5.double().mean()),
            "balanced_accuracy": sum(tp[y] / actual[y] for y in actual) / len(actual),
            "macro_f1": sum(2 * tp[y] / (actual[y] + predicted[y]) for y in all_labels) / len(all_labels)}


def _random_data():
    rng = torch.Generator().manual_seed(9811)
    gallery = torch.nn.functional.normalize(torch.randn(56, 11, generator=rng), dim=1)
    queries = torch.nn.functional.normalize(torch.randn(19, 11, generator=rng), dim=1)
    return gallery, torch.arange(56) % 7, queries, torch.arange(19) % 7


@pytest.mark.parametrize("temperature", [0.07, 0.2])
def test_matches_independent_dino_dense_voting_and_reuses_one_search(monkeypatch, temperature):
    gallery, labels, queries, truth = _random_data()
    original = classification.exact_topk
    calls = []

    def search(*args, **kwargs):
        calls.append(kwargs["depth"])
        return original(*args, **kwargs)

    monkeypatch.setattr(classification, "exact_topk", search)
    result = classification.evaluate_knn(gallery, labels, queries, truth, dim=11, device="cpu",
                                         n_neighbors=(1, 20), temperature=temperature,
                                         gallery_chunk=13, query_block=6)
    assert calls == [20]
    for n in (1, 20):
        expected = _dino_reference(gallery, labels, queries, truth, n, temperature)
        for metric, value in expected.items():
            assert result[f"knn_{n}_{metric}"] == pytest.approx(value)
    assert result["depth"] == 20
    assert result["top5_width"] == {"1": 1, "20": 5}
    assert result["seconds"] >= 0 and result["voting_seconds"] >= 0


def test_sparse_codes_match_dense_codes_and_one_nn_matches_existing_retrieval():
    gallery = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
                            [0.9, 0.1, 0., 0.], [0., 0.9, 0.1, 0.], [0., 0., 0.9, 0.1]])
    queries = torch.tensor([[0.9, 0.2, 0., 0.], [0., 0.9, 0.2, 0.], [0., 0., 0., 0.]])
    gallery = torch.nn.functional.normalize(gallery, dim=1)
    queries = torch.nn.functional.normalize(queries, dim=1)
    labels, truth = [10, 20, 30, 10, 20, 30], [10, 30, 10]

    def compact(codes):
        values, indices = codes.topk(2, dim=1)
        return {"indices": indices.int().numpy(), "values": values.numpy()}

    dense = classification.evaluate_knn(gallery, labels, queries, truth,
                                         dim=4, device="cpu", n_neighbors=(1, 3))
    sparse = classification.evaluate_knn(compact(gallery), labels, compact(queries), truth,
                                          dim=4, device="cpu", n_neighbors=(1, 3))
    retrieval = evaluate_retrieval(compact(gallery), labels, compact(queries), truth, dim=4, device="cpu")
    for name in dense:
        if name.startswith("knn_"):
            assert sparse[name] == dense[name]
    assert sparse["knn_1_top1"] == retrieval["recall_at_1"]
    assert sparse["empty_queries"] == 1
    assert sparse["knn_1_top1"] == pytest.approx(1 / 3)


def test_multineighbor_vote_is_not_recall_at_rank():
    gallery = torch.nn.functional.normalize(torch.tensor([[1., 0.], [1., 0.1], [1., -0.1]]), dim=1)
    queries = torch.tensor([[1., 0.]])
    labels, truth = [0, 1, 1], [0]
    result = classification.evaluate_knn(gallery, labels, queries, truth, dim=2, device="cpu",
                                         n_neighbors=(1, 3), temperature=0.07)
    retrieval = evaluate_retrieval(gallery, labels, queries, truth, dim=2, device="cpu", ranks=(1, 3))
    assert retrieval["recall_at_3"] == result["knn_1_top1"] == 1
    assert result["knn_3_top1"] == 0


def test_neighbor_and_class_ties_are_deterministic_across_chunks():
    gallery = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    labels = ["zebra", "ant", "zebra", "ant"]
    queries, truth = torch.tensor([[1., 0.], [0., 1.]]), ["ant", "zebra"]
    results = [classification.evaluate_knn(gallery, labels, queries, truth, dim=2, device="cpu",
                                            n_neighbors=(1, 2, 4), query_block=block,
                                            gallery_chunk=chunk)
               for chunk, block in ((1, 1), (2, 2), (4, 1))]
    for result in results:
        assert result["knn_1_top1"] == 0.5  # first gallery position wins
        assert result["knn_2_top1"] == 0.5  # equal class vote -> ant before zebra
        assert result["knn_2_macro_f1"] == pytest.approx(1 / 3)
    metrics = [{k: v for k, v in result.items() if k.startswith("knn_")} for result in results]
    assert metrics[0] == metrics[1] == metrics[2]


def test_empty_query_abstains_even_when_first_support_label_is_correct():
    result = classification.evaluate_knn(np.eye(2, dtype=np.float32), [0, 1],
                                          np.zeros((1, 2), dtype=np.float32), [0],
                                          dim=2, device="cpu", n_neighbors=(1, 2))
    assert result["empty_queries"] == 1
    assert all(value == 0 for key, value in result.items() if key.startswith("knn_"))
    assert result["empty_query_policy"] == classification.EMPTY_QUERY_POLICY


def test_balanced_accuracy_and_macro_f1_count_abstentions_as_false_negatives():
    result = classification.evaluate_knn(np.eye(2, dtype=np.float32), [0, 1],
        np.array([[1, 0], [1, 0], [0, 0], [0, 1]], dtype=np.float32), [0, 0, 0, 1],
        dim=2, device="cpu", n_neighbors=(1,))
    assert result["knn_1_top1"] == 0.75
    assert result["knn_1_balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert result["knn_1_macro_f1"] == pytest.approx((4 / 5 + 1) / 2)


def test_normalization_and_extreme_temperature_are_finite():
    result = classification.evaluate_knn(np.array([[2, 0], [0, 3]], dtype=np.float32), [0, 1],
                                          np.array([[4, 0]], dtype=np.float32), [0],
                                          dim=2, device="cpu", normalize=True,
                                          temperature=1e-8, n_neighbors=(1, 2))
    assert result["knn_1_top1"] == result["knn_2_top1"] == 1
    assert result["normalization"] == "l2_during_search"


@pytest.mark.parametrize("neighbors", [(), (0,), (True,), (1.5,), (1, 1), (3,)])
def test_invalid_neighbor_counts_fail_without_search(monkeypatch, neighbors):
    monkeypatch.setattr(classification, "exact_topk", lambda *a, **k: pytest.fail("search must not run"))
    with pytest.raises(ValueError):
        classification.evaluate_knn(np.eye(2), [0, 1], np.eye(2), [0, 1],
                                      dim=2, device="cpu", n_neighbors=neighbors)


@pytest.mark.parametrize("temperature", [0, -1, True, float("nan"), float("inf"), "0.07"])
def test_invalid_temperature_fails(temperature):
    with pytest.raises(ValueError, match="temperature"):
        classification.validate_knn((1,), temperature)


@pytest.mark.parametrize("extra,match", [
    ({"gallery_ids": ["same", "b"], "query_ids": ["same"]}, "overlap"),
    ({"gallery_ids": ["a", "b"]}, "both"),
    ({"gallery_ids": ["a", "a"], "query_ids": ["q"]}, "unique"),
])
def test_invalid_sample_identity_is_rejected(extra, match):
    with pytest.raises(ValueError, match=match):
        classification.evaluate_knn(np.eye(2), [0, 1], np.array([[1., 0.]]), [0],
                                      dim=2, device="cpu", n_neighbors=(1,), **extra)


def test_unknown_query_class_and_misaligned_labels_are_rejected():
    with pytest.raises(ValueError, match="covered"):
        classification.evaluate_knn(np.eye(2), [0, 1], np.array([[1., 0.]]), [2],
                                      dim=2, device="cpu", n_neighbors=(1,))
    with pytest.raises(ValueError, match="row counts"):
        classification.evaluate_knn(np.eye(2), [0, 1], np.eye(2), [0],
                                      dim=2, device="cpu", n_neighbors=(1,))


def test_no_gradient_is_recorded_for_frozen_representations():
    gallery = torch.eye(2, requires_grad=True)
    queries = torch.tensor([[1., 0.]], requires_grad=True)
    result = classification.evaluate_knn(gallery, [0, 1], queries, [0],
                                          dim=2, device="cpu", n_neighbors=(1,))
    assert result["knn_1_top1"] == 1
    assert gallery.grad is queries.grad is None
