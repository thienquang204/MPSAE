"""DINO classification settings, cache isolation, and frozen feature reuse."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mpsae.benchmark.config import Campaign, load_campaign
from mpsae.benchmark.run import evaluation_identity, evaluate_point
from mpsae.config import BACKBONES
from mpsae.io import canonical_hash


def test_optional_smoke_campaign_classifies_eurosat_and_retrieves_sop():
    campaign = load_campaign("configs/smoke.yaml")
    assert campaign.task_for("eurosat") == "knn_classification"
    assert campaign.task_for("sop") == "retrieval"
    assert campaign.evaluation_options("eurosat")["knn"] == {
        "n_neighbors": [1, 20], "temperature": 0.07}
    assert campaign.evaluation_options("sop")["knn"] is None
    assert campaign.resolved()["downstream"]["tasks"]["eurosat"] == "knn_classification"


@pytest.mark.parametrize("options", [
    {"tasks": {"sop": "knn_classification"}}, {"tasks": {"eurosat": "typo"}},
    {"tasks": {"flowers102": "retrieval"}}, {"tasks": []},
    {"knn": {"n_neighbors": []}}, {"knn": {"n_neighbors": [1, 1]}},
    {"knn": {"n_neighbors": [True]}}, {"knn": {"n_neighbors": [0]}},
    {"knn": {"temperature": 0}}, {"knn": {"temperature": float("inf")}},
    {"knn": {"temperature": float("nan")}}, {"knn": {"temperature": True}},
    {"knn": {"typo": 20}},
])
def test_invalid_classifier_protocols_fail_at_plan_time(options):
    with pytest.raises(ValueError):
        Campaign.from_mapping({"downstream": options})


def test_classifier_changes_do_not_change_training_cells_or_presets():
    original = load_campaign("configs/thesis.yaml")
    original = replace(original, downstream={**deepcopy(original.downstream),
        "datasets": ["eurosat", "sop"],
        "tasks": {"eurosat": "knn_classification", "sop": "retrieval"}})
    changed = replace(original, downstream={**deepcopy(original.downstream),
        "knn": {"n_neighbors": [1, 5], "temperature": 0.1}})
    assert changed.plan()["plan_id"] != original.plan()["plan_id"]
    assert original.source_cells() == changed.source_cells()
    assert original.resolved()["method_presets"] == changed.resolved()["method_presets"]
    assert original.plan()["source_cells"] == 18
    assert original.plan()["downstream_sparse_points"] == 84


def _features():
    import numpy as np
    generator = np.random.default_rng(42)
    gallery = SimpleNamespace(name="train", features=generator.normal(size=(24, 6)).astype("float32"),
                              labels=np.repeat(np.arange(3), 8), ids=[f"support:{i}" for i in range(24)])
    query = SimpleNamespace(name="test", features=generator.normal(size=(6, 6)).astype("float32"),
                            labels=np.repeat(np.arange(3), 2), ids=[f"query:{i}" for i in range(6)])
    return SimpleNamespace(name="eurosat", gallery=gallery, query=query, exclude_self=False,
                           protocol="fixture-train-test", identity={"fixture": "unchanged"}, manifest={})


def test_classifier_identity_changes_with_knobs_and_support_query_metadata():
    features = _features()
    arguments = {"k": 8, "device": "cpu", "gallery_chunk": 10, "query_block": 5}
    retrieval = evaluation_identity(features, None, **arguments)
    original = evaluation_identity(features, None, task="knn_classification", **arguments)
    assert "task" not in retrieval and retrieval["ranks"] == [1, 10, 100]
    assert canonical_hash(original) != canonical_hash(retrieval)
    for settings in ({"n_neighbors": [1, 5]}, {"temperature": 0.1}):
        assert evaluation_identity(features, None, task="knn_classification", knn=settings,
                                   **arguments) != original
    for split_name, key in (("gallery", "labels"), ("query", "labels"), ("gallery", "ids"), ("query", "ids")):
        altered = deepcopy(features)
        split = getattr(altered, split_name)
        getattr(split, key)[0] = 2 if key == "labels" else "a-different-sample"
        assert evaluation_identity(altered, None, task="knn_classification", **arguments) != original


def test_dense_classification_cache_is_separate_from_retrieval_and_reused(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from mpsae.evaluate import classification
    features = _features()
    spec = replace(BACKBONES["rn18"], dim=6, k_list=(2, 4), e2_anchors=(2, 4))
    old = evaluate_point(features, spec, tmp_path, device="cpu", task="retrieval")
    new = evaluate_point(features, spec, tmp_path, device="cpu")
    assert new["task"] == "knn_classification"
    assert new["evaluation_id"] != old["evaluation_id"]
    assert new["comparison_id"] != old["comparison_id"]
    assert "recall_at_1" not in new["metrics"]
    assert "knn_20_top1" in new["metrics"]
    assert new["metrics"]["knn_1_top1"] == old["metrics"]["recall_at_1"]

    def no_search(*args, **kwargs):
        raise AssertionError("completed classification must load from cache")

    monkeypatch.setattr(classification, "evaluate_knn", no_search)
    assert evaluate_point(features, spec, tmp_path, device="cpu") == new
    assert (tmp_path / "evaluations" / old["evaluation_id"] / "result.json").exists()
