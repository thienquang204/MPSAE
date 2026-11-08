"""Configuration, provenance and cache boundaries of the unified campaign."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mpsae.benchmark.config import Campaign, load_campaign
from mpsae.benchmark.run import (campaign_paths, campaign_status, checkpoint_record,
                                 evaluation_identity, verify_feature_provenance, _report)
from mpsae.io import atomic_json, canonical_hash
from mpsae.config import Cell


def test_default_campaign_is_source_training_plus_frozen_transfer():
    campaign = load_campaign("configs/thesis.yaml")
    plan = campaign.plan()
    assert plan["source_cells"] == 690
    assert plan["downstream_sparse_points"] == 1350
    assert plan["downstream_dense_points"] == 15
    assert campaign.downstream["mode"] == "frozen"


def test_smoke_and_scientific_workspaces_never_overlap(tmp_path):
    campaign = load_campaign("configs/smoke.yaml")
    normal = campaign_paths(campaign, tmp_path)
    smoke = campaign_paths(campaign, tmp_path, smoke=True)
    assert normal["workspace"] != smoke["workspace"]
    assert smoke["workspace"] == tmp_path / "smoke-smoke"
    assert campaign_status(campaign, tmp_path)["state"]["complete"] is False
    assert not normal["state"].exists()  # status has no filesystem side effects


@pytest.mark.parametrize("payload", [
    {"fine_tune": True}, {"downstream": {"mode": "adapted"}},
    {"name": "../escape"}, {"seeds": [True]}, {"execution": {"max_steps": -1}},
    {"imagenet": {"e1": False, "e2": True}},
    {"downstream": {"train_k": [3]}},
    {"downstream": {"eval_k": [9999]}},
    {"imagenet": {"enabled": False}, "downstream": {"enabled": False}},
    {"downstream": {"typo": True}},
])
def test_invalid_or_adaptation_configs_are_rejected(payload):
    with pytest.raises(ValueError):
        Campaign.from_mapping(payload)


def test_modern_backbones_support_explicit_per_backbone_budgets():
    campaign = Campaign.from_mapping({"backbones": ["rn18", "vit_b16"], "downstream": {
        "train_k": {"rn18": [8], "vit_b16": [12]},
        "eval_k": {"rn18": [2, 8], "vit_b16": [3, 12]}}})
    assert campaign.budgets("downstream", "train_k", "vit_b16") == (12,)
    assert {c.k for c in campaign.downstream_cells() if c.backbone == "vit_b16"} == {12}


def test_downstream_selection_does_not_change_source_cells():
    first = Campaign.from_mapping({"imagenet": {"train_k": [8]},
                                   "downstream": {"datasets": ["eurosat"]}})
    second = Campaign.from_mapping({"imagenet": {"train_k": [8]},
                                    "downstream": {"datasets": ["eurosat", "sop"]}})
    assert first.source_cells() == second.source_cells()
    assert first.plan()["plan_id"] != second.plan()["plan_id"]


def _weights():
    return {"repository": "timm/resnet18.a1_in1k", "revision": "a" * 40,
            "file": "model.safetensors", "sha256": "b" * 64}


def _completed(root, *, sidecar_digest="c" * 64):
    cell = Cell("rn18", "csr", 8, 1227)
    out = root / "runs" / cell.slug
    identity = {"backbone": cell.backbone, "method": cell.method, "target_k": cell.k,
                "seed": cell.seed, "data": {"train": {"sha256": "c" * 64, "rows": 10}}}
    signature = canonical_hash(identity)
    atomic_json(out / "summary.json", {"identity": identity, "signature": signature,
                                       "cell": cell.__dict__, "complete": True})
    atomic_json(out / "config.json", {"identity": identity, "signature": signature,
                                      "code_version": "2.0.0"})
    (out / "checkpoint.pt").write_bytes(b"source checkpoint bytes")
    manifest = {"sha256": sidecar_digest, "weights": _weights(), "input_size": 224,
                "normalize": [[0.5] * 3, [0.5] * 3]}
    atomic_json(root / "embeddings" / "rn18" / "1K_train_rn18.npz.json", manifest)
    return cell, out


def test_old_checkpoint_provenance_requires_matching_source_archive_digest(tmp_path):
    cell, out = _completed(tmp_path, sidecar_digest="d" * 64)
    with pytest.raises(RuntimeError, match="no pinned backbone"):
        checkpoint_record(cell, tmp_path)
    # ImageNet-only use remains compatible without downstream sidecars.
    record = checkpoint_record(cell, tmp_path, require_provenance=False)
    assert record["backbone_weights"] is None


def test_checkpoint_catalog_does_not_modify_original_source_artifacts(tmp_path):
    cell, out = _completed(tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    record = checkpoint_record(cell, tmp_path)
    assert record["backbone_weights"] == _weights()
    assert record["source_code_version"] == "2.0.0"
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    saved = json.loads((out / "summary.json").read_text())
    saved["cell"]["seed"] = 99
    atomic_json(out / "summary.json", saved)
    with pytest.raises(RuntimeError, match="selected source cell"):
        checkpoint_record(cell, tmp_path)


def test_shared_features_must_match_each_checkpoint_weights_and_preprocessing():
    features = SimpleNamespace(manifest={"extraction": {"weights": _weights(), "preprocessing": {
        "mean": [0.5] * 3, "std": [0.5] * 3, "center_crop": 224}}})
    checkpoint = {"backbone_weights": _weights(), "backbone_preprocessing": {
        "input_size": 224, "normalize": [[0.5] * 3, [0.5] * 3]}}
    verify_feature_provenance(features, checkpoint)
    bad = deepcopy(checkpoint)
    bad["backbone_weights"]["revision"] = "e" * 40
    with pytest.raises(RuntimeError, match="different backbone weights"):
        verify_feature_provenance(features, bad)
    bad = deepcopy(checkpoint)
    bad["backbone_preprocessing"]["normalize"][0][0] = 0.1
    with pytest.raises(RuntimeError, match="normalization/input size"):
        verify_feature_provenance(features, bad)


def test_evaluation_cache_depends_on_weights_data_protocol_budget_and_precision():
    features = SimpleNamespace(identity={"dataset": "flowers", "revision": "a"}, exclude_self=False)
    checkpoint = {"checkpoint_id": "weights-a", "signature": "fit-a"}
    def identity(**changes):
        arguments = {"k": 8, "device": "cpu", "gallery_chunk": 10, "query_block": 5, **changes}
        return canonical_hash(evaluation_identity(features, checkpoint, **arguments))
    original = identity()
    assert identity(k=16) != original
    assert identity(device="cuda:0") != original
    checkpoint["checkpoint_id"] = "weights-b"
    assert identity() != original
    checkpoint["checkpoint_id"] = "weights-a"
    features.identity["revision"] = "b"
    assert identity() != original


def test_missing_evaluations_cannot_be_reported_as_a_complete_campaign(tmp_path):
    campaign = Campaign.from_mapping({"backbones": ["rn18"], "methods": ["csr"], "seeds": [1227],
        "imagenet": {"enabled": False}, "downstream": {"datasets": ["sop"],
                                                          "train_k": [8], "eval_k": [8]}})
    paths = campaign_paths(campaign, tmp_path)
    catalog = {"checkpoints": {}, "evaluations": [], "imagenet": [], "dense": {}}
    manifest = _report(campaign, paths, catalog, [])
    assert manifest["complete"] is False
    assert manifest["downstream"]["counts"]["missing_points"] == 2
    assert manifest["expected"]["source_cells"] == 1
