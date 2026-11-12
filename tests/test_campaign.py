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


@pytest.mark.parametrize("path", ["configs/thesis.yaml", "configs/precomputed.yaml"])
def test_default_campaign_is_imagenet_only_with_selective_probes(path):
    campaign = load_campaign(path)
    plan = campaign.plan()
    assert plan["source_cells"] == 39
    assert plan["imagenet_e1_points"] == 39
    assert plan["imagenet_e2_points"] == 222
    assert plan["imagenet_shared_points"] == 39
    assert plan["imagenet_unique_points"] == 222
    assert plan["imagenet_probe_points"] == 84
    assert plan["downstream_sparse_points"] == 0
    assert plan["downstream_dense_points"] == 0
    assert campaign.downstream["enabled"] is False
    assert campaign.imagenet["timing"] is False
    assert campaign.imagenet["timing_warmup"] == campaign.imagenet["timing_rounds"] == 5
    from mpsae.embeddings_hub import campaign_selection
    assert campaign_selection(campaign) == (["imagenet"], list(campaign.backbones))
    assert campaign.downstream["mode"] == "frozen"
    assert campaign.execution["precomputed_only"] is True
    assert campaign.seeds == (1227,)
    assert campaign.target_backbones("sop") == ()
    assert campaign.backbones == ("ff2048", "vit_b16", "convnext_tiny", "dinov2_vits14")
    assert campaign.downstream["datasets"] == ["sop"]


def test_retrieval_only_campaign_reuses_main_sources_without_imagenet_work(tmp_path):
    from mpsae.embeddings_hub import campaign_selection

    main = load_campaign("configs/thesis.yaml")
    retrieval = load_campaign("configs/retrieval.yaml")
    plan = retrieval.plan()
    assert retrieval.source_cells() == [c for c in main.source_cells() if c.backbone == "ff2048"]
    assert retrieval.backbones == ("ff2048",)
    assert retrieval.downstream["datasets"] == ["sop"]
    assert retrieval.evaluation_options("sop") == main.evaluation_options("sop")
    assert retrieval.budgets("downstream", "eval_k", "ff2048") == (2, 16, 32, 256)
    assert all(retrieval.imagenet[key] is False
               for key in ("enabled", "prepare", "e1", "e2", "linear_probe", "timing"))
    assert retrieval.execution["train_missing"] is False
    assert retrieval.execution["precomputed_only"] is True
    assert retrieval.execution["continue_on_error"] is False
    assert campaign_selection(retrieval) == (["sop"], ["ff2048"])
    assert (plan["source_cells"], plan["downstream_sparse_points"],
            plan["downstream_dense_points"]) == (12, 48, 1)
    assert plan["storage_estimate"]["imagenet_features_bytes"] == {}
    assert plan["storage_estimate"]["total_imagenet_features_bytes"] == 0
    assert plan["storage_estimate"]["active_writer_upper_bytes"] == 0
    assert campaign_paths(retrieval, tmp_path)["workspace"] == campaign_paths(main, tmp_path)["workspace"]
    assert campaign_paths(retrieval, tmp_path)["reports"] != campaign_paths(main, tmp_path)["reports"]


def test_dataset_backbone_selection_changes_plan_without_dropping_other_targets():
    campaign = Campaign.from_mapping({"backbones": ["rn18", "vit_b16"],
        "methods": ["csr"], "seeds": [1227], "downstream": {
            "datasets": ["eurosat", "sop"], "backbones": {"sop": ["rn18"]}}})
    assert campaign.target_backbones("eurosat") == ("rn18", "vit_b16")
    assert {cell.backbone for cell in campaign.downstream_cells("sop")} == {"rn18"}
    assert campaign.plan()["downstream_dense_points"] == 3
    assert campaign.plan()["downstream_sparse_points"] == 20


@pytest.mark.parametrize("config", ["configs/thesis.yaml", "configs/precomputed.yaml"])
@pytest.mark.parametrize("backbone,expected,training", [
    ("ff2048", (2, 8, 16, 32, 256), (8, 16, 32, 256)),
    ("vit_b16", (3, 12, 48, 96, 192, 384), (12, 48, 192)),
    ("convnext_tiny", (3, 12, 48, 96, 192, 384), (12, 48, 192)),
    ("dinov2_vits14", (3, 12, 48, 96, 192, 384), (12, 48, 192)),
])
def test_campaigns_restrict_training_and_e2_to_requested_budgets(config, backbone, expected, training):
    from mpsae.config import BACKBONES
    campaign = load_campaign(config)
    assert {cell.k for cell in campaign.source_cells() if cell.backbone == backbone} == set(training)
    assert campaign.budgets("imagenet", "eval_k", backbone) == expected
    assert campaign.downstream_cells() == []
    assert campaign.resolved()["resolved_budgets"][backbone]["imagenet_eval_k"] == expected
    # Campaign selection does not remove the wider reference grids from the registry.
    assert set(expected) <= set(range(1, BACKBONES[backbone].dim + 1))
    assert len(BACKBONES[backbone].training_budgets) > 1


@pytest.mark.parametrize("backbone,expected", [
    ("ff2048", (2, 8, 16, 32, 256)), ("rn18", (2, 8, 16, 32, 64, 128, 256)),
    ("vit_b16", (3, 12, 48, 96, 192, 384)),
    ("convnext_tiny", (3, 12, 48, 96, 192, 384)),
    ("dinov2_vits14", (3, 12, 48, 96, 192, 384)),
])
def test_default_inference_selection_matches_all_evaluation_families(backbone, expected):
    from mpsae.config import inference_budgets
    campaign = Campaign.from_mapping({"backbones": [backbone]})
    assert inference_budgets(backbone) == expected
    assert campaign.budgets("imagenet", "eval_k", backbone) == expected
    assert campaign.budgets("downstream", "eval_k", backbone) == expected


def test_new_source_models_are_imagenet_only():
    campaign = load_campaign("configs/thesis.yaml")
    from mpsae.config import METHODS
    for method in METHODS:
        expected = {("ff2048", method, k, 1227) for k in (8, 16, 32, 256)}
        selected = {(c.backbone, c.method, c.k, c.seed)
                    for c in campaign.source_cells() if c.backbone == "ff2048" and c.method == method}
        assert selected == expected
        assert campaign.downstream_cells() == []
    assert sum(len(campaign.budgets("imagenet", "eval_k", c.backbone))
               for c in campaign.source_cells()) == 222


@pytest.mark.parametrize("targets", [[], {"flowers102": ["rn18"]},
    {"sop": []}, {"sop": ["vit_b16"]}, {"sop": ["rn18", "rn18"]}])
def test_invalid_dataset_backbone_selections_are_rejected(targets):
    with pytest.raises(ValueError, match="downstream.backbones"):
        Campaign.from_mapping({"backbones": ["rn18"],
            "downstream": {"datasets": ["sop"], "backbones": targets}})


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
    {"imagenet": {"eval_k": [True]}}, {"imagenet": {"eval_k": [8, 8]}},
    {"imagenet": {"eval_k": [1024]}},  # exceeds rn18's width
    {"imagenet": {"enabled": False}, "downstream": {"enabled": False}},
    {"downstream": {"typo": True}},
    {"execution": {"precomputed_only": True}},
    {"execution": {"storage_mode": "direct_v1", "precomputed_only": "yes"}},
])
def test_invalid_or_adaptation_configs_are_rejected(payload):
    with pytest.raises(ValueError):
        Campaign.from_mapping(payload)


@pytest.mark.parametrize("selection", [[2], {"typo": [2]}, {"rn18": [True]},
    {"rn18": [2, 2]}, {"rn18": [999]}, {"rn18": [4]}, {"rn18": None}])
def test_invalid_probe_selections_fail_before_training(selection):
    with pytest.raises(ValueError, match="imagenet.probe_k"):
        Campaign.from_mapping({"backbones": ["rn18"], "imagenet": {
            "train_k": [8], "eval_k": [2, 8], "probe_k": selection}})


def test_probe_selection_defaults_opt_in_and_master_switch():
    from dataclasses import replace

    original = Campaign.from_mapping({"backbones": ["ff2048", "rn18"],
        "imagenet": {"train_k": [8], "eval_k": [2, 8]}, "downstream": {"enabled": False}})
    assert original.probe_budgets("rn18") is None
    assert original.wants_probe("rn18", 2)
    assert "probe_k" not in original.resolved()["imagenet"]
    selected = replace(original, imagenet={**original.imagenet, "probe_k": {"ff2048": [2]}})
    assert selected.source_cells() == original.source_cells()
    assert selected.resolved()["method_presets"] == original.resolved()["method_presets"]
    assert selected.plan()["plan_id"] != original.plan()["plan_id"]
    assert selected.wants_probe("ff2048", 2)
    assert not selected.wants_probe("ff2048", 8)
    assert not selected.wants_probe("rn18", 2)  # omitted backbones opt out
    disabled = replace(selected, imagenet={**selected.imagenet, "linear_probe": False})
    assert not disabled.wants_probe("ff2048", 2)


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
