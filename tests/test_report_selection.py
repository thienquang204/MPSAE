"""Campaign reports must not discover stale ImageNet runs outside their plan."""
from __future__ import annotations

import csv
import json

import pytest

from mpsae.report import aggregate, figures
from mpsae.config import Cell, METHOD_VARIANTS


@pytest.mark.parametrize("backbone", ["ff2048", "vit_b16", "convnext_tiny", "rn18", "dinov2_vits14"])
def test_campaign_filters_old_probes_and_timing_without_erasing_cache(backbone):
    from copy import deepcopy
    from mpsae.benchmark.config import load_campaign
    from mpsae.benchmark.run import _selected_imagenet_record

    campaign = load_campaign("configs/thesis.yaml")
    budgets = campaign.budgets("imagenet", "eval_k", backbone)
    anchor = campaign.budgets("imagenet", "train_k", backbone)[0]
    def point(k):
        return {"k": k, "top1": 70, "probe_top1": 80, "probe_top5": 90,
                "probe_settings": {"epochs": 10}, "probe_epochs": 10,
                "retrieval_ms": 1.0, "index_bytes": 256, "timing_status": "success",
                "timing_config": {"rounds": 2000}, "timing": {"median": 1.0}}
    record = {"backbone": backbone, "k": anchor, "e1": point(anchor),
              "e2": {"points": [point(k) for k in (*budgets, 1)], "cached_points": [point(512)]}}
    original = deepcopy(record)
    selected = _selected_imagenet_record(campaign, record)
    assert record == original
    assert "cached_points" not in selected["e2"]
    assert [p["k"] for p in selected["e2"]["points"]] == list(budgets)
    for p in [selected["e1"], *selected["e2"]["points"]]:
        assert p["top1"] == 70
        assert ("probe_top1" in p) is campaign.wants_probe(backbone, p["k"])
        assert not any(k.startswith("probe_") for k in p) or campaign.wants_probe(backbone, p["k"])
        assert not set(p) & {"retrieval_ms", "index_bytes", "timing_status", "timing_config", "timing"}


def test_report_only_filters_old_catalog_and_skips_downstream(tmp_path, monkeypatch):
    from mpsae.benchmark.config import Campaign
    from mpsae.benchmark import run, report

    campaign = Campaign.from_mapping({"backbones": ["rn18"], "methods": ["csr"], "seeds": [1227],
        "imagenet": {"train_k": [8], "eval_k": [2, 8], "probe_k": {}, "timing": False},
        "downstream": {"enabled": False}})
    paths = run.campaign_paths(campaign, tmp_path)
    record = _run(tmp_path / "runs", seed=1227)
    record["backbone"] = "rn18"
    record["e1"].update(probe_top1=90)
    record["e2"] = {"points": [{"k": k, "top1": 60, "probe_top1": 90} for k in (2, 8)]}
    catalog = {"checkpoints": {"fixture": {}}, "imagenet": [record], "dense": {}, "evaluations": []}
    _write(paths["state"] / "catalog.json", catalog)
    before = (paths["state"] / "catalog.json").read_bytes()
    captured = []

    def no_downstream(*args, **kwargs):
        pytest.fail("disabled downstream must not produce a report")

    def image_report(*args, records, **kwargs):
        captured.extend(records)
        return {"skipped": []}

    monkeypatch.setattr(report, "build_report", no_downstream)
    monkeypatch.setattr(figures, "build_all", image_report)
    result = run.report_campaign(campaign, tmp_path)
    assert result["complete"]
    assert result["downstream"] is None
    assert not (paths["reports"] / "downstream").exists()
    assert "probe_top1" not in captured[0]["e1"]
    assert all("probe_top1" not in p for p in captured[0]["e2"]["points"])
    assert (paths["state"] / "catalog.json").read_bytes() == before


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(root, *, method="csr", k=8, seed=7, top1=60.0, log=True):
    path = root / Cell("ff2048", method, k, seed).slug
    config = {"code_version": "2.0.0", "total_steps": 10}
    if method in METHOD_VARIANTS:
        config["identity"] = {"objective": {"variant": METHOD_VARIANTS[method]}}
    e1 = {"top1": top1, "k": k, "seed": seed, "method": method, "backbone": "ff2048"}
    _write(path / "config.json", config)
    _write(path / "eval_e1.json", e1)
    if log:
        _write(path / "train.jsonl", {"step": 10, "elapsed": 2.0, "loss_total": 0.2,
                                      "training_k": k})
    return {"backbone": "ff2048", "method": method, "k": k, "seed": seed,
            "path": str(path), "config": config, "e1": e1, "e2": None,
            "has_train_log": log, "code_version": "2.0.0", "complete": True, "missing": []}


def _tables_only(monkeypatch):
    def no_matplotlib():
        raise ImportError("tables-only test")

    monkeypatch.setattr(figures, "apply_style", no_matplotlib)


def test_explicit_report_reads_only_selected_cells_and_configured_seeds(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    chosen = _run(root)
    ignored = _run(root, k=16, seed=8, top1=99)
    (root / "ff2048" / "csr" / "k16" / "seed8" / "train.jsonl").write_text(
        "invalid stale JSON", encoding="utf-8")
    _write(root / "ff2048" / "dense_reference.json", {"top1": 99})

    def unexpected_discovery(*args, **kwargs):
        raise AssertionError("explicit reports must not discover old artefacts")

    monkeypatch.setattr(figures, "load_runs", unexpected_discovery)
    monkeypatch.setattr(figures, "load_dense_references", unexpected_discovery)
    _tables_only(monkeypatch)
    manifest = figures.build_all(root, tmp_path / "reports", records=[chosen, ignored],
                                dense_references={"ff2048": {"top1": 71.19}}, seeds=(7,))
    assert manifest["selection"] == "explicit"
    assert manifest["seeds"] == [7]
    assert manifest["counts"]["runs"] == manifest["counts"]["training_logs"] == 1
    with open(manifest["tables"]["e1_accuracy"]["csv"], encoding="utf-8", newline="") as handle:
        row, = list(csv.DictReader(handle))
    assert float(row["top1_mean"]) == 60
    assert row["seeds_missing"] == ""
    with open(manifest["tables"]["dense_reference"]["csv"], encoding="utf-8", newline="") as handle:
        dense, = list(csv.DictReader(handle))
    assert float(dense["top1_mean"]) == 71.19


def test_explicit_empty_selection_never_falls_back_to_workspace(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    _run(root)
    (root / "ff2048" / "csr" / "k8" / "seed7" / "eval_e1.json").write_text(
        "invalid stale evaluation", encoding="utf-8")
    (root / "ff2048" / "csr" / "k8" / "seed7" / "train.jsonl").write_text(
        "invalid stale training log", encoding="utf-8")
    _tables_only(monkeypatch)
    manifest = figures.build_all(root, tmp_path / "reports", records=[], seeds=(7,))
    assert manifest["counts"]["runs"] == manifest["counts"]["training_logs"] == 0
    assert "dense_reference" not in manifest["tables"]


def test_default_build_all_keeps_legacy_workspace_discovery(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    _run(root, seed=1227)
    _write(root / "ff2048" / "dense_reference.json", {"top1": 71.19})
    _tables_only(monkeypatch)
    manifest = figures.build_all(root, tmp_path / "reports")
    assert manifest["selection"] == "workspace"
    assert manifest["counts"]["runs"] == manifest["counts"]["training_logs"] == 1
    assert "dense_reference" in manifest["tables"]


def test_custom_seed_is_used_in_paired_tables_and_delta_figures(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    records = [_run(root, method="csr", top1=60), _run(root, method="mpsae", top1=61)]
    monkeypatch.setattr(figures, "apply_style", lambda: {})
    for name in ("figure_backbone_summary", "figure_e1_curves", "figure_e2_generalisation",
                 "figure_e2_nnz", "figure_efficiency", "figure_training"):
        monkeypatch.setattr(figures, name, lambda *args, **kwargs: [])
    calls = []

    def delta(*args, **kwargs):
        calls.append(kwargs["seeds"])
        return []

    monkeypatch.setattr(figures, "figure_e1_delta", delta)
    manifest = figures.build_all(root, tmp_path / "reports", records=records, seeds=(7,))
    with open(manifest["tables"]["paired_mpsae_vs_csr"]["csv"], encoding="utf-8", newline="") as handle:
        row, = list(csv.DictReader(handle))
    assert float(row["delta_top1_mean"]) == 1
    assert row["seeds_paired"] == "7" and row["seeds_missing"] == ""
    assert calls == [(7,), (7,)]


def test_training_summary_explicit_selection_ignores_malformed_unselected_log(tmp_path):
    root = tmp_path / "runs"
    _run(root)
    stale = root / "ff2048" / "mpsae" / "k8" / "seed7" / "train.jsonl"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("not JSON", encoding="utf-8")
    rows = aggregate.training_summary(root, selected_cells=[("ff2048", "csr", 8, 7)])
    assert len(rows) == 1 and rows[0]["method"] == "csr"
    assert aggregate.training_summary(root, selected_cells=[]) == []


def test_imagenet_reports_exclude_retired_methods_from_mixed_explicit_records(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    records = [_run(root, method=method, top1=value)
               for method, value in (("csr", 59), ("csrv2", 60), ("mpsae", 61),
                                     ("mpsaev2", 99), ("csr_multitopk", 98))]
    for item in records:
        item["e2"] = {"anchor_k": 8, "points": [{"k": 16, "top1": item["e1"]["top1"]}]}
        if item["method"] in ("mpsaev2", "csr_multitopk"):
            # Exclusion happens before validation of old objective metadata or logs.
            item["config"] = None
            (root / Cell("ff2048", item["method"], 8, 7).slug / "train.jsonl").write_text(
                "invalid retired log", encoding="utf-8")
    _tables_only(monkeypatch)
    manifest = figures.build_all(root, tmp_path / "reports", records=records, seeds=(7,))
    assert manifest["counts"] == {"runs": 3, "runs_with_e1": 3, "e1_cells": 3,
                                  "e2_points": 3, "training_logs": 3}
    for table in ("e1_accuracy", "e2_generalisation", "training_summary"):
        with open(manifest["tables"][table]["csv"], encoding="utf-8", newline="") as handle:
            assert {row["method"] for row in csv.DictReader(handle)} == {"csr", "csrv2", "mpsae"}
    assert {name for name in manifest["tables"] if name.startswith("paired_")} == {
        "paired_mpsae_vs_csr", "paired_mpsae_vs_csrv2"}
    assert {row["method"] for row in aggregate.aggregate_e1(records)} == {"csr", "csrv2", "mpsae"}
    assert {row["method"] for row in aggregate.aggregate_e2(records)} == {"csr", "csrv2", "mpsae"}


def test_workspace_reports_never_parse_retired_results_or_logs(tmp_path):
    root = tmp_path / "runs"
    _run(root, seed=1227)
    for directory in ("csr_multitopk", "mpsaev2", "mpsaev2__fixed_ot_cl_v1"):
        old = root / "ff2048" / directory / "k8" / "seed1227"
        old.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "eval_e1.json", "eval_e2.json", "train.jsonl"):
            (old / name).write_text("invalid historical JSON", encoding="utf-8")
    assert [row["method"] for row in aggregate.load_runs(root)] == ["csr"]
    assert [row["method"] for row in aggregate.training_summary(root)] == ["csr"]
    assert aggregate.training_summary(root, selected_cells=[
        ("ff2048", "mpsaev2", 8, 1227), ("ff2048", "csr_multitopk", 8, 1227)]) == []
    for old in root.glob("ff2048/*/k8/seed1227/config.json"):
        if old.parents[2].name != "csr":
            assert old.read_text(encoding="utf-8") == "invalid historical JSON"


def test_explicit_duplicate_training_cells_are_rejected(tmp_path):
    root = tmp_path / "runs"
    chosen = _run(root)
    with pytest.raises(ValueError, match="duplicate training cells"):
        figures.build_all(root, tmp_path / "reports", records=[chosen, chosen], seeds=(7,))


def test_e2_plot_keeps_training_k_fixed_and_exports_single_seed_curves(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.collections import PolyCollection
    records = []
    for anchor in (8, 16):
        for method in ("csr", "mpsae"):
            item = _run(tmp_path / "runs", method=method, k=anchor, seed=1227, log=False)
            item["e2"] = {"anchor_k": anchor, "points": [
                {"k": k, "top1": anchor + k / 10 + (1 if method == "mpsae" else 0)}
                for k in (2, 8, 16, 32)]}
            records.append(item)
    rows = aggregate.aggregate_e2(records, seeds=(1227,))
    assert all(row["top1_n"] == 1 and row["top1_std"] is None and row["complete"] for row in rows)
    captured = []
    original_save = figures.save_figure
    def save(fig, *args, **kwargs):
        captured.append(fig)
        return original_save(fig, *args, **kwargs)
    monkeypatch.setattr(figures, "save_figure", save)
    paths = figures.figure_e2_generalisation(rows, tmp_path / "figures", "ff2048", seeds=(1227,))
    assert {path.suffix for path in paths} == {".png", ".pdf", ".txt"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    for ax, anchor in zip(captured[0].axes, (8, 16)):
        curves = [line for line in ax.lines if line.get_label() in ("CSR", "MPSAE")]
        assert len(curves) == 2
        assert list(curves[0].get_xdata()) == [2, 8, 16, 32]
        assert list(curves[0].get_ydata()) == pytest.approx([anchor + k / 10 for k in (2, 8, 16, 32)])
        assert ax.get_xlabel() == "Inference $K$"
        assert not any(isinstance(collection, PolyCollection) for collection in ax.collections)
        assert all(collection.get_offsets()[0, 0] == anchor for collection in ax.collections)
    caption = next(path for path in paths if path.suffix == ".txt").read_text(encoding="utf-8")
    assert "seed 1227 only (n=1)" in caption
    assert "no between-seed standard deviation" in caption
