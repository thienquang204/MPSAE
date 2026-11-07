"""Campaign reports must not discover stale ImageNet runs outside their plan."""
from __future__ import annotations

import csv
import json

import pytest

from mpsae.report import aggregate, figures


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(root, *, method="csr", k=8, seed=7, top1=60.0, log=True):
    path = root / "ff2048" / method / f"k{k}" / f"seed{seed}"
    config = {"code_version": "2.0.0", "total_steps": 10}
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


def test_explicit_duplicate_training_cells_are_rejected(tmp_path):
    root = tmp_path / "runs"
    chosen = _run(root)
    with pytest.raises(ValueError, match="duplicate training cells"):
        figures.build_all(root, tmp_path / "reports", records=[chosen, chosen], seeds=(7,))
