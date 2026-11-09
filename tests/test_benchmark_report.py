"""Frozen-transfer reports preserve protocol, checkpoint and seed provenance."""
from __future__ import annotations

import csv
import json
import math

import pytest

from mpsae.benchmark import report


def record(method="mpsae", seed=1227, value=0.7, *, train_k=8, eval_k=8,
           protocol="hf-test-to-train", comparison_id="data-revision-a", mode="frozen"):
    dense = method == "dense"
    if dense:
        seed = train_k = eval_k = None
    evaluation_id = f"{method}-{seed}-{train_k}-{eval_k}-{protocol}-{comparison_id}-{mode}"
    return {"dataset": "flowers102", "protocol": protocol, "comparison_id": comparison_id,
            "backbone": "ff2048", "method": method, "seed": seed,
            "train_k": train_k, "eval_k": eval_k, "mode": mode,
            "evaluation_id": evaluation_id, "checkpoint_id": f"weights-{method}-{seed}",
            "complete": True,
            "metrics": {"recall_at_1": value, "map_at_r": value / 2,
                        "nnz_mean": 7.0, "code_bytes_per_image": 64,
                        "query_count": 819, "gallery_count": 6551}}


def test_aggregation_keeps_fraction_units_and_sample_standard_deviation():
    records = [record(seed=seed, value=value)
               for seed, value in zip((1227, 2026, 2027), (0.5, 0.7, 0.9))]
    row, = report.aggregate_records(records)
    assert row["recall_at_1_mean"] == pytest.approx(0.7)
    assert row["recall_at_1_std"] == pytest.approx(0.2)
    assert row["recall_at_1_n"] == row["n"] == 3
    assert row["seeds_missing"] == []
    assert row["complete"]
    assert len(row["checkpoint_ids"]) == 3


def test_different_budgets_modes_protocols_and_revisions_never_merge():
    records = [record(), record(train_k=16), record(eval_k=16),
               record(mode="another-protocol"), record(protocol="test-to-test"),
               record(comparison_id="data-revision-b")]
    rows = report.aggregate_records(records)
    assert len(rows) == len(records)
    assert all(row["n"] == 1 and row["seeds_missing"] == [] for row in rows)


def test_same_seed_with_two_checkpoint_evaluations_is_rejected():
    first = record()
    second = {**first, "evaluation_id": "different-evaluation", "checkpoint_id": "other-weights"}
    with pytest.raises(ValueError, match="one benchmark condition and seed"):
        report.aggregate_records([first, second])


def test_identical_evaluation_is_deduplicated_but_conflicts_are_rejected():
    first = record()
    row, = report.aggregate_records([first, dict(first)])
    assert row["n"] == 1
    conflicting = {**first, "metrics": {**first["metrics"], "recall_at_1": 0.1}}
    with pytest.raises(ValueError, match="conflicting records"):
        report.aggregate_records([first, conflicting])


def test_dense_reference_is_one_measurement_without_missing_training_seeds():
    row, = report.aggregate_records([record("dense")])
    assert row["n"] == 1
    assert row["seeds_present"] == row["seeds_missing"] == []
    assert row["recall_at_1_std"] is None
    assert row["train_k"] is row["eval_k"] is None


def test_paired_differences_use_only_common_seeds():
    records = [record("csr", 1227, 0.5), record("csr", 2026, 0.6),
               record("mpsae", 1227, 0.6), record("mpsae", 2026, 0.8),
               record("mpsae", 2027, 0.99)]
    row, = report.paired_differences(records, expected_seeds=(1227, 2026, 2027))
    assert row["delta_recall_at_1_mean"] == pytest.approx(0.15)
    assert row["delta_recall_at_1_std"] == pytest.approx(math.sqrt(0.005))
    assert row["delta_recall_at_1_n"] == row["n"] == 2
    assert row["seeds_paired"] == [1227, 2026]
    assert row["seeds_unpaired"] == row["seeds_missing"] == [2027]
    assert len(row["evaluation_pairs"]) == 2


def test_paired_differences_never_cross_dataset_revision():
    assert report.paired_differences([
        record("csr", comparison_id="one"), record("mpsae", comparison_id="two")]) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 70.0])
def test_nonfinite_or_percentage_accuracies_cannot_be_reported(value):
    with pytest.raises(ValueError):
        report.aggregate_records([record(value=value)])


def test_tables_and_manifest_survive_unavailable_matplotlib(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError("matplotlib unavailable")

    monkeypatch.setattr(report, "_render_figures", unavailable)
    unfinished = {**record("csr"), "complete": False}
    manifest = report.build_report([record(), unfinished], tmp_path,
                                   expected_points=3, failures=[{"task": "failed-task"}],
                                   expected_seeds=(1227, 2026, 2027))
    assert manifest["complete"] is False
    assert manifest["counts"]["completed_points"] == 1
    assert manifest["counts"]["missing_points"] == 2
    assert manifest["incomplete_evaluation_ids"] == [unfinished["evaluation_id"]]
    assert "matplotlib unavailable" in manifest["skipped"][0]
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    with (tmp_path / "tables" / "retrieval.csv").open(encoding="utf-8", newline="") as handle:
        row, = list(csv.DictReader(handle))
    assert float(row["recall_at_1_mean"]) == pytest.approx(0.7)
    assert json.loads(row["seeds_missing"]) == [2026, 2027]


def test_report_uses_only_supplied_results_and_records_unknown_completeness(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_render_figures", lambda *args: ({}, [], {}))
    (tmp_path / "stale-evaluation.json").write_text(json.dumps(record("csr")), encoding="utf-8")
    manifest = report.build_report([record(), record()], tmp_path, expected_seeds=(1227,))
    assert manifest["counts"]["provided_records"] == 2
    assert manifest["counts"]["completed_points"] == 1
    assert manifest["complete"] is None
    rows = json.loads((tmp_path / "tables" / "retrieval.json").read_text(encoding="utf-8"))
    assert len(rows) == 1 and rows[0]["method"] == "mpsae"
    assert rows[0]["seeds_missing"] == []


def test_report_complete_requires_the_expected_evaluations(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_render_figures", lambda *args: ({}, [], {}))
    manifest = report.build_report([record()], tmp_path, expected_points=1, expected_seeds=(1227,))
    assert manifest["complete"] is True
    with pytest.raises(ValueError, match="more supplied evaluations"):
        report.build_report([record(), record("csr")], tmp_path, expected_points=1)


def test_report_cannot_claim_completion_when_plot_export_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_render_figures", lambda *args:
                        ({}, ["quality: RuntimeError: PDF export failed"], {}))
    manifest = report.build_report([record()], tmp_path, expected_points=1, expected_seeds=(1227,))
    assert not manifest["complete"]
    assert manifest["plot_failures"] == ["quality: RuntimeError: PDF export failed"]
    assert manifest["counts"]["completed_points"] == 1


def test_expected_point_count_does_not_replace_seed_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_render_figures", lambda *args: ({}, [], {}))
    manifest = report.build_report([record(seed=999)], tmp_path, expected_points=1,
                                   expected_seeds=(1227,))
    assert not manifest["complete"]


def test_raw_evaluation_table_retains_metric_policies_and_per_class_results(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_render_figures", lambda *args: ({}, [], {}))
    item = record()
    item["metrics"].update(empty_query_policy="miss", per_class={"rose": {"recall_at_1": 0.8}},
                           macro_recall_at_10=0.9)
    manifest = report.build_report([item], tmp_path, expected_points=1, expected_seeds=(1227,))
    raw = json.loads((tmp_path / "tables" / "evaluations.json").read_text(encoding="utf-8"))
    assert raw[0]["metrics"]["empty_query_policy"] == "miss"
    assert raw[0]["metrics"]["per_class"] == {"rose": {"recall_at_1": 0.8}}
    aggregate = json.loads((tmp_path / "tables" / "retrieval.json").read_text(encoding="utf-8"))
    assert aggregate[0]["macro_recall_at_10_mean"] == pytest.approx(0.9)


def test_figures_render_and_export_when_matplotlib_is_installed(tmp_path):
    pytest.importorskip("matplotlib")
    records = [record("dense"), record("csr", value=0.6), record("mpsae", value=0.7)]
    manifest = report.build_report(records, tmp_path, expected_points=3, expected_seeds=(1227,))
    assert len(manifest["figures"]) == 4, manifest["skipped"]
    assert manifest["skipped"] == []
    for paths in manifest["figures"].values():
        assert {path.rsplit(".", 1)[-1] for path in paths} == {"png", "pdf", "txt"}
        assert all(__import__("pathlib").Path(path).is_file() for path in paths)
        caption = next(path for path in paths if path.endswith(".txt"))
        assert "seed 1227 only (n=1)" in __import__("pathlib").Path(caption).read_text(encoding="utf-8")
