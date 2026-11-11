"""Classification reports preserve DINO voting settings and inference budgets."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mpsae.benchmark import report


def classification(method="mpsae", *, seed=1227, eval_k=8, neighbors=(1, 20),
                   temperature=0.07, value=0.7, identity=""):
    dense = method == "dense"
    classifier = {"name": "dino_weighted_knn", "n_neighbors": list(neighbors),
                  "temperature": temperature, "empty_query_policy": "counted_and_scored_as_miss",
                  "tie_rule": "score_desc_gallery_position_asc",
                  "class_tie_rule": "class_label_asc"}
    return {"dataset": "eurosat", "protocol": "train_support_test_query",
            "comparison_id": "same-supplied-comparison", "task": "knn_classification",
            "classifier": classifier, "backbone": "ff2048", "method": method,
            "seed": None if dense else seed, "train_k": None if dense else 8,
            "eval_k": None if dense else eval_k, "mode": "frozen", "complete": True,
            "evaluation_id": f"{method}-{seed}-{eval_k}-{temperature}-{neighbors}-{identity}",
            "checkpoint_id": f"weights-{method}-{seed}",
            "metrics": {**{f"knn_{n}_{field}": value for n in neighbors
                            for field in ("top1", "top5", "balanced_accuracy", "macro_f1")},
                        "query_count": 100, "gallery_count": 200, "seconds": 0.5,
                        "voting_seconds": 0.01, "nnz_mean": min(eval_k, 7),
                        "code_bytes_per_image": eval_k * 8}}


def retrieval():
    record = classification(identity="old-retrieval")
    record.pop("classifier")
    record.pop("task")  # old saved results still mean retrieval
    record["metrics"] = {"recall_at_1": 0.4, "map_at_r": 0.2}
    return record


def test_classification_dynamic_metrics_have_seed_stats_and_shared_cost():
    rows = report.aggregate_records([
        classification(seed=1227, neighbors=(1, 5), value=0.6),
        classification(seed=2026, neighbors=(1, 5), value=0.8)], expected_seeds=(1227, 2026))
    row, = rows
    assert row["task"] == "knn_classification"
    for neighbors in (1, 5):
        for metric in ("top1", "top5", "balanced_accuracy", "macro_f1"):
            assert row[f"knn_{neighbors}_{metric}_mean"] == pytest.approx(0.7)
            assert row[f"knn_{neighbors}_{metric}_std"] == pytest.approx(2 ** 0.5 * 0.1)
            assert row[f"knn_{neighbors}_{metric}_n"] == 2
    assert row["search_ms_per_query_mean"] == 5.0
    assert row["voting_seconds_mean"] == 0.01
    assert not any(key.startswith("recall_at") for key in row)


@pytest.mark.parametrize("change", [
    {"temperature": 0.2}, {"n_neighbors": [1, 5]},
    {"empty_query_policy": "other-policy"}, {"class_tie_rule": "other-rule"},
])
def test_same_comparison_id_cannot_merge_or_pair_distinct_classifier_settings(change):
    left = classification("csr")
    right = classification("mpsae", neighbors=tuple(change.get("n_neighbors", (1, 20))))
    right["classifier"].update(change)
    assert len(report.aggregate_records([left, right])) == 2
    assert report.paired_differences([left, right]) == []
    # Two same-seed evaluations are valid distinct conditions when the classifier changes.
    right["method"] = "csr"
    assert len(report.aggregate_records([left, right])) == 2


def test_same_classifier_pairs_active_methods_at_each_inference_k():
    records = [classification(method, eval_k=k, value=value)
               for k in (8, 32) for method, value in (("csr", 0.5), ("mpsae", 0.7))]
    rows = report.paired_differences(records)
    assert [row["eval_k"] for row in rows] == [8, 32]
    assert all(row["delta_knn_20_top1_mean"] == pytest.approx(0.2) for row in rows)
    assert all(row["classifier"]["temperature"] == 0.07 for row in rows)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 70, True])
def test_classifier_metrics_reject_nonfinite_percentage_and_boolean_values(value):
    record = classification()
    record["metrics"]["knn_20_macro_f1"] = value
    with pytest.raises((ValueError, TypeError)):
        report.aggregate_records([record])


@pytest.mark.parametrize("change", [{"temperature": 0}, {"temperature": float("nan")},
                                    {"n_neighbors": [1, 1]}, {"n_neighbors": [True]},
                                    {"name": "unknown-vote"}])
def test_invalid_classifier_specification_is_rejected(change):
    record = classification()
    record["classifier"].update(change)
    with pytest.raises(ValueError):
        report.aggregate_records([record])


def test_neighbor_metrics_must_match_declared_neighbors_and_task():
    item = classification()
    item["metrics"]["knn_5_top1"] = 0.8
    with pytest.raises(ValueError, match="declared"):
        report.aggregate_records([item])
    item = classification()
    item["metrics"].pop("knn_20_top1")
    with pytest.raises(ValueError, match="every declared"):
        report.aggregate_records([item])
    item = classification()
    item["metrics"]["recall_at_1"] = 0.8
    with pytest.raises(ValueError, match="retrieval accuracy"):
        report.aggregate_records([item])


@pytest.mark.parametrize("field,value", [("temperature", 0.2), ("n_neighbors", [1, 5]),
                                        ("empty_query_policy", "predict-class-zero")])
def test_classifier_header_cannot_relabel_metric_policy(field, value):
    item = classification()
    item["metrics"][field] = value
    with pytest.raises(ValueError, match="disagrees"):
        report.aggregate_records([item])


def test_mixed_report_separates_tasks_and_keeps_one_record_per_representation(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_render_figures", lambda *args: ({}, [], {}))
    records = [retrieval(), classification(), classification("dense"),
               {"method": "mpsaev2", "metrics": "not read"},
               {"method": "csr_multitopk", "metrics": "not read"}]
    original = deepcopy(records)
    manifest = report.build_report(records, tmp_path, expected_points=3)
    assert manifest["complete"]
    assert manifest["counts"]["completed_points"] == 3
    assert records == original
    tables = {name: json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
              for name, paths in manifest["tables"].items()}
    assert len(tables["retrieval"]) == 1
    assert len(tables["classification"]) == 2
    assert len(tables["evaluations"]) == 3
    assert tables["retrieval"][0]["task"] == "retrieval"
    assert {row["method"] for row in tables["classification"]} == {"dense", "mpsae"}
    assert tables["classification"][0]["classifier"]["n_neighbors"] == [1, 20]


def test_plots_export_distinct_neighbor_panels_without_reference_curves(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from mpsae.report import style
    captured = []
    original = style.save_figure

    def save(fig, output, name, *, caption):
        captured.append((name, [ax.get_title() for ax in fig.axes],
                         [ax.get_ylabel() for ax in fig.axes],
                         [line.get_label() for ax in fig.axes for line in ax.lines]))
        return original(fig, output, name, caption=caption)

    monkeypatch.setattr(style, "save_figure", save)
    records = [classification(method, eval_k=k, value=value)
               for method, value in (("csr", 0.6), ("mpsae", 0.7), ("dense", 0.99))
               for k in ((8, 32) if method != "dense" else (8,))]
    manifest = report.build_report(records, tmp_path, expected_points=5)
    assert manifest["skipped"] == []
    assert len(manifest["figures"]) == 6
    main = next(value for value in captured if value[0] == "classification_ff2048_frozen_train-k8")
    assert main[1] == ["EuroSAT · 1-NN · T=0.07", "EuroSAT · 20-NN · T=0.07"]
    assert main[2] == ["Accuracy (%)", "Accuracy (%)"]
    assert set(main[3]) == {"CSR", "MPSAE"}
    balance = next(value for value in captured if value[0].startswith("classification_balance_"))
    assert balance[2] == ["Balanced acc. (%)", "Macro-F1 (%)"] * 2
    assert not any("Dense" in label or "MPSAEv2" in label or "CSR-MTK" in label
                   for _, _, _, labels in captured for label in labels)
    for paths in manifest["figures"].values():
        assert {Path(path).suffix for path in paths} == {".png", ".pdf", ".txt"}
        assert all(Path(path).is_file() and Path(path).stat().st_size for path in paths)


def test_classification_panels_are_paginated_without_combining_temperature_settings(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from mpsae.report import style
    captured = []

    def save(fig, output, name, *, caption):
        captured.append((name, [ax.get_title() for ax in fig.axes if ax.get_visible()], caption))
        return [output / f"{name}.pdf"]

    monkeypatch.setattr(style, "save_figure", save)
    rows = report.aggregate_records([classification(temperature=t) for t in (0.05, 0.07, 0.2)])
    paths = report._draw_classification(rows, tmp_path, "classification", "Fixed train K")
    assert [path.name for path in paths] == ["classification.pdf", "classification_part2.pdf"]
    assert all(len(titles) <= 4 for _, titles, _ in captured)
    assert all(caption.startswith(f"Part {i} of 2.") for i, (_, _, caption) in enumerate(captured, 1))
    assert len({title for _, titles, _ in captured for title in titles}) == 6
