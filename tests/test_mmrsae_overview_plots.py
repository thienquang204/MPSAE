"""Measured-data contracts for the overview plots (no CUDA or datasets)."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mmrsae_reporting as reporting
from repair_data import atomic_json


def timed_point(k=8, *, gpu="test GPU", batch=32, top1=61., **extra):
    return {"K": k, "top1": top1, "actual_nnz_mean": k - .5, "correct": 61, "total": 100,
            "timed": True, "mean_retrieval_milliseconds_per_batch": .5 + k / 100,
            "device_name": gpu, "torch_version": "test", "cuda_version": "test", "dtype": "float32",
            "query_batch_size": batch, "gallery_count": 1000, "query_count": 100,
            "gallery_ids_sha256": "gallery", "query_ids_sha256": "queries",
            "timing": {"version": 2, "primary_statistic": "median_of_interleaved_repeat_totals",
                       "cache_policy": "warm_repeated_search_of_each_resident_query_batch",
                       "scope": "resident_search_only", "warmup_per_shape": 3, "repeats_per_batch": 5},
            **extra}


class OverviewDataTests(unittest.TestCase):
    def test_only_warm_repeated_timing_is_used_and_cohorts_exclude_k(self):
        small = reporting._timing_fields(timed_point(8))
        large = reporting._timing_fields(timed_point(256))
        self.assertEqual(small["timing_cohort"], large["timing_cohort"])
        for change in ({"gpu": "other GPU"}, {"batch": 64}, {"gallery_ids_sha256": "other rows"}):
            self.assertNotEqual(small["timing_cohort"], reporting._timing_fields(timed_point(**change))["timing_cohort"])
        for invalid in ({"timed": False}, {"timing": {}}, {"mean_retrieval_milliseconds_per_batch": -1},
                        {"mean_retrieval_milliseconds_per_batch": float("nan")}):
            result = reporting._timing_fields(timed_point(**invalid))
            self.assertIsNone(result["retrieval_batch_ms"])
            self.assertIsNone(result["timing_cohort"])

    def test_compute_macro_average_requires_same_k_and_entire_backbone_set(self):
        def point(method, backbone, k, top1, hours=2.):
            return {"method": method, "backbone": backbone, "K": k, "top1": top1,
                    "training_gpu_hours": hours, "training_time_scope": "updates_and_key_refresh",
                    "training_device_name": "same GPU"}
        rows = [point("dlmmsae", "resnet50", 8, 70., 3.), point("dlmmsae", "resnet18", 8, 50., 1.),
                point("csr", "resnet50", 8, 60.), point("csr", "resnet18", 16, 55.),
                point("mmrsae", "resnet50", 8, 62.), point("mmrsae", "resnet18", 8, 58., None)]
        compute = reporting.compute_rows(rows)
        averages = [row for row in compute if row["aggregation"] == "macro_accuracy_sum_compute"]
        self.assertEqual(len(averages), 1)
        self.assertEqual(averages[0]["method"], "dlmmsae")
        self.assertEqual(averages[0]["K"], 8)
        self.assertEqual(averages[0]["top1"], 60.)
        self.assertEqual(averages[0]["training_gpu_hours"], 4.)
        self.assertEqual(averages[0]["backbone_set"], "resnet50+resnet18")

    def test_fit_wall_seconds_never_substitute_for_training_gpu_hours(self):
        rows = [{"method": "csr", "backbone": "resnet50", "K": 8, "top1": 60., "fit_seconds": 3600.}]
        self.assertEqual(reporting.compute_rows(rows), [])

    def test_new_method_styles_are_distinct_and_stable(self):
        self.assertIn("dlmmsae", reporting.METHODS)
        self.assertIn("dlmmsaev2", reporting.METHODS)
        self.assertEqual(len(set(reporting._style(method) for method in reporting.METHODS)), len(reporting.METHODS))

    def test_classification_and_retrieval_use_independent_values_and_dimensions(self):
        point = {"method": "dlmmsae", "backbone": "resnet50", "K": 8, "top1": 62.,
                 "classification_top1": 74., "classification_actual_nnz_mean": 7.25,
                 "actual_nnz_mean": 6.5, "full_parent_top1": 72., "full_parent_classification_top1": 80.}
        plt = reporting._pyplot()
        captured = []

        def capture(unused, fig, target, stem):
            captured.append((stem, [(list(ax.lines[0].get_xdata()), list(ax.lines[0].get_ydata()), ax.get_ylabel())
                                    for ax in fig.axes]))
            plt.close(fig)
            return []

        with mock.patch.object(reporting, "_save", side_effect=capture):
            reporting.plot_active_dimensions(plt, [point], Path("unused"))
        self.assertEqual(captured[0][1][0][:2], ([7.25], [74.]))
        self.assertEqual(captured[0][1][1][:2], ([6.5], [62.]))
        self.assertIn("Linear-probe", captured[0][1][0][2])
        self.assertIn("1-NN", captured[0][1][1][2])


class OverviewReportTests(unittest.TestCase):
    def fixture(self, root, method, *, hardware="same GPU", with_measurements=True):
        trial = root / method
        atomic_json(trial / "trial.json", {"settings": {"variant": method, "k": 8,
                    "evaluation": "dev", "eval_topk": "auto"}, "code": {"source": "same"}})
        atomic_json(trial / "gpu_preflight.json", {"device_name": hardware, "torch_version": "same",
                                                 "device_total_bytes": 1234})
        for backbone in reporting.BACKBONES:
            points = {str(k): timed_point(k, top1=60. + k / 100) for k in (8, 16, 32)}
            summary = {"complete": True, "updates": 5, "fit_seconds": 30000.,
                       "retrieval": {**points["8"], "per_topk": points}, "full_parent": timed_point(512, top1=75.)}
            if with_measurements:
                summary.update({"training_gpu_hours": 2., "training_seconds": 7200.,
                                "training_device_name": hardware, "training_time_scope": "updates_and_key_refresh",
                                "classification": {"protocol": "frozen-linear-probe-v1",
                                  "per_topk": {str(k): {"top1": 70. + k / 100, "actual_nnz_mean": k - 1.,
                                                       "query_actual_nnz_mean": k - .25} for k in (8, 16, 32)},
                                  "full_parent": {"top1": 80.}}})
            atomic_json(trial / backbone / "summary.json", summary)

    def test_hardware_changes_split_comparison_groups(self):
        with tempfile.TemporaryDirectory(prefix="overview_") as directory:
            root = Path(directory)
            self.fixture(root, "dlmmsae", hardware="GPU A")
            self.fixture(root, "csr", hardware="GPU B")
            self.assertEqual(len(reporting.collect(root)), 2)

    def test_report_renders_all_requested_views_and_exports_measured_numbers(self):
        with tempfile.TemporaryDirectory(prefix="overview_") as directory:
            root = Path(directory)
            self.fixture(root, "dlmmsae")
            self.fixture(root, "dlmmsaev2")
            with contextlib.redirect_stdout(io.StringIO()):
                result = reporting.report(root)
            group = next(iter(result.values()))
            target = Path(group["directory"])
            for stem in ("top1_by_k", "top1_by_k_backbones", "classification_top1_by_k", "accuracy_by_active_dim",
                         "top1_vs_retrieval_ms", "top1_vs_gpu_hours_resnet50", "top1_vs_gpu_hours_resnet18",
                         "top1_vs_gpu_hours_resnet50_and_resnet18"):
                for suffix in ("png", "pdf"):
                    self.assertGreater((target / f"{stem}.{suffix}").stat().st_size, 100)
                    self.assertIn(f"{stem}.{suffix}", group["figures"])
            coverage = json.loads((target / "plot_coverage.json").read_text())
            self.assertEqual(coverage["retrieval_accuracy_points"], 12)
            self.assertEqual(coverage["classification_accuracy_points"], 12)
            self.assertEqual(coverage["warm_search_timing_points"], 12)
            self.assertEqual(coverage["macro_average_compute_points"], 6)
            with (target / "results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            first = rows[0]
            self.assertEqual(float(first["classification_top1"]), 70.08)
            self.assertEqual(float(first["top1"]), 60.08)
            self.assertEqual(float(first["classification_actual_nnz_mean"]), 7.75)
            self.assertEqual(float(first["actual_nnz_mean"]), 7.5)
            self.assertEqual(float(first["training_gpu_hours"]), 2.)
            self.assertTrue((target / "overview.md").is_file())

    def test_legacy_summary_has_missing_classification_and_compute_not_zero(self):
        with tempfile.TemporaryDirectory(prefix="overview_") as directory:
            root = Path(directory)
            self.fixture(root, "csr", with_measurements=False)
            group = next(iter(reporting.collect(root).values()))
            rows = reporting.budget_rows(group)
            self.assertTrue(all(row["classification_top1"] is None for row in rows))
            self.assertTrue(all(row["training_gpu_hours"] is None for row in rows))
            self.assertEqual(reporting.compute_rows(rows), [])

    def test_only_selected_methods_are_reported_and_old_annealed_name_is_readable(self):
        with tempfile.TemporaryDirectory(prefix="overview_") as directory:
            root = Path(directory)
            for method in ("csr", "csrv2", "dlmmsae", "dlmmsae-annealed",
                           "mmrsae", "mmrsae-rel", "mmrsae-wide", "mmrsae-annealed", "mpsae"):
                self.fixture(root, method)
            group = next(iter(reporting.collect(root).values()))
            self.assertEqual(set(group["trials"]), {"csr", "csrv2", "dlmmsae", "dlmmsaev2"})
            rows = reporting.budget_rows(group)
            self.assertEqual({row["method"] for row in rows}, set(reporting.METHODS))
            self.assertEqual(len(rows), 4 * 2 * 3)
            self.assertTrue(all(row["top1"] == 60. + row["K"] / 100 for row in rows))
            source = json.loads((root / "dlmmsae-annealed/trial.json").read_text())
            self.assertEqual(source["settings"]["variant"], "dlmmsae-annealed")
            self.assertEqual(reporting._label("dlmmsaev2"), "DLMMSAEv2")
            self.assertEqual(reporting._label("csrv2"), "CSRv2")


if __name__ == "__main__":
    unittest.main()
