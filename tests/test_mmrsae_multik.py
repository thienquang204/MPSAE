"""CPU contracts for angular objectives, CSRv2 and multi-budget reporting.

Retrieval is mocked: these tests do not start CUDA, download data or fit ImageNet.
"""
import contextlib
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

import mmrsae_imagenet as runner
import mmrsae_reporting as reporting
import mpsae_repair as core
from mmrsae import METHODS, MMRSAE, MMRSAEConfig
from repair_data import atomic_json


class AngularAndScheduleTests(unittest.TestCase):
    def test_all_methods_reject_neighborhood_ot_overrides(self):
        for method in METHODS:
            self.assertEqual(MMRSAEConfig(input_dim=6, hidden_dim=32, k=2,
                                         variant=method).mixture_weight, 0.)
            with self.assertRaisesRegex(ValueError, "angular|Baseline"):
                MMRSAEConfig(input_dim=6, hidden_dim=32, k=2,
                             variant=method, mixture_weight=.2)
        with self.assertRaisesRegex(ValueError, "angular OT only"):
            core.research_loss(None, torch.ones(2, 6), 2, mixture_weight=.2)

    def test_angular_dictionary_objective_needs_no_kl_bank_and_backpropagates(self):
        model = MMRSAE(MMRSAEConfig(input_dim=6, hidden_dim=32, k=2,
            mask=False, relation_weight=0., microbatch=4))
        x = torch.linspace(.1, 2.4, 24).reshape(4, 6)
        model.initialize_dictionary(x)
        with mock.patch.object(core, "neighborhood_cost", side_effect=AssertionError("neighborhood OT called")):
            loss, stats, _ = model.loss(x, update_activity=False)
            loss.backward()
        self.assertFalse(model.needs_bank)
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("costnbr_mean", stats)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_csrv2_changes_training_budget_and_matches_csr_at_endpoint(self):
        common = dict(input_dim=6, hidden_dim=32, k=2, anneal_start_k=16)
        csr = MMRSAE(MMRSAEConfig(**common, variant="csr"))
        v2 = MMRSAE(MMRSAEConfig(**common, variant="csrv2"))
        budgets = [v2.config.training_k(step, 10) for step in range(10)]
        self.assertEqual(budgets[0], 16)
        self.assertEqual(budgets[6:], [2] * 4)
        self.assertEqual(budgets, sorted(budgets, reverse=True))
        self.assertIsNone(v2.repair)
        x = torch.linspace(.1, 2.4, 24).reshape(4, 6)
        _, stats, _ = v2.loss(x, step=0, total_steps=10, update_activity=False)
        self.assertEqual(stats["training_k"], 16)
        fixed, _, _ = csr.loss(x, step=9, total_steps=10, update_activity=False)
        annealed, end_stats, _ = v2.loss(x, step=9, total_steps=10, update_activity=False)
        torch.testing.assert_close(fixed, annealed)
        fixed.backward()
        annealed.backward()
        for left, right in zip(csr.parameters(), v2.parameters()):
            torch.testing.assert_close(left.grad, right.grad)
        self.assertEqual(end_stats["training_k"], 2)

    def test_dictionary_inference_supports_every_requested_budget(self):
        model = MMRSAE(MMRSAEConfig(input_dim=6, hidden_dim=32, k=2))
        x = torch.linspace(.1, 2.4, 24).reshape(4, 6)
        model.initialize_dictionary(x)
        before = model.sae.inactive_steps.clone()
        for k in (1, 2, 4, 8, 16, 32):
            z = model.encode(x, k)
            self.assertEqual(z.shape, (4, 32))
            self.assertTrue((z.ge(0) & torch.isfinite(z)).all())
            self.assertTrue(z.gt(0).sum(1).le(k).all())
            indices, values = model.encode(x, k, compact=True)
            self.assertEqual(indices.shape, (4, k))
            self.assertEqual(values.dtype, torch.float32)
        torch.testing.assert_close(before, model.sae.inactive_steps)

    def test_mmrsae_anneals_training_k_and_all_representation_budgets(self):
        model = MMRSAE(MMRSAEConfig(input_dim=6, hidden_dim=128, k=8,
            variant="mmrsae-annealed", relation_weight=0., ot_weight=0., mask=False))
        x = torch.linspace(.1, 2.4, 24).reshape(4, 6)
        model.initialize_dictionary(x)
        schedule = [model.config.training_k(step, 10) for step in range(10)]
        self.assertEqual(schedule[0], 64)
        self.assertEqual(schedule[6:], [8] * 4)
        self.assertEqual(schedule, sorted(schedule, reverse=True))
        for step, expected in ((0, 64), (3, 36), (6, 8), (9, 8)):
            _, stats, views = model.loss(x, step=step, total_steps=10, update_activity=False)
            self.assertEqual(stats["training_k"], expected)
            self.assertEqual(stats["auxiliary_budget"], min(4 * expected, 128))
            for name, multiple in (("final_k", 1), ("final_2k", 2),
                                   ("raw_4k", 4), ("raw_auxiliary", 4)):
                torch.testing.assert_close(views[name], model.encode(x, min(multiple * expected, 128)))


class MultiBudgetEvaluationTests(unittest.TestCase):
    def arguments(self, *extra):
        return runner.build_parser().parse_args(list(extra))

    def test_auto_custom_and_invalid_budgets(self):
        args = self.arguments()
        self.assertEqual(runner.evaluation_budgets(args, 512), [2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.assertEqual(runner.evaluation_budgets(args, 2048)[-2:], [1024, 2048])
        args = self.arguments("--hidden-dim", "16", "--k", "2")
        self.assertEqual(runner.evaluation_budgets(args, 512), [2, 4, 8, 16])
        args.eval_topk = "4,1,8"
        self.assertEqual(runner.evaluation_budgets(args, 512), [1, 2, 4, 8])
        for invalid in ("", "8,8", "0,8", "-1", "32", "eight", "8,"):
            args.eval_topk = invalid
            with self.assertRaises(ValueError):
                runner.evaluation_budgets(args, 512)

    def test_plan_lists_budgets_without_loading_data_or_initializing_gpu(self):
        output = io.StringIO()
        with mock.patch("mmrsae_data.prepare_imagenet_features", side_effect=AssertionError("data opened")), \
                mock.patch("torch.cuda.init", side_effect=AssertionError("CUDA initialized")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(["--plan", "--variant", "csrv2"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["evaluation_topk"]["resnet50"][-1], 2048)
        self.assertEqual(plan["ot_cost"], "none")

    def test_preflight_checks_the_largest_inference_budget(self):
        with mock.patch("mmrsae_gpu.preflight", return_value={"gpu_search_probe": "passed"}) as preflight, \
                mock.patch("mmrsae_data.prepare_imagenet_features", side_effect=AssertionError("data opened")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--preflight-only", "--eval-topk", "8,128,1024"]), 0)
        self.assertEqual(preflight.call_args.args[1]["model"]["k"], 1024)

    def evaluate_fixture(self, root, fail_at=None):
        args = self.arguments("--variant", "csrv2", "--hidden-dim", "32", "--eval-topk", "2,4,16")
        model = MMRSAE(runner.method_config(args, 6))
        bundle = SimpleNamespace(train=None, train_labels=None, gallery_ids=[0, 1], query_ids=[2],
                                 query=None, query_labels=None, provenance={"identity": {"data": "fixed"}})
        checkpoint = root / "last.pt"
        checkpoint.write_bytes(b"checkpoint-identity")
        calls = []

        def benchmark(*positional, **kwargs):
            k = positional[6]["model"]["k"]
            calls.append((k, positional[4], positional[7]))
            if k == fail_at:
                raise RuntimeError("interrupted evaluation")
            return {"K": k, "top1": 50. + k / 10, "total": 10, "correct": 5,
                    "actual_nnz_mean": k, "empty_query_codes": 0, "empty_query_fraction": 0.,
                    "empty_gallery_codes": 0}

        with mock.patch("repair_benchmark.benchmark", side_effect=benchmark), \
                mock.patch.object(runner, "log_wandb_metrics"), mock.patch.object(runner, "update_wandb_summary"):
            if fail_at is not None:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    runner.evaluate(args, bundle, model, root, checkpoint, {"data_sha256": "data"}, 5)
                result = json.loads((root / "retrieval/endpoint/result.json").read_text())
            else:
                result = runner.evaluate(args, bundle, model, root, checkpoint, {"data_sha256": "data"}, 5)
        return result, calls, model

    def test_each_budget_is_evaluated_and_persisted_with_the_same_model(self):
        with tempfile.TemporaryDirectory(prefix="mk_") as directory:
            root = Path(directory)
            result, calls, model = self.evaluate_fixture(root)
            self.assertEqual([item[0] for item in calls], [2, 4, 8, 16])
            self.assertTrue(all(item[1] is model.sae for item in calls))
            self.assertEqual(set(result["per_topk"]), {"2", "4", "8", "16"})
            self.assertEqual(result["top1"], result["per_topk"]["8"]["top1"])
            self.assertTrue(result["complete"])
            for k in result["evaluation_topk"]:
                self.assertTrue((root / f"retrieval/endpoint/k_{k}/result.json").is_file())

    def test_interrupted_sweep_keeps_completed_budgets(self):
        with tempfile.TemporaryDirectory(prefix="mk_") as directory:
            result, _, _ = self.evaluate_fixture(Path(directory), fail_at=16)
            self.assertFalse(result["complete"])
            self.assertEqual(set(result["per_topk"]), {"2", "4", "8"})


class ReportingTests(unittest.TestCase):
    def fixture(self, root, method, retrieval, *, complete=True):
        trial = root / method
        atomic_json(trial / "trial.json", {"settings": {"variant": method, "k": 8,
            "evaluation": "dev", "eval_topk": "auto"}, "code": {"source": "same"}})
        if complete:
            atomic_json(trial / "resnet50/summary.json", {"complete": True, "updates": 5,
                        "retrieval": retrieval})
        else:
            atomic_json(trial / "resnet50/retrieval/endpoint/result.json", retrieval)
            atomic_json(trial / "resnet50/failure.json", {"error": "interrupted"})

    def test_legacy_and_partial_results_do_not_invent_accuracy_points(self):
        with tempfile.TemporaryDirectory(prefix="mr_") as directory:
            root = Path(directory)
            self.fixture(root, "csr", {"K": 8, "top1": 55.})
            self.fixture(root, "csrv2", {"per_topk": {"8": {"top1": 56.}, "16": {"top1": 57.}},
                         "complete": False}, complete=False)
            groups = reporting.collect(root)
            self.assertEqual(len(groups), 1)
            rows = reporting.budget_rows(next(iter(groups.values())))
            self.assertEqual({(r["method"], r["K"]) for r in rows}, {("csr", 8), ("csrv2", 8), ("csrv2", 16)})

    def test_full_report_contains_all_methods_budgets_and_real_figure_files(self):
        with tempfile.TemporaryDirectory(prefix="mr_") as directory:
            root = Path(directory)
            for offset, method in enumerate(reporting.METHODS):
                points = {str(k): {"K": k, "top1": 50. + offset + k / 100,
                          "total": 100, "empty_query_fraction": 0.} for k in (8, 16, 32)}
                self.fixture(root, method, {**points["8"], "per_topk": points,
                             "budget_evaluation": "single_checkpoint_multi_k"})
            with contextlib.redirect_stdout(io.StringIO()):
                index = reporting.report(root)
            group = next(iter(index.values()))
            target = Path(group["directory"])
            with (target / "results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3 * len(reporting.METHODS))
            self.assertEqual({r["method"] for r in rows}, set(reporting.METHODS))
            self.assertEqual({int(r["K"]) for r in rows}, {8, 16, 32})
            self.assertEqual(group["evaluation_topk"], [8, 16, 32])
            for filename in ("top1_by_k.png", "top1_by_k.pdf", "top1_by_method.png", "results.md"):
                self.assertGreater((target / filename).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
