"""Authored source checks for the stage/gate contracts; no real data or GPU needed.

These tests intentionally exercise decision logic with fake measurements.
They do not establish accuracy, solver convergence, or CUDA correctness.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from make_plan import load_config
from mpsae_repair_experiment import ResearchRun, gate_decision, screening_decision, require_unchanged
from repair_data import canonical_hash


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_triage_accepts_simpler_relation_candidate(self):
        rows = {
            "C": {"top1": 50.0},
            "S": {"top1": 50.4, "diagnostics_acceptable": True, "encoding_ratio_to_C": 2.0},
            "N": {"top1": 50.2, "diagnostics_acceptable": True, "encoding_ratio_to_C": 2.8},
        }
        result = screening_decision(rows, self.config)
        self.assertTrue(result["promising"])
        self.assertTrue(result["arms"]["S"]["promising"])
        self.assertFalse(result["arms"]["N"]["promising"])

    def test_missing_signal_or_cost_never_passes(self):
        rows = {
            "C": {"top1": 50.0},
            "S": {"top1": 55.0, "encoding_ratio_to_C": 1.0},
            "N": {"top1": 55.0, "diagnostics_acceptable": True},
        }
        self.assertFalse(screening_decision(rows, self.config)["promising"])
        rows["S"]["diagnostics_acceptable"] = True
        rows["S"]["encoding_ratio_to_C"] = float("nan")
        rows["N"]["encoding_ratio_to_C"] = 3.01
        self.assertFalse(screening_decision(rows, self.config)["promising"])

    def test_locked_gate_requires_strictly_positive_interval(self):
        comparison = {"delta_percentage_points": .4, "ci_lower_percentage_points": 0.0}
        self.assertFalse(gate_decision(comparison, True, True, True, self.config)["promoted"])
        comparison["ci_lower_percentage_points"] = .001
        self.assertTrue(gate_decision(comparison, True, True, True, self.config)["promoted"])
        for flags in ((False, True, True), (True, False, True), (True, True, False)):
            self.assertFalse(gate_decision(comparison, *flags, self.config)["promoted"])
        comparison["delta_percentage_points"] = .299
        self.assertFalse(gate_decision(comparison, True, True, True, self.config)["promoted"])

    def test_wide_stage_requires_the_n_family_to_pass(self):
        run = object.__new__(ResearchRun)
        run.decisions = {"triage": {"promising": True, "arms": {"S": {"promising": True}, "N": {"promising": False}}}}
        run.require_promising()
        with self.assertRaises(ValueError):
            run.require_promising("N")


class LockTests(unittest.TestCase):
    def make_run(self):
        run = object.__new__(ResearchRun)
        run.config = load_config()
        run.output = Path("unused_source_test_output")
        run.args = SimpleNamespace(candidate="N")
        run.identity = {"source": "locked-source", "code": "locked-code", "configuration": "locked-config"}
        locked = {"candidate": "N", "identity": copy.deepcopy(run.identity),
                  "checkpoints": {"C": "c-digest", "S": "s-digest", "N": "n-digest"},
                  "gate_policy": copy.deepcopy(run.config["gates"])}
        locked["lock_sha256"] = canonical_hash(locked)
        run.decisions = {"lock": locked}
        return run

    def test_lock_checks_candidate_code_and_checkpoint_identity(self):
        run = self.make_run()
        checksums = run.decisions["lock"]["checkpoints"]
        with mock.patch("mpsae_repair_experiment.sha256_file", side_effect=lambda path: checksums[path.parent.name]):
            self.assertEqual(run.verify_lock("N")["candidate"], "N")
            with self.assertRaisesRegex(ValueError, "candidate cannot change"):
                run.verify_lock("S")
        with mock.patch("mpsae_repair_experiment.sha256_file", return_value="different"):
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                run.verify_lock("N")
        run.identity["code"] = "changed-evaluation-code"
        with self.assertRaisesRegex(ValueError, "configuration/code/source"):
            run.verify_lock("N")

    def test_training_is_closed_after_candidate_lock(self):
        run = self.make_run()
        with self.assertRaisesRegex(ValueError, "locked"):
            run.require_unlocked()

    def test_holdout_marker_precedes_first_query_read_and_survives_failure(self):
        run = self.make_run()
        locked = copy.deepcopy(run.decisions["lock"])
        run.verify_lock = mock.Mock(return_value=locked)
        order = []
        run.save_decisions = mock.Mock(side_effect=lambda: order.append("durable-open-marker"))
        def failed_query(*args, **kwargs):
            order.append("query-evaluation")
            self.assertEqual(run.decisions["holdout"]["status"], "opened")
            raise RuntimeError("simulated interruption")
        run.evaluate = mock.Mock(side_effect=failed_query)
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            run.holdout()
        self.assertEqual(order, ["durable-open-marker", "query-evaluation"])
        self.assertEqual(run.decisions["holdout"]["lock_sha256"], locked["lock_sha256"])

    def test_completed_gate_reuses_report_without_query_evaluation(self):
        run = self.make_run()
        locked = run.decisions["lock"]
        run.verify_lock = mock.Mock(return_value=locked)
        previous = {"status": "complete", "lock_sha256": locked["lock_sha256"], "verdict": {"promoted": False}}
        run.decisions["holdout"] = previous
        run.evaluate = mock.Mock(side_effect=AssertionError("completed gate must not be searched again"))
        self.assertIs(run.holdout(), previous)
        run.evaluate.assert_not_called()

    def test_previously_opened_gate_cannot_be_rebound(self):
        run = self.make_run()
        run.verify_lock = mock.Mock(return_value=run.decisions["lock"])
        run.decisions["holdout"] = {"status": "opened", "lock_sha256": "another-lock"}
        run.evaluate = mock.Mock()
        with self.assertRaisesRegex(ValueError, "already opened"):
            run.holdout()
        run.evaluate.assert_not_called()

    def test_existing_configuration_is_immutable(self):
        with mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("mpsae_repair_experiment.read_json", return_value={"seed": 1227}), \
             mock.patch("mpsae_repair_experiment.atomic_json") as writer:
            require_unchanged("unused.json", {"seed": 1227})
            writer.assert_not_called()
            with self.assertRaisesRegex(ValueError, "identity differs"):
                require_unchanged("unused.json", {"seed": 1228})


if __name__ == "__main__":
    unittest.main()
