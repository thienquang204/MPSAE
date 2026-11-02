"""Linear-probe classification contracts, with no GPU or filesystem fixtures."""
import contextlib
import copy
import io
import json
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import mmrsae_imagenet as runner
import mmrsae_probe as probe


class MemoryFiles:
    """Minimal paths for probing checkpoint/result behavior without disk I/O."""
    def __init__(self):
        self.files = {}
        owner = self

        class MemoryPath(str):
            def __new__(cls, value):
                return str.__new__(cls, str(value).replace("\\", "/"))

            def __truediv__(self, other):
                return type(self)(str(self).rstrip("/") + "/" + str(other))

            def exists(self):
                return str(self) in owner.files

            def mkdir(self, **unused):
                pass

            def read_text(self, **unused):
                return json.dumps(owner.files[str(self)])

        self.Path = MemoryPath

    def save(self, path, value):
        self.files[str(path)] = copy.deepcopy(value)

    def load(self, path, *, expected_identity):
        value = copy.deepcopy(self.files[str(path)])
        if value["identity"] != expected_identity:
            raise ValueError("Unexpected fixture checkpoint identity")
        return value


class ProbeTrainingTests(unittest.TestCase):
    def fixture(self):
        values = np.repeat(np.eye(2, dtype=np.float32), 12, axis=0)
        labels = np.repeat(np.arange(2, dtype=np.int64), 12)
        return {"values": values}, labels, {"values": np.eye(2, dtype=np.float32)}, np.arange(2)

    def fit(self, train, labels, query, targets, **overrides):
        options = dict(width=2, num_classes=2, device="cpu", epochs=4,
                       batch_size=8, lr=.2, weight_decay=0., seed=17)
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return probe.fit_linear_probe(train, labels, query, targets, **options)

    def test_separable_frozen_features_are_learned_without_consuming_global_rng(self):
        train, labels, query, targets = self.fixture()
        before_features = train["values"].copy()
        before_rng = torch.random.get_rng_state().clone()
        result = self.fit(train, labels, query, targets)
        self.assertEqual(result["top1"], 100.)
        self.assertEqual(result["correct"], 2)
        self.assertLess(result["loss"], math.log(2))
        np.testing.assert_array_equal(train["values"], before_features)
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))

    def test_compact_batch_expansion_preserves_coordinates_and_zero_values(self):
        values = np.array([[.8, .6], [0., 1.], [.4, .2]], dtype=np.float32)
        codes = {"values": values, "indices": np.array([[5, 1], [7, 0], [2, 6]], dtype=np.int32)}
        expanded = probe.code_batch(codes, np.array([1, 0]), 8, "cpu")
        expected = torch.zeros(2, 8)
        expected[0, 0] = 1.
        expected[1, 5], expected[1, 1] = .8, .6
        torch.testing.assert_close(expanded, expected)
        self.assertEqual(expanded.shape, (2, 8))
        self.assertEqual(codes["values"].shape, (3, 2))
        self.assertEqual(codes["indices"].shape, (3, 2))
        np.testing.assert_array_equal(codes["values"], values)
        with self.assertRaises(ValueError):
            probe.code_batch({"values": values[:1], "indices": np.array([[8, 0]])}, [0], 8, "cpu")
        with self.assertRaises(ValueError):
            probe.code_batch({"values": np.array([[np.nan, 1.]])}, [0], 2, "cpu")

    def test_query_labels_only_change_scoring_and_never_the_fitted_head(self):
        train, labels, query, targets = self.fixture()
        files = MemoryFiles()
        with mock.patch.object(probe, "Path", files.Path), \
                mock.patch.object(probe, "save_checkpoint", side_effect=files.save), \
                mock.patch.object(probe, "load_checkpoint", side_effect=files.load):
            left = self.fit(train, labels, query, targets, checkpoint_path="left.pt", identity={"run": 1})
            right = self.fit(train, labels, query, targets[::-1], checkpoint_path="right.pt", identity={"run": 2})
        self.assertEqual(left["loss"], right["loss"])
        self.assertEqual(left["top1"], 100.)
        self.assertEqual(right["top1"], 0.)
        for name, parameter in files.files["left.pt"]["head"].items():
            torch.testing.assert_close(parameter, files.files["right.pt"]["head"][name], rtol=0., atol=0.)

    def test_interrupted_partial_epoch_replays_exactly_from_last_epoch_checkpoint(self):
        train, labels, query, targets = self.fixture()
        files = MemoryFiles()
        original_batch = probe.code_batch
        calls = 0

        def interrupt_mid_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            # Three updates finish epoch 0; interrupt after one update of epoch 1.
            if calls == 5:
                raise RuntimeError("simulated interruption")
            return original_batch(*args, **kwargs)

        with mock.patch.object(probe, "Path", files.Path), \
                mock.patch.object(probe, "save_checkpoint", side_effect=files.save), \
                mock.patch.object(probe, "load_checkpoint", side_effect=files.load):
            uninterrupted = self.fit(train, labels, query, targets,
                checkpoint_path="continuous.pt", identity={"fixture": "fixed"})
            with mock.patch.object(probe, "code_batch", side_effect=interrupt_mid_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    self.fit(train, labels, query, targets,
                        checkpoint_path="resumed.pt", identity={"fixture": "fixed"})
            self.assertEqual(files.files["resumed.pt"]["completed_epochs"], 1)
            resumed = self.fit(train, labels, query, targets,
                checkpoint_path="resumed.pt", identity={"fixture": "fixed"})
            # A completed checkpoint scores without fitting again.
            with mock.patch.object(probe, "save_checkpoint", side_effect=AssertionError("unexpected training")):
                repeated = self.fit(train, labels, query, targets,
                    checkpoint_path="resumed.pt", identity={"fixture": "fixed"})
        self.assertEqual(uninterrupted["loss"], resumed["loss"])
        self.assertEqual(uninterrupted["top1"], resumed["top1"])
        self.assertEqual(resumed["loss"], repeated["loss"])
        self.assertEqual(resumed["train_seconds"], repeated["train_seconds"])
        continuous, restored = files.files["continuous.pt"], files.files["resumed.pt"]
        self.assertEqual(restored["completed_epochs"], 4)
        for name, parameter in continuous["head"].items():
            torch.testing.assert_close(parameter, restored["head"][name], rtol=0., atol=0.)
        for parameter, state in continuous["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, restored["optimizer"]["state"][parameter][name], rtol=0., atol=0.)

    def test_invalid_labels_are_rejected_without_silent_integer_coercion(self):
        train, labels, query, targets = self.fixture()
        for invalid in (labels[:-1], np.full_like(labels, -1), np.full_like(labels, 2),
                        labels.astype(float) + .2, np.full(len(labels), np.nan)):
            with self.subTest(labels=invalid[:2]):
                with self.assertRaises(ValueError):
                    self.fit(train, invalid, query, targets)
        with self.assertRaises(ValueError):
            self.fit(train, labels, query, np.array([0., .9]))


class ProbeAggregationTests(unittest.TestCase):
    def test_per_k_heads_parent_and_partial_resume_are_separate(self):
        args = runner.build_parser().parse_args(["--hidden-dim", "8", "--k", "2", "--eval-topk", "2,4"])
        bundle = SimpleNamespace(train=np.ones((8, 4), dtype=np.float32),
            train_labels=np.arange(8) % 2, query=None, query_labels=None,
            fit_ids=np.arange(6), gallery_ids=np.arange(6), query_ids=np.array([6, 7]),
            provenance={"identity": {"data": "fixed"}})
        model = SimpleNamespace(config=SimpleNamespace(input_dim=4, hidden_dim=8), sae=object(), training=True)
        model.parameters = lambda: iter([SimpleNamespace(device=torch.device("cuda:0"))])
        model.eval = lambda: setattr(model, "training", False)
        model.train = lambda mode: setattr(model, "training", mode)
        files = MemoryFiles()
        calls = []

        def encoded(features, rows, sae, repair, config, root, identity, role, device):
            width = 4 if sae is None else config["model"]["k"]
            values = np.ones((len(rows), width), dtype=np.float32)
            return {"values": values}, {"nonzero_sum": width * len(rows)}

        def fit(*args, **kwargs):
            calls.append((kwargs["identity"]["K"], str(kwargs["checkpoint_path"]), kwargs["width"]))
            if len(calls) == 2:
                raise RuntimeError("interrupted between K budgets")
            return {"top1": 75., "total": 2, "train_seconds": .1}

        with mock.patch.object(probe, "Path", files.Path), \
                mock.patch.object(probe, "atomic_json", side_effect=files.save), \
                mock.patch.object(probe, "sha256_file", return_value="fixed-source"), \
                mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.get_device_name", return_value="synthetic-gpu"), \
                mock.patch("repair_benchmark._encoded_cache", side_effect=encoded), \
                mock.patch.object(probe, "fit_linear_probe", side_effect=fit), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "interrupted between"):
                probe.evaluate_linear_probes(args, bundle, model, "run", "fixed-checkpoint")
            path = "run/classification/endpoint/result.json"
            partial = files.files[path]
            self.assertFalse(partial["complete"])
            self.assertEqual(set(partial["per_topk"]), {"2"})
            self.assertTrue(model.training)
            complete = probe.evaluate_linear_probes(args, bundle, model, "run", "fixed-checkpoint")
            self.assertTrue(complete["complete"])
            self.assertEqual(set(complete["per_topk"]), {"2", "4"})
            self.assertIsNotNone(complete["full_parent"])
            self.assertTrue(model.training)
            before = len(calls)
            repeated = probe.evaluate_linear_probes(args, bundle, model, "run", "fixed-checkpoint")
            self.assertEqual(before, len(calls))
            self.assertEqual(complete, repeated)
            with self.assertRaisesRegex(ValueError, "different experiment"):
                probe.evaluate_linear_probes(args, bundle, model, "run", "changed-checkpoint")
        self.assertEqual([k for k, _, _ in calls], [2, 4, 4, None])
        self.assertEqual([width for _, _, width in calls], [8, 8, 8, 4])
        self.assertTrue(calls[0][1].endswith("/2/last.pt"))
        self.assertTrue(calls[-1][1].endswith("/full_parent/last.pt"))
        self.assertEqual(complete["per_topk"]["2"]["actual_nnz_mean"], 2.)
        # Dense representation width is not a measured nonzero count.
        self.assertIsNone(complete["full_parent"]["actual_nnz_mean"])
        self.assertIsNone(complete["full_parent"]["query_actual_nnz_mean"])


if __name__ == "__main__":
    unittest.main()
