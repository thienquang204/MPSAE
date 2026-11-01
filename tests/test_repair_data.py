"""Small synthetic checks for split integrity, cache memory use, and paired reporting."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

from compare_predictions import compare_predictions
from make_plan import build_plan, load_config, matrix_rows, validate_config
from prepare_splits import build_indices, prepare_splits
from repair_data import _read_npy_header, exclusive_lock, inspect_npz, load_splits, open_feature_cache, sha256_file


class PlannerTests(unittest.TestCase):
    def test_default_budget_and_deduplicated_confirmation(self):
        config = load_config()
        plan = build_plan(config)
        self.assertEqual([s["cumulative_updates"] for s in plan["stages"]], [1000, 1750, 2250])
        self.assertEqual(plan["confirmation"]["updates_per_fit"], 3130)
        self.assertEqual(plan["confirmation"]["final_batch_rows"], 3215)
        self.assertEqual(build_plan(config, "S")["confirmation"]["fits"], 4)
        rows = {row["arm"]: row for row in matrix_rows(config)}
        self.assertEqual((rows["A"]["auxiliary_start"], rows["A"]["auxiliary_end"], rows["A"]["anneal_updates"]), (64, 32, 175))
        self.assertEqual(rows["F"]["auxiliary_end"], 64)

    def test_invalid_protocol_changes_rejected(self):
        for field, key, value in [("optimizer", "name", "AdamW"), ("transport", "mass_normalization", True), ("bank", "confirmation_exclusion", "per_row")]:
            config = copy.deepcopy(load_config())
            config[field][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_invalid_numeric_settings_and_pins_fail_before_planning(self):
        for section, key, value in [("bank", "student_temperature", float("nan")),
                                    ("transport", "group_size", 1),
                                    ("retrieval", "query_batch_size", 32.5),
                                    ("gates", "bootstrap_replicates", 0),
                                    ("confirmation", "seeds", [1227, 1227]),
                                    ("source", "train_sha256", "not-a-release-digest")]:
            config = copy.deepcopy(load_config())
            config[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_arm_names_keep_their_declared_scientific_meaning(self):
        for arm, key, value in [("C", "repair", True), ("S", "ot_cost", "old"),
                                ("N", "mask", False), ("F", "auxiliary", "anneal"),
                                ("O", "rel_weight", .5)]:
            config = copy.deepcopy(load_config())
            config["arms"][arm][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)


@unittest.skipIf(np is None, "NumPy is needed for synthetic data tests")
class SplitAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="repair_data_")
        self.root = Path(self.temporary.name)
        self.path = self.root / "source.npz"
        self.features = np.arange(60 * 7, dtype=np.float32).reshape(60, 7)
        self.labels = np.repeat(np.arange(3), 20)
        np.savez_compressed(self.path, data=self.features, label=self.labels)
        self.parameters = dict(fit_per_class=5, cheap_per_class=2, holdout_per_class=3, anchor_count=5)

    def tearDown(self):
        self.temporary.cleanup()

    def test_inspection_never_loads_feature_array(self):
        original = np.load
        names = []
        def checked_load(stream, *args, **kwargs):
            names.append(stream.name)
            self.assertNotEqual(stream.name, "data.npy")
            return original(stream, *args, **kwargs)
        with mock.patch("numpy.load", side_effect=checked_load):
            metadata, labels = inspect_npz(self.path)
        self.assertEqual(names, ["label.npy"])
        self.assertEqual(metadata["feature_shape"], [60, 7])
        np.testing.assert_array_equal(labels, self.labels)

    def test_public_header_readers_preserve_versions_layout_and_byte_order(self):
        class HeaderOnlyStream(io.BytesIO):
            def __init__(self, contents, header_end):
                super().__init__(contents)
                self.header_end = header_end

            def read(self, size=-1):
                if size < 0 or self.tell() + size > self.header_end:
                    raise AssertionError("Header inspection tried to read feature data")
                return super().read(size)

        for version in ((1, 0), (2, 0), (3, 0)):
            for order in ("C", "F"):
                for dtype in ("<f4", ">f4"):
                    with self.subTest(version=version, order=order, dtype=dtype):
                        array = np.array(self.features, dtype=dtype, order=order)
                        encoded = io.BytesIO()
                        np.lib.format.write_array(encoded, array, version=version, allow_pickle=False)
                        contents = encoded.getvalue()
                        header_end = len(contents) - array.nbytes
                        stream = HeaderOnlyStream(contents, header_end)
                        shape, fortran_order, actual_dtype = _read_npy_header(stream)
                        self.assertEqual(shape, array.shape)
                        self.assertEqual(fortran_order, order == "F")
                        self.assertEqual(actual_dtype, array.dtype)
                        self.assertEqual(stream.tell(), header_end)

    def test_inspection_accepts_all_npy_versions_and_rejects_objects(self):
        labels = io.BytesIO()
        np.save(labels, self.labels, allow_pickle=False)
        for version in ((1, 0), (2, 0), (3, 0)):
            with self.subTest(version=version):
                features = io.BytesIO()
                np.lib.format.write_array(features, self.features, version=version, allow_pickle=False)
                with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("data.npy", features.getvalue())
                    archive.writestr("label.npy", labels.getvalue())
                metadata, actual_labels = inspect_npz(self.path, hash_archive=False)
                self.assertEqual(metadata["feature_shape"], list(self.features.shape))
                np.testing.assert_array_equal(actual_labels, self.labels)

        # A header alone is enough to reject object data, without unpickling.
        objects = io.BytesIO()
        np.lib.format.write_array_header_2_0(objects, {
            "shape": self.features.shape, "fortran_order": False, "descr": "|O",
        })
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("data.npy", objects.getvalue())
            archive.writestr("label.npy", labels.getvalue())
        with self.assertRaisesRegex(ValueError, "numeric, two-dimensional"):
            inspect_npz(self.path, hash_archive=False)

    def test_v3_header_decodes_unicode_and_rejects_invalid_headers(self):
        array = np.zeros(2, dtype=[("\u03bb", "<f4")])
        encoded = io.BytesIO()
        np.lib.format.write_array(encoded, array, version=(3, 0), allow_pickle=False)
        encoded.seek(0)
        self.assertEqual(_read_npy_header(encoded)[2], array.dtype)

        magic = np.lib.format.magic(3, 0)
        malformed = (
            magic + b"\x01\x00",  # incomplete length
            magic + (10_001).to_bytes(4, "little"),  # reject before reading body
            magic + (100).to_bytes(4, "little") + b"{}",  # truncated body
        )
        for contents in malformed:
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    _read_npy_header(io.BytesIO(contents))
        for body in (b"\xff", b"{}", b"{'shape': (-1, 2), 'fortran_order': False, 'descr': '<f4'}",
                     b"{'shape': (1, 2), 'fortran_order': 0, 'descr': '<f4'}"):
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    _read_npy_header(io.BytesIO(magic + len(body).to_bytes(4, "little") + body))

    def test_disjoint_balanced_deterministic_pools(self):
        fingerprint = sha256_file(self.path)
        first = build_indices(self.labels, fingerprint, **self.parameters)
        second = build_indices(self.labels, fingerprint, **self.parameters)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        for key, count in [("fit_indices", 5), ("dev_cheap_indices", 2), ("dev_holdout_indices", 3)]:
            np.testing.assert_array_equal(np.bincount(self.labels[first[key]]), [count] * 3)
        self.assertEqual(len(first["anchor_indices"]), 5)
        self.assertEqual(len(np.unique(self.labels[first["anchor_indices"]])), 3)
        combined = np.concatenate([first[key] for key in ("fit_indices", "dev_full_indices", "anchor_indices")])
        self.assertEqual(len(np.unique(combined)), 35)
        np.testing.assert_array_equal(first["fit_indices"], first["gallery_indices"])
        changed = build_indices(self.labels, "different-source", **self.parameters)
        self.assertFalse(np.array_equal(first["fit_indices"], changed["fit_indices"]))

    def test_immutable_manifest_and_tamper_detection(self):
        output = self.root / "splits"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            first = prepare_splits(self.path, output, **self.parameters)
            before = (output / "split_indices.npz").read_bytes()
            second = prepare_splits(self.path, output, **self.parameters)
            self.assertEqual(first, second)
            self.assertEqual(before, (output / "split_indices.npz").read_bytes())
            different = dict(self.parameters, fit_per_class=4)
            with self.assertRaisesRegex(ValueError, "different recipe"):
                prepare_splits(self.path, output, **different)
        arrays, manifest = load_splits(output, expected_source_sha256=sha256_file(self.path))
        self.assertEqual(manifest["distinct_source_rows"], 35)
        self.assertFalse(manifest["official_validation_opened"])
        self.assertEqual(len(arrays["dev_full_indices"]), 15)
        with (output / "split_indices.npz").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_splits(output)

    def test_streamed_read_only_memmap_and_source_check(self):
        cache = self.root / "cache"
        mapped, labels, source = open_feature_cache(self.path, cache, expected_sha256=sha256_file(self.path))
        self.assertIsInstance(mapped, np.memmap)
        self.assertFalse(mapped.flags.writeable)
        np.testing.assert_array_equal(mapped[[0, 17, 59]], self.features[[0, 17, 59]])
        np.testing.assert_array_equal(labels, self.labels)
        mapped._mmap.close()
        mapped2, _, _ = open_feature_cache(self.path, cache, expected_sha256=source["sha256"])
        mapped2._mmap.close()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            open_feature_cache(self.path, cache, expected_sha256="wrong")

    def test_too_few_class_rows_fails(self):
        with self.assertRaisesRegex(ValueError, "needs at least"):
            build_indices(np.array([0, 0, 1, 1]), "source", **self.parameters)

    def test_os_lock_is_reusable_without_deleting_lock_file(self):
        path = self.root / "writer.lock"
        with exclusive_lock(path):
            self.assertTrue(path.exists())
            with self.assertRaisesRegex(RuntimeError, "writer lock"):
                with exclusive_lock(path):
                    self.fail("A second independent handle acquired an overlapping lock")
        self.assertTrue(path.exists())
        with exclusive_lock(path):
            self.assertGreaterEqual(path.stat().st_size, 1)


@unittest.skipIf(np is None, "NumPy is needed for bootstrap tests")
class PredictionTests(unittest.TestCase):
    def test_all_seeds_share_each_query_draw(self):
        # The per-query mean difference is zero: independent seed resampling
        # would incorrectly produce a nonzero-width interval here.
        baseline, candidate = {}, {}
        for seed in (10, 20):
            for query in range(8):
                label = query // 4
                baseline[(seed, str(query))] = (label, label if seed == 10 else 9)
                candidate[(seed, str(query))] = (label, label if seed == 20 else 9)
        report = compare_predictions(baseline, candidate, replicates=100, seed=13)
        self.assertEqual(report["delta_percentage_points"], 0)
        self.assertEqual(report["ci95_percentage_points"], [0, 0])
        self.assertEqual(report["seeds"], [10, 20])

    def test_label_or_identity_mismatch_rejected(self):
        baseline = {(0, "a"): (1, 1), (0, "b"): (2, 0)}
        candidate = {(0, "a"): (1, 0), (0, "b"): (3, 3)}
        with self.assertRaisesRegex(ValueError, "Labels disagree"):
            compare_predictions(baseline, candidate)
        with self.assertRaisesRegex(ValueError, "matching seed/query"):
            compare_predictions(baseline, {(0, "a"): (1, 1)})

    def test_stratification_preserves_class_counts(self):
        baseline, candidate = {}, {}
        for query, label in enumerate([0, 0, 1, 1, 1, 1]):
            baseline[(1, str(query))] = (label, 9 if label == 0 else label)
            candidate[(1, str(query))] = (label, label if label == 0 else 9)
        report = compare_predictions(baseline, candidate, replicates=100)
        self.assertAlmostEqual(report["delta_percentage_points"], -100 / 3)
        self.assertAlmostEqual(report["ci95_percentage_points"][0], -100 / 3)
        self.assertAlmostEqual(report["ci95_percentage_points"][1], -100 / 3)


if __name__ == "__main__":
    unittest.main()
