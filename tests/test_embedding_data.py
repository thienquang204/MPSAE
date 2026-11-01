"""Prepared-embedding I/O contracts without network, raw images or CUDA work."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

import mmrsae_data as data
from repair_data import open_feature_cache, sha256_file


@unittest.skipIf(np is None, "NumPy is required for archive tests")
class FeatureMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="embedding_io_")
        self.root = Path(self.temporary.name)
        self.maps = []
        self.values = np.arange(60 * 7, dtype=np.float32).reshape(60, 7)
        self.labels = np.repeat(np.arange(3), 20)

    def tearDown(self):
        for mapped in self.maps:
            mapped._mmap.close()
        self.temporary.cleanup()

    def open(self, path, **options):
        mapped, labels, metadata = open_feature_cache(path, self.root / "cache", **options)
        self.maps.append(mapped)
        return mapped, labels, metadata

    def test_stored_zip64_archive_is_mapped_without_feature_copy(self):
        path = self.root / "stored.npz"
        np.savez(path, data=self.values, label=self.labels)
        mapped, labels, metadata = self.open(path, expected_sha256=sha256_file(path))
        self.assertIsInstance(mapped, np.memmap)
        self.assertEqual(Path(mapped.filename), path)
        self.assertFalse(mapped.flags.writeable)
        np.testing.assert_array_equal(mapped[[0, 17, 59]], self.values[[0, 17, 59]])
        np.testing.assert_array_equal(labels, self.labels)
        self.assertFalse((self.root / "cache" / metadata["sha256"] / "data.npy").exists())

    def test_stored_fortran_and_big_endian_layouts_are_preserved(self):
        for order in ("C", "F"):
            for dtype in ("<f4", ">f4"):
                with self.subTest(order=order, dtype=dtype):
                    path = self.root / f"{order}_{dtype[0] == '>'}.npz"
                    values = np.array(self.values, dtype=dtype, order=order)
                    np.savez(path, data=values, label=self.labels)
                    mapped, _, _ = self.open(path)
                    self.assertEqual(mapped.dtype, values.dtype)
                    np.testing.assert_array_equal(mapped, values)

    def test_compressed_archive_is_streamed_once_and_hash_is_reused(self):
        path = self.root / "compressed.npz"
        np.savez_compressed(path, data=self.values, label=self.labels)
        expected = sha256_file(path)
        first, _, metadata = self.open(path, expected_sha256=expected)
        self.assertNotEqual(Path(first.filename), path)
        with mock.patch("repair_data.sha256_file", side_effect=AssertionError("unchanged file was rehashed")):
            second, _, _ = self.open(path, expected_sha256=expected)
        np.testing.assert_array_equal(second, self.values)
        self.assertEqual(len(list((self.root / "cache" / metadata["sha256"]).glob("data.npy"))), 1)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.open(path, expected_sha256="0" * 64)

    def test_changed_archive_cannot_reuse_checksum_receipt(self):
        path = self.root / "source.npz"
        np.savez_compressed(path, data=self.values, label=self.labels)
        expected = sha256_file(path)
        mapped, _, _ = self.open(path, expected_sha256=expected)
        mapped._mmap.close()
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.open(path, expected_sha256=expected)

    def test_legacy_label_aliases_still_work(self):
        path = self.root / "labels.npz"
        np.savez(path, data=self.values, targets=self.labels[:, None])
        mapped, labels, _ = self.open(path, label_keys=("label", "labels", "target", "targets"))
        np.testing.assert_array_equal(mapped, self.values)
        np.testing.assert_array_equal(labels, self.labels)

    def test_truncated_stored_payload_is_rejected_before_mapping(self):
        encoded = io.BytesIO()
        np.save(encoded, self.values, allow_pickle=False)
        labels = io.BytesIO()
        np.save(labels, self.labels, allow_pickle=False)
        path = self.root / "bad.npz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data.npy", encoded.getvalue()[:-4])
            archive.writestr("label.npy", labels.getvalue())
        with self.assertRaisesRegex(ValueError, "payload length"):
            self.open(path)


@unittest.skipIf(np is None, "NumPy is required for embedding-source tests")
class PreparedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="embedding_source_")
        self.root = Path(self.temporary.name)
        self.maps = []
        self.args = types.SimpleNamespace(
            embeddings_dir=self.root / "local", cache_dir=self.root / "cache", download=True,
            embeddings_repo=data.EMBEDDINGS_REPO, embeddings_repo_type="model", embeddings_revision="main",
            embedding_source="precomputed", evaluation="dev", train_size=500, dev_size=200,
            anchor_count=100, data_seed=42, max_val=0)
        self.counts = mock.patch.dict(data.COUNTS, {"train": 1000, "validation": 1000})
        self.counts.start()

    def tearDown(self):
        for mapped in self.maps:
            mapped._mmap.close()
        self.counts.stop()
        self.temporary.cleanup()

    def archive(self, backbone="resnet18", split="train", folder=None):
        folder = folder or self.args.embeddings_dir
        folder.mkdir(parents=True, exist_ok=True)
        suffix = "rn18" if backbone == "resnet18" else "ff2048"
        name = "train" if split == "train" else "val"
        path = folder / f"1K_{name}_{suffix}.npz"
        width = 512 if backbone == "resnet18" else 2048
        labels = np.arange(1000, dtype=np.int64)
        values = np.ones((len(labels), width), dtype=np.float32)
        (np.savez if backbone == "resnet18" else np.savez_compressed)(path, data=values, label=labels)
        receipt = {"sha256": sha256_file(path), "split": split}
        if backbone == "resnet18":
            receipt.update(recipe="rn18-imagenet-export-v1", rows=len(labels), dimension=512,
                labels_sha256=data._array_hash(labels), transform="center_crop224of256_resize256;mean_std0.5;fp32",
                parent={"repository": "timm/resnet18.a1_in1k", "dimension": 512}, source={"split": split})
        else:
            receipt.update(repository=data.FF2048_REPO, revision="a" * 40, parent=data.FF2048_PARENT,
                           filename=f"FF2048_RN50_Embeds/{path.name}")
        path.with_suffix(".json").write_text(json.dumps(receipt), encoding="utf-8")
        return path, receipt

    def read(self, backbone="resnet18", split="train"):
        result = data._precomputed_split(split, self.args, backbone)
        self.maps.append(result[0])
        return result

    def test_both_local_backbones_work_offline_and_keep_original_parents(self):
        self.args.download = False
        for backbone in ("resnet18", "resnet50"):
            self.archive(backbone)
            with mock.patch.object(data, "_resolved_release", side_effect=AssertionError("network used")):
                features, labels, source = self.read(backbone)
            self.assertIsInstance(features, np.memmap)
            self.assertTrue(source["identity_verified"])
            self.assertEqual(source["receipt"]["sha256"], source["sha256"])
            self.assertEqual(len(labels), 1000)

    def test_hub_download_requests_only_one_pair_and_then_reuses_it_offline(self):
        remote = self.root / "remote"
        path, receipt = self.archive(folder=remote)
        release = {"repository": data.EMBEDDINGS_REPO, "repo_type": "model", "revision": "b" * 40,
                   "files": {p.name: {"size": p.stat().st_size,
                              "sha256": receipt["sha256"] if p.suffix == ".npz" else None}
                             for p in remote.iterdir()}}
        calls = []

        def download(repo, filename, **kwargs):
            calls.append((repo, filename, kwargs))
            target = Path(kwargs["local_dir"]) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(remote / filename, target)
            return str(target)

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = download
        with mock.patch.object(data, "_resolved_release", return_value=release), mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            _, _, online_source = self.read()
        self.assertEqual([call[1] for call in calls], ["1K_train_rn18.json", "1K_train_rn18.npz"])
        self.assertTrue(all(call[2]["repo_type"] == "model" and "cache_dir" not in call[2] for call in calls))
        self.args.download = False
        with mock.patch.object(data, "_resolved_release", side_effect=AssertionError("offline request")):
            _, _, offline_source = self.read()
        self.assertEqual(online_source, offline_source)
        self.assertFalse((self.args.embeddings_dir / "1K_val_rn18.npz").exists())

    def test_missing_offline_pair_fails_without_images_or_network(self):
        self.args.download = False
        with mock.patch.object(data, "_resolved_release", side_effect=AssertionError("network used")):
            with self.assertRaisesRegex(FileNotFoundError, "copy the NPZ and JSON together"):
                self.read()

    def test_bad_receipt_and_wrong_local_archive_are_rejected(self):
        path, receipt = self.archive()
        for field, value in (("split", "validation"), ("dimension", 2048), ("sha256", "0" * 64)):
            with self.subTest(field=field):
                bad = dict(receipt, **{field: value})
                path.with_suffix(".json").write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.read()

    def test_remote_archive_hash_must_match_its_json_before_large_download(self):
        path, receipt = self.archive(folder=self.root / "remote")
        release = {"files": {path.name: {"sha256": "0" * 64}}}
        with mock.patch.object(data, "_resolved_release", return_value=release), \
                mock.patch.object(data, "_hub_file", return_value=(path.with_suffix(".json"), None)) as fetch:
            with self.assertRaisesRegex(ValueError, "Hub archive SHA-256"):
                self.read()
        self.assertEqual(fetch.call_count, 1)

    def test_explicit_ff2048_path_without_json_keeps_legacy_authentication(self):
        path, _ = self.archive("resnet50")
        path.with_suffix(".json").unlink()
        self.args.ff2048_train = path
        sentinel = object()
        with mock.patch.object(data, "_ff2048_split", return_value=sentinel) as old_loader:
            self.assertIs(data._precomputed_split("train", self.args, "resnet50"), sentinel)
        old_loader.assert_called_once_with("train", self.args)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for runner routing")
    def test_dev_and_official_routing_preserve_pool_boundaries(self):
        import torch

        self.archive()
        self.archive(split="validation")
        self.args.download = False
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(data, "_resolved_release", side_effect=AssertionError("network used")):
            dev = data.prepare_imagenet_features(self.args, "resnet18", "cuda:0")
            self.maps.append(dev.train)
            self.assertIsNone(dev.query)
            self.assertFalse(dev.provenance["official_validation_opened"])
            np.testing.assert_array_equal(dev.fit_ids, dev.gallery_ids)
            self.assertEqual(np.intersect1d(dev.query_ids, dev.fit_ids).size, 0)
            self.assertEqual(np.intersect1d(dev.query_ids, dev.anchor_ids).size, 0)
            self.args.evaluation = "official"
            self.args.max_val = 100
            official = data.prepare_imagenet_features(self.args, "resnet18", "cuda:0")
            self.maps.extend([official.train, official.query])
            self.assertEqual(len(official.query_ids), 100)
            self.assertTrue(official.provenance["official_validation_opened"])

    def test_plan_never_prepares_data(self):
        import mmrsae_imagenet as runner

        output = io.StringIO()
        with mock.patch.object(data, "prepare_imagenet_features", side_effect=AssertionError("data opened")), contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(["--plan"]), 0)
        settings = json.loads(output.getvalue())["settings"]
        self.assertEqual(settings["embedding_source"], "precomputed")
        self.assertEqual(settings["embeddings_repo"], data.EMBEDDINGS_REPO)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for GPU-preflight routing")
    def test_preflight_only_never_exports_or_opens_data(self):
        import mmrsae_imagenet as runner

        with mock.patch("mmrsae_gpu.preflight", return_value={"gpu_search_probe": "passed"}), \
                mock.patch.object(runner, "method_config"), \
                mock.patch.object(data, "prepare_imagenet_features", side_effect=AssertionError("data opened")), \
                mock.patch("export_rn18_embeddings.export_split", side_effect=AssertionError("images opened")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--preflight-only", "--backbones", "resnet18"]), 0)


if __name__ == "__main__":
    unittest.main()
