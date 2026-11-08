"""Disk lifecycle failures and resume must never destroy the only usable data."""
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from mpsae.benchmark.config import Campaign
from mpsae.benchmark import storage
from mpsae.benchmark.run import campaign_paths
from mpsae.embed.archive import FeatureSink, finalize_archive
from mpsae.embed.direct import release_sink
from mpsae.io import atomic_json, canonical_hash, open_npz_memmap, read_json, save_npz
from mpsae.remote import RemoteSession


def config(**execution):
    return Campaign.from_mapping({"backbones": ["rn18", "ff2048"],
        "execution": {"storage_mode": "direct_v1", "cleanup_raw": True, **execution}})


def test_direct_namespace_does_not_reuse_historical_fits(tmp_path):
    assert campaign_paths(config(), tmp_path)["workspace"] == tmp_path / "direct-v1"
    assert campaign_paths(Campaign(), tmp_path)["workspace"] == tmp_path
    assert "storage_mode" not in Campaign().resolved()["execution"]


def test_stored_zip64_npz_maps_original_file_without_duplicate(tmp_path):
    source = tmp_path / "features.npz"
    values = np.arange(35, dtype=np.float32).reshape(7, 5)
    digest = save_npz(source, data=values, labels=np.arange(7))
    mapped, labels, info = open_npz_memmap(source, tmp_path / "cache", expected_sha256=digest)
    try:
        assert isinstance(mapped, np.memmap)
        assert Path(mapped.filename) == source
        np.testing.assert_array_equal(mapped, values)
        np.testing.assert_array_equal(labels, np.arange(7))
        assert info["storage"] == "zip_mmap"
        assert not (tmp_path / "cache").exists()
    finally:
        mapped._mmap.close()


def test_zip_mmap_rejects_nonfinite_payload(tmp_path):
    source = tmp_path / "bad.npz"
    np.savez(source, data=np.array([[np.nan]], dtype=np.float32), labels=np.array([0]))
    with pytest.raises(ValueError, match="nonfinite"):
        open_npz_memmap(source, tmp_path / "cache")


def test_published_sink_reclaimed_without_losing_archive(tmp_path):
    identity = {"fixture": "direct"}
    with FeatureSink(tmp_path / "cache", rows=2, dim=3, identity=identity) as sink:
        sink.write(np.ones((2, 3), dtype=np.float32), np.array([0, 1]))
        features, labels = sink.publish()
    archive = tmp_path / "out.npz"
    finalize_archive(archive, features, labels, manifest={"identity": canonical_hash(identity)})
    release_sink(sink, features, labels)
    assert not (sink.directory / "features.npy").exists()
    assert not (sink.directory / "sink.json").exists()
    mapped, _, _ = open_npz_memmap(archive, tmp_path / "reader")
    np.testing.assert_array_equal(mapped, np.ones((2, 3)))
    mapped._mmap.close()


def test_cleanup_rejects_owner_and_external_paths(tmp_path):
    owner = tmp_path / "work"
    owner.mkdir()
    outside = tmp_path / "keep.txt"
    outside.write_text("keep")
    for target in (owner, outside, owner / ".." / "keep.txt"):
        with pytest.raises(ValueError):
            storage.remove_owned(target, owner)
    assert outside.read_text() == "keep"


def test_cleanup_handles_hf_internal_symlinks_without_following_external_ones(tmp_path):
    owner = tmp_path / "owned"
    cache = owner / "hub"
    cache.mkdir(parents=True)
    blob = cache / "blob"
    blob.write_bytes(b"cache")
    link = cache / "snapshot-link"
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("symlink creation privilege unavailable")
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    external = cache / "external-link"
    external.symlink_to(outside)
    with pytest.raises(ValueError, match="linked cleanup"):
        storage.remove_owned(cache, owner)
    assert outside.read_bytes() == b"keep"
    external.unlink()
    storage.remove_owned(cache, owner)
    assert not cache.exists()
    assert outside.read_bytes() == b"keep"


def test_provenance_does_not_import_optional_ffcv(monkeypatch):
    import builtins
    from mpsae.embed.archive import library_versions
    original = builtins.__import__
    def imports(name, *args, **kwargs):
        if name.split(".")[0] in {"ffcv", "numba", "cupy"}:
            pytest.fail("direct provenance must not import the legacy runtime")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", imports)
    assert "numpy" in library_versions()


@pytest.mark.parametrize("complete", [False, True])
def test_migration_reclaims_only_known_ffcv_with_complete_originals(tmp_path, monkeypatch, complete):
    from mpsae.data import imagenet
    folder = tmp_path / "ffcv"
    folder.mkdir()
    redundant = folder / "train_500_0.50_90.ffcv"
    redundant.write_bytes(b"regenerable")
    unrelated = folder / "another-experiment.ffcv"
    unrelated.write_bytes(b"keep")
    monkeypatch.setattr(imagenet, "imagenet_folder_stats", lambda *a:
        {"images": imagenet.EXPECTED_ROWS["train"] if complete else 10, "classes": 1000})
    if complete:
        result = storage.reclaim_legacy_ffcv(tmp_path)
        assert result["reclaimed_bytes"] == len(b"regenerable")
        assert not redundant.exists()
    else:
        with pytest.raises(RuntimeError, match="incomplete"):
            storage.reclaim_legacy_ffcv(tmp_path)
        assert redundant.exists()
    assert unrelated.read_bytes() == b"keep"


@pytest.mark.parametrize("failure", [False, True])
def test_source_cleanup_requires_all_backbones_and_successful_backup(tmp_path, monkeypatch, failure):
    root = tmp_path / "direct-v1"
    root.mkdir()
    raw = tmp_path / "imagenet"
    for split in ("train", "val"):
        (raw / split).mkdir(parents=True)
        (raw / split / "original.jpg").write_bytes(b"original")
    checked = []
    monkeypatch.setattr(storage, "validate_source_accuracy", lambda *a: {"top1": 1.0})
    monkeypatch.setattr(storage, "verified_source", lambda r, name:
        (checked.append(name), [root / f"{name}.npz"])[1])
    def backup(files):
        assert set(p.stem for p in files) == {"rn18", "ff2048"}
        if not (raw / "direct-retired.json").exists():
            assert (raw / "train" / "original.jpg").exists()
        if failure:
            raise RuntimeError("backup unavailable")
        return {"revision": "verified"}
    observer = SimpleNamespace(verify_backup=backup)
    kwargs = dict(device="cuda:0", workers=0, batch_size=2, observer=observer,
                  task=lambda name, op: op())
    if failure:
        with pytest.raises(RuntimeError, match="backup unavailable"):
            storage.prepare_sources(config(), tmp_path, root, **kwargs)
        assert (raw / "train" / "original.jpg").exists()
        assert not (raw / "direct-retired.json").exists()
    else:
        storage.prepare_sources(config(), tmp_path, root, **kwargs)
        assert not (raw / "train").exists()
        assert read_json(raw / "direct-retired.json")["state"] == "raw_deleted"
        # Rerun never downloads retired raw images; verified features suffice.
        storage.prepare_sources(config(), tmp_path, root, **kwargs)


def test_missing_features_after_retirement_do_not_trigger_redownload(tmp_path, monkeypatch):
    root = tmp_path / "direct-v1"
    atomic_json(tmp_path / "imagenet" / "direct-retired.json", {"state": "raw_deleted"})
    monkeypatch.setattr(storage, "verified_source", lambda *a: None)
    with pytest.raises(RuntimeError, match="Restore verified embeddings"):
        storage.prepare_sources(config(), tmp_path, root, device="cuda:0", workers=0,
                                batch_size=2, observer=SimpleNamespace(verify_backup=lambda x: {}),
                                task=lambda name, op: op())


def test_local_cleanup_fails_before_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "verified_source", lambda *a: pytest.fail("must stop before scanning/download"))
    with pytest.raises(RuntimeError, match="cleanup_raw requires verified HF backup"):
        storage.prepare_sources(config(), tmp_path, tmp_path / "direct-v1", device="cuda:0",
                                workers=0, batch_size=2, observer=None, task=lambda name, op: op())


def test_dense_gate_failure_retains_raw_even_when_archives_are_complete(tmp_path, monkeypatch):
    root = tmp_path / "direct-v1"
    raw = tmp_path / "imagenet" / "train"
    raw.mkdir(parents=True)
    (raw / "original.jpg").write_bytes(b"image")
    monkeypatch.setattr(storage, "verified_source", lambda r, n: [root / f"{n}.npz"])
    def failed_gate(*args):
        raise RuntimeError("dense gate failed")
    monkeypatch.setattr(storage, "validate_source_accuracy", failed_gate)
    observer = SimpleNamespace(verify_backup=lambda files: pytest.fail("must not backup/cleanup"))
    with pytest.raises(RuntimeError, match="dense gate failed"):
        storage.prepare_sources(config(), tmp_path, root, device="cuda:0", workers=0,
                                batch_size=2, observer=observer, task=lambda name, op: op())
    assert (raw / "original.jpg").exists()


@pytest.mark.parametrize("fault", [None, "missing", "hash", "upload"])
def test_remote_backup_verifies_hashes_and_namespace(tmp_path, monkeypatch, fault):
    session = RemoteSession(config(), tmp_path)
    session.repo = "fixture/MPSAE"
    path = session.root / "embeddings" / "rn18" / "train.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture archive")
    remote_name = "direct-v1/embeddings/rn18/train.npz"
    uploads = []
    def upload(**kwargs):
        if fault == "upload":
            raise OSError("failed")
        uploads.append(kwargs)
    def entries(repo, names, **kwargs):
        assert names == [remote_name] and kwargs["revision"] == "pinned-commit"
        if fault == "missing":
            return []
        return [SimpleNamespace(path=remote_name, size=path.stat().st_size,
            lfs={"sha256": "wrong" if fault == "hash" else hashlib.sha256(path.read_bytes()).hexdigest()})]
    session.api = SimpleNamespace(create_repo=lambda **kw: None,
        repo_info=lambda **kw: SimpleNamespace(private=True, sha="pinned-commit"),
        upload_folder=upload, get_paths_info=entries)
    if fault:
        with pytest.raises((RuntimeError, OSError)):
            session.verify_backup([path])
    else:
        receipt = session.verify_backup([path])
        assert receipt["revision"] == "pinned-commit"
        assert uploads[0]["path_in_repo"] == "direct-v1/embeddings/rn18"


def test_feature_receipt_resumes_without_raw_dataset(tmp_path):
    from mpsae.benchmark.extract import FeatureSplit, FeatureBundle
    path = tmp_path / "benchmarks" / "test.npz"
    ids = ["one", "two"]
    manifest = finalize_archive(path, np.ones((2, 3), dtype=np.float32), np.array([0, 1]),
                                manifest={"ids_sha256": canonical_hash(ids)})
    split = FeatureSplit("test", None, None, ids, manifest, path)
    bundle = FeatureBundle("sop", "fixture", "rn18", split, split, True, {}, {})
    receipt = tmp_path / "benchmarks" / "receipt.json"
    storage.save_bundle(bundle, receipt, tmp_path, {"revision": "fixed"})
    restored = storage.load_bundle_receipt(receipt, tmp_path, {"revision": "fixed"})
    assert restored.gallery is restored.query
    assert restored.gallery.ids == ids
    assert restored.gallery.rows == 2
    restored.gallery.features._mmap.close()
    assert storage.load_bundle_receipt(receipt, tmp_path, {"revision": "changed"}) is None
