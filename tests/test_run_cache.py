"""IO/cache orchestration contracts with stand-in training and evaluation.

No neural numerics are mocked as evidence of model correctness. These tests
exercise real memmaps, cleanup, and E1 cache decisions without importing torch.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import mpsae
from mpsae.config import BACKBONES, Cell
from mpsae.io import append_jsonl, open_npz_memmap, read_jsonl, save_npz, truncate_jsonl


def test_interrupted_metric_tail_is_repaired_before_resume(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_bytes(b'{"step": 10}\n{"step": 20}\n{"step":')
    assert read_jsonl(path) == [{"step": 10}, {"step": 20}]
    assert truncate_jsonl(path, 2) == 2
    append_jsonl(path, {"step": 30})
    assert read_jsonl(path) == [{"step": 10}, {"step": 20}, {"step": 30}]


def test_valid_unterminated_metric_gets_separator_before_append(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"step": 10}', encoding="utf-8")
    assert truncate_jsonl(path, 10) == 1
    append_jsonl(path, {"step": 20})
    assert read_jsonl(path) == [{"step": 10}, {"step": 20}]


@pytest.mark.parametrize("payload", [b'{"step":}\n', b'{"step":}\n{"step": 20}'])
def test_corrupt_committed_metric_lines_are_not_silently_ignored(tmp_path, payload):
    import json

    path = tmp_path / "train.jsonl"
    path.write_bytes(payload)
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


@pytest.mark.parametrize("compressed,dtype,fortran,label_key", [
    (False, "float32", False, "labels"),
    (True, "float32", False, "label"),
    (False, "float64", True, "labels"),
    (True, "int32", False, "labels"),
])
def test_npz_cache_streams_data_without_materialising_archive_member(tmp_path, monkeypatch,
                                                                  compressed, dtype, fortran, label_key):
    values = np.arange(24, dtype=dtype).reshape(8, 3)
    if fortran:
        values = np.asfortranarray(values)
    archive = tmp_path / "features.npz"
    save = np.savez_compressed if compressed else np.savez
    save(archive, data=values, **{label_key: np.arange(8)})
    npz_type = type(np.load(archive))
    original_getitem = npz_type.__getitem__

    def prohibit_full_matrix(self, key):
        if key in ("data", "data.npy"):
            pytest.fail("the full embedding matrix must not be loaded into RAM")
        return original_getitem(self, key)

    monkeypatch.setattr(npz_type, "__getitem__", prohibit_full_matrix)
    data, labels, metadata = open_npz_memmap(archive, tmp_path / "cache")
    try:
        assert isinstance(data, np.memmap)
        assert data.dtype == np.float32 and data.flags.c_contiguous
        np.testing.assert_array_equal(data, values.astype(np.float32))
        np.testing.assert_array_equal(labels, np.arange(8))
        assert metadata["label_key"] == label_key
    finally:
        data._mmap.close()


def test_nonfinite_embeddings_do_not_publish_cache_receipt(tmp_path):
    archive = tmp_path / "features.npz"
    np.savez(archive, data=np.array([[np.nan]], dtype=np.float32), labels=np.array([0]))
    with pytest.raises(ValueError, match="nonfinite"):
        open_npz_memmap(archive, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("cache.json"))


def test_archive_metadata_cannot_replace_validated_embeddings(tmp_path):
    archive = tmp_path / "features.npz"
    with pytest.raises(ValueError, match="cannot replace"):
        save_npz(archive, data=np.ones((2, 3)), labels=np.array([0, 1]),
                 extra={"data": np.array([[np.nan]])})
    assert not archive.exists()


@pytest.fixture
def runs(monkeypatch):
    # The module imports train at its boundary. Supply only that heavy boundary
    # while loading the actual runs.py under an isolated module name.
    train = ModuleType("mpsae.train")
    train.FitSettings = lambda: SimpleNamespace(device="cpu")
    train.fit = lambda *args, **kwargs: {"signature": "fit", "train_seconds": 1.0, "gpu_hours": 1 / 3600}
    train.load_fitted = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "mpsae.train", train)
    monkeypatch.setattr(mpsae, "train", train, raising=False)
    spec = importlib.util.spec_from_file_location("mpsae._runs_contract_test",
                                                  Path(__file__).parents[1] / "mpsae" / "runs.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _bundle():
    return SimpleNamespace(
        identity={"train": {"sha256": "a" * 64}, "validation": {"sha256": "b" * 64}},
        train=SimpleNamespace(features=np.ones((3, 2), dtype=np.float32), labels=np.array([0, 0, 1])),
        validation=SimpleNamespace(features=np.ones((2, 2), dtype=np.float32), labels=np.array([0, 1])))


@pytest.mark.parametrize("backbone,recompute", [("ff2048", False), ("vit_b16", True)])
def test_dense_reference_reuses_only_compatible_zero_detection(
        runs, tmp_path, monkeypatch, backbone, recompute):
    from mpsae.io import atomic_json, canonical_hash

    bundle = _bundle()
    bundle.train.rows = 3
    path = tmp_path / backbone / "dense_reference.json"
    previous = {"identity": canonical_hash(bundle.identity), "top1": 65.0}
    atomic_json(path, previous)
    calls = []

    def score(*args, **kwargs):
        calls.append(True)
        return {"top1": 66.0}

    monkeypatch.setattr(runs, "_evaluate_module", lambda: SimpleNamespace(dense_top1=score))
    settings = runs.RunSettings(root=tmp_path)
    result = runs.dense_reference(BACKBONES[backbone], bundle, settings)
    assert bool(calls) is recompute
    assert result["top1"] == (66.0 if recompute else 65.0)
    runs.dense_reference(BACKBONES[backbone], bundle, settings)
    assert len(calls) == int(recompute)  # New policy is reusable on the next run.


def _fake_evaluator(fail_at=None):
    captured = {"directories": [], "maps": []}

    def encode(model, features, rows, k, *, cache_dir, identity, **kwargs):
        target = cache_dir / identity["split"]
        target.mkdir(parents=True, exist_ok=True)
        captured["directories"].append(cache_dir)
        indices = np.lib.format.open_memmap(target / "indices.npy", mode="w+", dtype=np.int32,
                                             shape=(len(features), k))
        values = np.lib.format.open_memmap(target / "values.npy", mode="w+", dtype=np.float32,
                                            shape=(len(features), k))
        indices[:] = np.arange(k)
        values[:] = 1
        captured["maps"].extend((indices, values))
        if fail_at == "encode" and identity["split"] == "validation":
            # Writers are not returned to evaluate_at: the traceback owns them.
            raise RuntimeError("encoding failed")
        return {"indices": indices, "values": values,
                "stats": {"nnz_mean": float(k), "nnz_min": k, "empty_codes": 0, "seconds": 0.01}}

    def score(gallery, glabels, queries, qlabels, **kwargs):
        assert gallery["values"][0, 0] == queries["values"][0, 0] == 1
        if fail_at == "score":
            raise RuntimeError("scoring failed")
        return {"top1": 75.0, "queries": len(qlabels), "empty_queries": 0, "seconds": 0.1}

    def probe(*args, epochs, **kwargs):
        if fail_at == "probe":
            raise RuntimeError("probing failed")
        return {"top1": 80.0, "top5": 100.0, "epochs": epochs, "settings": {"optimizer": "adam"}}

    def timing(*args, **kwargs):
        raise RuntimeError("CUDA is unavailable")

    evaluator = SimpleNamespace(encode_split=encode, top1_accuracy=score,
                                linear_probe=probe, time_sparse_matmul=timing)
    return evaluator, captured


@pytest.mark.parametrize("fail_at", [None, "encode", "score", "probe"])
def test_transient_cache_closes_maps_and_removes_only_owned_point(runs, tmp_path, monkeypatch, fail_at):
    evaluator, captured = _fake_evaluator(fail_at)
    monkeypatch.setattr(runs, "_evaluate_module", lambda: evaluator)
    shared = tmp_path / "cache" / "codes"
    shared.mkdir(parents=True)
    sentinel = shared / "keep.txt"
    sentinel.write_text("existing retained cache", encoding="utf-8")
    settings = runs.RunSettings(cache_dir=tmp_path / "cache", retain_codes=False,
                                 fit=SimpleNamespace(device="cpu"))
    kwargs = dict(k=2, signature="fit", want_probe=True, want_timing=True)
    arguments = (Cell("ff2048", "csr", 2, 1227), _bundle(), object(), BACKBONES["ff2048"], settings)
    if fail_at:
        with pytest.raises(RuntimeError, match="failed"):
            runs.evaluate_at(*arguments, **kwargs)
    else:
        result = runs.evaluate_at(*arguments, **kwargs)
        assert result["probe_epochs"] == result["probe_settings"]["epochs"] == 10
        assert result["timing_status"] == "skipped"
    assert captured["maps"] and all(array._mmap.closed for array in captured["maps"])
    assert all(not directory.exists() for directory in captured["directories"])
    assert list((tmp_path / "cache" / "transient-codes").iterdir()) == []
    assert sentinel.read_text(encoding="utf-8") == "existing retained cache"


def test_retention_default_keeps_legacy_cache_and_transient_calls_are_unique(runs, tmp_path, monkeypatch):
    evaluator, captured = _fake_evaluator()
    monkeypatch.setattr(runs, "_evaluate_module", lambda: evaluator)
    settings = runs.RunSettings(cache_dir=tmp_path / "cache", fit=SimpleNamespace(device="cpu"),
                                 linear_probe=False, measure_timing=False)
    arguments = (Cell("ff2048", "csr", 2, 1227), _bundle(), object(), BACKBONES["ff2048"])
    try:
        runs.evaluate_at(*arguments, settings, k=2, signature="fit", want_probe=False, want_timing=False)
        assert settings.retain_codes is True
        assert all(path == settings.cache_dir / "codes" for path in captured["directories"])
        assert (settings.cache_dir / "codes" / "train" / "values.npy").is_file()
    finally:
        for array in captured["maps"]:
            if not array._mmap.closed:
                array._mmap.close()
    transient = replace(settings, retain_codes=False)
    with runs._code_cache(transient, {"fit": "same", "k": 2}) as first:
        first_name = first.name
    with runs._code_cache(transient, {"fit": "same", "k": 2}) as second:
        assert second.name != first_name
        assert second.name[:25] == first_name[:25]


def test_cached_e1_refreshes_probe_epochs_and_attempts_timing_once(runs, tmp_path, monkeypatch):
    calls = []

    def evaluate(*args, **kwargs):
        settings = args[4]
        calls.append((settings.linear_probe, settings.probe_epochs, settings.measure_timing))
        point = {"top1": 75.0, "nnz_mean": 2.0,
                 "timing_status": "skipped" if settings.measure_timing else "not_requested"}
        if settings.measure_timing:
            point["timing_config"] = settings.timing_identity()
        if settings.linear_probe:
            point.update(probe_top1=80.0, probe_top5=100.0, probe_epochs=settings.probe_epochs,
                         probe_settings={"epochs": settings.probe_epochs})
        return point

    monkeypatch.setattr(runs, "evaluate_at", evaluate)
    cell, bundle = Cell("ff2048", "csr", 8, 1227), _bundle()
    settings = runs.RunSettings(root=tmp_path / "runs", fit=SimpleNamespace(device="cpu"),
                                 linear_probe=False, measure_timing=False, run_e2=False)
    runs.run_cell(cell, bundle, settings, code_version="2.0.0")
    runs.run_cell(cell, bundle, settings, code_version="2.0.0")
    assert len(calls) == 1
    settings = replace(settings, linear_probe=True)
    runs.run_cell(cell, bundle, settings, code_version="2.0.0")
    runs.run_cell(cell, bundle, settings, code_version="2.0.0")
    assert len(calls) == 2
    settings = replace(settings, probe_epochs=20)
    runs.run_cell(cell, bundle, settings, code_version="2.0.0")
    assert len(calls) == 3
    settings = replace(settings, measure_timing=True)
    runs.run_cell(cell, bundle, settings, code_version="2.0.0")
    runs.run_cell(cell, bundle, settings, code_version="2.0.0")
    assert len(calls) == 4  # CPU skip is recorded and is reusable.


def test_legacy_probe_and_timing_refresh_when_provenance_is_missing(runs):
    settings = runs.RunSettings(fit=SimpleNamespace(device="cpu"), measure_timing=False)
    assert not runs._cached_e1_complete({"top1": 75, "probe_top1": 80}, settings)
    point = {"top1": 75, "probe_top1": 80, "probe_settings": {"epochs": 10}}
    assert runs._cached_e1_complete(point, settings)
    settings = replace(settings, measure_timing=True)
    assert not runs._cached_e1_complete(point, settings)
    assert not runs._cached_e1_complete({**point, "timing_status": "skipped"}, settings)
    assert not runs._cached_e1_complete({**point, "timing": {"milliseconds_per_batch": 1}}, settings)
    point.update(timing_config=settings.timing_identity(), timing_status="skipped")
    assert runs._cached_e1_complete(point, settings)
    assert not runs._cached_e1_complete(point, replace(settings, timing_retry_skipped=True))
    point["timing_status"] = "success"
    assert runs._cached_e1_complete(point, settings)
    assert not runs._cached_e1_complete(point, replace(settings, timing_rounds=50))


def test_interrupted_timing_preserves_accuracy_and_probe_and_only_retries_timing(runs, tmp_path, monkeypatch):
    from mpsae.io import read_json

    evaluator, captured = _fake_evaluator()
    monkeypatch.setattr(runs, "_evaluate_module", lambda: evaluator)
    cell = Cell("ff2048", "csr", 8, 1227)
    settings = runs.RunSettings(root=tmp_path / "runs", cache_dir=tmp_path / "cache",
                               retain_codes=False, run_e2=False,
                               fit=SimpleNamespace(device="cpu"))
    artifact = runs.cell_dir(settings, cell) / "eval_e1.json"

    def interrupt(*args, **kwargs):
        saved = read_json(artifact)
        assert saved["top1"] == 75.0
        assert saved["probe_top1"] == 80.0
        assert saved["timing_status"] == "attempted"
        assert saved["timing_config"] == settings.timing_identity()
        raise KeyboardInterrupt("timing interrupted")

    evaluator.time_sparse_matmul = interrupt
    with pytest.raises(KeyboardInterrupt, match="timing interrupted"):
        runs.run_cell(cell, _bundle(), settings, code_version="2.0.0")
    assert all(array._mmap.closed for array in captured["maps"])

    def no_recompute(*args, **kwargs):
        pytest.fail("durable accuracy/probe must be reused")

    evaluator.top1_accuracy = evaluator.linear_probe = no_recompute
    evaluator.time_sparse_matmul = lambda *a, **kw: {"milliseconds_per_batch": 3.0}
    result = runs.run_cell(cell, _bundle(), settings, code_version="2.0.0")
    assert result["e1"]["top1"] == 75.0
    assert result["e1"]["timing_status"] == "success"
    assert result["e1"]["retrieval_ms"] == 3.0


def test_e2_selection_reuses_old_points_without_reporting_unselected_budgets(runs, tmp_path, monkeypatch):
    from mpsae.io import atomic_json, read_json

    cell = Cell("ff2048", "csr", 8, 1227)
    settings = runs.RunSettings(root=tmp_path / "runs", e2_k_values=(8, 64, 256, 1024),
                               linear_probe=False, measure_timing=False,
                               fit=SimpleNamespace(device="cpu"))
    path = runs.cell_dir(settings, cell) / "eval_e2.json"
    old = [{"k": k, "top1": 70.0 + k / 10000, "nnz_mean": float(k)} for k in (2, 8, 64)]
    atomic_json(path, {"signature": "fit", "points": old})
    calls = []

    def evaluate(*args, k, **kwargs):
        calls.append(k)
        return {"k": k, "top1": 75.0, "nnz_mean": float(k)}

    monkeypatch.setattr(runs, "evaluate_at", evaluate)
    result = runs.run_e2(cell, _bundle(), settings, signature="fit", e1=old[1])
    assert calls == [256, 1024]
    assert [point["k"] for point in result["points"]] == [8, 64, 256, 1024]
    assert result["points"][:2] == old[1:]
    assert read_json(path)["cached_points"] == old[:1]

    # A later selection can reuse archived points, without any recomputation.
    calls.clear()
    result = runs.run_e2(cell, _bundle(), replace(settings, e2_k_values=(2, 64)),
                         signature="fit", e1=old[1])
    assert calls == []
    assert [point["k"] for point in result["points"]] == [2, 64]
    assert {point["k"] for point in result["cached_points"]} == {8, 256, 1024}


def test_e2_without_override_uses_current_inference_grid(runs, tmp_path, monkeypatch):
    cell = Cell("rn18", "csr", 8, 1227)
    calls = []
    def evaluate(*args, k, **kwargs):
        calls.append(k)
        return {"k": k, "top1": 75.0, "nnz_mean": float(k)}
    monkeypatch.setattr(runs, "evaluate_at", evaluate)
    settings = runs.RunSettings(root=tmp_path / "runs", fit=SimpleNamespace(device="cpu"),
                               linear_probe=False, measure_timing=False)
    result = runs.run_e2(cell, _bundle(), settings, signature="fit", e1={"k": 8, "top1": 75.0})
    assert calls == [16, 32, 256]
    assert [point["k"] for point in result["points"]] == [16, 32, 256]


def test_e2_upgrades_legacy_accuracy_reuses_anchor_and_resumes_probe(runs, tmp_path, monkeypatch):
    from mpsae.io import atomic_json, read_json

    cell = Cell("ff2048", "csr", 8, 1227)
    settings = runs.RunSettings(root=tmp_path / "runs", cache_dir=tmp_path / "cache",
                               retain_codes=False, e2_k_values=(2, 8),
                               fit=SimpleNamespace(device="cpu"))
    path = runs.cell_dir(settings, cell) / "eval_e2.json"
    old = {"k": 2, "top1": 75.0, "nnz_mean": 2.0}
    atomic_json(path, {"signature": "fit", "points": [old]})
    anchor = {"k": 8, "top1": 75.0, "probe_top1": 80.0, "probe_epochs": 10,
              "nnz_mean": 7.0, "timing_config": settings.timing_identity(), "timing_status": "success"}
    evaluator, captured = _fake_evaluator()
    monkeypatch.setattr(runs, "_evaluate_module", lambda: evaluator)

    def no_accuracy(*args, **kwargs):
        pytest.fail("completed 1-NN must be reused")

    def interrupt_timing(*args, **kwargs):
        saved = read_json(path)["points"][0]
        assert saved["probe_top1"] == 80.0
        assert saved["signature"] == "fit"
        raise KeyboardInterrupt("interrupted after probe")

    evaluator.top1_accuracy = no_accuracy
    evaluator.time_sparse_matmul = interrupt_timing
    with pytest.raises(KeyboardInterrupt):
        runs.run_e2(cell, _bundle(), settings, signature="fit", e1=anchor)
    evaluator.linear_probe = no_accuracy
    evaluator.time_sparse_matmul = lambda *a, **kw: {"milliseconds_per_batch": 2.0}
    result = runs.run_e2(cell, _bundle(), settings, signature="fit", e1=anchor)
    assert result["points"][1] == anchor
    assert result["points"][0]["timing_status"] == "success"
    assert result["points"][0]["retrieval_ms"] == 2.0
    evaluator.encode_split = no_accuracy
    assert runs.run_e2(cell, _bundle(), settings, signature="fit", e1=anchor) == result
