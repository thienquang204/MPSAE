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


def test_legacy_probe_without_epochs_refreshes_but_legacy_timing_is_accepted(runs):
    settings = runs.RunSettings(fit=SimpleNamespace(device="cpu"), measure_timing=False)
    assert not runs._cached_e1_complete({"top1": 75, "probe_top1": 80}, settings)
    point = {"top1": 75, "probe_top1": 80, "probe_settings": {"epochs": 10}}
    assert runs._cached_e1_complete(point, settings)
    settings = replace(settings, measure_timing=True)
    assert not runs._cached_e1_complete(point, settings)
    assert runs._cached_e1_complete({**point, "timing_status": "skipped"}, settings)
    assert runs._cached_e1_complete({**point, "timing": {"milliseconds_per_batch": 1}}, settings)
