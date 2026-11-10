"""Execute the real campaign orchestration with explicit heavy-boundary doubles.

These are integration contracts for source reuse, frozen target evaluation,
provenance, disk result caching and final status. Stand-in encoders/rankers are
not evidence about neural numerics; those have separate torch-dependent tests.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import mpsae
from mpsae import __version__
from mpsae.benchmark import run
from mpsae.benchmark.config import Campaign
from mpsae.benchmark.data import DATASETS
from mpsae.benchmark.extract import FeatureBundle, FeatureSplit
from mpsae.config import BACKBONES
from mpsae.io import atomic_json, canonical_hash, read_json


def _weights(revision="a"):
    return {"repository": "timm/resnet18.a1_in1k", "revision": revision * 40,
            "file": "model.safetensors", "sha256": "b" * 64}


def _source_identity(cell, bundle, preset, *, total_steps, per_epoch, code_version):
    # A deliberately small orchestration fixture identity, not a pretend SAE
    # recipe. Both the source artifacts and the requested import use this seam.
    return {"format": "runner-source-fixture-v1", "code_version": code_version,
            "backbone": cell.backbone, "method": cell.method,
            "target_k": cell.k, "seed": cell.seed, "data": bundle.identity,
            "schedule": {"total_steps": total_steps, "batches_per_epoch": per_epoch}}


def _campaign(*, methods=("csr", "mpsae"), datasets=("eurosat", "flowers102", "sop")):
    return Campaign.from_mapping({
        "name": "runner_contract", "backbones": ["rn18"], "methods": list(methods),
        "seeds": [1227], "imagenet": {"enabled": False},
        "downstream": {"datasets": list(datasets), "train_k": [8], "eval_k": [2, 8]},
        "execution": {"train_missing": False, "continue_on_error": True},
    })


def _write_source(root, cell, *, complete=True, weights=None):
    directory = root / "runs" / cell.slug
    data = {"backbone": cell.backbone, "dim": BACKBONES[cell.backbone].dim,
            "train": {"sha256": "c" * 64, "rows": 10, "recipe": "source-fixture"},
            "validation": {"sha256": "d" * 64, "rows": 4, "recipe": "source-fixture"}}
    identity = _source_identity(cell, SimpleNamespace(identity=data), BACKBONES[cell.backbone].preset,
                                total_steps=10, per_epoch=1, code_version=__version__)
    signature = canonical_hash(identity)
    atomic_json(directory / "config.json", {
        "identity": identity, "signature": signature, "code_version": __version__,
        "backbone_weights": weights or _weights(),
        "backbone_preprocessing": {"input_size": 224, "normalize": [[0.5] * 3, [0.5] * 3]},
    })
    atomic_json(directory / "summary.json", {
        "identity": identity, "signature": signature, "cell": cell.__dict__,
        "complete": complete, "steps": 10,
    })
    (directory / "checkpoint.pt").write_bytes(f"source-only:{cell.slug}".encode())
    return directory


def _feature_bundle(spec, name, weights):
    metadata = {"weights": deepcopy(weights), "preprocessing": {
        "mean": [0.5] * 3, "std": [0.5] * 3, "center_crop": 224}}
    gallery = FeatureSplit("test" if name == "sop" else "train",
                           np.ones((4, spec.dim), dtype=np.float32),
                           np.array([0, 0, 1, 1], dtype=np.int64),
                           [f"{name}:gallery:{index}" for index in range(4)], {})
    query = gallery if name == "sop" else FeatureSplit(
        "test", np.ones((2, spec.dim), dtype=np.float32), np.array([0, 1], dtype=np.int64),
        [f"{name}:query:{index}" for index in range(2)], {})
    identity = {"dataset": name, "source_revision": "f" * 40, "backbone": spec.name,
                "weights": deepcopy(weights), "gallery_ids": gallery.ids, "query_ids": query.ids}
    return FeatureBundle(name, DATASETS[name].protocol, spec.name, gallery, query, name == "sop",
                         identity, {"extraction": metadata, "scientific": True})


@pytest.fixture
def heavy_boundaries(monkeypatch):
    """Patch runtime dependencies while keeping runner/cache/provenance real."""
    from mpsae import features, pipeline
    from mpsae.benchmark import data, extract, report
    from mpsae.evaluate import ranking
    from mpsae.report import figures
    import mpsae.evaluate as evaluation_package

    captured = SimpleNamespace(models=[], source_loads=[], fit_calls=[], dataset_loads=[],
                               feature_calls=[], encode_calls=[], rank_calls=[], report_calls=[])

    class FrozenModel:
        def __init__(self, cell):
            self.cell = cell
            self.training = True
            self.requires_gradient = True

        def eval(self):
            self.training = False
            return self

        def requires_grad_(self, enabled):
            self.requires_gradient = enabled
            return self

    def forbidden_fit(*args, **kwargs):
        captured.fit_calls.append((args, kwargs))
        raise AssertionError("this evaluation-only campaign must never train a model")

    def load_fitted(cell, directory, spec, *, device):
        captured.source_loads.append((cell.slug, str(directory)))
        model = FrozenModel(cell)
        captured.models.append(model)
        return model

    torch = ModuleType("torch")
    torch.__version__ = "orchestration-double"
    torch.cuda = SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False)
    train = ModuleType("mpsae.train")
    train.FitSettings = lambda **kwargs: SimpleNamespace(**kwargs)
    train.fit = forbidden_fit
    train.fit_identity = _source_identity
    train.compatible_fit_identity = lambda saved, wanted: canonical_hash(saved) == canonical_hash(wanted)
    train.load_fitted = load_fitted
    runs = ModuleType("mpsae.runs")
    runs.RunSettings = lambda **kwargs: SimpleNamespace(**kwargs)
    runs.run_cell = forbidden_fit
    runs.dense_reference = forbidden_fit
    runs._close_code_arrays = lambda codes: None
    runs._close_failed_cache_maps = lambda *args: None
    for key, value in (("torch", torch), ("mpsae.train", train), ("mpsae.runs", runs)):
        monkeypatch.setitem(sys.modules, key, value)
    monkeypatch.setitem(mpsae.__dict__, "train", train)
    monkeypatch.setitem(mpsae.__dict__, "runs", runs)
    original_find_spec = importlib.util.find_spec

    def boundary_available(name, *args, **kwargs):
        # These runtime boundaries are replaced below; the preflight contract
        # itself has separate tests and must not require downloading packages.
        if name in {"torchvision", "timm", "datasets", "huggingface_hub"}:
            return importlib.machinery.ModuleSpec(name, loader=None)
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", boundary_available)

    def no_imagenet(*args, **kwargs):
        raise AssertionError("downstream-only evaluation must not load or prepare ImageNet")

    monkeypatch.setattr(features, "load_bundle", no_imagenet)
    for name in ("stage_embed", "stage_ffcv", "stage_imagenet"):
        monkeypatch.setattr(pipeline, name, no_imagenet)

    def dataset(name, *args, **kwargs):
        captured.dataset_loads.append(name)
        return SimpleNamespace(name=name, training_on_target=False)

    def frozen_features(spec, bundle, *args, expected_weights=None, **kwargs):
        assert bundle.training_on_target is False
        assert expected_weights is not None
        captured.feature_calls.append((spec.name, bundle.name, deepcopy(expected_weights)))
        return _feature_bundle(spec, bundle.name, expected_weights)

    monkeypatch.setattr(data, "load_dataset_bundle", dataset)
    monkeypatch.setattr(extract, "extract_bundle", frozen_features)

    encoder = ModuleType("mpsae.evaluate.encode")

    def encode(model, features, row_ids, k, *, cache_dir, identity, **kwargs):
        assert not model.training and not model.requires_gradient
        assert row_ids is None
        captured.encode_calls.append((model.cell.slug, k, identity["split"]))
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "owned-point-cache.txt").write_text("transient", encoding="utf-8")
        return {"indices": np.tile(np.arange(k, dtype=np.int32), (len(features), 1)),
                "values": np.full((len(features), k), 1 / k**0.5, dtype=np.float32),
                "stats": {"nnz_mean": float(k), "seconds": 0.01}}

    encoder.encode_split = encode
    monkeypatch.setitem(sys.modules, "mpsae.evaluate.encode", encoder)
    monkeypatch.setitem(evaluation_package.__dict__, "encode", encoder)

    def score(gallery, gallery_labels, queries, query_labels, **kwargs):
        captured.rank_calls.append(kwargs)
        if kwargs["exclude_self"]:
            assert kwargs["gallery_ids"] == kwargs["query_ids"]
        else:
            assert not set(kwargs["gallery_ids"]) & set(kwargs["query_ids"])
        return {"units": "fraction", "recall_at_1": 0.5, "recall_at_10": 1.0,
                "recall_at_100": 1.0, "precision_at_10": 0.5,
                "map_at_r": 0.5, "queries": len(query_labels), "seconds": 0.01}

    monkeypatch.setattr(ranking, "evaluate_retrieval", score)

    def downstream_report(records, out_dir, *, expected_points, failures, expected_seeds):
        captured.report_calls.append((len(records), expected_points, tuple(expected_seeds)))
        output = {"complete": not failures and len(records) == expected_points,
                  "counts": {"completed_points": len(records), "expected_points": expected_points}}
        atomic_json(out_dir / "manifest.json", output)
        return output

    monkeypatch.setattr(report, "build_report", downstream_report)
    monkeypatch.setattr(figures, "build_all", no_imagenet)
    return captured


def test_missing_ffcv_fails_before_downloading_any_dataset(tmp_path, monkeypatch, heavy_boundaries):
    from mpsae.report import figures

    campaign = Campaign.from_mapping({
        "name": "missing_dependencies", "backbones": ["rn18"], "methods": ["csr"],
        "seeds": [1227], "imagenet": {"train_k": [8]},
        "downstream": {"datasets": ["eurosat"], "train_k": [8], "eval_k": [8]},
    })
    monkeypatch.setattr(sys.modules["torch"].cuda, "is_available", lambda: True)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs:
                        None if name in {"ffcv", "cupy"}
                        else original_find_spec(name, *args, **kwargs))
    monkeypatch.setattr(figures, "build_all", lambda *args, **kwargs: {"skipped": []})
    with pytest.raises(RuntimeError, match="ImageNet GPU extraction needs ffcv, cupy") as failure:
        run.run_campaign(campaign, tmp_path, device="cuda:0")
    assert "bash setup_server.sh" in str(failure.value)
    assert "source activate_server.sh" in str(failure.value)
    assert "Docker" not in str(failure.value)
    assert heavy_boundaries.dataset_loads == []
    assert heavy_boundaries.source_loads == []
    assert heavy_boundaries.fit_calls == []


def test_broken_installed_ffcv_fails_before_downloading_any_dataset(tmp_path, monkeypatch, heavy_boundaries):
    from mpsae.report import figures
    from mpsae import ffcv_compat

    monkeypatch.setattr(ffcv_compat, "prepare_ffcv", lambda: "test")

    campaign = Campaign.from_mapping({
        "name": "broken_dependency", "backbones": ["rn18"], "methods": ["csr"],
        "seeds": [1227], "imagenet": {"train_k": [8]},
        "downstream": {"datasets": ["eurosat"], "train_k": [8], "eval_k": [8]},
    })
    monkeypatch.setattr(sys.modules["torch"].cuda, "is_available", lambda: True)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs:
                        importlib.machinery.ModuleSpec(name, loader=None) if name in {"ffcv", "cupy"}
                        else original_find_spec(name, *args, **kwargs))
    original_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "ffcv":
            raise OSError("libturbojpeg.so missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(figures, "build_all", lambda *args, **kwargs: {"skipped": []})
    with pytest.raises(RuntimeError, match="ffcv is installed but cannot import") as failure:
        run.run_campaign(campaign, tmp_path, device="cuda:0")
    assert "libturbojpeg.so missing" in str(failure.value)
    assert "bash setup_server.sh" in str(failure.value)
    assert heavy_boundaries.dataset_loads == []
    assert heavy_boundaries.source_loads == []


def test_failed_dense_reference_prevents_source_training(tmp_path, monkeypatch, heavy_boundaries):
    from mpsae import features
    from mpsae.report import figures

    campaign = Campaign.from_mapping({
        "name": "dense_failure", "backbones": ["rn18"], "methods": ["csr"],
        "seeds": [1227], "imagenet": {"train_k": [8]},
        "downstream": {"enabled": False},
    })
    for split in ("train", "val"):
        archive = tmp_path / "embeddings" / "rn18" / f"1K_{split}_rn18.npz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"fixture")
    monkeypatch.setattr(features, "load_bundle", lambda *args: object())

    def failed_reference(*args, **kwargs):
        raise RuntimeError("dense search failed")

    monkeypatch.setattr(sys.modules["mpsae.runs"], "dense_reference", failed_reference)
    monkeypatch.setattr(figures, "build_all", lambda *args, **kwargs: {"skipped": []})
    with pytest.raises(RuntimeError, match="partial report"):
        run.run_campaign(campaign, tmp_path, device="cpu", workers=0)
    assert heavy_boundaries.fit_calls == heavy_boundaries.source_loads == []
    status = run.campaign_status(campaign, tmp_path)
    assert status["state"]["tasks"]["imagenet/dense/rn18"]["state"] == "failed"


def test_downstream_campaign_reuses_source_and_resumes_all_dataset_budget_points(tmp_path, heavy_boundaries):
    campaign = _campaign()
    sources = [_write_source(tmp_path, cell) for cell in campaign.source_cells()]
    before = {str(path): path.read_bytes() for directory in sources for path in directory.iterdir()}

    observed = []
    observer = SimpleNamespace(on_task=lambda name, result: observed.append(name))
    manifest = run.run_campaign(campaign, tmp_path, device="cpu", workers=0, observer=observer)
    status = run.campaign_status(campaign, tmp_path)
    catalog = status["catalog"]
    expected = campaign.plan()
    assert manifest["complete"] and status["state"]["complete"]
    assert manifest["completed"] == manifest["expected"] == {
        "source_cells": 2, "imagenet_e1": 0, "imagenet_e2": 0}
    assert expected["downstream_sparse_points"] == 12
    assert expected["downstream_dense_points"] == 3
    assert len([name for name in observed if name.startswith("retrieval/")]) == 12
    assert len([name for name in observed if name.startswith("source/")]) == 2
    assert len(catalog["evaluations"]) == 15
    assert heavy_boundaries.fit_calls == []
    assert len(heavy_boundaries.models) == 2
    assert all(not model.training and not model.requires_gradient for model in heavy_boundaries.models)
    assert Counter(name for _, name, _ in heavy_boundaries.feature_calls) == {
        "eurosat": 1, "flowers102": 1, "sop": 1}
    assert len(heavy_boundaries.rank_calls) == 15
    assert all(point["mode"] == "frozen" and not point["training_on_target"]
               for point in catalog["evaluations"])
    for checkpoint in catalog["checkpoints"].values():
        points = [p for p in catalog["evaluations"] if p["checkpoint_id"] == checkpoint["checkpoint_id"]]
        assert len(points) == 6
        assert {p["dataset"] for p in points} == {"eurosat", "flowers102", "sop"}
        assert {p["eval_k"] for p in points} == {2, 8}
        assert {p["train_k"] for p in points} == {8}
    assert before == {str(path): path.read_bytes() for directory in sources for path in directory.iterdir()}
    rank_count, encode_count = len(heavy_boundaries.rank_calls), len(heavy_boundaries.encode_calls)
    repeated = run.run_campaign(campaign, tmp_path, device="cpu", workers=0)
    assert repeated["complete"]
    assert len(heavy_boundaries.rank_calls) == rank_count  # actual JSON result cache hit
    assert len(heavy_boundaries.encode_calls) == encode_count
    assert heavy_boundaries.fit_calls == []
    assert len(run.campaign_status(campaign, tmp_path)["catalog"]["evaluations"]) == 15
    assert not list((tmp_path / "cache" / "benchmark_codes").glob("*/owned-point-cache.txt"))


@pytest.mark.parametrize("incomplete", [False, True])
def test_missing_or_incomplete_source_never_trains_and_writes_partial_report(
        tmp_path, heavy_boundaries, incomplete):
    campaign = _campaign(methods=("csr",))
    cell = campaign.source_cells()[0]
    if incomplete:
        _write_source(tmp_path, cell, complete=False)
    with pytest.raises(RuntimeError, match="partial report"):
        run.run_campaign(campaign, tmp_path, device="cpu", workers=0)
    status = run.campaign_status(campaign, tmp_path)
    task = status["state"]["tasks"][f"source/{cell.slug}"]
    assert task["state"] == "failed"
    assert "source training disabled" in task["error"]
    assert not status["state"]["complete"]
    assert heavy_boundaries.fit_calls == heavy_boundaries.models == []
    assert heavy_boundaries.encode_calls == heavy_boundaries.rank_calls == []
    report = read_json(Path(status["report_manifest"]))
    assert report is not None and report["complete"] is False
    assert report["completed"]["source_cells"] == 0
    assert report["downstream"]["counts"]["expected_points"] == 9
    assert report["downstream"]["counts"]["completed_points"] == 0


def test_shared_features_are_rejected_for_another_source_backbone_revision(tmp_path, heavy_boundaries):
    campaign = _campaign(datasets=("eurosat",))
    first, second = campaign.source_cells()
    _write_source(tmp_path, first)
    _write_source(tmp_path, second, weights=_weights("e"))
    with pytest.raises(RuntimeError, match="partial report"):
        run.run_campaign(campaign, tmp_path, device="cpu", workers=0)
    status = run.campaign_status(campaign, tmp_path)
    points = status["catalog"]["evaluations"]
    assert len(heavy_boundaries.feature_calls) == 1  # real runner reused the bundle
    assert len(points) == 3  # dense plus first checkpoint's two budgets only
    assert {point["method"] for point in points} == {"dense", first.method}
    failed = status["state"]["tasks"][f"provenance/eurosat/{second.slug}"]
    assert failed["state"] == "failed" and "different backbone weights" in failed["error"]
    assert heavy_boundaries.fit_calls == []
    assert not read_json(Path(status["report_manifest"]))["complete"]


def test_direct_campaign_processes_targets_sequentially_and_resumes_without_download(
        tmp_path, monkeypatch, heavy_boundaries):
    from mpsae.benchmark import data, extract, storage
    from mpsae.embed.archive import finalize_archive
    campaign = replace(_campaign(), execution={**_campaign().execution,
                                              "storage_mode": "direct_v1", "cleanup_raw": True})
    root = run.campaign_paths(campaign, tmp_path)["workspace"]
    for cell in campaign.source_cells():
        _write_source(root, cell)
    events = []
    original_load = data.load_dataset_bundle
    original_extract = extract.extract_bundle
    def load(name, cache, **kwargs):
        events.append(("load", name))
        (cache / "hf_datasets").mkdir(parents=True)
        (cache / "hf_datasets" / "raw.arrow").write_bytes(b"raw")
        return original_load(name, cache, **kwargs)
    def embeddings(spec, dataset, out, cache, **kwargs):
        bundle = original_extract(spec, dataset, out, cache, **kwargs)
        for split in {id(bundle.gallery): bundle.gallery, id(bundle.query): bundle.query}.values():
            split.path = out / dataset.name / spec.name / f"{split.name}.npz"
            split.manifest = finalize_archive(split.path, split.features, split.labels,
                manifest={"ids_sha256": canonical_hash(split.ids)})
        events.append(("extract", dataset.name))
        return bundle
    monkeypatch.setattr(data, "load_dataset_bundle", load)
    monkeypatch.setattr(extract, "extract_bundle", embeddings)
    def verified(files):
        name = next(p.relative_to(root / "benchmarks").parts[0] for p in files)
        events.append(("backup", name))
        assert all(p.is_file() for p in files)
        return {"verified": name}
    observer = SimpleNamespace(on_task=lambda *a: None, verify_backup=verified)
    result = run.run_campaign(campaign, tmp_path, device="cpu", workers=0, observer=observer)
    assert result["complete"]
    assert events == [(action, name) for name in campaign.downstream["datasets"]
                      for action in ("load", "extract", "backup")]
    assert not list((root / "cache" / "raw_datasets").rglob("raw.arrow"))
    events.clear()
    result = run.run_campaign(campaign, tmp_path, device="cpu", workers=0, observer=observer)
    assert result["complete"]
    assert events == [("backup", name) for name in campaign.downstream["datasets"]]
    assert heavy_boundaries.fit_calls == []


def test_real_cpu_smoke_campaign_when_torch_is_available(tmp_path, monkeypatch):
    """A real SAE smoke integration, separate from the boundary-double tests."""
    pytest.importorskip("torch")
    original = BACKBONES["rn18"]
    tiny = replace(original, dim=12, k_list=(2, 4, 8), e2_anchors=(8,),
                   preset=replace(original.preset, hidden_dim=64, k_aux=8,
                                  batch_size=16, epochs=1))
    monkeypatch.setitem(BACKBONES, "rn18", tiny)
    campaign = Campaign.from_mapping({
        "name": "real_cpu_smoke", "backbones": ["rn18"], "methods": ["csr"],
        "seeds": [1227], "imagenet": {"train_k": [8], "eval_k": [4, 8], "e2": True,
                                       "linear_probe": False, "timing": False},
        "downstream": {"datasets": ["eurosat"], "train_k": [8], "eval_k": [2, 8]},
        "execution": {"max_steps": 2, "retain_codes": False},
    })
    manifest = run.run_campaign(campaign, tmp_path, device="cpu", workers=0,
                                 embed_batch_size=16, smoke=True)
    status = run.campaign_status(campaign, tmp_path, smoke=True)
    assert manifest["complete"] and status["state"]["complete"]
    assert manifest["completed"] == {"source_cells": 1, "imagenet_e1": 1, "imagenet_e2": 1}
    e2 = read_json(tmp_path / "smoke-real_cpu_smoke" / "runs" / "rn18" / "csr"
                   / "k8" / "seed1227" / "eval_e2.json")
    assert [point["k"] for point in e2["points"]] == [4, 8]
    assert len(status["catalog"]["evaluations"]) == 3
    assert all(not point["scientific"] for point in status["catalog"]["evaluations"])
    assert Path(status["workspace"]) != tmp_path
