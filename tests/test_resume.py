"""Checkpoint reuse must preserve old experiments and reject changed recipes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

pytest.importorskip("torch")
np = pytest.importorskip("numpy")
import torch  # noqa: E402

from mpsae.config import BACKBONES, Cell  # noqa: E402
from mpsae.features import BatchSource, Bundle, Split  # noqa: E402
from mpsae.train import FitSettings, fit, fit_identity, load_fitted  # noqa: E402


@pytest.mark.parametrize("interrupt_at", [None, 10, 12])
def test_mpsaev2_extends_ten_epochs_to_identical_fifteen_epoch_fit(
        experiment, tmp_path, monkeypatch, interrupt_at):
    from mpsae import config, train
    from mpsae.benchmark.run import checkpoint_record
    from mpsae.io import read_json, read_jsonl

    cell, bundle, spec, settings = experiment
    cell = replace(cell, method="mpsaev2")
    experiment = cell, bundle, spec, settings
    out = tmp_path / "runs" / cell.slug
    with monkeypatch.context() as old_schedule:
        old_schedule.setitem(config.METHOD_EPOCHS, "mpsaev2", 10)
        old = _fit(experiment, out)
    assert old["steps"] == 10
    with pytest.raises(RuntimeError, match="15-epoch"):
        load_fitted(cell, out, spec, device="cpu")
    with pytest.raises(RuntimeError, match="15-epoch"):
        checkpoint_record(cell, tmp_path, require_provenance=False)

    if interrupt_at is not None:
        compute = train.methods.compute_step
        def interrupted(*args, **kwargs):
            if kwargs["step"] == interrupt_at:
                raise RuntimeError("interrupted extension")
            return compute(*args, **kwargs)
        with monkeypatch.context() as stopped:
            stopped.setattr(train.methods, "compute_step", interrupted)
            with pytest.raises(RuntimeError, match="interrupted extension"):
                _fit(experiment, out)
        assert not read_json(out / "summary.json")["complete"]
        with pytest.raises(FileNotFoundError, match="completed"):
            load_fitted(cell, out, spec, device="cpu")

    extended = _fit(experiment, out)
    reference = tmp_path / "reference"
    fresh = _fit(experiment, reference)
    assert extended["steps"] == fresh["steps"] == 15
    assert extended["signature"] == fresh["signature"] != old["signature"]
    assert extended["identity"]["objective"] == old["identity"]["objective"]
    assert extended["identity"]["optimizer"] == old["identity"]["optimizer"]
    left = torch.load(out / "checkpoint.pt", weights_only=False)
    right = torch.load(reference / "checkpoint.pt", weights_only=False)
    assert left["next_step"] == 15
    assert left["identity"]["schedule"]["epochs"] == 15
    torch.testing.assert_close(left["model"], right["model"], rtol=0, atol=0)
    torch.testing.assert_close(left["optimizer"], right["optimizer"], rtol=0, atol=0)
    assert [row["step"] for row in read_jsonl(out / "train.jsonl")] == list(range(1, 16))
    assert not load_fitted(cell, out, spec, device="cpu").training
    assert checkpoint_record(cell, tmp_path, require_provenance=False)["signature"] == fresh["signature"]
    before = _snapshot(out)
    assert _fit(experiment, out) == extended
    assert _snapshot(out) == before


@pytest.mark.parametrize("change", ["lr", "cl_coef", "seed", "data"])
def test_epoch_extension_still_rejects_any_other_recipe_change(
        experiment, tmp_path, monkeypatch, change):
    from mpsae import config
    cell, bundle, spec, settings = experiment
    cell = replace(cell, method="mpsaev2")
    with monkeypatch.context() as old_schedule:
        old_schedule.setitem(config.METHOD_EPOCHS, "mpsaev2", 10)
        _fit((cell, bundle, spec, settings), tmp_path)
    before = _snapshot(tmp_path)
    if change in ("lr", "cl_coef"):
        spec = replace(spec, preset=replace(spec.preset, **{change: getattr(spec.preset, change) * 2}))
    elif change == "seed":
        cell = replace(cell, seed=99)
    else:
        identity = deepcopy(bundle.identity)
        identity["train"]["sha256"] = "f" * 64
        bundle = replace(bundle, identity=identity)
    with pytest.raises(RuntimeError, match="different recipe"):
        _fit((cell, bundle, spec, settings), tmp_path)
    assert _snapshot(tmp_path) == before


@pytest.fixture
def experiment():
    generator = np.random.default_rng(54)
    train = generator.normal(size=(12, 6)).astype(np.float32)
    val = generator.normal(size=(4, 6)).astype(np.float32)
    labels = np.arange(12, dtype=np.int64) % 3
    vlabels = np.arange(4, dtype=np.int64) % 3
    identity = {"backbone": "ff2048", "dim": 6}
    for name, array in (("train", train), ("validation", val)):
        identity[name] = {"sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                          "rows": len(array), "recipe": None, "source": None}
    bundle = Bundle("ff2048", Split("train", train, labels, {}),
                    Split("validation", val, vlabels, {}), identity)
    preset = replace(BACKBONES["ff2048"].preset, hidden_dim=32, k_aux=4,
                     batch_size=4, epochs=1, ot_microbatch=2)
    spec = replace(BACKBONES["ff2048"], dim=6, preset=preset,
                   k_list=(2, 4), e2_anchors=(2, 4))
    cell = Cell("ff2048", "csr", 2, 1227)
    settings = FitSettings(device="cpu", residency="cpu", checkpoint_every=1, log_every=1)
    return cell, bundle, spec, settings


def _fit(experiment, out):
    cell, bundle, spec, settings = experiment
    return fit(cell, bundle, out, settings, code_version="2.0.0", spec=spec)


def _snapshot(out):
    return {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in out.iterdir() if path.is_file()}


def _enrich(bundle):
    identity = deepcopy(bundle.identity)
    for name in ("train", "validation"):
        identity[name].update(recipe="existing-extraction-v1", source="pinned-backbone-revision")
    return replace(bundle, identity=identity)


def test_completed_fit_reuses_files_without_erasing_metrics(experiment, tmp_path):
    first = _fit(experiment, tmp_path)
    before = _snapshot(tmp_path)
    second = _fit(experiment, tmp_path)
    assert second == first
    assert second["final_metrics"]["loss_total"] > 0
    assert _snapshot(tmp_path) == before


def test_completed_fit_accepts_only_missing_provenance_enrichment(experiment, tmp_path):
    first = _fit(experiment, tmp_path)
    before = _snapshot(tmp_path)
    cell, bundle, spec, settings = experiment
    second = _fit((cell, _enrich(bundle), spec, settings), tmp_path)
    assert second == first
    assert second["identity"]["data"]["train"]["recipe"] is None
    assert _snapshot(tmp_path) == before


def test_new_config_records_backbone_provenance_without_changing_fit_identity(experiment, tmp_path):
    from mpsae.io import read_json

    cell, bundle, spec, settings = experiment
    manifest = {"weights": {"repository": "source", "revision": "a" * 40,
                             "file": "model.pt", "sha256": "b" * 64},
                "input_size": 224, "normalize": [[0.5] * 3, [0.5] * 3],
                "preprocessing": {"crop": 224, "resize": 256}}
    bundle = replace(bundle, train=replace(bundle.train, manifest=manifest))
    first = _fit((cell, bundle, spec, settings), tmp_path)
    config = read_json(tmp_path / "config.json")
    assert config["backbone_weights"] == manifest["weights"]
    assert config["backbone_preprocessing"] == {key: manifest[key] for key in
                                                ("input_size", "normalize", "preprocessing")}
    assert "backbone_weights" not in first["identity"]
    before = _snapshot(tmp_path)
    _fit(experiment, tmp_path)  # A later missing sidecar must not erase saved provenance.
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("change", ["lr", "nested_coef", "seed", "data", "rows", "code_version"])
def test_changed_recipe_is_rejected_before_metadata_is_written(experiment, tmp_path, change):
    _fit(experiment, tmp_path)
    before = _snapshot(tmp_path)
    cell, bundle, spec, settings = experiment
    version = "2.0.0"
    if change == "lr":
        spec = replace(spec, preset=replace(spec.preset, lr=spec.preset.lr * 2))
    elif change == "nested_coef":
        spec = replace(spec, preset=replace(spec.preset, nested_coef=0.25))
    elif change == "seed":
        cell = replace(cell, seed=2026)
    elif change in ("data", "rows"):
        identity = deepcopy(bundle.identity)
        identity["train"]["sha256" if change == "data" else "rows"] = "f" * 64 if change == "data" else 13
        bundle = replace(bundle, identity=identity)
    else:
        version = "changed-training-implementation"
    with pytest.raises(RuntimeError, match="different recipe"):
        fit(cell, bundle, tmp_path, settings, code_version=version, spec=spec)
    assert _snapshot(tmp_path) == before


def test_ot_refinement_budget_is_part_of_fit_identity(experiment):
    cell, bundle, spec, settings = experiment
    cell = replace(cell, method="mpsae")
    default = fit_identity(cell, bundle, spec.preset, total_steps=3, per_epoch=3, code_version="v")
    changed = fit_identity(cell, bundle, replace(spec.preset, ot_max_refine_iters=3000),
                           total_steps=3, per_epoch=3, code_version="v")
    assert default != changed
    assert default["objective"]["ot"]["max_refine_iters"] == 8000


@pytest.mark.parametrize("kwargs", [{"log_every": 0}, {"checkpoint_every": 0},
                                     {"max_steps": -1}, {"max_steps": True}])
def test_invalid_fit_settings_are_rejected_before_writing(kwargs):
    with pytest.raises(ValueError):
        FitSettings(**kwargs)


def test_load_fitted_rejects_wrong_cell(experiment, tmp_path):
    _fit(experiment, tmp_path)
    cell, bundle, spec, settings = experiment
    with pytest.raises(RuntimeError, match="different training cell"):
        load_fitted(replace(cell, seed=2026), tmp_path, spec, device="cpu")


def test_two_campaigns_cannot_write_the_same_fit(experiment, tmp_path):
    from mpsae.io import exclusive_lock

    with exclusive_lock(tmp_path / "fit.lock", timeout=0):
        with pytest.raises(TimeoutError):
            _fit(experiment, tmp_path)
    assert not (tmp_path / "config.json").exists()


def test_known_provenance_change_is_not_legacy_enrichment(experiment, tmp_path):
    cell, bundle, spec, settings = experiment
    bundle = _enrich(bundle)
    _fit((cell, bundle, spec, settings), tmp_path)
    before = _snapshot(tmp_path)
    identity = deepcopy(bundle.identity)
    identity["train"]["recipe"] = "different-extraction"
    with pytest.raises(RuntimeError, match="different recipe"):
        _fit((cell, replace(bundle, identity=identity), spec, settings), tmp_path)
    assert _snapshot(tmp_path) == before


def test_partial_checkpoint_enrichment_resumes_exact_weights(experiment, tmp_path, monkeypatch):
    partial, reference = tmp_path / "partial", tmp_path / "reference"
    original = BatchSource.epoch

    def interrupt_after_one(self, epoch, *, skip_batches=0):
        for index, batch, labels in original(self, epoch, skip_batches=skip_batches):
            if index == 1:
                raise RuntimeError("simulated interruption")
            yield index, batch, labels

    with monkeypatch.context() as patch:
        patch.setattr(BatchSource, "epoch", interrupt_after_one)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            _fit(experiment, partial)
    saved = torch.load(partial / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["next_step"] == 1
    original_config = (partial / "config.json").read_bytes()
    cell, bundle, spec, settings = experiment
    resumed = _fit((cell, _enrich(bundle), spec, settings), partial)
    uninterrupted = _fit(experiment, reference)
    assert resumed["signature"] == saved["signature"] == uninterrupted["signature"]
    assert resumed["identity"] == saved["identity"]
    assert resumed["steps"] == 3 and resumed["final_metrics"]
    assert (partial / "config.json").read_bytes() == original_config
    left = torch.load(partial / "checkpoint.pt", map_location="cpu", weights_only=False)
    right = torch.load(reference / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(value, right["model"][key]) for key, value in left["model"].items())


def test_missing_summary_recovers_final_metrics_without_rewriting_checkpoint(experiment, tmp_path):
    first = _fit(experiment, tmp_path)
    checkpoint = (tmp_path / "checkpoint.pt").read_bytes()
    (tmp_path / "summary.json").unlink()
    recovered = _fit(experiment, tmp_path)
    assert recovered["final_metrics"] == first["final_metrics"]
    assert recovered["signature"] == first["signature"]
    assert (tmp_path / "checkpoint.pt").read_bytes() == checkpoint


def test_config_only_mismatch_is_also_rejected_before_writes(experiment, tmp_path):
    _fit(experiment, tmp_path)
    (tmp_path / "checkpoint.pt").unlink()
    (tmp_path / "summary.json").unlink()
    before = _snapshot(tmp_path)
    cell, bundle, spec, settings = experiment
    changed = replace(spec, preset=replace(spec.preset, lr=1e-3))
    with pytest.raises(RuntimeError, match="different recipe"):
        _fit((cell, bundle, changed, settings), tmp_path)
    assert _snapshot(tmp_path) == before


def test_cpu_fit_never_calls_cuda_memory_or_rng_state_helpers(experiment, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU fit attempted a CUDA operation")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for name in ("synchronize", "reset_peak_memory_stats", "max_memory_allocated",
                 "get_rng_state_all", "set_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    summary = _fit(experiment, tmp_path)
    assert summary["peak_memory_bytes"] is None
    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert "cuda" not in checkpoint["rng"]
