"""Only MPSAE fits above training K=64 use the reduced MMPOT coefficient."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mpsae import config, methods
from mpsae.io import atomic_json, canonical_hash
from mpsae.model import build_sae
from mpsae.train import compatible_fit_identity, fit_identity, load_fitted


def test_campaign_marks_only_the_affected_training_cell_not_inference_budgets():
    from mpsae.benchmark.config import Campaign, load_campaign

    current = load_campaign("configs/thesis.yaml")
    assert current.resolved()["cell_objective_overrides"] == {
        "ff2048/mpsae/k256/seed1227": {"ot_weight": 0.5},
        "vit_b16/mpsae/k192/seed1227": {"ot_weight": 0.5},
        "convnext_tiny/mpsae/k192/seed1227": {"ot_weight": 0.5},
        "rn18/mpsae/k128/seed1227": {"ot_weight": 0.5},
        "dinov2_vits14/mpsae/k192/seed1227": {"ot_weight": 0.5}}
    low_k = Campaign.from_mapping({"backbones": ["ff2048"], "methods": ["mpsae"],
        "imagenet": {"train_k": [8], "eval_k": [256]},
        "downstream": {"train_k": [8], "eval_k": [256]}})
    assert "cell_objective_overrides" not in low_k.resolved()


def test_effective_training_preset_changes_only_the_high_k_ot_coefficient():
    from dataclasses import asdict

    original = config.FF2048_PRESET
    low = config.training_preset("mpsae", original, k=64)
    high = config.training_preset("mpsae", original, k=65)
    assert asdict(high) == {**asdict(low), "ot_weight": 0.5}
    assert low.ot_weight == 2.0
    assert original.ot_weight == 2.0  # resolving a fit never mutates the shared preset


@pytest.mark.parametrize("k,weight", [(8, 2.0), (32, 2.0), (64, 2.0), (65, 0.5), (128, 0.5), (256, 0.5)])
def test_described_weight_uses_training_budget_not_nested_view_width(k, weight):
    preset = config.FF2048_PRESET
    assert config.effective_ot_weight("mpsae", k, preset) == weight
    assert methods.describe("mpsae", k, preset)["ot"]["weight"] == weight


@pytest.mark.parametrize("method", ["csr", "csrv2", "csr_multitopk", "mpsaev2"])
def test_other_methods_preserve_their_preset_weight(method):
    preset = replace(config.FF2048_PRESET, ot_weight=1.7)
    assert config.effective_ot_weight(method, 256, preset) == 1.7
    description = methods.describe(method, 256, preset)
    if method == "mpsaev2":
        assert description["ot"]["weight"] == 1.7
    else:
        assert "ot" not in description


@pytest.mark.parametrize("k,weight", [(8, 2.0), (64, 2.0), (65, 0.5), (256, 0.5)])
def test_actual_ot_component_and_gradient_use_the_requested_coefficient(monkeypatch, k, weight):
    preset = replace(config.FF2048_PRESET, hidden_dim=1024, k_aux=8, ot_microbatch=4)
    model = build_sae(6, preset.hidden_dim, seed=1227)
    batch = torch.randn(4, 6, generator=torch.Generator().manual_seed(42))
    captured = {}

    def gap(main, middle, wide, **kwargs):
        raw = main.square().sum() + 0.3 * middle.square().sum() + 0.1 * wide.square().sum()
        captured["raw"] = raw
        return raw, {}

    monkeypatch.setattr(methods.ot, "matching_gap", gap)
    output = methods.compute_step("mpsae", model, batch, k=k, preset=preset,
                                  step=0, total_steps=1, update_activity=False)
    assert output.training_k == k
    assert float(captured["raw"].detach()) > 0
    assert torch.allclose(output.components["ot"], weight * captured["raw"])
    # Gradient check uses the real encoder/gates and independent raw-gap graph,
    # rather than comparing full objectives whose gates differ across budgets.
    parameters = tuple(model.parameters())
    actual = torch.autograd.grad(output.components["ot"], parameters, retain_graph=True, allow_unused=True)
    reference = torch.autograd.grad(captured["raw"], parameters, allow_unused=True)
    assert any(value is not None and torch.count_nonzero(value) > 0 for value in reference)
    for observed, raw in zip(actual, reference):
        assert (observed is None) == (raw is None)
        if raw is not None:
            torch.testing.assert_close(observed, weight * raw)
    assert output.metrics["loss_ot"] == pytest.approx(float(captured["raw"].detach()))


@pytest.mark.parametrize("k,compatible", [(8, True), (64, True), (65, False), (256, False)])
def test_old_weight_identity_is_compatible_only_at_unchanged_budgets(k, compatible):
    cell = config.Cell("rn18", "mpsae", k, 1227)
    bundle = SimpleNamespace(identity={"train": {"sha256": "a" * 64, "rows": 4}})
    requested = fit_identity(cell, bundle, config.TIMM_PRESET, total_steps=1,
                             per_epoch=1, code_version="test")
    old = deepcopy(requested)
    old["objective"]["ot"]["weight"] = 2.0
    assert compatible_fit_identity(old, requested) is compatible
    assert (canonical_hash(old) == canonical_hash(requested)) is compatible


def _historical_checkpoint(tmp_path, monkeypatch, k):
    original = config.BACKBONES["rn18"]
    preset = replace(original.preset, hidden_dim=512, k_aux=8, batch_size=4, epochs=1)
    spec = replace(original, dim=128, preset=preset, k_list=(64, 65), e2_anchors=(64, 65))
    monkeypatch.setitem(config.BACKBONES, "rn18", spec)
    cell = config.Cell("rn18", "mpsae", k, 1227)
    bundle = SimpleNamespace(identity={"train": {"sha256": "a" * 64, "rows": 4}},
                             train=SimpleNamespace(rows=4))
    identity = fit_identity(cell, bundle, preset, total_steps=1, per_epoch=1, code_version="test")
    identity["objective"]["ot"]["weight"] = 2.0  # completed historical recipe
    signature = canonical_hash(identity)
    directory = tmp_path / "runs" / cell.slug
    directory.mkdir(parents=True)
    model = build_sae(spec.dim, preset.hidden_dim, seed=cell.seed)
    torch.save({"identity": identity, "signature": signature, "next_step": 1,
                "model": model.state_dict()}, directory / "checkpoint.pt")
    atomic_json(directory / "summary.json", {"identity": identity, "signature": signature,
                "complete": True, "steps": 1, "cell": cell.__dict__})
    atomic_json(directory / "config.json", {"identity": identity, "signature": signature,
                "code_version": "test"})
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    return cell, spec, bundle, directory, signature, before


@pytest.mark.parametrize("k", [64, 65])
def test_old_checkpoint_loading_and_catalog_respect_weight_without_rewriting_files(tmp_path, monkeypatch, k):
    from mpsae.benchmark.run import checkpoint_record

    cell, spec, _, directory, signature, before = _historical_checkpoint(tmp_path, monkeypatch, k)
    if k <= 64:
        assert not load_fitted(cell, directory, spec, device="cpu").training
        assert checkpoint_record(cell, tmp_path, require_provenance=False)["signature"] == signature
    else:
        with pytest.raises(RuntimeError, match="weight|recipe|objective"):
            load_fitted(cell, directory, spec, device="cpu")
        with pytest.raises(RuntimeError, match="weight|recipe|objective"):
            checkpoint_record(cell, tmp_path, require_provenance=False)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_high_budget_old_fit_cannot_resume_or_force_overwrite_different_recipe(tmp_path, monkeypatch):
    from mpsae.train import FitSettings, _fit

    cell, spec, bundle, directory, _, before = _historical_checkpoint(tmp_path, monkeypatch, 65)
    for force in (False, True):
        with pytest.raises(RuntimeError, match="different recipe"):
            _fit(cell, bundle, directory, FitSettings(device="cpu", force=force),
                 spec=spec, code_version="test")
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_high_budget_report_cannot_relabel_old_weight(tmp_path):
    from mpsae.report.figures import build_all

    record = {"backbone": "rn18", "method": "mpsae", "k": 65, "seed": 1227,
              "config": {"identity": {"objective": {"ot": {"weight": 2.0}}}},
              "e1": {"top1": 90.0}, "e2": None, "complete": True}
    with pytest.raises(RuntimeError, match="weight"):
        build_all(tmp_path, tmp_path / "reports", records=[record])
    assert not (tmp_path / "reports").exists()


def test_complete_e2_cache_does_not_bypass_changed_training_weight(tmp_path):
    from mpsae import runs

    cell = config.Cell("rn18", "mpsae", 256, 1227)
    directory = tmp_path / "runs" / cell.slug
    atomic_json(directory / "config.json", {"identity": {"objective": {"ot": {"weight": 2.0}}}})
    point = {"k": 256, "top1": 90.0}
    atomic_json(directory / "eval_e2.json", {"signature": "old-fit", "points": [point]})
    settings = runs.RunSettings(root=tmp_path / "runs", linear_probe=False, measure_timing=False,
                                e2_k_values=(256,))
    assert runs._cached_e1_complete(point, settings)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(RuntimeError, match="weight"):
        runs.run_e2(cell, None, settings, signature="old-fit", e1=point)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
