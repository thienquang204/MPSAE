"""Historical MPSAEv2 artifacts stay identifiable and excluded from experiments.

These tests exercise storage, report, checkpoint and stage-receipt boundaries;
the loss and gradient equivalence tests live in test_methods.py.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pytest

from mpsae import __version__, config, pipeline
from mpsae.benchmark.config import Campaign
from mpsae.io import atomic_json, canonical_hash, read_json, read_jsonl
from mpsae.report import aggregate


def test_only_hybrid_fit_directory_changes():
    for method in config.IMPLEMENTED_METHODS:
        cell = config.Cell("rn18", method, 8, 1227)
        if method == "mpsaev2":
            assert cell.slug == "rn18/mpsaev2__fixed_ot_cl_v1/k8/seed1227"
        else:
            assert cell.slug == f"rn18/{method}/k8/seed1227"
        decoded = config.method_from_directory(config.method_directory(method))
        assert decoded == (method if method in config.METHODS else None)
    assert config.method_from_directory("mpsaev2") is None
    assert config.method_from_directory("mpsaev2__retired_variant") is None


def test_historical_variant_changes_do_not_change_active_plans(monkeypatch):
    campaign = Campaign.from_mapping({"backbones": ["rn18"],
        "seeds": [1227], "imagenet": {"train_k": [8]},
        "downstream": {"enabled": False}})
    assert "method_variants" not in campaign.resolved()
    before = campaign.plan()["plan_id"]
    monkeypatch.setitem(config.METHOD_VARIANTS, "mpsaev2", "future_objective")
    assert campaign.plan()["plan_id"] == before


def test_report_discovery_and_selected_logs_ignore_all_retired_hybrid_directories(tmp_path):
    root = tmp_path / "runs"
    cell = config.Cell("ff2048", "mpsaev2", 8, 1227)
    current = root / cell.slug
    legacy = root / "ff2048" / "mpsaev2" / "k8" / "seed1227"
    legacy.mkdir(parents=True)
    # Ignored artifacts must not even be parsed under the new method label.
    (legacy / "config.json").write_text("invalid historical JSON", encoding="utf-8")
    (legacy / "train.jsonl").write_text("invalid historical log", encoding="utf-8")
    atomic_json(legacy / "eval_e1.json", {"top1": 99.0})
    atomic_json(current / "config.json", {"code_version": __version__, "total_steps": 1,
        "identity": {"objective": {"variant": config.METHOD_VARIANTS["mpsaev2"]}}})
    atomic_json(current / "eval_e1.json", {"top1": 65.0})
    (current / "train.jsonl").write_text(json.dumps({"step": 1, "elapsed": 2.0,
        "loss_total": 0.25, "training_k": 8}) + "\n", encoding="utf-8")

    records = aggregate.load_runs(root)
    assert records == []
    for selection in (None, [("ff2048", "mpsaev2", 8, 1227)]):
        rows = aggregate.training_summary(root, selected_cells=selection)
        assert rows == []
    assert (legacy / "train.jsonl").read_text() == "invalid historical log"
    assert read_json(current / "eval_e1.json")["top1"] == 65.0


@pytest.mark.parametrize("saved_variant", [None, "retired_annealing"])
@pytest.mark.parametrize("payload", ["e1", "e2"])
def test_explicit_retired_results_are_excluded_regardless_of_objective(tmp_path, monkeypatch,
                                                                    saved_variant, payload):
    from mpsae.report.figures import build_all
    monkeypatch.setattr("mpsae.report.figures.apply_style", lambda: {})

    record = {"backbone": "ff2048", "method": "mpsaev2", "k": 8, "seed": 1227,
              "config": {"identity": {"objective": {"variant": saved_variant}}}}
    record[payload] = ({"top1": 99.0} if payload == "e1"
                       else {"points": [{"k": 8, "top1": 99.0}]})
    # Historical objective validation remains available for checkpoint audits,
    # but retired results cannot enter an active report in either variant.
    with pytest.raises(ValueError, match="objective variant"):
        aggregate.validate_result_variant(record)
    manifest = build_all(tmp_path, tmp_path / "report", records=[record])
    assert manifest["counts"]["runs"] == manifest["counts"]["training_logs"] == 0
    record["config"]["identity"]["objective"]["variant"] = config.METHOD_VARIANTS["mpsaev2"]
    aggregate.validate_result_variant(record)
    actual = (aggregate.aggregate_e1 if payload == "e1" else aggregate.aggregate_e2)([record])
    assert actual == []


def test_result_discovery_ignores_retired_partial_and_complete_directories(tmp_path):
    directory = tmp_path / "runs" / config.Cell("ff2048", "mpsaev2", 8, 1227).slug
    directory.mkdir(parents=True)
    partial = aggregate.load_runs(tmp_path)
    assert partial == []
    assert aggregate.aggregate_e1(partial) == aggregate.aggregate_e2(partial) == []
    atomic_json(directory / "eval_e1.json", {"top1": 99.0})
    assert aggregate.load_runs(tmp_path) == []


def test_real_hybrid_fit_logs_both_regularizers_at_fixed_k(tmp_path, tiny_hybrid):
    import math
    from mpsae.train import fit, load_fitted

    torch, cell, spec, bundle, settings = tiny_hybrid
    spec = replace(spec, preset=replace(spec.preset, epochs=2))
    settings = replace(settings, max_steps=2, log_every=1, checkpoint_every=1)
    summary = fit(cell, bundle, tmp_path, settings, spec=spec, code_version=__version__)
    assert summary["complete"] and summary["steps"] == 2
    objective = summary["identity"]["objective"]
    assert objective["variant"] == config.METHOD_VARIANTS["mpsaev2"]
    assert "annealing" not in objective and "ot" in objective and "contrastive" in objective
    logs = read_jsonl(tmp_path / "train.jsonl")
    assert len(logs) == 2 and {row["training_k"] for row in logs} == {cell.k}
    for row in logs:
        assert all(math.isfinite(row[key]) for key in ("loss_total", "loss_ot", "loss_contrastive"))
        assert row["loss_contrastive"] > 0
    model = load_fitted(cell, tmp_path, spec, device="cpu")
    with torch.no_grad():
        z = model.encode(torch.from_numpy(bundle.train.features), cell.k)
    assert (torch.count_nonzero(z, dim=1) <= cell.k).all()


@pytest.fixture
def tiny_hybrid(monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from mpsae.features import Bundle, Split
    from mpsae.train import FitSettings

    original = config.BACKBONES["ff2048"]
    preset = replace(original.preset, hidden_dim=32, k_aux=4, batch_size=4,
                     epochs=1, ot_microbatch=2)
    spec = replace(original, dim=6, preset=preset, k_list=(2, 4), e2_anchors=(2, 4))
    monkeypatch.setitem(config.BACKBONES, "ff2048", spec)
    features = np.arange(24, dtype=np.float32).reshape(4, 6) / 24
    split = Split("train", features, np.array([0, 0, 1, 1]), {})
    bundle = Bundle("ff2048", split, split,
                    {"train": {"sha256": "a" * 64, "rows": 4}})
    cell = config.Cell("ff2048", "mpsaev2", 2, 1227)
    settings = FitSettings(device="cpu", residency="cpu", max_steps=1)
    return torch, cell, spec, bundle, settings


def _checkpoint(directory, experiment, *, retired):
    torch, cell, spec, bundle, settings = experiment
    from mpsae.model import build_sae
    from mpsae.train import fit_identity

    identity = fit_identity(cell, bundle, spec.preset, total_steps=15, per_epoch=1,
                            code_version=__version__)
    if retired:
        identity = deepcopy(identity)
        identity["objective"].pop("variant")
        identity["objective"].pop("contrastive")
        identity["objective"]["annealing"] = {"shape": "linear", "k_init": 8,
            "k_final": cell.k, "decay_ratio": 0.7, "reference": "CSRv2 get_current_k"}
    signature = canonical_hash(identity)
    model = build_sae(spec.dim, spec.preset.hidden_dim, seed=cell.seed)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": identity, "signature": signature,
                "model": model.state_dict(), "next_step": 15}, directory / "checkpoint.pt")
    atomic_json(directory / "config.json", {"identity": identity, "signature": signature,
                                            "code_version": __version__})
    atomic_json(directory / "summary.json", {"identity": identity, "signature": signature,
        "complete": True, "steps": 15, "cell": cell.__dict__})
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def test_current_checkpoint_loads_but_copied_annealed_checkpoint_is_rejected(tmp_path, tiny_hybrid):
    from mpsae.train import load_fitted
    from mpsae.benchmark.run import checkpoint_record

    _, cell, spec, _, _ = tiny_hybrid
    directory = tmp_path / "runs" / cell.slug
    _checkpoint(directory, tiny_hybrid, retired=False)
    assert not load_fitted(cell, directory, spec, device="cpu").training
    assert checkpoint_record(cell, tmp_path, require_provenance=False)["cell"] == cell.__dict__

    # A folder name alone cannot establish which objective trained its weights.
    before = _checkpoint(directory, tiny_hybrid, retired=True)
    with pytest.raises(RuntimeError, match="variant|objective|recipe"):
        load_fitted(cell, directory, spec, device="cpu")
    with pytest.raises(RuntimeError, match="variant|objective|recipe"):
        checkpoint_record(cell, tmp_path, require_provenance=False)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_old_fit_identity_is_rejected_without_rewriting_artifacts(tmp_path, tiny_hybrid):
    from mpsae.train import _fit

    _, cell, spec, bundle, settings = tiny_hybrid
    before = _checkpoint(tmp_path, tiny_hybrid, retired=True)
    with pytest.raises(RuntimeError, match="different recipe"):
        _fit(cell, bundle, tmp_path, settings, spec=spec, code_version=__version__)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_profiler_rejects_retired_checkpoint_before_writing_output(tmp_path, tiny_hybrid):
    from mpsae import profile_train

    directory = tmp_path / "old"
    before = _checkpoint(directory, tiny_hybrid, retired=True)
    output = tmp_path / "profile.json"
    with pytest.raises(SystemExit):
        profile_train.build_parser().parse_args([
            "--device", "cpu", "--backbone", "ff2048", "--method", "mpsaev2", "--k", "2",
            "--warmup", "0", "--steps", "1", "--checkpoint", str(directory / "checkpoint.pt"),
            "--out", str(output)])
    assert not output.exists()
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("method", ["csr", "csrv2", "mpsae"])
def test_active_method_legacy_receipts_remain_reusable(tmp_path, monkeypatch, method):
    options = pipeline.PipelineOptions(paths=pipeline.Paths(tmp_path), backbones=("ff2048",),
        methods=(method,), seeds=(1227,), k_values=(8,),
        skip=tuple(name for name in pipeline.STAGES if name not in ("env", "train", "report")))
    options.paths.create()
    old_identity = canonical_hash({"version": __version__, "options": asdict(options)})
    for name in ("env", "train", "report"):
        atomic_json(options.paths.state / f"{name}.json",
                    {"ok": True, "identity": old_identity, "seconds": 0, "detail": {}})
    calls = []
    monkeypatch.setattr(pipeline, "setup_logging", lambda *args, **kwargs: None)
    for name in ("env", "train", "report"):
        def handler(options, stage_name=name):
            calls.append(stage_name)
            return {}
        monkeypatch.setitem(pipeline.HANDLERS, name, handler)
    pipeline.run_all(options)
    assert calls == []
    calls.clear()
    pipeline.run_all(options)
    assert calls == []


def test_legacy_status_does_not_count_removed_methods(tmp_path):
    cell = config.Cell("ff2048", "mpsae", 8, 1227)
    options = pipeline.PipelineOptions(paths=pipeline.Paths(tmp_path), backbones=(cell.backbone,),
                                        methods=(cell.method,), seeds=(cell.seed,), k_values=(cell.k,))
    legacy = options.paths.runs / cell.backbone / "mpsaev2" / "k8" / "seed1227" / "eval_e1.json"
    atomic_json(legacy, {"top1": 99})
    assert pipeline.status(options)["done"] == 0
    atomic_json(options.paths.runs / cell.slug / "eval_e1.json", {"top1": 65})
    assert pipeline.status(options)["done"] == 1
    assert read_json(legacy) == {"top1": 99}
