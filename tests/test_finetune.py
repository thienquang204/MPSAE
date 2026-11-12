"""Joint gradients, replayable resumes, source isolation and sparse evaluation."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from mpsae.finetune.config import DEFAULTS, load_config
from mpsae.config import BACKBONES, Cell
from mpsae.io import read_json, sha256_file


def test_pilot_is_nine_additional_fits_with_existing_probe_selection():
    config = load_config("configs/finetune.yaml")
    plan = config.plan()
    assert len(config.cells()) == plan["additional_fits"] == 9
    assert {(c.backbone, c.k) for c in config.cells()} == {("ff2048", 8), ("convnext_tiny", 12), ("vit_b16", 12)}
    assert {c.method for c in config.cells()} == {"csr", "csrv2", "mpsae"}
    assert plan["imagenet_e2_points"] == 51
    assert plan["imagenet_unique_points"] == plan["frozen_control_1nn_points"] == 51
    assert plan["imagenet_probe_points"] == 24
    assert plan["downstream_sparse_points"] == 0
    assert not config.campaign.imagenet["timing"]
    assert config.training["epochs"] == 3 and config.training["batch_size"] == 16
    # All cell training hyperparameters are overridden uniformly, while source recipes are untouched.
    from mpsae.benchmark.config import load_campaign
    assert load_campaign("configs/thesis.yaml").plan()["source_cells"] == 39


@pytest.mark.parametrize("key,value", [("epochs", 0), ("batch_size", 1), ("backbone_lr", float("nan")),
    ("sae_lr", -1), ("precision", "fp16"), ("ce_weight", 0), ("freeze_bn_stats", "yes"), ("typo", 1)])
def test_invalid_finetune_settings_are_rejected(tmp_path, key, value):
    import yaml
    payload = yaml.safe_load(Path("configs/finetune.yaml").read_text())
    payload["finetune"][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        load_config(path)


def fixture_models(method="csr"):
    import torch
    from torch import nn
    from mpsae.model import build_sae
    torch.manual_seed(91)
    spec = replace(BACKBONES["ff2048"], dim=16, k_list=(2, 8, 16), e2_anchors=(8,),
        preset=replace(BACKBONES["ff2048"].preset, hidden_dim=64, k_aux=8, ot_microbatch=4))
    network = nn.Sequential(nn.Flatten(), nn.Linear(192, 16), nn.Tanh(), nn.Dropout(0.1))
    teacher = deepcopy(network).eval().requires_grad_(False)
    sae = build_sae(16, 64, seed=7)
    head = nn.Linear(64, 4)
    return Cell("ff2048", method, 8, 7), spec, network, teacher, sae, head


@pytest.mark.parametrize("method", ["csr", "csrv2", "mpsae"])
def test_joint_loss_updates_backbone_and_sae_but_not_teacher(method):
    torch = pytest.importorskip("torch")
    from mpsae.finetune.train import joint_step
    cell, spec, network, teacher, sae, head = fixture_models(method)
    before = deepcopy(teacher.state_dict())
    images, labels = torch.randn(4, 3, 8, 8), torch.arange(4)
    loss, metrics = joint_step(cell, spec, network, teacher, sae, head, images, labels,
        options={**DEFAULTS, "precision": "float32"}, step=0, total=6)
    loss.backward()
    assert metrics["training_k"] == 8  # CSRv2 does not restart source annealing
    assert metrics["nnz_mean"] <= 8
    for module in (network, sae, head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    for key, tensor in teacher.state_dict().items():
        torch.testing.assert_close(tensor, before[key], atol=0, rtol=0)
    assert torch.isfinite(loss)


def test_teacher_target_has_no_gradient_and_default_objective_is_unchanged():
    torch = pytest.importorskip("torch")
    from mpsae.methods import compute_step
    cell, spec, _, _, sae, _ = fixture_models()
    x = torch.randn(4, 16, requires_grad=True)
    target = torch.randn(4, 16, requires_grad=True)
    arguments = dict(k=8, preset=spec.preset, step=0, total_steps=4, update_activity=False)
    original = compute_step("csr", sae, x, **arguments)
    same = compute_step("csr", sae, x, reconstruction_target=x.detach(), **arguments)
    torch.testing.assert_close(original.loss, same.loss, atol=0, rtol=0)
    anchored = compute_step("csr", sae, x, reconstruction_target=target, **arguments)
    anchored.loss.backward()
    assert target.grad is None and x.grad is not None
    with pytest.raises(ValueError, match="reconstruction target"):
        compute_step("csr", sae, x, reconstruction_target=target[:, :4], **arguments)


def augment(image):
    import torch
    return image + torch.rand_like(image) * 0.01


def test_interrupted_joint_training_resumes_exactly_and_rejects_changed_recipe(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils.data import TensorDataset
    from mpsae.finetune import train
    torch.manual_seed(77)
    dataset = TensorDataset(torch.randn(12, 3, 8, 8), torch.arange(12) % 4)
    options = {**DEFAULTS, "precision": "float32", "epochs": 2, "batch_size": 4, "checkpoint_every": 1,
               "log_every": 1}
    def fit(directory, **changes):
        cell, spec, network, teacher, sae, _ = fixture_models()
        return train.fit_joint(cell, spec, network, teacher, sae, dataset, augment, directory,
            options={**options, **changes}, source_identity={"source": "fixed"}, data_identity={"data": "fixture"},
            preprocessing={"augmentation": "seeded"}, device="cpu", classes=4)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    fit(full)
    actual_step = train.joint_step
    def interrupt(*args, step, **kwargs):
        if step == 2:
            raise KeyboardInterrupt("simulate interrupted image batch")
        return actual_step(*args, step=step, **kwargs)
    monkeypatch.setattr(train, "joint_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fit(resumed)
    checkpoint = resumed / "checkpoint.pt"
    assert torch.load(checkpoint, weights_only=False)["next_step"] == 2
    monkeypatch.setattr(train, "joint_step", actual_step)
    fit(resumed)
    first, second = (torch.load(p / "checkpoint.pt", weights_only=False) for p in (full, resumed))
    for group in ("backbone", "sae", "classifier"):
        for key in first[group]:
            torch.testing.assert_close(first[group][key], second[group][key], atol=0, rtol=0)
    before = sha256_file(checkpoint)
    monkeypatch.setattr(train, "joint_step", lambda *args, **kwargs: pytest.fail("completed fit retrained"))
    assert fit(resumed)["complete"]
    assert sha256_file(checkpoint) == before
    with pytest.raises(RuntimeError, match="changed fine-tuning recipe"):
        fit(resumed, backbone_lr=0.01)
    assert sha256_file(checkpoint) == before


def test_batchnorm_statistics_freeze_but_affine_gradients_train():
    torch = pytest.importorskip("torch")
    from torch import nn
    from mpsae.finetune.train import train_mode
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Flatten())
    train_mode(model, True)
    before = model[0].running_mean.clone()
    model(torch.randn(4, 3, 8, 8)).square().mean().backward()
    assert torch.equal(before, model[0].running_mean)
    assert model[0].weight.grad is not None


def test_frozen_sources_required_without_modifying_them(tmp_path):
    pytest.importorskip("torch")
    from mpsae.finetune.run import preflight_sources
    root = tmp_path / "frozen"
    root.mkdir()
    sentinel = root / "existing.npz"
    sentinel.write_bytes(b"cached embedding")
    with pytest.raises(FileNotFoundError, match="missing completed source fit"):
        preflight_sources(load_config("configs/finetune.yaml"), root)
    assert list(root.iterdir()) == [sentinel]


@pytest.mark.parametrize("capped,completed", [(False, True), (True, True), (False, False)])
def test_source_preflight_requires_full_compatible_schedule(tmp_path, monkeypatch, capped, completed):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from mpsae import __version__, train
    from mpsae.benchmark import run as benchmark
    from mpsae.config import training_preset
    from mpsae.finetune.run import preflight_sources
    from mpsae.io import atomic_json
    config = load_config("configs/finetune.yaml")
    loaded = []

    def record(cell, root):
        preset = training_preset(cell.method, BACKBONES[cell.backbone].preset, k=cell.k)
        per_epoch = -(-1_281_167 // preset.batch_size)
        total = preset.epochs * per_epoch
        identity = train.fit_identity(cell, SimpleNamespace(identity={"train": {"rows": 1_281_167}}),
            preset, total_steps=total - 1 if capped else total, per_epoch=per_epoch,
            code_version=__version__)
        path = root / "runs" / cell.slug / "checkpoint.pt"
        atomic_json(path.parent / "summary.json", {"steps": total if completed else total - 1})
        return {"path": str(path), "training_identity": identity, "cell": cell.__dict__,
                "backbone_weights": {"same": cell.backbone}}

    monkeypatch.setattr(benchmark, "checkpoint_record", record)
    monkeypatch.setattr(train, "load_fitted", lambda cell, *a, **kw: loaded.append(cell.slug))
    if capped or not completed:
        with pytest.raises(RuntimeError, match="capped training recipe|full training schedule"):
            preflight_sources(config, tmp_path)
        assert not loaded
    else:
        assert len(preflight_sources(config, tmp_path)) == len(loaded) == 9


def test_all_nine_smoke_cells_are_sparse_and_resume_without_training(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from mpsae.finetune import run as runner, train
    from mpsae.report import figures
    from mpsae.benchmark import extract
    # Plot rendering is exercised by the standalone smoke command; this test
    # checks actual neural updates/extraction/search and disk cache reuse.
    monkeypatch.setattr(figures, "build_all", lambda *args, **kwargs: {"skipped": []})
    monkeypatch.setattr(extract, "_network", lambda *args, **kwargs: pytest.fail("smoke downloaded weights"))
    config = load_config("configs/finetune.yaml")
    result = runner.run(config, tmp_path, device="cpu", workers=0, smoke=True)
    assert result["complete"] and not result["scientific"]
    root = runner.output_root(config, tmp_path, True)
    catalog = read_json(root / "campaigns" / config.name / config.plan()["plan_id"] / "catalog.json")
    assert len(catalog["imagenet"]) == len(catalog["frozen"]) == 9
    for record in catalog["imagenet"]:
        for point in record["e2"]["points"]:
            assert 0 <= point["nnz_mean"] <= point["k"]
            assert "probe_top1" not in point
    checkpoint = next((root / "runs").rglob("checkpoint.pt"))
    before = sha256_file(checkpoint)
    monkeypatch.setattr(train, "joint_step", lambda *args, **kwargs: pytest.fail("resume retrained"))
    from mpsae import runs
    monkeypatch.setattr(runs, "evaluate_at", lambda *args, **kwargs: pytest.fail("resume recomputed metrics"))
    assert runner.run(config, tmp_path, device="cpu", workers=0, smoke=True)["complete"]
    assert sha256_file(checkpoint) == before
    assert not (tmp_path / "direct-v1").exists()
