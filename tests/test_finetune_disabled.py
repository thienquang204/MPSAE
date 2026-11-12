import json

import pytest

from mpsae.finetune.config import load_config


def test_active_plan_has_no_additional_fits_or_evaluations():
    config = load_config("configs/finetune.yaml")
    assert not config.enabled and config.cells() == []
    plan = config.plan()
    for key in ("additional_fits", "source_cells", "imagenet_e1_points", "imagenet_e2_points",
                "imagenet_unique_points", "imagenet_probe_points", "frozen_control_1nn_points",
                "transfer_knn_points", "transfer_retrieval_points", "transfer_records"):
        assert plan[key] == 0
    assert plan["source_checkpoints_required"] == []
    from mpsae.benchmark.config import load_campaign
    assert load_campaign("configs/thesis.yaml").plan()["source_cells"] == 39
    archived = load_config("configs/archive/finetune_gradient_replay.yaml")
    assert archived.enabled and archived.plan()["additional_fits"] == 9


@pytest.mark.parametrize("command", ["plan", "run", "transfer", "prepare", "restore-sources",
                                     "report", "transfer-report", "sync"])
def test_disabled_cli_does_no_work_or_remote_writes(command, monkeypatch, tmp_path, capsys):
    from mpsae import cli
    from mpsae.finetune.__main__ import main
    monkeypatch.setattr(cli, "load_dotenv", lambda *a: pytest.fail("disabled CLI loaded credentials"))
    sentinel = tmp_path / "checkpoint.pt"
    sentinel.write_bytes(b"existing result")
    main([command, "--workspace", str(tmp_path), "--backup", "--download-images"])
    result = json.loads(capsys.readouterr().out)
    assert result["enabled"] is False and result["additional_fits"] == 0
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing result"


@pytest.mark.parametrize("module,function", [("run", "run"), ("transfer", "run_transfer")])
def test_direct_runner_also_refuses_disabled_campaign(module, function, tmp_path):
    from importlib import import_module
    runner = getattr(import_module(f"mpsae.finetune.{module}"), function)
    with pytest.raises(RuntimeError, match="omitted"):
        runner(load_config("configs/finetune.yaml"), tmp_path)
    assert not list(tmp_path.iterdir())
