"""Retired methods must not enter any current experiment or profile."""
from pathlib import Path

import pytest

from mpsae import cli
from mpsae.benchmark.config import Campaign, load_campaign
from mpsae.config import METHODS, Cell, grid
from mpsae.pipeline import PipelineOptions


RETIRED = ("csr_multitopk", "mpsaev2")


def test_every_shipped_config_and_default_grid_excludes_retired_methods():
    assert METHODS == ("csr", "csrv2", "mpsae")
    assert {cell.method for cell in grid()} == set(METHODS)
    for path in Path("configs").glob("*.yaml"):
        campaign = load_campaign(path)
        assert set(campaign.methods) <= set(METHODS), path
        assert {cell.method for cell in campaign.source_cells()} <= set(METHODS), path
        for dataset in campaign.downstream["datasets"]:
            assert {cell.method for cell in campaign.downstream_cells(dataset)} <= set(METHODS)


@pytest.mark.parametrize("method", RETIRED)
def test_explicit_experiment_selections_reject_retired_methods(method):
    with pytest.raises(ValueError, match="methods"):
        Campaign.from_mapping({"methods": [method]})
    with pytest.raises(ValueError, match="not enabled"):
        Campaign(methods=(method,))
    with pytest.raises(ValueError, match="not enabled"):
        PipelineOptions(methods=(method,))
    with pytest.raises(ValueError, match="not enabled"):
        grid(methods=(method,))
    # Validate the selection even when no training K would match.
    with pytest.raises(ValueError, match="not enabled"):
        grid(methods=(method,), k_values=(9999,))


@pytest.mark.parametrize("method", RETIRED)
@pytest.mark.parametrize("source", ["flag", "environment"])
def test_legacy_cli_rejects_retired_methods_without_creating_workspace(
        tmp_path, monkeypatch, method, source):
    for key in ("MPSAE_CONFIG", "MPSAE_METHODS"):
        monkeypatch.delenv(key, raising=False)
    workspace = tmp_path / "untouched"
    args = ["--env-file", str(tmp_path / "absent.env"), "--workspace", str(workspace)]
    if source == "flag":
        args += ["--methods", method]
    else:
        monkeypatch.setenv("MPSAE_METHODS", method)
    with pytest.raises(SystemExit, match="unknown method"):
        cli.main(args + ["plan"])
    assert not workspace.exists()


@pytest.mark.parametrize("method", RETIRED)
def test_direct_training_evaluation_and_profiler_reject_before_io(tmp_path, method):
    pytest.importorskip("torch")
    from mpsae import profile_train, runs

    cell = Cell("ff2048", method, 8, 1227)
    settings = runs.RunSettings(root=tmp_path / "runs")
    with pytest.raises(ValueError, match="not enabled"):
        runs.run_cell(cell, None, settings, code_version="test")
    with pytest.raises(ValueError, match="not enabled"):
        runs.run_e2(cell, None, settings, signature="test", e1={})
    with pytest.raises(ValueError, match="not enabled"):
        runs.run_grid([cell], settings, code_version="test")
    with pytest.raises(SystemExit):
        profile_train.build_parser().parse_args(["--method", method])
    args = profile_train.build_parser().parse_args([])
    args.method = method
    with pytest.raises(ValueError, match="not enabled"):
        profile_train.run_profile(args)
    assert not (tmp_path / "runs").exists()
