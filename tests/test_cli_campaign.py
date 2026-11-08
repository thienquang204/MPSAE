"""One config drives Docker commands; legacy metadata commands still work."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mpsae import cli

ROOT = Path(__file__).resolve().parents[1]


def test_dotenv_is_loaded_before_building_parser_defaults(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.os, "environ", dict(cli.os.environ))
    for key in ("MPSAE_CONFIG", "MPSAE_BACKBONES", "MPSAE_METHODS", "MPSAE_SEEDS", "MPSAE_K"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / "settings.env"
    env.write_text("MPSAE_BACKBONES=ff2048\nMPSAE_METHODS=csr\nMPSAE_SEEDS=1227\nMPSAE_K=8\n")
    assert cli.main(["--env-file", str(env), "plan"]) == 0
    output = capsys.readouterr().out
    assert "1 training cells" in output
    assert "methods: csr" in output


@pytest.mark.parametrize("command", ["plan", "status"])
def test_docker_default_campaign_metadata_needs_no_torch_or_credentials(
        tmp_path, monkeypatch, capsys, command):
    monkeypatch.setenv("MPSAE_CONFIG", str(ROOT / "configs" / "smoke.yaml"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert cli.main(["--env-file", str(tmp_path / "absent.env"), "--workspace", str(tmp_path), command]) == 0
    payload = json.loads(capsys.readouterr().out)
    plan = payload["plan"] if command == "status" else payload
    assert plan["name"] == "smoke"
    assert plan["downstream_sparse_points"] == 18
    assert not (tmp_path / "campaigns").exists()


def test_campaign_cli_rejects_silently_ignored_scientific_overrides(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(ROOT / "configs" / "smoke.yaml"), "--methods", "csr", "plan"])
    assert error.value.code == 2
    assert "select experiments in YAML" in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [("--workers", "-1"), ("--embed-batch-size", "0"),
                                          ("--max-steps", "-1"), ("--force", "typo")])
def test_invalid_runtime_options_are_rejected_early(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([option, value, "plan"])
    assert error.value.code == 2


def test_explicit_campaign_report_rejects_legacy_out(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(ROOT / "configs" / "smoke.yaml"), "--workspace", str(tmp_path),
                  "report", "--out", str(tmp_path / "other")])
    assert error.value.code == 2
    assert "fixed by campaign name/plan" in capsys.readouterr().err


def test_legacy_smoke_uses_a_separate_workspace(tmp_path):
    args = cli.build_parser().parse_args(["--workspace", str(tmp_path), "--smoke", "run"])
    assert cli.to_options(args).paths.workspace == tmp_path / "smoke-legacy"
