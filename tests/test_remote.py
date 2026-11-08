"""Cloud behavior tested with explicit fake HF/W&B clients; no network access."""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from mpsae import remote
from mpsae.benchmark.config import Campaign
from mpsae.io import atomic_json, read_json
from mpsae.logs import METRIC_SINK, MetricWriter


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setattr(remote.os, "environ", dict(remote.os.environ))
    for name in ("WANDB_API_KEY", "WANDB_TOKEN", "WANDB_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_fixture_secret")
    monkeypatch.setenv("WANDB_TOKEN", "wb_fixture_secret")
    state = SimpleNamespace(created=[], uploads=[], runs=[], exists=False, private=True,
                            fail_upload=False, fail_log=False, init_arguments=[])

    class Api:
        def __init__(self, token):
            assert token == "hf_fixture_secret"

        def whoami(self):
            return {"name": "researcher"}

        def create_repo(self, **kwargs):
            state.created.append(kwargs)
            if not state.exists:
                state.private = kwargs["private"]
            state.exists = True

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=state.private)

        def upload_folder(self, **kwargs):
            if state.fail_upload:
                raise OSError("upload failed with hf_fixture_secret")
            state.uploads.append(kwargs)

    class Run:
        def __init__(self):
            self.metrics, self.definitions, self.artifacts, self.finishes = [], [], [], []

        def log(self, record):
            if state.fail_log:
                raise OSError("logging failed with wb_fixture_secret")
            self.metrics.append(record)

        def define_metric(self, *args, **kwargs):
            self.definitions.append((args, kwargs))

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def finish(self, **kwargs):
            self.finishes.append(kwargs)

    class Artifact:
        def __init__(self, *args, **kwargs):
            self.directories = []

        def add_dir(self, path):
            self.directories.append(path)

    def init(**kwargs):
        state.init_arguments.append(kwargs)
        run = Run()
        state.runs.append(run)
        return run

    hf = ModuleType("huggingface_hub")
    hf.HfApi = Api
    wandb = ModuleType("wandb")
    wandb.init = init
    wandb.Settings = lambda **kwargs: kwargs
    wandb.Artifact = Artifact
    wandb.Table = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    campaign = Campaign.from_mapping({"name": "remote_fixture", "backbones": ["rn18"],
        "methods": ["csr"], "seeds": [1227], "imagenet": {"enabled": False},
        "downstream": {"datasets": ["sop"], "train_k": [8], "eval_k": [8]}})
    return remote.RemoteSession(campaign, tmp_path), state


def test_only_tokens_resolve_private_mpsae_repo_and_cvpr_project(services):
    session, state = services
    session.start()
    assert session.repo == "researcher/MPSAE"
    assert state.created == [{"repo_id": "researcher/MPSAE", "repo_type": "dataset",
                              "private": True, "exist_ok": True}]
    assert state.init_arguments[0]["project"] == "cvpr"
    assert "entity" not in state.init_arguments[0]
    assert remote.os.environ["WANDB_API_KEY"] == "wb_fixture_secret"
    metadata = (session.directory / "status.json").read_text()
    assert "hf_fixture_secret" not in metadata and "wb_fixture_secret" not in metadata


def test_repo_is_recreated_and_wandb_run_resumed_after_deletion(services):
    session, state = services
    session.start()
    run_id = state.init_arguments[-1]["id"]
    state.exists = False
    again = remote.RemoteSession(session.campaign, session.root)
    again.start()
    assert state.exists and len(state.created) == 2
    assert state.init_arguments[-1]["id"] == run_id
    assert state.init_arguments[-1]["resume"] == "allow"


def test_incomplete_tracking_receipt_generates_a_new_resumable_id(services):
    session, state = services
    atomic_json(session.directory / "wandb.json", {"project": "cvpr"})
    session.start()
    assert state.init_arguments[0]["id"]
    assert read_json(session.directory / "wandb.json")["id"] == state.init_arguments[0]["id"]


def test_public_existing_repo_refuses_artifact_upload(services):
    session, state = services
    state.exists, state.private = True, False
    with pytest.raises(RuntimeError, match="exists publicly"):
        session.start()
    assert not state.uploads and not state.runs


def test_backup_selects_artifacts_without_env_raw_images_or_sdk_files(services):
    session, state = services
    session.start()
    atomic_json(session.root / "runs/a/config.json", {"identity": "source"})
    (session.root / "runs/a/checkpoint.pt").write_bytes(b"checkpoint")
    (session.root / "runs/a/checkpoint.partial").write_bytes(b"incomplete")
    (session.root / "runs/a/.env").write_text("SECRET=keep-local")
    atomic_json(session.root / "runs/a/wandb/metadata.json", {"private": "SDK"})
    atomic_json(session.root / "runs/a/codes/metadata.json", {"private": "temporary"})
    (session.root / "runs/a/image.jpg").write_bytes(b"raw image")
    atomic_json(session.root / "imagenet/raw.json", {"not": "an artifact"})
    session.upload(session.root / "runs")
    assert state.uploads[-1]["allow_patterns"] == ["a/checkpoint.pt", "a/config.json"]
    with pytest.raises(ValueError, match="allowlist"):
        session.upload(session.root / "imagenet")
    with pytest.raises(ValueError, match="inside the workspace"):
        session.upload(session.root.parent)


def test_checkpoint_backup_failure_is_retried_at_final_sync_and_secrets_redacted(services):
    session, state = services
    session.start()
    atomic_json(session.root / "runs/rn18/csr/k8/seed1227/summary.json", {"complete": True})
    state.fail_upload = True
    session.on_task("source/rn18/csr/k8/seed1227", None)
    metadata = (session.directory / "status.json").read_text()
    assert "[REDACTED]" in metadata and "hf_fixture_secret" not in metadata
    state.fail_upload = False
    assert session.finish(scientific_success=True)
    assert session.status["hf_sync_complete"] is True
    assert any(call["path_in_repo"] == "runs" for call in state.uploads)


def test_remote_training_sink_preserves_local_jsonl_and_uses_per_cell_steps(services):
    session, state = services
    session.start()
    writer = MetricWriter(session.root / "runs/train.jsonl", stage="rn18/csr/k8/seed1227")
    token = METRIC_SINK.set(session.training_metrics)
    try:
        writer.log({"loss_total": 1.5}, step=10)
        writer.log({"loss_total": 1.0}, step=20)
        state.fail_log = True
        writer.log({"loss_total": 0.5}, step=30)
    finally:
        METRIC_SINK.reset(token)
    assert len(writer.path.read_text().splitlines()) == 3
    assert len(state.runs[0].definitions) == 1
    assert state.runs[0].metrics[0]["train/rn18/csr/k8/seed1227/step"] == 10
    assert session.wandb_failed
    assert "wb_fixture_secret" not in (session.directory / "status.json").read_text()


def test_hf_only_does_not_require_wandb_token_or_project(services, monkeypatch):
    session, state = services
    monkeypatch.delenv("WANDB_TOKEN")
    monkeypatch.delenv("WANDB_API_KEY")
    session.start()
    assert not state.runs
    assert session.finish(scientific_success=True)


def test_smoke_creates_no_remote_repo_or_run(services):
    session, state = services
    smoke = remote.RemoteSession(session.campaign, session.root, smoke=True)
    smoke.start()
    smoke.on_task("source/example", None)
    assert smoke.finish(scientific_success=True)
    assert not state.created and not state.runs and not state.uploads


def test_finish_exports_report_and_marks_failed_science_as_failed_run(services):
    session, state = services
    session.start()
    atomic_json(session.paths["reports"] / "manifest.json", {"complete": False})
    atomic_json(session.paths["state"] / "catalog.json", {"evaluations": []})
    assert session.finish(scientific_success=False)
    assert state.runs[0].artifacts
    assert state.runs[0].finishes == [{"exit_code": 1}]
    assert session.status["scientific_success"] is False


def test_wandb_is_finished_even_when_final_report_logging_fails(services):
    session, state = services
    session.start()
    state.fail_log = True
    assert not session.finish(scientific_success=True)
    assert state.runs[0].finishes == [{"exit_code": 1}]
    assert session.status["wandb_complete"] is False
    assert "wb_fixture_secret" not in (session.directory / "status.json").read_text()


def test_main_backs_up_partial_results_and_restores_sink_on_campaign_failure(services, monkeypatch, tmp_path):
    session, state = services
    from mpsae.benchmark import run as runner

    def fail(*args, **kwargs):
        assert kwargs["observer"] is session
        atomic_json(session.root / "runs/partial/summary.json", {"complete": False})
        raise RuntimeError("fixture scientific failure")

    monkeypatch.setattr(remote, "load_campaign", lambda path: session.campaign)
    monkeypatch.setattr(remote, "RemoteSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(runner, "run_campaign", fail)
    with pytest.raises(RuntimeError, match="fixture scientific failure"):
        remote.main(["run", "--env-file", str(tmp_path / "absent.env")])
    assert METRIC_SINK.get() is None
    assert any(call["path_in_repo"] == "runs" for call in state.uploads)
    assert state.runs[0].finishes == [{"exit_code": 1}]


@pytest.mark.parametrize("scientific_error", [RuntimeError("original scientific failure"), KeyboardInterrupt()])
def test_cleanup_failure_does_not_replace_original_scientific_error(services, monkeypatch, scientific_error):
    session, state = services
    from mpsae.benchmark import run as runner

    def fail(*args, **kwargs):
        raise scientific_error

    def fail_finish(**kwargs):
        raise OSError("fixture backup disk failure with hf_fixture_secret")

    monkeypatch.setattr(runner, "run_campaign", fail)
    monkeypatch.setattr(session, "finish", fail_finish)
    args = SimpleNamespace(command="run", workspace=str(session.root), device="cpu",
                           workers=0, embed_batch_size=2, smoke=False)
    with pytest.raises(type(scientific_error)) as caught:
        remote.execute(session, args)
    assert caught.value is scientific_error
    assert METRIC_SINK.get() is None
    assert "hf_fixture_secret" not in "".join(getattr(caught.value, "__notes__", []))
