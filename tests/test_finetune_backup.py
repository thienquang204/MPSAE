from pathlib import Path
from types import SimpleNamespace

import pytest

from mpsae.finetune import backup


def test_checkpoint_upload_is_throttled_and_finished_fit_is_left_to_summary_upload(monkeypatch, tmp_path):
    times = iter([0, 10, 30, 31, 80])
    monkeypatch.setattr(backup.time, "monotonic", lambda: next(times))
    calls = []
    session = SimpleNamespace(root=tmp_path, ensure_repository=lambda: None,
                              upload=calls.append, warn=lambda *a: pytest.fail(str(a)))
    observer = backup.IncrementalBackup(session, interval=30)
    for step in (1, 2, 3, 4):
        observer.checkpoint(tmp_path / "runs", step, 4)
    assert calls == [tmp_path / "runs"]
    observer.results()
    assert calls[1:] == [tmp_path / "evaluations", tmp_path / "campaigns"]


def test_failed_intermediate_upload_does_not_destroy_training_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(backup.time, "monotonic", lambda: 0)
    warnings = []
    def broken(*args):
        raise ConnectionError("fixture")
    observer = backup.IncrementalBackup(SimpleNamespace(ensure_repository=broken, warn=lambda *a: warnings.append(a)))
    observer.last_checkpoint = -3600
    observer.checkpoint(tmp_path, 100, 1000)
    assert warnings and observer.last_checkpoint == 0


def test_sync_rejects_failed_backup_without_training(monkeypatch, tmp_path):
    from mpsae import remote
    class Session:
        directory = tmp_path
        def __init__(self, *args): pass
        def start(self): pass
        def finish(self, **kwargs): return False
    monkeypatch.setattr(remote, "RemoteSession", Session)
    config = SimpleNamespace(name="fixture", plan=lambda: {"plan_id": "p"})
    with pytest.raises(RuntimeError, match="synchronization incomplete"):
        backup.sync(config, tmp_path)


def test_restore_only_required_checkpoints_and_provenance_and_preserves_conflicts(tmp_path, monkeypatch):
    import huggingface_hub
    from mpsae.finetune import run
    from mpsae.finetune.restore import restore_sources
    cell = SimpleNamespace(slug="ff2048/csr/k8/seed1227")
    config = SimpleNamespace(cells=lambda: [cell], campaign=SimpleNamespace(backbones=["ff2048"]))
    required = [f"direct-v1/runs/{cell.slug}/{name}" for name in ("config.json", "summary.json", "checkpoint.pt")]
    sidecar = "direct-v1/embeddings/ff2048/1K_train_ff2048.npz.json"
    forbidden = ["direct-v1/embeddings/ff2048/1K_train_ff2048.npz", "direct-v1/runs/other/checkpoint.pt"]
    class Api:
        def __init__(self, **kwargs): pass
        def repo_info(self, *args, **kwargs): return SimpleNamespace(sha="a" * 40)
        def list_repo_files(self, *args, **kwargs): return required + [sidecar] + forbidden
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    downloaded = []
    blob = tmp_path / "blob"
    blob.write_bytes(b"fixture")
    def fetch(repo, name, **kwargs):
        assert kwargs["revision"] == "a" * 40
        downloaded.append(name)
        return str(blob)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    monkeypatch.setattr(run, "preflight_sources", lambda *args: {})
    root = tmp_path / "work"
    result = restore_sources(config, root, "owner/repo")
    assert result["files"] == 4 and set(downloaded) == set(required + [sidecar])
    path = root / required[0]
    path.write_text("different existing checkpoint")
    with pytest.raises(RuntimeError, match="existing source differs"):
        restore_sources(config, root, "owner/repo")
    assert path.read_text() == "different existing checkpoint"
