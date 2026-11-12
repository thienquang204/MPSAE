"""Fresh-server boundaries; never installs packages or downloads weights in tests."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from mpsae.scene.config import SceneConfig
from mpsae.scene import server

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash') if os.name != 'nt' else 'C:/Program Files/Git/bin/bash.exe'
HAS_BASH = bool(BASH and Path(BASH).is_file())


@pytest.fixture
def boundaries(monkeypatch, tmp_path):
    import mpsae.scene.__main__ as cli
    import mpsae.scene.run as runner
    calls = {'restore': [], 'run': [], 'publish': []}
    monkeypatch.setattr(server, 'doctor', lambda *a: {'verified': True})
    monkeypatch.setattr(runner, 'source_records', lambda *a: [])
    def restore(config, **kwargs):
        calls['restore'].append(kwargs)
        return {'source_root': str(tmp_path / 'source'), 'revision': 'fixed-commit'}
    def run(config, workspace, source_root, **kwargs):
        calls['run'].append(str(source_root))
        return {'report': str(tmp_path / 'report'), 'complete': True}
    monkeypatch.setattr(cli, 'restore', restore)
    monkeypatch.setattr(runner, 'run', run)
    monkeypatch.setattr(server, 'publish_results', lambda *a: calls['publish'].append(a) or 'upload')
    return calls


def launch(tmp_path, **kwargs):
    return server.bootstrap(SceneConfig(), tmp_path, repo='test/models', revision='main',
                            prefix='direct-v1', device='cpu', **kwargs)


def test_fresh_bootstrap_and_pinned_resume(tmp_path, boundaries):
    first = launch(tmp_path)
    assert first['complete'] and first['source_revision'] == 'fixed-commit'
    assert not first['published'] and not boundaries['publish']
    assert boundaries['run'] == [str(tmp_path / 'source')]
    second = launch(tmp_path)
    assert second == first
    assert len(boundaries['restore']) == 1
    assert json.loads((tmp_path / 'latest.json').read_text())['evaluation_complete']


def test_explicit_refresh_and_publication(tmp_path, boundaries):
    launch(tmp_path)
    assert launch(tmp_path, refresh_source=True, publish=True)['published']
    assert len(boundaries['restore']) == 2 and len(boundaries['publish']) == 1


def test_missing_cached_source_redownloads_pinned_revision(tmp_path, boundaries, monkeypatch):
    import mpsae.scene.run as runner
    launch(tmp_path)
    def missing(*a):
        raise FileNotFoundError('cache removed')
    monkeypatch.setattr(runner, 'source_records', missing)
    launch(tmp_path)
    assert boundaries['restore'][-1]['revision'] == 'fixed-commit'


def test_failed_evaluation_records_stage_without_secret(tmp_path, boundaries, monkeypatch):
    import mpsae.scene.run as runner
    def broken(*a, **kw):
        raise RuntimeError('hf_SECRET_NOT_FOR_LOGS')
    monkeypatch.setattr(runner, 'run', broken)
    with pytest.raises(RuntimeError):
        launch(tmp_path)
    raw = (tmp_path / 'status.json').read_text()
    state = json.loads(raw)
    assert state['stage'] == 'evaluate' and not state['complete']
    assert 'hf_SECRET' not in raw
    assert list((tmp_path / 'sources').glob('*.json'))  # retry can reuse downloads
    assert not (tmp_path / 'latest.json').exists()


def test_failed_gpu_preflight_downloads_nothing(tmp_path, boundaries, monkeypatch):
    def broken(*a):
        raise RuntimeError('CUDA unavailable')
    monkeypatch.setattr(server, 'doctor', broken)
    with pytest.raises(RuntimeError):
        launch(tmp_path)
    assert boundaries['restore'] == boundaries['run'] == []


def test_missing_remote_artifacts_fails_before_download(monkeypatch):
    from mpsae.scene.__main__ import restore
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        dataset_info=lambda *a, **kw: SimpleNamespace(sha='revision', siblings=[])))
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *a, **kw: pytest.fail('must not download'))
    with pytest.raises(FileNotFoundError, match='checkpoint.pt'):
        restore(SceneConfig(), repo='x/y', revision='main', prefix='direct-v1', cache_dir='unused')


@pytest.mark.skipif(not HAS_BASH, reason='Bash unavailable')
@pytest.mark.parametrize('driver,capability,cuda,ok', [
    ('570.124.06', '12.0', 'cu128', True), ('580.65.06', '8.0;12.0', 'cu130', True),
    ('550.1', '12.0', 'cu128', False), ('580.65.06', '10.3', 'cu128', False),
    ('580.65.06', '10.3', 'cu130', True), ('580.65.06', '99.0', 'cu130', False)])
def test_driver_capability_guards(driver, capability, cuda, ok):
    result = subprocess.run([BASH, '-c', 'source setup_scene_server.sh; scene_cuda_profile "$1" "$2" "$3"',
                             'check', driver, capability, cuda], cwd=ROOT, capture_output=True, text=True)
    assert (result.returncode == 0) == ok, result.stderr


@pytest.mark.skipif(not HAS_BASH, reason='Bash unavailable')
def test_scripts_parse_and_help_without_install():
    for name in ('setup_scene_server.sh', 'scene_only.sh'):
        subprocess.run([BASH, '-n', name], cwd=ROOT, check=True)
        result = subprocess.run([BASH, name, '--help'], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0 and 'Usage:' in result.stdout


@pytest.mark.skipif(not HAS_BASH, reason='Bash unavailable')
def test_launcher_uses_restored_bootstrap_and_preserves_spaces(tmp_path):
    (tmp_path / 'scene_only.sh').write_text((ROOT / 'scene_only.sh').read_text(encoding='utf-8'), encoding='utf-8')
    venv = tmp_path / 'eval env' / 'bin'
    venv.mkdir(parents=True)
    python = venv / 'python'
    python.write_text('#!/usr/bin/env bash\nprintf "<%s>\\n" "$@"\n', encoding='utf-8')
    python.chmod(0o755)
    result = subprocess.run([BASH, 'scene_only.sh', '--venv', 'eval env', '--workspace', 'work with spaces'],
                             cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '<bootstrap>' in result.stdout and '<work with spaces>' in result.stdout
    assert 'train' not in result.stdout


@pytest.mark.skipif(not HAS_BASH, reason='Bash unavailable')
def test_launcher_stops_when_setup_fails(tmp_path):
    (tmp_path / 'scene_only.sh').write_text((ROOT / 'scene_only.sh').read_text(encoding='utf-8'), encoding='utf-8')
    (tmp_path / 'setup_scene_server.sh').write_text('#!/usr/bin/env bash\nexit 13\n', encoding='utf-8')
    result = subprocess.run([BASH, 'scene_only.sh', '--setup'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 13
    assert 'bootstrap' not in result.stdout


def test_requirements_are_lean_and_pinned():
    lines = (ROOT / 'requirements-scene.txt').read_text().lower().splitlines()
    specs = [line for line in lines if line and not line.startswith('#')]
    assert all('==' in line for line in specs)
    assert not any(word in '\n'.join(specs) for word in ('ffcv', 'cupy', 'wandb', 'numba'))


def test_doctor_actual_cpu_path_without_optional_dataset_stack(monkeypatch):
    import mpsae.server as general
    monkeypatch.setattr(general, 'verify_data_stack', lambda: {'note': 'data-stack mocked in this CPU test'})
    result = server.doctor('cpu')
    assert result['verified'] and not result['training']
