import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location('onefile', ROOT/'scripts/segmentation_server_onefile.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_matches_kaggle_and_does_not_launch(tmp_path, capsys, monkeypatch):
    module = launcher()
    monkeypatch.setattr(module, 'command', lambda *a, **kw: pytest.fail('plan launched a command'))
    module.main(['--skip-setup', '--repo', str(ROOT), '--workspace', str(tmp_path/'unused'), '--plan'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['sae_fits'] == 33 and plan['total_evaluations'] == 34
    assert [p['mpsae_ot_mass'] for p in plan['plans']] == [.9, .95, 1]
    assert plan['plans'][-1]['solver'] == 'balanced-log-sinkhorn-v1'
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('args', [['--masses', '1.1'], ['--masses', 'nan'],
                                ['--masses', '.9', '.9'], ['--methods', 'csr']])
def test_invalid_grid_before_install(args):
    with pytest.raises(SystemExit):
        launcher().main(args)


def test_setup_reused_and_no_implicit_git_pull(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = launcher()
    repo = tmp_path/'repo'
    for name in ('mpsae/patch_scene/weight_sweep.py', 'requirements-patch-scene.txt',
                 'scripts/setup_segmentation_server.sh'):
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    calls = []
    def command(argv, repo=None):
        calls.append(argv)
        python = tmp_path/'repo/.venv-segmentation/bin/python'
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setattr(module.shutil, 'which', lambda x: '/usr/bin/'+x)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: 'refactor\n')
    monkeypatch.setattr(module, 'command', command)
    args = SimpleNamespace(repo=repo, workspace=tmp_path, skip_setup=False, update=False)
    module.bootstrap(args)
    module.bootstrap(args)
    assert len(calls) == 1 and calls[0][0] == 'bash'
    (repo/'requirements-patch-scene.txt').write_text('changed')
    module.bootstrap(args)
    assert len(calls) == 2


def test_local_two_mass_two_worker_smoke_and_resume(tmp_path):
    pytest.importorskip('torch')
    module = launcher()
    workspace = tmp_path/'run'
    args = ['--skip-setup', '--repo', str(ROOT), '--workspace', str(workspace),
            '--masses', '.9', '1', '--weights', '.5', '--train-k', '3',
            '--smoke', '--devices', 'cpu', 'cpu']
    module.main(args)
    archive = workspace/'SMOKE_ONLY/segmentation-results.zip'
    with ZipFile(archive) as bundle:
        report = json.loads(bundle.read('results.json'))
        assert report['complete'] and report['completed'] == report['expected'] == 5
        assert sum(r['method'] == 'csr' for r in report['results']) == 1
        assert {r['mmpot_mass'] for r in report['results'] if r['method'] == 'mpsae'} == {.9, 1}
        assert len([n for n in bundle.namelist() if n.endswith('weight-sweep-results.zip')]) == 2
        assert 'mass_weight_comparison.csv' in bundle.namelist()
    checkpoints = list((workspace/'voc/SMOKE_ONLY/fits').glob('*/checkpoint.pt'))
    assert len(checkpoints) == 4
    stamps = {p: p.stat().st_mtime_ns for p in checkpoints}
    module.main(args)
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())
