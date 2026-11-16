from dataclasses import replace
import json

import pytest

from mpsae.io import atomic_json
from mpsae.patch_scene.artifacts import Hub
from mpsae.patch_scene.config import Config, load
from mpsae.patch_scene.execution import Task, evaluation_tasks
from mpsae.patch_scene.grid import grid_configs, grid_plan
from mpsae.patch_scene.report import expected_keys
from mpsae.patch_scene.scheduler import run_parallel


def test_selection_identity_counts_and_validation():
    full = load('configs/patch_scene_nyuv2.yaml')
    selected = replace(full, methods=('mpsae',), include_dense=False, mpsae_ot_weight=0.1)
    assert 'methods' not in full.identity()['settings']
    assert 'include_dense' not in full.identity()['settings']
    assert selected.source_identity() == full.source_identity()
    assert selected.plan()['campaign'] != full.plan()['campaign']
    assert selected.plan()['sae_fits'] == selected.plan()['total_evaluations'] == 6
    assert selected.plan()['dense_evaluations'] == 0
    assert len(expected_keys(selected)) == 6
    assert all(k[2] == 'mpsae' for k in expected_keys(selected))
    tasks = evaluation_tasks(selected, Task('prepare', 'siglip2', 'nyuv2'))
    assert len(tasks) == 3 and all(t.method == 'mpsae' for t in tasks)
    assert all(selected.preset('mpsae', e).ot_weight == 0.1 for e in selected.encoder_names)
    assert grid_plan(selected, grid_configs(selected, [0.1, 0.5]))['total_evaluations'] == 12
    for bad in ([], ['mpsae', 'mpsae'], ['unknown'], 'mpsae'):
        with pytest.raises(ValueError, match='methods'):
            Config(methods=bad)
    with pytest.raises(ValueError, match='include_dense'):
        Config(include_dense='false')


def test_mpsae_only_two_workers_export_and_resume(tmp_path):
    config = Config(datasets=('nyuv2',), encoders=('siglip2', 'dinov2'),
                    train_k=(3, 12, 48), evaluation_mode='matched',
                    methods=('mpsae',), include_dense=False, mpsae_ot_weight=0.1)
    hub = Hub(tmp_path)
    campaign = config.plan()['campaign']
    atomic_json(hub.root/f'campaigns/{campaign}/settings/config.json', config.plan())
    def run():
        run_parallel(config, {}, hub, campaign, list(config.encoder_names), ['nyuv2'],
                     ['cpu', 'cpu'], smoke=True, cpu_threads=1)
    run()
    receipt = json.loads((hub.root/f'campaigns/{campaign}/table/results.json').read_text())
    assert receipt['complete'] and receipt['completed'] == receipt['expected'] == 6
    assert {r['method'] for r in receipt['results']} == {'mpsae'}
    assert all(r['train_k'] == r['inference_k'] for r in receipt['results'])
    status = json.loads((hub.root/f'campaigns/{campaign}/scheduler.json').read_text())
    assert status['completed_jobs'] == status['total_jobs'] == 8
    files = [*(hub.root/'fits').glob('*/checkpoint.pt'),
             *(hub.root/'evaluations').glob('*/result.json')]
    assert len(files) == 12
    stamps = {p: p.stat().st_mtime_ns for p in files}
    assert (hub.root/'patch-scene-results.zip').exists()
    run()
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())


def test_kaggle_export_rejects_stale_campaign(tmp_path):
    import importlib.util
    from pathlib import Path
    from zipfile import ZipFile
    spec = importlib.util.spec_from_file_location('kaggle_launcher', Path('scripts/kaggle_segmentation.py'))
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    archive, output = tmp_path/'archive.zip', tmp_path/'download.zip'
    with ZipFile(archive, 'w') as bundle:
        bundle.writestr('config.json', json.dumps({'campaign': 'old'}))
    assert not launcher.copy_current_results(archive, output, 'new')
    assert not output.exists()
    assert launcher.copy_current_results(archive, output, 'old')
    assert archive.read_bytes() == output.read_bytes()
