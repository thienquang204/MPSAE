from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import pytest

from mpsae.io import atomic_json
from mpsae.patch_scene.config import load
from mpsae.patch_scene.grid import grid_configs, write_grid_tables
from mpsae.patch_scene.mass_sweep import rebuild_index
from mpsae.patch_scene.report import expected_keys
from mpsae.patch_scene.weight_sweep import weight_plan, export_grid, main
from mpsae.transport import BALANCED_SOLVER


def selected(dataset='voc'):
    return replace(load(f'configs/patch_scene_{dataset}.yaml'), encoders=('siglip2',),
                   mpsae_ot_mass=1., evaluation_mode='matched', inference_k=None)


@pytest.mark.parametrize('dataset', ['voc', 'nyuv2', 'cityscapes'])
def test_balanced_grid_preserves_protocol_and_shares_baselines(dataset):
    config = selected(dataset)
    plan = weight_plan(config, [.5, 1., 2.])
    assert plan['solver'] == BALANCED_SOLVER
    assert plan['sae_fits'] == 15 and plan['total_evaluations'] == 16
    assert plan['cross_k_evaluations'] == 0
    assert weight_plan(config, [2., 1., .5]) == plan
    partial = replace(config, mpsae_ot_mass=.9)
    assert weight_plan(partial, [.5, 1., 2.])['sweep_id'] != plan['sweep_id']
    for c in grid_configs(config, [.5, 1., 2.]):
        assert c.source_identity() == partial.source_identity()
        for method in ('csr', 'csrv2'):
            assert c.preset(method, 'siglip2') == partial.preset(method, 'siglip2')
        assert c.preset('mpsae', 'siglip2').ot_mass == 1
        assert c.epochs == config.epochs and c.lr == config.lr


def fake_results(workspace, config):
    rows = []
    for dataset, encoder, method, train, infer in expected_keys(config):
        rows.append(dict(dataset=dataset, encoder=encoder, method=method, train_k=train,
                         inference_k=infer, complete=True, seed=config.seed,
                         miou=.5, pixel_accuracy=.7, mean_nnz=infer, gallery_mean_nnz=infer,
                         identity={'method': method, 'k': train,
                                   'weight': config.mpsae_ot_weight if method == 'mpsae' else None}))
    folder = workspace/'campaigns'/config.plan()['campaign']
    atomic_json(folder/'settings/config.json', config.plan())
    atomic_json(folder/'cells/voc-siglip2/rows.json', {'results': rows})
    return rows


def test_export_recovers_all_weights_not_last_zip_and_validates_mass(tmp_path):
    config = selected()
    grid = grid_configs(config, [.5, 1., 2.])
    (tmp_path/'patch-scene-results.zip').write_bytes(b'unrelated stale campaign')
    rows = fake_results(tmp_path, grid[0])
    partial, archive = export_grid(tmp_path, config, [.5, 1., 2.])
    assert partial['completed'] == 10 and partial['expected'] == 16
    assert not partial['complete']
    for c in grid[1:]:
        fake_results(tmp_path, c)
    result, archive = export_grid(tmp_path, config, [.5, 1., 2.])
    assert result['complete'] and result['completed'] == 16
    assert sum(r['method'] == 'csr' for r in result['results']) == 3
    assert sum(r['method'] == 'mpsae' for r in result['results']) == 9
    for row in result['results']:
        if row['method'] == 'mpsae':
            assert row['mmpot_mass'] == 1 and row['solver'] == BALANCED_SOLVER
        else:
            assert row['mmpot_weight'] is None and row['solver'] is None
    with ZipFile(archive) as bundle:
        assert {'plan.json', 'grid_results.csv', 'grid_results.tex', 'grid_comparisons.csv',
                'experiment-index.csv'} <= set(bundle.namelist())
        assert len([n for n in bundle.namelist() if n.endswith('settings/config.json')]) == 3
        assert not any(n.endswith('.pt') for n in bundle.namelist())
        assert 'mpsae_lambda_0.5' in bundle.read('grid_comparisons.csv').decode()
        assert 'balanced MMOT' in bundle.read('grid_results.tex').decode()
    with pytest.raises(ValueError, match='settings differ'):
        write_grid_tables([(replace(grid[0], mpsae_ot_mass=.9), 'voc', {'results': rows})],
                          {'voc': config}, [.5, 1., 2.], tmp_path/'bad')


def test_plan_no_download_and_mpsae_only_selection(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('plan ran subprocess'))
    main(['--workspace', str(tmp_path/'unused'), '--encoders', 'dinov2', '--methods', 'mpsae',
          '--train-k', '12', '--mass', '1', '--weights', '.1', '.5', '--no-dense', '--plan'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['total_evaluations'] == 2 and plan['sae_fits'] == 2
    assert plan['settings']['epochs'] == 10 and plan['solver'] == BALANCED_SOLVER
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('args', [
    ['--mass', '1.01'], ['--mass', '0'], ['--weights', '.5', '.5'],
    ['--weights', 'nan'], ['--weights', '0'], ['--train-k', '192'],
    ['--methods', 'csr'], ['--cpu-threads', '0'],
])
def test_invalid_grid_rejected_before_run(args):
    with pytest.raises(SystemExit):
        main(args+['--plan'])


def test_kaggle_rejects_ambiguous_controls_before_install():
    spec = importlib.util.spec_from_file_location('kaggle_weights', Path('scripts/kaggle_segmentation.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for kwargs in ({'mmpot_weights': [.5], 'mmpot_masses': [1]},
                   {'mmpot_weights': [.5], 'mmpot_weight': 2},
                   {'mmpot_mass': 1, 'mmpot_masses': [.9, 1]},
                   {'mmpot_weights': [.5], 'methods': ['csr']}):
        with pytest.raises(ValueError):
            module.main(**kwargs)


def test_balanced_grid_two_workers_resume_and_partial_checkpoint_isolation(tmp_path):
    pytest.importorskip('torch')
    workspace = tmp_path/'work'
    args = ['--workspace', str(workspace), '--encoders', 'siglip2', '--train-k', '3',
            '--weights', '.5', '1', '--devices', 'cpu', 'cpu', '--smoke']
    main(args+['--mass', '1'])
    root = workspace/'SMOKE_ONLY'
    config = replace(selected(), train_k=(3,))
    result = json.loads((root/'ablations'/weight_plan(config, [.5, 1.])['sweep_id']/'grid_results.json').read_text())
    assert result['complete'] and result['completed'] == result['expected'] == 5
    checkpoints = list((root/'fits').glob('*/checkpoint.pt'))
    assert len(checkpoints) == 4  # two baselines + two MPSAE weights
    stamps = {p: p.stat().st_mtime_ns for p in checkpoints}
    main(args+['--mass', '1'])
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())
    main(args+['--mass', '.9'])
    assert len(list((root/'fits').glob('*/checkpoint.pt'))) == 6
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())
    records, errors = rebuild_index(root)
    assert not errors
    assert {r['s'] for r in records if r['method'] == 'mpsae'} == {.9, 1.}
    for c in grid_configs(config, [.5, 1.]):
        metrics = next(r for r in records if r['method'] == 'mpsae' and r['campaign_id'] == c.plan()['campaign'])['diagnostics']
        assert metrics['ot_marginal_l1'] <= 1e-4
    main(args+['--mass', '1', '--export-only'])
