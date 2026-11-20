from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from mpsae.io import atomic_json
from mpsae.patch_scene.config import load
from mpsae.patch_scene.mass_sweep import (DEFAULT_MASSES, mass_configs, sweep_plan,
    export_sweep, rebuild_index, main, campaign_records)

ROOT = Path(__file__).resolve().parents[1]


def selected():
    return replace(load(ROOT/'configs/patch_scene_voc.yaml'), methods=('mpsae',),
                   include_dense=False, encoders=('siglip2',))


def test_plan_preserves_cache_identities_and_baseline_presets():
    base = selected()
    grid = mass_configs(base)
    plan = sweep_plan(grid)
    assert plan['sae_fits'] == 18 and plan['total_evaluations'] == 54
    assert plan['matched_evaluations'] == 18 and plan['cross_k_evaluations'] == 36
    assert len(set(plan['campaigns'])) == 6
    assert grid[4].identity() == base.identity()
    assert 'mpsae_ot_mass' not in base.identity()['settings']
    for c in grid:
        assert c.source_identity() == base.source_identity()
        assert c.preset('csr', 'siglip2') == base.preset('csr', 'siglip2')
        assert c.preset('mpsae', 'siglip2').ot_mass == c.mpsae_ot_mass
        assert c.preset('mpsae', 'siglip2').ot_weight == 2
        assert c.epochs == 10 and c.evaluation_mode == 'sweep'
    with pytest.raises(ValueError):
        mass_configs(base, [0.9, 0.9])
    for value in (0, 1.01, -.1, True, float('nan'), float('inf'), '0.9'):
        with pytest.raises(ValueError, match='mass'):
            replace(base, mpsae_ot_mass=value)
    assert sweep_plan(mass_configs(base, list(reversed(DEFAULT_MASSES)))) == plan


def fake_campaign(workspace, config, *, complete=True):
    campaign = config.plan()['campaign']
    root = workspace/'campaigns'/campaign
    atomic_json(root/'settings/config.json', config.plan())
    rows = []
    if complete:
        for train_k, infer_k in config.evaluation_pairs:
            rows.append({'dataset': 'voc', 'encoder': 'siglip2', 'method': 'mpsae',
                         'train_k': train_k, 'inference_k': infer_k, 'seed': config.seed,
                         'complete': True, 'miou': .4+config.mpsae_ot_mass/10,
                         'pixel_accuracy': .7, 'mean_nnz': infer_k,
                         'identity': {'fit': f'fit-{config.mpsae_ot_mass}-{train_k}'}})
        atomic_json(root/'cells/voc-siglip2/rows.json', {'results': rows})
    return campaign


def test_export_preserves_history_and_separates_weights(tmp_path):
    base = selected()
    historical = replace(base, mpsae_ot_weight=.1)
    old_id = fake_campaign(tmp_path, historical)
    grid = mass_configs(base, [.5, .95])
    fake_campaign(tmp_path, grid[0])
    second = fake_campaign(tmp_path, grid[1], complete=False)
    result, archive = export_sweep(tmp_path, grid)
    assert not result['complete'] and result['completed'] == 9 and result['expected'] == 18
    records, errors = rebuild_index(tmp_path)
    assert not errors and len(records) == 27
    assert any(r['campaign_id'] == old_id and r['lambda'] == .1 for r in records)
    assert all(r['lambda'] == 2 for r in result['results'])
    assert {r['status'] for r in records if r['campaign_id'] == second} == {'pending'}
    atomic_json(tmp_path/f'campaigns/{second}/run-state.json', {'status': 'failed', 'error': 'interrupted download'})
    export_sweep(tmp_path, grid)
    assert all(r['status'] == 'failed' for r in campaign_records(tmp_path, grid[1], second))
    fake_campaign(tmp_path, grid[1])
    result, archive = export_sweep(tmp_path, grid)
    assert result['complete'] and result['completed'] == 18
    with ZipFile(archive) as bundle:
        assert 'matched_comparisons.tex' in bundle.namelist()
        assert 'experiment-index.csv' in bundle.namelist()
        assert 'experiment-index.json' in bundle.namelist()
        assert 's_0.5' in bundle.read('matched_comparisons.csv').decode()
        assert not any(n.endswith('.pt') for n in bundle.namelist())
    # Rebuilding a smaller sweep never deletes history in the global register.
    export_sweep(tmp_path, grid[:1])
    assert len(rebuild_index(tmp_path)[0]) == 27


def test_mass_solver_production_group_cpu():
    torch = pytest.importorskip('torch')
    from mpsae import ot
    torch.set_num_threads(1)
    torch.manual_seed(1227)
    pre = torch.randn(32, 128)
    views = []
    for k in (3, 6, 12):
        values, indices = pre.topk(k, dim=-1)
        views.append(torch.zeros_like(pre).scatter(-1, indices, values).relu())
    cases = torch.stack([ot.angular_cost(*views), torch.full((32,32,32), 2/3), torch.rand(32,32,32)])
    for mass in DEFAULT_MASSES:
        costs = cases.detach().requires_grad_()
        losses, stats = ot._batched_matching_gap_from_costs(costs, mass=mass, eta=.2,
            max_iters=400, max_refine_iters=8000, tol=1e-4, require_feasible=True)
        grad, = torch.autograd.grad(losses.sum(), costs)
        assert torch.isfinite(grad).all() and losses.min() >= -1e-4
        assert max(s['mass_error'] for s in stats) <= 1e-4
        assert max(s['cap_violation'] for s in stats) <= 1e-4
        for i in range(len(cases)):
            scalar = cases[i].detach().requires_grad_()
            loss, _, delta = ot.matching_gap_from_cost(scalar, mass=mass, eta=.2,
                max_iters=400, max_refine_iters=8000, tol=1e-4)
            torch.testing.assert_close(loss, losses[i], atol=5e-5, rtol=1e-5)
            scalar_grad, = torch.autograd.grad(loss, scalar)
            torch.testing.assert_close(scalar_grad, delta, atol=1e-7, rtol=1e-6)


def test_balanced_mass_has_separate_fit_identity_and_solver_metadata(tmp_path):
    from mpsae import methods
    from mpsae.transport import BALANCED_SOLVER
    base=selected()
    partial,balanced=mass_configs(base,[.9,1.])
    assert partial.identity()==base.identity()
    assert balanced.source_identity()==base.source_identity()
    assert partial.identity()!=balanced.identity()
    assert methods.describe('mpsae',3,balanced.preset('mpsae','siglip2'))['ot']['solver']==BALANCED_SOLVER
    campaign=fake_campaign(tmp_path,balanced)
    assert all(r['solver']==BALANCED_SOLVER for r in campaign_records(tmp_path,balanced,campaign))


def test_smoke_sweep_two_workers_all_masses_and_resume(tmp_path):
    pytest.importorskip('torch')
    import yaml
    # Small K grid keeps this integration check short; separate plan test covers all 54 points.
    config = replace(selected(), train_k=(3,), inference_k=(3,12))
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config.identity()['settings']))
    args = ['--config', str(path), '--workspace', str(tmp_path/'work'), '--smoke',
            '--devices', 'cpu', 'cpu']
    main(args)
    root = tmp_path/'work/SMOKE_ONLY'
    plan = sweep_plan(mass_configs(config))
    result = json.loads((root/f'ablations/{plan["sweep_id"]}/results.json').read_text())
    assert result['complete'] and result['completed'] == 12
    checkpoints = list((root/'fits').glob('*/checkpoint.pt'))
    assert len(checkpoints) == 6
    assert len({r['checkpoint_sha256'] for r in result['results']}) == 6
    stamps = {p: p.stat().st_mtime_ns for p in checkpoints}
    main(args+['--masses', '0.9'])
    assert all(p.stat().st_mtime_ns == stamp for p,stamp in stamps.items())
    assert len(rebuild_index(root)[0]) == 12
    main(args+['--export-only'])
    main(args+['--masses','1.0'])
    assert len(list((root/'fits').glob('*/checkpoint.pt')))==7
    assert len(rebuild_index(root)[0])==14
