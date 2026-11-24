from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

from mpsae.benchmark.mass_config import MassConfig
from mpsae.benchmark.retrieval_studies import conditions, plan_for, main

ROOT = Path(__file__).resolve().parents[1]


def settings():
    return yaml.safe_load((ROOT/'configs/retrieval_studies_26.yaml').read_text())


def test_requested_grid_and_no_extra_1024_weight_runs():
    s = settings()
    p = plan_for(s)
    assert p['unique_fits'] == p['unique_evaluations'] == 26
    assert p['study_conditions'] == {'mass': 16, 'optimizer': 10}
    assert p['overlapping_conditions'] == 0
    from mpsae.config import BACKBONES
    for name, c in conditions(s):
        for cell, mass in c.cells():
            preset = c.spec(cell, mass).preset
            assert preset.epochs == 10 and preset.lr == 4e-5
            if name == 'mass':
                assert preset.batch_size == 1024 and mass in (.2, .4, .6, 1)
                assert preset.ot_weight == 2
            else:
                assert cell.backbone == 'ff2048' and preset.batch_size == 4096
                assert mass == 1 and preset.ot_weight in (.1, .5, 1, 2, 2.5)
                baseline = BACKBONES[cell.backbone].preset
                for field in ('batch_size', 'epochs', 'lr', 'adam_eps', 'adam_betas', 'weight_decay'):
                    assert getattr(preset, field) == getattr(baseline, field)


def test_batch_override_identity_and_real_update_counts():
    from mpsae.benchmark.mass_sweep import _synthetic
    from mpsae.train import fit_identity, total_updates
    from mpsae import __version__
    c = MassConfig(backbones=('ff2048',), train_k={'ff2048': [8]}, masses=(1,))
    cell, mass = c.cells()[0]
    bundle = _synthetic(c.spec(cell, mass, smoke=True))
    original = c.spec(cell, mass).preset
    larger = replace(c, batch_size=4096).spec(cell, mass).preset
    class Full:
        class train:
            rows = 1_281_167
    assert total_updates(Full(), original) == (1252, 12520)
    assert total_updates(Full(), larger) == (313, 3130)
    old = fit_identity(cell, bundle, original, total_steps=12520, per_epoch=1252, code_version=__version__)
    explicit = fit_identity(cell, bundle, original, total_steps=12520, per_epoch=1252,
                            code_version=__version__, batch_size_override=1024)
    assert old == explicit
    new = fit_identity(cell, bundle, larger, total_steps=3130, per_epoch=313,
                       code_version=__version__, batch_size_override=4096)
    assert new['schedule']['batch_size'] == 4096 and new['optimizer'] == old['optimizer']
    assert old != new


def test_low_mass_solver_feasibility_and_gradients():
    torch = pytest.importorskip('torch')
    from mpsae import ot
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        costs = torch.stack([torch.rand(32,32,32), torch.full((32,32,32), 2/3)]).requires_grad_()
        for mass in (.2, .4, .6, 1):
            losses, records = ot._batched_matching_gap_from_costs(costs, mass=mass, eta=.2,
                max_iters=400, max_refine_iters=8000, tol=1e-4, require_feasible=True)
            assert losses.min() >= -1e-4
            assert all(r['mass_error'] <= 1e-4 and r['cap_violation'] <= 1e-4 for r in records)
            assert torch.isfinite(torch.autograd.grad(losses.sum(), costs)[0]).all()
    finally:
        torch.set_num_threads(previous)


def test_both_studies_smoke_batch_record_plots_and_resume(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from mpsae.benchmark import mass_sweep
    s = settings()
    s['train_k'] = {'ff2048': [3], 'vit_b16': [3]}
    s['optimizer_weights'] = [.1, 2]
    path = tmp_path/'suite.yaml'
    path.write_text(yaml.safe_dump(s))
    args = ['--config', str(path), '--workspace', str(tmp_path/'out'), '--source-root', str(tmp_path/'source'),
            '--smoke', '--device', 'cpu', '--cpu-threads', '1']
    # Synthetic data too is built just once per backbone, then borrowed by each campaign.
    original = mass_sweep._synthetic
    calls = []
    def build(spec):
        calls.append(spec.name)
        return original(spec)
    monkeypatch.setattr(mass_sweep, '_synthetic', build)
    assert main(args) == 0
    assert sorted(calls) == ['ff2048', 'vit_b16']
    root = tmp_path/'out/SMOKE_ONLY'
    latest = json.loads((root/'studies-latest.json').read_text())
    with ZipFile(latest['archive']) as z:
        assert {'mass_accuracy.pdf', 'mass_accuracy.png', 'weight_accuracy.pdf',
                'weight_accuracy.png', 'mass_table.tex', 'optimizer_table.tex', 'results.csv'} <= set(z.namelist())
        result = json.loads(z.read('results.json'))
        assert result['complete'] and result['completed'] == result['expected'] == 10
        assert not result['plan']['scientific']
        for r in result['results']:
            assert r['training_identity']['schedule']['batch_size'] == (4096 if r['study'] == 'optimizer' else 1024)
    checkpoints = list((root/'fits').glob('*/checkpoint.pt'))
    assert len(checkpoints) == 10
    stamps = {p: p.stat().st_mtime_ns for p in checkpoints}
    assert main(args) == 0
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())
    assert main(args+['--export-only']) == 0


def test_missing_results_export_as_missing(tmp_path):
    from mpsae.benchmark.retrieval_studies import selected_rows
    from mpsae.benchmark.retrieval_studies_report import export
    plan = {**plan_for(settings()), 'scientific': False}
    result = export(tmp_path, plan, selected_rows(tmp_path, plan))
    assert result['completed'] == 0 and result['expected'] == 26 and not result['complete']
    assert '--' in (tmp_path/'studies'/plan['suite_id']/'mass_table.tex').read_text()


def test_standalone_cli_forwards_no_upload_path(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('retrieval_server', ROOT/'scripts/retrieval_server_onefile.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    monkeypatch.setattr(mod, 'command', lambda argv, **kw: calls.append(list(map(str, argv))))
    mod.main(['--skip-setup', '--repo', str(ROOT), '--workspace', str(tmp_path), '--plan'])
    assert '--download-embeddings' not in calls[-1] and '--plan' in calls[-1]
    mod.main(['--skip-setup', '--repo', str(ROOT), '--workspace', str(tmp_path)])
    assert '--download-embeddings' in calls[-1]
    assert '--publish' not in calls[-1]
    mod.main(['--skip-setup', '--repo', str(ROOT), '--workspace', str(tmp_path), '--local-embeddings'])
    assert '--download-embeddings' not in calls[-1]


def test_failed_input_load_leaves_downloadable_status_and_tables(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(['--workspace', str(tmp_path/'out'), '--source-root', str(tmp_path/'missing'),
              '--device', 'cpu'])
    archive = tmp_path/'out/retrieval-studies-results.zip'
    with ZipFile(archive) as z:
        report = json.loads(z.read('results.json'))
        assert not report['complete'] and report['completed'] == 0
        assert report['run_state']['status'] == 'failed'
        assert 'FileNotFoundError' in report['run_state']['error']
        assert 'mass_table.tex' in z.namelist()
