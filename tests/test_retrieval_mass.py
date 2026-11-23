from dataclasses import replace
import json
from pathlib import Path
import pytest

from mpsae.benchmark.mass_config import MassConfig
from mpsae.benchmark.mass_sweep import main, _source_fit
from mpsae.io import atomic_json, canonical_hash
from mpsae.transport import BALANCED_SOLVER

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_threads():
    torch=pytest.importorskip('torch')
    old=torch.get_num_threads()
    yield
    torch.set_num_threads(old)


def test_config_plan_default_and_flexible_selection():
    config=MassConfig.load(ROOT/'configs/retrieval_mass.yaml')
    assert config.plan()['fits']==16 and config.plan()['evaluations']==16
    for cell,mass in config.cells():
        preset=config.spec(cell,mass).preset
        assert (preset.batch_size,preset.epochs,preset.lr)==(1024,10,4e-5)
        assert preset.ot_weight==2 and preset.ot_mass==mass
    expanded=replace(config,backbones=('convnext_tiny','dinov2_vits14'),
                     train_k={'convnext_tiny':[12,48],'dinov2_vits14':[12]},
                     evaluation_mode='sweep',inference_k={'convnext_tiny':[3,12,48], 'dinov2_vits14':[3,12]})
    assert expanded.plan()['fits']==12 and expanded.plan()['evaluations']==32
    assert expanded.plan()['sweep_id']!=config.plan()['sweep_id']
    for kwargs in ({'masses':(True,)},{'masses':(0.,)},{'masses':(1.01,)},
                   {'masses':(.9,.9)},{'backbones':('unknown',)}, {'lr':float('nan')},
                   {'train_k':{'ff2048':[4096],'vit_b16':[12]}},
                   {'inference_k':{'ff2048':[8]}}):
        with pytest.raises(ValueError):
            replace(config,**kwargs)


def test_high_k_recipe_is_visible_and_not_silently_overridden():
    config=MassConfig(backbones=('ff2048',),train_k={'ff2048':[256]})
    assert all(row['preset']['ot_weight']==.5 for row in config.plan()['recipes'])
    with pytest.raises(ValueError,match='K>64'):
        replace(config,ot_weight=2.)


def test_legacy_reuse_rejects_different_mass_and_broken_signature(tmp_path):
    from mpsae.config import Cell
    cell=Cell('ff2048','mpsae',8,1227)
    identity={'objective':{'ot':{'mass':.9}},'schedule':{'total_steps':10},'data':{}}
    folder=tmp_path/'runs'/cell.slug
    atomic_json(folder/'summary.json',dict(identity=identity,signature=canonical_hash(identity),complete=True,steps=10))
    atomic_json(folder/'config.json',dict(identity=identity,signature=canonical_hash(identity)))
    (folder/'checkpoint.pt').write_bytes(b'actual contents are checked by load_fitted')
    assert _source_fit(cell,identity,tmp_path)[0]==folder
    changed={**identity,'objective':{'ot':{'mass':1}}}
    assert _source_fit(cell,changed,tmp_path) is None
    atomic_json(folder/'config.json',dict(identity=identity,signature='corrupt'))
    with pytest.raises(RuntimeError,match='Inconsistent'):
        _source_fit(cell,identity,tmp_path)


def test_smoke_train_evaluate_resume_and_export_history(tmp_path):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    import yaml
    from zipfile import ZipFile
    config=MassConfig(backbones=('vit_b16',),train_k={'vit_b16':[3]},
                      evaluation_mode='sweep',inference_k={'vit_b16':[3,6]})
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config.plan()['config']))
    args=['--config',str(path),'--workspace',str(tmp_path/'ablation'),
          '--source-root',str(tmp_path/'source'),'--smoke','--device','cpu','--cpu-threads','1']
    assert main(args)==0
    root=tmp_path/'ablation/SMOKE_ONLY'
    checkpoints=list((root/'fits').glob('*/checkpoint.pt'))
    assert len(checkpoints)==2
    stamps={p:p.stat().st_mtime_ns for p in checkpoints}
    result=json.loads((root/'sweeps'/config.plan()['sweep_id']/'results.json').read_text())
    assert result['complete'] and result['completed']==4
    assert len({r['checkpoint_sha256'] for r in result['results']})==2
    assert {r['solver'] for r in result['results']}=={'partial-greenkhorn-v1',BALANCED_SOLVER}
    assert all(0<=r['gallery_mean_nnz']<=r['inference_k'] for r in result['results'])
    assert main(args)==0
    assert stamps=={p:p.stat().st_mtime_ns for p in checkpoints}
    assert main(args+['--masses','1'])==0
    assert len(list((root/'fits').glob('*/checkpoint.pt')))==2
    assert len(json.loads((root/'experiment-index.json').read_text()))==6
    assert main(args+['--export-only'])==0
    archive=root/'sweeps'/config.plan()['sweep_id']/'retrieval-mass-results.zip'
    with ZipFile(archive) as z:
        assert {'table.tex','results.csv','summary.csv','runtime.json','plan.json'}<=set(z.namelist())
        assert not any(name.endswith('.pt') for name in z.namelist())


def test_failed_load_is_recorded_and_exportable(tmp_path):
    pytest.importorskip('torch')
    args=['--config',str(ROOT/'configs/retrieval_mass.yaml'),'--workspace',str(tmp_path/'out'),
          '--source-root',str(tmp_path/'missing'),'--backbones','vit_b16','--train-k','vit_b16=12',
          '--masses','1','--device','cpu','--continue-on-error']
    assert main(args)==1
    latest=json.loads((tmp_path/'out/latest.json').read_text())
    result=json.loads((tmp_path/'out/sweeps'/latest['sweep_id']/'results.json').read_text())
    assert result['completed']==0 and result['results'][0]['status']=='failed'
    assert main(args+['--export-only'])==0


def test_resume_after_evaluation_failure_does_not_retrain(tmp_path,monkeypatch):
    import yaml
    from mpsae import runs
    config=MassConfig(backbones=('vit_b16',),train_k={'vit_b16':[3]},masses=(1.,),
                      evaluation_mode='sweep',inference_k={'vit_b16':[3,6]})
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config.plan()['config']))
    args=['--config',str(path),'--workspace',str(tmp_path/'ablation'),
          '--source-root',str(tmp_path/'source'),'--smoke','--device','cpu','--cpu-threads','1',
          '--continue-on-error']
    original=runs.evaluate_at
    def failure(*a,**kw):
        if kw['k']==6:
            raise RuntimeError('simulated evaluator interruption')
        return original(*a,**kw)
    monkeypatch.setattr(runs,'evaluate_at',failure)
    assert main(args)==1
    root=tmp_path/'ablation/SMOKE_ONLY'
    checkpoint=next((root/'fits').glob('*/checkpoint.pt'))
    stamp=checkpoint.stat().st_mtime_ns
    result=json.loads((root/'sweeps'/config.plan()['sweep_id']/'results.json').read_text())
    assert result['completed']==1
    assert {r['status'] for r in result['results']}=={'failed','complete'}
    monkeypatch.setattr(runs,'evaluate_at',original)
    assert main(args)==0
    assert checkpoint.stat().st_mtime_ns==stamp
