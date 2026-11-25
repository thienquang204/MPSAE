import csv
from dataclasses import asdict,replace
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
import yaml

from mpsae.benchmark.matched_config import MatchedConfig
from mpsae.benchmark.matched_study import main as matched_main
from mpsae.benchmark.matched_crossk import main as cross_main
from mpsae.benchmark.lr_ablation import main,study_plan
from mpsae.benchmark.lr_report import paired,summaries,export
from mpsae.io import read_json


def test_plan_and_only_lr_changes():
    base=MatchedConfig()
    new,plan=study_plan(base,'source',[8,32],[2,8,32,128])
    assert (plan['new_fits'],plan['new_matched_1nn'],plan['new_cross_1nn'],plan['new_probes'])==(12,12,12,2)
    assert plan['new_accuracy_evaluations']==26
    assert plan['baseline_cross_max_missing']==36
    assert plan['expected_unique_1nn']==96 and plan['expected_probe_scores']==8
    for cell,batch in new.cells():
        original=base.spec(cell,batch).preset
        changed=new.spec(cell,batch).preset
        assert replace(changed,lr=original.lr)==original
        assert changed.batch_size==4096 and changed.lr==8e-5 and changed.epochs==10
    with pytest.raises(ValueError):study_plan(replace(base,lr=1e-4),'source',[8],[8])


def test_paired_seed_differences():
    def row(name,seed,value):
        return dict(condition=name,seed=seed,train_k=8,inference_k=2,kind='cross',top1=value)
    rows=[row('mpsae_lr4e-5',1227,50),row('mpsae_lr4e-5',2026,60),
          row('mpsae_lr8e-5',1227,54),row('mpsae_lr8e-5',2026,62)]
    pairs,summary=paired(rows)
    assert [r['delta_pp'] for r in pairs]==[4,2]
    assert summary[0]['mean_delta_pp']==3
    assert summary[0]['std_delta_pp']==pytest.approx(2**.5)


def test_incompatible_runtime_masks_performance_not_accuracy(tmp_path):
    from mpsae.io import atomic_json
    _,plan=study_plan(MatchedConfig(),'source',[8,32],[2,8,32,128])
    atomic_json(tmp_path/'matched/cell.json',dict(condition='mpsae_lr4e-5',method='mpsae',lr=4e-5,
        batch_size=4096,seed=1227,train_k=8,inference_k=8,kind='matched',status='complete',
        performance_comparable=False,retrieval={'top1':70,'timing':{'microseconds_per_query':12}},nnz=8))
    export(tmp_path,plan,figures=False)
    with (tmp_path/'reports/accuracy_per_seed.csv').open() as f:row=next(csv.DictReader(f))
    assert float(row['top1'])==70 and row['retrieval_us']==''


def test_end_to_end_reuse_new_training_only_resume_and_zip(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    cfg=MatchedConfig(seeds=(1227,),k_values=(2,8),probe_k=8,batch_sizes=(8,),probe_epochs=1,
                      query_batch=8,gallery_chunk=16,build_entry_cap=128,encode_batch_size=32)
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(asdict(cfg)))
    work=tmp_path/'work'
    common=['--workspace',str(work),'--device','cpu','--smoke','--cpu-threads','1']
    assert matched_main(['--config',str(config),*common])==0
    cross_main([*common,'--train-k','2','8','--inference-k','2','8','16'])
    root=work/'SMOKE_ONLY'
    original={p:p.read_bytes() for sub in ('fits','studies','crossk') for p in (root/sub).rglob('*') if p.is_file()}
    latest=(root/'latest.json').read_bytes()
    import mpsae.train as train
    import mpsae.benchmark.matched_evaluate as evaluation
    fit=train.fit;retrieve=evaluation.retrieval;fits=[];evaluations=[]
    def controlled_fit(cell,data,path,settings,**kw):
        assert cell.method=='mpsae'
        assert kw['spec'].preset.lr==8e-5 and settings.batch_size_override==8
        assert not (path/'checkpoint.pt').exists(), 'New LR must start from scratch'
        fits.append(path)
        return fit(cell,data,path,settings,**kw)
    def counted_retrieval(*a,**kw):
        evaluations.append(1)
        return retrieve(*a,**kw)
    monkeypatch.setattr(train,'fit',controlled_fit)
    monkeypatch.setattr(evaluation,'retrieval',counted_retrieval)
    args=[*common,'--train-k','2','8','--inference-k','2','8','16','--log-every','1']
    main(args)
    assert len(fits)==2
    assert len(evaluations)==6, 'Two new matched and four new cross-K evaluations; baseline cross-K reused'
    directory=Path(read_json(root/'lr_ablation/latest.json')['directory'])
    result=read_json(directory/'results.json')
    assert result['complete_1nn']==result['expected_1nn']==24
    assert result['probe_scores']==result['expected_probe_scores']==4
    rows=[read_json(p) for p in (directory/'matched').glob('*.json')]
    assert len(rows)==8
    assert len({r['checkpoint_sha256'] for r in rows})==8
    old={r['train_k']:r for r in rows if r['condition']=='mpsae_lr4e-5'}
    for r in rows:
        if r['condition']!='mpsae_lr8e-5':continue
        before=old[r['train_k']]['training_identity'];after=r['training_identity']
        assert before['objective']==after['objective'] and before['schedule']==after['schedule']
        assert before['optimizer']['lr']==4e-5 and after['optimizer']['lr']==8e-5
    assert len(read_json(next((directory/'losses').glob('*.json')))['samples'])>0
    assert (directory/'reports/loss_total.pdf').exists()
    assert (directory/'reports/crossk_accuracy.pdf').exists()
    with (directory/'reports/accuracy_per_seed.csv').open() as f:
        table=list(csv.DictReader(f))
    assert len(table)==24
    assert len({(r['condition'],r['seed'],r['train_k'],r['inference_k']) for r in table})==24
    with ZipFile(directory/'lr-ablation-results.zip') as z:
        assert 'reports/linear_probe.csv' in z.namelist()
        assert not any(n.endswith('.pt') or n.startswith('training/') for n in z.namelist())
    def forbidden(*a,**kw):raise AssertionError('Resume must reuse completed work')
    monkeypatch.setattr(train,'fit',forbidden);monkeypatch.setattr(evaluation,'retrieval',forbidden)
    main(args);main(args+['--report-only'])
    assert original=={p:p.read_bytes() for sub in ('fits','studies','crossk') for p in (root/sub).rglob('*') if p.is_file()}
    assert (root/'latest.json').read_bytes()==latest
    assert not list((directory/'cache/transient-codes').glob('*'))


def test_missing_baseline_stops_before_training(tmp_path,monkeypatch):
    from mpsae.io import atomic_json
    cfg=MatchedConfig()
    plan=cfg.plan();plan['scientific']=True
    atomic_json(tmp_path/'latest.json',{'study_id':plan['study_id']})
    atomic_json(tmp_path/'studies'/plan['study_id']/'plan.json',plan)
    with pytest.raises(SystemExit):main(['--workspace',str(tmp_path)])
    assert not list(tmp_path.rglob('checkpoint.pt'))
