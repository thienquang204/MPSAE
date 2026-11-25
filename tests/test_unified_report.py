import csv
from dataclasses import asdict
from zipfile import ZipFile

import pytest

from mpsae.benchmark.matched_config import MatchedConfig
from mpsae.benchmark.lr_ablation import study_plan,condition
from mpsae.benchmark.lr_report import export,PERFORMANCE
from mpsae.benchmark.unified_report import export_unified
from mpsae.io import atomic_json,read_json


def rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def test_all_batches_lr_seeds_cross_and_zip_deduplication(tmp_path):
    config=MatchedConfig(k_values=(8,32))
    original=config.plan();source=tmp_path/'studies'/original['study_id']
    atomic_json(source/'plan.json',original)
    new,plan=study_plan(config,original['study_id'],[8,32],[2,8,32,128])
    out=tmp_path/'lr_ablation'/plan['study_id'];out.mkdir(parents=True)
    atomic_json(out/'session.json',{'runtime':{'device':'current'},'evaluation':{}})
    previous=tmp_path/'crossk/old'
    atomic_json(previous/'plan.json',{'source_study':original['study_id'],'source_config':asdict(config)})
    records=[];values={}
    for cell,b in config.cells():
        score=(40 if b==1024 else 60)+(2 if cell.seed==2026 else 0)
        sha=f'{cell.method}-{b}-{cell.k}-{cell.seed}'
        r={'method':cell.method,'batch_size':b,'seed':cell.seed,'k':cell.k,'status':'complete',
           'checkpoint_sha256':sha,'runtime':{'device':'old' if b==1024 else 'current'},
           'identity':{'config':asdict(config),'implementation':{}},
           'training_identity':{'optimizer':{'lr':4e-5},'data':{'id':'features'}},
           'retrieval':{'top1':score,'timing':{'microseconds_per_query':25}},'query_nnz':cell.k,'nnz':cell.k}
        if cell.k==config.probe_k:r['probe']={'fit':{'top1':score+10}}
        atomic_json(source/'cells'/f'b{b}-{cell.method}-k{cell.k}-s{cell.seed}'/'result.json',r)
        values[(cell.method,b,cell.seed,cell.k)]=r
        if b==4096:
            record={**r,'condition':condition(cell.method),'lr':4e-5,'train_k':cell.k,
                    'inference_k':cell.k,'kind':'matched','performance_comparable':True}
            atomic_json(out/'matched'/f'{record["condition"]}-k{cell.k}-s{cell.seed}.json',record)
            records.append(record)
        # All cross conditions also emit a diagonal copy, which must not erase probes.
        for ki in (cell.k,128):
            cross={**r,'train_k':cell.k,'inference_k':ki,'retrieval':{'top1':score-5},
                   'identity':{'data':{'id':'features'},'session':{'runtime':{'device':'current'},'implementation':{}}}}
            cross.pop('probe',None)
            atomic_json(previous/'points'/f'{sha}-infer{ki}'/'result.json',cross)
    for cell,b in new.cells():
        r=values[(cell.method,b,cell.seed,cell.k)]
        for ki in (cell.k,128):
            record={**r,'condition':condition('mpsae',True),'lr':8e-5,'train_k':cell.k,
                    'inference_k':ki,'kind':'matched' if ki==cell.k else 'cross',
                    'retrieval':{'top1':r['retrieval']['top1']+3},'performance_comparable':True,
                    'checkpoint_sha256':'new-'+r['checkpoint_sha256']}
            if ki!=cell.k:record.pop('probe',None)
            key=(f'{record["condition"]}-k{cell.k}-s{cell.seed}' if ki==cell.k else
                 f'{record["condition"]}-train{cell.k}-infer{ki}-s{cell.seed}')
            atomic_json(out/record['kind']/f'{key}.json',record);records.append(record)
    # Same old cross point is also present in the LR result: counted once.
    old=read_json(next((previous/'points').glob('*4096*infer128/result.json')))
    r={**old,'condition':condition(old['method']),'lr':4e-5,'kind':'cross','performance_comparable':True}
    atomic_json(out/'cross'/f'{r["condition"]}-train{r["train_k"]}-infer128-s{r["seed"]}.json',r)
    records.append(r)
    original_bytes={p:p.read_bytes() for directory in (source,previous) for p in directory.rglob('*') if p.is_file()}
    result=export(out,plan,figures=False)
    status=result['unified']
    assert status['matched_1nn_scores']==28 and status['expected_matched_1nn']==28
    assert status['unique_1nn_scores']==56 and status['cross_1nn_scores']==28
    assert status['probe_scores']==14
    table=rows(out/'reports/unified_per_seed.csv')
    assert len({tuple(r[k] for k in ('backbone','method','batch_size','lr','seed','train_k','inference_k')) for r in table})==56
    summary=rows(out/'reports/unified_summary.csv')
    for batch,expected in [('1024',41),('4096',61)]:
        r=next(r for r in summary if (r['method'],r['batch_size'],r['lr'],r['train_k'],r['inference_k'])==('csr',batch,'4e-05','8','8'))
        assert float(r['top1_mean'])==expected and float(r['top1_std'])==pytest.approx(2**.5)
        assert int(r['top1_n'])==2
    r=next(r for r in table if r['batch_size']=='1024' and r['kind']=='matched')
    assert r['retrieval_us']=='' and r['top1']!=''
    assert len(rows(out/'reports/unified_matched_table.csv'))==7
    with ZipFile(out/'lr-ablation-results.zip') as archive:
        assert 'reports/unified_matched.tex' in archive.namelist()
        assert 'reports/unified_per_seed.csv' in archive.namelist()
    assert original_bytes=={p:p.read_bytes() for directory in (source,previous) for p in directory.rglob('*') if p.is_file()}
    # A foreign study or mismatched checkpoint must not contaminate the union.
    bad={**old,'checkpoint_sha256':'unrelated','inference_k':256}
    atomic_json(previous/'points/bad/result.json',bad)
    status=export_unified(out,plan,records,PERFORMANCE)
    assert status['unique_1nn_scores']==56 and len(status['skipped'])==1


def test_partial_and_missing_source_are_explicit(tmp_path):
    config=MatchedConfig();_,plan=study_plan(config,config.plan()['study_id'],[8,32],[2,8,32,128])
    out=tmp_path/'lr_ablation'/plan['study_id'];(out/'reports').mkdir(parents=True)
    status=export_unified(out,plan,[],PERFORMANCE)
    assert not status['original_source_available'] and status['expected_matched_1nn'] is None
    assert status['unique_1nn_scores']==0
    atomic_json(tmp_path/'studies'/plan['source_study']/'plan.json',config.plan())
    status=export_unified(out,plan,[],PERFORMANCE)
    assert status['expected_matched_1nn']==84 and status['matched_1nn_scores']==0
    assert len(rows(out/'reports/unified_matched_table.csv'))==7
