"""Paper receipts are evidence; tests use temporary synthetic metadata only."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/cvpr2027'))
from common import digest,read_json,write_json
from registry import build_plan,jobs_for,table_conditions
from ingest import index_source,documents,rebuild_index
from reporting import report,verify_reference,bundle


@pytest.fixture(scope='module')
def plan():
    return build_plan()


def find(plan, **selector):
    return next(r for r in plan['conditions'] if all(r.get(k)==v for k,v in selector.items()))


def main_receipt(folder,row,value=61.25):
    identity={'backbone':row['encoder'],'method':row['method'],'target_k':row['train_k'],'seed':row['seed'],
              'objective':{'hidden_dim':8192},'optimizer':{'lr':row['lr']},
              'schedule':{'batch_size':row['batch_size'],'epochs':row['epochs'],'total_steps':3130},
              'data':{'train':{'sha256':'train-fixture'},'validation':{'sha256':'val-fixture'}}}
    if row['method']=='mpsae':
        identity['objective']['ot']={'mass':row['mass'],'weight':row['loss_weight']}
    signature=digest(identity)
    write_json(folder/'config.json',{'identity':identity,'signature':signature})
    write_json(folder/'summary.json',{'complete':True,'identity':identity,'signature':signature,'steps':3130})
    point={'signature':signature,'backbone':row['encoder'],'method':row['method'],'seed':row['seed'],
           'k':row['inference_k'],'top1':value,'nnz_mean':row['inference_k'],'empty_queries':0}
    write_json(folder/'eval_e2.json',dict(point,anchor_k=row['train_k'],points=[point]))
    return point


def test_catalog_counts_and_table_reuse(plan):
    assert (plan['unique_sae_fits'],plan['probe_fits'],plan['unique_evaluations'])==(238,84,671)
    assert len(table_conditions(plan,'1')[1])==222
    assert len(table_conditions(plan,'2')[1])==130
    assert len(table_conditions(plan,'5')[1])==6
    assert len(table_conditions(plan,'12')[1])==36
    assert len(table_conditions(plan,'13')[1])==48
    assert {r['condition_id'] for r in table_conditions(plan,'10')[1]} <= {r['condition_id'] for r in table_conditions(plan,'2')[1]}
    assert all(r['dataset'] not in ('ade20k','cityscapes') for r in plan['conditions'])
    assert all(r['method'] in ('dense','csr','csrv2','mpsae') for r in plan['conditions'])


def test_paper_nyu_override_and_cross_budget_weights():
    assert all(j['config']['mpsae_ot_weight']==2 and j['config']['siglip2_mpsae_ot_weight'] is None for j in jobs_for('nyuv2_main'))
    assert all(j['config']['evaluation_mode']=='sweep' and j['config']['inference_k']==[3,12,48] for j in jobs_for('voc_weight'))


def test_source_lock():
    from runners import check_source_lock
    check_source_lock()


def test_torch_free_planning_and_reporting(tmp_path):
    script="""
import builtins,sys
real=builtins.__import__
def guarded(name,*a,**kw):
    if name.split('.')[0] in ('torch','transformers','datasets'):
        raise RuntimeError('Unexpected heavyweight import: '+name)
    return real(name,*a,**kw)
builtins.__import__=guarded
sys.path.insert(0,sys.argv[1])
from registry import build_plan
from reporting import report
p=build_plan()
report(sys.argv[2],p,tables=['5'],plots=False)
"""
    subprocess.run([sys.executable,'-B','-c',script,str(ROOT/'experiments/cvpr2027'),str(tmp_path)],cwd=ROOT,check=True)


def test_original_imagenet_receipts_deduplicate_and_preserve_inference_k(tmp_path,plan):
    row=find(plan,task='imagenet_1nn',encoder='ff2048',method='mpsae',train_k=8,inference_k=2,mass=.9)
    source=tmp_path/'source'; out=tmp_path/'out'
    main_receipt(source/'runs/ff2048/mpsae/k8/seed1227',row)
    first=index_source(source,out,plan)
    second=index_source(source,out,plan)
    assert first['complete']==second['complete']==1
    found=next(r for r in second['results'] if r['condition_id']==row['condition_id'])
    assert found['result']['value']==61.25 and found['result']['train_k']==8 and found['result']['inference_k']==2


def test_conflicting_reruns_are_not_averaged(tmp_path,plan):
    row=find(plan,task='imagenet_1nn',encoder='ff2048',method='csr',train_k=8,inference_k=2)
    main_receipt(tmp_path/'one',row,10.)
    main_receipt(tmp_path/'two',row,20.)
    index_source(tmp_path/'one',tmp_path/'out',plan)
    status=index_source(tmp_path/'two',tmp_path/'out',plan)
    r=next(r for r in status['results'] if r['condition_id']==row['condition_id'])
    assert r['status']=='conflict' and r['result'] is None
    write_json(tmp_path/'out/selections.json',{row['condition_id']:r['candidates'][0]})
    assert rebuild_index(tmp_path/'out',plan)['complete']==1


def test_segmentation_zip_uses_saved_weight_and_fraction_units(tmp_path,plan):
    job=next(jobs_for('voc_weight'))
    cfg=job['config']
    row={'complete':True,'dataset':'voc','encoder':'siglip2','method':'mpsae','seed':1227,
         'train_k':3,'inference_k':48,'miou':.4451,'pixel_accuracy':.8,
         'identity':{'fit':'checkpoint','support_indices':'indices','neighbors':30,'temperature':.02},
         'mean_nnz':40.,'gallery_mean_nnz':41.}
    zip_path=tmp_path/'results.zip'
    with ZipFile(zip_path,'w') as z:
        z.writestr('results.json',json.dumps({'results':[row]}))
        z.writestr('config.json',json.dumps({'settings':cfg}))
        z.writestr('checkpoint.pt',b'never opened')
    status=index_source(zip_path,tmp_path/'out',plan)
    found=[r['result'] for r in status['results'] if r['status']=='complete']
    assert len(found)==1
    assert found[0]['loss_weight']==.1 and found[0]['value']==pytest.approx(44.51)
    assert found[0]['pixel_accuracy_percent']==80 and found[0]['metadata_level']=='campaign_config'


def test_mismatched_signature_rejected(tmp_path,plan):
    row=find(plan,task='imagenet_1nn',encoder='ff2048',method='csr',train_k=8,inference_k=2)
    main_receipt(tmp_path/'source',row)
    write_json(tmp_path/'source/summary.json',{'complete':True,'signature':'wrong'})
    with pytest.raises(ValueError,match='signature'):
        index_source(tmp_path/'source',tmp_path/'out',plan)


def test_full_patch_fit_metadata_preserves_metric_and_checks_preset(tmp_path,plan):
    from dataclasses import asdict
    from mpsae.patch_scene.config import Config
    job=next(jobs_for('voc_main'))
    c=Config(**job['config'])
    identity={'preset':asdict(c.preset('mpsae','siglip2')), 'method':'mpsae', 'k':3, 'seed':1227}
    cfg={'settings':job['config']}
    row={'complete':True,'dataset':'voc','encoder':'siglip2','method':'mpsae','seed':1227,
         'train_k':3,'inference_k':12,'miou':.4312,'mean_nnz':11.,'identity':{'fit':'sha-fit'}}
    write_json(tmp_path/'source/campaigns/run/settings/config.json',cfg)
    write_json(tmp_path/'source/campaigns/run/results.json',{'results':[row]})
    write_json(tmp_path/'source/fits/fit/summary.json',{'complete':True,'identity':identity,'checkpoint_sha256':'sha-fit'})
    status=index_source(tmp_path/'source',tmp_path/'out',plan)
    found=[r['result'] for r in status['results'] if r['status']=='complete']
    assert len(found)==1 and found[0]['value']==pytest.approx(43.12)
    identity['preset']['weight_decay']=.25
    write_json(tmp_path/'source/fits/fit/summary.json',{'complete':True,'identity':identity,'checkpoint_sha256':'sha-fit'})
    status=index_source(tmp_path/'source',tmp_path/'different',plan)
    assert status['complete']==0
    exclusions=next((tmp_path/'different/imports').glob('*/excluded.json'))
    assert 'preset.weight_decay' in read_json(exclusions)[0]['reason']


def test_full_retrieval_receipt_uses_saved_training_identity(tmp_path,plan):
    from dataclasses import asdict
    from mpsae.benchmark.mass_config import MassConfig
    from mpsae.config import Cell
    from mpsae.methods import describe
    cfg=MassConfig(**next(jobs_for('retrieval_mass'))['config'])
    cell=Cell('ff2048','mpsae',8,1227)
    p=cfg.spec(cell,.2).preset
    identity={'objective':describe('mpsae',8,p),'schedule':{'batch_size':p.batch_size,'epochs':p.epochs},
              'optimizer':{'lr':p.lr,'weight_decay':p.weight_decay,'eps':p.adam_eps,'betas':p.adam_betas}}
    row={'status':'complete','backbone':cell.backbone,'method':'mpsae','seed':cell.seed,
         'train_k':8,'inference_k':2,'top1_percent':55.1,'gallery_mean_nnz':2.,
         'training_identity':identity,'signature':digest(identity),'checkpoint_sha256':'fit'}
    write_json(tmp_path/'source/results.json',{'scientific':True,'results':[row]})
    status=index_source(tmp_path/'source',tmp_path/'out',plan)
    found=[r['result'] for r in status['results'] if r['status']=='complete']
    assert len(found)==1 and found[0]['mass']==.2 and found[0]['value']==55.1


def test_archive_path_traversal_rejected(tmp_path):
    path=tmp_path/'bad.zip'
    with ZipFile(path,'w') as z:
        z.writestr('../results.json','{}')
    with pytest.raises(ValueError,match='Unsafe'):
        documents(path)


def test_synthetic_and_unprovenanced_results_excluded(tmp_path,plan):
    write_json(tmp_path/'source/results.json',{'scientific':False,'results':[{'backbone':'ff2048','top1_percent':99}]})
    assert index_source(tmp_path/'source',tmp_path/'out',plan)['complete']==0
    write_json(tmp_path/'source/results.json',{'results':[{'complete':True,'dataset':'voc','encoder':'siglip2',
               'method':'mpsae','seed':1227,'train_k':3,'inference_k':3,'miou':.99,'identity':{}}]})
    assert index_source(tmp_path/'source',tmp_path/'out',plan)['complete']==0


def test_sparsity_violation_rejected(tmp_path,plan):
    row=find(plan,task='imagenet_1nn',encoder='ff2048',method='csr',train_k=8,inference_k=2)
    point=main_receipt(tmp_path/'source',row)
    point['nnz_mean']=3
    write_json(tmp_path/'source/eval_e2.json',dict(point,anchor_k=8,points=[point]))
    with pytest.raises(ValueError,match='violates K'):
        index_source(tmp_path/'source',tmp_path/'out',plan)


def test_reports_keep_missing_blank_and_real_zero(tmp_path,plan):
    row=find(plan,task='imagenet_1nn',encoder='ff2048',method='mpsae',train_k=8,inference_k=8,mass=.9)
    main_receipt(tmp_path/'source',row,0.)
    index_source(tmp_path/'source',tmp_path/'out',plan)
    manifest=report(tmp_path/'out',plan,tables=['1','2','5'],plots=True)
    assert manifest['complete']==1
    assert 'reports/figures/figure_02_resnet50.pdf' in manifest['figures']
    text=(tmp_path/'out/reports/tables/table_01.tex').read_text()
    assert '--' in text and '0.00' in text
    with (tmp_path/'out/results/metrics.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==671 and sum(r['value']!='' for r in rows)==1
    comparisons=verify_reference(tmp_path/'out')
    assert any(r['status']=='different' for r in comparisons)
    archive=bundle(tmp_path/'out')
    with ZipFile(archive) as z:
        assert 'reports/tables/table_01.tex' in z.namelist()
        assert not any(n.endswith('.pt') for n in z.namelist())


def test_report_refuses_missing_results_by_default(tmp_path):
    result=subprocess.run([sys.executable,'-B',str(ROOT/'experiments/cvpr2027/reproduce.py'),
                           'report','--output',str(tmp_path),'--tables','1','--no-plots'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==1 and 'missing/conflicting' in result.stderr


def test_patch_evaluate_only_never_trains_or_resumes(tmp_path):
    import numpy as np
    import torch
    from dataclasses import replace
    from unittest.mock import patch
    from mpsae.patch_scene.train import fit
    from mpsae.patch_scene.config import Config
    from mpsae.patch_scene.artifacts import Hub
    torch.set_num_threads(1)
    c=Config(k=3,encoders=['siglip2'],datasets=['voc'])
    p=replace(c.preset(),epochs=1,batch_size=8,hidden_dim=16,k_aux=4)
    x=np.random.default_rng(7).normal(size=(16,8)).astype('float32')
    y=np.eye(4,dtype='float32')[np.arange(16)%4]
    hub=Hub(tmp_path); identity={'encoder':'siglip2'}
    with pytest.raises(FileNotFoundError,match='evaluation-only'):
        fit(c,'csr',x,y,identity,hub,'cpu',preset=p,allow_training=False)
    fit(c,'csr',x,y,identity,hub,'cpu',preset=p,stop_after=1)
    with pytest.raises(FileNotFoundError,match='evaluation-only'):
        fit(c,'csr',x,y,identity,hub,'cpu',preset=p,allow_training=False)
    _,trained=fit(c,'csr',x,y,identity,hub,'cpu',preset=p)
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('optimizer ran')):
        _,reused=fit(c,'csr',x,y,identity,hub,'cpu',preset=p,allow_training=False)
    assert reused['checkpoint_sha256']==trained['checkpoint_sha256']


def test_image_evaluate_only_requires_completed_fit(tmp_path):
    from dataclasses import replace
    from unittest.mock import patch
    import torch
    from mpsae.config import Cell
    from mpsae.benchmark.mass_sweep import _synthetic
    from mpsae.benchmark.mass_config import MassConfig
    from mpsae.train import fit,FitSettings
    torch.set_num_threads(1)
    cfg=MassConfig(backbones=('vit_b16',),train_k={'vit_b16':[3]})
    original=Cell('vit_b16','mpsae',3,1227)
    spec=cfg.spec(original,.9,smoke=True)
    cell=Cell('vit_b16','csr',3,1227)
    spec=replace(spec,preset=replace(spec.preset,batch_size=16,epochs=1))
    data=_synthetic(spec)
    settings=FitSettings(device='cpu',max_steps=2)
    with pytest.raises(FileNotFoundError,match='evaluation-only'):
        fit(cell,data,tmp_path,replace(settings,allow_training=False),code_version='test',spec=spec)
    trained=fit(cell,data,tmp_path,settings,code_version='test',spec=spec)
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('optimizer ran')):
        reused=fit(cell,data,tmp_path,replace(settings,allow_training=False),code_version='test',spec=spec)
    assert trained['signature']==reused['signature']
    before=(tmp_path/'checkpoint.pt').read_bytes()
    with pytest.raises(FileNotFoundError,match='evaluation-only'):
        fit(cell,data,tmp_path,replace(settings,allow_training=False,force=True),code_version='test',spec=spec)
    assert (tmp_path/'checkpoint.pt').read_bytes()==before
