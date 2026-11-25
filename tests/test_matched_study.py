import csv
import json
from pathlib import Path
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from mpsae.benchmark.matched_config import MatchedConfig
from mpsae.benchmark.matched_evaluate import retrieval
from mpsae.benchmark.matched_report import summarise
from mpsae.benchmark.matched_study import main
from mpsae.features import BatchSource
from mpsae.io import read_json
from mpsae.model import build_sae
from mpsae.train import fit_identity, total_updates


def test_plan_counts_and_equal_optimizer():
    c=MatchedConfig()
    p=c.plan()
    assert (p['sae_fits'],p['one_nn_evaluations'],p['probe_fits'],p['accuracy_evaluations'])==(72,72,12,84)
    assert p['updates_per_fit']=={'1024':12520,'4096':3130}
    assert len({(r['method'],r['k'],r['seed'],r['batch_size']) for r in p['recipes']})==72
    assert {(r['preset']['epochs'],r['preset']['lr']) for r in p['recipes']}=={(10,4e-5)}
    assert all(r['preset']['batch_size']==r['batch_size'] for r in p['recipes'])
    assert all(r['preset']['ot_weight']==(.5 if r['k']>64 else 2.) for r in p['recipes'] if r['method']=='mpsae')


@pytest.mark.parametrize('changes',[{'seeds':(1227,1227)}, {'k_values':(2,8)}, {'batch_sizes':(0,)},
                                  {'epochs':0},{'lr':float('nan')},{'backbone':'rn18'},{'methods':('mpsaev2',)}])
def test_invalid_configs(changes):
    with pytest.raises(ValueError):
        replace(MatchedConfig(),**changes)


def test_same_initialization_and_batch_sequence():
    c=MatchedConfig(k_values=(2,8),probe_k=8)
    weights=[]; orders=[]
    x=np.zeros((40,8),dtype='float32'); labels=np.arange(40)%4
    for method in c.methods:
        cell=next(cell for cell,b in c.cells() if cell.method==method and cell.k==2 and cell.seed==1227)
        model=build_sae(8,64,seed=cell.seed,dead_threshold=30,device='cpu')
        weights.append(model.encoder.weight.detach().clone())
        source=BatchSource(x,labels,batch_size=16,seed=cell.seed,device='cpu',residency='cpu')
        orders.append(source.permutation(1))
    assert all(torch.equal(weights[0],w) for w in weights)
    assert all(np.array_equal(orders[0],o) for o in orders)


def test_sparse_retrieval_matches_dense_with_ties_and_empty_queries():
    torch.set_num_threads(1)
    c=MatchedConfig(k_values=(2,),probe_k=2,query_batch=2,gallery_chunk=2,build_entry_cap=8)
    gallery={'indices':np.array([[0,1],[0,1],[2,3]],dtype='int32'),
             'values':np.array([[1,0],[1,0],[1,0]],dtype='float32')}
    queries={'indices':np.array([[0,3],[1,3],[2,3]],dtype='int32'),
             'values':np.array([[1,0],[0,0],[1,0]],dtype='float32')}
    result=retrieval(gallery,queries,np.array([0,1,2]),np.array([0,0,2]),4,'cpu',c)
    from mpsae.evaluate.knn import top1_accuracy
    dense=top1_accuracy(gallery,np.array([0,1,2]),queries,np.array([0,0,2]),dim=4,device='cpu',gallery_chunk=2,query_block=2)
    assert result['top1']==dense['top1']==pytest.approx(200/3)
    assert result['empty_queries']==1
    assert result['index']['nnz']==3
    assert result['timing']['status']=='unavailable'


def test_probe_returns_frozen_head_matching_score():
    from mpsae.evaluate.probe import linear_probe
    torch.set_num_threads(1)
    x=np.eye(4,dtype='float32').repeat(3,axis=0)
    y=np.arange(4).repeat(3)
    result=linear_probe(x,y,x,y,hidden_dim=4,device='cpu',num_classes=4,epochs=1,batch_size=4,return_head=True)
    head=result['head']
    assert all(not p.requires_grad for p in head.parameters())
    assert 100*head(torch.from_numpy(x)).argmax(1).eq(torch.from_numpy(y)).float().mean().item()==result['top1']
    ordinary=linear_probe(x,y,x,y,hidden_dim=4,device='cpu',num_classes=4,epochs=1,batch_size=4)
    assert 'head' not in ordinary and ordinary['top1']==result['top1']


def test_no_fabricated_std_or_seed_averaging():
    row={'method':'csr','k':2,'batch_size':1024,'status':'complete','top1':50.,'seed':1227}
    s=summarise([row])[0]
    assert s['top1_n']==1 and s['top1_std'] is None
    s=summarise([row,{**row,'seed':2026,'top1':54.}])[0]
    assert s['top1_mean']==52. and s['top1_std']==pytest.approx(8**.5)


def test_smoke_all_methods_two_batches_seeds_resume_export(tmp_path):
    import yaml
    torch.set_num_threads(1)
    config=MatchedConfig(k_values=(2,8),probe_k=8,batch_sizes=(8,16),probe_epochs=1,query_batch=8,gallery_chunk=16,
                         build_entry_cap=128,encode_batch_size=32)
    path=tmp_path/'study.yaml'
    path.write_text(yaml.safe_dump(asdict(config)))
    args=['--config',str(path),'--workspace',str(tmp_path/'work'),'--smoke','--device','cpu','--cpu-threads','1']
    assert main(args)==0
    root=tmp_path/'work/SMOKE_ONLY'
    folder=root/'studies'/config.plan()['study_id']
    result=read_json(folder/'results.json')
    assert result['complete']==24 and result['scientific'] is False
    fits=list((root/'fits').glob('*/checkpoint.pt'))
    assert len(fits)==24
    stamps={str(p):p.stat().st_mtime_ns for p in fits}
    for row in result['results']:
        assert row['training_identity']['schedule']['batch_size']==row['batch_size']
        assert 0<=row['nnz']<=row['k']
    with (folder/'reports/accuracy.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream)))==36
    assert (folder/'reports/top1.pdf').is_file()
    assert (folder/'matched-results.zip').is_file()
    assert main(args)==0
    assert stamps=={str(p):p.stat().st_mtime_ns for p in fits}
    assert main(args+['--report-only'])==0
    assert not list((root/'cache/transient-codes').glob('*'))


def test_evaluate_only_does_not_create_checkpoint(tmp_path):
    import yaml
    config=MatchedConfig(methods=('csr',),seeds=(1227,),k_values=(2,),probe_k=2,batch_sizes=(8,))
    path=tmp_path/'study.yaml'
    path.write_text(yaml.safe_dump(asdict(config)))
    with pytest.raises(SystemExit):
        main(['--config',str(path),'--workspace',str(tmp_path/'work'),'--smoke','--device','cpu','--evaluate-only'])
    assert not list(tmp_path.rglob('checkpoint.pt'))
