"""One resident sparse index serves full ImageNet accuracy and retrieval timing."""
import gc
from pathlib import Path

import numpy as np
import torch

from ..evaluate.encode import CodeSource
from ..evaluate.knn import fp32_matmul
from ..evaluate.timing import build_csr_shards, sparse_sharded_top1
from ..io import atomic_json, read_json, sha256_file
from .matched_profile import Stage, measure


def retrieval(gallery,queries,train_labels,val_labels,dim,device,config):
    device=torch.device(device)
    with Stage(device) as build:
        shards,index=build_csr_shards(gallery,dim=dim,device=device,gallery_chunk=config.gallery_chunk,
                                    build_entry_cap=config.build_entry_cap,query_batch=config.query_batch)
    query=CodeSource(queries,dim=dim,device=device)
    labels=torch.as_tensor(np.array(train_labels,copy=True),device=device,dtype=torch.long)
    truth=torch.as_tensor(np.array(val_labels,copy=True),device=device,dtype=torch.long)
    if len(labels)!=index['rows'] or len(truth)!=query.rows:
        raise ValueError('Retrieval label count mismatch')
    correct=empty=0
    with Stage(device) as accuracy,torch.no_grad(),fp32_matmul():
        for start in range(0,query.rows,config.query_batch):
            stop=min(query.rows,start+config.query_batch)
            dense=query.dense(start,stop)
            _,winner=sparse_sharded_top1(shards,dense)
            nonempty=dense.ne(0).any(dim=1)
            correct+=int((labels[winner].eq(truth[start:stop])&nonempty).sum())
            empty+=int((~nonempty).sum())
            if start==0 or stop==query.rows or stop%(config.query_batch*10)==0:
                print(f'[matched 1-NN] {stop:,}/{query.rows:,} queries',flush=True)
        del dense,winner,nonempty
    # Deterministic, evenly spaced FULL query batches; same rows for every method.
    n=min(config.query_batch,query.rows)
    blocks=max(1,query.rows//n)
    selected=sorted(set(np.linspace(0,blocks-1,min(5,blocks),dtype=int).tolist()))
    batches=[query.dense(i*n,(i+1)*n).contiguous() for i in selected]
    cursor=0
    def search():
        nonlocal cursor
        sparse_sharded_top1(shards,batches[cursor%len(batches)])
        cursor+=1
    with Stage(device) as timing_memory,torch.no_grad(),fp32_matmul():
        timing=measure(search,device,config,n)
    timing.update(timing_scope='resident CSR search+argmax; excludes encoding, index build and query transfer',
                  sampled_query_starts=[i*n for i in selected],query_batch=n,
                  backend='torch_sparse_csr_sharded_int32_matmul',precision='float32_tf32_disabled')
    return {'top1':100*correct/query.rows,'correct':correct,'queries':query.rows,'empty_queries':empty,
            'index':index,'index_build':build,'accuracy':accuracy,'timing':timing,'timing_memory':timing_memory}


def probe(gallery,queries,train_labels,val_labels,dim,device,config,seed,folder,identity,*,smoke=False):
    from ..evaluate.probe import linear_probe
    device=torch.device(device)
    folder=Path(folder)
    path=folder/'probe.pt'
    cached=read_json(folder/'probe.json')
    if cached and cached.get('identity')!=identity:
        raise RuntimeError('Probe identity mismatch')
    classes=4 if smoke else 1000
    if cached and path.is_file() and sha256_file(path)==cached['checkpoint_sha256']:
        state=torch.load(path,map_location=device,weights_only=True)
        head=torch.nn.Linear(dim,classes).to(device)
        head.load_state_dict(state)
        result=cached['fit']
        memory=cached['fit_and_validation_memory']
    else:
        with Stage(device) as memory:
            result=linear_probe(gallery,train_labels,queries,val_labels,hidden_dim=dim,device=device,
                                epochs=1 if smoke else config.probe_epochs,
                                batch_size=min(config.probe_batch_size,16) if smoke else config.probe_batch_size,
                                lr=config.probe_lr,seed=seed,num_classes=classes,return_head=True)
        head=result.pop('head')
        folder.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.partial')
        torch.save({k:v.detach().cpu() for k,v in head.state_dict().items()},temporary)
        temporary.replace(path)
        atomic_json(folder/'probe.json',{'identity':identity,'fit':result,'fit_and_validation_memory':memory,
                                        'checkpoint_sha256':sha256_file(path)})
    head.eval().requires_grad_(False)
    # Release training allocations before measuring inference memory.
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
    with Stage(device) as inference_memory:
        query=CodeSource(queries,dim=dim,device=device,prefer_resident=True)
        n=min(config.query_batch,query.rows)
        blocks=max(1,query.rows//n)
        starts=sorted(set((np.linspace(0,blocks-1,min(5,blocks),dtype=int)*n).tolist()))
        cursor=0
        def classify():
            nonlocal cursor
            first=starts[cursor%len(starts)]
            head(query.dense(first,first+n)).argmax(dim=1)
            cursor+=1
        with torch.no_grad(),fp32_matmul():
            timing=measure(classify,device,config,n)
    timing.update(timing_scope='compact-code expansion + linear head + argmax; excludes SAE encoding and head fitting',
                  sampled_query_starts=starts,precision='float32_tf32_disabled')
    return {'fit':result,'fit_and_validation_memory':memory,'inference_memory':inference_memory,'timing':timing,
            'head_parameter_bytes':sum(p.numel()*p.element_size() for p in head.parameters()),
            'checkpoint_sha256':sha256_file(path)}
