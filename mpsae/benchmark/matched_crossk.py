"""Evaluate completed matched-study checkpoints at new K values. Never train."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re

from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, sha256_file
from .matched_config import MatchedConfig

ROOT=Path(__file__).resolve().parents[2]
IMPLEMENTATION=('mpsae/model.py','mpsae/evaluate/encode.py','mpsae/evaluate/timing.py',
                'mpsae/benchmark/matched_evaluate.py')


def source_plan(root,study_id=None):
    sid=study_id or (read_json(root/'latest.json') or {}).get('study_id')
    if not isinstance(sid,str) or not re.fullmatch(r'[a-f0-9]{20}',sid):
        raise ValueError('Specify --study-id or use a matched workspace containing latest.json')
    directory=root/'studies'/sid
    plan=read_json(directory/'plan.json')
    if not plan or plan.get('study_id')!=sid:
        raise ValueError(f'Missing or inconsistent original study plan: {directory}')
    return directory,plan,MatchedConfig(**plan['config'])


def selection(config,anchors,budgets):
    for name,values in (('train K',anchors),('inference K',budgets)):
        if not values or len(set(values))!=len(values) or any(type(k) is not int or k<1 or k>2048 for k in values):
            raise ValueError(f'Invalid {name} selection')
    if not set(anchors)<=set(config.k_values):
        raise ValueError('Train K must select existing checkpoints from the original study')
    cells=[(c,b) for c,b in config.cells() if c.k in anchors]
    diagonal=sum(c.k in budgets for c,b in cells)
    return cells,{'checkpoints':len(cells),'sweep_1nn':len(cells)*len(budgets),
                  'existing_diagonal':diagonal,'new_off_diagonal':len(cells)*len(budgets)-diagonal,
                  'new_training':0,'combined_accuracy_results':config.plan()['accuracy_evaluations']+len(cells)*len(budgets)-diagonal}


def checkpoint(root,source,cell,batch):
    key=f'b{batch}-{cell.method}-k{cell.k}-s{cell.seed}'
    row=read_json(source/'cells'/key/'result.json') or {}
    fid=row.get('fit_id')
    if not isinstance(fid,str) or not re.fullmatch(r'[a-f0-9]{24}',fid):
        raise FileNotFoundError(f'{key}: source fit record missing; no training will be started')
    folder=(root/'fits'/fid).resolve()
    if not folder.is_relative_to(root.resolve()):
        raise ValueError('Checkpoint path escapes workspace')
    summary=read_json(folder/'summary.json') or {}
    identity=summary.get('identity',{})
    if not summary.get('complete') or not (folder/'checkpoint.pt').is_file():
        raise FileNotFoundError(f'{key}: completed checkpoint required; no training will be started')
    if identity!=row.get('training_identity') or canonical_hash(identity)!=summary.get('signature'):
        raise ValueError(f'{key}: training identity mismatch')
    expected={'backbone':cell.backbone,'method':cell.method,'target_k':cell.k,'seed':cell.seed}
    if any(identity.get(k)!=v for k,v in expected.items()) or identity['schedule']['batch_size']!=batch:
        raise ValueError(f'{key}: checkpoint belongs to another training condition')
    if sha256_file(folder/'checkpoint.pt')!=row.get('checkpoint_sha256'):
        raise ValueError(f'{key}: checkpoint checksum mismatch')
    return row,summary,folder


def reusable(row,signature,config,rt):
    old=row.get('identity',{})
    current={n:hashlib.sha256((ROOT/n).read_text(encoding='utf-8').encode()).hexdigest() for n in IMPLEMENTATION}
    return (old.get('fit')==signature and row.get('retrieval',{}).get('top1') is not None
            and all(old.get('implementation',{}).get(k)==v for k,v in current.items())
            and old.get('runtime')==rt
            and all(old.get('config',{}).get(k)==getattr(config,k) for k in
                    ('query_batch','gallery_chunk','build_entry_cap','encode_batch_size','warmup','rounds')))


def run(args):
    from .crossk_report import export
    root=args.workspace.resolve()
    if args.smoke:
        root=root/'SMOKE_ONLY'
    source,original,config=source_plan(root,args.study_id)
    if bool(original.get('scientific'))==args.smoke:
        raise ValueError('Synthetic and scientific workspaces must not be mixed')
    cells,counts=selection(config,args.train_k,args.inference_k)
    identity={'protocol':'matched-crossk-v1','source_study':original['study_id'],
              'train_k':sorted(args.train_k),'inference_k':sorted(args.inference_k)}
    plan={**identity,**counts,'scientific':not args.smoke,'source_config':original['config']}
    if args.plan:
        print(json.dumps(plan,indent=2)); return counts
    output=root/'crossk'/canonical_hash(identity)[:20]
    with exclusive_lock(root/'.matched-study.lock',timeout=0):
        output.mkdir(parents=True,exist_ok=True)
        if args.report_only:
            return export(output,plan,source,figures=True,bundle=True)
        import torch
        from ..features import load_bundle
        from ..train import load_fitted,FitSettings
        from ..runs import RunSettings,_code_cache,_close_code_arrays,_close_failed_cache_maps
        from ..evaluate.encode import encode_split
        from .matched_evaluate import retrieval
        from .matched_profile import Stage
        from .matched_study import runtime
        torch.set_num_threads(args.cpu_threads)
        # Resolve saved fit IDs, never recompute them from today's source code hash.
        records=[(c,b,*checkpoint(root,source,c,b)) for c,b in cells]
        rt=runtime(args.device)
        perf=json.loads(json.dumps({k:rt.get(k) for k in ('torch','cuda','device_name','uuid','capability','packages','python')}))
        impl={n:hashlib.sha256((ROOT/n).read_text(encoding='utf-8').encode()).hexdigest() for n in
              (*IMPLEMENTATION,'mpsae/evaluate/knn.py','mpsae/benchmark/matched_crossk.py')}
        session={'runtime':perf,'implementation':impl}
        previous=read_json(output/'session.json')
        if previous is not None and previous!=session:
            raise ValueError('Cross-K runtime/source changed. Preserve this output and use a separate workspace copy to avoid mixing measurements.')
        atomic_json(output/'session.json',session)
        atomic_json(output/'plan.json',plan)
        atomic_json(root/'crossk/latest.json',{'directory':str(output),'plan':plan})
        if args.smoke:
            from .mass_sweep import _synthetic
            data=_synthetic(config.spec(cells[0][0],cells[0][1],smoke=True))
        else:
            from ..config import BACKBONES
            data=load_bundle(BACKBONES[config.backbone],args.source_root/'embeddings',args.source_root/'cache')
        try:
            if any(summary['identity']['data']!=data.identity for c,b,row,summary,path in records):
                raise ValueError('Embedding data differ from the checkpoint training data')
            for cell,batch,old,summary,path in records:
                spec=config.spec(cell,batch,smoke=args.smoke)
                model=None
                try:
                    for k in sorted(args.inference_k):
                        key=f'b{batch}-{cell.method}-train{cell.k}-infer{k}-s{cell.seed}'
                        dest=output/'points'/key/'result.json'
                        point_identity={'checkpoint_sha256':old['checkpoint_sha256'],'train_k':cell.k,
                                        'inference_k':k,'data':data.identity,'session':session}
                        saved=read_json(dest)
                        if saved and saved.get('identity')==point_identity and saved.get('status')=='complete':
                            print(f'[cross-K] reuse {key}',flush=True); continue
                        point={'method':cell.method,'batch_size':batch,'seed':cell.seed,'train_k':cell.k,
                               'inference_k':k,'k':k,'identity':point_identity,'status':'running',
                               'fit_id':old['fit_id'],'checkpoint_sha256':old['checkpoint_sha256']}
                        if k==cell.k and reusable(old,summary['signature'],config,perf):
                            point.update({n:old[n] for n in ('retrieval','nnz','query_nnz','gallery_stats','query_stats','encoding') if n in old})
                            point.update(status='complete',reused_diagonal=True)
                        else:
                            print(f'[cross-K] evaluate {key}',flush=True)
                            if model is None:
                                model=load_fitted(cell,path,spec,device='cpu').requires_grad_(False)
                            model.to(args.device)
                            settings=RunSettings(cache_dir=root/'crossk/cache',fit=FitSettings(device=args.device),retain_codes=False)
                            gallery=queries=None
                            with _code_cache(settings,point_identity) as cache:
                                try:
                                    with Stage(args.device) as encoding:
                                        gallery=encode_split(model,data.train.features,None,k,device=args.device,
                                            batch_size=config.encode_batch_size,cache_dir=cache,identity={**point_identity,'split':'train'})
                                        queries=encode_split(model,data.validation.features,None,k,device=args.device,
                                            batch_size=config.encode_batch_size,cache_dir=cache,identity={**point_identity,'split':'validation'})
                                    model.to('cpu')  # same retrieval memory scope as the original evaluator
                                    point.update(encoding=encoding,nnz=gallery['stats']['nnz_mean'],query_nnz=queries['stats']['nnz_mean'],
                                                 gallery_stats=gallery['stats'],query_stats=queries['stats'])
                                    if not (0<=point['nnz']<=k and 0<=point['query_nnz']<=k):
                                        raise RuntimeError('Sparse representation exceeds inference K nonzeros')
                                    point['retrieval']=retrieval(gallery,queries,data.train.labels,data.validation.labels,
                                                                spec.preset.hidden_dim,args.device,config)
                                    point.update(status='complete',reused_diagonal=False)
                                except BaseException as error:
                                    point.update(status='failed',error=f'{type(error).__name__}: {error}')
                                    atomic_json(dest,point)
                                    _close_failed_cache_maps(error,cache)
                                    raise
                                finally:
                                    _close_code_arrays(gallery); _close_code_arrays(queries)
                                    gallery=queries=None
                        atomic_json(dest,point)
                        export(output,plan,source,figures=False,bundle=False)
                finally:
                    del model
                    gc.collect()
                    if str(args.device).startswith('cuda'):
                        torch.cuda.empty_cache()
                    export(output,plan,source,figures=False,bundle=True)
        finally:
            result=export(output,plan,source,figures=True,bundle=True)
            for split in (data.train,data.validation):
                mapping=getattr(split.features,'_mmap',None)
                if mapping is not None:
                    mapping.close()
        print(f'Cross-K results: {output}/crossk-results.zip',flush=True)
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--source-root',type=Path,default=ROOT/'work/direct-v1')
    p.add_argument('--study-id')
    p.add_argument('--train-k',nargs='+',type=int,default=[8,32])
    p.add_argument('--inference-k',nargs='+',type=int,default=[2,8,32,128])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    if args.cpu_threads<1:
        p.error('--cpu-threads must be positive')
    if not args.smoke and not args.plan and not args.report_only and not args.device.startswith('cuda'):
        p.error('Use CUDA for scientific profiling')
    try:
        run(args)
    except (ValueError,FileNotFoundError) as error:
        p.error(str(error))


if __name__=='__main__':
    main()
