"""Separate MPSAE learning-rate extension of a completed matched study."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, read_jsonl
from .matched_crossk import checkpoint, source_plan, selection, IMPLEMENTATION

ROOT=Path(__file__).resolve().parents[2]
PERF=('torch','cuda','device_name','uuid','capability','packages','python')


def normalize(value):
    return json.loads(json.dumps(value))


def condition(method,new=False):
    return f'{method}_lr'+('8e-5' if new else '4e-5')


def study_plan(config,source_id,anchors,budgets,*,smoke=False):
    if set(config.methods)!={'csr','csrv2','mpsae'} or config.lr!=4e-5 or config.epochs!=10:
        raise ValueError('Source must compare CSR, CSRv2 and MPSAE at 10 epochs and lr=4e-5')
    batch=8 if smoke else 4096
    if batch not in config.batch_sizes:
        raise ValueError(f'Source study must contain batch {batch}')
    selected=replace(config,batch_sizes=(batch,))
    _,counts=selection(selected,anchors,budgets)
    new=replace(selected,methods=('mpsae',),lr=8e-5)
    n=len(new.cells());off=counts['new_off_diagonal']//3
    identity={'protocol':'matched-lr-v1','source_study':source_id,'new_config':asdict(new),
              'train_anchors':sorted(anchors),'inference_k':sorted(budgets),'scientific':not smoke}
    return new,{**identity,'study_id':canonical_hash(identity)[:20],
                'source_config':asdict(selected),'new_fits':n,'new_matched_1nn':n,
                'new_cross_1nn':off,'new_probes':len(config.seeds),
                'new_accuracy_evaluations':n+off+len(config.seeds),
                'baseline_cross_max_missing':3*off,'expected_unique_1nn':4*(n+off),
                'expected_probe_scores':4*len(config.seeds)}


def save_matched(output,row,path,*,new,perf):
    name=condition(row['method'],new)
    key=f'{name}-k{row["k"]}-s{row["seed"]}'
    record={**row,'condition':name,'lr':8e-5 if new else 4e-5,'train_k':row['k'],
            'inference_k':row['k'],'kind':'matched','performance_comparable':row.get('runtime')==perf,
            'checkpoint_path':str(path/'checkpoint.pt')}
    atomic_json(output/'matched'/f'{key}.json',record)
    if row['method']=='mpsae':
        # Logged minibatch samples, not fabricated epoch-average losses.
        logs=read_jsonl(path/'train.jsonl')
        per_epoch=row['training_identity']['schedule']['batches_per_epoch']
        atomic_json(output/'losses'/f'{key}.json',{
            'condition':name,'train_k':row['k'],'seed':row['seed'],'available':bool(logs),
            'samples':[{**r,'epoch_progress':r['step']/per_epoch} for r in logs if 'step' in r]})


def cached_cross(root,old,cell,k,session):
    """Find previous cross-K receipts by saved checkpoint hash, not new fit IDs."""
    key=f'b{old["batch_size"]}-{cell.method}-train{cell.k}-infer{k}-s{cell.seed}'
    for path in sorted((root/'crossk').glob(f'*/points/{key}/result.json')):
        r=read_json(path) or {};identity=r.get('identity',{})
        previous=identity.get('session',{})
        previous_config=(read_json(path.parents[2]/'plan.json') or {}).get('source_config',{})
        same_timing=all(previous_config.get(n)==old['identity']['config'].get(n) for n in
                        ('query_batch','gallery_chunk','build_entry_cap','encode_batch_size','warmup','rounds'))
        if (r.get('status')=='complete' and r.get('checkpoint_sha256')==old['checkpoint_sha256']
            and identity.get('data')==old['training_identity']['data']
            and all(previous.get('implementation',{}).get(n)==session['evaluation'][n]
                    for n in (*IMPLEMENTATION,'mpsae/evaluate/knn.py'))
            and (r.get('train_k'),r.get('inference_k'))==(cell.k,k)):
            return r,previous.get('runtime')==session['runtime'] and same_timing
    return None,False


def run(args):
    from .lr_report import export
    root=args.workspace.resolve()/('SMOKE_ONLY' if args.smoke else '')
    source,original,base=source_plan(root,args.study_id)
    if bool(original.get('scientific'))==args.smoke:
        raise ValueError('Synthetic and scientific studies cannot be mixed')
    new,plan=study_plan(base,original['study_id'],args.train_k,args.inference_k,smoke=args.smoke)
    output=root/'lr_ablation'/plan['study_id']
    if args.plan:
        print(json.dumps(plan,indent=2));return plan
    with exclusive_lock(root/'.matched-study.lock',timeout=0):
        if args.report_only:
            if not (output/'plan.json').is_file():
                raise FileNotFoundError(f'No saved LR study at {output}')
            return export(output,plan,figures=True)
        import torch
        from . import matched_study
        from ..features import load_bundle
        from ..config import BACKBONES
        from ..train import load_fitted,FitSettings
        from ..runs import RunSettings,_code_cache,_close_code_arrays,_close_failed_cache_maps
        from ..evaluate.encode import encode_split
        from .matched_evaluate import retrieval
        from .matched_profile import Stage
        torch.set_num_threads(args.cpu_threads)
        rt=matched_study.runtime(args.device)
        perf=normalize({k:rt.get(k) for k in PERF})
        hashes={n:hashlib.sha256((ROOT/n).read_text(encoding='utf-8').encode()).hexdigest()
                for n in (*IMPLEMENTATION,'mpsae/evaluate/knn.py','mpsae/evaluate/probe.py')}
        session={'runtime':perf,'evaluation':hashes,'training':matched_study._source_hash(rt)}
        prior=read_json(output/'session.json')
        if prior is not None and prior!=session:
            raise ValueError('LR study runtime or numerical code changed; preserve outputs before starting a separate comparison')
        baseline=[];batch=new.batch_sizes[0]
        # Validate every baseline before launching expensive new training.
        for cell,b in base.cells():
            if b!=batch:
                continue
            row,summary,path=checkpoint(root,source,cell,b)
            if row.get('status')!='complete' or row.get('retrieval',{}).get('top1') is None:
                raise ValueError(f'Baseline evaluation incomplete: {path}; baseline training is never started here')
            if cell.k==base.probe_k and row.get('probe',{}).get('fit',{}).get('top1') is None:
                raise ValueError(f'Baseline probe missing: {path}')
            impl=row.get('identity',{}).get('implementation',{})
            if any(impl.get(n)!=hashes[n] for n in (*IMPLEMENTATION,'mpsae/evaluate/probe.py')):
                raise ValueError('Baseline evaluator differs from current code; rerun original evaluation before comparing')
            if row['training_identity']['optimizer']['lr']!=4e-5:
                raise ValueError('Baseline checkpoint has the wrong learning rate')
            baseline.append((cell,b,row,summary,path,False))
        atomic_json(output/'session.json',session);atomic_json(output/'plan.json',plan)
        atomic_json(root/'lr_ablation/latest.json',{'directory':str(output),'study_id':plan['study_id']})
        for cell,b,row,summary,path,_ in baseline:
            save_matched(output,row,path,new=False,perf=perf)
        training=output/'training'
        trainroot=training/('SMOKE_ONLY' if args.smoke else '')
        newsource=trainroot/'studies'/new.plan()['study_id']
        def sync_new(campaign=None):
            for cell,b in new.cells():
                r=read_json(newsource/'cells'/f'b{b}-{cell.method}-k{cell.k}-s{cell.seed}'/'result.json')
                if r and r.get('status')=='complete':
                    save_matched(output,r,trainroot/'fits'/r['fit_id'],new=True,perf=perf)
            export(output,plan,figures=False)
        state={'status':'training'};atomic_json(output/'state.json',state)
        data=None
        try:
            if args.smoke:
                from .mass_sweep import _synthetic
                data=_synthetic(new.spec(*new.cells()[0],smoke=True))
            else:
                data=load_bundle(BACKBONES[base.backbone],args.source_root/'embeddings',args.source_root/'cache')
            if any(s['identity']['data']!=data.identity for c,b,r,s,p,n in baseline):
                raise ValueError('Embedding data differ from saved baseline checkpoint data')
            from ..train import fit_identity,total_updates
            from .. import __version__
            for cell,b,row,summary,path,_ in baseline:
                spec=base.spec(cell,b,smoke=args.smoke)
                per_epoch,total=total_updates(data,spec.preset,max_steps=2 if args.smoke else 0)
                expected=fit_identity(cell,data,spec.preset,total_steps=total,per_epoch=per_epoch,
                                      code_version=__version__,batch_size_override=b)
                if any(expected[k]!=row['training_identity'].get(k) for k in ('objective','optimizer','schedule','precision')):
                    raise ValueError(f'Baseline recipe differs from current matched recipe: {path}')
            sync_new()
            trainargs=SimpleNamespace(**vars(args))
            trainargs.workspace=training
            trainargs.embeddings_dir=args.source_root/'embeddings';trainargs.cache_dir=args.source_root/'cache'
            trainargs.download_embeddings=False;trainargs.evaluate_only=False
            trainargs.retime=False;trainargs.continue_on_error=False
            # Separate workspace + lr in fit identity: never warm-start from baseline weights.
            matched_study.run(new,trainargs,on_progress=sync_new)
            sync_new()
            records=baseline+[(c,b,*checkpoint(trainroot,newsource,c,b),True) for c,b in new.cells()]
            state['status']='inference';atomic_json(output/'state.json',state)
            if any(s['identity']['data']!=data.identity for c,b,r,s,p,n in records):
                raise ValueError('Embedding data differ from saved checkpoint data')
            for cell,b,row,summary,path,isnew in records:
                if cell.k not in args.train_k:
                    continue
                name=condition(cell.method,isnew);model=None
                try:
                    for k in sorted(args.inference_k):
                        if k==cell.k:
                            continue  # diagonal already saved as matched, no duplicate computation
                        dest=output/'cross'/f'{name}-train{cell.k}-infer{k}-s{cell.seed}.json'
                        identity={'checkpoint_sha256':row['checkpoint_sha256'],'inference_k':k,'session':session}
                        point=read_json(dest)
                        if point and point.get('identity')==identity and point.get('status')=='complete':
                            print(f'[LR inference] reuse {dest.stem}',flush=True);continue
                        previous,comparable=cached_cross(root,row,cell,k,session) if not isnew else (None,False)
                        point={**(previous or {}),'condition':name,'method':cell.method,'batch_size':b,
                               'seed':cell.seed,'lr':8e-5 if isnew else 4e-5,'train_k':cell.k,
                               'inference_k':k,'k':k,'kind':'cross','identity':identity,
                               'checkpoint_sha256':row['checkpoint_sha256'],'status':'running',
                               'performance_comparable':comparable if previous else True,
                               'reused_previous_cross':previous is not None}
                        if previous is None:
                            print(f'[LR inference] evaluate {dest.stem}',flush=True)
                            spec=new.spec(cell,b,smoke=args.smoke) if isnew else base.spec(cell,b,smoke=args.smoke)
                            if model is None:
                                model=load_fitted(cell,path,spec,device='cpu').requires_grad_(False)
                            model.to(args.device)
                            settings=RunSettings(cache_dir=output/'cache',fit=FitSettings(device=args.device),retain_codes=False)
                            gallery=queries=None
                            with _code_cache(settings,identity) as cache:
                                try:
                                    with Stage(args.device) as encoding:
                                        gallery=encode_split(model,data.train.features,None,k,device=args.device,
                                            batch_size=base.encode_batch_size,cache_dir=cache,identity={**identity,'split':'train'})
                                        queries=encode_split(model,data.validation.features,None,k,device=args.device,
                                            batch_size=base.encode_batch_size,cache_dir=cache,identity={**identity,'split':'validation'})
                                    model.to('cpu')
                                    point.update(encoding=encoding,nnz=gallery['stats']['nnz_mean'],query_nnz=queries['stats']['nnz_mean'],
                                                 gallery_stats=gallery['stats'],query_stats=queries['stats'])
                                    if not (0<=point['nnz']<=k and 0<=point['query_nnz']<=k):
                                        raise RuntimeError('Code exceeds inference K')
                                    point['retrieval']=retrieval(gallery,queries,data.train.labels,data.validation.labels,
                                                                 spec.preset.hidden_dim,args.device,base)
                                except BaseException as error:
                                    _close_failed_cache_maps(error,cache)
                                    raise
                                finally:
                                    _close_code_arrays(gallery);_close_code_arrays(queries)
                        point['status']='complete';atomic_json(dest,point)
                        export(output,plan,figures=False)
                finally:
                    del model;gc.collect()
                    if args.device.startswith('cuda'):
                        torch.cuda.empty_cache()
            state['status']='complete'
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            raise
        finally:
            atomic_json(output/'state.json',state)
            if data is not None:
                for split in (data.train,data.validation):
                    mapping=getattr(split.features,'_mmap',None)
                    if mapping is not None:mapping.close()
            export(output,plan,figures=True)
        print(f'Results: {output}/lr-ablation-results.zip',flush=True)
        return read_json(output/'results.json')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True,help='Completed original matched-study workspace')
    p.add_argument('--source-root',type=Path,default=ROOT/'work/direct-v1')
    p.add_argument('--study-id')
    p.add_argument('--train-k',type=int,nargs='+',default=[8,32],help='Cross-K anchor checkpoints; all original K still train')
    p.add_argument('--inference-k',type=int,nargs='+',default=[2,8,32,128])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--residency',choices=['auto','cuda','cpu','mmap'],default='auto')
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--checkpoint-every',type=int,default=512)
    for flag in ('plan','report-only','smoke'):p.add_argument('--'+flag,action='store_true')
    args=p.parse_args(argv);args.source_root=args.source_root.resolve()
    if min(args.cpu_threads,args.log_every,args.checkpoint_every)<1:p.error('Runtime counts must be positive')
    if not (args.smoke or args.plan or args.report_only) and not args.device.startswith('cuda'):
        p.error('Scientific profiling requires CUDA')
    try:
        run(args)
    except (ValueError,FileNotFoundError) as error:p.error(str(error))


if __name__=='__main__':main()
