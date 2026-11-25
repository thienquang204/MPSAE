"""Standalone controlled CSR/CSRv2/MPSAE comparison on cached ResNet-50 features."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import subprocess
import sys

from .. import __version__
from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, sha256_file
from .matched_config import MatchedConfig

ROOT=Path(__file__).resolve().parents[2]
SOURCES=('mpsae/config.py','mpsae/train.py','mpsae/features.py','mpsae/model.py','mpsae/methods.py',
         'mpsae/losses.py','mpsae/ot.py','mpsae/ot_balanced.py','mpsae/evaluate/probe.py',
         'mpsae/evaluate/encode.py','mpsae/evaluate/timing.py','mpsae/benchmark/matched_evaluate.py',
         'mpsae/benchmark/matched_profile.py')


def runtime(device):
    from ..timing_config import runtime_fingerprint
    import importlib.metadata
    import hashlib
    result=runtime_fingerprint(device)
    result['python']=sys.version
    result['packages']={n:importlib.metadata.version(n) for n in ('numpy','psutil','matplotlib')}
    result['source_sha256']={n:hashlib.sha256((ROOT/n).read_text(encoding='utf-8').encode()).hexdigest() for n in SOURCES}
    try:
        result['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        result['git_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except (OSError,subprocess.CalledProcessError):
        result['git_commit']=None
    return result


def _source_hash(runtime):
    # Reporting/profiling edits must not invalidate compatible trained SAEs.
    return {k:v for k,v in runtime['source_sha256'].items() if k in SOURCES[:8]}


def run(config,args,*,on_progress=None):
    from .matched_report import export
    plan=config.plan()
    root=args.workspace/('SMOKE_ONLY' if args.smoke else '')
    campaign=root/'studies'/plan['study_id']
    plan['scientific']=not args.smoke
    with exclusive_lock(root/'.matched-study.lock',timeout=0):
        atomic_json(campaign/'plan.json',plan)
        atomic_json(root/'latest.json',{'study_id':plan['study_id'],'path':str(campaign)})
        if args.report_only:
            return export(campaign,plan,plots=True,bundle=True)
        import torch
        from ..config import BACKBONES
        from ..features import load_bundle
        from ..train import fit, fit_identity, total_updates, FitSettings, load_fitted
        from ..logs import setup_logging
        from ..runs import RunSettings, _code_cache, _close_code_arrays, _close_failed_cache_maps
        from ..evaluate.encode import encode_split
        from .matched_evaluate import retrieval, probe
        from .matched_profile import Stage
        torch.set_num_threads(args.cpu_threads)
        setup_logging(root/'logs',name='matched-study')
        rt=runtime(args.device)
        performance_runtime={k:rt.get(k) for k in ('torch','cuda','device_name','uuid','capability','packages','python')}
        atomic_json(campaign/'runtime.json',rt)
        if args.download_embeddings and not args.smoke:
            from .retrieval_studies import download_features
            download_features(args,[config.backbone])
        first,b=config.cells()[0]
        if args.smoke:
            from .mass_sweep import _synthetic
            bundle=_synthetic(config.spec(first,b,smoke=True))
        else:
            bundle=load_bundle(BACKBONES[config.backbone],args.embeddings_dir,args.cache_dir)
            if bundle.train.rows!=1281167 or bundle.validation.rows!=50000:
                raise ValueError('Main ImageNet protocol requires all 1,281,167 train and 50,000 validation rows')
        state={'status':'running'}
        atomic_json(campaign/'state.json',state)
        try:
            for number,(cell,batch) in enumerate(config.cells(),1):
                key=f'b{batch}-{cell.method}-k{cell.k}-s{cell.seed}'
                target=campaign/'cells'/key
                settings=FitSettings(device=args.device,residency=args.residency,max_steps=2 if args.smoke else 0,
                                     checkpoint_every=args.checkpoint_every,log_every=args.log_every,
                                     batch_size_override=batch,allow_training=not args.evaluate_only)
                spec=config.spec(cell,batch,smoke=args.smoke)
                per_epoch,total=total_updates(bundle,spec.preset,max_steps=settings.max_steps)
                identity=fit_identity(cell,bundle,spec.preset,total_steps=total,per_epoch=per_epoch,
                                      code_version=__version__,batch_size_override=batch)
                signature=canonical_hash(identity)
                fit_id=canonical_hash({'training':identity,'implementation':_source_hash(rt)})[:24]
                out=root/'fits'/fit_id
                eval_identity={'fit':signature,'implementation':rt['source_sha256'],
                               'config':asdict(config),'runtime':performance_runtime,'smoke':args.smoke}
                # Normalize tuple/list representations before checking saved receipts.
                eval_identity=json.loads(json.dumps(eval_identity))
                row=read_json(target/'result.json') or {}
                if row.get('identity')==eval_identity and row.get('status')=='complete' and not args.retime:
                    print(f'[matched {number}/{plan["sae_fits"]}] reuse {key}',flush=True)
                    continue
                row={'method':cell.method,'k':cell.k,'seed':cell.seed,'batch_size':batch,
                     'identity':eval_identity,'status':'running','fit_id':fit_id,'fit_path':str(out),
                     'training_identity':identity,'runtime':performance_runtime,'scientific':not args.smoke}
                atomic_json(target/'result.json',row)
                model=gallery=queries=None
                try:
                    print(f'[matched {number}/{plan["sae_fits"]}] {key}',flush=True)
                    before=read_json(out/'summary.json') or {}
                    complete=before.get('complete') and (out/'checkpoint.pt').is_file()
                    if complete:
                        summary=fit(cell,bundle,out,settings,code_version=__version__,spec=spec)
                    else:
                        profiles=read_json(out/'profiles.json',[])
                        covered=not (out/'checkpoint.pt').is_file() or bool(profiles and profiles[0].get('covers_from_start'))
                        train_profile={}
                        try:
                            with Stage(args.device) as train_profile:
                                summary=fit(cell,bundle,out,settings,code_version=__version__,spec=spec)
                        finally:
                            if not args.evaluate_only:
                                profiles.append({'runtime':performance_runtime,'profile':train_profile,'covers_from_start':covered})
                                atomic_json(out/'profiles.json',profiles)
                    if summary['identity']['schedule']['batch_size']!=batch:
                        raise RuntimeError('Training batch override was not applied')
                    profiles=read_json(out/'profiles.json',[])
                    matching=[p['profile'] for p in profiles if p['runtime']==performance_runtime]
                    row.update(train_seconds=summary['train_seconds'],steps=summary['steps'],
                               training_profiles=matching,training_residency=summary['residency'],
                               train_memory_complete=bool(matching) and len(matching)==len(profiles) and profiles[0].get('covers_from_start',False),
                               checkpoint_sha256=sha256_file(out/'checkpoint.pt'))
                    atomic_json(target/'result.json',row)
                    model=load_fitted(cell,out,spec,device=args.device)
                    cache_settings=RunSettings(root=root/'fits',cache_dir=root/'cache',fit=settings,retain_codes=False)
                    with _code_cache(cache_settings,eval_identity) as cache:
                        try:
                            with Stage(args.device) as encoding:
                                gallery=encode_split(model,bundle.train.features,None,cell.k,device=args.device,
                                    batch_size=config.encode_batch_size,cache_dir=cache,
                                    identity={'fit':signature,'split':'train','data':bundle.identity})
                                queries=encode_split(model,bundle.validation.features,None,cell.k,device=args.device,
                                    batch_size=config.encode_batch_size,cache_dir=cache,
                                    identity={'fit':signature,'split':'validation','data':bundle.identity})
                            row.update(encoding=encoding,nnz=gallery['stats']['nnz_mean'],query_nnz=queries['stats']['nnz_mean'],
                                       gallery_stats=gallery['stats'],query_stats=queries['stats'],
                                       compact_code_bytes=sum(x['indices'].nbytes+x['values'].nbytes for x in (gallery,queries)))
                            if not (0<=row['nnz']<=cell.k and 0<=row['query_nnz']<=cell.k):
                                raise RuntimeError('Sparse representation exceeds K nonzeros')
                            del model
                            model=None
                            row['retrieval']=retrieval(gallery,queries,bundle.train.labels,bundle.validation.labels,
                                                       spec.preset.hidden_dim,args.device,config)
                            atomic_json(target/'result.json',row)
                            if cell.k==config.probe_k:
                                probe_id={'fit':signature,'checkpoint_sha256':row['checkpoint_sha256'],
                                          'epochs':config.probe_epochs,'batch':config.probe_batch_size,
                                          'lr':config.probe_lr,'seed':cell.seed,'smoke':args.smoke,
                                          'implementation':rt['source_sha256']['mpsae/evaluate/probe.py']}
                                row['probe']=probe(gallery,queries,bundle.train.labels,bundle.validation.labels,
                                    spec.preset.hidden_dim,args.device,config,cell.seed,
                                    out/'probes'/canonical_hash(probe_id)[:20],probe_id,smoke=args.smoke)
                        except BaseException as error:
                            _close_failed_cache_maps(error,cache)
                            raise
                        finally:
                            _close_code_arrays(gallery)
                            _close_code_arrays(queries)
                            gallery=queries=None
                    row['status']='complete'
                    atomic_json(target/'result.json',row)
                except BaseException as error:
                    row.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                               error=f'{type(error).__name__}: {error}')
                    atomic_json(target/'result.json',row)
                    if not args.continue_on_error or not isinstance(error,Exception):
                        raise
                finally:
                    del model,gallery,queries
                    gc.collect()
                    if str(args.device).startswith('cuda'):
                        torch.cuda.empty_cache()
                    export(campaign,plan,plots=False,bundle=False)
                    if on_progress is not None:
                        on_progress(campaign)
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            raise
        finally:
            if state['status']=='running':
                state['status']='finished'
            atomic_json(campaign/'state.json',state)
            result=export(campaign,plan,plots=True,bundle=True)
            for split in (bundle.train,bundle.validation):
                mapping=getattr(split.features,'_mmap',None)
                if mapping is not None:
                    mapping.close()
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/matched_resnet50.yaml')
    p.add_argument('--workspace',type=Path,default=ROOT/'work/matched-resnet50')
    p.add_argument('--source-root',type=Path,default=ROOT/'work/direct-v1')
    p.add_argument('--embeddings-dir',type=Path)
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--embedding-repo')
    p.add_argument('--download-embeddings',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--residency',choices=['auto','cuda','cpu','mmap'],default='auto')
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--checkpoint-every',type=int,default=512)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--retime',action='store_true',help='refresh evaluation/profile measurements, retaining completed SAE and probe fits')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    args=p.parse_args(argv)
    try:
        config=MatchedConfig.load(args.config)
        if any(getattr(args,k)<1 for k in ('cpu_threads','log_every','checkpoint_every')):
            raise ValueError('Runtime counts must be positive')
        if args.plan:
            print(json.dumps(config.plan(),indent=2))
            return 0
        if not args.smoke and not args.report_only and not args.device.startswith('cuda'):
            raise ValueError('Scientific performance measurements require CUDA; use --smoke for CPU tests')
        args.workspace=args.workspace.resolve()
        args.source_root=args.source_root.resolve()
        args.embeddings_dir=(args.embeddings_dir or args.source_root/'embeddings').resolve()
        args.cache_dir=(args.cache_dir or args.source_root/'cache').resolve()
        result=run(config,args)
        print(f'Results: {result["directory"]}; {result["complete"]}/{result["expected"]} fits evaluated')
        return 0 if result['complete']==result['expected'] else 1
    except (ValueError,FileNotFoundError) as error:
        p.error(str(error))


if __name__=='__main__':
    raise SystemExit(main())
