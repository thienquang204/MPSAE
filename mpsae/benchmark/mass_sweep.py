"""ImageNet MPSAE transport-mass ablation, with local resumable artifacts.

python -m mpsae.benchmark.mass_sweep --config configs/retrieval_mass.yaml --plan
Only cached global image embeddings are used. No backbone or downstream fits.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import gc
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from .. import __version__
from ..config import BACKBONES
from ..io import (append_jsonl, atomic_json, canonical_hash, exclusive_lock,
                  read_json, sha256_file)
from ..transport import solver_name
from .mass_config import MassConfig

PROTOCOL = 'imagenet-mass-exact-1nn-v1'
ROOT = Path(__file__).resolve().parents[2]
FIELDS = ['backbone', 'method', 'train_k', 'inference_k', 'seed', 's', 'lambda',
          'solver', 'status', 'top1_percent', 'gallery_mean_nnz', 'empty_queries',
          'train_seconds', 'fit_id', 'signature', 'checkpoint_sha256', 'checkpoint_path',
          'evaluation_id', 'reused_source_fit', 'reused_source_evaluation', 'error']


def _key(cell, mass, k):
    return f'{cell.slug}/s{mass:g}/infer{k}'


def _csv(path, rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(stream.getvalue(), encoding='utf-8')
    os.replace(tmp, path)


def export(root, plan, rows):
    """Export after each cell, including failures and unfinished conditions."""
    folder = root/'sweeps'/plan['sweep_id']
    folder.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    completed = sum(row['status'] == 'complete' for row in ordered)
    result = dict(sweep_id=plan['sweep_id'], complete=completed == plan['evaluations'],
                  scientific=plan.get('scientific',True),
                  completed=completed, expected=plan['evaluations'], results=ordered)
    atomic_json(folder/'results.json', result)
    _csv(folder/'results.csv', ordered, FIELDS)
    groups = {}
    for row in ordered:
        if row['status'] == 'complete':
            key = (row['backbone'], row['train_k'], row['inference_k'], row['s'])
            groups.setdefault(key, []).append(row)
    summary = []
    for (b, kt, ki, s), values in sorted(groups.items()):
        scores = [r['top1_percent'] for r in values]
        summary.append(dict(backbone=b, train_k=kt, inference_k=ki, s=s,
                            top1_mean=statistics.mean(scores),
                            top1_std=statistics.stdev(scores) if len(scores)>1 else None,
                            n=len(scores), expected_seeds=len(plan['config']['seeds']),
                            gallery_mean_nnz=statistics.mean(r['gallery_mean_nnz'] for r in values)))
    _csv(folder/'summary.csv', summary, ['backbone','train_k','inference_k','s',
         'top1_mean','top1_std','n','expected_seeds','gallery_mean_nnz'])
    tex = (['% SYNTHETIC SMOKE TEST -- NOT SCIENTIFIC RESULTS'] if not result['scientific'] else [])
    tex += [r'\begin{tabular}{lrrrrrr}', r'\toprule',
           r'Encoder & $K_{\rm train}$ & $K_{\rm test}$ & $s$ & 1-NN (\%) & nnz & Seeds \\',
           r'\midrule']
    names = {'ff2048':'ResNet-50', 'rn18':'ResNet-18', 'vit_b16':'ViT-B/16',
             'convnext_tiny':'ConvNeXt-T', 'dinov2_vits14':'DINOv2-S/14'}
    for r in summary:
        score = f"{r['top1_mean']:.2f}"
        if r['top1_std'] is not None:
            score += r' $\pm$ ' + f"{r['top1_std']:.2f}"
        tex.append(f"{names[r['backbone']]} & {r['train_k']} & {r['inference_k']} & "
                   f"{r['s']:g} & {score} & {r['gallery_mean_nnz']:.2f} & "
                   f"{r['n']}/{r['expected_seeds']} " + r'\\')
    tex += [r'\bottomrule', r'\end{tabular}',
            '% booktabs required. nnz: measured gallery mean. Dispersion: sample SD across seeds.',
            '% Missing conditions remain pending/failed in results.csv; never replace them with zero.']
    (folder/'table.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    history = []
    for path in sorted((root/'sweeps').glob('*/results.json')):
        payload = read_json(path)
        history.extend(dict(row, sweep_id=payload['sweep_id']) for row in payload['results'])
    atomic_json(root/'experiment-index.json', history)
    _csv(root/'experiment-index.csv', history, ['sweep_id', *FIELDS])
    (folder/'README.txt').write_text(
        ('SYNTHETIC SMOKE TEST -- NOT SCIENTIFIC RESULTS\n' if not result['scientific'] else '') +
        'ImageNet exact 1-NN; train support / validation queries. Local artifacts only.\n'
        's=1: balanced entropic MMOT matching gap; s<1: partial matching gap.\n'
        'results.csv includes every requested condition; summary.csv/table.tex aggregate completed seeds.\n'
        'This ZIP contains results/provenance, not model weights or embeddings. Keep the workspace to resume.\n'
        f"Completed {completed}/{plan['evaluations']}.\n", encoding='utf-8')
    archive = folder/'retrieval-mass-results.zip'
    tmp = archive.with_suffix('.zip.partial')
    with ZipFile(tmp, 'w', ZIP_DEFLATED) as z:
        for path in sorted(folder.iterdir()):
            if path.suffix in ('.json', '.jsonl', '.csv', '.tex', '.txt'):
                z.write(path, path.name)
        for name in ('experiment-index.csv', 'experiment-index.json'):
            z.write(root/name, name)
    os.replace(tmp, archive)
    atomic_json(root/'latest.json', {'sweep_id': plan['sweep_id'], 'archive': str(archive),
                                   'completed': completed, 'expected': plan['evaluations']})
    print(f'[mass retrieval] {completed}/{plan["evaluations"]} complete; {archive}', flush=True)
    return result


def _runtime(args):
    import torch
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(started_at=datetime.now(timezone.utc).isoformat(), git_commit=commit, git_dirty=dirty,
                python=sys.version, torch=str(torch.__version__), device=args.device,
                gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
                smoke=args.smoke, source_root=str(args.source_root),
                sources={f:sha256_file(ROOT/f) for f in (
                    'mpsae/ot.py','mpsae/ot_balanced.py','mpsae/methods.py',
                    'mpsae/benchmark/mass_sweep.py','mpsae/benchmark/mass_config.py')})


def _synthetic(spec):
    import numpy as np
    from ..features import Bundle, Split
    from ..io import array_hash
    rng = np.random.default_rng(1227)
    splits = []
    identity = {'backbone':spec.name, 'dim':spec.dim}
    for name, count in (('train',64), ('validation',16)):
        x = rng.normal(size=(count,spec.dim)).astype('float32')
        y = np.arange(count, dtype='int64') % 4
        metadata = dict(sha256=array_hash(x), rows=count, recipe='synthetic-mass-smoke-v1', source=None)
        splits.append(Split(name,x,y,metadata))
        identity[name] = metadata
    return Bundle(spec.name,*splits,identity)


def _source_fit(cell, requested, source_root):
    """Read-only reuse requires the complete objective, data and schedule identity."""
    from ..train import compatible_fit_identity
    out = source_root/'runs'/cell.slug
    summary, config = read_json(out/'summary.json'), read_json(out/'config.json')
    if not summary or not summary.get('complete') or not config or not (out/'checkpoint.pt').is_file():
        return None
    identity = summary.get('identity')
    if not isinstance(identity,dict) or not compatible_fit_identity(identity,requested):
        return None
    signature = canonical_hash(identity)
    if (summary.get('signature') != signature or config.get('signature') != signature
            or canonical_hash(config.get('identity')) != signature
            or summary.get('steps') != requested['schedule']['total_steps']):
        raise RuntimeError(f'Inconsistent completed source checkpoint: {out}')
    return out, summary


def run(config, args):
    root = args.workspace.resolve()/('SMOKE_ONLY' if args.smoke else '')
    plan = config.plan()
    plan['scientific'] = not args.smoke
    folder = root/'sweeps'/plan['sweep_id']
    rows = {}
    for cell,s in config.cells():
        preset = config.spec(cell,s,smoke=args.smoke).preset
        for k in config.budgets(cell):
            rows[_key(cell,s,k)] = dict(backbone=cell.backbone,method='mpsae',train_k=cell.k,
                inference_k=k,seed=cell.seed,s=s,**{'lambda':preset.ot_weight},
                solver=solver_name(s),status='pending')
    with exclusive_lock(root/'.mass-retrieval.lock', timeout=0):
        previous = read_json(folder/'plan.json')
        if previous is not None and canonical_hash(previous) != canonical_hash(plan):
            raise RuntimeError('Immutable sweep plan changed')
        atomic_json(folder/'plan.json',plan)
        old = read_json(folder/'results.json') or {}
        for row in old.get('results',[]):
            from ..config import Cell
            key = _key(Cell(row['backbone'],'mpsae',row['train_k'],row['seed']),row['s'],row['inference_k'])
            if key in rows:
                rows[key] = row
        if args.export_only:
            return export(root,plan,rows)
        import torch
        from ..features import load_bundle
        from ..logs import setup_logging
        from ..runs import RunSettings, evaluate_at, _cached_e1_complete
        from ..train import FitSettings, fit, fit_identity, total_updates, load_fitted
        # Previous results are evidence, not permission to bypass source and
        # checkpoint validation on a new invocation.
        for row in rows.values():
            row['status'] = 'pending'
        torch.set_num_threads(args.cpu_threads)
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device unavailable: {args.device}')
        setup_logging(root/'logs',name='retrieval-mass')
        runtime = _runtime(args)
        append_jsonl(folder/'attempts.jsonl',runtime)
        if not (folder/'runtime.json').exists():
            atomic_json(folder/'runtime.json',runtime)
        export(root,plan,rows)
        for backbone in config.backbones:
            selected = [(c,s) for c,s in config.cells() if c.backbone == backbone]
            first,s = selected[0]
            bundle = None
            try:
                bundle = (_synthetic(config.spec(first,s,smoke=True)) if args.smoke else
                          load_bundle(BACKBONES[backbone], args.embeddings_dir, args.cache_dir))
                if not args.smoke:
                    if (bundle.train.rows,bundle.validation.rows) != (1_281_167,50_000):
                        raise ValueError('Scientific retrieval requires the full 1,281,167/50,000 ImageNet splits')
                    if any('synthetic' in str(bundle.identity[x]).lower() for x in ('train','validation')):
                        raise ValueError('Synthetic embeddings require --smoke')
                for cell,mass in selected:
                    keys = [_key(cell,mass,k) for k in config.budgets(cell)]
                    model = None
                    try:
                        for key in keys:
                            rows[key] = {**rows[key],'status':'running','error':None}
                        export(root,plan,rows)
                        spec = config.spec(cell,mass,smoke=args.smoke)
                        fits = FitSettings(device=args.device, max_steps=2 if args.smoke else 0,
                            residency=args.residency, log_every=args.log_every,
                            checkpoint_every=args.checkpoint_every)
                        per_epoch,total = total_updates(bundle,spec.preset,max_steps=fits.max_steps)
                        identity = fit_identity(cell,bundle,spec.preset,total_steps=total,
                                                per_epoch=per_epoch,code_version=__version__)
                        fit_id = canonical_hash(identity)
                        out = root/'fits'/fit_id
                        reused = (None if args.smoke or not args.reuse_source or
                                  (out/'checkpoint.pt').exists() else
                                  _source_fit(cell,identity,args.source_root))
                        print(f'[mass retrieval] {cell.slug}; s={mass:g}; {solver_name(mass)}',flush=True)
                        if reused:
                            out,summary = reused
                            print(f'[reuse source fit] {out}',flush=True)
                        else:
                            summary = fit(cell,bundle,out,fits,code_version=__version__,spec=spec)
                        signature = summary['signature']
                        # Verify model contents even when reusing all evaluation numbers.
                        model = load_fitted(cell,out,spec,device=args.device)
                        checkpoint_hash = sha256_file(out/'checkpoint.pt')
                        settings = RunSettings(root=root/'fits',cache_dir=args.cache_dir,fit=fits,
                            encode_batch_size=args.encode_batch_size, gallery_chunk=args.gallery_chunk,
                            query_block=args.query_block,linear_probe=False,measure_timing=False,
                            run_e2=False,retain_codes=False)
                        for k,key in zip(config.budgets(cell),keys):
                            eval_identity = dict(protocol=PROTOCOL,fit=signature,checkpoint=checkpoint_hash,
                                data=bundle.identity,k=k,device_precision='cpu-fp32' if args.device.startswith('cpu')
                                else 'cuda-autocast-fp16',gallery_chunk=args.gallery_chunk,query_block=args.query_block)
                            eid = canonical_hash(eval_identity)
                            path = root/'evaluations'/eid/'result.json'
                            cached = read_json(path)
                            if cached and cached.get('identity') != eval_identity:
                                raise RuntimeError(f'Evaluation identity mismatch: {path}')
                            point = cached.get('point') if cached else None
                            reused_eval = bool(cached and cached.get('reused_source_evaluation'))
                            if not (_cached_e1_complete(point,settings) and point.get('k') == k
                                    and point.get('signature') == signature and 'nnz_mean' in point):
                                legacy = read_json(out/'eval_e1.json') if reused and k == cell.k else None
                                if (legacy and legacy.get('signature') == signature and legacy.get('k') == k
                                        and 'nnz_mean' in legacy and _cached_e1_complete(legacy,settings)):
                                    point,reused_eval = legacy,True
                                else:
                                    point = evaluate_at(cell,bundle,model,spec,settings,k=k,signature=signature,
                                                        want_probe=False,want_timing=False)
                                atomic_json(path,dict(identity=eval_identity,point=point,
                                            reused_source_evaluation=reused_eval))
                            if not 0 <= point['nnz_mean'] <= k:
                                raise RuntimeError('Measured sparse-code nnz exceeds the requested K')
                            rows[key].update(status='complete',top1_percent=point['top1'],
                                gallery_mean_nnz=point['nnz_mean'],empty_queries=point['empty_queries'],
                                fit_id=fit_id,signature=signature,checkpoint_sha256=checkpoint_hash,
                                checkpoint_path=str(out/'checkpoint.pt'),evaluation_id=eid,
                                reused_source_fit=bool(reused),reused_source_evaluation=reused_eval,
                                train_seconds=summary['train_seconds'],
                                final_training_metrics=summary.get('final_metrics',{}),
                                training_identity=summary['identity'],evaluation_identity=eval_identity)
                            export(root,plan,rows)
                    except BaseException as error:
                        for key in keys:
                            if rows[key]['status'] != 'complete':
                                rows[key].update(status='interrupted' if isinstance(error,KeyboardInterrupt)
                                                 else 'failed',error=f'{type(error).__name__}: {error}')
                        export(root,plan,rows)
                        if not args.continue_on_error or not isinstance(error,Exception):
                            raise
                    finally:
                        del model
                        gc.collect()
            except BaseException as error:
                for cell,mass in selected:
                    for k in config.budgets(cell):
                        row=rows[_key(cell,mass,k)]
                        if row['status'] in ('pending','running'):
                            row.update(status='failed',error=f'{type(error).__name__}: {error}')
                export(root,plan,rows)
                if not args.continue_on_error or not isinstance(error,Exception):
                    raise
            finally:
                if bundle is not None:
                    for split in (bundle.train,bundle.validation):
                        mapping=getattr(split.features,'_mmap',None)
                        if mapping is not None:
                            mapping.close()
                del bundle
                gc.collect()
        return export(root,plan,rows)


def _budgets(items):
    result = {}
    for item in items:
        name,values = item.split('=',1)
        if name in result:
            raise ValueError(f'duplicate backbone budget: {name}')
        result[name] = [int(v) for v in values.split(',')]
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/retrieval_mass.yaml')
    parser.add_argument('--workspace',type=Path,default=Path('work/retrieval-mass'))
    parser.add_argument('--source-root',type=Path,default=Path('work/direct-v1'))
    parser.add_argument('--embeddings-dir',type=Path)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--backbones',nargs='+',choices=list(BACKBONES))
    parser.add_argument('--train-k',nargs='+',metavar='BACKBONE=K,K')
    parser.add_argument('--inference-k',nargs='+',metavar='BACKBONE=K,K',help='enables an all-checkpoint inference sweep')
    parser.add_argument('--masses',nargs='+',type=float)
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--ot-weight',type=float)
    parser.add_argument('--ot-eta',type=float)
    parser.add_argument('--ot-microbatch',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--residency',choices=('auto','cuda','cpu','mmap'),default='auto')
    parser.add_argument('--encode-batch-size',type=int,default=1024)
    parser.add_argument('--gallery-chunk',type=int,default=32768)
    parser.add_argument('--query-block',type=int,default=8192)
    parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--checkpoint-every',type=int,default=100)
    parser.add_argument('--reuse-source',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--download-embeddings',action='store_true',help='download cached features only; never extract images or upload results')
    parser.add_argument('--embedding-repo',default=None)
    parser.add_argument('--continue-on-error',action='store_true')
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv)
    payload=MassConfig.load(args.config).__dict__.copy()
    if args.backbones:
        payload['backbones']=args.backbones
        payload['train_k']={b:payload['train_k'].get(b,[8] if b in ('ff2048','rn18') else [12]) for b in args.backbones}
        payload['inference_k']={b:v for b,v in payload['inference_k'].items() if b in args.backbones}
    if args.train_k:
        payload['train_k'].update(_budgets(args.train_k))
    if args.inference_k:
        payload.update(evaluation_mode='sweep',inference_k=_budgets(args.inference_k))
    for name in ('masses','seeds','epochs','lr','ot_weight','ot_eta','ot_microbatch'):
        if getattr(args,name) is not None:
            payload[name]=getattr(args,name)
    config=MassConfig(**payload)
    for name in ('cpu_threads','encode_batch_size','gallery_chunk','query_block','log_every','checkpoint_every'):
        if getattr(args,name)<1:
            parser.error(f'{name} must be positive')
    if args.plan:
        print(json.dumps(config.plan(),indent=2))
        return 0
    args.workspace=args.workspace.resolve()
    args.source_root=args.source_root.resolve()
    args.embeddings_dir=(args.embeddings_dir or args.source_root/'embeddings').resolve()
    args.cache_dir=(args.cache_dir or args.source_root/'cache').resolve()
    if args.download_embeddings and not args.smoke and not args.export_only:
        if args.embeddings_dir != args.source_root/'embeddings':
            parser.error('--download-embeddings uses --source-root/embeddings; omit --embeddings-dir')
        from ..cli import load_dotenv
        from ..embeddings_hub import Hub, DEFAULT_REPO, sync, link_campaign
        load_dotenv('.env')
        pairs=[('imagenet',b) for b in config.backbones]
        hub_root=args.source_root.parent/'embeddings-hub'
        hub=Hub.connect(args.embedding_repo or os.environ.get('MPSAE_EMBEDDINGS_REPO',DEFAULT_REPO))
        sync(pairs,hub_root,args.source_root.parent,hub,download_only=True)
        link_campaign(hub_root,args.source_root,pairs)
    result=run(config,args)
    return 0 if result['complete'] or args.export_only else 1


if __name__=='__main__':
    raise SystemExit(main())
