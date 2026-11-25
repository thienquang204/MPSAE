"""Read-only merge of original matched/cross-K results and the LR extension."""
import math
import statistics

from ..io import atomic_json,read_json
from .matched_report import METRICS,NAMES,flatten,write_csv

KEYS=('backbone','method','batch_size','lr','seed','train_k','inference_k')
GROUPS=tuple(k for k in KEYS if k!='seed')
EVALUATION=('mpsae/model.py','mpsae/evaluate/encode.py','mpsae/evaluate/timing.py',
            'mpsae/benchmark/matched_evaluate.py')
TIMING=('query_batch','gallery_chunk','build_entry_cap','encode_batch_size','warmup','rounds')


def aggregate(rows):
    result=[]
    for key in sorted({tuple(r[k] for k in GROUPS) for r in rows}):
        selected=[r for r in rows if tuple(r[k] for k in GROUPS)==key]
        entry={**dict(zip(GROUPS,key)),'n':len(selected)}
        entry['kind']='matched' if entry['train_k']==entry['inference_k'] else 'cross'
        for metric in METRICS:
            values=[r[metric] for r in selected if r.get(metric) is not None and math.isfinite(r[metric])]
            entry.update({metric+'_mean':statistics.mean(values) if values else None,
                          metric+'_std':statistics.stdev(values) if len(values)>1 else None,
                          metric+'_n':len(values)})
        result.append(entry)
    return result


def formatted(row,metric):
    if not row or row.get(metric+'_mean') is None:return '--'
    value=f'{row[metric+"_mean"]:.2f}'
    sd=row[metric+'_std']
    return value+(f' $\\pm$ {sd:.2f}' if sd is not None else '')


def export_unified(output,plan,records,performance_fields):
    """No checkpoint loading or inference; current LR records win overlaps."""
    root=output.parents[1]
    source=root/'studies'/plan['source_study']
    original=read_json(source/'plan.json')
    folder=output/'reports'
    session=read_json(output/'session.json') or {}
    expected_hashes=session.get('evaluation',{})
    current_runtime=session.get('runtime')
    merged={};skipped=[];conflicts=[];baseline={}
    def compatible(hashes,names):
        return all(hashes.get(n)==expected_hashes[n] for n in names if n in expected_hashes)
    def relative(path):
        return str(path.relative_to(root)).replace('\\','/')
    def add(record,lr,train_k,inference_k,path,origin,comparable):
        row={**flatten(record),'backbone':plan['new_config']['backbone'],
             'lr':lr,'train_k':train_k,'inference_k':inference_k,
             'kind':'matched' if train_k==inference_k else 'cross',
             'performance_comparable':comparable,'origin':origin,'source_path':relative(path),
             'source_study':plan['study_id'] if origin.startswith('lr_') else plan['source_study']}
        if row.get('top1') is None:return
        if train_k!=inference_k:
            # A cross-budget retrieval receipt cannot supply a matched probe score.
            for metric in METRICS:
                if metric.startswith(('probe_','train_')):row[metric]=None
        if not comparable:
            for metric in performance_fields:row[metric]=None
        key=tuple(row[k] for k in KEYS)
        previous=merged.get(key)
        if previous and previous['top1']!=row['top1']:
            conflicts.append({'key':list(key),'previous_source':previous['source_path'],
                              'selected_source':row['source_path'],'previous_top1':previous['top1'],'selected_top1':row['top1']})
        merged[key]=row
    cfg=original['config'] if original else None
    if cfg:
        allowed={(m,b,s,k) for m in cfg['methods'] for b in cfg['batch_sizes'] for s in cfg['seeds'] for k in cfg['k_values']}
        for path in sorted((source/'cells').glob('*/result.json')):
            r=read_json(path)
            key=(r.get('method'),r.get('batch_size'),r.get('seed'),r.get('k'))
            if r.get('status')!='complete' or key not in allowed:continue
            identity=r.get('training_identity',{})
            if identity.get('optimizer',{}).get('lr')!=cfg['lr']:
                skipped.append({'path':relative(path),'reason':'source learning-rate mismatch'});continue
            baseline[key]=r
            evaluation=r.get('identity',{})
            if not compatible(evaluation.get('implementation',{}),(*EVALUATION,'mpsae/evaluate/probe.py')):
                skipped.append({'path':relative(path),'reason':'incompatible original evaluator'});continue
            comparable=(current_runtime is not None and r.get('runtime')==current_runtime
                        and all(evaluation.get('config',{}).get(n)==plan['source_config'].get(n) for n in TIMING))
            add(r,cfg['lr'],r['k'],r['k'],path,'original_matched',comparable)
        for campaign in sorted((root/'crossk').glob('*/plan.json')):
            previous=read_json(campaign)
            if previous.get('source_study')!=plan['source_study']:continue
            for path in sorted((campaign.parent/'points').glob('*/result.json')):
                r=read_json(path)
                if r.get('status')!='complete' or r.get('train_k')==r.get('inference_k'):continue
                b=baseline.get((r.get('method'),r.get('batch_size'),r.get('seed'),r.get('train_k')))
                identity=r.get('identity',{});old_session=identity.get('session',{})
                if (not b or not b.get('checkpoint_sha256')
                    or r.get('checkpoint_sha256')!=b['checkpoint_sha256']
                    or identity.get('data')!=b['training_identity'].get('data')):
                    skipped.append({'path':relative(path),'reason':'cross-K checkpoint/data mismatch'});continue
                if not compatible(old_session.get('implementation',{}),(*EVALUATION,'mpsae/evaluate/knn.py')):
                    skipped.append({'path':relative(path),'reason':'incompatible cross-K evaluator'});continue
                comparable=(current_runtime is not None and old_session.get('runtime')==current_runtime
                            and all(previous.get('source_config',{}).get(n)==plan['source_config'].get(n) for n in TIMING))
                add(r,cfg['lr'],r['train_k'],r['inference_k'],path,'previous_cross',comparable)
    # This extension's copies override old entries, with the same logical key.
    for r in records:
        if r.get('status')!='complete':continue
        if r['kind']=='matched':
            filename=f'{r["condition"]}-k{r["train_k"]}-s{r["seed"]}.json'
            path=output/'matched'/filename
        else:
            filename=f'{r["condition"]}-train{r["train_k"]}-infer{r["inference_k"]}-s{r["seed"]}.json'
            path=output/'cross'/filename
        add(r,r['lr'],r['train_k'],r['inference_k'],path,'lr_'+r['kind'],bool(r.get('performance_comparable')))
    rows=[merged[k] for k in sorted(merged)]
    summary=aggregate(rows)
    fields=[*KEYS,'kind','epochs','steps','ot_weight','performance_comparable',*METRICS,'checkpoint_sha256','origin','source_study','source_path']
    write_csv(folder/'unified_per_seed.csv',rows,fields)
    write_csv(folder/'unified_summary.csv',summary,[*GROUPS,'kind','n']+[m+s for m in METRICS for s in ('_mean','_std','_n')])
    probes=[r for r in rows if r.get('probe_top1') is not None]
    write_csv(folder/'unified_linear_probe.csv',probes,fields)
    conditions={(m,b,cfg['lr']) for m in cfg['methods'] for b in cfg['batch_sizes']} if cfg else set()
    nc=plan['new_config']
    conditions.update((m,b,nc['lr']) for m in nc['methods'] for b in nc['batch_sizes'])
    conditions.update((r['method'],r['batch_size'],r['lr']) for r in rows)
    ks=sorted(set((cfg or nc)['k_values'])|set(nc['k_values']))
    lookup={(r['method'],r['batch_size'],r['lr'],r['train_k'],r['inference_k']):r for r in summary}
    table=[]
    tex=[r'\begin{tabular}{lrr'+('r'*len(ks))+'}',r'\toprule',
         'Method & Batch & LR & '+' & '.join(f'$K={k}$' for k in ks)+r' \\',r'\midrule']
    probe_tex=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Method & Batch & LR & $K$ & $n$ & Probe (\%) \\',r'\midrule']
    for m,b,lr in sorted(conditions,key=lambda c:(c[1],c[0],c[2])):
        entry={'method':m,'batch_size':b,'lr':lr}
        values=[]
        for k in ks:
            r=lookup.get((m,b,lr,k,k));value=formatted(r,'top1')
            entry[f'k{k}_accuracy']=value.replace(' $\\pm$ ',' +/- ')
            entry[f'k{k}_n']=r['top1_n'] if r else 0
            values.append(value)
        table.append(entry)
        tex.append(f'{NAMES[m]} & {b} & {lr:.0e} & '+' & '.join(values)+r' \\')
        k=plan['source_config']['probe_k'];r=lookup.get((m,b,lr,k,k))
        probe_tex.append(f'{NAMES[m]} & {b} & {lr:.0e} & {k} & {r["probe_top1_n"] if r else 0} & '+formatted(r,'probe_top1')+r' \\')
    write_csv(folder/'unified_matched_table.csv',table,['method','batch_size','lr']+[f'k{k}_{s}' for k in ks for s in ('accuracy','n')])
    for name,lines in [('unified_matched',tex),('unified_linear_probe',probe_tex)]:
        (folder/f'{name}.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n',encoding='utf-8')
    expected_old=(len(cfg['methods'])*len(cfg['batch_sizes'])*len(cfg['seeds'])*len(cfg['k_values'])) if cfg else None
    result={'original_source_available':bool(cfg),'source_study':plan['source_study'],
            'unique_1nn_scores':len(rows),'matched_1nn_scores':sum(r['kind']=='matched' for r in rows),
            'cross_1nn_scores':sum(r['kind']=='cross' for r in rows),'probe_scores':len(probes),
            'expected_matched_1nn':expected_old+plan['new_fits'] if expected_old is not None else None,
            'skipped':skipped,'duplicate_accuracy_conflicts':conflicts,
            'note':'Read-only union of available results; no missing evaluation is launched. LR receipts win overlapping points; matched receipts supply diagonal/probe scores.'}
    atomic_json(folder/'unified_status.json',result)
    return result
