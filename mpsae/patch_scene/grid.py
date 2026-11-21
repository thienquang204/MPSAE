"""MMPOT-weight ablation: separate MPSAE fits, shared baseline caches, explicit tables."""
import csv
from dataclasses import replace
import math
from pathlib import Path

from ..io import atomic_json
from ..transport import solver_name
from .report import expected_keys, NAMES, DATASET_NAMES


def grid_configs(config, weights):
    if (not weights or any(isinstance(w,bool) or not math.isfinite(w) or w<=0 for w in weights)
            or len(set(weights))!=len(weights)):
        raise ValueError('OT weights must be distinct finite positive numbers')
    if config.evaluation_mode!='matched' or any(k>64 for k in config.training_ks):
        raise ValueError('This grid supports matched K<=64; larger K has a separate loss-weight policy')
    return [replace(config,mpsae_ot_weight=float(w)) for w in sorted(weights)]


def grid_plan(config, grid):
    cells=len(config.encoder_names)*len(config.dataset_names)
    n=cells*len(config.training_ks)
    weights=[c.mpsae_ot_weight for c in grid]
    sparse=n*(len([m for m in config.methods if m!='mpsae']) + (len(grid) if 'mpsae' in config.methods else 0))
    dense=cells if config.include_dense else 0
    return {**config.plan(), 'ot_weights':weights, 'mpsae_ot_weights':{e:weights for e in config.encoder_names},
            'sae_fits':sparse,
            'sparse_evaluations':sparse, 'matched_evaluations':sparse,
            'cross_k_evaluations':0, 'dense_evaluations':dense,
            'total_evaluations':sparse+dense,
            'grid_campaigns':[c.plan() for c in grid],
            'count_scope':'unique fits/evaluations; dense, CSR and CSRv2 caches reused across weights'}


def write_grid_tables(receipts, configs, weights, directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    weights=sorted(float(w) for w in weights)
    expected=set()
    for config in configs.values():
        for key in expected_keys(config):
            for weight in weights if key[2]=='mpsae' else [None]:
                expected.add((*key,weight))
    lookup={}; campaigns=[]
    for config,dataset,receipt in receipts:
        weight=config.mpsae_ot_weight
        if dataset not in configs or config.plan()['campaign'] not in {
                c.plan()['campaign'] for c in grid_configs(configs[dataset], weights)}:
            raise ValueError('Receipt settings differ from the requested weight grid')
        campaigns.append({'dataset':dataset,'weight':weight,'mass':config.mpsae_ot_mass,
                          'solver':solver_name(config.mpsae_ot_mass),'campaign':config.plan()['campaign']})
        for row in receipt['results']:
            w=weight if row['method']=='mpsae' else None
            key=(row['dataset'],row['encoder'],row['method'],row.get('train_k'),row.get('inference_k'),w)
            if key not in expected or not row.get('complete'):
                raise ValueError('Unexpected/incomplete row in MMPOT grid')
            old=lookup.get(key)
            if old is not None:
                for field in ('identity','miou','pixel_accuracy','mean_nnz','gallery_mean_nnz'):
                    if old.get(field)!=row.get(field):
                        raise ValueError(f'Inconsistent shared baseline/result: {key}, {field}')
            lookup[key]={**row,'mmpot_weight':w,
                         'mmpot_mass':config.mpsae_ot_mass if w is not None else None,
                         'solver':solver_name(config.mpsae_ot_mass) if w is not None else None,
                         'seed':config.seed}
    ordered=sorted(lookup,key=lambda key:tuple('' if x is None else str(x) for x in key))
    rows=[lookup[key] for key in ordered]
    atomic_json(directory/'grid_results.json',{'complete':len(rows)==len(expected),
        'completed':len(rows),'expected':len(expected),'weights':weights,'campaigns':campaigns,
        'selection':'Report all weights. No best-weight selection on NYUv2 official test.', 'results':rows})
    fields=['dataset','encoder','method','train_k','inference_k','seed','mmpot_mass','mmpot_weight','solver','miou_percent',
            'pixel_accuracy_percent','mean_nnz','gallery_mean_nnz']
    with (directory/'grid_results.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({**{f:row.get(f) for f in fields},'miou_percent':100*row['miou'],
                             'pixel_accuracy_percent':100*row['pixel_accuracy']})
    comparisons=[]
    for dataset,config in configs.items():
        for encoder in config.encoder_names:
            for k in config.training_ks:
                out={'dataset':dataset,'encoder':encoder,'k':k,'seed':config.seed,'mmpot_mass':config.mpsae_ot_mass}
                for method in ('dense','csr','csrv2','mpsae'):
                    for w in weights if method=='mpsae' else [None]:
                        train=infer=None if method=='dense' else k
                        row=lookup.get((dataset,encoder,method,train,infer,w))
                        name=f'mpsae_lambda_{w:g}' if w is not None else method
                        out[name]=100*row['miou'] if row else None
                comparisons.append(out)
    with (directory/'grid_comparisons.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    lines=[r'% Requires \usepackage{booktabs}; scores are mIoU (%).',r'\begin{table*}[t]',r'\centering',
        r'\caption{Transport-loss weight ablation at matched training/inference sparsity. $s=1$ uses balanced MMOT; $s<1$ uses partial MMPOT. All weights are reported; no test-set model selection.}',
        r'\begin{tabular}{llrr'+'r'*(3+len(weights))+'}',r'\toprule',
        'Dataset & Encoder & $K$ & $s$ & Dense & CSR & CSRv2 & '+' & '.join(f'MPSAE ($\\lambda={w:g}$)' for w in weights)+r' \\',r'\midrule']
    for row in comparisons:
        scores=[row[f] for f in ['dense','csr','csrv2',*[f'mpsae_lambda_{w:g}' for w in weights]]]
        lines.append(' & '.join([DATASET_NAMES[row['dataset']],NAMES[row['encoder']],str(row['k']),f"{row['mmpot_mass']:g}",
                                *['--' if s is None else f'{s:.2f}' for s in scores]])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
        r'\par\smallskip\footnotesize VOC and Cityscapes: validation; NYUv2: official test. Single seed; CSRv2 uses training labels. Shared support bank; baselines evaluated once. -- denotes unavailable results.',r'\end{table*}']
    (directory/'grid_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
