"""Tables and figures for four matched-batch learning-rate conditions."""
from pathlib import Path
import statistics
from zipfile import ZipFile,ZIP_DEFLATED

from ..io import read_json,atomic_json
from .matched_report import flatten,write_csv,METRICS

NAMES={'csr_lr4e-5':'CSR (4e-5)','csrv2_lr4e-5':'CSRv2 (4e-5)',
       'mpsae_lr4e-5':'MPSAE (4e-5)','mpsae_lr8e-5':'MPSAE (8e-5)'}
COLORS=dict(zip(NAMES,('#0072B2','#009E73','#D55E00','#CC79A7')))
IDS=['condition','method','lr','batch_size','seed','train_k','inference_k','kind','status','performance_comparable']
LOSS_FIELDS=('loss_total','loss_mse_k','loss_mse_4k','loss_ot')
PERFORMANCE=('train_seconds','train_gpu_gib','train_rss_gib','retrieval_us','retrieval_gpu_gib',
             'encoding_seconds','probe_train_seconds','probe_fit_gpu_gib','probe_us','probe_inference_gpu_gib')


def summaries(rows):
    result=[]
    keys=('condition','kind','train_k','inference_k')
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected=[r for r in rows if tuple(r[k] for k in keys)==key and r['status']=='complete']
        out={**dict(zip(keys,key)),'n':len(selected)}
        for metric in METRICS:
            values=[r[metric] for r in selected if r.get(metric) is not None]
            out.update({metric+'_n':len(values),metric+'_mean':statistics.mean(values) if values else None,
                        metric+'_std':statistics.stdev(values) if len(values)>1 else None})
        result.append(out)
    return result


def paired(rows):
    pairs=[]
    old={(r['seed'],r['train_k'],r['inference_k']):r for r in rows if r['condition']=='mpsae_lr4e-5'}
    for r in rows:
        baseline=old.get((r['seed'],r['train_k'],r['inference_k']))
        if r['condition']!='mpsae_lr8e-5' or not baseline:continue
        for metric in ('top1','probe_top1'):
            if r.get(metric) is not None and baseline.get(metric) is not None:
                pairs.append({**{k:r[k] for k in ('seed','train_k','inference_k','kind')},'metric':metric,
                              'delta_pp':r[metric]-baseline[metric]})
    aggregated=[]
    for t,k,m in sorted({(r['train_k'],r['inference_k'],r['metric']) for r in pairs}):
        vals=[r['delta_pp'] for r in pairs if (r['train_k'],r['inference_k'],r['metric'])==(t,k,m)]
        aggregated.append(dict(train_k=t,inference_k=k,metric=m,n=len(vals),mean_delta_pp=statistics.mean(vals),
                               std_delta_pp=statistics.stdev(vals) if len(vals)>1 else None))
    return pairs,aggregated


def render_figures(folder,summary,losses,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        if not plan['scientific']:fig.suptitle('SYNTHETIC SMOKE TEST',color='crimson')
        fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight',dpi=300)
        plt.close(fig)
    def lines(ax,rows,metric,x):
        for name in NAMES:
            selected=sorted((r for r in rows if r['condition']==name and r[metric+'_mean'] is not None),key=lambda r:r[x])
            if not selected:continue
            ax.plot([r[x] for r in selected],[r[metric+'_mean'] for r in selected],marker='o',
                    markersize=3,color=COLORS[name],label=NAMES[name])
            for r in selected:
                if r[metric+'_std'] is not None:
                    ax.errorbar(r[x],r[metric+'_mean'],yerr=r[metric+'_std'],color=COLORS[name],capsize=2)
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=7)
    with plt.rc_context({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False}):
        matched=[r for r in summary if r['kind']=='matched']
        for metric,label in [('top1','1-NN accuracy (%)'),('query_nnz','Mean query nnz'),
                              ('retrieval_us','Retrieval (us/query)'),('index_gib','Index (GiB)'),
                              ('retrieval_gpu_gib','Retrieval peak VRAM (GiB)'),
                              ('train_seconds','SAE training (s)'),('train_gpu_gib','Training peak VRAM (GiB)'),
                              ('probe_top1','Linear probe accuracy (%)'),('probe_us','Probe inference (us/query)'),
                              ('probe_inference_gpu_gib','Probe peak VRAM (GiB)')]:
            if not any(r[metric+'_mean'] is not None for r in matched):continue
            fig,ax=plt.subplots(figsize=(4.5,3.1))
            if metric.startswith('probe_'):
                selected=[r for r in matched if r[metric+'_mean'] is not None]
                for i,r in enumerate(selected):
                    ax.bar(i,r[metric+'_mean'],color=COLORS[r['condition']],width=.65)
                    if r[metric+'_std'] is not None:ax.errorbar(i,r[metric+'_mean'],yerr=r[metric+'_std'],color='black',capsize=3)
                ax.set_xticks(range(len(selected)),[NAMES[r['condition']] for r in selected],rotation=20,ha='right')
                ax.set_title(f'Train/inference K={plan["source_config"]["probe_k"]}')
            else:
                lines(ax,matched,metric,'train_k')
                ks=plan['source_config']['k_values']
                ax.set(xscale='log',xticks=ks,xlabel='Train / inference K');ax.set_xticklabels(ks);ax.minorticks_off()
            ax.set_ylabel(label);save(fig,'matched_'+metric)
        anchors=plan['train_anchors']
        fig,axes=plt.subplots(1,len(anchors),figsize=(4*len(anchors),3.1),squeeze=False)
        for ax,k in zip(axes[0],anchors):
            selected=[r for r in summary if r['train_k']==k and r['inference_k'] in plan['inference_k']]
            lines(ax,selected,'top1','inference_k')
            ax.set(xscale='log',xticks=plan['inference_k'],xlabel='Inference K',ylabel='1-NN accuracy (%)',title=f'Train K={k}')
            ax.set_xticklabels(plan['inference_k']);ax.minorticks_off()
        save(fig,'crossk_accuracy')
        ks=plan['source_config']['k_values'];cols=min(3,len(ks));rows=(len(ks)+cols-1)//cols
        for metric in LOSS_FIELDS:
            if not any(r.get(metric) is not None for r in losses):continue
            fig,axes=plt.subplots(rows,cols,figsize=(3.7*cols,2.8*rows),squeeze=False)
            for ax,k in zip(axes.flat,ks):
                for name in ('mpsae_lr4e-5','mpsae_lr8e-5'):
                    for i,seed in enumerate(plan['source_config']['seeds']):
                        samples=sorted((r for r in losses if r['condition']==name and r['train_k']==k and r['seed']==seed
                                        and r.get(metric) is not None),key=lambda r:r['step'])
                        if samples:ax.plot([r['epoch_progress'] for r in samples],[r[metric] for r in samples],
                                            color=COLORS[name],linestyle='-' if i==0 else '--',linewidth=1,marker='.',markersize=2,
                                            label=f'{NAMES[name]}, seed {seed}')
                ax.set(xlabel='Epoch progress',ylabel=metric.replace('loss_','').replace('_',' '),title=f'Train K={k}')
                ax.grid(alpha=.2);ax.legend(fontsize=6,frameon=False)
            for ax in list(axes.flat)[len(ks):]:ax.set_visible(False)
            save(fig,metric)


def export(output,plan,*,figures=False):
    folder=output/'reports';folder.mkdir(parents=True,exist_ok=True)
    records=[read_json(p) for sub in ('matched','cross') for p in sorted((output/sub).glob('*.json'))]
    rows=[]
    for r in records:
        row={**flatten(r),**{k:r.get(k) for k in IDS}}
        if not r.get('performance_comparable'):
            # Historical accuracy is valid, but speed/peak-memory measurements
            # from a different machine/runtime must not imply a fair comparison.
            for metric in PERFORMANCE:row[metric]=None
        rows.append(row)
    summary=summaries(rows)
    write_csv(folder/'accuracy_per_seed.csv',rows,[*IDS,*METRICS,'checkpoint_sha256'])
    write_csv(folder/'summary.csv',summary,['condition','kind','train_k','inference_k','n']+
              [m+s for m in METRICS for s in ('_mean','_std','_n')])
    matched=[r for r in rows if r['kind']=='matched']
    write_csv(folder/'matched_accuracy.csv',matched,[*IDS,*METRICS])
    cross=[r for r in rows if r['train_k'] in plan['train_anchors'] and r['inference_k'] in plan['inference_k']]
    write_csv(folder/'crossk_accuracy.csv',cross,[*IDS,*METRICS])
    write_csv(folder/'linear_probe.csv',[r for r in matched if r.get('probe_top1') is not None],
              [*IDS,'probe_top1','probe_us','probe_train_seconds','probe_fit_gpu_gib','probe_inference_gpu_gib'])
    pairs,deltas=paired(rows)
    write_csv(folder/'paired_deltas.csv',pairs,['seed','train_k','inference_k','kind','metric','delta_pp'])
    write_csv(folder/'paired_summary.csv',deltas,['train_k','inference_k','metric','n','mean_delta_pp','std_delta_pp'])
    losses=[];missing=[]
    for path in sorted((output/'losses').glob('*.json')):
        record=read_json(path)
        if not record['available']:missing.append(path.stem)
        # If a resumed log repeats a step, retain its last recorded sample.
        samples={r['step']:r for r in record['samples']}
        losses.extend({**r,**{k:record[k] for k in ('condition','train_k','seed')}} for r in samples.values())
    write_csv(folder/'loss_samples.csv',losses,['condition','train_k','seed','step','epoch_progress',*LOSS_FIELDS,'ot_weight'])
    for kind,subset in [('matched',[r for r in summary if r['kind']=='matched']),
                        ('crossk',[r for r in summary if r['train_k'] in plan['train_anchors'] and r['inference_k'] in plan['inference_k']])]:
        tex=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Condition & Train $K$ & Infer $K$ & $n$ & 1-NN (\%) \\',r'\midrule']
        for r in subset:
            value=r['top1_mean'];sd=r['top1_std']
            text='--' if value is None else f'{value:.2f}'+(f' $\\pm$ {sd:.2f}' if sd is not None else '')
            tex.append(f"{NAMES[r['condition']]} & {r['train_k']} & {r['inference_k']} & {r['n']} & {text}"+r' \\')
        (folder/f'{kind}.tex').write_text('\n'.join(tex+[r'\bottomrule',r'\end{tabular}'])+'\n',encoding='utf-8')
    tex=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Condition & $K$ & $n$ & Probe (\%) & Time ($\mu$s/query) \\',r'\midrule']
    for r in summary:
        if r['kind']!='matched' or r['probe_top1_mean'] is None:continue
        def formatted(metric):
            val=r[metric+'_mean'];sd=r[metric+'_std']
            return '--' if val is None else f'{val:.2f}'+(f' $\\pm$ {sd:.2f}' if sd is not None else '')
        tex.append(f"{NAMES[r['condition']]} & {r['train_k']} & {r['probe_top1_n']} & {formatted('probe_top1')} & {formatted('probe_us')}"+r' \\')
    (folder/'linear_probe.tex').write_text('\n'.join(tex+[r'\bottomrule',r'\end{tabular}'])+'\n',encoding='utf-8')
    from .unified_report import export_unified
    unified=export_unified(output,plan,records,PERFORMANCE)
    result={'complete_1nn':sum(r.get('top1') is not None for r in rows),'expected_1nn':plan['expected_unique_1nn'],
            'probe_scores':sum(r.get('probe_top1') is not None for r in rows),
            'expected_probe_scores':plan['expected_probe_scores'],'missing_loss_logs':missing,
            'scientific':plan['scientific'],'directory':str(output),'unified':unified}
    atomic_json(output/'results.json',result)
    if figures:render_figures(folder,summary,losses,plan)
    old=read_json(output/'report_status.json') or {}
    atomic_json(output/'report_status.json',{'complete_1nn':result['complete_1nn'],
        'figures_complete_at':result['complete_1nn'] if figures else old.get('figures_complete_at'),
        'note':'Tables/ZIP refresh per completed checkpoint or inference point; figures at exit/report-only. Losses are logged minibatch samples.'})
    path=output/'lr-ablation-results.zip';temporary=path.with_suffix('.partial')
    with ZipFile(temporary,'w',ZIP_DEFLATED) as archive:
        for p in sorted(output.glob('*.json')):archive.write(p,p.relative_to(output))
        for sub in ('reports','matched','cross','losses'):
            for p in sorted((output/sub).glob('*')):
                if p.is_file() and p.suffix in ('.csv','.json','.tex','.pdf','.png'):archive.write(p,p.relative_to(output))
    temporary.replace(path)
    return result
