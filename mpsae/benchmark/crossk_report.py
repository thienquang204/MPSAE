"""Cross-K reports keep train K, inference K and original matched results distinct."""
from pathlib import Path
import statistics
from zipfile import ZipFile,ZIP_DEFLATED

from ..io import read_json,atomic_json
from .matched_report import write_csv,get,gib,NAMES

METRICS=('top1','nnz','query_nnz','retrieval_us','index_gib','retrieval_gpu_gib')
FIELDS=['method','batch_size','seed','train_k','inference_k','status','reused_diagonal',*METRICS,
        'query_nnz0_percent','query_nnz1_percent','query_nnz2_percent','checkpoint_sha256']


def flatten(r):
    h=get(r,'query_stats','nnz_histogram')
    return {**{k:r.get(k) for k in FIELDS},'top1':get(r,'retrieval','top1'),
            'retrieval_us':get(r,'retrieval','timing','microseconds_per_query'),
            'index_gib':gib(get(r,'retrieval','index','index_bytes')),
            'retrieval_gpu_gib':gib(get(r,'retrieval','timing_memory','gpu_peak_allocated_bytes')),
            **{f'query_nnz{i}_percent':100*h[i]/sum(h) if h and len(h)==3 and sum(h) else None for i in range(3)}}


def plot(folder,summary,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'csr':'#0072B2','csrv2':'#009E73','mpsae':'#D55E00'}
    markers={'csr':'o','csrv2':'s','mpsae':'^'}
    cfg=plan['source_config']
    with plt.rc_context({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False}):
        for metric,label in [('top1','1-NN accuracy (%)'),('nnz','Mean active latents'),
                             ('retrieval_us','Retrieval (us/query)'),('index_gib','Index (GiB)')]:
            if not any(r[metric+'_mean'] is not None for r in summary):
                continue
            fig,axes=plt.subplots(len(cfg['batch_sizes']),len(plan['train_k']),
                                  figsize=(3.4*len(plan['train_k']),2.8*len(cfg['batch_sizes'])),squeeze=False)
            for bi,b in enumerate(cfg['batch_sizes']):
                for ti,t in enumerate(plan['train_k']):
                    ax=axes[bi,ti]
                    for m in cfg['methods']:
                        rows=sorted((r for r in summary if (r['method'],r['batch_size'],r['train_k'])==(m,b,t)),key=lambda r:r['inference_k'])
                        ax.plot([r['inference_k'] for r in rows],[r[metric+'_mean'] if r[metric+'_mean'] is not None else np.nan for r in rows],
                                marker=markers[m],color=colors[m],label=NAMES[m],markersize=4)
                        for r in rows:
                            if r[metric+'_std'] is not None:
                                ax.errorbar(r['inference_k'],r[metric+'_mean'],yerr=r[metric+'_std'],color=colors[m],capsize=2)
                    ax.set(xscale='log',xticks=plan['inference_k'],xlabel='Inference K',ylabel=label,title=f'Train K={t}, batch {b}')
                    ax.set_xticklabels([str(k) for k in plan['inference_k']]);ax.minorticks_off()
                    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=7)
            if not plan['scientific']:
                fig.suptitle('SYNTHETIC SMOKE TEST',color='crimson')
            fig.tight_layout()
            for ext in ('pdf','png'):
                fig.savefig(folder/f'{metric}.{ext}',bbox_inches='tight',dpi=300)
            plt.close(fig)


def export(output,plan,source,*,figures,bundle):
    output=Path(output);reports=output/'reports';reports.mkdir(parents=True,exist_ok=True)
    points=[];cfg=plan['source_config']
    for b in cfg['batch_sizes']:
        for m in cfg['methods']:
            for seed in cfg['seeds']:
                for kt in plan['train_k']:
                    for ki in plan['inference_k']:
                        key=f'b{b}-{m}-train{kt}-infer{ki}-s{seed}'
                        points.append(read_json(output/'points'/key/'result.json') or
                                      dict(method=m,batch_size=b,seed=seed,train_k=kt,inference_k=ki,status='pending'))
    rows=[flatten(r) for r in points]
    summary=[]
    for b,m,t,k in sorted({(r['batch_size'],r['method'],r['train_k'],r['inference_k']) for r in rows}):
        selected=[r for r in rows if (r['batch_size'],r['method'],r['train_k'],r['inference_k'])==(b,m,t,k) and r['status']=='complete']
        row=dict(batch_size=b,method=m,train_k=t,inference_k=k,n=len(selected))
        for metric in METRICS:
            values=[r[metric] for r in selected if r[metric] is not None]
            row.update({metric+'_mean':statistics.mean(values) if values else None,
                        metric+'_std':statistics.stdev(values) if len(values)>1 else None,metric+'_n':len(values)})
        summary.append(row)
    write_csv(reports/'crossk.csv',rows,FIELDS)
    write_csv(reports/'summary.csv',summary,['method','batch_size','train_k','inference_k','n']+
              [m+s for m in METRICS for s in ('_mean','_std','_n')])
    # Keep all original 72 matched 1-NN scores and 12 probes; replace overlapping
    # diagonal entries, never duplicate them or average old and new measurements.
    combined={}
    for path in sorted((source/'cells').glob('*/result.json')):
        r=read_json(path)
        if r.get('status')!='complete':
            continue
        base=dict(method=r['method'],batch_size=r['batch_size'],seed=r['seed'],train_k=r['k'],inference_k=r['k'])
        for metric,value in [('1nn',get(r,'retrieval','top1')),('linear_probe',get(r,'probe','fit','top1'))]:
            if value is not None:
                key=(*base.values(),metric)
                combined[key]={**base,'metric':metric,'value':value,'source':'original_matched'}
    for r in rows:
        if r['status']=='complete':
            base={k:r[k] for k in ('method','batch_size','seed','train_k','inference_k')}
            combined[(*base.values(),'1nn')]={**base,'metric':'1nn','value':r['top1'],'source':'crossk'}
    write_csv(reports/'combined_accuracy.csv',list(combined.values()),
              ['method','batch_size','seed','train_k','inference_k','metric','value','source'])
    tex=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Method & Batch & Train $K$ & Infer $K$ & $n$ & 1-NN (\%) \\',r'\midrule']
    for r in summary:
        val=r['top1_mean'];sd=r['top1_std']
        score='--' if val is None else f'{val:.2f}'+(f' $\\pm$ {sd:.2f}' if sd is not None else '')
        tex.append(' & '.join([NAMES[r['method']],str(r['batch_size']),str(r['train_k']),str(r['inference_k']),str(r['n']),score])+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (reports/'crossk.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    result={'complete':sum(r['status']=='complete' for r in rows),'expected':len(rows),
            'combined_accuracy_rows':len(combined),'scientific':plan['scientific'],'results':points}
    atomic_json(output/'results.json',result)
    if figures:
        plot(reports,summary,plan)
    atomic_json(output/'report_status.json',{'complete':result['complete'],
        'plots_complete_at':result['complete'] if figures else (read_json(output/'report_status.json') or {}).get('plots_complete_at'),
        'note':'CSV/JSON refresh each point; ZIP each checkpoint; figures at exit or --report-only'})
    if bundle:
        path=output/'crossk-results.zip';tmp=path.with_suffix('.partial')
        with ZipFile(tmp,'w',ZIP_DEFLATED) as archive:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.suffix in ('.json','.csv','.tex','.pdf','.png'):
                    archive.write(p,p.relative_to(output))
        tmp.replace(path)
    return result
