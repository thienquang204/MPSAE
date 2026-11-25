"""Local CSV/LaTeX tables, paired seed differences and vector figures."""
import csv
import io
import math
from pathlib import Path
import statistics
from zipfile import ZipFile, ZIP_DEFLATED

from ..io import atomic_json, read_json

METRICS=('top1','nnz','query_nnz','train_seconds','train_gpu_gib','train_rss_gib',
         'retrieval_us','retrieval_gpu_gib','index_gib','encoding_seconds',
         'probe_top1','probe_train_seconds','probe_fit_gpu_gib','probe_us','probe_inference_gpu_gib')
NAMES={'csr':'CSR','csrv2':'CSRv2','mpsae':'MPSAE'}


def get(record,*keys):
    for key in keys:
        record=record.get(key) if isinstance(record,dict) else None
    return record


def gib(value):
    return value/1024**3 if value is not None else None


def write_csv(path,rows,fields):
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    temporary=path.with_suffix('.partial')
    temporary.write_text(buffer.getvalue(),encoding='utf-8')
    temporary.replace(path)


def flatten(record):
    profiles=record.get('training_profiles',[])
    def peak(name):
        values=[p[name] for p in profiles if p.get(name) is not None]
        return max(values) if values and record.get('train_memory_complete') else None
    return {**{k:record.get(k) for k in ('method','k','seed','batch_size','status','fit_id','checkpoint_sha256','error')},
            'top1':get(record,'retrieval','top1'),'nnz':record.get('nnz'),'query_nnz':record.get('query_nnz'),
            'train_seconds':record.get('train_seconds') if record.get('train_memory_complete') else None,'steps':record.get('steps'),
            'epochs':get(record,'training_identity','schedule','epochs'),
            'lr':get(record,'training_identity','optimizer','lr'),
            'ot_weight':get(record,'training_identity','objective','ot','weight'),
            'train_gpu_gib':gib(peak('gpu_peak_allocated_bytes')),
            'train_rss_gib':gib(peak('rss_peak_sampled_bytes')),
            'retrieval_us':get(record,'retrieval','timing','microseconds_per_query'),
            'retrieval_gpu_gib':gib(get(record,'retrieval','timing_memory','gpu_peak_allocated_bytes')),
            'index_gib':gib(get(record,'retrieval','index','index_bytes')),
            'encoding_seconds':get(record,'encoding','wall_seconds'),
            'probe_top1':get(record,'probe','fit','top1'),
            'probe_train_seconds':get(record,'probe','fit','train_seconds'),
            'probe_fit_gpu_gib':gib(get(record,'probe','fit_and_validation_memory','gpu_peak_allocated_bytes')),
            'probe_us':get(record,'probe','timing','microseconds_per_query'),
            'probe_inference_gpu_gib':gib(get(record,'probe','inference_memory','gpu_peak_allocated_bytes'))}


def summarise(rows):
    result=[]
    for batch,k,method in sorted({(r['batch_size'],r['k'],r['method']) for r in rows}):
        selected=[r for r in rows if (r['batch_size'],r['k'],r['method'])==(batch,k,method) and r['status']=='complete']
        row={'batch_size':batch,'k':k,'method':method,'n':len(selected)}
        for name in METRICS:
            values=[r[name] for r in selected if r.get(name) is not None and math.isfinite(r[name])]
            row[name+'_n']=len(values)
            row[name+'_mean']=statistics.mean(values) if values else None
            row[name+'_std']=statistics.stdev(values) if len(values)>1 else None
        result.append(row)
    return result


def plots(folder,summary,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'csr':'#0072B2','csrv2':'#009E73','mpsae':'#D55E00'}
    markers={'csr':'o','csrv2':'s','mpsae':'^'}
    batches=plan['config']['batch_sizes']
    def save(fig,name):
        if not plan['scientific']:
            fig.suptitle('SYNTHETIC SMOKE TEST',fontsize=11,color='crimson')
        fig.tight_layout()
        for ext in ('pdf','png'):
            fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight',dpi=300)
        plt.close(fig)
    with plt.rc_context({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False}):
        for metric,label in [('top1','1-NN accuracy (%)'),('nnz','Mean active latents'),('retrieval_us','Retrieval (us/query)'),
                             ('train_seconds','SAE training (s)'),('train_gpu_gib','Peak training VRAM (GiB)'),
                             ('index_gib','Retrieval index (GiB)'),('retrieval_gpu_gib','Peak retrieval VRAM (GiB)')]:
            if not any(r[metric+'_mean'] is not None for r in summary):
                continue
            fig,axes=plt.subplots(1,len(batches),figsize=(3.4*len(batches),2.8),squeeze=False)
            for ax,batch in zip(axes[0],batches):
                for method in plan['config']['methods']:
                    selected=sorted((r for r in summary if r['batch_size']==batch and r['method']==method),key=lambda r:r['k'])
                    vals=[r[metric+'_mean'] if r[metric+'_mean'] is not None else np.nan for r in selected]
                    if not any(np.isfinite(vals)):
                        continue
                    # A single completed seed has no error bar, not a fabricated zero std.
                    ax.plot([r['k'] for r in selected],vals,color=colors[method],marker=markers[method],label=NAMES[method],markersize=4)
                    for r in selected:
                        if r[metric+'_std'] is not None:
                            ax.errorbar(r['k'],r[metric+'_mean'],yerr=r[metric+'_std'],color=colors[method],capsize=2)
                ax.set(xscale='log',xticks=plan['config']['k_values'],xlabel='K',ylabel=label,title=f'Train batch {batch}')
                ax.set_xticklabels([str(k) for k in plan['config']['k_values']])
                ax.minorticks_off()
                ax.grid(axis='y',alpha=.2)
                if ax.lines:
                    ax.legend(frameon=False,fontsize=8)
            save(fig,metric)
        for cost,label in [('retrieval_us','Retrieval (us/query)'),('index_gib','Retrieval index (GiB)')]:
            if not any(r[cost+'_mean'] is not None for r in summary):
                continue
            fig,axes=plt.subplots(1,len(batches),figsize=(3.4*len(batches),2.8),squeeze=False)
            for ax,batch in zip(axes[0],batches):
                for method in plan['config']['methods']:
                    selected=sorted((r for r in summary if r['batch_size']==batch and r['method']==method and
                                     r[cost+'_mean'] is not None and r['top1_mean'] is not None),key=lambda r:r['k'])
                    if selected:
                        ax.plot([r[cost+'_mean'] for r in selected],[r['top1_mean'] for r in selected],
                                color=colors[method],marker=markers[method],label=NAMES[method],markersize=4)
                ax.set(xlabel=label,ylabel='1-NN accuracy (%)',title=f'Train batch {batch}')
                if ax.lines:
                    ax.legend(frameon=False,fontsize=8)
            save(fig,'accuracy_vs_'+cost)
        fig,axes=plt.subplots(1,3,figsize=(10,2.8))
        for ax,metric,label in zip(axes,('probe_top1','probe_us','probe_inference_gpu_gib'),
                                   ('Probe accuracy (%)','Inference (us/query)','Peak inference VRAM (GiB)')):
            for bi,batch in enumerate(batches):
                for mi,method in enumerate(plan['config']['methods']):
                    row=next((r for r in summary if r['batch_size']==batch and r['method']==method and r['k']==plan['config']['probe_k']),None)
                    if row and row[metric+'_mean'] is not None:
                        ax.bar(bi*4+mi,row[metric+'_mean'],color=colors[method],width=.8,
                               yerr=row[metric+'_std'],capsize=2)
            ax.set(xticks=[i*4+1 for i in range(len(batches))],xticklabels=[f'Batch {b}' for b in batches],
                   ylabel=label,title=f'Probe K={plan["config"]["probe_k"]}')
            if not ax.patches:
                ax.text(.5,.5,'No measurement',transform=ax.transAxes,ha='center',va='center',color='0.4')
        from matplotlib.patches import Patch
        axes[-1].legend(handles=[Patch(color=colors[m],label=NAMES[m]) for m in plan['config']['methods']],frameon=False,fontsize=8)
        save(fig,'linear_probe')


def export(folder,plan,*,plots=True,bundle=True):
    folder=Path(folder)
    records=[]
    for recipe in plan['recipes']:
        key=f'b{recipe["batch_size"]}-{recipe["method"]}-k{recipe["k"]}-s{recipe["seed"]}'
        record=read_json(folder/'cells'/key/'result.json') or {**recipe,'status':'pending'}
        records.append(record)
    rows=[flatten(r) for r in records]
    summary=summarise(rows)
    reports=folder/'reports'
    reports.mkdir(parents=True,exist_ok=True)
    raw_fields=['method','k','seed','batch_size','status','fit_id','checkpoint_sha256','steps','epochs','lr','ot_weight',*METRICS,'error']
    write_csv(reports/'results.csv',rows,raw_fields)
    fields=['method','k','batch_size','n']+[m+s for m in METRICS for s in ('_mean','_std','_n')]
    write_csv(reports/'summary.csv',summary,fields)
    long=[]
    for row in rows:
        for metric in ('top1','probe_top1') if row['k']==plan['config']['probe_k'] else ('top1',):
            long.append({**row,'metric':metric,'value':row.get(metric)})
    write_csv(reports/'accuracy.csv',long,['method','k','seed','batch_size','status','metric','value'])
    paired=[]
    for baseline in ('csr','csrv2'):
        for b,k in sorted({(r['batch_size'],r['k']) for r in rows}):
            for metric in ('top1','probe_top1'):
                values=[]
                for seed in plan['config']['seeds']:
                    a=next((r.get(metric) for r in rows if r['status']=='complete' and (r['method'],r['seed'],r['k'],r['batch_size'])==('mpsae',seed,k,b)),None)
                    c=next((r.get(metric) for r in rows if r['status']=='complete' and (r['method'],r['seed'],r['k'],r['batch_size'])==(baseline,seed,k,b)),None)
                    if a is not None and c is not None:
                        values.append(a-c)
                if values:
                    paired.append({'baseline':baseline,'batch_size':b,'k':k,'metric':metric,'n':len(values),
                                   'delta_mean':statistics.mean(values),'delta_std':statistics.stdev(values) if len(values)>1 else None})
    write_csv(reports/'paired_differences.csv',paired,['baseline','batch_size','k','metric','n','delta_mean','delta_std'])
    def fmt(row,key):
        value=row.get(key+'_mean'); std=row.get(key+'_std')
        return '--' if value is None else f'{value:.2f}'+(f' $\\pm$ {std:.2f}' if std is not None else '')
    for name,chosen,metrics,labels in (
        ('retrieval',summary,('top1','retrieval_us','index_gib','train_gpu_gib','train_seconds'),
         ['1-NN (\\%)','Query ($\\mu$s)','Index (GiB)','Train VRAM (GiB)','Train (s)']),
        ('linear_probe',[r for r in summary if r['k']==plan['config']['probe_k']],
         ('probe_top1','probe_us','probe_inference_gpu_gib','probe_train_seconds'),
         ['Top-1 (\\%)','Inference ($\\mu$s)','VRAM (GiB)','Fit (s)'])):
        header=['Batch','$K$','Method','$n$',*labels]
        lines=['% Mean +/- sample std across seeds. -- is unavailable; consult *_n in CSV.',
               '% SYNTHETIC SMOKE ONLY' if not plan['scientific'] else '% Original standalone measurements',
               r'\begin{tabular}{'+'lllr'+'r'*len(metrics)+'}',r'\toprule',' & '.join(header)+r' \\',r'\midrule']
        lines+=[' & '.join([str(r['batch_size']),str(r['k']),NAMES[r['method']],str(r['n']),*[fmt(r,m) for m in metrics]])+r' \\' for r in chosen]
        lines += [r'\bottomrule',r'\end{tabular}','']
        (reports/f'{name}.tex').write_text('\n'.join(lines),encoding='utf-8')
    result={'directory':str(folder),'expected':plan['sae_fits'],
            'complete':sum(r['status']=='complete' for r in rows),'accuracy_evaluations':len(long),
            'scientific':plan['scientific'],'results':records}
    atomic_json(folder/'results.json',result)
    if plots and any(r['status']=='complete' for r in rows):
        globals()['plots'](reports,summary,plan)
    if bundle:
        archive=folder/'matched-results.zip'
        temporary=archive.with_suffix('.partial')
        with ZipFile(temporary,'w',ZIP_DEFLATED) as z:
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix in ('.json','.csv','.tex','.pdf','.png'):
                    z.write(path,path.relative_to(folder))
        temporary.replace(archive)
        result['archive']=str(archive)
    return result
