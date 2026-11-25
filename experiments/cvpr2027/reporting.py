"""Paper tables and figures generated only from explicitly indexed evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from common import HERE, digest, read_json, read_yaml, write_csv, write_json, write_text
from registry import SELECTORS, table_conditions

NAMES = {'ff2048':'ResNet-50','vit_b16':'ViT-B/16','convnext_tiny':'ConvNeXt-T',
         'dinov2_vits14':'DINOv2-S/14','siglip2':'SigLIP 2 Base','dinov2':'DINOv2 Base',
         'csr':'CSR','csrv2':'CSRv2','mpsae':'MPSAE','dense':'Dense','voc':'VOC','nyuv2':'NYUv2'}
FIELDS = ['condition_id',*SELECTORS,'metric','value','status','candidate_id','metadata_level',
          'pixel_accuracy_percent','gallery_mean_nnz','query_mean_nnz','empty_queries',
          'epochs_completed','steps','query_nnz_histogram','checkpoint_sha256','training_signature','sources']


def flat(entry):
    return {**entry, **(entry.get('result') or {}), 'status':entry['status']}


def fmt(value, places=2):
    return '--' if value is None else f'{value:.{places}f}'


def count(value):
    return '--' if value is None else str(value)


def esc(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')


def score(row):
    return row.get('value') if row and row.get('status')=='complete' else None


def ranked(values):
    rounded = [None if x is None else round(x,2) for x in values]
    levels = sorted({x for x in rounded if x is not None},reverse=True)
    complete = all(x is not None for x in rounded)
    return [('\\textbf{'+fmt(v)+'}' if complete and levels and v==levels[0] else
             '\\underline{'+fmt(v)+'}' if complete and len(levels)>1 and v==levels[1] else fmt(v))
            for v in rounded]


def matrix_table(rows, layout):
    """Return headers and already formatted rows; no averaging over checkpoints."""
    if layout == 'k2':
        headers = ['Encoder',r'$K_{tr}\to K_{inf}$','1-NN', 'Gallery nnz','Empty','Query nnz=0','nnz=1','nnz=2']
        body=[]
        for r in sorted(rows,key=lambda r:(r['encoder'],r['train_k'])):
            h=r.get('query_nnz_histogram')
            percentages=[100*h[i]/sum(h) if h and len(h)>i and sum(h) else None for i in range(3)]
            body.append([esc(NAMES[r['encoder']]),f"{r['train_k']} $\\to$ 2",fmt(score(r)),
                         fmt(r.get('gallery_mean_nnz')),count(r.get('empty_queries')),*map(fmt,percentages)])
        return headers,body
    if layout == 'inference':
        encoders=sorted({r['encoder'] for r in rows})
        by_encoder={e:sorted({r['inference_k'] for r in rows if r['encoder']==e}) for e in encoders}
        width=max(map(len,by_encoder.values()))
        diagnostics=any(r['task']=='imagenet_1nn' for r in rows)
        headers=['Encoder',r'$K_{tr}$','Method',*['']*width,'Avg.']+(['nnz','Empty'] if diagnostics else [])
        lookup={(r['encoder'],r['train_k'],r['method'],r['inference_k']):r for r in rows}
        body=[]; previous=None
        for e,kt in sorted({(r['encoder'],r['train_k']) for r in rows}):
            budgets=by_encoder[e]
            pad=['']*(width-len(budgets))
            if previous!=e:
                body.append([esc(NAMES[e]),'','',*[f'$K={k}$' for k in budgets],*pad,'']+(['',''] if diagnostics else []))
                previous=e
            methods=['csr','csrv2','mpsae']
            best={k:ranked([score(lookup.get((e,kt,m,k))) for m in methods]) for k in budgets}
            for i,m in enumerate(methods):
                selected=[lookup.get((e,kt,m,k)) for k in budgets]
                actual=[score(r) for r in selected if r is not None]
                avg=sum(actual)/len(actual) if actual and all(v is not None for v in actual) else None
                matched=lookup.get((e,kt,m,kt)) or {}
                body.append(['',str(kt),NAMES[m],
                    *[best[k][i]+(r'$^{\dagger}$' if k==kt and lookup.get((e,kt,m,k)) else '') for k in budgets],
                    *pad,fmt(avg)]+([fmt(matched.get('gallery_mean_nnz'),1),count(matched.get('empty_queries'))] if diagnostics else []))
        return headers,body
    if layout in ('mass','weight'):
        axis='mass' if layout=='mass' else 'loss_weight'
        values=sorted({r[axis] for r in rows})
        headers=['Dataset','Encoder',r'$K_{tr}\to K_{inf}$',*[f'${"s" if axis=="mass" else chr(92)+"lambda"}={v:g}$' for v in values]]
        body=[]
        for d,e,kt,ki in sorted({(r['dataset'],r['encoder'],r['train_k'],r['inference_k']) for r in rows}):
            points={r[axis]:r for r in rows if (r['dataset'],r['encoder'],r['train_k'],r['inference_k'])==(d,e,kt,ki)}
            body.append([NAMES.get(d,d),esc(NAMES[e]),f'{kt} $\\to$ {ki}'+(r'$^{\dagger}$' if kt==ki else ''),
                         *ranked([score(points.get(v)) for v in values])])
        return headers,body
    # Main segmentation / joint sweeps: dense and supervised baselines are reused.
    joint=layout=='joint'
    weights=sorted({r['loss_weight'] for r in rows if r['method']=='mpsae'})
    headers=['Dataset','Encoder',r'$K_{tr}\to K_{inf}$']+(['$s$'] if joint else [])+['Dense','CSR','CSRv2']+[f'$\\lambda={v:g}$' for v in weights]
    gains=not joint and 2.0 in weights
    if gains:
        headers += [r'$\Delta_{best}$',r'Ret. (\%)']
    body=[]
    keys=sorted({(r['dataset'],r['encoder'],r['train_k'],r['inference_k']) for r in rows if r['method']!='dense'})
    for d,e,kt,ki in keys:
        selected=[r for r in rows if r['dataset']==d and r['encoder']==e]
        dense=next((score(r) for r in selected if r['method']=='dense'),None)
        selected=[r for r in selected if (r['train_k'],r['inference_k'])==(kt,ki)]
        baseline=[next((score(r) for r in selected if r['method']==m),None) for m in ['csr','csrv2']]
        masses=sorted({r['mass'] for r in selected if r['method']=='mpsae'})
        for mass in masses:
            points={r['loss_weight']:score(r) for r in selected if r['method']=='mpsae' and r['mass']==mass}
            values=baseline+[points.get(w) for w in weights]
            line=[NAMES[d],esc(NAMES[e]),f'{kt} $\\to$ {ki}'+(r'$^{\dagger}$' if kt==ki else '')]
            line+=([f'{mass:g}'] if joint else [])+[fmt(dense)]+ranked(values)
            if gains:
                default=points.get(2.)
                delta=default-max(baseline) if default is not None and all(v is not None for v in baseline) else None
                retention=100*default/dense if default is not None and dense else None
                line += [fmt(delta),fmt(retention,1)]
            body.append(line)
    return headers,body


def make_figures(output, rows, plan, folder=HERE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory=Path(output)/'reports/figures'
    directory.mkdir(parents=True,exist_ok=True)
    files=[]
    style={'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
           'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300}
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('pdf','png'):
            path=directory/f'{name}.{suffix}'
            fig.savefig(path,bbox_inches='tight',metadata={'Creator':'MPSAE paper reproduction'} if suffix=='pdf' else None)
            files.append(path.relative_to(output).as_posix())
        plt.close(fig)
    with plt.rc_context(style):
        main=[r for r in rows if r['task']=='imagenet_1nn' and r['encoder']=='ff2048'
              and 'imagenet_main' in r['studies'] and r['train_k']==r['inference_k']]
        if any(score(r) is not None for r in main):
            fig,ax=plt.subplots(figsize=(3.5,2.7))
            for m,color,marker in [('csr','#0072B2','o'),('csrv2','#009E73','s'),('mpsae','#D55E00','^')]:
                points=sorted((r for r in main if r['method']==m),key=lambda r:r['train_k'])
                ax.plot([r['train_k'] for r in points],[score(r) if score(r) is not None else np.nan for r in points],
                        color=color,marker=marker,markersize=4,linewidth=1.5,label=NAMES[m])
            ax.set(xscale='log',xlabel='Train K',ylabel='1-NN (%)',title='ResNet-50',xticks=[8,16,32,256])
            ax.set_xticklabels(['8','16','32','256']); ax.legend(frameon=False,fontsize=8)
            ax.grid(axis='y',alpha=.2); save(fig,'figure_02_resnet50')
        for table in ('3','4','6','7','8','9','11'):
            spec,conditions=table_conditions(plan,table,folder)
            wanted={r['condition_id'] for r in conditions}
            selected=[r for r in rows if r['condition_id'] in wanted and r['method']=='mpsae']
            if not any(score(r) is not None for r in selected):
                continue
            panels=sorted({(r['encoder'],r['train_k']) for r in selected})
            fig,axes=plt.subplots(1,len(panels),figsize=(3.2*len(panels),2.8),squeeze=False)
            for ax,(encoder,kt) in zip(axes[0],panels):
                points=[r for r in selected if r['encoder']==encoder and r['train_k']==kt]
                ax.set_title(f'{NAMES[encoder]}, train K={kt}',fontsize=9)
                if spec['layout']=='joint':
                    masses=sorted({r['mass'] for r in points}); weights=sorted({r['loss_weight'] for r in points})
                    values=np.full((len(masses),len(weights)),np.nan)
                    for r in points:
                        if score(r) is not None:
                            values[masses.index(r['mass']),weights.index(r['loss_weight'])]=score(r)
                    im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap='viridis')
                    for i in range(len(masses)):
                        for j in range(len(weights)):
                            if np.isfinite(values[i,j]):
                                ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,
                                        color='white' if im.norm(values[i,j])<.5 else 'black')
                    ax.set(xticks=range(len(weights)),xticklabels=weights,yticks=range(len(masses)),yticklabels=masses,
                           xlabel=r'$\lambda$',ylabel='Mass s')
                else:
                    axis='loss_weight' if spec['layout']=='weight' else 'mass'
                    for i,ki in enumerate(sorted({r['inference_k'] for r in points})):
                        line=sorted((r for r in points if r['inference_k']==ki),key=lambda r:r[axis])
                        ax.plot([r[axis] for r in line],[score(r) if score(r) is not None else np.nan for r in line],
                                marker=('o','s','^')[i%3],color=('#0072B2','#D55E00','#009E73')[i%3],
                                linewidth=1.5,markersize=4,label=f'Inference K={ki}')
                    ax.set(xlabel=r'$\lambda$' if axis=='loss_weight' else 'Mass s',
                           ylabel='1-NN (%)' if points[0]['task']=='imagenet_1nn' else 'mIoU (%)')
                    ax.legend(frameon=False,fontsize=7); ax.grid(axis='y',alpha=.2)
            save(fig,f'table_{int(table):02d}_sensitivity')
    return files


def report(output, plan, *, tables=None, plots=True, folder=HERE):
    from ingest import rebuild_index
    output=Path(output)
    index=rebuild_index(output,plan)
    rows=[flat(r) for r in index['results']]
    write_csv(output/'results/metrics.csv',rows,FIELDS)
    inventory=[]
    for study in plan['studies']:
        points=[r for r in rows if study in r['studies']]
        inventory.append({'study':study,'expected':len(points),'complete':sum(r['status']=='complete' for r in points),
                          'conflicts':sum(r['status']=='conflict' for r in points),
                          'missing':sum(r['status']=='missing' for r in points)})
    write_csv(output/'results/study_status.csv',inventory,['study','expected','complete','conflicts','missing'])
    selected=tables or list(read_yaml(folder/'paper_map.yaml')['tables'])
    receipts={}
    for table in selected:
        spec,conditions=table_conditions(plan,table,folder)
        ids={r['condition_id'] for r in conditions}
        points=[r for r in rows if r['condition_id'] in ids]
        complete=sum(r['status']=='complete' for r in points)
        headers,body=matrix_table(points,spec['layout'])
        caption=spec['title']+f'. Available: {complete}/{len(points)} conditions. Single seed; -- means unavailable or conflicting.'
        if spec['layout'] in ('inference','mass','weight','segmentation','joint'):
            caption+=' Dagger marks matched train/inference K.'
        if spec['layout']=='inference' and any(r['task']=='imagenet_1nn' for r in points):
            caption+=' nnz and Empty describe the matched-K evaluation.'
        if spec['layout']=='segmentation' and any(r.get('loss_weight')==2. for r in points):
            caption+=' Delta and retention use MPSAE at lambda=2, versus the best CSR baseline and the dense reference, respectively.'
        lines=[r'% Generated from indexed original receipts; requires booktabs and graphicx.',
               f'% {caption}',r'\begin{table*}[t]',r'\centering',r'\small',
               '\\caption{'+esc(caption)+'}',f'\\label{{tab:paper-{table}}}',r'\resizebox{\textwidth}{!}{%',
               '\\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
        lines += [' & '.join(line)+r' \\' for line in body]
        lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}','']
        stem=f'table_{int(table):02d}'
        write_text(output/f'reports/tables/{stem}.tex','\n'.join(lines))
        write_csv(output/f'reports/tables/{stem}.csv',points,FIELDS)
        receipts[table]={'expected':len(points),'complete':complete,'condition_ids':sorted(ids)}
    figures=make_figures(output,rows,plan,folder) if plots else []
    preview=r'\documentclass{article}'+'\n'+r'\usepackage[margin=15mm]{geometry}'+'\n'+r'\usepackage{booktabs,graphicx}'+'\n'+r'\begin{document}'+'\n'
    preview+=''.join(f'\\input{{table_{int(t):02d}.tex}}\n\\clearpage\n' for t in selected)+r'\end{document}'+'\n'
    write_text(output/'reports/tables/preview.tex',preview)
    manifest={'schema':1,'paper':plan['paper'],'expected':index['expected'],'complete':index['complete'],
              'conflicts':index['conflicts'],'tables':receipts,'figures':figures,
              'index_sha256':digest(index),'verification':'Original receipts indexed; fresh GPU reproduction not implied'}
    write_json(output/'reports/manifest.json',manifest)
    return manifest


def verify_reference(output, folder=HERE):
    """Only compare declared reference cells; PDF-transcribed values never fill results."""
    path=folder/'reference/reported_values.csv'
    rows=read_json(Path(output)/'results/index.json',{}).get('results',[])
    lookup={r['condition_id']:flat(r) for r in rows}
    comparisons=[]
    with path.open(encoding='utf-8',newline='') as stream:
        for reference in csv.DictReader(stream):
            row=lookup.get(reference['condition_id'])
            value=score(row)
            status='missing' if value is None else 'match' if fmt(value)==fmt(float(reference['value'])) else 'different'
            comparisons.append({**reference,'observed':value,'status':status})
    write_csv(Path(output)/'reports/reference_check.csv',comparisons,
              ['table','condition_id','value','observed','status','source'])
    return comparisons


def bundle(output, folder=HERE):
    output=Path(output)
    archive=output/'cvpr2027-results.zip'
    if not (output/'reports/manifest.json').is_file():
        raise FileNotFoundError('Run report before bundle')
    allowed={'.json','.jsonl','.csv','.tex','.pdf','.png','.md','.txt','.yaml','.sha256'}
    with ZipFile(archive.with_suffix('.partial'),'w',ZIP_DEFLATED) as z:
        for area in ('results','reports','imports'):
            for path in sorted((output/area).rglob('*')):
                if path.is_file() and not path.is_symlink() and path.suffix in allowed:
                    z.write(path,path.relative_to(output))
        for name in ('catalog.yaml','paper_map.yaml','README.md'):
            z.write(folder/name,'experiment/'+name)
        for area in ('configs','reference','environment'):
            for path in sorted((folder/area).rglob('*')):
                if path.is_file() and path.suffix in allowed:
                    z.write(path,'experiment/'+str(path.relative_to(folder)).replace('\\','/'))
        if (output/'selections.json').is_file():
            z.write(output/'selections.json','selections.json')
    archive.with_suffix('.partial').replace(archive)
    return archive
