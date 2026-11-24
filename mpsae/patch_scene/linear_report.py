"""Tables and vector figures for the frozen linear segmentation readout."""
import os
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np

from ..io import atomic_json, read_json
from ..benchmark.retrieval_studies_report import write_csv
from ..benchmark.study_figures import STYLE, COLORS, save

NAMES = {'dense': 'Dense', 'csr': 'CSR', 'csrv2': 'CSRv2', 'mpsae': 'MPSAE'}


def figures(folder, rows, scientific):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        for dataset, encoder in sorted({(r['dataset'], r['encoder']) for r in rows}):
            selected = [r for r in rows if (r['dataset'], r['encoder']) == (dataset, encoder)]
            complete = [r for r in selected if r['status'] == 'complete']
            dense = next((r for r in complete if r['method'] == 'dense'), None)
            prefix = f'{dataset}_{encoder}'
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
            for ax, metric, label, scale in zip(axes, ('miou', 'pixel_accuracy', 'mean_nnz'),
                    ('mIoU (%)', 'Pixel accuracy (%)', 'Mean active latents'), (100, 100, 1)):
                for i, method in enumerate(('csr', 'csrv2', 'mpsae')):
                    points = sorted((r for r in selected if r['method'] == method), key=lambda r: r['train_k'])
                    if not points:
                        continue
                    ax.plot([r['train_k'] for r in points],
                            [r[metric]*scale if r['status'] == 'complete' else np.nan for r in points],
                            label=NAMES[method], color=COLORS[i], marker=('s','^','o')[i])
                if dense is not None and metric != 'mean_nnz':
                    ax.axhline(dense[metric]*scale, color='0.4', linestyle='--', label='Dense')
                ax.set_xticks(sorted({r['train_k'] for r in selected if r['train_k'] is not None}))
                ax.set(xlabel='Train / inference K', ylabel=label)
                if metric != 'mean_nnz':
                    low, high = ax.get_ylim()
                    ax.set_ylim(max(0, low), min(100, high))
                ax.grid(axis='y', alpha=.2)
                ax.legend(frameon=False, fontsize=8)
            save(fig, folder, f'{prefix}_linear_accuracy', scientific)
            fig, ax = plt.subplots(figsize=(6, 3.3))
            sparse = [r for r in selected if r['method'] != 'dense']
            delta = [(r['miou']-dense['miou'])*100 if dense and r['status'] == 'complete' else np.nan for r in sparse]
            ax.bar(range(len(sparse)), delta, color=[COLORS[('csr','csrv2','mpsae').index(r['method'])] for r in sparse])
            ax.set_xticks(range(len(sparse)), [f"{NAMES[r['method']]}\nK={r['train_k']}" for r in sparse], fontsize=8)
            ax.axhline(0, color='0.4', linewidth=.8)
            ax.set_ylabel('mIoU difference from dense (pp)')
            ax.grid(axis='y', alpha=.2)
            save(fig, folder, f'{prefix}_dense_difference', scientific)
            fig, ax = plt.subplots(figsize=(5.5, 3.3))
            for i, r in enumerate(complete):
                history = r.get('history', [])
                ax.plot([h['epoch'] for h in history], [h['loss'] for h in history],
                        label=NAMES[r['method']] + (f" K={r['train_k']}" if r['train_k'] is not None else ''),
                        color=COLORS[i % len(COLORS)], linestyle='--' if i >= len(COLORS) else '-')
            ax.set(xlabel='Probe epoch', ylabel='Training cross-entropy')
            if complete:
                ax.legend(frameon=False, fontsize=8, ncol=2)
            ax.grid(axis='y', alpha=.2)
            save(fig, folder, f'{prefix}_probe_convergence', scientific)
            if complete:
                matrix = np.array([[np.nan if v is None else v*100 for v in r['class_iou']] for r in complete])
                fig, ax = plt.subplots(figsize=(10, max(2.5, .45*len(complete))))
                im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, cmap='cividis', aspect='auto')
                ax.set_yticks(range(len(complete)), [NAMES[r['method']] + (f" K={r['train_k']}" if r['train_k'] is not None else '') for r in complete])
                ax.set_xticks(range(matrix.shape[1]))
                ax.set_xlabel('Class ID (0 is VOC background)')
                fig.colorbar(im, ax=ax, label='Class IoU (%)')
                save(fig, folder, f'{prefix}_class_iou', scientific)


def export(root, folder, plan, rows):
    folder.mkdir(parents=True, exist_ok=True)
    completed = sum(r['status'] == 'complete' for r in rows)
    result = {'plan': plan, 'results': rows, 'completed': completed, 'expected': plan['evaluations'],
              'complete': completed == plan['evaluations'], 'run_state': read_json(folder/'run-state.json')}
    atomic_json(folder/'results.json', result)
    tabular = []
    for row in rows:
        r = dict(row)
        if r['status'] == 'complete':
            r['miou_percent'], r['pixel_accuracy_percent'] = 100*r['miou'], 100*r['pixel_accuracy']
        else:
            for field in ('miou_percent', 'pixel_accuracy_percent', 'mean_nnz', 'head_parameters', 'empty_patches'):
                r.pop(field, None)
        tabular.append(r)
    write_csv(folder/'linear_results.csv', tabular, ['dataset', 'encoder', 'method', 'train_k', 'inference_k', 'status',
        'miou_percent', 'pixel_accuracy_percent', 'mean_nnz', 'empty_patches', 'training_rows', 'head_parameters',
        'checkpoint_sha256', 'sae_checkpoint', 'error'])
    lines = [r'% Requires booktabs. Frozen encoder and SAE; supervised affine readout.',
             r'\begin{tabular}{lllrrrr}', r'\toprule',
             r'Dataset & Encoder & Method & $K$ & mIoU (\%) & Acc. (\%) & nnz \\', r'\midrule']
    if not plan['scientific']:
        lines.insert(0, '% SYNTHETIC SMOKE TEST -- NOT SCIENTIFIC RESULTS')
    for r in tabular:
        def fmt(field):
            return f'{r[field]:.2f}' if r.get(field) is not None else '--'
        lines.append(f"{r['dataset'].upper()} & {r['encoder']} & {NAMES[r['method']]} & "
            f"{r['train_k'] if r['train_k'] is not None else '--'} & {fmt('miou_percent')} & "
            f"{fmt('pixel_accuracy_percent')} & {fmt('mean_nnz')} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (folder/'linear_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Do not repeatedly render identical figures while a long fit is still running.
    from ..io import canonical_hash
    signature = canonical_hash({'format': 2, 'rows': [(r['status'] == 'complete', r.get('checkpoint_sha256'), r.get('miou')) for r in rows]})
    drawn = read_json(folder/'figure-state.json') or {}
    if drawn.get('signature') != signature or not all((folder/n).exists() for n in drawn.get('files', [])):
        figures(folder, rows, plan['scientific'])
        atomic_json(folder/'figure-state.json', {'signature': signature,
            'files': [p.name for p in folder.iterdir() if p.suffix in ('.pdf', '.png')]})
    (folder/'README.txt').write_text(
        'Frozen patch linear segmentation, not nearest-neighbor/Hummingbird decoding.\n'
        'Inputs: L2-normalized dense features or post-ReLU Top-K SAE codes.\n'
        'All cached training patches (not the support bank); soft patch targets; no validation selection.\n'
        'Standard affine classifier: zero inputs predict the learned bias; empty patches are reported.\n'
        'Dense and sparse heads have different input dimensions; see head_parameters in CSV.\n'
        'mIoU/accuracy are percentages in CSV and tables; fractions in result JSON.\n'
        'Single seed; no significance claim or fabricated error bars. Retain workspace for checkpoints.\n', encoding='utf-8')
    archive = folder/'linear-segmentation-results.zip'
    with ZipFile(archive.with_suffix('.partial'), 'w', ZIP_DEFLATED) as z:
        for path in sorted(folder.iterdir()):
            if path.suffix in ('.json', '.csv', '.tex', '.txt', '.pdf', '.png'):
                z.write(path, path.name)
    os.replace(archive.with_suffix('.partial'), archive)
    latest = root/'linear-segmentation-results.zip'
    shutil.copyfile(archive, latest.with_suffix('.partial'))
    os.replace(latest.with_suffix('.partial'), latest)
    atomic_json(root/'linear-latest.json', {'archive': str(archive), 'campaign_id': plan['campaign_id'],
                                          'completed': completed, 'expected': plan['evaluations']})
    print(f'[linear segmentation] {completed}/{plan["evaluations"]}; {archive}', flush=True)
    return result
