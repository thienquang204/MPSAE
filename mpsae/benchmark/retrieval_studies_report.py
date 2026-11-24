"""Local tables/figures for the two declared retrieval studies; no external curves."""
import csv
import io
import os
import statistics
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from ..io import atomic_json, read_json, canonical_hash

NAMES = {'ff2048': 'ResNet-50', 'vit_b16': 'ViT-B/16'}
KEYS = ['study', 'backbone', 'train_k', 'inference_k', 's', 'lambda', 'batch_size', 'epochs']


def write_csv(path, rows, fields):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(text.getvalue(), encoding='utf-8')
    os.replace(temp, path)


def summarize(rows, seeds):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in KEYS), []).append(row)
    summary = []
    for key, values in sorted(groups.items()):
        complete = [r for r in values if r['status'] == 'complete']
        scores = [r['top1_percent'] for r in complete]
        summary.append({**dict(zip(KEYS, key)), 'n': len(scores), 'expected_seeds': len(seeds),
                        'top1_mean': statistics.mean(scores) if scores else None,
                        'top1_std': statistics.stdev(scores) if len(scores)>1 else None,
                        'gallery_mean_nnz': statistics.mean(r['gallery_mean_nnz'] for r in complete) if complete else None,
                        'empty_queries': statistics.mean(r['empty_queries'] for r in complete) if complete else None,
                        'updates': complete[0].get('training_identity', {}).get('schedule', {}).get('total_steps') if complete else None})
    return summary


def figures(directory, summary, scientific):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
              'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    files = []
    with plt.rc_context(styles):
        for study, xkey, xlabel, filename in (
                ('mass', 's', 'Transport mass $s$', 'mass_accuracy'),
                ('optimizer', 'lambda', r'OT weight $\lambda$', 'weight_accuracy')):
            selected = [r for r in summary if r['study'] == study and r['train_k'] == r['inference_k']]
            if not selected:
                continue
            backbones = sorted({r['backbone'] for r in selected})
            fig, axes = plt.subplots(1, len(backbones), figsize=(4.1*len(backbones), 3.3), squeeze=False)
            for ax, backbone in zip(axes[0], backbones):
                points = [r for r in selected if r['backbone'] == backbone]
                curves = sorted({(r['train_k'], r['batch_size']) for r in points})
                batches = {b for _, b in curves}
                for i, (k, batch) in enumerate(curves):
                    series = sorted((r for r in points if r['train_k'] == k and r['batch_size'] == batch), key=lambda r: r[xkey])
                    # Preserve missing positions as NaN so partial curves never bridge them.
                    x = [r[xkey] for r in series]
                    y = [r['top1_mean'] if r['top1_mean'] is not None else float('nan') for r in series]
                    label = f'K={k}' + (f', B={batch}' if len(batches)>1 else '')
                    ax.plot(x, y, marker=('o', 's', '^')[i % 3], linewidth=1.7, markersize=4,
                            color=('#0072B2', '#D55E00', '#009E73')[i % 3], label=label)
                    for r in series:
                        if r['top1_std'] is not None:
                            ax.errorbar(r[xkey], r['top1_mean'], yerr=r['top1_std'], color='0.3', capsize=2, fmt='none')
                ax.set(title=NAMES[backbone], xlabel=xlabel, ylabel='1-NN accuracy (%)')
                ax.set_xticks(sorted({r[xkey] for r in points}))
                ax.grid(axis='y', alpha=.2)
                ax.legend(frameon=False)
            if not scientific:
                fig.suptitle('SYNTHETIC SMOKE TEST', color='red', fontsize=10)
            fig.tight_layout()
            for ext in ('pdf', 'png'):
                path = directory/f'{filename}.{ext}'
                fig.savefig(path, dpi=300, bbox_inches='tight')
                files.append(path)
            plt.close(fig)
    return files


def export(root, plan, rows):
    directory = root/'studies'/plan['suite_id']
    directory.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, plan['settings']['seeds'])
    unique = {}
    for row in rows:
        key = (row['backbone'], row['train_k'], row['inference_k'], row['seed'], row['s'], row['lambda'], row['batch_size'], row['epochs'])
        if row['status'] == 'complete':
            previous = unique.get(key)
            if previous and any(previous.get(k) != row.get(k) for k in ('signature', 'checkpoint_sha256', 'top1_percent')):
                raise ValueError('Overlapping studies disagree on a reused condition')
            unique[key] = row
    result = {'complete': all(r['status'] == 'complete' for r in rows),
              'completed': len(unique), 'expected': plan['unique_evaluations'], 'plan': plan, 'results': rows,
              'run_state': read_json(directory/'run-state.json')}
    atomic_json(directory/'results.json', result)
    fields = [*KEYS, 'seed', 'status', 'top1_percent', 'gallery_mean_nnz', 'query_mean_nnz', 'empty_queries',
              'gallery_nnz_histogram', 'query_nnz_histogram',
              'train_seconds', 'signature', 'checkpoint_path', 'checkpoint_sha256', 'solver', 'error']
    write_csv(directory/'results.csv', rows, fields)
    summary_fields = [*KEYS, 'top1_mean', 'top1_std', 'gallery_mean_nnz', 'empty_queries', 'updates', 'n', 'expected_seeds']
    write_csv(directory/'summary.csv', summary, summary_fields)
    sparsity = []
    for row in rows:
        if row['inference_k'] != 2:
            continue
        record = {k: row[k] for k in (*KEYS, 'seed', 'status')}
        for split in ('gallery', 'query'):
            counts = row.get(f'{split}_nnz_histogram') if row['status'] == 'complete' else None
            for n in range(3):
                record[f'{split}_nnz{n}_percent'] = 100*counts[n]/sum(counts) if counts and sum(counts) else None
        sparsity.append(record)
    write_csv(directory/'k2_sparsity.csv', sparsity, [*KEYS, 'seed', 'status',
        *[f'{split}_nnz{n}_percent' for split in ('gallery', 'query') for n in range(3)]])
    for study in ('mass', 'optimizer', 'extreme'):
        chosen = [r for r in summary if r['study'] == study]
        if not chosen:
            continue
        write_csv(directory/f'{study}_summary.csv', chosen, summary_fields)
        lines = [r'% Requires \usepackage{booktabs}.', r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
                 r'Backbone & $K_{tr}$ & $K_{inf}$ & $s$ & $\lambda$ & Batch & 1-NN (\%) & nnz & Seeds \\', r'\midrule']
        if not plan['scientific']:
            lines.insert(0, '% SYNTHETIC SMOKE TEST -- NOT BENCHMARK RESULTS')
        for r in chosen:
            score = '--' if r['top1_mean'] is None else f"{r['top1_mean']:.2f}"
            if r['top1_std'] is not None:
                score += r' $\pm$ '+f"{r['top1_std']:.2f}"
            nnz = '--' if r['gallery_mean_nnz'] is None else f"{r['gallery_mean_nnz']:.2f}"
            lines.append(f"{NAMES[r['backbone']]} & {r['train_k']} & {r['inference_k']} & {r['s']:g} & {r['lambda']:g} & "
                         f"{r['batch_size']} & {score} & {nnz} & {r['n']}/{r['expected_seeds']} " + r'\\')
        lines += [r'\bottomrule', r'\end{tabular}',
                  '% Mean and sample SD across seeds; -- means unavailable. No best-setting selection.',
                  '% Epochs are fixed, optimizer updates depend on batch size. nnz is measured on the gallery.']
        (directory/f'{study}_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    signature = canonical_hash({'summary': summary, 'scientific': plan['scientific'], 'format': 'cross-k-v1'})
    drawn = read_json(directory/'figure-state.json') or {}
    if drawn.get('signature') != signature or not all((directory/n).exists() for n in drawn.get('files', [])):
        figures(directory, summary, plan['scientific'])
        from .study_figures import extended_figures
        extended_figures(directory, summary, plan['scientific'])
        atomic_json(directory/'figure-state.json', {'signature': signature,
            'files': [p.name for p in directory.iterdir() if p.suffix in ('.pdf', '.png')]})
    (directory/'README.txt').write_text(
        f"Unique completed conditions: {result['completed']}/{result['expected']}.\n"
        'MPSAE only. Mass study: ResNet-50/ViT-B/16. Weight study: ResNet-50.\n'
        'See plan.json for actual settings. No published curves or test-based selection.\n'
        'Plots: matched accuracy, inference sweeps, nnz, empty queries, heatmaps and K=2 comparison; PDF/300-dpi PNG.\n'
        'Tables: *_table.tex. Training K and inference K are separate columns and aggregation keys.\n'
        'Missing values are not zeros. Larger batches have fewer updates at equal epochs.\n'
        'No model weights or embeddings in this ZIP. Keep the workspace/source files to resume.\n', encoding='utf-8')
    archive = directory/'retrieval-studies-results.zip'
    temporary = archive.with_suffix('.zip.partial')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as bundle:
        for path in sorted(directory.iterdir()):
            if path.suffix in ('.csv', '.json', '.tex', '.txt', '.png', '.pdf'):
                bundle.write(path, path.name)
        for campaign in plan['campaigns']:
            folder = root/'sweeps'/campaign['sweep_id']
            for name in ('plan.json', 'runtime.json', 'attempts.jsonl', 'results.json'):
                if (folder/name).exists():
                    bundle.write(folder/name, f"campaigns/{campaign['sweep_id']}/{name}")
    os.replace(temporary, archive)
    latest_zip = root/'retrieval-studies-results.zip'
    latest_tmp = latest_zip.with_suffix('.zip.partial')
    shutil.copyfile(archive, latest_tmp)
    os.replace(latest_tmp, latest_zip)
    atomic_json(root/'studies-latest.json', {'archive': str(archive), 'suite_id': plan['suite_id'],
                                          'completed': result['completed'], 'expected': result['expected']})
    print(f"[retrieval studies] {result['completed']}/{result['expected']} unique evaluations; {archive}", flush=True)
    return result
