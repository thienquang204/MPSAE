"""Matched-budget and cross-budget paper tables, with one coordinator writer."""
import csv
import math
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from ..io import read_json, atomic_json
from .config import Config, ENCODERS, DATASETS, METHODS

NAMES = {'siglip2': 'SigLIP 2', 'dinov2': 'DINOv2', 'clip': 'CLIP'}
DATASET_NAMES = {'voc': 'VOC', 'ade20k': 'ADE20K'}
TABLE_FILES = ('results.csv', 'comparisons.csv', 'results.tex', 'table_voc.tex',
               'table_ade20k.tex', 'pixel_accuracy.tex', 'inference_sweep.tex',
               'sweep_voc.tex', 'sweep_ade20k.tex', 'tables_preview.tex', 'results.json')


def row_key(row):
    sparse = row['method'] != 'dense'
    return (row['dataset'], row['encoder'], row['method'],
            row.get('train_k', 12) if sparse else None,
            row.get('inference_k', row.get('k')) if sparse else None)


def expected_keys(config):
    return {(d, e, 'dense', None, None) for d in DATASETS for e in config.encoder_names} | {
        (d, e, m, train_k, k) for d in DATASETS for e in config.encoder_names
        for m in METHODS for train_k in config.training_ks for k in config.inference_ks}


def collect_tables(hub, campaign):
    settings = read_json(hub.root/f'campaigns/{campaign}/settings/config.json')
    config = Config(**settings['settings']) if settings else Config()
    rows = []
    for dataset in DATASETS:
        for encoder in config.encoder_names:
            folder = hub.pull(f'campaigns/{campaign}/cells/{dataset}-{encoder}')
            rows.extend((read_json(folder/'rows.json') or {}).get('results', []))
    if not rows:
        raise FileNotFoundError('No completed evaluations found for this campaign; run a job first.')
    relative = f'campaigns/{campaign}/table'
    write_tables(rows, hub.root/relative, config)
    hub.push(relative)
    archive = export_results(hub.root, campaign)
    print(f'[download] {archive} ({len(rows)}/{len(expected_keys(config))} completed evaluations)', flush=True)
    return hub.root/relative


def export_results(workspace, campaign):
    """Small atomic download; no feature caches, weights or private credentials."""
    workspace = Path(workspace)
    root = workspace/'campaigns'/campaign
    table = root/'table'
    result = read_json(table/'results.json')
    if not result or not result.get('results'):
        raise FileNotFoundError('No completed evaluation results to export')
    sources = [(table/name, name) for name in TABLE_FILES]
    for folder, name in [('settings', 'config.json'), ('sources', 'sources.json')]:
        if (root/folder/name).is_file():
            sources.append((root/folder/name, name))
    archive = workspace/'patch-scene-results.zip'
    temporary = archive.with_suffix('.zip.partial')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as handle:
        for path, name in sources:
            handle.write(path, arcname=name)
        handle.writestr('README.txt',
            f'Campaign: {campaign}\nCompleted evaluations: {result["completed"]}/{result["expected"]}\n'
            'results.tex: main mIoU table; table_voc.tex/table_ade20k.tex: separate dataset tables.\n'
            'For multiple training budgets, main tables show matched train K = inference K.\n'
            'inference_sweep.tex and sweep_voc.tex/sweep_ade20k.tex: all train/inference combinations.\n'
            'pixel_accuracy.tex: supplementary pixel accuracy table.\n'
            'Use \\usepackage{booktabs} and \\input{results.tex} in your paper.\n'
            'Compile tables_preview.tex with pdflatex to preview all tables.\n'
            'results.csv: metrics (%) and measured sparsity; comparisons.csv: gains in percentage points.\n'
            'results.json: detailed metrics, missing entries and provenance.\n'
            'Bold = best sparse method only when all three scores exist; ties at printed precision are bold.\n'
            'Delta = MPSAE minus max(CSR, CSRv2) at the same encoder, dataset, train K and inference K.\n'
            '-- means missing; negative deltas are retained. No significance claim from a single seed.\n'
            f'Training K: {result["train_ks"]}; inference K: {result["inference_k"]}.\n'
            'Each training budget has a separate checkpoint, reused across inference budgets.\n'
            'CSRv2 uses training-mask labels in its loss. Dataset-specific adaptation, not cross-dataset transfer.\n'
            'This is a results archive, not a training-resume backup.\n'
            'Model checkpoints and feature caches remain in the Kaggle workspace.\n')
    os.replace(temporary, archive)
    return archive


def _score(row, metric):
    value = row.get(metric) if row else None
    return 100*value if value is not None else None


def _delta(a, b):
    return a-b if a is not None and b is not None else None


def _csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _comparison(lookup, dataset, encoder, train_k, k, metric):
    scores = {m: _score(lookup.get((dataset, encoder, m, train_k, k)), metric) for m in METHODS}
    dense = _score(lookup.get((dataset, encoder, 'dense', None, None)), metric)
    baseline = max(scores['csr'], scores['csrv2']) if all(scores[m] is not None for m in ('csr', 'csrv2')) else None
    return dense, scores, baseline, _delta(scores['mpsae'], baseline)


def _latex(lookup, config, *, datasets, metric='miou', label='patch-sae', matched=False):
    combined = len(datasets) > 1
    pairs = [(t, k) for t in config.training_ks for k in config.inference_ks if not matched or t == k]
    show_train = not matched and len(config.training_ks) > 1
    metric_name = 'mIoU' if metric == 'miou' else 'pixel accuracy'
    expected = {key for key in expected_keys(config) if key[0] in datasets
                and (key[2] == 'dense' or (key[3], key[4]) in pairs)}
    count = len(set(lookup) & expected)
    status = f'Partial results ({count}/{len(expected)} evaluations). ' if count != len(expected) else ''
    title = ' / '.join(DATASET_NAMES[d] for d in datasets)
    protocol = (r'Matched budgets: $K_{\mathrm{train}}=K_{\mathrm{inf}}=K$.' if matched else
                r'Each training checkpoint is evaluated across inference budgets.' if show_train else
                f'SAEs trained at $K_{{\\mathrm{{train}}}}={config.training_ks[0]}$.')
    lines = [r'% Requires \usepackage{booktabs}. Generated from measured results only.',
             r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
             '\\caption{'+status+f'{title} patch retrieval segmentation: {metric_name} (\\%). '
             + 'Frozen vision encoders. '+protocol+'}',
             '\\label{tab:'+label+'}', '\\begin{tabular}{'+('l' if combined else '')+'l'+('r' if show_train else '')+'rrrrrr}',
             r'\toprule', ('Dataset & ' if combined else '')
             + 'Encoder & '+(r'$K_{\mathrm{train}}$ & ' if show_train else '')
             + (r'$K$' if matched else r'$K_{\mathrm{inf}}$')
             + r' & Dense & CSR & CSRv2$^{\dagger}$ & MPSAE & $\Delta$ (pp) \\',
             r'\midrule']
    for di, dataset in enumerate(datasets):
        for ei, encoder in enumerate(config.encoder_names):
            if di or ei:
                lines.append(r'\midrule' if ei == 0 else r'\addlinespace[3pt]')
            for ki, (train_k, k) in enumerate(pairs):
                dense, scores, _, delta = _comparison(lookup, dataset, encoder, train_k, k, metric)
                formatted = {m: f'{value:.2f}' if value is not None else '--' for m, value in scores.items()}
                if all(value is not None for value in scores.values()):
                    best = max(float(value) for value in formatted.values())
                    formatted = {m: r'\textbf{'+value+'}' if float(value) == best else value
                                 for m, value in formatted.items()}
                cells = ([DATASET_NAMES[dataset] if ei == ki == 0 else ''] if combined else [])
                cells += [NAMES[encoder] if ki == 0 else '']
                if show_train:
                    cells += [str(train_k)]
                cells += [str(k),
                          f'{dense:.2f}' if dense is not None else '--',
                          *(formatted[m] for m in METHODS),
                          f'{0.0 if abs(delta) < .005 else delta:+.2f}' if delta is not None else '--']
                lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
              r'\parbox{0.98\textwidth}{\footnotesize '
              'All encoders are ViT-B ('+', '.join(f'{NAMES[e]}: patch {ENCODERS[e][1]}' for e in config.encoder_names)+'). '
              + f'Single seed ({config.seed}); separate SAE fitting on each dataset\'s training split. '
              + r'Bold: best sparse method per row, including ties at displayed precision. '
              r'$\Delta$: MPSAE minus the stronger of CSR and CSRv2, in percentage points. '
              r'$^{\dagger}$CSRv2 uses training-mask labels in its SAE loss. '
              r'$K_{\mathrm{inf}}$ bounds nonzero entries; actual counts are in the CSV. '
              r'Dense is evaluated once per encoder/dataset and repeated for comparison. '
              r'No significance claim; -- denotes unavailable results. '
              r'Hummingbird-style retrieval with dataset-specific patch SAE training, not paper reproduction.}',
              r'\end{table*}']
    return '\n'.join(lines)+'\n'


def write_tables(rows, directory, config=None):
    config = config or Config()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lookup = {}
    expected = expected_keys(config)
    for row in rows:
        key = row_key(row)
        if key in lookup:
            raise ValueError('ambiguous table: duplicate dataset/encoder/method/train K/inference K')
        if key not in expected or row.get('seed') != config.seed or not row.get('complete', False):
            raise ValueError(f'row outside the completed campaign: {key}')
        for metric in ('miou', 'pixel_accuracy'):
            value = row.get(metric)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'invalid {metric}: {value}')
        lookup[key] = row
    comparisons, metrics = [], []
    for dataset in DATASETS:
        for encoder in config.encoder_names:
            for train_k, k in ((t, k) for t in config.training_ks for k in config.inference_ks):
                dense, scores, baseline, delta = _comparison(lookup, dataset, encoder, train_k, k, 'miou')
                comparisons.append({'dataset': dataset, 'encoder': encoder, 'train_k': train_k,
                    'inference_k': k, 'dense_miou_percent': dense,
                    **{m+'_miou_percent': scores[m] for m in METHODS},
                    'best_baseline_miou_percent': baseline, 'mpsae_gain_over_best_baseline_pp': delta,
                    'mpsae_gain_over_dense_pp': _delta(scores['mpsae'], dense)})
    fields = ['dataset', 'encoder', 'method', 'k', 'train_k', 'inference_k', 'seed',
              'miou_percent', 'pixel_accuracy_percent', 'gain_over_dense_pp', 'gain_over_best_baseline_pp',
              'mean_nnz', 'gallery_mean_nnz', 'query_images', 'query_seconds', 'search_workspace_bytes',
              'peak_cuda_bytes', 'supervision']
    for key, row in sorted(lookup.items()):
        out = {f: row.get(f) for f in fields}
        out.update(train_k=key[3], inference_k=key[4])
        out['miou_percent'], out['pixel_accuracy_percent'] = _score(row, 'miou'), _score(row, 'pixel_accuracy')
        dense = _score(lookup.get((key[0], key[1], 'dense', None, None)), 'miou')
        out['gain_over_dense_pp'] = _delta(out['miou_percent'], dense) if key[2] != 'dense' else None
        if key[2] != 'dense':
            _, _, baseline, _ = _comparison(lookup, key[0], key[1], key[3], key[4], 'miou')
            out['gain_over_best_baseline_pp'] = _delta(out['miou_percent'], baseline)
        metrics.append(out)
    _csv(directory/'results.csv', fields, metrics)
    _csv(directory/'comparisons.csv', list(comparisons[0]), comparisons)
    matched = len(config.training_ks) > 1 and set(config.training_ks) <= set(config.inference_ks)
    (directory/'results.tex').write_text(_latex(lookup, config, datasets=tuple(DATASETS), matched=matched), encoding='utf-8')
    (directory/'inference_sweep.tex').write_text(
        _latex(lookup, config, datasets=tuple(DATASETS), label='patch-sae-sweep'), encoding='utf-8')
    for dataset in DATASETS:
        (directory/f'table_{dataset}.tex').write_text(
            _latex(lookup, config, datasets=(dataset,), label=f'patch-sae-{dataset}', matched=matched), encoding='utf-8')
        (directory/f'sweep_{dataset}.tex').write_text(
            _latex(lookup, config, datasets=(dataset,), label=f'patch-sae-sweep-{dataset}'), encoding='utf-8')
    (directory/'pixel_accuracy.tex').write_text(
        _latex(lookup, config, datasets=tuple(DATASETS), metric='pixel_accuracy', label='patch-sae-pa', matched=matched), encoding='utf-8')
    (directory/'tables_preview.tex').write_text(
        '\\documentclass{article}\n\\usepackage[a4paper,margin=18mm]{geometry}\n'
        '\\usepackage{booktabs}\n\\begin{document}\n\\input{results.tex}\n\\clearpage\n'
        '\\input{table_voc.tex}\n\\clearpage\n\\input{table_ade20k.tex}\n\\clearpage\n'
        '\\input{pixel_accuracy.tex}\n\\clearpage\n\\input{sweep_voc.tex}\n\\clearpage\n'
        '\\input{sweep_ade20k.tex}\n\\end{document}\n', encoding='utf-8')
    missing = sorted(expected-set(lookup))
    atomic_json(directory/'results.json', {'complete': not missing, 'completed': len(lookup),
        'expected': len(expected), 'train_k': config.plan()['train_k'], 'train_ks': list(config.training_ks),
        'inference_k': list(config.inference_ks), 'main_table': 'matched_budgets' if matched else 'inference_sweep',
        'missing': [dict(zip(('dataset', 'encoder', 'method', 'train_k', 'inference_k'), key)) for key in missing],
        'results': [lookup[key] for key in sorted(lookup)]})
