import csv
from pathlib import Path

from ..io import read_json, atomic_json
from .config import ENCODERS, DATASETS


def collect_tables(hub, campaign):
    rows = []
    for dataset in DATASETS:
        for encoder in ENCODERS:
            folder = hub.pull(f'campaigns/{campaign}/cells/{dataset}-{encoder}')
            rows.extend((read_json(folder/'rows.json') or {}).get('results', []))
    relative = f'campaigns/{campaign}/table'
    write_tables(rows, hub.root/relative)
    hub.push(relative)
    return hub.root/relative


def write_tables(rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = [(r['dataset'], r['encoder'], r['method']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('ambiguous table: duplicate dataset/encoder/method; select a single campaign')
    fields = ['dataset', 'encoder', 'method', 'k', 'seed', 'miou_percent', 'pixel_accuracy_percent',
              'mean_nnz', 'gallery_mean_nnz', 'query_images', 'query_seconds', 'search_workspace_bytes',
              'peak_cuda_bytes', 'supervision']
    with (directory/'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r['dataset'], r['encoder'], r['method'])):
            out = {k: r.get(k) for k in fields}
            out['miou_percent'] = 100*r['miou'] if r['miou'] is not None else ''
            out['pixel_accuracy_percent'] = 100*r['pixel_accuracy'] if r['pixel_accuracy'] is not None else ''
            writer.writerow(out)
    lookup = {key: r for key, r in zip(keys, rows)}
    names = {'siglip2': 'SigLIP 2-B/16', 'dinov2': 'DINOv2-B/14', 'clip': 'CLIP-B/16'}
    lines = [r'\begin{tabular}{llrrrr}', r'\hline',
             r'Dataset & Encoder & Dense & CSR & CSRv2 & MPSAE \\', r'\hline']
    for dataset in DATASETS:
        for encoder in ENCODERS:
            values = []
            for method in ('dense', 'csr', 'csrv2', 'mpsae'):
                r = lookup.get((dataset, encoder, method))
                values.append(f'{100*r["miou"]:.2f}' if r and r['miou'] is not None else '--')
            lines.append(' & '.join(['VOC' if dataset == 'voc' else 'ADE20K', names[encoder], *values])+r' \\')
    lines += [r'\hline', r'\end{tabular}',
              '% mIoU (%); K=12 for all SAEs; single seed; frozen encoders.',
              '% CSRv2 uses training semantic labels in its SAE objective.',
              '% Dataset-specific patch SAE training; adapted Hummingbird protocol, not paper reproduction.',
              '% -- means not completed, never zero.']
    (directory/'results.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    atomic_json(directory/'results.json', {'complete': len(rows) == 24, 'completed': len(rows),
                                         'expected': 24, 'results': rows})
