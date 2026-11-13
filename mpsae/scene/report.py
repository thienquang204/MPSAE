"""Measured curves only, with consistent methods and frozen/adapted styles."""
import csv
import json
import math
from pathlib import Path

from ..io import atomic_json, canonical_hash

NAMES = {'dense': 'Dense', 'csr': 'CSR', 'csrv2': 'CSRv2', 'mpsae': 'MPSAE'}
TITLES = {'ff2048': 'ResNet-50', 'vit_b16': 'ViT-B/16',
          'dinov2_vits14': 'DINOv2-S/14', 'convnext_tiny': 'ConvNeXt-Tiny'}
COLORS = {'csr': '#009E73', 'csrv2': '#7950AD', 'mpsae': '#D55E00', 'dense': '#555555'}
MARKERS = {'csr': 's', 'csrv2': 'D', 'mpsae': 'o', 'dense': '^'}


def _rows(payload):
    backbone = payload.get('campaign', {}).get('plan', {}).get('config', {}).get('backbone', '')
    return [{**row, 'backbone': row.get('backbone', backbone),
             'variant': row.get('variant', 'dense' if row['method'] == 'dense' else 'frozen')}
            for row in payload['results']]


def _save(fig, directory, stem):
    import matplotlib.pyplot as plt
    for extension in ('pdf', 'png'):
        fig.savefig(directory / f'{stem}.{extension}', dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _axes_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7)


def _legend(fig, rows):
    from matplotlib.lines import Line2D
    methods = [m for m in NAMES if m != 'dense' and any(r['method'] == m for r in rows)]
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=NAMES[m]) for m in methods]
    if any(r['method'] == 'dense' for r in rows):
        handles.append(Line2D([], [], color=COLORS['dense'], ls=':', label='Dense'))
    variants = {r['variant'] for r in rows if r['method'] != 'dense'}
    if variants:
        handles += [Line2D([], [], color='0.25', ls=ls, label=name)
                    for variant, ls, name in [('frozen', '--', 'Frozen'), ('adapted', '-', 'Fine-tuned')]
                    if variant in variants]
    fig.legend(handles=handles, loc='outside lower center', ncol=min(6, len(handles)),
               frameon=False, handlelength=1.8, columnspacing=1.2)


def _curves(ax, rows, xkey, ykey, *, xscale=1, yscale=100):
    for method in ('csr', 'csrv2', 'mpsae'):
        for variant in ('frozen', 'adapted'):
            selected = sorted((r for r in rows if r['method'] == method and r['variant'] == variant),
                              key=lambda r: r['inference_k'])
            selected = [r for r in selected if r.get(xkey) is not None and r.get(ykey) is not None]
            if not selected:
                continue
            adapted = variant == 'adapted'
            ax.plot([r[xkey]*xscale for r in selected], [r[ykey]*yscale for r in selected],
                color=COLORS[method], marker=MARKERS[method], markersize=4,
                markerfacecolor=COLORS[method] if adapted else 'white', markeredgewidth=.9,
                ls='-' if adapted else '--', lw=1.5 if adapted else 1.1, alpha=1 if adapted else .75)


def _k_axis(ax, rows):
    budgets = sorted({r['inference_k'] for r in rows if r['inference_k'] is not None})
    if budgets:
        ax.set_xscale('log', base=2)
        ax.set_xticks(budgets, [str(k) for k in budgets])
        ax.minorticks_off()
    ax.set_xlabel('Inference K')


def _measured_axis(ax):
    """Readable numeric ticks without crowding on a wide measured-resource range."""
    from matplotlib.ticker import FixedLocator, FuncFormatter
    low, high = ax.get_xlim()
    positive = max(low, .001)
    first = math.floor(math.log2(positive))
    last = math.ceil(math.log2(max(high, positive*2)))
    stride = max(1, math.ceil((last-first)/5))
    ticks = [2**i for i in range(first, last+1, stride) if low <= 2**i <= high]
    if low <= 0:
        ticks.insert(0, 0)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.minorticks_off()


def plot(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = _rows(payload)
    for row in rows:
        count = row.get('query_images', 0)
        row['search_seconds_per_image'] = row['search_seconds']/count if count else None
    with (path.parent / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        columns = ('backbone', 'method', 'variant', 'train_k', 'inference_k', 'checkpoint_id',
                   'miou', 'pixel_accuracy', 'bank_nnz_mean', 'query_nnz_mean', 'query_nnz_max',
                   'query_empty_patches', 'index_bytes', 'search_seconds', 'search_seconds_per_image')
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    groups = [(b, [r for r in rows if r['backbone'] == b]) for b in dict.fromkeys(r['backbone'] for r in rows)]
    if not groups:
        raise ValueError('no scene results to plot')
    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
             'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
             'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .7}
    with plt.rc_context(style):
        for stem, panels in (
            ('segmentation', [('inference_k', 'miou', 'mIoU (%)'),
                              ('inference_k', 'pixel_accuracy', 'Pixel accuracy (%)')]),
            ('retrieval', [('query_nnz_mean', 'miou', 'mIoU (%)'),
                           ('index_bytes', 'miou', 'mIoU (%)')])):
            fig, axes = plt.subplots(len(groups), 2, squeeze=False,
                figsize=(7.2, 2.6*len(groups)+.35), layout='constrained')
            for group_index, (backbone, selected) in enumerate(groups):
                for column, (xkey, ykey, ylabel) in enumerate(panels):
                    ax = axes[group_index, column]
                    _axes_style(ax)
                    _curves(ax, selected, xkey, ykey, xscale=1/2**20 if xkey == 'index_bytes' else 1)
                    dense = next((r for r in selected if r['method'] == 'dense'), None)
                    if xkey == 'inference_k':
                        _k_axis(ax, selected)
                        if dense and dense.get(ykey) is not None:
                            ax.axhline(100*dense[ykey], color=COLORS['dense'], ls=':', lw=1.2)
                    else:
                        ax.set_xscale('symlog', linthresh=.001 if xkey == 'index_bytes' else 1, base=2)
                        ax.set_xlabel('Mean active dims.' if xkey == 'query_nnz_mean' else 'Key storage (MiB)')
                        if dense and dense.get(ykey) is not None:
                            ax.scatter(dense[xkey]/2**20 if xkey == 'index_bytes' else dense[xkey],
                                       100*dense[ykey], color=COLORS['dense'], marker='^', s=23, zorder=5)
                        _measured_axis(ax)
                    ax.set_ylabel(ylabel)
                    ax.set_title(TITLES.get(backbone, backbone), loc='left')
                    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=min(100, ax.get_ylim()[1]))
            _legend(fig, rows)
            _save(fig, path.parent, stem)
        gains = []
        for row in rows:
            if row['variant'] != 'adapted':
                continue
            old = next((r for r in rows if (r['backbone'], r['method'], r['inference_k'], r['variant']) ==
                       (row['backbone'], row['method'], row['inference_k'], 'frozen')), None)
            if old and row.get('miou') is not None and old.get('miou') is not None:
                gains.append({**row, 'gain': row['miou']-old['miou']})
        for stem, values, metric, ylabel, scale in (
                ('adaptation_gain', gains, 'gain', 'mIoU gain (pp)', 100),
                ('search_time', rows, 'search_seconds_per_image', 'Search time (s/image)', 1)):
            if not any(r.get(metric) is not None for r in values):
                continue
            columns = min(2, len(groups))
            nrows = math.ceil(len(groups)/columns)
            fig, axes = plt.subplots(nrows, columns, squeeze=False,
                                     figsize=(3.6*columns, 2.6*nrows+.35), layout='constrained')
            for ax in axes.flat[len(groups):]:
                ax.set_visible(False)
            for ax, (backbone, selected) in zip(axes.flat, groups):
                _axes_style(ax)
                _curves(ax, [r for r in values if r['backbone'] == backbone], 'inference_k', metric, yscale=scale)
                _k_axis(ax, selected)
                if stem == 'adaptation_gain':
                    ax.axhline(0, color='0.5', lw=.8)
                else:
                    dense = next((r for r in selected if r['method'] == 'dense'), None)
                    if dense and dense[metric] is not None:
                        ax.axhline(dense[metric], color=COLORS['dense'], ls=':', lw=1.2)
                ax.set_ylabel(ylabel)
                ax.set_title(TITLES.get(backbone, backbone), loc='left')
            _legend(fig, values)
            _save(fig, path.parent, stem)
    (path.parent / 'figure_notes.txt').write_text(
        'Solid/filled: fine-tuned SAE. Dashed/open: frozen SAE. Dotted/triangle: measured dense baseline.\n'
        'Curves connect increasing inference K. No interpolation, smoothing or published scores.\n'
        'Scores are percentages in figures and fractions in CSV/JSON; gains are percentage points.\n'
        'Single seed; no confidence intervals claimed. CSRv2 uses dominant patch labels during adaptation.\n'
        'Key storage counts feature keys only. Search timing excludes extraction/encoding and may include\n'
        'cached measurements; it is not an end-to-end latency benchmark or proof of sparse-kernel speedup.\n',
        encoding='utf-8')


def suite_report(reports, workspace, plan):
    rows, campaigns, adaptations = [], [], []
    for report in reports:
        path = Path(report['report']) / 'results.json'
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not payload.get('complete'):
            raise ValueError('cannot combine incomplete scene reports')
        rows.extend(_rows(payload))
        campaigns.append(payload['campaign'])
        adaptations.extend(payload.get('adaptations', []))
    campaign = {'version': 'voc-scene-suite-v1', 'plan': plan, 'campaigns': campaigns}
    output = Path(workspace) / 'reports' / ('suite-' + canonical_hash(campaign)[:20]) / 'results.json'
    atomic_json(output, {'complete': True, 'campaign': campaign, 'adaptations': adaptations, 'results': rows})
    plot(output)
    return output
