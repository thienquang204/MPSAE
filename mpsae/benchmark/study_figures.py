"""Small-multiple figures; each curve has one fixed training recipe."""
import math
import numpy as np

COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}


def save(fig, folder, name, scientific=True):
    import matplotlib.pyplot as plt
    if not scientific:
        fig.suptitle('SYNTHETIC SMOKE TEST — NOT BENCHMARK RESULTS', color='red', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(folder/f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def extended_figures(folder, summary, scientific=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .retrieval_studies_report import NAMES
    with plt.rc_context(STYLE):
        for study, parameter, symbol in (('mass', 's', '$s$'), ('optimizer', 'lambda', r'$\lambda$')):
            rows = [r for r in summary if r['study'] == study]
            panels = sorted({(r['backbone'], r['train_k'], r['batch_size']) for r in rows})
            if not panels:
                continue
            columns = min(2, len(panels))
            for metric, ylabel, suffix in (('top1_mean', '1-NN accuracy (%)', 'inference'),
                    ('gallery_mean_nnz', 'Mean active latents', 'active_latents'),
                    ('empty_queries', 'Empty queries (count)', 'empty_queries')):
                fig, axes = plt.subplots(math.ceil(len(panels)/columns), columns,
                    figsize=(4.5*columns, 3.25*math.ceil(len(panels)/columns)), squeeze=False)
                for ax, (backbone, kt, batch) in zip(axes.flat, panels):
                    points = [r for r in rows if (r['backbone'], r['train_k'], r['batch_size']) == (backbone, kt, batch)]
                    for i, value in enumerate(sorted({r[parameter] for r in points})):
                        line = sorted((r for r in points if r[parameter] == value), key=lambda r: r['inference_k'])
                        xs = [r['inference_k'] for r in line]
                        ys = [np.nan if r[metric] is None else r[metric] for r in line]
                        ax.plot(xs, ys, marker=('o','s','^','D','v')[i % 5], color=COLORS[i % 5],
                                label=f'{symbol}={value:g}', markersize=4, linewidth=1.5)
                        if metric == 'top1_mean':
                            for r in line:
                                if r['top1_std'] is not None:
                                    ax.errorbar(r['inference_k'], r[metric], yerr=r['top1_std'],
                                                color=COLORS[i % 5], capsize=2)
                    ticks = sorted({r['inference_k'] for r in points})
                    ax.set_xscale('log', base=2)
                    ax.set_xticks(ticks, [str(x) for x in ticks])
                    ax.set(title=f'{NAMES[backbone]} · train K={kt}', xlabel='Inference K', ylabel=ylabel)
                    ax.grid(axis='y', alpha=.2)
                    ax.legend(frameon=False, fontsize=8)
                for ax in list(axes.flat)[len(panels):]:
                    ax.set_visible(False)
                save(fig, folder, f'{study}_{suffix}', scientific)

            # Heatmap exposes interactions without choosing the best validation setting.
            fig, axes = plt.subplots(1, len(panels), figsize=(3.6*len(panels), 3.4), squeeze=False)
            finite = [r['top1_mean'] for r in rows if r['top1_mean'] is not None]
            for ax, panel in zip(axes.flat, panels):
                points = [r for r in rows if (r['backbone'], r['train_k'], r['batch_size']) == panel]
                xs, ys = sorted({r['inference_k'] for r in points}), sorted({r[parameter] for r in points})
                values = np.full((len(ys), len(xs)), np.nan)
                for r in points:
                    if r['top1_mean'] is not None:
                        values[ys.index(r[parameter]), xs.index(r['inference_k'])] = r['top1_mean']
                im = ax.imshow(np.ma.masked_invalid(values), cmap='cividis', aspect='auto',
                               vmin=min(finite) if finite else 0, vmax=max(finite) if finite else 100)
                ax.set_xticks(range(len(xs)), xs)
                ax.set_yticks(range(len(ys)), [f'{y:g}' for y in ys])
                ax.set(xlabel='Inference K', ylabel=symbol, title=f'{NAMES[panel[0]]}\nTrain K={panel[1]}')
                for yi in range(len(ys)):
                    for xi in range(len(xs)):
                        v = values[yi, xi]
                        ax.text(xi, yi, '--' if np.isnan(v) else f'{v:.2f}', ha='center', va='center',
                                fontsize=8, color='white' if finite and v < (min(finite)+max(finite))/2 else 'black')
                fig.colorbar(im, ax=ax, label='1-NN (%)', shrink=.8)
            save(fig, folder, f'{study}_heatmap', scientific)

        # Compare training K only where mass, weight, batch and inference K all agree.
        points = [r for r in summary if r['inference_k'] == 2 and r['s'] == 1 and
                  r['lambda'] == 2 and r['batch_size'] == 1024]
        backbones = sorted({r['backbone'] for r in points})
        if backbones:
            fig, axes = plt.subplots(1, len(backbones), figsize=(4*len(backbones), 3.3), squeeze=False)
            for ax, b in zip(axes.flat, backbones):
                series = sorted((r for r in points if r['backbone'] == b), key=lambda r: r['train_k'])
                ax.plot([r['train_k'] for r in series],
                        [np.nan if r['top1_mean'] is None else r['top1_mean'] for r in series],
                        color=COLORS[0], marker='o')
                ax.set_xticks([r['train_k'] for r in series])
                ax.set(title=NAMES[b], xlabel='Training K', ylabel='1-NN at inference K=2 (%)')
                ax.grid(axis='y', alpha=.2)
            save(fig, folder, 'k2_training_comparison', scientific)
