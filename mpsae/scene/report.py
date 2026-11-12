"""Standalone figures; no published reference curves."""
import csv
import json
from pathlib import Path


def plot(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload['results']
    backbone = payload.get('campaign', {}).get('plan', {}).get('config', {}).get('backbone', '')
    for row in rows:
        row.setdefault('backbone', backbone)
    names = {'dense': 'Dense', 'csr': 'CSR', 'csrv2': 'CSRv2', 'mpsae': 'MPSAE'}
    with (path.parent / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        columns = ('backbone', 'method', 'train_k', 'inference_k', 'miou', 'pixel_accuracy',
                   'query_nnz_mean', 'query_empty_patches', 'index_bytes', 'search_seconds')
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
        titles = {'ff2048': 'ResNet-50', 'vit_b16': 'ViT-B/16', 'dinov2_vits14': 'DINOv2-S/14', 'convnext_tiny': 'ConvNeXt-Tiny'}
        train_k = next((r['train_k'] for r in rows if r['method'] != 'dense'), '')
        fig.suptitle(f'{titles.get(backbone, backbone)} | Train K={train_k}')
        dense = next(r for r in rows if r['method'] == 'dense')
        sparse = [r for r in rows if r['method'] != 'dense']
        if len({r['inference_k'] for r in sparse}) == 1:
            axes[0].bar([names[r['method']] for r in rows], [100*r['miou'] for r in rows])
            axes[0].set_xlabel(f'SAE K = {sparse[0]["inference_k"]}')
        else:
            for method in dict.fromkeys(r['method'] for r in sparse):
                selected = sorted((r for r in sparse if r['method'] == method), key=lambda r:r['inference_k'])
                axes[0].plot([r['inference_k'] for r in selected], [100*r['miou'] for r in selected],
                             marker='o', label=names[method])
            axes[0].axhline(100*dense['miou'], color='0.4', ls='--', label='Dense')
            axes[0].set_xscale('log', base=2)
            budgets = sorted({r['inference_k'] for r in sparse})
            axes[0].set_xticks(budgets, [str(k) for k in budgets])
            axes[0].set_xlabel('Inference K')
            axes[0].legend(fontsize=8)
        for method in dict.fromkeys(r['method'] for r in rows):
            selected = sorted((r for r in rows if r['method'] == method), key=lambda r:r['index_bytes'])
            axes[1].plot([r['index_bytes']/2**20 for r in selected], [100*r['miou'] for r in selected],
                         marker='o', label=names[method])
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Key storage (MiB)')
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_ylabel('mIoU (%)')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.2)
        for extension in ('pdf', 'png'):
            fig.savefig(path.parent / f'segmentation.{extension}', dpi=300)
        plt.close(fig)
