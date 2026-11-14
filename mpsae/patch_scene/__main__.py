"""Kaggle patch campaign: train K=12 once; reuse each SAE across inference K."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json

from ..io import atomic_json, canonical_hash, exclusive_lock, read_json
from .config import load, ENCODERS, DATASETS, METHODS
from .artifacts import Hub


def run_cell(config, encoder, dataset, sources, hub, device, campaign, *, smoke=False):
    import torch
    from .features import prepare, training_bank
    from .train import fit
    from .evaluate import evaluate
    from .report import collect_tables, row_key
    if smoke:
        from .smoke import fixture
        manifest, features, labels, selected = fixture(hub, config, encoder, dataset)
    else:
        _, manifest = prepare(config, encoder, dataset, sources, hub, device)
        features, labels, selected = training_bank(manifest, hub, config)
    cell_rel = f'campaigns/{campaign}/cells/{dataset}-{encoder}'
    cell_folder = hub.pull(cell_rel)
    rows = (read_json(cell_folder/'rows.json') or {}).get('results', [])
    for method in ('dense', *METHODS):
        model, record = None, None
        if method != 'dense':
            preset = replace(config.preset(), hidden_dim=64, epochs=1, batch_size=16,
                             k_aux=32, ot_microbatch=4) if smoke else None
            model, record = fit(config, method, features, labels, manifest['identity'], hub, device, preset=preset)
        # Finish/reuse train-K first, then extend without fitting another model.
        grid = sorted(config.inference_ks, key=lambda k: (k != config.k, k)) if model is not None else (None,)
        for k in grid:
            print(f'[job {dataset}/{encoder}/{method}] train K={config.k if model is not None else None}; inference K={k}', flush=True)
            result = evaluate(config, encoder, dataset, method, model, record, features, labels,
                              selected, manifest, hub, device, inference_k=k)
            rows = [r for r in rows if row_key(r) != row_key(result)]
            rows.append(result)
            atomic_json(cell_folder/'rows.json', {'results': rows})
            hub.push(cell_rel)
            table_folder = collect_tables(hub, campaign)
        del model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    print(f'[table] {table_folder / "results.tex"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'export', 'restore-table', 'smoke', 'doctor'))
    parser.add_argument('--config', default='configs/patch_scene_kaggle.yaml')
    parser.add_argument('--workspace', default='work/patch-scene-v2')
    parser.add_argument('--encoder', choices=(*ENCODERS, 'all'), default='dinov2')
    parser.add_argument('--dataset', choices=(*DATASETS, 'all'), default='voc')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repo', help='Optional HF artifact repository; omitted means local artifacts only')
    parser.add_argument('--publish', action='store_true', help='Back up shards, fits, evaluation progress and tables')
    args = parser.parse_args()
    config = load(args.config)
    if args.command == 'plan':
        print(json.dumps(config.plan(), indent=2))
        return
    if args.command == 'doctor':
        from ..scene.server import doctor
        import transformers
        print(json.dumps({'transformers': transformers.__version__, **doctor(args.device)}, indent=2))
        return
    if args.publish and not args.repo:
        parser.error('--publish requires --repo')
    smoke = args.command == 'smoke'
    if smoke and (args.repo or args.publish):
        parser.error('smoke uses synthetic data; publishing is prohibited')
    workspace = Path(args.workspace)/'SMOKE_ONLY' if smoke else Path(args.workspace)
    campaign = canonical_hash(config.identity())[:20]
    with exclusive_lock(workspace/'.campaign.lock', timeout=0):
        hub = Hub(workspace, args.repo, publish=args.publish)
        settings_rel = f'campaigns/{campaign}/settings'
        atomic_json(hub.root/settings_rel/'config.json', config.plan())
        hub.push(settings_rel)
        if args.command in ('export', 'restore-table'):
            from .report import collect_tables
            path = collect_tables(hub, campaign)/'results.tex'
            print(path)
            return
        print('[artifacts] ' + ('optional HF repository enabled' if args.repo else
              f'local only: {workspace}; no account/token or uploads required'), flush=True)
        if not smoke:
            from ..scene.server import doctor
            doctor(args.device)
            from .features import resolve_sources
            sources = resolve_sources(config, hub)
        else:
            sources = {}
            import torch
            torch.set_num_threads(1)
        encoders = list(ENCODERS) if args.encoder == 'all' else [args.encoder]
        datasets = list(DATASETS) if args.dataset == 'all' else [args.dataset]
        for dataset in datasets:
            for encoder in encoders:
                atomic_json(workspace/'status.json', {'stage': 'running', 'dataset': dataset, 'encoder': encoder,
                            'campaign': campaign, 'smoke': smoke})
                run_cell(config, encoder, dataset, sources, hub, args.device, campaign, smoke=smoke)
        atomic_json(workspace/'status.json', {'stage': 'selected_jobs_complete', 'campaign': campaign, 'smoke': smoke})


if __name__ == '__main__':
    main()
