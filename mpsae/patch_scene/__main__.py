"""Kaggle patch SAE train/inference grids, with independent workers per GPU."""
import argparse
from pathlib import Path
import json

from ..io import atomic_json, canonical_hash, exclusive_lock, read_json
from .config import load, ENCODERS, DATASETS, METHODS
from .artifacts import Hub


def run_cell(config, encoder, dataset, sources, hub, device, campaign, *, smoke=False):
    from .execution import Task, TaskRunner, evaluation_tasks, record_result
    runner = TaskRunner(config, sources, hub, device, smoke=smoke)
    prepare = Task('prepare', encoder, dataset)
    emit = lambda result: record_result(hub, campaign, result)
    runner.run(prepare, emit)
    for task in evaluation_tasks(config, prepare):
        runner.run(task, emit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'export', 'restore-table', 'smoke', 'doctor'))
    parser.add_argument('--config', default='configs/patch_scene_kaggle.yaml')
    parser.add_argument('--workspace', default='work/patch-scene-v2')
    parser.add_argument('--encoder', choices=(*ENCODERS, 'all'), default='all')
    parser.add_argument('--dataset', choices=(*DATASETS, 'all'), default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', nargs='+', help='Independent workers, e.g. cuda:0 cuda:1; overrides --device')
    parser.add_argument('--cpu-threads', type=int, default=2, help='CPU threads per parallel GPU worker')
    parser.add_argument('--repo', help='Optional HF artifact repository; omitted means local artifacts only')
    parser.add_argument('--publish', action='store_true', help='Back up shards, fits, evaluation progress and tables')
    args = parser.parse_args()
    config = load(args.config)
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
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
    devices = args.devices or [args.device]
    if len(devices) > 1 and (args.repo or args.publish):
        parser.error('multi-GPU runs currently save locally; omit --repo/--publish')
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
        encoders = list(config.encoder_names) if args.encoder == 'all' else [args.encoder]
        if any(e not in config.encoder_names for e in encoders):
            parser.error('--encoder must be selected in the configuration')
        datasets = list(DATASETS) if args.dataset == 'all' else [args.dataset]
        if len(devices) > 1:
            from .scheduler import validate_devices
            devices = validate_devices(devices, smoke=smoke)
        if not smoke:
            if len(devices) == 1:
                from ..scene.server import doctor
                doctor(devices[0])
            from .features import resolve_sources
            sources = resolve_sources(config, hub)
        else:
            sources = {}
            import torch
            torch.set_num_threads(1)
        atomic_json(workspace/'status.json', {'stage': 'running', 'campaign': campaign, 'devices': devices,
                    'encoders': encoders, 'datasets': datasets, 'smoke': smoke})
        try:
            if len(devices) > 1:
                from .scheduler import run_parallel
                run_parallel(config, sources, hub, campaign, encoders, datasets, devices,
                             smoke=smoke, cpu_threads=args.cpu_threads)
            else:
                for dataset in datasets:
                    for encoder in encoders:
                        run_cell(config, encoder, dataset, sources, hub, devices[0], campaign, smoke=smoke)
        except BaseException:
            atomic_json(workspace/'status.json', {'stage': 'interrupted_or_failed', 'campaign': campaign, 'smoke': smoke})
            raise
        atomic_json(workspace/'status.json', {'stage': 'selected_jobs_complete', 'campaign': campaign, 'smoke': smoke})


if __name__ == '__main__':
    main()
