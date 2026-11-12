"""python -m mpsae.scene: opt-in evaluation with no training side effects."""
import argparse
import json
from pathlib import Path, PurePosixPath

from .config import load_config


def restore(config, *, repo, revision, prefix, cache_dir):
    """Fetch selected completed fits + tiny provenance sidecars, never ImageNet arrays."""
    from huggingface_hub import HfApi, snapshot_download
    prefix = prefix.strip('/')
    if not prefix or '..' in PurePosixPath(prefix).parts or '\\' in prefix:
        raise ValueError('prefix must be a relative Hub folder such as direct-v1')
    info = HfApi().dataset_info(repo, revision=revision)
    commit = info.sha
    patterns = [f'{prefix}/runs/{cell.slug}/{name}' for cell in config.cells()
                for name in ('checkpoint.pt', 'summary.json', 'config.json')]
    patterns += [f'{prefix}/embeddings/{b}/*.json' for b in sorted({c.backbone for c in config.cells()})]
    patterns += [f'{prefix}/embeddings/*.json']
    available = {entry.rfilename for entry in info.siblings}
    missing = [name for name in patterns if '*' not in name and name not in available]
    if missing:
        raise FileNotFoundError(f'{repo}@{commit} is missing required completed-fit artifacts: '
                                + ', '.join(missing) + '. No training will be started.')
    root = snapshot_download(repo, repo_type='dataset', revision=commit,
                             allow_patterns=patterns, cache_dir=cache_dir)
    from .run import source_records
    source = Path(root) / prefix
    source_records(config, source)
    return {'source_root': str(source), 'revision': commit,
            'note': 'Pass this source_root to prepare/run; no ImageNet feature arrays downloaded.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'restore', 'prepare', 'run', 'plot', 'publish',
                                            'bootstrap', 'doctor'))
    parser.add_argument('--config', default='configs/scene_voc.yaml')
    parser.add_argument('--workspace', default='work/scene-v1')
    parser.add_argument('--source-root', default='work/direct-v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repo', default='quangnguyenopt/MPSAE')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--prefix', default='direct-v1')
    parser.add_argument('--results', help='Path to one completed results.json (plot/publish)')
    parser.add_argument('--env-file', default='.env')
    parser.add_argument('--refresh-source', action='store_true', help='Explicitly resolve the latest source revision again')
    parser.add_argument('--publish', action='store_true', help='Upload completed bootstrap reports to --repo')
    parser.add_argument('--expected-cuda', help='Doctor: require this torch CUDA runtime, e.g. 12.8')
    args = parser.parse_args()
    if args.command not in ('plan', 'doctor'):
        # Must happen before importing huggingface_hub (it reads environment on import).
        from ..cli import load_dotenv
        load_dotenv(args.env_file)
    if args.command in ('bootstrap', 'run', 'prepare'):
        from ..logs import setup_logging
        setup_logging(Path(args.workspace) / 'logs', name='scene')
    if args.command == 'doctor':
        from .server import doctor
        result = doctor(args.device, args.expected_cuda)
    elif args.command in ('plot', 'publish'):
        if not args.results:
            parser.error('--results is required')
        path = Path(args.results)
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not payload.get('complete'):
            raise ValueError('report is incomplete')
        if args.command == 'plot':
            from .report import plot
            plot(path)
            result = {'plots': str(path.parent)}
        else:
            from .server import publish_results
            result = {'commit': publish_results(path, args.repo)}
    else:
        config = load_config(args.config)
        if args.command == 'plan':
            result = config.plan()
        elif args.command == 'restore':
            result = restore(config, repo=args.repo, revision=args.revision, prefix=args.prefix,
                             cache_dir=str(Path(args.workspace) / 'hf_checkpoints'))
        elif hasattr(config, 'groups'):
            from .suite import run_suite
            result = run_suite(config, args)
        elif args.command == 'bootstrap':
            from .server import bootstrap
            from ..io import exclusive_lock
            with exclusive_lock(Path(args.workspace) / '.scene.lock'):
                result = bootstrap(config, args.workspace, repo=args.repo, revision=args.revision,
                                   prefix=args.prefix, device=args.device,
                                   refresh_source=args.refresh_source, publish=args.publish)
        else:
            from .run import run
            from ..io import exclusive_lock
            with exclusive_lock(Path(args.workspace) / '.scene.lock'):
                result = run(config, args.workspace, args.source_root, device=args.device,
                             prepare_only=args.command == 'prepare')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
