"""Sequential scene suites sharing one cache root; each report remains independent."""
from pathlib import Path
from ..io import atomic_json, canonical_hash, exclusive_lock


def run_suite(config, args):
    from .server import bootstrap
    from .run import run, source_records
    workspace = Path(args.workspace)
    summary = {'complete': False, 'plan': config.plan(), 'reports': []}
    index = workspace / 'suites' / (canonical_hash(config.plan())[:20] + '.json')
    with exclusive_lock(workspace / '.scene.lock'):
        if args.command != 'bootstrap':
            source_records(config, args.source_root)
        for group in config.groups:
            print(f'[scene suite] {group.backbone}: inference K={group.inference_k}', flush=True)
            if args.command == 'bootstrap':
                result = bootstrap(group, workspace, repo=args.repo, revision=args.revision,
                    prefix=args.prefix, device=args.device, refresh_source=args.refresh_source,
                    publish=args.publish)
            else:
                result = run(group, workspace, args.source_root, device=args.device,
                             prepare_only=args.command == 'prepare')
            summary['reports'].append({'backbone': group.backbone, **result})
            atomic_json(index, summary)
        summary['complete'] = True
        atomic_json(index, summary)
        atomic_json(workspace / 'latest_suite.json', {'index': str(index), 'reports': summary['reports']})
    return summary
