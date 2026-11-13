"""Sequential scene suites sharing one cache root; each report remains independent."""
from pathlib import Path
from ..io import atomic_json, canonical_hash, exclusive_lock


def run_suite(config, args):
    from .server import bootstrap, publish_adaptation, publish_results
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
                             prepare_only=args.command == 'prepare',
                             on_adapted=(lambda r: publish_adaptation(Path(r['path']).parent, args.repo))
                                        if args.publish else None)
                if args.publish and args.command == 'run':
                    publish_results(Path(result['report']) / 'results.json', args.repo)
            summary['reports'].append({'backbone': group.backbone, **result})
            atomic_json(index, summary)
        summary['complete'] = True
        if args.command != 'prepare':
            from .report import suite_report
            combined = suite_report(summary['reports'], workspace, config.plan())
            if combined:
                summary['combined_report'] = str(combined.parent)
                if args.publish:
                    publish_results(combined, args.repo)
        atomic_json(index, summary)
        atomic_json(workspace / 'latest_suite.json', {'index': str(index), 'reports': summary['reports'],
                                                   'combined_report': summary.get('combined_report')})
    return summary
