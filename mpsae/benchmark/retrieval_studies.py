"""Two local ImageNet MPSAE studies sharing fits, evaluations and mapped embeddings."""
from dataclasses import asdict
import json
from pathlib import Path

from ..io import atomic_json, canonical_hash, exclusive_lock, read_json
from .mass_config import MassConfig
from . import mass_sweep


def conditions(settings, studies=('mass', 'optimizer')):
    allowed = {'train_k', 'seeds', 'epochs', 'lr', 'mass_values', 'mass_batch_size', 'mass_weight',
               'optimizer_mass', 'optimizer_batches', 'optimizer_weights', 'ot_eta', 'ot_microbatch'}
    if set(settings) - allowed:
        raise ValueError(f'Unknown suite settings: {set(settings)-allowed}')
    if not settings['train_k'] or set(settings['train_k']) - {'ff2048', 'vit_b16'}:
        raise ValueError('This suite supports only ResNet-50 (ff2048) and ViT-B/16 (vit_b16)')
    configs = []
    common = {k: settings[k] for k in ('seeds', 'epochs', 'lr', 'ot_eta', 'ot_microbatch')}
    for b, ks in settings['train_k'].items():
        base = dict(backbones=(b,), train_k={b: ks}, **common)
        if 'mass' in studies:
            configs.append(('mass', MassConfig(**base, masses=settings['mass_values'],
                batch_size=settings['mass_batch_size'], ot_weight=settings['mass_weight'])))
        if b == 'ff2048' and 'optimizer' in studies:
            batches, weights = settings['optimizer_batches'], settings['optimizer_weights']
            if not batches or not weights or len(set(batches)) != len(batches) or len(set(weights)) != len(weights):
                raise ValueError('Optimizer batches/weights must be nonempty and distinct')
            for batch in sorted(batches):
                for weight in sorted(weights):
                    configs.append(('optimizer', MassConfig(**base, masses=(settings['optimizer_mass'],),
                                    batch_size=batch, ot_weight=weight)))
    if not configs:
        raise ValueError('No conditions selected')
    return configs


def plan_for(settings, studies=('mass', 'optimizer')):
    configs = conditions(settings, studies)
    unique, totals = set(), {}
    for name, config in configs:
        totals[name] = totals.get(name, 0) + config.plan()['fits']
        for cell, mass in config.cells():
            unique.add(canonical_hash({'cell': cell.slug, 'preset': asdict(config.spec(cell, mass).preset)}))
    identity = {'kind': 'imagenet-two-studies-v1', 'settings': settings, 'studies': list(studies)}
    return {**identity, 'suite_id': canonical_hash(identity)[:20], 'unique_fits': len(unique),
            'unique_evaluations': len(unique), 'study_conditions': totals,
            'overlapping_conditions': sum(totals.values())-len(unique),
            'campaigns': [{'study': name, **config.plan()} for name, config in configs],
            'protocol': 'ImageNet exact 1-NN; matched training/inference K; MPSAE only',
            'optimizer_comparison': 'equal epochs, not equal updates; Adam settings fixed; no LR scaling'}


def selected_rows(root, plan):
    rows = []
    for campaign in plan['campaigns']:
        config = MassConfig(**campaign['config'])
        payload = read_json(root/'sweeps'/campaign['sweep_id']/'results.json') or {}
        lookup = {(r['backbone'], r['train_k'], r['seed'], r['s']): r for r in payload.get('results', [])}
        for cell, mass in config.cells():
            preset = config.spec(cell, mass).preset
            row = lookup.get((cell.backbone, cell.k, cell.seed, mass), {})
            rows.append({**row, 'study': campaign['study'], 'backbone': cell.backbone, 'method': 'mpsae',
                         'train_k': cell.k, 'inference_k': cell.k, 'seed': cell.seed, 's': mass,
                         'lambda': preset.ot_weight, 'batch_size': preset.batch_size, 'epochs': preset.epochs,
                         'status': row.get('status', 'pending'), 'campaign': campaign['sweep_id']})
    return rows


def download_features(args, backbones):
    from ..cli import load_dotenv
    from ..embeddings_hub import Hub, DEFAULT_REPO, sync, link_campaign
    import os
    if args.embeddings_dir != args.source_root/'embeddings':
        raise ValueError('Downloads require --source-root/embeddings; omit --embeddings-dir')
    load_dotenv('.env')
    pairs = [('imagenet', b) for b in backbones]
    hub_root = args.source_root.parent/'embeddings-hub'
    hub = Hub.connect(args.embedding_repo or os.environ.get('MPSAE_EMBEDDINGS_REPO', DEFAULT_REPO))
    sync(pairs, hub_root, args.source_root.parent, hub, download_only=True)
    link_campaign(hub_root, args.source_root, pairs)


def run(settings, args):
    from .retrieval_studies_report import export
    plan = {**plan_for(settings, args.studies), 'scientific': not args.smoke}
    configs = conditions(settings, args.studies)
    root = args.workspace/('SMOKE_ONLY' if args.smoke else '')
    with exclusive_lock(root/'.retrieval-studies.lock', timeout=0):
        output = root/'studies'/plan['suite_id']
        atomic_json(output/'plan.json', plan)
        if args.export_only:
            return export(root, plan, selected_rows(root, plan))
        from ..features import load_bundle
        from ..config import BACKBONES
        from ..logs import setup_logging
        setup_logging(root/'logs', name='retrieval-studies')
        state = {'status': 'running'}
        atomic_json(output/'run-state.json', state)
        try:
            backbones = sorted({c.backbones[0] for _, c in configs})
            if args.download_embeddings and not args.smoke:
                download_features(args, backbones)
            for backbone in backbones:
                selected = [(name, c) for name, c in configs if c.backbones == (backbone,)]
                config = selected[0][1]
                cell, mass = config.cells()[0]
                bundle = (mass_sweep._synthetic(config.spec(cell, mass, smoke=True)) if args.smoke else
                          load_bundle(BACKBONES[backbone], args.embeddings_dir, args.cache_dir))
                try:
                    for name, config in selected:
                        print(f'\n[study {name}] {backbone}; batch={config.batch_size}; '
                              f'lambda={config.ot_weight}; masses={config.masses}', flush=True)
                        try:
                            mass_sweep.run(config, args, bundles={backbone: bundle})
                        finally:
                            export(root, plan, selected_rows(root, plan))
                finally:
                    for split in (bundle.train, bundle.validation):
                        mapping = getattr(split.features, '_mmap', None)
                        if mapping is not None:
                            mapping.close()
                    del bundle
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                         error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if state['status'] == 'running':
                state['status'] = 'finished'
            atomic_json(output/'run-state.json', state)
            result = export(root, plan, selected_rows(root, plan))
        return result


def main(argv=None):
    import yaml
    parser = mass_sweep.argument_parser()
    parser.description = __doc__
    parser.set_defaults(config='configs/retrieval_studies.yaml', workspace=Path('work/retrieval-studies'),
                        checkpoint_every=512, log_every=100, query_block=2048)
    parser.add_argument('--studies', nargs='+', choices=('mass', 'optimizer'), default=['mass', 'optimizer'])
    parser.add_argument('--optimizer-batches', nargs='+', type=int)
    parser.add_argument('--optimizer-weights', nargs='+', type=float)
    parser.add_argument('--optimizer-mass', type=float)
    args = parser.parse_args(argv)
    settings = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.backbones:
        settings['train_k'] = {b: settings['train_k'].get(b, [8] if b == 'ff2048' else [12]) for b in args.backbones}
    if args.train_k:
        settings['train_k'].update(mass_sweep._budgets(args.train_k))
    for source, target in (('masses', 'mass_values'), ('batch_size', 'mass_batch_size'), ('ot_weight', 'mass_weight'),
                           ('seeds', 'seeds'), ('epochs', 'epochs'), ('lr', 'lr'), ('ot_eta', 'ot_eta'),
                           ('ot_microbatch', 'ot_microbatch'), ('optimizer_batches', 'optimizer_batches'),
                           ('optimizer_weights', 'optimizer_weights'), ('optimizer_mass', 'optimizer_mass')):
        if getattr(args, source) is not None:
            settings[target] = getattr(args, source)
    try:
        if args.inference_k:
            raise ValueError('These studies use matched K; use mass_sweep for cross-K inference')
        if len(set(args.studies)) != len(args.studies):
            raise ValueError('Duplicate study selection')
        for name in ('cpu_threads', 'encode_batch_size', 'gallery_chunk', 'query_block', 'log_every', 'checkpoint_every'):
            if getattr(args, name) < 1:
                raise ValueError(f'{name} must be positive')
        plan = plan_for(settings, args.studies)
    except ValueError as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    args.workspace = args.workspace.resolve()
    args.source_root = args.source_root.resolve()
    args.embeddings_dir = (args.embeddings_dir or args.source_root/'embeddings').resolve()
    args.cache_dir = (args.cache_dir or args.source_root/'cache').resolve()
    result = run(settings, args)
    return 0 if result['complete'] or args.export_only else 1


if __name__ == '__main__':
    raise SystemExit(main())
