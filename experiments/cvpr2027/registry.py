"""Expand the paper recipes using the existing, torch-free configuration classes."""
from __future__ import annotations

from dataclasses import asdict
from itertools import product

from common import HERE, digest, inside, merge, read_json, read_yaml

SELECTORS = ('task', 'dataset', 'encoder', 'method', 'seed', 'train_k', 'inference_k',
             'mass', 'loss_weight', 'batch_size', 'epochs', 'lr')


def condition_key(row):
    return {k: row.get(k) for k in SELECTORS}


def condition_id(row):
    return digest(condition_key(row))[:20]


def _point(task, dataset, encoder, method, seed, kt, ki, preset, protocol):
    sparse = method != 'dense'
    p = asdict(preset) if preset is not None else {}
    row = dict(task=task, dataset=dataset, encoder=encoder, method=method, seed=seed,
               train_k=kt, inference_k=ki,
               mass=p.get('ot_mass') if method == 'mpsae' else None,
               loss_weight=p.get('ot_weight') if method == 'mpsae' else None,
               batch_size=p.get('batch_size'), epochs=p.get('epochs'), lr=p.get('lr'),
               metric='miou_percent' if task == 'segmentation_nn' else 'top1_percent',
               preset=p if sparse else None, protocol=protocol)
    row['condition_id'] = condition_id(row)
    row['fit_key'] = digest({k: v for k, v in row.items()
                             if k not in ('task', 'inference_k', 'metric', 'condition_id')})[:20] if sparse else None
    return row


def job_conditions(runner, payload):
    from mpsae.config import BACKBONES, training_preset
    if runner == 'imagenet':
        from mpsae.benchmark.config import Campaign
        c = Campaign.from_mapping(payload)
        for cell in c.source_cells():
            preset = training_preset(cell.method, BACKBONES[cell.backbone].preset, k=cell.k)
            for ki in sorted(set(c.budgets('imagenet', 'eval_k', cell.backbone)) | {cell.k}):
                args = ('imagenet', cell.backbone, cell.method, cell.seed, cell.k, ki, preset)
                yield _point('imagenet_1nn', *args, 'imagenet-global-v1')
                if c.wants_probe(cell.backbone, ki):
                    yield _point('imagenet_probe', *args, 'imagenet-global-v1')
    elif runner == 'retrieval':
        from mpsae.benchmark.mass_config import MassConfig
        c = MassConfig(**payload)
        for cell, mass in c.cells():
            for ki in c.budgets(cell):
                yield _point('imagenet_1nn', 'imagenet', cell.backbone, cell.method,
                             cell.seed, cell.k, ki, c.spec(cell, mass).preset, 'imagenet-global-v1')
    elif runner == 'segmentation':
        from mpsae.patch_scene.config import Config, require_supported
        c = Config(**payload)
        require_supported(c)
        for dataset in c.dataset_names:
            for encoder in c.encoder_names:
                protocol = {k: getattr(c, k) for k in (
                    'image_size', 'image_width', 'all_train_patches', 'train_patches',
                    'support_patches', 'neighbors', 'temperature', 'cache_dtype',
                    'early_stopping', 'early_min_epochs', 'early_patience', 'early_min_delta')}
                if c.include_dense:
                    yield _point('segmentation_nn', dataset, encoder, 'dense', c.seed,
                                 None, None, None, protocol)
                for method in c.methods:
                    for kt, ki in c.evaluation_pairs:
                        yield _point('segmentation_nn', dataset, encoder, method, c.seed,
                                     kt, ki, c.preset(method, encoder), protocol)
    else:
        raise ValueError(f'Unknown runner: {runner}')


def jobs_for(study, folder=HERE):
    entry = read_yaml(folder / 'catalog.yaml')['studies'][study]
    recipe = read_yaml(inside(folder, entry['config']))
    unknown = set(recipe) - {'runner', 'base', 'variants', 'grid', 'description'}
    if unknown:
        raise ValueError(f'{study}: unknown recipe fields {unknown}')
    grid = recipe.get('grid', {})
    for variant in recipe.get('variants', [{}]):
        for values in product(*grid.values()):
            config = merge(merge(recipe['base'], variant), dict(zip(grid, values)))
            job = {'runner': recipe['runner'], 'config': config}
            yield {**job, 'job_id': digest(job)[:20], 'study': study}


def build_plan(studies=None, folder=HERE, *, check_lock=True):
    catalog = read_yaml(folder / 'catalog.yaml')
    selected = list(catalog['studies']) if not studies or studies == ['all'] else studies
    if len(selected) != len(set(selected)) or set(selected) - set(catalog['studies']):
        raise ValueError('Select distinct experiment names from catalog.yaml, or all')
    jobs, conditions, counts = {}, {}, {}
    for study in selected:
        points = set()
        fits = set()
        for job in jobs_for(study, folder):
            jobs.setdefault(job['job_id'], {**job, 'studies': []})['studies'].append(study)
            for row in job_conditions(job['runner'], job['config']):
                key = row['condition_id']
                if key in conditions and conditions[key]['protocol'] != row['protocol']:
                    raise ValueError(f'Conflicting protocols for {key}; version the experiment')
                conditions.setdefault(key, {**row, 'studies': [], 'job_ids': []})
                for field, value in (('studies', study), ('job_ids', job['job_id'])):
                    if value not in conditions[key][field]:
                        conditions[key][field].append(value)
                points.add(key)
                if row['fit_key']:
                    fits.add(row['fit_key'])
        counts[study] = {'evaluations': len(points), 'sae_fits': len(fits)}
        if counts[study] != catalog['studies'][study]['expected']:
            raise ValueError(f'{study}: counts changed: {counts[study]}')
    ordered = sorted(conditions.values(), key=lambda r: r['condition_id'])
    scientific = [{k: v for k, v in r.items() if k not in ('studies', 'job_ids')} for r in ordered]
    if check_lock:
        lock = read_json(folder / 'reference' / 'recipes.lock.json')
        if lock is None:
            raise ValueError('Missing recipes.lock.json; paper recipes must be reviewed and frozen')
        for row in scientific:
            if lock['conditions'].get(row['condition_id']) != digest(row):
                raise ValueError(f"Frozen recipe changed: {row['condition_id']}. Review the recipe lock before running.")
    return {'schema': 1, 'paper': catalog['paper'], 'studies': counts,
            'unique_evaluations': len(ordered),
            'unique_sae_fits': len({r['fit_key'] for r in ordered if r['fit_key']}),
            'probe_fits': sum(r['task'] == 'imagenet_probe' for r in ordered),
            'jobs': list(jobs.values()), 'conditions': ordered}


def table_conditions(plan, table, folder=HERE):
    spec = read_yaml(folder / 'paper_map.yaml')['tables'][str(table)]
    result = []
    for row in plan['conditions']:
        if not set(row['studies']) & set(spec['studies']):
            continue
        if any(row.get(k) not in (v if isinstance(v, list) else [v]) for k, v in spec.get('where', {}).items()):
            continue
        if spec.get('matched') and row['method'] != 'dense' and row['train_k'] != row['inference_k']:
            continue
        if row['method'] == 'mpsae' and any(row.get(k) not in spec[k] for k in ('mass', 'loss_weight') if k in spec):
            continue
        result.append(row)
    return spec, result
