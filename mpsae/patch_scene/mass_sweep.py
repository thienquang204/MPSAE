"""VOC MPSAE mass ablation; immutable plans, reusable fits and rebuildable reports.

Run with --plan before launch; --export-only rebuilds tables/index without training.
Partial masses retain the existing solver; s=1 uses balanced log-Sinkhorn.
"""
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from ..io import atomic_json, append_jsonl, canonical_hash, exclusive_lock, read_json
from ..transport import solver_name
from .config import Config, load, require_supported
from .report import expected_keys, row_key, NAMES, DATASET_NAMES

DEFAULT_MASSES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
ROOT = Path(__file__).resolve().parents[2]
SOLVER = 'partial-greenkhorn-v1'  # names existing equations; no solver change


def mass_configs(config, masses=DEFAULT_MASSES):
    if (not isinstance(masses, (list, tuple)) or not masses
            or any(isinstance(s, bool) or not isinstance(s, (int, float))
                   or not math.isfinite(s) or not 0 < s <= 1 for s in masses)
            or len(set(masses)) != len(masses)):
        raise ValueError('masses must be distinct finite numbers satisfying 0 < s <= 1')
    require_supported(config)
    if config.dataset_names != ('voc',) or config.methods != ('mpsae',) or config.include_dense:
        raise ValueError('Mass ablation requires VOC, methods=[mpsae], include_dense=false')
    if any(k > 64 for k in config.training_ks):
        raise ValueError('Use K<=64 to avoid the separate high-K MMPOT-weight policy')
    return [replace(config, mpsae_ot_mass=float(s)) for s in sorted(masses)]


def sweep_plan(configs):
    campaigns = [c.plan() for c in configs]
    identity = {'kind': 'mpsae-mass-v1', 'campaigns': [p['campaign'] for p in campaigns]}
    return {**identity, 'sweep_id': canonical_hash(identity)[:20],
            'masses': [c.mpsae_ot_mass for c in configs],
            'solver': SOLVER if all(c.mpsae_ot_mass < 1 for c in configs) else
                {str(c.mpsae_ot_mass): solver_name(c.mpsae_ot_mass) for c in configs},
            'sae_fits': sum(p['sae_fits'] for p in campaigns),
            'total_evaluations': sum(p['total_evaluations'] for p in campaigns),
            'matched_evaluations': sum(p['matched_evaluations'] for p in campaigns),
            'cross_k_evaluations': sum(p['cross_k_evaluations'] for p in campaigns),
            'campaign_plans': campaigns}


def _csv(path, rows, fields):
    # Atomic replacement prevents readers from observing half-written summaries.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(stream.getvalue(), encoding='utf-8')
    os.replace(temporary, path)


FIELDS = ['dataset', 'encoder', 'method', 'train_k', 'inference_k', 'seed',
          's', 'lambda', 'campaign_id', 'status', 'solver', 'checkpoint_path',
          'checkpoint_sha256', 'result_path', 'campaign_table_path', 'git_revision',
          'miou_percent', 'pixel_accuracy_percent', 'mean_nnz', 'gallery_mean_nnz',
          'epochs_completed', 'training_patches', 'error']


def campaign_records(workspace, config, campaign):
    """Reconstruct completed and pending conditions from durable campaign rows."""
    workspace = Path(workspace)
    folder = workspace/'campaigns'/campaign
    run_state = read_json(folder/'run-state.json') or {}
    runtime = read_json(folder/'runtime.json') or {}
    expected = expected_keys(config)
    lookup = {}
    # The coordinator saves these atomically before regenerating aggregate tables.
    for cell in sorted((folder/'cells').glob('*/rows.json')):
        for row in read_json(cell).get('results', []):
            key = row_key(row)
            if key not in expected or not row.get('complete') or row.get('seed') != config.seed:
                raise ValueError(f'Unexpected result in campaign {campaign}: {key}')
            if key in lookup:
                raise ValueError(f'Duplicate result in campaign {campaign}: {key}')
            lookup[key] = row
    if not lookup:
        # Also support restored exports whose cell receipts were not retained.
        for row in (read_json(folder/'table/results.json') or {}).get('results', []):
            key = row_key(row)
            if key not in expected or not row.get('complete') or row.get('seed') != config.seed or key in lookup:
                raise ValueError(f'Invalid restored result in campaign {campaign}: {key}')
            lookup[key] = row
    records = []
    for key in sorted(expected):
        dataset, encoder, method, train_k, infer_k = key
        row = lookup.get(key, {})
        training = row.get('training', {})
        relative_fit = training.get('path')
        summary = {}
        if relative_fit:
            fit_path = (workspace/relative_fit).resolve()
            if not fit_path.is_relative_to(workspace.resolve()):
                raise ValueError('Checkpoint path escapes workspace')
            summary = read_json(fit_path/'summary.json') or {}
            if method == 'mpsae' and summary:
                preset = summary['identity']['preset']
                requested = config.preset('mpsae', encoder)
                if preset['ot_mass'] != requested.ot_mass or preset['ot_weight'] != requested.ot_weight:
                    raise ValueError('Saved fit mass/weight differs from campaign settings')
                if summary.get('checkpoint_sha256') != row.get('identity', {}).get('fit'):
                    raise ValueError('Evaluation references a different checkpoint')
        metrics = (read_json(workspace/relative_fit/'progress.json') or {}).get('metrics', {}) if relative_fit else {}
        status = ('complete' if row else 'failed' if run_state.get('status') == 'failed'
                  else 'incomplete' if run_state.get('status') in ('running', 'interrupted', 'complete') or lookup
                  else 'pending')
        record = dict(zip(('dataset', 'encoder', 'method', 'train_k', 'inference_k'), key))
        record.update(seed=config.seed, s=config.mpsae_ot_mass if method == 'mpsae' else None,
                      **{'lambda': config.preset(method, encoder).ot_weight if method == 'mpsae' else None},
                      campaign_id=campaign, status=status,
                      solver=solver_name(config.mpsae_ot_mass) if method == 'mpsae' else None,
                      checkpoint_path=f'{relative_fit}/checkpoint.pt' if relative_fit else None,
                      checkpoint_sha256=row.get('identity', {}).get('fit'),
                      result_path=f"evaluations/{canonical_hash(row['identity'])[:20]}/result.json" if row.get('identity') else None,
                      campaign_table_path=f'campaigns/{campaign}/table/results.json',
                      git_revision=runtime.get('git_revision'),
                      miou_percent=100*row['miou'] if row else None,
                      pixel_accuracy_percent=100*row['pixel_accuracy'] if row else None,
                      mean_nnz=row.get('mean_nnz'), gallery_mean_nnz=row.get('gallery_mean_nnz'),
                      epochs_completed=training.get('epochs_completed'), training_patches=training.get('training_patches'),
                      error=run_state.get('error') if status == 'failed' else None,
                      diagnostics={k: v for k, v in metrics.items() if k.startswith('ot_') or k in ('loss_ot', 'loss_total')},
                      provenance=row.get('identity'))
        records.append(record)
    return records


def rebuild_index(workspace):
    """Scan all historical campaigns, not only this invocation's mass list."""
    workspace = Path(workspace)
    records, errors = [], []
    for settings in sorted((workspace/'campaigns').glob('*/settings/config.json')):
        try:
            plan = read_json(settings)
            config = Config(**plan['settings'])
            campaign = settings.parent.parent.name
            if campaign != config.plan()['campaign']:
                raise ValueError('Configuration/campaign ID mismatch')
            records.extend(campaign_records(workspace, config, campaign))
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append({'path': str(settings.relative_to(workspace)), 'error': str(error)})
    atomic_json(workspace/'experiment-index.json', {'experiments': records, 'errors': errors})
    _csv(workspace/'experiment-index.csv', records, FIELDS)
    return records, errors


def export_sweep(workspace, configs):
    workspace = Path(workspace)
    plan = sweep_plan(configs)
    directory = workspace/'ablations'/plan['sweep_id']
    directory.mkdir(parents=True, exist_ok=True)
    rows = [row for c in configs for row in campaign_records(workspace, c, c.plan()['campaign'])]
    completed = sum(r['status'] == 'complete' for r in rows)
    result = {'complete': completed == len(rows), 'completed': completed, 'expected': len(rows),
              'plan': plan, 'results': rows, 'selection': 'Report all masses; no best-mass selection.'}
    atomic_json(directory/'results.json', result)
    _csv(directory/'results.csv', rows, FIELDS)
    matched = [r for r in rows if r['train_k'] == r['inference_k']]
    _csv(directory/'matched_results.csv', matched, FIELDS)
    groups = {}
    for row in matched:
        key = (row['dataset'], row['encoder'], row['train_k'], row['seed'], row['lambda'])
        groups.setdefault(key, {})[row['s']] = row['miou_percent']
    comparisons = []
    for key, scores in sorted(groups.items()):
        comparisons.append({**dict(zip(('dataset', 'encoder', 'train_k', 'seed', 'lambda'), key)),
                            **{f's_{s:g}': scores.get(s) for s in plan['masses']}})
    fields = ['dataset', 'encoder', 'train_k', 'seed', 'lambda', *[f's_{s:g}' for s in plan['masses']]]
    _csv(directory/'matched_comparisons.csv', comparisons, fields)
    lines = [r'% Requires \usepackage{booktabs}.', r'\begin{table*}[t]', r'\centering',
             r'\caption{VOC MPSAE transported-mass ablation. Matched training/inference $K$; mIoU (\%).}',
             r'\begin{tabular}{llrrr'+'r'*len(plan['masses'])+'}', r'\toprule',
             r'Dataset & Encoder & $K$ & Seed & $\lambda$ & '+
             ' & '.join(f'$s={s:g}$' for s in plan['masses'])+r' \\', r'\midrule']
    for row in comparisons:
        values = [DATASET_NAMES[row['dataset']], NAMES[row['encoder']], str(row['train_k']),
                  str(row['seed']), f"{row['lambda']:g}"]
        values += ['--' if row[f's_{s:g}'] is None else f"{row[f's_{s:g}']:.2f}" for s in plan['masses']]
        lines.append(' & '.join(values)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\par\smallskip\footnotesize Single-seed results; -- denotes unavailable results. '
              r'All masses are reported. Full inference sweeps are in results.csv.', r'\end{table*}']
    (directory/'matched_comparisons.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    _, errors = rebuild_index(workspace)
    atomic_json(directory/'index-errors.json', errors)
    (directory/'README.txt').write_text(
        f"Sweep {plan['sweep_id']}: {completed}/{len(rows)} evaluations.\n"
        'results.csv: all train/inference K pairs, mass, weight, seed, status, and paths.\n'
        'matched_comparisons.csv/.tex: matched K comparison across masses.\n'
        'results.json: full records and last logged solver diagnostics (not a convergence certificate).\n'
        'Missing values are not zero. Individual campaign receipts are in campaigns/.\n'
        'experiment-index.csv/json includes historical campaigns, not only this sweep.\n'
        'No models or feature caches in this ZIP. Preserve the full workspace to resume.\n', encoding='utf-8')
    archive = directory/'mass-sweep-results.zip'
    temporary = archive.with_suffix('.zip.partial')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as bundle:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.csv', '.tex', '.txt'):
                bundle.write(path, path.name)
        for name in ('experiment-index.csv', 'experiment-index.json'):
            bundle.write(workspace/name, name)
        for campaign in plan['campaigns']:
            folder = workspace/'campaigns'/campaign
            for name in ('runtime.json', 'attempts.jsonl', 'run-state.json'):
                if (folder/name).exists():
                    bundle.write(folder/name, f'campaigns/{campaign}/{name}')
            for subdir in ('table', 'settings'):
                for path in sorted((folder/subdir).glob('*')):
                    if path.suffix in ('.json', '.csv', '.tex') and path.name != 'bundle.json':
                        bundle.write(path, f'campaigns/{campaign}/{subdir}/{path.name}')
    os.replace(temporary, archive)
    print(f'[mass sweep] {completed}/{len(rows)} evaluations; results: {archive}', flush=True)
    return result, archive


def _runtime(mass=0.9):
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    from ..io import sha256_file
    return {'git_revision': revision, 'git_dirty': dirty, 'python': sys.version,
            'solver': solver_name(mass), 'solver_sha256': sha256_file(ROOT/'mpsae/ot.py'),
            **({'balanced_solver_sha256': sha256_file(ROOT/'mpsae/ot_balanced.py')}
               if mass == 1 else {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/patch_scene_voc.yaml')
    parser.add_argument('--workspace', type=Path, default=Path('work/voc'))
    parser.add_argument('--masses', type=float, nargs='+', default=list(DEFAULT_MASSES))
    parser.add_argument('--encoders', nargs='+', choices=('siglip2', 'dinov2'), default=None)
    parser.add_argument('--weight', type=float, help='Fixed MMPOT coefficient; defaults to config')
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    # Default to SigLIP2; an explicit config/CLI encoder subset remains selectable.
    config = replace(load(args.config), methods=('mpsae',), include_dense=False,
                     encoders=tuple(args.encoders) if args.encoders else ('siglip2',))
    if args.weight is not None:
        config = replace(config, mpsae_ot_weight=args.weight)
    if args.cpu_threads < 1:
        parser.error('cpu-threads must be positive')
    configs = mass_configs(config, args.masses)
    plan = sweep_plan(configs)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    workspace = args.workspace.resolve()
    actual = workspace/'SMOKE_ONLY' if args.smoke else workspace
    directory = actual/'ablations'/plan['sweep_id']
    with exclusive_lock(actual/'.mass-sweep.lock', timeout=0):
        directory.mkdir(parents=True, exist_ok=True)
        plan_path = directory/'plan.json'
        old_plan = read_json(plan_path)
        if old_plan is not None and canonical_hash(old_plan) != canonical_hash(plan):
            raise RuntimeError('Immutable sweep plan differs from existing plan')
        if old_plan is None:
            atomic_json(plan_path, plan)
        for c in configs:
            folder = actual/'campaigns'/c.plan()['campaign']
            if not (folder/'settings/config.json').exists():
                atomic_json(folder/'settings/config.json', c.plan())
        if args.export_only:
            export_sweep(actual, configs)
            return
        export_sweep(actual, configs)
        import yaml
        for c in configs:
            campaign = c.plan()['campaign']
            folder = actual/'campaigns'/campaign
            config_file = directory/f'config-{campaign}.yaml'
            config_file.write_text(yaml.safe_dump(c.identity()['settings'], sort_keys=False), encoding='utf-8')
            runtime = _runtime(c.mpsae_ot_mass)
            if not (folder/'runtime.json').exists():
                atomic_json(folder/'runtime.json', runtime)
            append_jsonl(folder/'attempts.jsonl', {**runtime, 'started_at': datetime.now(timezone.utc).isoformat()})
            state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()}
            atomic_json(folder/'run-state.json', state)
            print(f'[mass sweep] s={c.mpsae_ot_mass:g}; weights={c.plan()["mpsae_ot_weights"]}; campaign={campaign}', flush=True)
            try:
                command = [sys.executable, '-u', '-m', 'mpsae.patch_scene', 'smoke' if args.smoke else 'run',
                           '--config', str(config_file), '--workspace', str(workspace), '--devices', *args.devices,
                           '--cpu-threads', str(args.cpu_threads)]
                subprocess.run(command, cwd=ROOT, check=True)
                rows = campaign_records(actual, c, campaign)
                if any(r['status'] != 'complete' for r in rows):
                    raise RuntimeError('Campaign returned without all expected evaluations')
                state.update(status='complete')
            except BaseException as error:
                state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
                raise
            finally:
                state['finished_at'] = datetime.now(timezone.utc).isoformat()
                atomic_json(folder/'run-state.json', state)
                export_sweep(actual, configs)


if __name__ == '__main__':
    main()
