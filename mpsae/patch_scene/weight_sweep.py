"""Matched-K loss-weight ablations for all supported segmentation datasets.

Use --mass 1 for balanced MMOT, or 0<s<1 for partial MMPOT. Only MPSAE
changes across weights; the existing feature, baseline and fit caches are reused.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from ..io import atomic_json, append_jsonl, canonical_hash, exclusive_lock, read_json
from ..transport import solver_name
from .config import load, require_supported, METHODS, SUPPORTED_ENCODERS
from .grid import grid_configs, grid_plan, write_grid_tables
# The mass-ablation registry already records arbitrary methods/datasets. Reuse
# its provenance checks and historical register rather than create a second one.
from .mass_sweep import campaign_records, rebuild_index, _runtime
from .report import write_tables

ROOT = Path(__file__).resolve().parents[2]


def weight_plan(config, weights):
    require_supported(config)
    if 'mpsae' not in config.methods:
        raise ValueError('A transport-weight sweep requires MPSAE in methods')
    grid = grid_configs(config, weights)
    identity = {'kind': 'segmentation-weight-v1',
                'campaigns': [c.plan()['campaign'] for c in grid]}
    return {**grid_plan(config, grid), **identity,
            'sweep_id': canonical_hash(identity)[:20],
            'solver': solver_name(config.mpsae_ot_mass)}


def export_grid(workspace, config, weights):
    """Rebuild this entire grid from durable rows, never the last campaign ZIP."""
    workspace = Path(workspace)
    plan = weight_plan(config, weights)
    directory = workspace/'ablations'/plan['sweep_id']
    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    for c in grid_configs(config, weights):
        campaign = c.plan()['campaign']
        folder = workspace/'campaigns'/campaign
        campaign_records(workspace, c, campaign)  # validate fit mass, weight and SHA
        rows = [r for p in sorted((folder/'cells').glob('*/rows.json'))
                for r in read_json(p).get('results', [])]
        if rows:
            # Recover if interrupted between persisting a row and its table.
            write_tables(rows, folder/'table', c)
        receipt = read_json(folder/'table/results.json') or {'results': []}
        for dataset in config.dataset_names:
            receipts.append((c, dataset, {'results': [r for r in receipt['results']
                                                       if r['dataset'] == dataset]}))
    write_grid_tables(receipts, {d: config for d in config.dataset_names}, weights, directory)
    _, errors = rebuild_index(workspace)
    atomic_json(directory/'index-errors.json', errors)
    atomic_json(directory/'plan.json', plan)
    result = read_json(directory/'grid_results.json')
    (directory/'README.txt').write_text(
        f"Weight sweep {plan['sweep_id']}: {result['completed']}/{result['expected']} evaluations.\n"
        f"Fixed s={config.mpsae_ot_mass:g}; solver={plan['solver']}; lambdas={plan['ot_weights']}.\n"
        'grid_results.csv: every unique result with mass, weight, solver, seed, accuracy and nnz.\n'
        'grid_comparisons.csv and grid_results.tex: matched-K mIoU comparisons.\n'
        'CSR, CSRv2 and dense are shared across weights; only MPSAE uses transport.\n'
        'Missing scores are unavailable, never zero. Report all weights.\n'
        'experiment-index.csv/json includes status and checkpoint paths for historical campaigns.\n'
        'Runtime records describe invocations; reused fits may predate them.\n'
        'No weights or feature caches are included. Preserve the full workspace to resume.\n',
        encoding='utf-8')
    archive = directory/'weight-sweep-results.zip'
    temporary = archive.with_suffix('.zip.partial')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as bundle:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in ('.csv', '.json', '.tex', '.txt'):
                bundle.write(path, path.name)
        for name in ('experiment-index.csv', 'experiment-index.json'):
            bundle.write(workspace/name, name)
        for campaign in plan['campaigns']:
            folder = workspace/'campaigns'/campaign
            for name in ('runtime.json', 'attempts.jsonl', 'run-state.json'):
                if (folder/name).exists():
                    bundle.write(folder/name, f'campaigns/{campaign}/{name}')
            for subdir in ('settings', 'sources', 'table'):
                for path in sorted((folder/subdir).glob('*')):
                    if path.suffix in ('.csv', '.json', '.tex') and path.name != 'bundle.json':
                        bundle.write(path, f'campaigns/{campaign}/{subdir}/{path.name}')
    os.replace(temporary, archive)
    print(f"[weight sweep] {result['completed']}/{result['expected']} unique evaluations; {archive}", flush=True)
    return result, archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/patch_scene_voc.yaml')
    parser.add_argument('--workspace', type=Path, default=Path('work/voc'))
    parser.add_argument('--weights', nargs='+', type=float, default=[.5, 1., 2.])
    parser.add_argument('--mass', type=float, help='1 = balanced MMOT; omitted keeps config mass')
    parser.add_argument('--encoders', nargs='+', choices=SUPPORTED_ENCODERS)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--train-k', nargs='+', type=int)
    parser.add_argument('--no-dense', action='store_true')
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--nyuv2-root')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    try:
        overrides = {'evaluation_mode': 'matched', 'inference_k': None}
        for key, value in (('encoders', args.encoders), ('methods', args.methods),
                           ('train_k', args.train_k), ('mpsae_ot_mass', args.mass)):
            if value is not None:
                overrides[key] = value
        if args.no_dense:
            overrides['include_dense'] = False
        config = replace(load(args.config), **overrides)
        plan = weight_plan(config, args.weights)
        if args.cpu_threads < 1:
            raise ValueError('cpu-threads must be positive')
    except ValueError as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    workspace = args.workspace.expanduser().resolve()
    actual = workspace/'SMOKE_ONLY' if args.smoke else workspace
    directory = actual/'ablations'/plan['sweep_id']
    with exclusive_lock(actual/'.weight-sweep.lock', timeout=0):
        directory.mkdir(parents=True, exist_ok=True)
        previous = read_json(directory/'plan.json')
        if previous is not None and canonical_hash(previous) != canonical_hash(plan):
            raise RuntimeError('Immutable weight-sweep plan differs from existing plan')
        atomic_json(directory/'plan.json', plan)
        configs = grid_configs(config, args.weights)
        for c in configs:
            path = actual/f'campaigns/{c.plan()["campaign"]}/settings/config.json'
            if not path.exists():
                atomic_json(path, c.plan())
        export_grid(actual, config, args.weights)
        if args.export_only:
            return
        import yaml
        for c in configs:
            campaign = c.plan()['campaign']
            folder = actual/'campaigns'/campaign
            config_file = directory/f'config-{campaign}.yaml'
            config_file.write_text(yaml.safe_dump(c.identity()['settings'], sort_keys=False), encoding='utf-8')
            runtime = _runtime(c.mpsae_ot_mass)
            if not (folder/'runtime.json').exists():
                atomic_json(folder/'runtime.json', runtime)
            state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()}
            append_jsonl(folder/'attempts.jsonl', {**runtime, **state})
            atomic_json(folder/'run-state.json', state)
            print(f'[weight sweep] s={c.mpsae_ot_mass:g}; lambda={c.mpsae_ot_weight:g}; solver={plan["solver"]}', flush=True)
            try:
                command = [sys.executable, '-u', '-m', 'mpsae.patch_scene',
                           'smoke' if args.smoke else 'run', '--config', str(config_file),
                           '--workspace', str(workspace), '--devices', *args.devices,
                           '--cpu-threads', str(args.cpu_threads)]
                if args.nyuv2_root:
                    command += ['--nyuv2-root', args.nyuv2_root]
                subprocess.run(command, cwd=ROOT, check=True)
                if any(r['status'] != 'complete' for r in campaign_records(actual, c, campaign)):
                    raise RuntimeError('Campaign returned without all expected evaluations')
                state.update(status='complete')
            except BaseException as error:
                state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
                raise
            finally:
                state['finished_at'] = datetime.now(timezone.utc).isoformat()
                atomic_json(folder/'run-state.json', state)
                export_grid(actual, config, args.weights)


if __name__ == '__main__':
    main()
