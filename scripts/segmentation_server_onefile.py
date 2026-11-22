#!/usr/bin/env python3
"""Standalone Linux/NVIDIA setup and local segmentation mass/weight benchmark.

Copy this file to a new server and run python3 segmentation_server_onefile.py.
Requires Python 3, git, curl, bash and a working NVIDIA driver. The repository,
Python environment, datasets, caches and results are prepared automatically.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

URL = 'https://github.com/thienquang204/MPSAE.git'


def command(argv, repo=None):
    print('\n>>>', ' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=repo, check=True)


def bootstrap(args):
    repo = args.repo or args.workspace/'code/MPSAE'
    repo = repo.expanduser().resolve()
    if args.skip_setup:
        if not (repo/'mpsae/patch_scene/weight_sweep.py').is_file():
            raise RuntimeError('--skip-setup requires --repo pointing to the updated repository')
        return repo, Path(sys.executable)
    if sys.platform != 'linux':
        raise RuntimeError('Automatic installation requires a Linux NVIDIA server')
    for executable in ('git', 'curl', 'bash', 'nvidia-smi'):
        if not shutil.which(executable):
            raise RuntimeError(f'Missing {executable}; use a Linux image with NVIDIA drivers, git and curl')
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        command(['git', 'clone', '--branch', 'refactor', '--single-branch', URL, repo])
    else:
        branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
        if branch != 'refactor':
            raise RuntimeError(f'Expected refactor branch, found {branch!r}')
        if args.update:
            command(['git', '-C', repo, 'pull', '--ff-only'])
    if not (repo/'mpsae/patch_scene/weight_sweep.py').is_file():
        raise RuntimeError('Push the weight-sweep update to refactor first; rerun with --update')
    python = repo/'.venv-segmentation/bin/python'
    signature = hashlib.sha256(b''.join((repo/p).read_bytes() for p in
        ('requirements-patch-scene.txt', 'scripts/setup_segmentation_server.sh'))).hexdigest()
    stamp = repo/'.venv-segmentation/.server-install-signature'
    if not python.exists() or not stamp.exists() or stamp.read_text() != signature:
        command(['bash', repo/'scripts/setup_segmentation_server.sh'], repo)
        stamp.write_text(signature)
    return repo, python


def selections(args, repo):
    from dataclasses import replace
    from mpsae.patch_scene.config import load
    from mpsae.patch_scene.weight_sweep import weight_plan
    configs = [replace(load(repo/f'configs/patch_scene_{args.dataset}.yaml'),
                       encoders=tuple(args.encoders), methods=tuple(args.methods),
                       train_k=tuple(args.train_k), inference_k=None, evaluation_mode='matched',
                       include_dense=not args.no_dense, mpsae_ot_mass=s) for s in sorted(args.masses)]
    plans = [weight_plan(c, args.weights) for c in configs]
    n = len(args.encoders)*len(args.train_k)
    fits = n*(len(args.methods)-1 + len(args.masses)*len(args.weights))
    identity = {'kind': 'server-mass-weight-v1', 'sweeps': [p['sweep_id'] for p in plans]}
    from mpsae.io import canonical_hash
    return configs, {**identity, 'experiment_id': canonical_hash(identity)[:20],
                     'dataset': args.dataset, 'sae_fits': fits,
                     'total_evaluations': fits+(0 if args.no_dense else len(args.encoders)),
                     'count_scope': 'Unique results; CSR/CSRv2/dense reused across all masses and weights',
                     'plans': plans, 'smoke_only': args.smoke}


def export_results(data_root, destination, plan):
    """Only collect explicitly selected sweep IDs, never stale ZIPs/globbed history."""
    from mpsae.io import atomic_json, read_json
    destination.mkdir(parents=True, exist_ok=True)
    lookup, comparisons, archives, statuses = {}, [], [], []
    for p in plan['plans']:
        folder = data_root/'ablations'/p['sweep_id']
        result = read_json(folder/'grid_results.json')
        statuses.append({'sweep_id': p['sweep_id'], 'mass': p['mpsae_ot_mass'],
                         'complete': bool(result and result['complete']),
                         'completed': result['completed'] if result else 0,
                         'expected': p['total_evaluations']})
        if result:
            for row in result['results']:
                key = tuple(row.get(k) for k in ('dataset', 'encoder', 'method', 'train_k',
                                                'inference_k', 'seed', 'mmpot_mass', 'mmpot_weight'))
                old = lookup.get(key)
                if old is not None and any(old.get(k) != row.get(k) for k in
                        ('identity', 'miou', 'pixel_accuracy', 'mean_nnz', 'gallery_mean_nnz')):
                    raise ValueError('Shared baseline results disagree across mass sweeps')
                lookup[key] = row
            with (folder/'grid_comparisons.csv').open(newline='', encoding='utf-8') as file:
                comparisons.extend(csv.DictReader(file))
        if (folder/'weight-sweep-results.zip').exists():
            archives.append((p['sweep_id'], folder/'weight-sweep-results.zip'))
    rows = list(lookup.values())
    summary = {'complete': len(rows) == plan['total_evaluations'] and all(s['complete'] for s in statuses),
               'completed': len(rows), 'expected': plan['total_evaluations'], 'sweeps': statuses,
               'plan': plan, 'results': rows}
    atomic_json(destination/'results.json', summary)
    atomic_json(destination/'plan.json', plan)
    fields = ['dataset', 'encoder', 'method', 'train_k', 'inference_k', 'seed', 'mmpot_mass',
              'mmpot_weight', 'solver', 'miou_percent', 'pixel_accuracy_percent', 'mean_nnz', 'gallery_mean_nnz']
    with (destination/'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k) for k in fields}, 'miou_percent': 100*row['miou'],
                             'pixel_accuracy_percent': 100*row['pixel_accuracy']})
    if comparisons:
        with (destination/'mass_weight_comparison.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    archive = destination/'segmentation-results.zip'
    temporary = archive.with_suffix('.zip.partial')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as bundle:
        for name in ('results.json', 'results.csv', 'plan.json', 'mass_weight_comparison.csv'):
            if (destination/name).exists():
                bundle.write(destination/name, name)
        for sweep_id, path in archives:
            bundle.write(path, f'sweeps/{sweep_id}/weight-sweep-results.zip')
        bundle.writestr('README.txt', 'All selected masses and weights; CSR/CSRv2/dense deduplicated.\n'
                         'Missing conditions are reported in results.json, not assigned zero scores.\n'
                         'Per-sweep ZIPs contain LaTeX tables, settings and provenance.\n'
                         'No models or feature caches. Retain the full workspace to resume.\n'
                         f'Synthetic smoke run: {plan["smoke_only"]}\n')
    os.replace(temporary, archive)
    print(f'\nResults: {summary["completed"]}/{summary["expected"]}; complete={summary["complete"]}\n{archive}', flush=True)
    return summary, archive


def worker(args, repo):
    sys.path.insert(0, str(repo))
    configs, plan = selections(args, repo)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan:
        return
    from mpsae.io import exclusive_lock
    data_workspace = args.workspace/args.dataset
    actual_data = data_workspace/'SMOKE_ONLY' if args.smoke else data_workspace
    output_root = args.workspace/'SMOKE_ONLY' if args.smoke else args.workspace
    destination = output_root/'results'/plan['experiment_id']
    os.environ.update(HF_HOME=str(args.workspace/'cache/huggingface'),
                      HF_DATASETS_CACHE=str(args.workspace/'cache/datasets'),
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false',
                      PYTHONUNBUFFERED='1')
    with exclusive_lock(args.workspace/'.server.lock', timeout=0):
        devices = args.devices
        if args.smoke:
            devices = devices or ['cpu']
        else:
            import torch
            devices = devices or [f'cuda:{i}' for i in range(min(2, torch.cuda.device_count()))]
            if not devices or any(not d.startswith('cuda:') for d in devices):
                raise RuntimeError('A CUDA GPU is required; check nvidia-smi and the installed driver')
            for device in devices:
                command([sys.executable, '-m', 'mpsae.patch_scene', 'doctor', '--device', device], repo)
        base = [sys.executable, '-u', '-m', 'mpsae.patch_scene.weight_sweep',
                '--config', repo/f'configs/patch_scene_{args.dataset}.yaml',
                '--workspace', data_workspace, '--encoders', *args.encoders,
                '--methods', *args.methods, '--train-k', *map(str, args.train_k),
                '--weights', *map(str, args.weights), '--devices', *devices,
                '--cpu-threads', str(args.cpu_threads)]
        if args.no_dense:
            base.append('--no-dense')
        if args.smoke:
            base.append('--smoke')
        try:
            if not args.smoke and args.dataset == 'nyuv2':
                root = args.workspace/'datasets/nyuv2'
                command([sys.executable, '-u', '-m', 'mpsae.patch_scene.prepare_nyuv2',
                         '--from-hf', '--destination', root], repo)
                base += ['--nyuv2-root', root]
            elif not args.smoke and args.dataset == 'cityscapes':
                command([sys.executable, '-u', '-m', 'mpsae.patch_scene.cityscapes_hf'], repo)
            for c in configs:
                try:
                    command(base+['--mass', str(c.mpsae_ot_mass)], repo)
                finally:
                    export_results(actual_data, destination, plan)
        finally:
            summary, archive = export_results(actual_data, destination, plan)
            shutil.copy2(archive, output_root/'segmentation-results.zip')
            print(f'Convenient download path: {output_root/"segmentation-results.zip"}', flush=True)
        if not summary['complete']:
            raise RuntimeError('The selected experiment is incomplete; see results.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('segmentation-run'))
    parser.add_argument('--repo', type=Path, help='Existing repository; otherwise cloned into workspace/code/MPSAE')
    parser.add_argument('--dataset', choices=('voc', 'nyuv2', 'cityscapes'), default='voc')
    parser.add_argument('--encoders', nargs='+', choices=('siglip2', 'dinov2'), default=['siglip2'])
    parser.add_argument('--methods', nargs='+', choices=('csr', 'csrv2', 'mpsae'), default=['csr', 'csrv2', 'mpsae'])
    parser.add_argument('--train-k', nargs='+', type=int, default=[3, 12, 48])
    parser.add_argument('--masses', nargs='+', type=float, default=[.9, .95, 1.])
    parser.add_argument('--weights', nargs='+', type=float, default=[.5, 1., 2.])
    parser.add_argument('--no-dense', action='store_true')
    parser.add_argument('--devices', nargs='+', help='Default: first one or two visible CUDA GPUs')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--update', action='store_true', help='Fast-forward an existing refactor checkout')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Isolated synthetic test, no dataset downloads')
    parser.add_argument('--skip-setup', action='store_true', help='Use current Python and --repo; environment already installed')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    import math
    if (any(not math.isfinite(s) or not 0 < s <= 1 for s in args.masses)
            or len(set(args.masses)) != len(args.masses) or args.cpu_threads < 1):
        parser.error('Use distinct masses 0<s<=1 and positive cpu-threads')
    if 'mpsae' not in args.methods:
        parser.error('Mass/weight experiments require mpsae in --methods')
    args.workspace = args.workspace.expanduser().resolve()
    if args.worker:
        worker(args, args.repo.resolve())
    else:
        repo, python = bootstrap(args)
        if args.skip_setup:
            worker(args, repo)
        else:
            forwarded = list(argv) if argv is not None else sys.argv[1:]
            command([python, '-u', Path(__file__).resolve(), *forwarded,
                     '--worker', '--repo', repo, '--workspace', args.workspace])


if __name__ == '__main__':
    main()
