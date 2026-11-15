"""Single-GPU, persistent, local-only NYUv2/Cityscapes campaigns.

Install once with bash scripts/setup_segmentation_server.sh. Repeating the run
command reuses compatible checkpoints/features. --plan performs no downloads.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(command, env, log):
    print('>>>', ' '.join(map(str, command)), flush=True)
    with log.open('a', encoding='utf-8') as handle:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', errors='replace', bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                if os.name == 'posix': os.killpg(process.pid, signal.SIGINT)
                else: process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix': os.killpg(process.pid, signal.SIGKILL)
                    else: process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def export_local(workspace, destination, campaign):
    """Copy only this campaign's tables, including valid partial results on failure."""
    from mpsae.io import read_json
    from mpsae.patch_scene.report import export_results
    table = workspace/'campaigns'/campaign/'table'
    receipt = read_json(table/'results.json')
    if not receipt or not receipt.get('results'):
        return None
    archive = export_results(workspace, campaign)
    destination.mkdir(parents=True, exist_ok=True)
    # Export directories include campaign ID, so different settings cannot mix.
    for path in table.iterdir():
        if path.suffix in ('.csv', '.tex', '.json') and path.name != 'bundle.json':
            shutil.copy2(path, destination/path.name)
    shutil.copy2(archive, destination/'results.zip')
    return receipt


def write_config(config, path):
    import yaml
    # Use YAML's emitter: JSON exponent literals such as 4e-05 are strings in
    # PyYAML's YAML-1.1 resolver, despite JSON otherwise being accepted by it.
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default='work/segmentation-server', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=('nyuv2','cityscapes'), default=['nyuv2','cityscapes'])
    parser.add_argument('--encoders', nargs='+', choices=('siglip2','dinov2'), default=['siglip2','dinov2'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--plan', action='store_true', help='Print resolved settings without downloading/running')
    parser.add_argument('--smoke', action='store_true', help='Synthetic CPU pipeline check; not benchmark results')
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or len(set(args.datasets)) != len(args.datasets) or len(set(args.encoders)) != len(args.encoders):
        parser.error('positive cpu-threads and distinct datasets/encoders required')
    from mpsae.patch_scene.config import load
    configs = {d: replace(load(ROOT/f'configs/patch_scene_{d}.yaml'), encoders=tuple(args.encoders))
               for d in args.datasets}
    plans = {d: c.plan() for d,c in configs.items()}
    print(json.dumps(plans, indent=2), flush=True)
    if args.plan: return
    workspace = args.workspace.expanduser().resolve()
    if args.smoke: workspace = workspace/'SMOKE_ONLY'
    workspace.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HOME=str(workspace/'cache/huggingface'),
        HF_DATASETS_CACHE=str(workspace/'cache/datasets'),
        OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH','')
    (workspace/'logs').mkdir(exist_ok=True)
    print(f'Persistent workspace: {workspace}; free disk: {shutil.disk_usage(workspace).free/2**30:.1f} GiB', flush=True)
    base = [sys.executable, '-u', '-m', 'mpsae.patch_scene']
    # Lock the orchestration as well as the underlying campaigns, before data preparation.
    from mpsae.io import exclusive_lock
    with exclusive_lock(workspace/'.server.lock', timeout=0):
        if not args.smoke:
            run(base+['doctor','--device',args.device], env, workspace/'logs/doctor.log')
        selected_outputs = []
        for dataset, config in configs.items():
            root = workspace/dataset
            root.mkdir(exist_ok=True)
            campaign = config.plan()['campaign']
            config_file = root/f'config-{campaign}.yaml'
            write_config(config, config_file)
            log = workspace/f'logs/{dataset}.log'
            extra = []
            if not args.smoke:
                if dataset == 'nyuv2':
                    data = workspace/'datasets/nyuv2'
                    run([sys.executable,'-u','-m','mpsae.patch_scene.prepare_nyuv2',
                         '--from-hf','--destination',data], env, log)
                    extra = ['--nyuv2-root', str(data)]
                else:
                    run([sys.executable,'-u','-m','mpsae.patch_scene.cityscapes_hf'], env, log)
            output = workspace/'results'/dataset/campaign
            actual_root = root/'SMOKE_ONLY' if args.smoke else root
            receipt = None
            try:
                run(base+['smoke' if args.smoke else 'run','--config',config_file,
                          '--workspace',root,'--device','cpu' if args.smoke else args.device,
                          '--cpu-threads',str(args.cpu_threads),*extra], env, log)
            finally:
                receipt = export_local(actual_root, output, campaign)
                if receipt:
                    print(f'Local tables: {output} ({receipt["completed"]}/{receipt["expected"]})', flush=True)
            if not receipt or not receipt['complete'] or receipt['expected'] != plans[dataset]['total_evaluations']:
                raise RuntimeError(f'{dataset} did not complete its expected evaluations')
            selected_outputs.append(output)
        bundle = workspace/'segmentation-results.zip'
        temporary = bundle.with_suffix('.zip.partial')
        with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
            for output in selected_outputs:
                for path in sorted(output.iterdir()):
                    if path.suffix != '.zip':
                        archive.write(path, path.relative_to(workspace/'results'))
                archive.write(output/'results.zip', output.relative_to(workspace/'results')/'results.zip')
        os.replace(temporary, bundle)
        print(f'COMPLETE: {bundle}\nCSV/LaTeX tables: {workspace/"results"}', flush=True)


if __name__ == '__main__':
    main()
