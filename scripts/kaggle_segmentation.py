"""Canonical Kaggle launcher for the supported VOC, NYUv2 and Cityscapes patch-SAE campaigns.

Notebook: runpy.run_path(..., init_globals={'DATASET': 'nyuv2'}, run_name='__main__').
No hyperparameters are copied here: canonical YAMLs own all training settings.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from zipfile import ZipFile


def run(command, cwd=None):
    command = [str(x) for x in command]
    print('\n>>>', ' '.join(command), flush=True)
    process = subprocess.Popen(command, cwd=cwd, env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except KeyboardInterrupt:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        process.stdout.close()


def copy_current_results(archive, destination, campaign):
    """Never relabel an earlier campaign's ZIP after a failed new run."""
    if not archive.exists():
        return False
    with ZipFile(archive) as bundle:
        if 'config.json' not in bundle.namelist():
            return False
        if json.loads(bundle.read('config.json')).get('campaign') != campaign:
            return False
    shutil.copy2(archive, destination)
    return True


def main(dataset='nyuv2', encoders=None, methods=None, include_dense=True, mmpot_weight=None):
    if encoders is not None:
        encoders = list(encoders)
        if not encoders or len(set(encoders)) != len(encoders) or any(e not in ('siglip2', 'dinov2') for e in encoders):
            raise ValueError('ENCODERS must be a nonempty subset of siglip2/dinov2')
    if dataset not in ('nyuv2', 'voc', 'cityscapes'):
        raise ValueError('Choose DATASET="nyuv2", "voc" or "cityscapes"')
    from IPython.display import FileLink, display
    import pandas as pd
    repo = Path(__file__).resolve().parents[1]
    working = Path('/kaggle/working')
    working.mkdir(parents=True, exist_ok=True)
    os.chdir(working)
    workspace = working/dataset
    workspace.mkdir(exist_ok=True)
    config = repo/f'configs/patch_scene_{dataset}.yaml'
    python = '/tmp/mpsae-env/bin/python'
    os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HOME='/tmp/mpsae-hf',
        HF_DATASETS_CACHE='/tmp/mpsae-datasets', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    print('Free disk GiB:', round(shutil.disk_usage(working).free/2**30, 1))
    run(['nvidia-smi'])
    run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    if not Path(python).exists():
        run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '--seed', '/tmp/mpsae-env'])
    run([python, '-m', 'pip', 'install', '--no-cache-dir', 'torch==2.9.1', 'torchvision==0.24.1',
         '--index-url', 'https://download.pytorch.org/whl/cu126'])
    run([python, '-m', 'pip', 'install', '--no-cache-dir', '-r', repo/'requirements-patch-scene.txt'])
    run([python, '-m', 'pip', 'check'])
    count = int(subprocess.check_output([python, '-c', 'import torch; print(torch.cuda.device_count())'], text=True))
    if count < 1:
        raise RuntimeError('Enable a GPU accelerator in Kaggle settings')
    devices = [f'cuda:{i}' for i in range(min(2, count))]
    base = [python, '-u', '-m', 'mpsae.patch_scene']
    for device in devices:
        run(base+['doctor', '--device', device], cwd=repo)
    if encoders is not None or methods is not None or not include_dense or mmpot_weight is not None:
        selected_config = workspace/'config-kaggle.yaml'
        overrides = {'include_dense': include_dense}
        if encoders is not None: overrides['encoders'] = encoders
        if methods is not None: overrides['methods'] = list(methods)
        if mmpot_weight is not None: overrides['mpsae_ot_weight'] = mmpot_weight
        run([python, '-c',
             'import sys,yaml,json; from pathlib import Path; c=yaml.safe_load(Path(sys.argv[1]).read_text()); c.update(json.loads(sys.argv[3])); Path(sys.argv[2]).write_text(yaml.safe_dump(c, sort_keys=False))',
             config, selected_config, json.dumps(overrides)], cwd=repo)
        config = selected_config
    plan = json.loads(subprocess.check_output(base+['plan', '--config', str(config)], cwd=repo, text=True))
    assert set(plan['datasets']) == {dataset}
    assert set(plan['encoders']) == set(encoders or ['siglip2', 'dinov2'])
    print(json.dumps(plan, indent=2))
    (workspace/'run-plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    extra = []
    if dataset == 'nyuv2':
        data_root = workspace/'dataset'
        run([python, '-u', '-m', 'mpsae.patch_scene.prepare_nyuv2', '--from-hf',
             '--destination', data_root], cwd=repo)
        extra = ['--nyuv2-root', str(data_root)]
    if dataset == 'cityscapes':
        # Download once before the two GPU workers open the same Parquet cache.
        run([python, '-u', '-m', 'mpsae.patch_scene.cityscapes_hf'], cwd=repo)
    print('Sampling: all_train_patches=', plan['settings'].get('all_train_patches', False),
          'support_patches=', plan['settings']['support_patches'], flush=True)
    # VOC is fetched automatically from its pinned HF dataset by the feature loader.
    suffix = '_' + '_'.join(encoders) if encoders else ''
    if methods is not None: suffix += '_' + '_'.join(methods)
    if mmpot_weight is not None: suffix += '_ot' + str(mmpot_weight).replace('.', 'p')
    result_zip = working/f'{dataset.upper()}{suffix}_results.zip'
    try:
        run(base+['run', '--config', str(config), '--workspace', str(workspace),
            '--dataset', dataset, '--devices', *devices, '--cpu-threads', '2', *extra], cwd=repo)
    finally:
        archive = workspace/'patch-scene-results.zip'
        if copy_current_results(archive, result_zip, plan['campaign']):
            display(FileLink(result_zip.name))
    run(base+['export', '--config', str(config), '--workspace', str(workspace)], cwd=repo)
    if not copy_current_results(workspace/'patch-scene-results.zip', result_zip, plan['campaign']):
        raise RuntimeError('Results ZIP does not belong to the selected campaign')
    with ZipFile(result_zip) as archive:
        receipt = json.loads(archive.read('results.json'))
        comparison = pd.read_csv(archive.open('comparisons.csv'))
    assert receipt['expected'] == plan['total_evaluations']
    print(f"Completed: {receipt['completed']}/{receipt['expected']}; complete={receipt['complete']}")
    # Main comparison is matched K for both datasets. VOC cross-K rows stay in CSV/ZIP.
    matched = comparison[comparison['train_k'] == comparison['inference_k']]
    columns = ['encoder', 'train_k']
    if include_dense: columns.append('dense_miou_percent')
    columns.extend(m + '_miou_percent' for m in (methods if methods is not None else ['csr', 'csrv2', 'mpsae']))
    display(matched[columns].round(2))
    display(FileLink(result_zip.name))
    print('Results ZIP excludes checkpoints/caches. Save notebook outputs to retain workspace:', workspace)


if __name__ == '__main__':
    main(globals().get('DATASET', 'nyuv2'), globals().get('ENCODERS'),
         globals().get('METHODS'), globals().get('INCLUDE_DENSE', True),
         globals().get('MMPOT_WEIGHT'))
