"""Canonical Kaggle launcher for the supported VOC and NYUv2 patch-SAE campaigns.

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


def main(dataset='nyuv2'):
    if dataset not in ('nyuv2', 'voc'):
        raise ValueError('Choose DATASET="nyuv2" or "voc"')
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
    plan = json.loads(subprocess.check_output(base+['plan', '--config', str(config)], cwd=repo, text=True))
    assert set(plan['datasets']) == {dataset}
    assert set(plan['encoders']) == {'siglip2', 'dinov2'}
    print(json.dumps(plan, indent=2))
    (workspace/'run-plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    extra = []
    if dataset == 'nyuv2':
        data_root = workspace/'dataset'
        run([python, '-u', '-m', 'mpsae.patch_scene.prepare_nyuv2', '--from-hf',
             '--destination', data_root], cwd=repo)
        extra = ['--nyuv2-root', str(data_root)]
    # VOC is fetched automatically from its pinned HF dataset by the feature loader.
    result_zip = working/f'{dataset.upper()}_results.zip'
    try:
        run(base+['run', '--config', str(config), '--workspace', str(workspace),
            '--dataset', dataset, '--devices', *devices, '--cpu-threads', '2', *extra], cwd=repo)
    finally:
        archive = workspace/'patch-scene-results.zip'
        if archive.exists():
            shutil.copy2(archive, result_zip)
            display(FileLink(result_zip.name))
    run(base+['export', '--config', str(config), '--workspace', str(workspace)], cwd=repo)
    shutil.copy2(workspace/'patch-scene-results.zip', result_zip)
    with ZipFile(result_zip) as archive:
        receipt = json.loads(archive.read('results.json'))
        comparison = pd.read_csv(archive.open('comparisons.csv'))
    assert receipt['expected'] == plan['total_evaluations']
    print(f"Completed: {receipt['completed']}/{receipt['expected']}; complete={receipt['complete']}")
    # Main comparison is matched K for both datasets. VOC cross-K rows stay in CSV/ZIP.
    matched = comparison[comparison['train_k'] == comparison['inference_k']]
    display(matched[['encoder', 'train_k', 'dense_miou_percent', 'csr_miou_percent',
                     'csrv2_miou_percent', 'mpsae_miou_percent']].round(2))
    display(FileLink(result_zip.name))
    print('Results ZIP excludes checkpoints/caches. Save notebook outputs to retain workspace:', workspace)


if __name__ == '__main__':
    main(globals().get('DATASET', 'nyuv2'))
