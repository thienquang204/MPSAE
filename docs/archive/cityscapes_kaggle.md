# Cityscapes fine train/validation — full-patch SAE benchmark

Enable Internet and select T4 x2 in Kaggle. Push the updated code to the GitHub
`refactor` branch first. Only Cityscapes is selected: 2,975 train and 500 validation
images; no ADE20K, VOC, test evaluation, coarse labels or extra video frames.

Register and activate an account at https://www.cityscapes-dataset.com/register/
and accept the dataset terms. Cell 4 can download the official ZIPs directly;
credentials are entered privately and are not saved. Alternatively attach your
official ZIPs/extracted dataset as a private Kaggle input and set DATA_ROOT.

Configuration: SigLIP 2 and DINOv2 frozen; CSR/CSRv2/MPSAE; independent train
K=3,12,48 and inference K=3,12,48. All valid patches at 448x896, disk streaming,
10 epochs, batch 1024, Adam LR 4e-5. 18 fits and 56 evaluation conditions.
Full patch coverage is at the declared resized resolution, not native resolution.

Feature caches (~18 GiB before masks/labels) and downloads go to /tmp; checkpoints
and result tables go to /kaggle/working. /tmp is ephemeral and not saved as notebook
output. Preserve caches separately if you want reuse across sessions. A results ZIP
does not contain checkpoints or caches. This full campaign may exceed one Kaggle
session; no runtime guarantee is implied. Use the same workspace to resume within
a session. There is no automatic transfer to Hugging Face.


## Cell 1 — Paths and caches

```python
import os, sys, json, subprocess, shutil
from pathlib import Path

REPO = Path('/kaggle/working/MPSAE')
WORK = Path('/kaggle/working/cityscapes')
DATA_ROOT = Path('/tmp/cityscapes-data')  # or your private /kaggle/input/... folder
PYTHON = '/tmp/mpsae-env/bin/python'
CONFIG = REPO / 'configs/patch_scene_cityscapes.yaml'
DEVICES = ['cuda:0', 'cuda:1']  # use ['cuda:0'] on one GPU
WORK.mkdir(parents=True, exist_ok=True)
cache = Path('/tmp/mpsae-city-features')
cache.mkdir(parents=True, exist_ok=True)
if not (WORK/'features').exists():
    (WORK/'features').symlink_to(cache, target_is_directory=True)
os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HOME='/tmp/mpsae-hf',
                  HF_DATASETS_CACHE='/tmp/mpsae-datasets', TOKENIZERS_PARALLELISM='false',
                  PYTHONUNBUFFERED='1')
print('Free disk GiB:', {p: round(shutil.disk_usage(p).free/2**30, 1)
                         for p in ['/tmp', '/kaggle/working']})
```

## Cell 2 — Get the code

```python
if not REPO.exists():
    subprocess.run(['git','clone','--branch','refactor','--single-branch',
                    'https://github.com/thienquang204/MPSAE.git',str(REPO)], check=True)
else:
    branch = subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'], text=True).strip()
    assert branch == 'refactor', f'Unexpected branch: {branch}'
    subprocess.run(['git','-C',str(REPO),'pull','--ff-only'], check=True)
assert CONFIG.exists(), 'Push the Cityscapes changes to GitHub, then rerun this cell.'
subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'], check=True)
subprocess.run(['nvidia-smi'], check=True)
```

## Cell 3 — Install the isolated environment

```python
subprocess.run([sys.executable,'-m','pip','install','-q','uv'], check=True)
if not Path(PYTHON).exists():
    subprocess.run([sys.executable,'-m','uv','venv','--python','3.11','--seed','/tmp/mpsae-env'], check=True)
subprocess.run([PYTHON,'-m','pip','install','--no-cache-dir','torch==2.9.1','torchvision==0.24.1',
                '--index-url','https://download.pytorch.org/whl/cu126'], check=True)
subprocess.run([PYTHON,'-m','pip','install','--no-cache-dir','-r',str(REPO/'requirements-patch-scene.txt')], check=True)
subprocess.run([PYTHON,'-m','pip','check'], check=True)
BASE = [PYTHON,'-u','-m','mpsae.patch_scene']
for device in DEVICES:
    subprocess.run(BASE+['doctor','--device',device], cwd=REPO, check=True)
```

## Cell 4 — Download or validate Cityscapes

```python
# If you attached both official ZIPs or extracted folders, this skips downloading.
attached = DATA_ROOT.exists() and (
    (any(DATA_ROOT.rglob('leftImg8bit_trainvaltest.zip')) and any(DATA_ROOT.rglob('gtFine_trainvaltest.zip'))) or
    any(DATA_ROOT.rglob('*_leftImg8bit.png')))
if not attached:
    from getpass import getpass
    credentials = {'username': getpass('Cityscapes username/email: '),
                   'password': getpass('Cityscapes password: ')}
    try:
        subprocess.run([PYTHON,'-u','-m','mpsae.patch_scene.download_cityscapes',str(DATA_ROOT)],
                       input=json.dumps(credentials), text=True, cwd=REPO, check=True)
    finally:
        credentials.clear()
subprocess.run([PYTHON,'-c',
    'import sys; from mpsae.patch_scene.cityscapes import load_local; '
    'd,_=load_local(sys.argv[1]); print({k:len(v) for k,v in d.items()})',
    str(DATA_ROOT)], cwd=REPO, check=True)
```

## Cell 5 — Verify the experiment

```python
plan = json.loads(subprocess.check_output(BASE+['plan','--config',str(CONFIG)], cwd=REPO, text=True))
assert set(plan['datasets']) == {'cityscapes'}
assert plan['settings']['all_train_patches'] is True
assert plan['settings']['epochs'] == 10 and not plan['settings']['early_stopping']
assert plan['input_hw'] == [448,896]
assert plan['train_ks'] == plan['inference_k'] == [3,12,48]
assert plan['sae_fits'] == 18 and plan['total_evaluations'] == 56
print(json.dumps(plan, indent=2))
(WORK/'run-plan.json').write_text(json.dumps(plan, indent=2))
```

## Cell 6 — Train and evaluate

```python
import signal
command = BASE+['run','--config',str(CONFIG),'--workspace',str(WORK),
                '--dataset','cityscapes','--cityscapes-root',str(DATA_ROOT),
                '--devices',*DEVICES,'--cpu-threads','2']
process = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(), start_new_session=True)
try:
    result = process.wait()
    if result:
        raise RuntimeError(f'Experiment exited with code {result}; inspect the worker logs.')
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
    archive = WORK/'patch-scene-results.zip'
    if archive.exists():
        shutil.copy2(archive, '/kaggle/working/Cityscapes_results.zip')
        print('Available results: /kaggle/working/Cityscapes_results.zip')
```

## Cell 7 — Export and download

```python
# Run only after the training process has finished or stopped.
from IPython.display import FileLink, display
subprocess.run(BASE+['export','--config',str(CONFIG),'--workspace',str(WORK)], cwd=REPO, check=True)
destination = Path('/kaggle/working/Cityscapes_results.zip')
shutil.copy2(WORK/'patch-scene-results.zip', destination)
from zipfile import ZipFile
with ZipFile(destination) as archive:
    receipt = json.loads(archive.read('results.json'))
print(f"Completed: {receipt['completed']}/{receipt['expected']}; complete={receipt['complete']}")
display(FileLink(str(destination)))
```
