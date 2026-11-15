# NYUv2-40 on Kaggle: train K, evaluate the same K

Enable Internet and a GPU accelerator (T4 x2 when available). Push the updated
code to the `refactor` branch before cloning it in Kaggle.

Download **Labeled dataset (~2.8 GB)** from
https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html and upload
`nyu_depth_v2_labeled.mat` as a Kaggle Dataset. Attach it with Add Input.
Do not select the 428 GB raw videos or a depth-only dataset. No NYU login or
Hugging Face token is needed by this notebook. The code fetches only two small
metadata files automatically; they can also be attached for offline setup:

- [splits.mat](https://raw.githubusercontent.com/ankurhanda/nyuv2-meta-data/b92d15c506876dada1ac4086938e30ebc4deb49b/splits.mat)
- [classMapping40.mat](https://raw.githubusercontent.com/ankurhanda/nyuv2-meta-data/b92d15c506876dada1ac4086938e30ebc4deb49b/classMapping40.mat)

Official split: **795 train / 654 test**. RGB only, 40 classes; label zero is void.
The pipeline's historical internal key `validation` means the official **test**
split for NYUv2; reports explicitly identify it. No test-based model selection.
All valid training patches at **336 x 448** are used, not a sampled training cap.
This is whole-image resizing preserving 4:3, not native-resolution patch training.

Frozen SigLIP 2 Base/16 and DINOv2 Base/14, independent CSR/CSRv2/MPSAE fitting at
K=3,12,48; **each checkpoint is evaluated only at its own training K**.
18 fits + 18 sparse evaluations + 2 dense evaluations (20 evaluations total).
10 epochs, batch 1024, Adam LR 4e-5; original method losses are unchanged.
CSRv2 uses training-mask classes in its loss; CSR/MPSAE do not. All methods use
the same train-only soft-label retrieval support selection (65536 patches per
encoder), 30 neighbors, temperature 0.02 and native-resolution test masks.

Each frozen encoder is extracted once. FP16 feature shards (~2.81 GiB combined,
before masks/labels) stay in WORK and are streamed for training; no depth arrays
are loaded. Two GPUs run independent jobs, one per GPU; their VRAM is not pooled.
Verified immutable feature files are not rehashed every epoch unless file metadata
changes. Checkpoints are saved every 512 steps and at training completion; logs
every 100 steps. Existing completed work in the same workspace is reused.

Reserve roughly 15–20 GB disk, 16–32 GB RAM. 12 GB VRAM is the target, not a measured
guarantee; the complete dataset/GPU campaign has not been benchmarked locally.
If extraction alone runs out of memory, lower `extraction_batch` before the first
run. Features, fits and results remain in `/kaggle/working/nyuv2`; save notebook
outputs to retain them across sessions. `/tmp` holds only the Python environment
and downloaded encoder weights and is ephemeral. The downloadable results ZIP
contains tables/provenance, **not** checkpoints or feature caches.

Copy each of the following blocks into a separate code cell.

## Cell 1: Paths

```python
import os, sys, json, subprocess, shutil
from pathlib import Path

os.chdir('/kaggle/working')
REPO = Path('/kaggle/working/MPSAE')
WORK = Path('/kaggle/working/nyuv2')
INPUT = Path('/kaggle/input')  # narrow to your dataset folder if multiple copies are attached
DATA_ROOT = WORK / 'dataset'
PYTHON = '/tmp/mpsae-env/bin/python'
CONFIG = REPO / 'configs/patch_scene_nyuv2.yaml'
WORK.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HOME='/tmp/mpsae-hf',
                  HF_DATASETS_CACHE='/tmp/mpsae-datasets', TOKENIZERS_PARALLELISM='false',
                  PYTHONUNBUFFERED='1')
print('Free working disk GiB:', round(shutil.disk_usage(WORK).free/2**30, 1))
```

## Cell 2: Get the updated code

```python
if not REPO.exists():
    subprocess.run(['git','clone','--branch','refactor','--single-branch',
                    'https://github.com/thienquang204/MPSAE.git',str(REPO)], check=True)
else:
    branch = subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'], text=True).strip()
    assert branch == 'refactor', f'Unexpected branch: {branch}'
    subprocess.run(['git','-C',str(REPO),'pull','--ff-only'], check=True)
assert CONFIG.exists(), 'Push the NYUv2 changes to GitHub first, then rerun this cell.'
subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'], check=True)
subprocess.run(['nvidia-smi'], check=True)
```

## Cell 3: Install and check GPUs

```python
subprocess.run([sys.executable,'-m','pip','install','-q','uv'], check=True)
if not Path(PYTHON).exists():
    subprocess.run([sys.executable,'-m','uv','venv','--python','3.11','--seed','/tmp/mpsae-env'], check=True)
subprocess.run([PYTHON,'-m','pip','install','--no-cache-dir','torch==2.9.1','torchvision==0.24.1',
                '--index-url','https://download.pytorch.org/whl/cu126'], check=True)
subprocess.run([PYTHON,'-m','pip','install','--no-cache-dir','-r',str(REPO/'requirements-patch-scene.txt')], check=True)
subprocess.run([PYTHON,'-m','pip','check'], check=True)
count = int(subprocess.check_output([PYTHON,'-c','import torch; print(torch.cuda.device_count())'], text=True))
assert count >= 1, 'Enable a GPU in Kaggle notebook settings.'
DEVICES = [f'cuda:{i}' for i in range(min(2,count))]
BASE = [PYTHON,'-u','-m','mpsae.patch_scene']
for device in DEVICES:
    subprocess.run(BASE+['doctor','--device',device], cwd=REPO, check=True)
print('Workers:', DEVICES)
```

## Cell 4: Validate the attached dataset

```python
# Links the large MAT from Input without copying it; downloads only small metadata.
subprocess.run([PYTHON,'-u','-m','mpsae.patch_scene.prepare_nyuv2',
                '--source',str(INPUT),'--destination',str(DATA_ROOT)], cwd=REPO, check=True)
```

## Cell 5: Verify exactly 20 evaluations

```python
plan = json.loads(subprocess.check_output(BASE+['plan','--config',str(CONFIG)], cwd=REPO, text=True))
assert set(plan['datasets']) == {'nyuv2'}
assert set(plan['encoders']) == {'siglip2','dinov2'}
assert plan['settings']['all_train_patches'] is True
assert plan['settings']['epochs'] == 10 and not plan['settings'].get('early_stopping', False)
assert plan['input_hw'] == [336,448]
assert plan['evaluation_mode'] == 'matched'
assert plan['evaluation_pairs'] == [[3,3],[12,12],[48,48]]
assert plan['sae_fits'] == 18 and plan['total_evaluations'] == 20
assert plan['cross_k_evaluations'] == 0
print(json.dumps(plan, indent=2))
(WORK/'run-plan.json').write_text(json.dumps(plan, indent=2))
```

## Cell 6: Train and evaluate

```python
import signal
command = BASE+['run','--config',str(CONFIG),'--workspace',str(WORK),
                '--dataset','nyuv2','--nyuv2-root',str(DATA_ROOT),
                '--devices',*DEVICES,'--cpu-threads','2']
process = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(), start_new_session=True)
try:
    result = process.wait()
    if result:
        raise RuntimeError(f'Experiment exited with code {result}; inspect {WORK}/campaigns/*/logs/.')
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
        shutil.copy2(archive, '/kaggle/working/NYUv2_results.zip')
        print('Available results: /kaggle/working/NYUv2_results.zip')
```

## Cell 7: Show the comparison table and download

```python
# Run after Cell 6 has finished or stopped.
from IPython.display import FileLink, display
from zipfile import ZipFile
import pandas as pd

subprocess.run(BASE+['export','--config',str(CONFIG),'--workspace',str(WORK)], cwd=REPO, check=True)
destination = Path('/kaggle/working/NYUv2_results.zip')
shutil.copy2(WORK/'patch-scene-results.zip', destination)
with ZipFile(destination) as archive:
    receipt = json.loads(archive.read('results.json'))
    table = pd.read_csv(archive.open('comparisons.csv'))
assert receipt['expected'] == 20
print(f"Completed: {receipt['completed']}/{receipt['expected']}; complete={receipt['complete']}")
display(table[['encoder','train_k','dense_miou_percent','csr_miou_percent',
               'csrv2_miou_percent','mpsae_miou_percent']].round(2))
display(FileLink('NYUv2_results.zip'))
```

`results.tex` is the main six-row mIoU comparison table (two encoders x three K).
`pixel_accuracy.tex` is supplementary. `results.csv` includes actual nonzero counts;
`results.json` includes provenance, class IoUs and training histories. No inference
sweep tables or plots are generated. A partial run clearly marks missing entries.
To resume in the same session, rerun Cell 6. To resume a later session, restore the
whole WORK folder, retain the same config, and reattach the labeled MAT at the same
path (or recreate only its symlink); a results ZIP alone cannot resume training.
