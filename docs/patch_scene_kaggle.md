# Kaggle segmentation: VOC, NYUv2 or Cityscapes

Supported experiments use frozen **SigLIP2 Base and DINOv2 Base** with fresh
CSR/CSRv2/MPSAE patch SAEs. Training algorithms and retrieval calculations are
unchanged by the organizational cleanup. Choose the dataset in the cell below.

| Setting | VOC | NYUv2-40 |
|---|---|---|
| Canonical config | `configs/patch_scene_voc.yaml` | `configs/patch_scene_nyuv2.yaml` |
| Split | 1464 train / 1449 validation | 795 train / 654 official test |
| Input | 448 x 448 | 336 x 448 |
| SAE training patches | Up to 262144; existing per-image sampling | All valid train patches, streamed |
| Train K | 3, 12, 48 | 3, 12, 48 |
| Evaluation | Each fit at K=3,12,48 (preserved VOC sweep) | Each fit only at its training K |
| Epochs | 10, no early stopping | Up to 20; training-loss early stopping |
| Fits / evaluations | 18 / 56 | 18 / 20 |

All three campaigns use batch 1024, Adam LR 4e-5, a separate 65536-patch retrieval bank,
30 neighbors and temperature 0.02. CSRv2 uses training-mask labels in its loss.
NYUv2 early stopping begins at epoch 15 after a full epoch at target K, with
patience 3 and relative improvement threshold 0.001; no test-based selection.
NYUv2 MPSAE uses OT weight 0.5 for SigLIP2 and 2.0 for DINOv2; other campaigns
use 2.0 for both. For only SigLIP2, use `scripts/kaggle_nyuv2_siglip2_cell.py`
(9 fits and 10 evaluations). The last completed epoch is evaluated. All settings live in the YAML files;
the launcher reads them rather than repeating their values in notebook checks.

Enable **Internet** and **GPU T4 x2** in Kaggle. One GPU also works. Push the
updated code to the GitHub `refactor` branch first. Paste this single code cell:

```python
# Enable Internet and GPU T4 x2 (one GPU also works). No Add Input or HF token needed.
DATASET = "nyuv2"  # Options: "nyuv2", "voc", "cityscapes".

from pathlib import Path
import subprocess
import runpy

REPO = Path("/kaggle/working/MPSAE")
if not REPO.exists():
    subprocess.run(["git", "clone", "--branch", "refactor", "--single-branch",
                    "https://github.com/thienquang204/MPSAE.git", str(REPO)], check=True)
else:
    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    assert branch == "refactor", f"Unexpected branch: {branch}"
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)

launcher = REPO / "scripts/kaggle_segmentation.py"
assert launcher.is_file(), "Push the organizational cleanup to GitHub, then rerun this cell."
runpy.run_path(str(launcher), init_globals={"DATASET": DATASET}, run_name="__main__")
```

The canonical implementation is `scripts/kaggle_segmentation.py`; the notebook
and this snippet only select the dataset and invoke it. No manual dataset upload
or Hugging Face token is required. VOC comes from its pinned HF dataset. NYUv2
downloads the labeled MAT from a pinned public HF mirror, verifies its SHA-256,
then applies the pinned official split and NYU40 mapping. Only RGB is used.

Outputs: `/kaggle/working/<dataset>/`, plus `NYUV2_results.zip` or `VOC_results.zip`.
The ZIP contains CSV/JSON/LaTeX, not weights or feature caches. The notebook displays
the matched-K comparison table; the VOC ZIP also retains its existing inference
sweep. Rerunning in the same workspace resumes compatible work. Save Kaggle outputs
to preserve the workspace across sessions; restore the whole workspace to resume
training, not just the small results ZIP. `/tmp` contains only the environment,
downloaded encoder weights and HF dataset cache and is ephemeral.

Each encoder is extracted once per dataset/config; independent jobs use up to two
GPUs. No backbone fine-tuning, ImageNet training, CLIP or ADE20K jobs
are launched. NYUv2 GPU runtime/peak-memory estimates still require a real run.
For local MAT input instead of the mirror:

```bash
python -m mpsae.patch_scene.prepare_nyuv2 --source /path/to/input --destination work/nyuv2/dataset
```

Use `python -m mpsae.patch_scene plan --config configs/patch_scene_voc.yaml`
or `configs/patch_scene_nyuv2.yaml` to inspect the exact campaign before running.


## Cityscapes from Hugging Face

Use `scripts/kaggle_cityscapes_cell.py` as one complete Kaggle cell (Internet on,
GPU T4 x2 recommended). Push these files to the GitHub refactor branch first.
The shared launcher accepts `DATASET="cityscapes"`; no manual data upload/token.
Canonical configuration: `configs/patch_scene_cityscapes.yaml`.

- Frozen SigLIP2 Base and DINOv2 Base; CSR, CSRv2, MPSAE.
- Train each method separately at K=3,12,48; evaluate only its matched K.
- 18 fits + 18 sparse evaluations + 2 dense evaluations = 20 total evaluations.
- Chris1/cityscapes pinned at f5245e5b4ce9ec8c1c00e17921e176c566914a7c;
  original 14 train and 3 validation Parquet shards, 2975/500 images; no test.
- Source masks store original labelIds in identical RGB channels. Validate and
  map the 19 evaluated classes to 0..18, all ignored IDs to 255. Reject actual
  color masks. IDs are image-byte hashes; reject duplicates and split overlap.
- Resize complete images to 336x672, preserving the 2:1 aspect ratio. All valid
  patches at this resolution train the SAE; no 262144 candidate-pool cap.
  This is not training on all native-resolution pixels/patches.
- Like current NYUv2, retrieve from 65536 references, sampled with the existing
  global randomized class-frequency rule. This is a declared memory-limited
  protocol, NOT the per-image sampler/full-bank Hummingbird or Franca benchmark.
  Dense and sparse comparisons share support selection. No hidden bank changes.
- SAE max 20 epochs, batch 1024, LR 4e-5; training-loss early stopping after
  epoch 15, patience 3, relative threshold .001. Frozen encoders; no validation
  selection. Extraction batch 2 and 16-image shards bound CPU/GPU memory.
- Native-resolution validation masks, weighted 30-NN, temperature .02; mIoU,
  pixel accuracy and measured nnz, with CSV/LaTeX tables. No cross-K sweep.

Original Parquets are read directly without another full Arrow dataset copy.
Features are extracted once per encoder, cached and streamed across all fits.
Two independent GPU workers share a task queue. One GPU is also supported.
Cached sources are in /tmp; reusable experiment state is in
`/kaggle/working/cityscapes`. Download `CITYSCAPES_results.zip` from the displayed
link or the notebook Output panel. Save Version to retain notebook outputs.
The ZIP includes tables and experiment settings, not model weights/feature caches.

Real Cityscapes GPU runtime has not been measured. All-patch training is larger
than NYUv2; do not assume completion within one Kaggle session. Resume using the
same complete workspace and configuration, not the small results ZIP alone.
