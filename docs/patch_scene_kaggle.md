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


## MPSAE only at a chosen MMPOT weight

The Kaggle launcher also accepts `METHODS`, `INCLUDE_DENSE`, and `MMPOT_WEIGHT`:

```python
runpy.run_path(str(REPO / "scripts/kaggle_segmentation.py"),
    init_globals={"DATASET": "nyuv2", "METHODS": ["mpsae"],
                  "INCLUDE_DENSE": False, "MMPOT_WEIGHT": 0.1},
    run_name="__main__")
```

This runs six fresh MPSAE fits (SigLIP2/DINOv2, K=3/12/48) and six matched-K
 evaluations, with no CSR, CSRv2 or dense evaluation jobs. It keeps the canonical
20-epoch maximum, early stopping, and all other NYUv2 settings. Features must
still be extracted on a fresh notebook; compatible caches/fits are reused when
present. Two available GPUs run independent jobs. The generated config is saved
in `/kaggle/working/nyuv2/config-kaggle.yaml`; tracked YAML files are unchanged.
Download `/kaggle/working/NYUV2_mpsae_ot0p1_results.zip`. CSV/JSON contain only
selected result rows. Comparison exports retain their standard schema, with
uncomputed baselines and gains left blank/`--`, never zero or fabricated scores.
Selection changes the campaign ID, but preserves feature and fit cache identities.
The default launcher still runs all methods and the dense reference.


## VOC transported-mass ablation (SigLIP2 / MPSAE)

Copy `scripts/kaggle_voc_mass_cell.py` into one fresh Kaggle cell after pushing
this update to the `refactor` branch. Enable Internet and GPU T4 x2. This installs
the environment once, downloads VOC once, and runs independent jobs on up to two
GPUs within each campaign. No Hugging Face token or output upload is needed.

- `MMPOT_MASSES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]` varies transported mass s.
- `MMPOT_WEIGHT = None` retains the VOC default lambda=2.0. A positive float
  overrides lambda, held fixed for the whole mass sweep. This is NOT s.
- SigLIP2 only, MPSAE only; no CSR, CSRv2, dense or DINOv2 jobs.
- Existing VOC settings: train K=3,12,48; each checkpoint evaluates K=3,12,48;
  10 epochs, batch 1024, LR 4e-5, up to 262144 sampled training patches,
  reference bank 65536, 30 neighbors, temperature .02. No early stopping.
- Six masses mean 18 fits and 54 evaluations (18 matched, 36 cross-K).
- The existing partial Greenkhorn solver is unchanged; 0<s<1 is enforced.
  s=1 is supported through a separate balanced log-Sinkhorn branch; no implicit
  loss normalization is added. Include `1.0` in `MMPOT_MASSES` (Kaggle) or
  `--masses 0.9 1.0` (mass-sweep CLI) to compare the endpoint. The existing six
  partial masses remain the default. Fits, tables and runtime provenance record
  the selected solver, and existing partial results remain reusable.
- Data/feature/source identities ignore s. Fit identities include s. Each new
  mass starts from the same seeded initialization, not another mass's checkpoint.
  Compatible existing s=.9 fits/evaluations can be reused, even from larger campaigns.

The launcher prints the immutable sweep ID and displays a download link named
`VOC_mpsae_mass_<sweep_id>_results.zip`. Results are also saved under:

```text
/kaggle/working/voc/
  experiment-index.csv                  all campaign conditions, including historical runs
  experiment-index.json                 same, with provenance and scan errors
  campaigns/<campaign_id>/              per-mass settings/results/checkpoints referenced
  ablations/<sweep_id>/
    plan.json                           immutable experiment list and resolved settings
    results.csv                         all mass/train-K/inference-K results and statuses
    results.json                        full provenance and last logged OT diagnostics
    matched_results.csv                 matched K subset, including accuracy/nnz
    matched_comparisons.csv              mIoU by mass, with lambda and seed explicit
    matched_comparisons.tex              paper table (no best-mass selection)
    mass-sweep-results.zip              combined results, even when partially completed
```

The register is rebuilt from all saved campaigns after each mass, so restarting
or running a different mass list does not erase older experiments. Complete,
pending, incomplete and failed entries are distinguished; missing values stay
blank, never zero. Checkpoint and result paths are relative to the workspace.
The ZIP also includes the experiment index and per-campaign settings/runtime/
attempt receipts. It excludes actual checkpoints, image data and feature caches.
Keep the full workspace to resume training; the ZIP preserves result/history
information but is not a training backup. Solver diagnostics are last logged
values, not a guarantee of optimality throughout training. Existing reused fits
may not have execution Git metadata from their original training invocation.

For an existing server environment, run from the repository root:

```bash
python -m mpsae.patch_scene.mass_sweep --plan
python -m mpsae.patch_scene.mass_sweep --workspace work/voc --devices cuda:0
# With two GPUs: --devices cuda:0 cuda:1
# Rebuild the same sweep tables/index without downloading/training:
python -m mpsae.patch_scene.mass_sweep --workspace work/voc --export-only
```

Use the same `--config`, `--masses` and `--weight` to rebuild a customized sweep.
The default CLI selects VOC/SigLIP2/MPSAE. `--encoders dinov2` is available for
explicit extensions. Re-running with unchanged settings resumes compatible work.
Do not run two launchers concurrently in the same workspace.


## Multiple methods and loss weights, including balanced MMOT (s=1)

Copy the complete contents of `scripts/kaggle_segmentation_weights_cell.py`
into one new Kaggle code cell, after pushing this update to the `refactor` branch.
Enable Internet and one GPU or T4 x2. The cell installs its environment once,
prepares the selected dataset once, and keeps results locally; no HF token or
result upload is required. Edit only the settings at the top:

```python
DATASET = 'voc'                       # also 'nyuv2' or 'cityscapes'
ENCODERS = ['siglip2']                # or ['siglip2', 'dinov2']
METHODS = ['csr', 'csrv2', 'mpsae']
TRAIN_KS = [3, 12, 48]
MMPOT_MASS = 1.0                     # transported mass s, NOT the loss weight
MMPOT_WEIGHTS = [0.5, 1.0, 2.0]     # coefficients lambda before the transport loss
INCLUDE_DENSE = True
```

For MPSAE this is `L_other + lambda * L_transport`. Exact `s=1` uses the shared
balanced log-domain Sinkhorn solver (`balanced-log-sinkhorn-v1`), with uniform
equality marginals and zero slack entropy. It does not approximate s with 0.999
or send full mass to the partial solver. `0<s<1` retains the existing partial
Greenkhorn path. The objective, encoder, optimization settings, patch selection,
and retrieval protocol are otherwise unchanged. The balanced matching-gap
gradient is the diagonal reference plan minus the solved transport plan.

CSR and CSRv2 have no MMOT term: they are fitted/evaluated once per encoder/K and
reused across weights. Each MPSAE weight/K starts its own seeded fit, and only an
exact compatible fit may resume. A previous partial-transport MPSAE checkpoint
cannot be resumed as balanced MMOT. All methods share the same cached features
and support bank; the two-GPU scheduler runs independent jobs, not DDP, so batch
size remains 1024. Existing default experiments/configuration files are unchanged.

The weight grid uses **matched K only**, including on VOC. With three K values
and three weights, SigLIP2 alone requires **15 fits, 16 unique evaluations**
(3 CSR + 3 CSRv2 + 9 MPSAE + 1 dense). Both encoders require 30 fits and 32
evaluations. For MPSAE alone, set `METHODS=['mpsae']` and `INCLUDE_DENSE=False`:
9 fits/evaluations per encoder. `TRAIN_KS` must be at most 64 for this grid;
the existing larger-K loss-weight policy is not overridden. VOC keeps 10 epochs;
NYUv2/Cityscapes keep their canonical 20-epoch maximum and early stopping.

Do not combine `MMPOT_WEIGHTS` with the singular `MMPOT_WEIGHT` or the different
`MMPOT_MASSES` ablation. Changing datasets only requires editing `DATASET`.

Download the displayed `<DATASET>_ot_weights_<sweep_id>_results.zip` link. It
contains **all selected weights**, not just the latest campaign:

- `plan.json`: dataset, encoders, methods, K, s, weights, solver and full settings.
- `grid_results.csv`: long table with s, lambda, seed, solver, mIoU, pixel accuracy
  and measured query/gallery nnz. Baselines appear once, with no transport weight.
- `grid_comparisons.csv` and `grid_results.tex`: matched-K mIoU comparison table.
- `experiment-index.csv/json`: all historical campaign statuses and fit paths.
- `campaigns/`: per-campaign settings, source provenance, runtime and results.

The combined archive is refreshed after each weight, also on an interrupted or
failed campaign. Unavailable results stay blank/`--`. Downloading the ordinary
`patch-scene-results.zip` still gives only one campaign; use the new combined ZIP.
Checkpoints and feature caches stay in `/kaggle/working/<dataset>` and are excluded
from the ZIP. Retain that complete workspace to resume across Kaggle sessions.

For an existing environment, the same driver works outside Kaggle:

```bash
python -m mpsae.patch_scene.weight_sweep --config configs/patch_scene_voc.yaml \
  --workspace work/voc --encoders siglip2 --methods csr csrv2 mpsae \
  --train-k 3 12 48 --mass 1 --weights 0.5 1 2 --devices cuda:0 cuda:1 --plan
# Remove --plan to run. Use --export-only to rebuild this grid without training.
```

For NYUv2, first use `prepare_nyuv2 --from-hf --destination <folder>` and pass
`--nyuv2-root <folder>`; the Kaggle launcher does this automatically. Retain all
weights in the report; do not select the best one on NYUv2's official test split.
