# Kaggle: training and inference sparsity on two GPUs

The active patch campaign uses **SigLIP 2 and DINOv2**, **VOC and ADE20K**, and
**CSR, CSRv2 and MPSAE**. CLIP remains readable for historical campaigns but is
excluded from the active config, extraction, fits and reports.

Train an independent SAE at each **K = 3, 12, 48**, then evaluate every checkpoint
at **inference K = 3, 12, 48**. The backbone is frozen; this does not enable the
retired nine backbone fine-tuning runs. Each training K has a separate fit identity
and checkpoint. Changing inference K reuses that checkpoint.

## Work count and reuse

| Work | Count |
|---|---:|
| Encoder/dataset pairs | 4 |
| Independent SAE fits: 4 x 3 methods x 3 training K | 36 |
| Matched train K = inference K evaluations | 36 |
| Cross-K evaluations | 72 |
| Sparse evaluations (matched + cross-K) | 108 |
| Dense references, measured once per encoder/dataset | 4 |
| Total evaluations | 112 |

The matched evaluations are already included in the full sweep, not repeated.
The previous suite had 18 fits and 60 evaluations including CLIP. The new suite
has 18 more fits and 52 more evaluation points in total, but the reusable subset
is only 12 fits / 40 evaluations for SigLIP 2 and DINOv2. With that full workspace
retained, the actual additional work is **24 fits and 72 evaluations**. The 20
old CLIP evaluation points are excluded, not deleted. If only the original
K=12-inference results exist, 16 evaluation points can be reused and 96 remain.

Old configs are preserved:

- `configs/patch_scene_kaggle_k12.yaml`: original 24-point suite.
- `configs/archive/patch_scene_kaggle_train12_three_encoders.yaml`: 60-point suite.

Existing K=12 feature, checkpoint and evaluation identities are preserved. Model
revisions are reused from the old source receipt. Cache reuse requires the original
workspace and compatible packages/settings; a results-only ZIP has no weights.
A new empty session performs all extraction, fitting and evaluation again.

## Two-GPU execution

Push the updated code to the GitHub `refactor` branch. On Kaggle, enable Internet
and select **T4 x2**. Open the updated `notebooks/patch_scene_kaggle.ipynb`, or keep
the existing Python 3.11 / CUDA 12.6 installation cells and use:

```python
import subprocess
from pathlib import Path

REPO = Path('/kaggle/working/MPSAE')
WORK = Path('/kaggle/working/patch-scene-v2')
PYTHON = '/tmp/mpsae-env/bin/python'
subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
CONFIG = str(REPO / 'configs/patch_scene_kaggle.yaml')
BASE = [PYTHON, '-u', '-m', 'mpsae.patch_scene']
subprocess.run(BASE + ['plan', '--config', CONFIG], cwd=REPO, check=True)
subprocess.run(BASE + ['run', '--config', CONFIG, '--workspace', str(WORK),
                      '--encoder', 'all', '--dataset', 'all',
                      '--devices', 'cuda:0', 'cuda:1', '--cpu-threads', '2'],
               cwd=REPO, check=True)
```

Use the new config for this grid: an old custom YAML retains its old selection.
The notebook creates an editable YAML outside the Git checkout using PyYAML;
do not serialize floats as raw JSON and assume the YAML loader keeps their types.

One spawned process owns each GPU. Preparation runs once per encoder/dataset,
then unlocks a dense job and independent method/training-K jobs. Each fitting job
evaluates all inference K before releasing its model. Workers prefer their current
feature bank when possible and take other ready work when it would otherwise be
idle. At most one CPU bank is retained per worker. Model initialization, batches,
learning rates and loss recipes do not change with GPU count. There is no DDP,
gradient accumulation or pooled VRAM. `--devices cuda:0` selects one GPU.

Only the coordinator writes campaign rows, tables and the ZIP. Workers write
distinct fit/evaluation artifacts; feature preparation must finish before consumers
start. The workspace lock prevents a second coordinator from modifying these
artifacts concurrently. Do not run `export` concurrently with the training command.

Assignments and a heartbeat are printed every 20 seconds. Inspect
`campaigns/<id>/scheduler.json` for active jobs, and `campaigns/<id>/logs/` for
detailed worker logs. A failed worker causes a nonzero exit; completed results
remain available. Rerun the same command to reuse completed work and saved progress.

Two GPUs can approach twice the throughput of serial independent jobs, but this
is not a measured speedup. Extraction, shared disk/CPU work, unequal job lengths
and higher K costs affect runtime. New K=48 fits/evaluations cannot be assumed to
take the same time as K=12. Use recorded `train_seconds` and `query_seconds` to
estimate time on the actual server; no fixed hour estimate is implied.

## Scientific protocol

| Setting | Value |
|---|---|
| Encoders | SigLIP 2-B/16, DINOv2-B/14 |
| Input / latent dimensions | 768 / 8192 |
| Training K | 3, 12, 48, independent initialization per fit |
| Inference K | 3, 12, 48 for every fit |
| SAE optimization | 10 epochs, batch 1024, Adam LR 4e-5 |
| Training / support patch limits | 262144 / 65536 |
| Retrieval | exact FP32 cosine; 30 neighbors; temperature 0.02 |
| Images | entire image resized to 448 square; no center crop |
| Seed | 1227 |

This is dataset-specific patch SAE learning: each method starts from fresh SAE
weights on the dataset's training split. Existing K=12 fits may be restored, but
new K=3/48 fits do not warm-start from them. Backbone weights stay frozen. All
methods use the same initialization seed, bank and batch ordering within a fit
setting. Final-epoch checkpoints are evaluated without validation-based selection.

CSRv2 retains its training-K curriculum and uses dominant nonvoid training-mask
labels in its contrastive loss. CSR/MPSAE do not use class targets in their loss.
The contrastive coefficient is 0.1 and temperature is 0.2. MPSAE retains MMPOT on
K, 2K and 4K views with weight 2.0 for all three selected training budgets. The
existing rule for target training K > 64 would use weight 0.5; changing inference
K never changes the trained objective. Optimizer and solver settings are unchanged.

VOC uses 1464 training / 1449 validation images and 21 classes including background;
ADE20K uses 20210 / 2000 images and 150 classes. Void pixels are ignored. Training
patches are sampled uniformly within image quotas from nonvoid patches. The
support bank uses the existing class-frequency-based sampler on that training bank
and is shared by all methods/budgets. This uses training annotations for retrieval;
CSRv2 has additional label supervision during SAE fitting.

SigLIP 2 gives a 28x28 spatial grid and DINOv2 gives 32x32. CLS is removed from
DINOv2. Cached backbone extraction uses FP16 autocast and FP16 storage with a
rounding-error check; SAE training, sparse encoding and ranking use FP32. Feature
caches are shared across K. Predictions aggregate soft patch labels with
temperature-weighted neighbors, then bilinearly upsample probabilities to native
image size. Empty query codes abstain and count as errors; similarity ties use
gallery order. Evaluation reports global-confusion-matrix mIoU and pixel accuracy.

K is an upper bound on nonzero entries, not the full latent dimension. Search uses
a dense transposed lookup workspace (about 2 GiB for 65536 x 8192 FP32 values),
although the codes are sparse. Report measured nonzero counts and workspace costs
separately. Query timing includes streaming/encoding/search/upsampling/metrics,
not just a search kernel. This is Hummingbird-style decoding with dataset-specific
SAEs, not a reproduction of Hummingbird pretraining or cross-dataset SAE transfer.

## Tables and local download

After every completed evaluation, the coordinator refreshes:

```text
/kaggle/working/patch-scene-v2/patch-scene-results.zip
```

The notebook copies it to `/kaggle/working/patch-scene-results.zip` when the run
cell exits and displays a download link. No Hugging Face token/upload is used.
The parallel runner currently supports local artifacts only; the single-GPU CLI
retains optional `--repo`/`--publish` support.

The ZIP includes:

- `results.tex`, `table_voc.tex`, `table_ade20k.tex`: matched train/inference K.
- `inference_sweep.tex`, `sweep_voc.tex`, `sweep_ade20k.tex`: full cross-K grid.
- `pixel_accuracy.tex`: matched-K pixel accuracy.
- `results.csv`, `comparisons.csv`, `results.json`: every condition, measured
  sparsity, per-condition gains, missing rows and provenance.
- `config.json`, optional `sources.json`, `README.txt`, and `tables_preview.tex`.

For a legacy single-training-K campaign the main table still displays its inference
sweep. Dense is evaluated once and repeated only for display. Delta compares MPSAE
with max(CSR, CSRv2) at the same dataset, encoder, train K and inference K. Best
sparse values are bold only when all three methods exist; rounded ties stay tied.
Missing values are `--`. No results are averaged across training budgets or selected
by validation performance. These are single-seed results, not significance tests.

Use `\usepackage{booktabs}` and `\input{results.tex}` in the paper. After downloading,
`pdflatex tables_preview.tex` previews the matched tables and dataset-specific sweeps.
Kaggle does not need LaTeX installed to generate the sources.

After the runner has stopped, rebuild the ZIP without fitting or GPU use:

```bash
/tmp/mpsae-env/bin/python -m mpsae.patch_scene export \
  --config configs/patch_scene_kaggle.yaml --workspace /kaggle/working/patch-scene-v2
```

If the notebook wrote a custom YAML, pass that same path to `--config` instead.
Download before ending the session or save a notebook version with output files.
The small ZIP excludes checkpoints and feature caches. Restore the full workspace
to resume in another session; saving notebook code alone does not retain results.

An offline parallel plumbing check uses synthetic data, never benchmark results:

```bash
python -m mpsae.patch_scene smoke --devices cpu cpu --cpu-threads 1 \
  --config configs/patch_scene_kaggle.yaml --workspace work/patch-smoke
```

Smoke outputs are isolated under `SMOKE_ONLY` and cannot be published.
