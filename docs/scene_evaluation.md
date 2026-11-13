# Patch SAE fine-tuning and segmentation

## ADE20K / SceneParse150

ADE20K is available in `configs/scene_ade20k.yaml`. Run it separately from VOC:

```bash
bash scene_only.sh --config configs/scene_ade20k.yaml --workspace /workspace/scene-ade20k-v1 --publish
```

On a completely new server, first run `bash scene_only.sh --setup-only` and
`.venv-scene/bin/hf auth login` as described below. No additional packages are needed.
The same source ImageNet SAEs are downloaded automatically; ADE adaptations start
from those sources, not from VOC-adapted weights.

The dataset is [zhoubolei/scene_parse_150](https://huggingface.co/datasets/zhoubolei/scene_parse_150),
configuration `scene_parsing`, using the Hugging Face Parquet conversion pinned to
`e660d866a1351c70bcf07925d2600b60bd6e3bc3`. Only train and validation shards are loaded,
without running remote dataset scripts. The protocol uses 20,210 training images,
2,000 validation images and 150 semantic classes. These are **validation scores**;
the unlabeled challenge test split is not evaluated. `scene_category` is an image-level
scene label and is never used as a semantic patch target.

Following the [official evaluation](https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/intersectionAndUnion.m),
raw mask ID 0 is ignored. IDs 1..150 map to internal 0..149; ignore maps to 255.
There is no scored background class. CSRv2 uses the dominant valid semantic class,
and all-void patches are excluded. ADE mIoU averages all 150 class IoUs (zero for
zero-union classes); pixel accuracy ignores void pixels. Empty sparse queries count
as abstentions/false negatives. Counts use int64, including class 149.

All four backbones, source K values, inference sweeps, decoder settings and SAE
adaptation hyperparameters match VOC. This adds 12 independent SAE fits and 160
evaluation points (78 adapted, 78 frozen, four dense). Running both configs gives
24 new SAE fits and 320 evaluation points. Backbone weights remain frozen.

The same 65,536-patch cap gives three sampled patches per training image, at most
60,630 total, before void exclusions. Five epochs at batch 1,024 take at most 300
updates per method. The training/cache budget is bounded, but extracting features
from all 20,210 training images is a larger first-run cost than VOC. Memory estimates
in `plan` count feature arrays; labels, original masks and download caches are extra.
The dataset-specific label contract is included in cache/report identities. Existing
VOC cache and report identities remain compatible. Raw features are reusable across
methods and inference K; train and validation features remain separate.
Per-image ADE confusion records store only nonzero integer counts, avoiding tens of
thousands of zero entries per image per inference K; aggregation remains exact.

ADE plots and CSVs identify the dataset explicitly. They have the same frozen/adapted
curves and measured dense baselines as VOC. This remains our adapted Hummingbird-style
retrieval benchmark, not a reproduction of a published ADE20K leaderboard result.
`--publish` uses the existing Hub paths `scene-v1/adaptations/<identity>` and
`scene-v1/reports/<identity>` for both datasets; dataset-aware identities prevent
collisions, and the report manifests/CSV identify which dataset was evaluated.

```bash
.venv-scene/bin/python -m mpsae.scene plan --config configs/scene_ade20k.yaml
```

This opt-in module applies existing ResNet-50, ViT-B/16, DINOv2-S/14 and
ConvNeXt-Tiny SAEs to local features. The default fine-tunes only the SAE on cached
VOC training patches, then compares frozen and adapted representations using the
same retrieval decoder. The backbone stays frozen and there is no trained segmentation
head. The main ImageNet campaign and disabled backbone fine-tuning campaign are unchanged.

## Adaptation settings

The archived frozen configuration is `configs/archive/scene_voc_frozen_multi.yaml`.
`configs/scene_voc.yaml` enables SAE adaptation: five epochs, batch 1,024, constant
Adam learning rate 1e-5, seed 1227. Fine-tune once at the source training K (8 for
ResNet-50, 12 for the other backbones), then sweep inference K. There are 12 new
fits and 78 adapted evaluation points, plus 78 frozen points and four dense baselines.

Start from the completed ImageNet SAE weights with a fresh optimizer and reset
activity counters. Both the tied dictionary and biases are trainable. Loss weights,
Adam betas/epsilon/weight decay, AuxK and OT solver settings are read from the source
checkpoint's recorded recipe, not silently replaced with current defaults. Reuse
`methods.compute_step` with a fixed target budget. CSRv2 does not restart its completed
pretraining annealing schedule. CSR keeps instance contrastive loss, CSRv2 keeps
supervised contrastive loss, and MPSAE keeps its K/2K/4K MMPOT matching gap.

CSRv2 uses the dominant nonvoid class of each patch; ties choose the smallest class
ID. Mixed patches remain in the dataset, and their purity and class distribution are
logged. Retrieval still uses the full soft class distributions. CSRv2 therefore has
supervised adaptation, while CSR/MPSAE do not use labels in their loss. All share the
same label-informed patch sampling. This is a comparison of method-specific adaptation
recipes, not a comparison with identical loss supervision. The new recipe is not a
claim to reproduce CSRv2's original backbone fine-tuning protocol.

Only VOC train patches enter gradient updates. Validation is used for scoring only;
the final epoch is selected without validation-based checkpoint selection. All patches
are shuffled across images using the same order for all methods. A final singleton
batch is merged into the preceding batch to keep contrastive/OT objectives defined.
Training uses FP32 with TF32 disabled. At the default upper bound, there are 315
updates per fit. `adaptation.residency: device` keeps the small training bank on the
GPU; `cpu` transfers batches instead. Checkpoints are saved every 50 updates and
at each epoch end. A resume uses the new adaptation optimizer, never the ImageNet optimizer.

Feature caches and frozen report identities remain compatible. Adapted checkpoint
identities include source bytes, training feature/label hashes, objective and optimizer
settings, K and seed. Changing the recipe creates a different fit. Source checkpoints
are never overwritten. Both support and query codes are regenerated with the adapted
SAE at each inference K. Raw feature caches can be reused; old sparse codes cannot.

## Plan and scientific interpretation

1. Validate completed source fits and their original backbone weight provenance.
2. Load VOC images/masks from a pinned Hugging Face revision. Extract the exact
   frozen backbone's spatial feature map before average pooling, once.
3. Within each backbone, sample the same annotated support patches for every method. Reuse one bank
   and all query features across dense, CSR, CSRv2 and MPSAE evaluation.
4. Adapt the SAE on cached training patches and compare it with the frozen source.
   Feed raw local features to both; normalize only retrieval keys and queries.
   Sweep inference K without fitting another checkpoint for every inference K.
5. Retrieve neighbors, average soft patch labels with similarity weights,
   upsample probabilities and score original-resolution masks. Save mIoU,
   pixel accuracy, per-class IoU, sparsity, key storage and search time.

The decoder follows the mechanism in
[Towards In-context Scene Understanding](https://arxiv.org/pdf/2306.01667),
Section 3.1 and Appendix A: cosine retrieval, soft patch-label distributions,
temperature-weighted aggregation, and bilinear upsampling. Defaults use 30
neighbors and temperature 0.02. SAE sparsity K and neighbor count are independent.

This is an **adapted protocol**, not reproduction of the Hummingbird tables:
spatial stride depends on the backbone, the bank is smaller, search is exact rather than
ScaNN, and preprocessing uses a deterministic square resize without augmentation.
The default standard VOC split also differs from augmented VOC training.
Most importantly, the SAE was trained on pooled image features, not local
features. Frozen evaluation measures direct transfer; SAE adaptation measures
downstream adaptation on those patches. Neither guarantees improved segmentation.
Always include the dense local
backbone baseline. No contextual pretraining or Hummingbird model is introduced.

## Data and geometry

The default [shijli/voc2012](https://huggingface.co/datasets/shijli/voc2012)
`segmentation` config has 1,464 support and 1,449 validation images; the optional
`segmentation_aug` config has 10,582 support images. Image IDs must be unique
and disjoint across support/query splits. Validation labels are used only for
scoring. Train masks construct the memory and supply dominant-class training targets
for CSRv2. CSR/MPSAE loss terms consume features only.

Images are bilinearly resized to the configured resolution; masks use nearest
interpolation. The spatial grid and feature width depend on the backbone. Mask values
0..20 are classes (including background); 255 is ignored. Average one-hot class
labels within each cell, excluding void pixels, then renormalize. Never average
numeric class IDs. Sampling uses the paper's per-image class-frequency heuristic,
excluding all-void cells. Original query masks are retained for scoring.

An all-zero query code abstains; its nearest-grid spatial region counts as false
negatives instead of being dropped. Equal retrieval scores use gallery order.
Absent classes with zero union are omitted from mIoU and recorded as null; ordinary
full VOC evaluation should include all 21 classes. Scores in JSON/CSV are fractions.

## Run

### Completely new Linux GPU server

Clone/update this repository and run the following **from its root**:

```bash
bash scene_only.sh --setup-only
.venv-scene/bin/hf auth login
bash scene_only.sh --workspace /workspace/scene-v1 --publish
```

Requirements: Linux x86_64, Python 3.10-3.12 with venv support, a working NVIDIA
driver, and internet access. On a minimal Ubuntu image, install the basic tools
first: `sudo apt-get update && sudo apt-get install -y git python3 python3-venv ca-certificates`.
Root containers can omit `sudo`. The installer does not change the host driver.
If the container's default Python is newer, use `--python /usr/bin/python3`
or another available Python 3.10-3.12 interpreter during setup.

`--setup-only` creates an isolated `.venv-scene` with pinned evaluation
dependencies. It uses the published [PyTorch 2.9.1 / torchvision 0.24.1 CUDA
wheels](https://pytorch.org/get-started/previous-versions/#v291), with cu128 by
default (RTX 5090 supported; this profile requires driver >=570.124.06).
Use `--cuda cu130` for the CUDA 13 profile (driver >=580.65.06). No system CUDA
toolkit, compiler, FFCV, CuPy, W&B or ImageNet preparation is needed.
The existing training `.venv` remains separate. Setup exercises actual backbone,
SAE and retrieval operations on the GPU, plus Arrow/HF image decoding.

If `HF_TOKEN` is already exported, or stored in the repository `.env`, setup and
evaluation can be started together:

```bash
bash scene_only.sh --setup --workspace /workspace/scene-v1
```

The launcher calls `python -m mpsae.scene bootstrap`. It restores the twelve
selected completed fits from `quangnguyenopt/MPSAE`, validates their backbone
provenance, downloads the backbone weights and VOC, extracts features, adapts the SAE, evaluates,
and writes plots. You do not need the old workspace or any ImageNet downloads.
All 36 required source artifacts were confirmed in the Hub at revision
`22aa69e63d565c238bf334a79fdf89ad3db7428d` during implementation. Use
`--revision <commit>` to request that revision explicitly.

Rerun the same command without `--setup` to resume. Source receipts pin the
checkpoint revision; later remote pushes do not silently replace it. An absent
local checkpoint cache is repaired at the pinned revision. Use `--refresh-source`
only when deliberately switching to newer remote checkpoints. Per-image results
and features remain reusable when their identities match.

For result backup, authenticate with a token that can write to the result repo:

```bash
bash scene_only.sh --workspace /workspace/scene-v1 --publish
```

Each completed adaptation is backed up immediately to
`scene-v1/adaptations/<identity>/` (checkpoint.pt, config.json, summary.json, train.jsonl),
before segmentation evaluation. Checkpoints include optimizer state for recovery.
This also uploads completed JSON/CSV/PDF/PNG reports to
`quangnguyenopt/MPSAE/scene-v1/reports/<identity>/`. Upload failures preserve local
results; rerunning with `--publish` retries the backup using completed evaluations.
It does not upload dataset caches, credentials, or images.

`status.json` records the current/failed stage; `latest.json` points to the latest
completed report. Project progress logs are in `logs/scene.log`. For long runs,
use `tmux` so disconnecting SSH does not terminate the process. Allow roughly
40 GiB free disk for environment installation, download caches and evaluation
artifacts; actual peak depends on pip caching and source checkpoint sizes.

### Existing environment / manual stages

From the repository root, using the existing environment:

```bash
python -m mpsae.scene plan --config configs/scene_voc.yaml
python -m mpsae.scene run --config configs/scene_voc.yaml --source-root work/direct-v1 --workspace work/scene-v1 --device cuda:0
```

The default suite has 12 existing SAE checkpoints, 12 new SAE adaptations and 160
evaluation points (including frozen comparisons and four dense baselines).
Each backbone runs sequentially and has its own dense baseline. Previously completed
frozen results are reused when the source, grid, data, precision and decoder settings match.

| Backbone | Train K | Inference K | Image size | Grid |
|---|---:|---|---:|---|
| ResNet-50 | 8 | 2, 4, 8, 16, 32, 64, 128, 256 | 512 | 16x16 |
| ViT-B/16 | 12 | 3, 12, 48, 96, 192, 384 | 448 | 28x28 |
| DINOv2-S/14 | 12 | 3, 12, 48, 96, 192, 384 | 448 | 32x32 |
| ConvNeXt-Tiny | 12 | 3, 12, 48, 96, 192, 384 | 448 | 14x14 |

Edit the `experiments` list in `configs/scene_voc.yaml` to select groups or budgets.
The original one-backbone K=8 configuration is preserved at
`configs/archive/scene_voc_resnet50_k8.yaml`; the flat config format still works.
`latest_suite.json` lists the report directories for a completed suite.

ViT/DINOv2 use final normalized patch tokens with CLS/register tokens excluded.
Their positional embeddings are interpolated by timm when loading at the scene
resolution. ConvNeXt runs its original pre-logit head on each one-cell feature
map so the local representation retains head normalization. This is a local
adaptation of an image-level head; averaging these local vectors need not equal
the original pooled vector. Original SAE checkpoints remain unchanged; adapted weights
are saved separately. DINO refers here
specifically to the existing DINOv2-S/14 backbone, not original DINO.

These compare image-level SAE-to-spatial transfer and patch adaptation. Different native grids and
head semantics mean cross-backbone mIoU is not a controlled sparsification-only
comparison: compare each SAE with its own dense spatial baseline first.

To restore existing source fits from HF on another machine:

```bash
hf auth login
python -m mpsae.scene restore --repo quangnguyenopt/MPSAE --prefix direct-v1
```

Use the printed `source_root` with `run`. This downloads only selected
checkpoints/configs/summaries and embedding **JSON sidecars**, not ImageNet images
or embedding arrays. Missing checkpoint or weight provenance fails before feature
extraction; a model name alone is insufficient to establish compatibility.

`prepare` accepts the same arguments as `run` and stops after feature caching.
Completed caches are verified by SHA256 and reused without dataset/backbone
downloads. Dataset `main` is resolved once to a commit in the workspace; change
the explicit revision to select a different dataset version. Feature cache keys
include weights, data revision, resize, precision, seed and support budget, but
not SAE method or inference K. Query features are streamed one image at a time;
each completed image's evaluation is saved so interruptions can resume.

The native extractor runs fp16 autocast on CUDA with float32 storage. SAE encoding
and retrieval use float32, TF32 disabled. This precision policy is explicit and
differs from the older ImageNet SAE autocast evaluation. Features are never
L2-normalized before SAE encoding.

## Outputs and resource budget

`work/scene-v1/reports/<identity>/` contains `results.json`, `results.csv`, a full
manifest, per-image resumable metrics, and publication figures (vector PDF, 400 dpi PNG):

- `segmentation`: mIoU and pixel accuracy versus inference K.
- `retrieval`: mIoU versus measured active dimensions and key storage.
- `adaptation_gain`: paired fine-tuned minus frozen mIoU, in percentage points.
- `search_time`: search seconds per validation image, explicitly a partial timing metric.

Method colors and markers are consistent. Solid/filled curves are fine-tuned;
dashed/open curves are frozen. Dense values are measured in this experiment. No
published curves, smoothing, or invented confidence intervals are included. Plots
connect increasing inference K even when actual nonzeros saturate. `figure_notes.txt`
documents units and interpretation. The completed suite adds combined plots/CSV under
`reports/suite-<identity>/`; `latest_suite.json` records that path.
Key storage excludes label values and runtime
buffers; search time is not end-to-end latency or a throughput benchmark.

Each default backbone uses at most 44 patches per support image: 64,416 total before
all-void exclusions. Dense keys occupy about 503 MiB; K=8 sparse keys about
3.9 MiB (int32 indices + float32 values). The ResNet raw spatial cache totals about 3.3 GiB
plus labels/masks. Search expands only chunks of compact codes; it does not
allocate a full bank-by-8,192 dense latent matrix or all-query similarity matrix.
Extraction batch size is independent of SAE training batch size. Reduce
`extraction_batch`, `gallery_chunk` or `query_block` for lower GPU peak memory.
Exact search can still be slow: these estimates are storage, not measured speed.

```bash
python -m mpsae.scene plot --results work/scene-v1/reports/IDENTITY/results.json
python -m mpsae.scene publish --repo quangnguyenopt/MPSAE --results work/scene-v1/reports/IDENTITY/results.json
```

Publishing is explicit and uploads adapted checkpoints/logs and result JSON/CSV/plots. Feature caches stay
local; copy the feature directory and its manifest to reuse it elsewhere. No
large upload or server experiment runs automatically when this module is added.

To download results and adapted checkpoints on another machine:

```bash
hf download quangnguyenopt/MPSAE --repo-type dataset --include "scene-v1/reports/*" "scene-v1/adaptations/*" --local-dir ./MPSAE-results
```

To reuse recovered adaptations, copy the downloaded `scene-v1/adaptations` directory
into the selected scene workspace, retaining its identity subdirectories. Rerun the
same pinned source and dataset settings. A matching completed adaptation is checked
and loaded without training again. Image/backbone extraction is still required if
the feature cache has not been copied to the new server.
