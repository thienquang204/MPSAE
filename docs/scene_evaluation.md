# Frozen spatial SAE evaluation

This opt-in module applies existing FF2048 ResNet-50 SAEs to local layer4
features. It does not fit a backbone, an SAE, or a segmentation head. The main
ImageNet campaign and disabled fine-tuning campaign are unchanged.

## Plan and scientific interpretation

1. Validate completed source fits and their original backbone weight provenance.
2. Load VOC images/masks from a pinned Hugging Face revision. Extract the exact
   frozen backbone's spatial feature map before average pooling, once.
3. Sample the same annotated support patches for every method. Reuse one bank
   and all query features across dense, CSR, CSRv2 and MPSAE evaluation.
4. Apply each SAE directly to raw local features. Normalize only retrieval keys
   and queries. Sweep inference K without fitting another checkpoint.
5. Retrieve neighbors, average soft patch labels with similarity weights,
   upsample probabilities and score original-resolution masks. Save mIoU,
   pixel accuracy, per-class IoU, sparsity, key storage and search time.

The decoder follows the mechanism in
[Towards In-context Scene Understanding](https://arxiv.org/pdf/2306.01667),
Section 3.1 and Appendix A: cosine retrieval, soft patch-label distributions,
temperature-weighted aggregation, and bilinear upsampling. Defaults use 30
neighbors and temperature 0.02. SAE sparsity K and neighbor count are independent.

This is an **adapted protocol**, not reproduction of the Hummingbird tables:
native ResNet stride is 32, the bank is smaller, search is exact rather than
ScaNN, and preprocessing uses a deterministic square resize without augmentation.
The default standard VOC split also differs from augmented VOC training.
Most importantly, the SAE was trained on pooled image features, not local
features. Matching channel width permits this transfer but does not guarantee
matching distributions or useful segmentation. Always include the dense local
backbone baseline. No contextual pretraining or Hummingbird model is introduced.

## Data and geometry

The default [shijli/voc2012](https://huggingface.co/datasets/shijli/voc2012)
`segmentation` config has 1,464 support and 1,449 validation images; the optional
`segmentation_aug` config has 10,582 support images. Image IDs must be unique
and disjoint across support/query splits. Validation labels are used only for
scoring. Train masks construct the memory, not gradient updates.

Images are bilinearly resized to 512x512; masks use nearest interpolation. The
unmodified ResNet produces 16x16 spatial cells with 2,048 channels. Mask values
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
bash scene_only.sh --workspace /workspace/scene-v1
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

The launcher calls `python -m mpsae.scene bootstrap`. It restores the three
selected completed fits from `quangnguyenopt/MPSAE`, validates their backbone
provenance, downloads the backbone weights and VOC, extracts features, evaluates,
and writes plots. You do not need the old workspace or any ImageNet downloads.
All nine required source artifacts were confirmed in the Hub at revision
`c22461f0585d60f2c4efc908a9c7230fb15ee1a6` during implementation. Use
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

This explicitly uploads completed JSON/CSV/PDF/PNG reports to
`quangnguyenopt/MPSAE/scene-v1/reports/<identity>/`. Upload failures preserve local
results; rerunning with `--publish` retries the backup using completed evaluations.
It does not upload dataset caches, credentials, or images.

`status.json` records the current/failed stage; `latest.json` points to the latest
completed report. Project progress logs are in `logs/scene.log`. For long runs,
use `tmux` so disconnecting SSH does not terminate the process. Allow roughly
25 GiB free disk for environment installation, download caches and evaluation
artifacts; actual peak depends on pip caching and source checkpoint sizes.

### Existing environment / manual stages

From the repository root, using the existing environment:

```bash
python -m mpsae.scene plan --config configs/scene_voc.yaml
python -m mpsae.scene run --config configs/scene_voc.yaml --source-root work/direct-v1 --workspace work/scene-v1 --device cuda:0
```

The default evaluates dense plus three K=8 source fits at inference K=8: four
evaluations, zero training runs. Change `inference_k` to `[2, 8, 16, 32, 256]`
for a larger sweep; `train_k` identifies which existing checkpoint is loaded.
Only FF2048 is supported in this first adapter; ViT needs a separate validated
patch-token adapter, including positional embeddings and token normalization.

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
manifest, per-image resumable metrics, and `segmentation.pdf/png`. Figures show
mIoU versus inference K (bars for a single K), and mIoU versus key storage, with
no published reference curves. Key storage excludes label values and runtime
buffers; search time is not end-to-end latency or a throughput benchmark.

The default uses at most 44 patches per support image: 64,416 total before
all-void exclusions. Dense keys occupy about 503 MiB; K=8 sparse keys about
3.9 MiB (int32 indices + float32 values). Raw spatial caches total about 3.3 GiB
plus labels/masks. Search expands only chunks of compact codes; it does not
allocate a full bank-by-8,192 dense latent matrix or all-query similarity matrix.
Extraction batch size is independent of SAE training batch size. Reduce
`extraction_batch`, `gallery_chunk` or `query_block` for lower GPU peak memory.
Exact search can still be slow: these estimates are storage, not measured speed.

```bash
python -m mpsae.scene plot --results work/scene-v1/reports/IDENTITY/results.json
python -m mpsae.scene publish --repo quangnguyenopt/MPSAE --results work/scene-v1/reports/IDENTITY/results.json
```

Publishing is explicit and uploads result JSON/CSV/plots only. Feature caches stay
local; copy the feature directory and its manifest to reuse it elsewhere. No
large upload or server experiment runs automatically when this module is added.
