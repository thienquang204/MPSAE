# MPSAE: 79-Evaluation Ablation and Segmentation Study

This experiment studies transport mass, transport-loss weight, inference
sparsity, and the semantic information preserved in sparse patch embeddings.
The vision encoders remain frozen throughout.

## Experiment inventory

| Study | SAE fits | Evaluations |
|---|---:|---:|
| ImageNet transport mass | 16 | 48 |
| ImageNet transport-loss weight | 10 | 20 |
| ImageNet dedicated K=2 training | 2 | 4 |
| VOC linear segmentation | 6 patch SAEs + 7 linear heads | 7 |
| **Total** | **34 SAEs + 7 linear heads** | **79** |

If the original 26 ImageNet checkpoints and their evaluations are retained,
this extension adds **53 evaluations, up to 8 SAEs, and 7 linear heads**.
The 79 configurations are evaluation points, not 79 separate SAE fits.

## Sparse representation and MPSAE objective

All SAEs use 8192 latent coordinates and a tied encoder/decoder:

$$
z_K(x)=\operatorname{ReLU}\!\left(\operatorname{TopK}
\left(W(x-b_{\mathrm{pre}})+b_{\mathrm{latent}}\right)\right),
\qquad \hat x_K=W^\top z_K+b_{\mathrm{pre}}.
$$

TopK retains selected coordinates and zeros the others. Thus
$\|z_K(x)\|_0\le K$. K=2 means at most two active entries in an
8192-dimensional vector; different examples can activate different coordinates.
It does not mean a conventional dense two-dimensional embedding.

The MPSAE objective is

$$
\mathcal L=\mathcal L_{\mathrm{rec}}(z_K)
+\tfrac18\mathcal L_{\mathrm{rec}}(z_{4K})
+\tfrac1{32}\mathcal L_{\mathrm{aux}}
+\lambda\,\mathcal G_s(z_K,z_{2K},z_{4K}).
$$

Reconstruction uses elementwise mean squared error. The auxiliary term trains
dead latents to explain reconstruction residuals. The transport matching-gap
regularizer couples the K, 2K and 4K views. Evaluation uses only the code at the
requested inference K, despite these wider views participating in training.

For mass s<1, the existing partial transport solver is used; s=1 uses balanced
log-Sinkhorn. The entropy coefficient is 0.2 and OT group size is 32.

## 1. ImageNet transport-mass study

Train only MPSAE on cached global embeddings from frozen ResNet-50
(MRL fixed-feature BlurPool, 2048 dimensions) and ViT-B/16
(augreg ImageNet checkpoint, 768 dimensions).

For every row below, train at each mass **s = 0.2, 0.4, 0.6, 1.0**:

| Backbone | Training K | Inference K |
|---|---:|---|
| ResNet-50 | 8 | 2, 8, 64 |
| ResNet-50 | 32 | 2, 32, 64 |
| ViT-B/16 | 12 | 2, 12, 96 |
| ViT-B/16 | 48 | 2, 48, 96 |

Fix lambda=2, batch size 1024 and 10 epochs. This gives **16 fits and 48
evaluations**, covering stronger compression, matched sparsity and a larger
inference budget. Raw losses across masses are not directly comparable:
transport mass changes objective scale and entropy contributions too.

## 2. ImageNet transport-weight study

Use ResNet-50 only, s=1, and **lambda = 0.1, 0.5, 1.0, 2.0, 2.5**.

| Training K | Inference K |
|---:|---|
| 8 | 2, 8 |
| 32 | 2, 32 |

Use batch size **4096** and **10 epochs**, matching the existing ResNet-50
CSR/CSRv2 optimizer recipe. This gives **10 fits and 20 evaluations**.
There is no additional batch-1024 weight sweep.

Equal epochs do not imply equal optimizer updates: the full ImageNet training
set yields 12520 updates at batch 1024 versus 3130 at batch 4096 over ten
epochs. Within this weight study, both batch size and update count are fixed.

## 3. Dedicated extreme-sparsity training

Train two additional MPSAE models at **K=2**, with s=1, lambda=2,
batch size 1024 and 10 epochs:

| Backbone | Training K | Inference K |
|---|---:|---|
| ResNet-50 | 2 | 2, 8 |
| ViT-B/16 | 2 | 2, 12 |

These **2 fits and 4 evaluations** distinguish training specifically at K=2
from truncating models trained at larger K. The principal comparison holds
inference K=2, backbone, mass, weight and optimizer settings fixed while
varying training K.

New K=2 results include query/gallery proportions with zero, one and two
nonzeros. These describe realized sparsity, not training convergence.

## Shared ImageNet training and evaluation settings

| Parameter | Value |
|---|---|
| SAE latent width | 8192 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.00004 |
| Adam betas | 0.9, 0.999 |
| Adam epsilon | 6.25e-10 |
| Coupled weight decay | 0.0001 |
| Seed | 1227 |
| OT entropy / group size | 0.2 / 32 |

The 1-NN gallery contains **1281167 ImageNet training images** and queries
contain **50000 validation images**. Gallery and query embeddings use the
same checkpoint and inference K, followed by L2 normalization. Each query
inherits the class of its exact nearest gallery neighbor. Report top-1
accuracy; all-zero queries count as incorrect.

Search is exact and chunked for bounded memory. Encoding preserves the existing
CUDA FP16-compute/FP32-storage convention; search uses FP32 with TF32 disabled.
No approximate search, ImageNet linear probe, timing-repeat loop, or backbone
fine-tuning is introduced in this suite.

## 4. VOC frozen linear-probe segmentation

Use **Pascal VOC 2012**, with 1464 training images, 1449 validation images,
and 21 classes including background. Ignore void pixels.

The frozen encoder is **SigLIP 2 Base, patch size 16**. Images are resized to
448x448 without cropping or augmentation, giving a 28x28 grid of
768-dimensional final post-LayerNorm patch tokens.

The training cache targets **262144 valid patches**. Sampling is uniform
within each image with approximately equal per-image quotas; record the actual
retained count. Each patch target is the class distribution of its non-void
pixels. Purely void patches are excluded from training.

| Representation | Training / inference K |
|---|---|
| Dense SigLIP 2 features | Not applicable |
| CSR | 2, 12 |
| CSRv2 | 2, 12 |
| MPSAE | 2, 12 |

The six patch SAEs use 8192 latents, 10 epochs, batch 1024 and LR 0.00004.
MPSAE fixes s=1 and lambda=2. CSR and CSRv2 retain their existing objectives,
with contrastive coefficient 0.1 and temperature 0.2. CSRv2 uses dominant
patch-class labels and sparsity annealing; its SAE training therefore uses
semantic supervision. This distinction must be disclosed in comparisons.

After SAE fitting, freeze both encoder and SAE and train an affine head:

$$
\ell_i=A\widetilde z_i+b,\qquad
\mathcal L_{\mathrm{probe}}=-\frac1N\sum_i\sum_c
q_{ic}\log\operatorname{softmax}(\ell_i)_c.
$$

Both dense and sparse inputs are L2-normalized. Train the head on **all cached
training patches**, not the 65536-patch nearest-neighbor support bank.

| Probe parameter | Value |
|---|---|
| Epochs | 20, fixed |
| Batch size | 4096 |
| Optimizer | Adam, betas 0.9/0.999, epsilon 1e-8 |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Initialization | Zero weights and bias |
| Seed | 1227 |
| Selection | Final epoch; no validation-based selection |

At evaluation, bilinearly upsample patch logits to the native mask resolution
with align_corners=False, then take pixel-wise argmax. Report **mIoU, pixel
accuracy, per-class IoU, mean nonzeros and empty-patch counts**.

This is supervised linear segmentation, separate from the existing
Hummingbird-style nearest-neighbor protocol. A zero sparse input predicts the
learned classifier bias; its frequency is reported. Dense and sparse heads
have different input widths (768 versus 8192), so parameter counts are saved.

## Setup on a new Linux NVIDIA server

After cloning the repository and pushing these changes to the refactor branch:

```bash
cd /workspace/MPSAE
git switch refactor
git pull --ff-only
bash scripts/setup_ablation79.sh
bash scripts/run_ablation79.sh --plan

mkdir -p /workspace/retrieval-studies
set -o pipefail
bash scripts/run_ablation79.sh \
  --workspace /workspace/retrieval-studies \
  --source-root /workspace/retrieval-studies/source \
  --download-embeddings \
  --device cuda:0 \
  2>&1 | tee -a /workspace/retrieval-studies/console.log
```

Use a server with a working NVIDIA driver. Setup installs Python 3.11,
CUDA 12.8 PyTorch wheels and the required dependencies in `.venv-retrieval`.
It does not install the host driver. Downloads supply cached ImageNet
embeddings, VOC and the SigLIP2 vision encoder as needed; there are no result
uploads to Hugging Face and no raw ImageNet download.

Keep the terminal connected. Repeat the same command to resume. To reuse
previous work, retain or transfer the full experiment workspace containing
checkpoints and evaluation receipts. GitHub and the downloadable results ZIP
alone do not contain the trained checkpoints.

Optional commands:

```bash
# Run one stage:
bash scripts/run_ablation79.sh --workspace /workspace/retrieval-studies --only retrieval
bash scripts/run_ablation79.sh --workspace /workspace/retrieval-studies --only segmentation

# Regenerate reports without training:
bash scripts/run_ablation79.sh --workspace /workspace/retrieval-studies --export-only
```

To reuse compatible existing VOC features/SAEs, supply
`--segmentation-workspace /path/to/existing/voc/workspace` on every invocation.
Reuse requires matching data, encoder revision, extraction recipe, objective
and training schedule. Omit `--download-embeddings` when ImageNet inputs are local.

## Outputs and efficiency

Download **`/workspace/retrieval-studies/ablation79-results.zip`**. It contains:

- 15 figure groups in vector PDF and 300-dpi PNG.
- Four booktabs LaTeX tables and detailed CSV/JSON results.
- Separate training-K and inference-K columns, sparsity diagnostics and status.
- Resolved settings and provenance; incomplete conditions remain blank.

Plots cover matched mass/weight sensitivity, inference sweeps, nonzero counts,
empty queries, interaction heatmaps, K=2 training comparisons, segmentation
accuracy, differences from the measured dense baseline, probe convergence,
and per-class IoU. No published reference curves are inserted.

ImageNet embeddings are mapped once per backbone. Compatible fits and
evaluations are reused independently of the inference grid. Sparse probe codes
are stored as indices and values; the affine head avoids materializing a full
N-by-8192 matrix. Probe data reside on GPU when memory permits, otherwise use
host batches. Encoder extraction, SAE training and probe fitting are separate
stages. Checkpoints and evaluation progress support resumption. Figures are
regenerated only when their data change or files are missing.

## Interpretation and verification

These are single-seed sensitivity experiments, not evidence of significance
across training seeds. Validation labels are not used for checkpoint selection
or early stopping in these configurations. Reporting the best validation
setting does not constitute an independent test result.

Implementation checks: **70 tests passed, 6 skipped**; the complete synthetic
workflow finished **79/79**. Tests cover checkpoint/evaluation reuse, K=2
nonzero counts, sparse/dense linear equivalence, resume behavior and existing
cache/segmentation paths. Actual RTX 5090 speed and memory have not been
measured in the CPU-only development environment.

## Configuration and implementation references

- [Current ImageNet configuration](configs/retrieval_studies.yaml)
- [Original 26-evaluation configuration](configs/retrieval_studies_26.yaml)
- [VOC linear-probe configuration](configs/linear_scene_voc.yaml)
- [Combined runner](mpsae/benchmark/ablation79.py)
- [Linear segmentation implementation](mpsae/patch_scene/linear_probe.py)
- [Additional operational details](docs/ablation79.md)
