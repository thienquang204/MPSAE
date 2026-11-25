# Unified Experiment Settings: Retrieval and Semantic Segmentation

This document consolidates the implemented experiment settings, especially the
two segmentation protocols: **frozen-encoder nearest-neighbor segmentation**
and **frozen-representation linear-probe segmentation**. It also locates them
relative to the current ImageNet ablation suite.

These are separate configurations, not one automatically combined experiment.
The 79-evaluation launcher runs **72 ImageNet evaluations and 7 linear
segmentation evaluations**. The nearest-neighbor segmentation ablations are
launched separately and are not included in that count.

## 1. Which segmentation experiment is which?

| Property | Nearest-neighbor segmentation | Linear-probe segmentation |
|---|---|---|
| Vision encoder | Frozen | Frozen |
| Patch SAE | Trained first, then frozen | Trained first, then frozen |
| Segmentation readout | Annotated support-patch retrieval | Learned affine classifier |
| Readout training | No fitted classifier | Supervised classifier training |
| Readout data | 65536 selected training patches | All retained training patches in the declared cache |
| Query prediction | Temperature-weighted neighbor label distributions | Linear logits |
| Upsampling | Bilinear probabilities, then argmax | Bilinear logits, then argmax |
| Empty sparse code | Abstains and counts as an error on valid pixels | Predicts learned classifier bias; frequency reported |
| Current datasets | VOC, NYUv2, Cityscapes | VOC in the 79-evaluation suite |
| Current encoders | SigLIP2 / DINOv2, selectable | SigLIP2 in the 79-evaluation suite |

Use the term **frozen-encoder** rather than “encoder-free”: both protocols use a
vision encoder. The nearest-neighbor readout requires no classifier fitting,
but the SAE still undergoes training. Neither configuration fine-tunes the
vision backbone.

## 2. Shared patch representations and SAE objectives

The supported encoders are:

- **SigLIP 2 Base/16:** `google/siglip2-base-patch16-256`.
- **DINOv2 Base/14:** `facebook/dinov2-base`.

Both provide 768-dimensional final post-LayerNorm spatial tokens. CLS tokens
are removed where applicable; pooled image features and image-text projections
are not used. Images are resized as specified below without cropping or random
augmentation. Dataset masks use nearest-neighbor resizing for patch targets;
final evaluation uses the original mask resolution.

Each patch target is a distribution over its covered non-void semantic pixels.
Purely void patches are excluded from SAE/probe training and support selection.
CSRv2 receives the dominant class of this distribution, with ties resolved by
the lowest class ID.

All SAEs use 8192 latents and tied encoder/decoder weights:

$$
z_K(x)=\operatorname{ReLU}\left(\operatorname{TopK}
\left(W(x-b_{\mathrm{pre}})+b_{\mathrm{latent}}\right)\right),
\qquad \hat x_K=W^\top z_K+b_{\mathrm{pre}}.
$$

The output remains an 8192-dimensional vector with **at most K nonzeros**.
Coordinate identities vary between examples. Larger K does not guarantee that
all selected preactivations are positive; actual nonzeros are measured.

The common reconstruction objective is

$$
\mathcal L_{\mathrm{base}}=
\mathcal L_{\mathrm{rec}}(z_k)+\tfrac18\mathcal L_{\mathrm{rec}}(z_{4k})
+\tfrac1{32}\mathcal L_{\mathrm{aux}}.
$$

Reconstruction is elementwise MSE. The auxiliary loss is variance-normalized
residual reconstruction using up to 512 auxiliary latents. Method-specific
terms are:

| Method | Additional term | Main sparsity during training |
|---|---|---|
| CSR | 0.1 times instance contrastive loss | Fixed target K |
| CSRv2 | 0.1 times class-based contrastive loss | Annealed k(t), then target K |
| MPSAE | lambda times transport matching gap over z_K, z_2K, z_4K | Fixed target K |

Contrastive temperature is 0.2. CSRv2 anneals from 64 for the K values in these
segmentation experiments, over the first 70% of the planned updates. It uses
semantic labels during SAE training; CSR and MPSAE do not use class targets
in their SAE losses. This distinction must remain explicit in comparisons.

All segmentation SAE recipes use Adam, LR 4e-5, batch 1024, betas (0.9,0.999),
epsilon 6.25e-10, coupled weight decay 1e-4 and seed 1227. Backbone extraction
is cached; patch features are stored as float16 with a quantization check,
while SAE optimization and nearest-neighbor code/search calculations use FP32
with TF32 disabled. Training uses wider sparsity views, but evaluation uses
only the code at the requested inference K.

## 3. Canonical nearest-neighbor dataset settings

These values describe the base dataset YAML files before mass/weight launcher
overrides. Patch grids are listed as SigLIP2 / DINOv2.

| Setting | VOC 2012 | NYUv2 | Cityscapes |
|---|---|---|---|
| Classes | 21, including background | 40 | 19 |
| Training images | 1464 | 795 | 2975 |
| Evaluation images | 1449 validation | 654 official test | 500 validation |
| Input | RGB | RGB only | RGB |
| Resolution, H x W | 448 x 448 | 336 x 448 | 336 x 672 |
| Patch grids | 28x28 / 32x32 | 21x28 / 24x32 | 21x42 / 24x48 |
| SAE training pool | Target 262144 valid patches | All valid training patches | All valid training patches |
| Support bank | Up to 65536 | Up to 65536 | Up to 65536 |
| SAE epochs | 10 | Up to 20 | Up to 20 |
| Training K | 3,12,48 | 3,12,48 | 3,12,48 |
| Base inference grid | 3,12,48 for every fit | Matched training K | Matched training K |
| Base encoder selection | SigLIP2 and DINOv2 | SigLIP2 and DINOv2 | SigLIP2 and DINOv2 |
| Base methods | CSR, CSRv2, MPSAE | CSR, CSRv2, MPSAE | CSR, CSRv2, MPSAE |
| Base SAE fits | 18 | 18 | 18 |
| Base evaluations, including dense | 56 | 20 | 20 |

VOC uses the pinned `shijli/voc2012` segmentation split. NYUv2 uses the official
795/654 split and NYU40 mapping. Cityscapes uses the pinned `Chris1/cityscapes`
mirror and maps original label IDs to the 19 evaluation classes. Void pixels
are ignored on all datasets.

VOC's training pool uses approximately equal per-image quotas, with uniform
sampling without replacement among each image's valid patches. The requested
262144 is a budget; actual retained counts are recorded. NYUv2 and Cityscapes
stream all valid training patches once per epoch, shuffling shards and patches.

NYUv2/Cityscapes early stopping monitors sample-weighted epoch training loss:
minimum epoch 15, relative improvement threshold 0.001, patience 3. Monitoring
also requires a full epoch at target K. Use the last completed epoch when the
plateau criterion triggers, or epoch 20. No evaluation masks select the epoch.

Without sweep overrides, MPSAE mass defaults to **s=0.9**. VOC and Cityscapes
default to **lambda=2**. NYUv2 has a SigLIP2-specific **lambda=0.5**, while
DINOv2 retains lambda=2. Explicit weight sweeps override these coefficients.

## 4. Support-bank selection and nearest-neighbor decoding

The support bank is selected from training data only and shared between dense
and sparse representations for compatible dataset/encoder settings. VOC selects
from its retained patch pool; full-patch datasets select from all valid
training patches. Evaluation patches never enter the support bank.

The implemented heuristic is a randomized class-frequency score, not direct
sampling with probabilities proportional to inverse class frequency. Let
I_ic indicate whether class c occurs in patch i, and f_c=sum_i I_ic. Draw
u_i uniformly in [0,1), calculate

$$
a_i=u_i\sum_c I_{ic}f_c,
$$

and retain the patches with the smallest scores. This favors patches containing
less frequent classes. Stable ordering resolves score ties. Full-patch
datasets compute class frequencies globally and maintain the best scores
incrementally to bound memory use.

For each query patch, L2-normalize its representation and the support vectors,
retrieve the **30 highest cosine-similarity neighbors**, and predict

$$
\hat y_i=\sum_{j\in\mathcal N_i}
\frac{\exp(\operatorname{cos}(z_i,m_j)/0.02)}
{\sum_{j'\in\mathcal N_i}\exp(\operatorname{cos}(z_i,m_{j'})/0.02)}y_j.
$$

Search is exact over the declared support bank. Similarity ties use support
position in ascending order. Bilinearly upsample class probabilities to the
original mask resolution, then take argmax. An all-zero query code abstains;
its mapped valid pixels count as errors rather than being discarded.

Report mIoU, pixel accuracy, per-class IoU, mean query/gallery nonzeros and empty
patch counts. This is a **budgeted Hummingbird-style nearest-neighbor protocol**,
not a claim to reproduce a published result using a different encoder, feature
resolution, support size, or sampling policy.

## 5. Nearest-neighbor segmentation ablation variants

### A. Mass-only VOC sweep

The dedicated Kaggle mass launcher selects VOC, SigLIP2 and MPSAE only, with
no dense reference. It uses:

- Masses: **0.5,0.6,0.7,0.8,0.9,0.95**.
- Fixed lambda: **2.0** unless explicitly overridden.
- Training K: **3,12,48**.
- Inference K: **3,12,48 for every checkpoint**.
- Total: **18 fits and 54 evaluations**.

An explicit mass of 1.0 is supported and uses the balanced solver. Adding it to
the six listed masses gives 21 fits and 63 evaluations. This cross-K mass-only
protocol differs from the matched-K weight grids below.

### B. Weight sweep at one mass

For VOC/SigLIP2, CSR/CSRv2/MPSAE, training K=3,12,48, dense included,
**s=1** and **lambda=0.5,1.0,2.0**:

- Train each MPSAE weight/K combination independently: 9 fits.
- Train CSR and CSRv2 once per K: 6 shared baseline fits.
- Evaluate each sparse checkpoint only at its training K: 15 evaluations.
- Evaluate the dense reference once: **16 evaluations total**.

CSR and CSRv2 have no transport-loss weight. Their fits and dense references
are reused across weights rather than counted again. Selecting both encoders
doubles this example to 30 SAE fits and 32 evaluations.

### C. Joint mass and weight grid on the server

The standalone segmentation server launcher's defaults are:

| Parameter | Default |
|---|---|
| Dataset / encoder | VOC / SigLIP2 |
| Methods | CSR, CSRv2, MPSAE; dense included |
| Training K | 3,12,48 |
| Inference | Matched K only |
| Masses | 0.9,0.95,1.0 |
| Weights | 0.5,1.0,2.0 |

This yields **27 MPSAE fits + 6 baseline fits = 33 SAE fits**, and **34 unique
evaluations** including one dense reference. Methods, encoders, dataset,
masses, weights and K are configurable. The matched weight grid supports K<=64.

For MPSAE, s<1 uses partial transport and s=1 uses balanced log-Sinkhorn with
uniform equality marginals. Entropy stays 0.2 and OT group size stays 32. No
mass-dependent rescaling is added. A different mass or weight requires its own
fit; a previous checkpoint is reused only when its complete recipe matches.
Raw losses across masses should not be interpreted as a common metric.

Every grid must be reported with its resolved settings. The base VOC YAML,
mass-only launcher and joint grid intentionally have different evaluation
counts; they must not be merged as if they were one protocol.

## 6. Linear-probe VOC study in the 79-evaluation suite

This study fixes **VOC and SigLIP2**, retains the 448x448 extraction and
262144-patch training budget, and compares:

- Dense features: one reference.
- CSR, CSRv2 and MPSAE, independently trained at **K=2 and K=12**.
- Every sparse checkpoint is evaluated at its own training K.

There are **6 patch SAEs and 7 linear heads/evaluations**. SAE training uses
10 epochs and the shared optimizer recipe above. MPSAE fixes **s=1, lambda=2**.
No mass or weight sweep is performed in this linear-probe configuration.

After fitting each SAE, freeze it. Train an affine head on L2-normalized dense
features or sparse codes using all retained training-pool patches:

$$
\ell_i=A\widetilde z_i+b,\qquad
\mathcal L_{\mathrm{probe}}=-\frac1N\sum_i\sum_c
q_{ic}\log\operatorname{softmax}(\ell_i)_c.
$$

| Probe parameter | Setting |
|---|---|
| Epochs | 20 fixed |
| Batch size | 4096 |
| Optimizer | Adam, betas 0.9/0.999, epsilon 1e-8 |
| Learning rate / weight decay | 0.001 / 0 |
| Initialization | Zero head weights and bias |
| Seed | 1227 |
| Class reweighting | None |
| Model selection | Final epoch; no validation selection |

Only the head receives gradients. Sparse logits are computed directly from
coordinate/value pairs, avoiding an N-by-8192 dense matrix. The 65536 support
bank setting is not used as the probe training set.

Bilinearly upsample logits with align_corners=False to native mask resolution,
then take argmax. Ignore void pixels. Empty sparse inputs use the learned bias,
as in a standard affine classifier; their counts remain visible. Report mIoU,
pixel accuracy, per-class IoU, mean nonzeros and probe training convergence.
Dense and sparse heads have different input widths (768/8192); parameter counts
are recorded. This readout is supervised and distinct from nearest-neighbor
segmentation.

## 7. Relationship to the ImageNet ablations

The combined `ablation79` launcher includes the linear segmentation study above
and these MPSAE-only ImageNet studies on frozen cached global embeddings:

| Study | Training settings | Inference settings | Evaluations |
|---|---|---|---:|
| Mass | ResNet-50 K=8,32; ViT-B/16 K=12,48; s=.2,.4,.6,1; lambda=2; batch 1024 | K=2, matched K, and 64 for ResNet / 96 for ViT | 48 |
| Weight | ResNet-50 K=8,32; s=1; lambda=.1,.5,1,2,2.5; batch 4096 | K=2 and matched K | 20 |
| Extreme sparsity | Both backbones train K=2; s=1; lambda=2; batch 1024 | ResNet: 2,8; ViT: 2,12 | 4 |
| Linear segmentation | VOC/SigLIP2, dense plus 3 methods at K=2,12 | Matched K | 7 |
| **Total** | | | **79** |

ImageNet fits use 10 epochs, Adam LR 4e-5 and seed 1227. Exact 1-NN evaluation
uses 1281167 training gallery images and 50000 validation queries, normalized
sparse codes, and empty queries scored as misses. Equal epochs at different
batch sizes do not imply equal optimizer-update counts. The current suite has
28 ImageNet SAEs, 6 patch SAEs and 7 heads; old compatible fits can be reused.

## 8. Entry points and local outputs

### Nearest-neighbor segmentation

- Base configs: [VOC](configs/patch_scene_voc.yaml),
  [NYUv2](configs/patch_scene_nyuv2.yaml),
  [Cityscapes](configs/patch_scene_cityscapes.yaml).
- Mass-only Kaggle cell: [kaggle_voc_mass_cell.py](scripts/kaggle_voc_mass_cell.py).
- Weight-grid Kaggle cell: [kaggle_segmentation_weights_cell.py](scripts/kaggle_segmentation_weights_cell.py).
- Joint server launcher: [segmentation_server_onefile.py](scripts/segmentation_server_onefile.py).

In an existing environment, preview a matched-K weight grid:

```bash
python -m mpsae.patch_scene.weight_sweep \
  --config configs/patch_scene_voc.yaml --workspace work/voc \
  --encoders siglip2 --methods csr csrv2 mpsae \
  --train-k 3 12 48 --mass 1 --weights 0.5 1 2 --devices cuda:0 --plan
```

Remove `--plan` to run. The mass-only entry point is
`python -m mpsae.patch_scene.mass_sweep --plan`. For a joint grid using an
already installed environment:

```bash
python scripts/segmentation_server_onefile.py \
  --skip-setup --repo . --workspace /workspace/segmentation \
  --dataset voc --encoders siglip2 --methods csr csrv2 mpsae \
  --train-k 3 12 48 --masses 0.9 0.95 1 --weights 0.5 1 2
```

The joint server archive is `<workspace>/segmentation-results.zip`. Mass-only
and weight-only runs provide combined sweep archives under their `ablations/`
directories. The ordinary `patch-scene-results.zip` represents a campaign and
must not be mistaken for the archive of every mass/weight condition.

### Linear segmentation / complete 79-evaluation suite

- Configuration: [linear_scene_voc.yaml](configs/linear_scene_voc.yaml).
- Full setup: `bash scripts/setup_ablation79.sh`.
- Linear segmentation only:

```bash
bash scripts/run_ablation79.sh \
  --workspace /workspace/retrieval-studies --only segmentation
```

Omit `--only segmentation` to include ImageNet. The combined archive is
`<workspace>/ablation79-results.zip`; the linear stage also produces
`linear-segmentation-results.zip` in its own workspace. All result storage is
local in these launch paths; artifact uploads are not required.

See [the 79-evaluation README](README_ABLATION79.md),
[Kaggle segmentation guide](docs/patch_scene_kaggle.md), and
[joint-grid server guide](docs/segmentation_server_onefile.md) for setup and
resumption details. Two-GPU segmentation scheduling runs independent jobs,
not distributed data parallel training; it does not double the configured batch.

## 9. Reporting and scientific interpretation

- Always report dataset, encoder checkpoint, training K, inference K, method,
  s, lambda, seed, training-pool size, and readout type.
- Nearest-neighbor results additionally require the support-bank size,
  selection rule, neighbor count and temperature.
- A dense reference is measured using the same encoder, data and readout;
  published reference scores are not substituted.
- All current defaults use one seed. Do not claim across-seed significance or
  invent uncertainty estimates from these runs.
- Preserve every declared mass/weight result. In particular, do not choose
  hyperparameters on NYUv2's official test set and call the selected score an
  independently evaluated result.
- Pending or failed conditions are blank, not zero. Reuse requires exact
  compatible identities; changing inference alone does not change a real fit.
- Retain full workspaces to resume: downloadable result archives exclude
  checkpoints, images and feature caches.
- Numeric settings describe configured experiments, not proof that a particular
  server run completed them. Use that run's plan, provenance and completion
  receipts when preparing the paper.

This file consolidates the current settings without changing any configuration
or launching additional experiments.
