# Local 79-evaluation extension

Run the saved ImageNet models over a small inference grid, add two K=2 fits,
and evaluate a supervised linear segmentation readout on VOC/SigLIP2. The
encoder and SAE stay frozen during probe fitting. There are no remote uploads.

## Exact experiment inventory

| Study | Fits | Evaluation grid | Evaluations |
|---|---:|---|---:|
| Mass: ResNet-50 train K=8,32 and ViT-B/16 train K=12,48; s=.2,.4,.6,1 | 16 existing | K=2, matched K, and K=64 (ResNet) or 96 (ViT) | 48 |
| Weight: ResNet-50 train K=8,32; lambda=.1,.5,1,2,2.5 | 10 existing | K=2 and matched K | 20 |
| Extreme sparsity: ResNet-50 and ViT-B/16 train K=2 | 2 new | ResNet: 2,8; ViT: 2,12 | 4 |
| VOC linear probe: dense + CSR/CSRv2/MPSAE at matched K=2,12 | Up to 6 patch SAEs + 7 small heads | Seven representations | 7 |
| Total | 34 SAEs including 26 existing; 7 heads | | **79** |

Assuming the original 26 evaluations and checkpoints are retained, this is
**53 additional evaluations, up to 8 new SAEs, and 7 linear heads**. Existing
training and matched evaluation identities are unchanged. Inference K belongs
to an evaluation identity and never changes a scientific fit's identity.

`configs/retrieval_studies_26.yaml` preserves the original configuration.
`configs/retrieval_studies.yaml` now declares the 72 ImageNet evaluations;
`configs/linear_scene_voc.yaml` declares the seven segmentation evaluations.
The runner reports actual counts when either configuration is edited.

## Setup and run

From the cloned refactor repository on a Linux NVIDIA server:

```bash
git pull --ff-only
bash scripts/setup_ablation79.sh
bash scripts/run_ablation79.sh --plan
bash scripts/run_ablation79.sh \
  --workspace /workspace/retrieval-studies \
  --source-root /workspace/retrieval-studies/source \
  --download-embeddings
```

**Use the same retrieval workspace as the completed 26-run experiment.** The
results ZIP contains no model weights; keep `fits/`, `evaluations/` and the
source embeddings to reuse previous work. The download flag fetches missing
cached ImageNet embeddings; it never builds ImageNet features or uploads data.
Omit it if the embeddings are already local. VOC and the SigLIP2 vision tower
are downloaded only if their local caches are missing. No raw ImageNet is used.

If an existing VOC patch workspace contains compatible features or SAEs, add:

```bash
--segmentation-workspace /workspace/segmentation/voc
```

Reuse is exact: encoder revision, extraction parameters, package versions,
data, objective and schedule must match. Changing these may require extracting
or fitting again. A K=12 patch SAE with a different mass/weight cannot be reused
as the s=1, lambda=2 model. CSR/CSRv2 recipes remain unaffected by MPSAE weights.

The setup uses Python 3.11 in `.venv-retrieval`, torch 2.9.1 and torchvision
0.24.1 CUDA 12.8 wheels, and the pinned patch-scene dependencies. A compatible
host driver is required; the script does not install a driver or CUDA compiler.
Re-running setup uses the existing environment and installed packages.

```bash
# Run stages independently (same workspace, no duplicated fits):
bash scripts/run_ablation79.sh --workspace /workspace/retrieval-studies --only retrieval
bash scripts/run_ablation79.sh --workspace /workspace/retrieval-studies --only segmentation
# Rebuild tables/plots/archive from saved receipts without training or downloads:
bash scripts/run_ablation79.sh --workspace /workspace/retrieval-studies --export-only
```

No tmux or background service is used. Keep the terminal connected. Repeating
the same command resumes local work. Retrieval preserves existing numerical
precision and exact full-gallery search. Query/gallery block defaults match the
original runs (2048/32768); changing them conservatively creates new evaluation
identities. Use `--query-block`/`--gallery-chunk` if lower peak memory is needed.

## Training recipes

ImageNet: tied SAE with 8192 latents, seed 1227, 10 epochs, Adam LR 4e-5,
betas .9/.999, epsilon 6.25e-10 and coupled weight decay 1e-4. Mass and K=2
studies use batch 1024, lambda=2. The K=2 study fixes s=1. The weight study
uses batch 4096 and s=1. OT entropy .2 and group size 32 remain fixed. s<1
uses the existing partial solver; s=1 uses balanced log-Sinkhorn.

VOC: frozen SigLIP2 Base, 448x448, final post-LayerNorm patch tokens, no image
projection. Use the existing per-image uniform valid-patch sampling with a
262144-patch target. The actual cached row count is recorded. Train CSR,
CSRv2 and MPSAE at K=2 and K=12 with the existing VOC 10-epoch, batch-1024,
LR-4e-5 recipe. MPSAE alone fixes s=1, lambda=2. CSRv2 still uses dominant
semantic-class targets and its existing K annealing; label use is not erased
by adding a common probe. This is a downstream patch-SAE study, not a claim
of label-free transfer for every method.

## Linear segmentation protocol

This is an additional supervised affine readout, **not** Hummingbird nearest
neighbor decoding and not a reproduction of another paper's tuned probe.
Use the same train-patch rows and soft nonvoid class distributions for every
head, with both dense and sparse input vectors L2-normalized (epsilon 1e-12).
The SAE and backbone never receive probe gradients. Sparse heads use
`b + sum(values * W[indices])`, mathematically identical to a linear map on
the full sparse vector without materializing an N-by-8192 matrix.

Heads use zero-initialized weights/bias, Adam (betas .9/.999, epsilon 1e-8),
LR .001, zero weight decay, batch 4096, 20 fixed epochs, seed 1227 and the
same epoch permutations. Loss is mean soft-target patch cross-entropy, no
class reweighting. The last completed epoch is used. No held-out masks are
used for fitting, early stopping, or choosing a hyperparameter. Validation
logits are bilinearly upsampled with align_corners=False, then argmaxed at
native mask resolution; void pixels are excluded from the confusion matrix.

Dense and sparse classifiers have different input widths (768 versus 8192),
so record their parameter counts. Zero sparse inputs use the learned bias,
the standard affine-classifier behavior; their frequency is reported. This
differs deliberately from 1-NN's empty-query-as-miss rule. K is an upper
bound, not a promise of exactly K positive entries. New K=2 retrieval outputs
include query/gallery 0/1/2-nonzero histograms. Historical cached results that
lack these optional diagnostics remain valid; missing diagnostics stay blank.

## Efficiency and outputs

- One ImageNet feature bundle is mapped per backbone across all studies.
- Existing fits and evaluation receipts are reused independently of sweep IDs.
- Retrieval stays exact over all 1281167 gallery and 50000 validation images;
  no extra linear probes, timing rounds, or validation subsets are added there.
- The segmentation encoder is released after extraction. Cache one compact
  normalized representation at a time; keep the probe bank on GPU when at
  least 2 GiB headroom remains, otherwise use host batches. Head checkpoints
  are atomic per epoch; evaluation progress is atomic per image shard.
- PDF/300-dpi PNG figures are regenerated only when their data change (or files
  are missing), and the two main stages run in separate processes to release
  GPU memory fully between them.

Download `<workspace>/ablation79-results.zip`. It contains `retrieval/` and
`segmentation/`, the resolved plan and completion counts. It is also produced
after a stage fails, with incomplete stages clearly marked. Per-stage archives
remain available. No checkpoints, images or large feature caches are bundled.

Retrieval figures: matched mass/weight curves, inference curves, measured nnz,
empty-query counts, mass/weight heatmaps, and a fixed-inference-K=2 comparison
across training K at identical s/lambda/batch settings. Every curve has a fixed
training recipe; unrelated K values are never averaged together. CSV and
booktabs LaTeX tables record both training and inference K.

Segmentation figures: mIoU/pixel accuracy/nnz against K, differences from the
measured dense reference, probe convergence, and per-class IoU. This dense
reference is measured in the same run, not copied from a publication. Results
are percentages in tables/CSV and fractions in detailed JSON.

These are single-seed sensitivity experiments: no significance claim, no
fabricated error bars, and no selecting the best validation condition as an
independent test result. Pending/failed conditions are blank, never zero.
CPU smoke tests verify the plumbing; throughput and peak memory require a
real server run. Example: `.venv-retrieval/bin/python -m
mpsae.benchmark.ablation79 --workspace /tmp/ablation79-smoke --smoke --device cpu`.

Implementation verification (2026-11-24): 70 focused/regression tests passed,
6 were skipped; the complete default synthetic workflow finished 79/79 and
exported 15 PDF/PNG figure groups and four LaTeX tables. Checks cover reuse of
the original training recipes and matched evaluations, real K=2 nonzero counts,
sparse/dense linear-logit and gradient equivalence, deterministic epoch resume,
and existing retrieval cache/segmentation behavior. Bash scripts pass syntax
validation. Fresh Linux installation and actual RTX 5090 performance have not
been exercised from the CPU-only development environment.
