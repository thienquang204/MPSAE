# Graduate thesis: three-method ImageNet ablation

This directory now has one experiment setting only: a side-by-side comparison
of **Matryoshka Representation Learning (MRL)**, **Contrastive Sparse
Representation (CSR)**, and **MP-SAE** on **ResNet-18** and **ResNet-50**.
The launcher always executes the complete 3-method × 2-backbone matrix.

## Compared methods

- **Matryoshka:** the selected ResNet is fine-tuned end to end with independent
  classifiers on nested, power-of-two feature prefixes. The objective is the
  weighted sum of the nested cross-entropies (default coefficient `1.0`).
- **CSR:** the ImageNet-pretrained ResNet is frozen. A width-`4d` tied Top-K
  sparse autoencoder is trained with independently weighted Top-K,
  Top-4K, auxiliary, and NCL components.
- **MP-SAE:** starts from exactly the same SAE initialization and frozen feature
  cache as CSR, trains independently weighted Top-K, nested Top-2K/Top-4K,
  auxiliary, and three-marginal partial-matching components, and runs for four
  additional epochs.

All methods are evaluated at the same budgets (`K=8,16,32,64,128,256`) using
the ImageNet train split as the gallery, validation as queries, and exact L2
1-nearest-neighbour search. ResNet-18 and ResNet-50 both use torchvision's
ImageNet-1K V1 weight recipe.

Model and optimizer weights are never written. The run keeps only reusable
pretrained/frozen feature caches plus histories, metrics, plots, tables, logs,
and a portable result ZIP. Because there are no training checkpoints, an
interrupted training arm starts again when the command is rerun.

## Run everything in one command

Accept the ImageNet terms at `ILSVRC/imagenet-1k`, then:

```bash
cp .env.example .env
# Edit .env and replace HF_TOKEN with a read-only Hugging Face token.
bash run_all_experiments_docker.sh
```

Requirements: Docker, an NVIDIA GPU, NVIDIA Container Toolkit, and enough disk
space for ImageNet plus ResNet-18/50 feature caches. The Docker image handles
Python dependencies, CUDA FAISS, dataset preparation, both backbones, all three
training arms, evaluation, plots, tables, and result packaging without further
input.

Outputs are written to:

```text
runs/three_method_ablation/
  resnet18/
    matryoshka/{history.json,results.json}
    csr/{history.json,results.json}
    mpsae/{history.json,results.json}
    summary.json
    comparison.csv
  resnet50/
    ...
  architecture_ablation_per_budget.csv
  architecture_ablation_effect_summary.csv
  architecture_ablation_summary.json
  architecture_ablation_effect.{png,pdf}
  architecture_ablation_results.{md,tex}
  artifact_manifest.json
  imagenet_architecture_ablation_results.zip
  RUN_COMPLETE.json
  logs/
```

The aggregate tables report all three pairwise effects: CSR minus Matryoshka,
MP-SAE minus Matryoshka, and MP-SAE minus CSR.

## Configuration

`.env.example` contains the complete setting. Important controls are:

```bash
ABLATION_EPOCHS=10
MPSAE_EXTRA_EPOCHS=4
ABLATION_BATCH_SIZE=1024
TOPK=8,16,32,64,128,256
TRAIN_K=32
MAX_TRAIN=0
MAX_VAL=0

MRL_CLASSIFICATION_WEIGHT=1.0
CSR_MAIN_RECON_WEIGHT=1.0
CSR_MULTI_TOPK_RECON_WEIGHT=0.125
CSR_AUX_RECON_WEIGHT=0.03125
CSR_CONTRASTIVE_WEIGHT=1.0
MPSAE_MAIN_RECON_WEIGHT=1.0
MPSAE_NESTED_RECON_WEIGHT=0.125
MPSAE_AUX_RECON_WEIGHT=0.03125
MPSAE_MMPOT_WEIGHT=1.3
```

`MPSAE_EXTRA_EPOCHS` is fixed to 4 by the launcher. `MAX_TRAIN=0` and
`MAX_VAL=0` use all ImageNet samples; smaller positive values are useful only
for code development. A CSR or MP-SAE component can be disabled with weight
`0`, provided at least one component for that method remains positive.
`MRL_CLASSIFICATION_WEIGHT` must be positive. Every selected coefficient is
stored in the run manifest, histories, result JSON, and aggregate protocol.
Online W&B monitoring is enabled automatically when `WANDB_API_KEY` is
supplied.

The lower-level `run_csr_vs_mmpot_imagenet.sh` entry point can run the same
study in an already prepared environment, but the Docker launcher above is the
intended portable, unattended interface.
