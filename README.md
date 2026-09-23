# Graduate thesis: five-arm ImageNet ablation

This directory contains one experiment setting: a side-by-side comparison of
**Matryoshka Representation Learning (MRL)**, **CSR v1**, **MPSAE v1**,
**CSRv2**, and **MPSAEv2** on **ResNet-18** and **ResNet-50**. The launcher
always executes the complete 5-arm by 2-backbone matrix.

## Compared methods

- **Matryoshka:** the selected ResNet is fine-tuned end to end with independent
  classifiers on nested, power-of-two feature prefixes. The objective is the
  weighted sum of the nested cross-entropies (default coefficient `1.0`).
- **CSR v1:** uses the frozen pretrained ResNet features and the CSR sparse
  autoencoder objective with fixed training support `K=32` by default.
- **MPSAE v1:** uses the matching frozen features, initialization, and fixed
  training support `K=32` with the MPSAE objective.
- **CSRv2:** the ImageNet-pretrained ResNet is frozen. A width-`4d` tied sparse
  autoencoder is trained with independently weighted Top-K, Top-4K, auxiliary,
  and NCL components. Its base K follows a half-cosine curriculum from 64 to 2
  over the first 70% of optimization steps and then remains at 2.
- **MPSAEv2:** starts from exactly the same SAE initialization and frozen
  feature cache as CSRv2. It trains independently weighted Top-K, nested
  Top-2K/Top-4K, auxiliary, and three-marginal partial-matching components,
  applies the same half-cosine curriculum to the base K of all three views,
  and runs for four additional epochs.

All five arms are evaluated at `K=8,16,32,64,128,256,512` for ResNet-18 and
at `K=8,16,32,64,128,256,512,1024,2048` for ResNet-50. All four CSR/MPSAE
sparse arms are also evaluated at `K=1,2,4`. Gallery and query embeddings are
unit normalized before exact L2 1-nearest-neighbour search. Matryoshka uses
dense CPU FAISS search; the four sparse arms use exact CPU SciPy CSR products.
Search-only mean latency per query is recorded and plotted separately for each
backbone. Both backbones use torchvision's ImageNet-1K V1 weight recipe.

This controlled experiment isolates progressive K annealing while keeping the
existing frozen-backbone sparse objectives matched. It does not claim to
reproduce the CSRv2 paper's separate supervised-contrastive or full-backbone
fine-tuning variants. Here, `MPSAEv2` specifically means MPSAE with the
CSRv2-style annealing curriculum.

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
Python dependencies, FAISS, dataset preparation, both backbones, all five
training arms, evaluation, plots, tables, and result packaging without further
input.

Outputs are written to:

```text
runs/three_method_ablation/
  resnet18/
    matryoshka/{history.json,results.json}
    csr/{history.json,results.json}
    mpsae/{history.json,results.json}
    csrv2/{history.json,results.json}
    mpsaev2/{history.json,results.json}
    summary.json
    comparison.csv
    ablation_resnet18_representation_accuracy_comparison.{png,pdf}
    ablation_resnet18_retrieval_time_comparison.{png,pdf}
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

The aggregate tables report all three pairwise effects: CSRv2 minus Matryoshka,
MPSAEv2 minus Matryoshka, and MPSAEv2 minus CSRv2.

## Configuration

`.env.example` contains the complete setting. Important controls are:

```bash
ABLATION_EPOCHS=10
MPSAEV2_EXTRA_EPOCHS=4
ABLATION_BATCH_SIZE=1024
SPARSE_EXTRA_TOPK=1,2,4
SPARSE_KNN_QUERY_BATCH=32
TRAIN_K=2
V1_TRAIN_K=32
ANNEAL_START_K=64
ANNEAL_FRACTION=0.7
MAX_TRAIN=0
MAX_VAL=0

MRL_CLASSIFICATION_WEIGHT=1.0
CSRV2_MAIN_RECON_WEIGHT=1.0
CSRV2_MULTI_TOPK_RECON_WEIGHT=0.125
CSRV2_AUX_RECON_WEIGHT=0.03125
CSRV2_CONTRASTIVE_WEIGHT=1.0
MPSAEV2_MAIN_RECON_WEIGHT=1.0
MPSAEV2_NESTED_RECON_WEIGHT=0.125
MPSAEV2_AUX_RECON_WEIGHT=0.03125
MPSAEV2_MMPOT_WEIGHT=1.3
```

`V1_TRAIN_K` is the fixed training support for both v1 arms. `TRAIN_K` is the
annealing target for both v2 arms. `MPSAEV2_EXTRA_EPOCHS` is fixed to 4 for
both MPSAE arms. `MAX_TRAIN=0` and
`MAX_VAL=0` use all ImageNet samples; smaller positive values are useful only
for code development. `ANNEAL_START_K` must be at least `TRAIN_K`, and
`ANNEAL_FRACTION` must be in `(0,1]`. The v1/v2 variants share their family
loss coefficients. A component can be disabled with weight `0`, provided at
least one component in that family remains positive.
`MRL_CLASSIFICATION_WEIGHT` must be positive. Every selected
coefficient and annealing parameter is stored in the run manifest, histories,
result JSON, and aggregate protocol. Online W&B monitoring is enabled
automatically when `WANDB_API_KEY` is supplied.

The lower-level `run_csr_vs_mmpot_imagenet.sh` entry point can run the same
study in an already prepared environment, but the Docker launcher above is the
intended portable, unattended interface. The legacy filename is retained for
command compatibility; its active sparse methods are CSR v1, MPSAE v1, CSRv2,
and MPSAEv2.
