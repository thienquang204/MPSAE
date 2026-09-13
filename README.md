# Graduate thesis experiment suite

This directory contains three reproducible ImageNet experiment families:

- `csr_vs_mmpot_imagenet.py`: a controlled architecture ablation of Matryoshka versus frozen-backbone MP-SAE. ResNet-18 and ResNet-50 are the default matched pair; Swin-T remains optional.
- `matryoshka_mmpot_experiment.py`: MRL versus the tractable pairwise MMPOT proxy.
- `matryoshka_real_mmpot_experiment.py`: MRL versus the true three-marginal partial-OT objective.

## One-command Docker workflow

Accept the ImageNet terms at `ILSVRC/imagenet-1k`, copy `.env.example` to `.env`, replace the token placeholder, and run one clearly named file:

```bash
cp .env.example .env
bash run_all_experiments_docker.sh
```

That launcher builds one image from the clearly named `Dockerfile.all-experiments`, prepares or reuses the persistent ImageNet volume, runs every experiment selected by `SUITE_EXPERIMENTS`, resumes checkpoints by default, generates plots/tables/loss records, and creates one portable ZIP. It requires Docker, an NVIDIA GPU, NVIDIA Container Toolkit, and sufficient storage for ImageNet and the feature caches.

The final files are under:

```text
runs/all_experiments/
  ALL_EXPERIMENTS_COMPLETE.json
  all_experiments_results.zip
  all_experiments_artifact_manifest.json
  csr_vs_mpsae/
  mmpot_proxy/
  mmpot_true/
  logs/
```

Re-running the same command reuses the Docker layers, validated dataset, feature caches, and checkpoints. Set `PULL_BASE_IMAGE=1` only when a fresh base image is wanted.

## Tracking and plots

Every training runner stores checkpoints, raw JSON/JSONL history, normalized CSV history, result JSON, and clearly named PNG/PDF figures.

For each CSR backbone, the important files are named with the backbone, for example:

```text
csr_resnet50_training_loss_history.csv
csr_resnet50_loss_component_impact.csv
csr_resnet50_loss_component_impact.json
csr_resnet50_training_loss_curves.png
csr_resnet50_training_procedure_overview.png
csr_resnet50_loss_component_impact.png
csr_resnet50_representation_accuracy_comparison.png
```

The CSR history records every Matryoshka head cross-entropy plus MP-SAE main, nested, auxiliary, and MMPOT losses in raw and objective-weighted form. The impact report records observed decrease and objective contribution. “Impact” is deliberately defined as measured contribution to the optimized objective, not as a causal ablation claim.

The cross-backbone architecture ablation additionally creates
`architecture_ablation_per_budget.csv`,
`architecture_ablation_effect_summary.csv`,
`architecture_ablation_summary.json`,
`architecture_ablation_effect.png/.pdf`,
`architecture_ablation_results.md/.tex`, and
`imagenet_architecture_ablation_results.zip`.

The method effect is MP-SAE top-1 minus Matryoshka top-1. Its summary includes
the mean, median, range, standard deviation, positive-budget fraction, relative
error reduction, and ResNet-50 versus ResNet-18 sensitivity. Aggregation first
verifies that both comparison arms and the controlled training/evaluation
settings match across architectures.

The pairwise and true MMPOT runners create similarly explicit `pairwise_mmpot_*` and `true_mmpot_*` reports, including training procedure, loss-component, and benchmark plots.

## Configuration

All normal controls live in `.env`. The defaults run all three experiment families. Useful development overrides include:

```bash
SUITE_EXPERIMENTS=csr_vs_mpsae
CSR_BACKBONES=resnet18,resnet50
CSR_MAX_TRAIN=50000
CSR_MAX_VAL=10000
CSR_EPOCHS=3
MMPOT_MAX_TRAIN_BATCHES=100
MMPOT_MAX_VAL_BATCHES=50
```

Outputs and caches have separate directories, and portable bundles exclude `.pt` checkpoints, model weights, and feature caches.

## Specialized entry points

The single Docker launcher is the recommended interface. These lower-level utilities remain available when only one stage is needed:

- `run_csr_vs_mmpot_imagenet.sh`: matched ResNet-18/ResNet-50 architecture ablation in an already prepared environment.
- `run_imagenet_download.sh`: optional background ImageNet download helper.
- `run_full_pipeline.sh`: compatibility alias for `run_all_experiments_docker.sh`.
- `container_pipeline.sh`: compatibility alias for the container-side all-experiments entry point.

No generated dataset, token, checkpoint, cache, or result artifact is stored in the Docker build context.
