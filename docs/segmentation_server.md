# NYUv2 and Cityscapes on one RTX 3090

Use a Linux server image with a working NVIDIA driver (`nvidia-smi`). The setup
script creates Python 3.11 in `.venv-segmentation`, installs the repository's
pinned dependencies and the CUDA 12.6 PyTorch 2.9.1 / torchvision 0.24.1 wheels,
then runs the segmentation environment check. It does not install GPU drivers.
The standalone CUDA toolkit is not required for these prebuilt wheels.

Push the updated `refactor` branch, including the segmentation cleanup, before
cloning it on the server. On an Ubuntu/Debian server, install git, curl and tmux
if missing (`sudo apt-get install git curl tmux`; omit sudo when root).

```bash
git clone --branch refactor --single-branch https://github.com/thienquang204/MPSAE.git
cd MPSAE
bash scripts/setup_segmentation_server.sh

# Preview both resolved campaigns; no downloads or training:
.venv-segmentation/bin/python scripts/run_segmentation_server.py --plan

# Keep the job alive when SSH disconnects:
tmux new -s segmentation
.venv-segmentation/bin/python -u scripts/run_segmentation_server.py \
  --workspace /workspace/segmentation
```

Choose a persistent, writable SSD directory for `--workspace`; it need not be
`/workspace`. Detach tmux with Ctrl+B then D. Reconnect using
`tmux attach -t segmentation`. After a process/server interruption, rerun the
same Python command with the same workspace. Do not run two launchers against
that workspace concurrently. Completed work and compatible checkpoints resume.

Default: NYUv2 followed by Cityscapes on `cuda:0`, two frozen encoders (SigLIP2
and DINOv2), three SAE methods (CSR/CSRv2/MPSAE), independent training at K=3,12,48,
matched-K evaluation. Each dataset has 18 fits and 20 evaluations including dense.
NYUv2 SigLIP2 MPSAE weight is 0.5; DINOv2 and Cityscapes retain 2.0.
Max 20 epochs, training-loss early stopping after epoch 15; canonical YAMLs own
all numerical settings. Support bank remains 65536 references, with all valid
resized-image patches used for SAE training. This is a memory-limited benchmark.

No Hugging Face output repository, token or publishing flag is used. Public
datasets and encoder weights are downloaded from HF. Cityscapes reads source
Parquets without a second Arrow copy. Feature caches are reused across methods
and K values; epochs stream patches from disk. One process uses the GPU, keeping
both datasets from competing for VRAM. CPU BLAS/OpenMP threads default to four
(or available CPU count if smaller); adjust with `--cpu-threads` if needed.
No automatic batch-size, precision or objective changes are made for the GPU.

## Local output

```text
/workspace/segmentation/
  cache/                          persistent HF/model downloads
  datasets/nyuv2/                  NYU input links and pinned metadata
  datasets/hf_dataset/             downloaded labeled NYU MAT
  nyuv2/                          feature caches, checkpoints, campaign receipts
  cityscapes/                     feature caches, checkpoints, campaign receipts
  logs/                           doctor.log, nyuv2.log, cityscapes.log
  results/<dataset>/<campaign>/   CSV, LaTeX, JSON and per-campaign results.zip
  segmentation-results.zip       combined results for this successful invocation
```

Results are exported after each completed dataset; valid partial tables are also
copied when evaluation/training fails. The combined ZIP is created only after
all requested datasets finish. An old combined ZIP can remain after a failed
rerun: inspect the campaign-specific results/receipt and logs for current status.
The underlying workspace additionally saves incremental tables during evaluation.
ZIPs contain metrics/provenance, not feature caches or model weights. Preserve
the entire workspace on persistent disk to resume training.

Read results directly on the server, for example:

```bash
find /workspace/segmentation/results -name comparisons.csv -print
tail -f /workspace/segmentation/logs/nyuv2.log
```

Download the final ZIP directly over SSH, from your local terminal:

```bash
scp -P YOUR_SSH_PORT USER@SERVER:/workspace/segmentation/segmentation-results.zip .
```

## Select a subset

```bash
# One dataset:
.venv-segmentation/bin/python scripts/run_segmentation_server.py \
  --workspace /workspace/segmentation --datasets cityscapes

# Only SigLIP2, both datasets (18 fits / 20 evaluations total):
.venv-segmentation/bin/python scripts/run_segmentation_server.py \
  --workspace /workspace/segmentation --encoders siglip2
```

Planning allowance: 32-64 GB system RAM and about 100 GB free SSD space, including
environment, inputs, cached features, and checkpoints. These are provisioning
estimates, not measured peak requirements. Full RTX 3090 GPU runtime is unmeasured;
all-patch Cityscapes training is larger than NYUv2 and may be a long-running job.
The launcher does not reduce epochs, data coverage or evaluation points to fit
a time limit. Use `--smoke` for a clearly separated synthetic CPU pipeline check.
