# Native A100 server setup

The default deployment runs Python directly on a Linux A100 server. One entry
point, `bash scripts/setup_server.sh`, prepares the complete environment. It
does not download ImageNet, access Hugging Face credentials or start training.

## Requirements

| Component | Native installation profile |
|---|---|
| OS / architecture | Ubuntu 22.04 or 24.04, x86_64 |
| GPU | NVIDIA A100 40 GB or 80 GB, visible to `nvidia-smi` |
| NVIDIA driver | Already installed by the provider, version >= 560.28.03 |
| Python | 3.10–3.12; Ubuntu's Python is installed if needed |
| Permissions | Root or sudo for system libraries and CUDA Toolkit |
| Storage / RAM | At least 500 GB SSD and 64 GB RAM; 128 GB RAM preferable |

The script uses root directly on provider images that have no `sudo`. For a
regular account it uses `sudo` only for system installation; the Python
environment belongs to the account running the script. Do not run the whole
script with `sudo` from a regular account.

## Install and run

From the checkout, for example `/workspace/MPSAE`:

```bash
cd /workspace/MPSAE
bash scripts/setup_server.sh --dry-run   # inspect the plan, no changes
bash scripts/setup_server.sh
source scripts/activate_server.sh
```

If `.env` did not exist, setup creates it with restrictive permissions. Existing
`.env`, datasets, embeddings and checkpoints are preserved. Fill in `HF_TOKEN`
after accepting the access conditions for `ILSVRC/imagenet-1k`, then:

```bash
python -m mpsae --config configs/thesis.yaml plan
python -m mpsae --config configs/smoke.yaml --smoke run
python -m mpsae --config configs/thesis.yaml run
```

The smoke run uses synthetic data in a separate workspace. Its results are not
scientific. A normal run executes the selected ImageNet and frozen downstream
experiments sequentially. Existing compatible fits and results are reused.

After reconnecting to the server, source the activation script again:

```bash
source scripts/activate_server.sh
python -m mpsae --config configs/thesis.yaml status
python -m mpsae --config configs/thesis.yaml report
```

The activation script sets the `.venv` interpreter, CUDA compiler/library paths
and `TORCH_CUDA_ARCH_LIST=8.0`. It does not select scientific experiments; those
are in `configs/thesis.yaml`. The optional shell launcher loads the same
environment automatically and records a console transcript:

```bash
bash scripts/pipeline.sh
bash scripts/pipeline.sh status
```

## What is installed and checked

The installer uses a fixed A100 profile rather than a moving latest CUDA wheel:

| Package | Version |
|---|---|
| CUDA Toolkit | 12.6, including nvcc and NVRTC |
| torch / torchvision | 2.9.1+cu126 / 0.24.1+cu126 |
| FFCV | 1.0.2 |
| CuPy CUDA 12 wheel | 13.6.0 |
| NumPy / Numba / OpenCV Python | 1.26.4 / 0.61.2 / 4.11.0.86 |

The paired PyTorch builds are listed in the
[official installation commands](https://pytorch.org/get-started/previous-versions/#v291).
[FFCV](https://github.com/libffcv/ffcv#installation) needs libjpeg-turbo,
OpenCV and a C++ compiler. Its GPU normalization imports CuPy, which is not
installed by FFCV itself. [CuPy 13.6 CUDA 12 wheels](https://pypi.org/project/cupy-cuda12x/13.6.0/)
require a CUDA 12 toolkit. The installer supplies that toolkit as well as the
Python packages; `requirements.txt` alone is insufficient.

On a newer NVIDIA driver the installer still uses this CUDA 12.6 profile.
It installs NVIDIA's versioned `cuda-toolkit-12-6` package, which
[does not include a driver](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-installation-guide-linux/index.html#meta-packages).
If the installed driver is too old, setup stops before installing packages;
ask the provider to update the driver. The installer does not replace it.

FFCV is mandatory. A build failure stops setup immediately, and success is
reported only after `pip check` and `python -m mpsae.server` pass. That verifier:

1. Runs PyTorch FP32 and FP16 GPU matrix multiplications.
2. Compiles and executes a CuPy NVRTC kernel.
3. Writes a tiny temporary JPEG FFCV dataset and executes the project's train
   and validation GPU preprocessing, checking labels and normalized pixels.
4. Imports the experiment dependencies and exports headless PNG/PDF figures.
5. Builds and executes the sparse retrieval CUDA extension for A100.

No real data or model weights are downloaded by these checks. Rerun them with:

```bash
source scripts/activate_server.sh
python -m mpsae.server
```

## Installation options and diagnosis

```bash
# Choose a supported Python when creating a new .venv:
bash scripts/setup_server.sh --python /usr/bin/python3.11

# Administrator already supplied CUDA 12.6 and all development libraries:
MPSAE_CUDA_HOME=/usr/local/cuda-12.6 bash scripts/setup_server.sh --skip-system-packages
```

An existing `.venv` must also use Python 3.10–3.12. An incompatible environment
is reported, not deleted. Install in a separate checkout if that environment
must be retained. `--skip-system-packages` still installs all Python packages
and performs every verification; it is not a way to skip FFCV or GPU checks.

Setup writes logs under `.setup-logs/` and the resolved Python versions to
`.venv/server-requirements.freeze.txt` after successful verification. No token
is read or written into these logs. On an installation failure, fix the first
error in the log and rerun the same setup command. Existing results are reused.

If a campaign reports missing `ffcv, cupy`, check the active interpreter:

```bash
source scripts/activate_server.sh
python -c 'import sys, torch, ffcv, cupy; print(sys.executable); print(torch.__version__, torch.version.cuda)'
python -m mpsae.server
python -m mpsae --config configs/thesis.yaml run
```

Empty reports after a preflight failure mean no evaluation was completed;
the final exception identifies the cause. Do not remove the dependency check
or change the source extraction recipe merely to bypass a missing package.

## Workspace and long jobs

Use fast local SSD storage. In `.env`, `MPSAE_WORKSPACE=/workspace/mpsae` moves
the inputs, features, checkpoints, campaign state and reports together;
`HF_HOME=/workspace/hf` moves the Hub cache. The default is `./work`.
See [benchmark.md](benchmark.md) for the exact layout and dataset protocols.

```bash
tmux new -s mpsae
source scripts/activate_server.sh
bash scripts/pipeline.sh
# Ctrl-b d detaches; tmux attach -t mpsae reconnects.
```

Experiments use one configured device (`cuda:0` by default); multiple visible
A100s do not automatically make this a distributed run. To select one GPU,
set `CUDA_VISIBLE_DEVICES` before launching Python. Scientific settings and
checkpoint identities are independent of Docker or the shell launcher.
