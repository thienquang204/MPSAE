# Native NVIDIA GPU server setup

The default deployment runs Python directly on a Linux NVIDIA GPU server. One entry
point, `bash all_in_one.sh`, installs the environment, runs the campaign
and saves artifacts to private HF `<account>/MPSAE` and W&B project `cvpr`.
For installation alone use `bash setup_server.sh`; that installer does
not download ImageNet, access Hugging Face credentials or start training.

## Requirements

| Component | Native installation profile |
|---|---|
| OS / architecture | Ubuntu 22.04 or 24.04, x86_64 or ARM64/SBSA |
| GPU | A100, RTX A6000, RTX 4090/5090, H100/H200, GB300; visible to `nvidia-smi` |
| NVIDIA driver | Already installed by the provider; see the selected profile below |
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
cp .env.example .env                    # only when creating a new .env
# Fill HF_TOKEN and WANDB_TOKEN in .env.
bash all_in_one.sh
```

Only tokens are needed. HF_TOKEN must have ImageNet read access plus write
access to create/upload the private dataset repository `<account>/MPSAE`.
`WANDB_TOKEN` (or `WANDB_API_KEY`) enables W&B; its project defaults to `cvpr`
and the SDK chooses the account's default entity. Without a W&B key the run
still backs up to HF. No repository/project names need to be entered.

The all-in-one command checks remote credentials before downloading datasets.
It uploads feature archives and source-fit artifacts after expensive steps,
then saves reports, evaluations and campaign state on completion or failure.
It excludes raw images, regenerable caches, env files and SDK internal files.
It never deletes old remote artifacts. Deleting the HF repo does not leave a
stale local uploaded flag: the next run creates it again and uploads the local
artifacts. If local artifacts were also deleted, they must be recomputed.

```bash
bash all_in_one.sh --skip-setup   # installed environment, resume
bash all_in_one.sh --sync-only   # retry backup, no training
bash all_in_one.sh --check       # check/create HF repo and W&B project
bash all_in_one.sh --smoke       # synthetic run, no remote writes
```

Cloud status is stored at `work/campaigns/<name>/<plan_id>/services/status.json`.
Tokens are not included in this file. W&B gets live per-cell training curves,
retrieval measurements, a result table and the final report artifact. Failures
to upload do not discard local results; an incomplete final sync returns an
error so it can be retried with `--sync-only`.

For a local run without cloud synchronization, the original commands below
remain available. To install and verify only:

```bash
bash setup_server.sh --dry-run
bash setup_server.sh
source activate_server.sh
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
source activate_server.sh
python -m mpsae --config configs/thesis.yaml status
python -m mpsae --config configs/thesis.yaml report
```

The activation script sets the `.venv` interpreter, CUDA compiler/library paths
and `TORCH_CUDA_ARCH_LIST` from the detected compute capabilities. It does not select scientific experiments; those
are in `configs/thesis.yaml`. The optional shell launcher loads the same
environment automatically and records a console transcript:

```bash
bash pipeline.sh
bash pipeline.sh status
```

## What is installed and checked

The installer selects a tested toolkit by compute capability, not by matching
GPU names or following a moving latest wheel. Newer drivers keep the same
default profile, and mixed-GPU machines use the newest profile needed by any
selected GPU. The supported pairings are:

| GPU | Compute capability | CUDA profile | Conservative driver floor |
|---|---|---|---|
| A100 | 8.0 | 12.6 / cu126 | 560.35.05 |
| RTX A6000 | 8.6 | 12.6 / cu126 | 560.35.05 |
| RTX 4090 | 8.9 | 12.6 / cu126 | 560.35.05 |
| H100 / H200 | 9.0 | 12.6 / cu126 | 560.35.05 |
| RTX 5090 | 12.0 | 12.8 / cu128 | 570.124.06 |
| GB300 / B300 | 10.3 | 12.9 / cu129 | 575.57.08 |

Capabilities follow [NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus).
CUDA 12.9 adds the compiler target for GB300's sm_103; CUDA 12.8 is insufficient
for compiling that target. Driver floors cover toolkit updates installed by
the versioned apt packages; see [NVIDIA's release table](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cuda-driver).

All three profiles share the following Python versions:

| Package | Version |
|---|---|
| torch / torchvision | 2.9.1 / 0.24.1, from the selected CUDA index |
| FFCV | 1.0.2 |
| CuPy CUDA 12 wheel | 13.6.0 |
| NumPy / Numba / OpenCV Python | 1.26.4 / 0.61.2 / 4.11.0.86 |

The builds are available from the official
[cu126](https://download.pytorch.org/whl/cu126/torch/),
[cu128](https://download.pytorch.org/whl/cu128/torch/) and
[cu129](https://download.pytorch.org/whl/cu129/torch/) indexes. ARM64 torchvision
wheels for this version have no `+cuXXX` suffix; the installer handles that
and uses NVIDIA's `sbsa` repository for the native ARM64 toolkit.
[FFCV](https://github.com/libffcv/ffcv#installation) needs libjpeg-turbo,
OpenCV and a C++ compiler. Its GPU normalization imports CuPy, which is not
installed by FFCV itself. [CuPy 13.6 CUDA 12 wheels](https://pypi.org/project/cupy-cuda12x/13.6.0/)
require a CUDA 12 toolkit. The installer supplies that toolkit as well as the
Python packages; `requirements.txt` alone is insufficient.

The installer uses NVIDIA's versioned `cuda-toolkit-12-6`, `cuda-toolkit-12-8`
or `cuda-toolkit-12-9` package, which
[does not include a driver](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-installation-guide-linux/index.html#meta-packages).
If the installed driver is too old, setup stops before installing packages;
ask the provider to update the driver. The installer does not replace it.

FFCV is mandatory. A build failure stops setup immediately, and success is
reported only after `pip check` and `python -m mpsae.server` pass. That verifier:

1. Checks runtime compatibility and PyTorch FP32/FP16 operations on every visible GPU.
2. Compiles and executes a CuPy NVRTC kernel on every visible GPU.
3. Writes a tiny temporary JPEG FFCV dataset and executes the project's train
   and validation GPU preprocessing, checking labels and normalized pixels.
4. Imports the experiment dependencies and exports headless PNG/PDF figures.
5. Builds the sparse retrieval CUDA extension for the selected architectures
   and executes its test on the first visible GPU.

No real data or model weights are downloaded by these checks. Rerun them with:

```bash
source activate_server.sh
python -m mpsae.server
```

## Installation options and diagnosis

```bash
# Choose a supported Python when creating a new .venv:
bash setup_server.sh --python /usr/bin/python3.11

# Preview automatic selection, or explicitly choose a newer compatible profile:
bash setup_server.sh --dry-run
bash setup_server.sh --cuda 12.9

# Select the physical GPU before installation and launching the run:
CUDA_VISIBLE_DEVICES=0 bash all_in_one.sh

# Administrator already supplied the selected toolkit and development libraries:
MPSAE_CUDA_HOME=/usr/local/cuda-12.9 bash setup_server.sh --cuda 12.9 --skip-system-packages
```

`--cuda auto` is the default. An explicitly selected toolkit older than the
GPU needs is rejected before any package installation. All-in-one forwards
`--cuda`, `--python` and `--skip-system-packages` to the installer. The server's
driver is never installed or downgraded by this script. Jetson is not SBSA and
is outside these native server profiles.

An existing `.venv` must also use Python 3.10–3.12. An incompatible environment
is reported, not deleted. Install in a separate checkout if that environment
must be retained. `--skip-system-packages` still installs all Python packages
and performs every verification; it is not a way to skip FFCV or GPU checks.

For a custom `--python`, install the matching development headers and venv
support (for example `python3.11-dev` and `python3.11-venv`), not only the
Ubuntu default `python3-dev`. Setup checks that the selected interpreter's
`Python.h` exists before downloading Python dependencies. The Ubuntu 24.04
profile selects `libglib2.0-0t64`; Ubuntu 22.04 uses `libglib2.0-0`.

Setup writes logs under `.setup-logs/`, verified GPU/CPU/runtime provenance to
`.venv/server-profile.json` and resolved Python versions to
`.venv/server-requirements.freeze.txt` after successful verification. No token
is read or written into these logs. On an installation failure, fix the first
error in the log and rerun the same setup command. Existing results are reused.

If a campaign reports missing `ffcv, cupy`, check the active interpreter:

```bash
source activate_server.sh
python -c 'import sys, torch, ffcv, cupy; print(sys.executable); print(torch.__version__, torch.version.cuda)'
python -m mpsae.server
python -m mpsae --config configs/thesis.yaml run
```

Empty reports after a preflight failure mean no evaluation was completed;
the final exception identifies the cause. Do not remove the dependency check
or change the source extraction recipe merely to bypass a missing package.

### PyArrow requires NumPy 2 during setup

The native FFCV stack uses NumPy 1.26.4. Arrow 26 requires NumPy 2 at
import time, even when `pip check` reports no broken requirements. The server
constraints pin PyArrow 25.0.1, and verification checks an actual NumPy/Arrow/
Hugging Face Dataset round trip before starting GPU checks.

After updating this checkout, repair an existing environment in place:

```bash
source activate_server.sh
python -m pip install --no-build-isolation -r requirements-server.txt
bash setup_server.sh
bash all_in_one.sh --skip-setup
```

Existing data and checkpoints are retained. Do not upgrade NumPy alone to
bypass this error, since the FFCV stack shares these native dependencies.

## Workspace and long jobs

After installing on the actual GPU server, run the regression suite and a
small synthetic campaign before the full run:

```bash
source activate_server.sh
python -m pytest tests/ -q
bash all_in_one.sh --skip-setup --smoke
bash all_in_one.sh --skip-setup
```

The smoke campaign uses its own workspace and never uploads to HF or W&B.
Legacy `python -m mpsae --smoke run` also isolates its files under
`<workspace>/smoke-legacy/`, preserving real source archives and checkpoints.
It checks the training/evaluation/reporting path; it does not establish real
dataset accuracy or certify every supported GPU. `setup_server.sh` separately
tests the installed GPU, FFCV decoding, CuPy JIT and CUDA retrieval extension.

Use fast local SSD storage. In `.env`, `MPSAE_WORKSPACE=/workspace/mpsae` moves
the inputs, features, checkpoints, campaign state and reports together;
`HF_HOME=/workspace/hf` moves the Hub cache. The default is `./work`.
See [benchmark.md](benchmark.md) for the exact layout and dataset protocols.

```bash
tmux new -s mpsae
source activate_server.sh
bash pipeline.sh
# Ctrl-b d detaches; tmux attach -t mpsae reconnects.
```

Experiments use one configured device (`cuda:0` by default); multiple visible
GPUs do not automatically make this a distributed run. To select one GPU,
set `CUDA_VISIBLE_DEVICES` before launching Python. Scientific settings and
checkpoint identities are independent of Docker or the shell launcher.

All entry scripts are at the repository root: `setup_server.sh`,
`activate_server.sh`, `all_in_one.sh`, `pipeline.sh`, and `capture_golden.py`.
The generated environment file is also at root (`env.sh`, gitignored). After
updating an older checkout, run the new installer once to regenerate this file.
