#!/usr/bin/env bash
# One-time environment setup for a rented GPU box (Vast.ai and similar).
#
#   ./scripts/setup_server.sh              # autodetect the CUDA build to install
#   ./scripts/setup_server.sh --no-ffcv    # skip FFCV (only the MRL recipe will work)
#
# FFCV is the awkward dependency: it needs libjpeg-turbo, opencv and pkg-config,
# and it is required by the CSRv2 embedding recipe (rn18, convnext_tiny, vit_b16,
# dinov2_vits14). The MRL recipe used for ff2048 does not need it.
set -Eeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

WANT_FFCV=1
for arg in "$@"; do
    case "$arg" in
        --no-ffcv) WANT_FFCV=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

die() { printf '\033[38;5;203merror:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "inspecting the GPU"
if ! command -v nvidia-smi >/dev/null; then
    die "nvidia-smi not found; this project needs a CUDA GPU"
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

# Build the arch list from the capabilities actually present. Generating it
# rather than hard-coding one matters under CUDA 13, whose nvcc rejects the
# architectures it dropped (sm_50 through sm_70) instead of ignoring them.
ARCHES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
          | tr -d ' ' | sort -uV | paste -sd';' -)"
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -V | head -1)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
cap_at_least() { [ "$(printf '%s\n%s\n' "$1" "$CAP" | sort -V | head -1)" = "$1" ]; }

# Index choice is constrained from both ends: the driver sets the highest CUDA
# runtime that can load, the GPU sets the lowest toolkit that still has kernels
# for it. Checked 2026-11: cu128 is frozen at torch 2.11.0, so it is only used
# where nothing newer can run; cu132 carries the current release.
if ! cap_at_least 7.5; then
    # CUDA 13 removed Maxwell, Pascal and Volta entirely.
    CU=cu126; WHY="sm_$CAP predates CUDA 13, which dropped it"
elif [ "${DRIVER:-0}" -ge 580 ]; then
    CU=cu132; WHY="driver $DRIVER supports the CUDA 13 runtime"
elif cap_at_least 10.0; then
    if [ "${DRIVER:-0}" -lt 570 ]; then
        die "Blackwell (sm_$CAP) needs driver >= 570, this box has $DRIVER.
  Ask the provider for a newer driver image; no wheel works around this."
    fi
    CU=cu128; WHY="Blackwell on driver $DRIVER; cu128 is the newest runtime this driver loads"
    echo "note: cu128 is frozen at torch 2.11.0. A driver >= 580 would get cu132 and torch 2.15." >&2
else
    CU=cu126; WHY="driver $DRIVER predates CUDA 13"
fi
# An operator who knows better can override, e.g. MPSAE_TORCH_INDEX=.../cu130
TORCH_INDEX="${MPSAE_TORCH_INDEX:-https://download.pytorch.org/whl/$CU}"

echo "compute capability: $CAP (arch list: $ARCHES)"
echo "driver:             ${DRIVER:-unknown}"
echo "torch wheel index:  $TORCH_INDEX  — $WHY"

step "system packages"
if command -v apt-get >/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        build-essential pkg-config git curl ca-certificates \
        libturbojpeg0-dev libopencv-dev python3-dev python3-venv
else
    echo "not a Debian/Ubuntu host; install build tools, pkg-config, libjpeg-turbo and opencv yourself"
fi

step "python environment"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

step "pytorch"
python -m pip install --index-url "$TORCH_INDEX" torch torchvision

step "project requirements"
python -m pip install -r requirements.txt ninja

if [[ "$WANT_FFCV" == 1 ]]; then
    step "ffcv (CSRv2 embedding recipe)"
    # NormalizeImage runs on CUDA and FFCV does not declare CuPy itself.
    CUPY_PACKAGE="$(python -c "import torch; print('cupy-cuda' + torch.version.cuda.split('.')[0] + 'x')")"
    python -m pip install "$CUPY_PACKAGE"
    # ffcv has no wheels and compiles against libjpeg-turbo/opencv; a failure
    # here is survivable, so report it clearly instead of aborting setup.
    if python -m pip install ffcv; then
        echo "ffcv installed"
    else
        cat >&2 <<'MSG'
ffcv failed to build. The ff2048 backbone (MRL recipe) still works:

    # Set backbones: [ff2048] in configs/thesis.yaml, then:
    ./scripts/pipeline.sh

For the timm backbones, try a conda environment instead:
    conda create -n mpsae python=3.11 -y && conda activate mpsae
    conda install -c conda-forge -c pytorch cupy pkg-config libjpeg-turbo opencv numba -y
    pip install ffcv
MSG
    fi
fi

step "writing scripts/env.sh"
cat > scripts/env.sh <<EOF
# Sourced by the pipeline; regenerate with scripts/setup_server.sh
export TORCH_CUDA_ARCH_LIST="$ARCHES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

step "verifying"
export TORCH_CUDA_ARCH_LIST="$ARCHES"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch cannot see a CUDA device"
index = torch.cuda.current_device()
print("device", torch.cuda.get_device_name(index),
      "sm_" + "".join(map(str, torch.cuda.get_device_capability(index))))
# A matmul proves kernels exist for this architecture, which is the usual
# failure mode of a too-old wheel on new silicon.
a = torch.randn(512, 512, device="cuda")
print("matmul ok:", float((a @ a.T).sum()) == float((a @ a.T).sum()))
for name in ("numpy", "timm", "datasets", "huggingface_hub", "matplotlib", "ffcv"):
    try:
        module = __import__(name)
        print(f"{name:16s} {getattr(module, '__version__', 'ok')}")
    except Exception as error:
        print(f"{name:16s} MISSING ({type(error).__name__})")
PY

cat <<'MSG'

Setup complete.

  1. cp .env.example .env   and paste HF_TOKEN
  2. ./scripts/pipeline.sh --smoke     (end-to-end rehearsal)
  3. ./scripts/pipeline.sh             (the real run)
MSG
