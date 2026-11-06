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
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
echo "compute capability: $CAP"

# Blackwell (sm_120: RTX 5090, RTX PRO 6000) only has kernels in cu128 builds.
case "$CAP" in
    12.*|11.*) TORCH_INDEX="https://download.pytorch.org/whl/cu128"; ARCHES="9.0;10.0;12.0" ;;
    10.*)      TORCH_INDEX="https://download.pytorch.org/whl/cu128"; ARCHES="9.0;10.0" ;;
    9.*)       TORCH_INDEX="https://download.pytorch.org/whl/cu124"; ARCHES="9.0" ;;
    8.*)       TORCH_INDEX="https://download.pytorch.org/whl/cu124"; ARCHES="8.0;8.6;8.9" ;;
    *)         TORCH_INDEX="https://download.pytorch.org/whl/cu124"; ARCHES="$CAP" ;;
esac
echo "torch wheel index: $TORCH_INDEX"
echo "TORCH_CUDA_ARCH_LIST: $ARCHES"

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
python -m pip install -r requirements.txt

if [[ "$WANT_FFCV" == 1 ]]; then
    step "ffcv (CSRv2 embedding recipe)"
    # ffcv has no wheels and compiles against libjpeg-turbo/opencv; a failure
    # here is survivable, so report it clearly instead of aborting setup.
    if python -m pip install ffcv; then
        echo "ffcv installed"
    else
        cat >&2 <<'MSG'
ffcv failed to build. The ff2048 backbone (MRL recipe) still works:

    ./scripts/pipeline.sh -- --backbones ff2048

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
