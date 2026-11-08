#!/usr/bin/env bash
# Native A100 setup: bash scripts/setup_server.sh
# Installs into .venv; never starts a benchmark or reads HF credentials.
set -Eeuo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
step() { printf '\n==> %s\n' "$*"; }

# Matched versions: requirements-server.txt / constraints/server-a100.txt.
# Keep CUDA 12.6 on A100 even with a newer driver, rather than following a
# moving wheel index. nvidia-smi's CUDA banner is not the installed toolkit.
select_a100_stack() {
    local driver_version="$1"
    [[ "$driver_version" =~ ^[0-9]+([.][0-9]+)*$ ]] || die "cannot parse NVIDIA driver: $driver_version"
    [[ "$(printf '%s\n' 560.28.03 "$driver_version" | sort -V | head -1)" == 560.28.03 ]] \
        || die "A100 setup needs driver >= 560.28.03 for CUDA 12.6; found $driver_version. Ask the server provider to update the driver."
    CUDA_VERSION=12.6
    TORCH_VERSION=2.9.1
    TORCHVISION_VERSION=0.24.1
    TORCH_INDEX=https://download.pytorch.org/whl/cu126
}

usage() {
    cat <<'HELP'
Usage: bash scripts/setup_server.sh [options]
  --dry-run               Inspect the server and print the plan without changes.
  --python PATH           Python 3.10, 3.11 or 3.12 for creating .venv.
  --skip-system-packages  Admin already installed CUDA 12.6 + JPEG/OpenCV headers.
  --help                  Show this help.

Default target: Ubuntu 22.04/24.04 x86_64, NVIDIA A100, driver >= 560.28.03.
Installs CUDA Toolkit 12.6 (toolkit only), PyTorch, FFCV, CuPy, all project
requirements, and verifies GPU operations including the CUDA extension.
Root and sudo accounts are supported. The NVIDIA driver is managed by the host.
HELP
}

ubuntu_repository() {
    [[ -r /etc/os-release ]] || die '/etc/os-release missing'
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" =~ ^(22.04|24.04)$ ]] \
        || die 'supported OS: Ubuntu 22.04 or 24.04; see docs/server.md'
    printf 'ubuntu%s\n' "${VERSION_ID//./}"
}

main() {
    local dry_run=0 skip_system=0 setup_python=""
    local root_path log_path distro temporary_dir candidate gpu_info driver_version
    local -a admin=()
    while (( $# )); do
        case "$1" in
            --dry-run) dry_run=1; shift ;;
            --skip-system-packages) skip_system=1; shift ;;
            --python) (( $# >= 2 )) || die '--python needs a path'; setup_python="$2"; shift 2 ;;
            --help|-h) usage; return ;;
            *) die "unknown option: $1 (use --help)" ;;
        esac
    done
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
        || die 'run this installer on the Linux x86_64 A100 server'
    distro="$(ubuntu_repository)"
    root_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$root_path"
    command -v nvidia-smi >/dev/null || die 'nvidia-smi missing; use an A100 server with the NVIDIA driver installed'
    gpu_info="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader)"
    printf '%s\n' "$gpu_info"
    [[ "$gpu_info" == *A100* ]] || die 'this installation profile targets NVIDIA A100'
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
    select_a100_stack "$driver_version"
    if [[ -n "${MPSAE_TORCH_INDEX:-}" && "$MPSAE_TORCH_INDEX" != "$TORCH_INDEX" ]]; then
        die 'MPSAE_TORCH_INDEX conflicts with the pinned A100 profile; unset it before setup'
    fi
    if [[ -z "$setup_python" ]]; then
        for candidate in python3.11 python3.12 python3.10 python3; do
            if command -v "$candidate" >/dev/null && "$candidate" -c \
                'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))'; then
                setup_python="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$setup_python" ]]; then
        [[ "$skip_system" == 0 ]] || die 'Python 3.10-3.12 missing; supply --python PATH'
        setup_python=/usr/bin/python3
    fi
    step 'installation plan'
    printf 'Python: %s\nCUDA Toolkit: %s\nPyTorch: %s / torchvision %s\nWheel index: %s\nEnvironment: %s/.venv\n' \
        "$setup_python" "$CUDA_VERSION" "$TORCH_VERSION" "$TORCHVISION_VERSION" "$TORCH_INDEX" "$root_path"
    printf 'FFCV + CuPy: required; versions in constraints/server-a100.txt\nSystem packages skipped: %s\n' "$skip_system"
    [[ "$dry_run" == 0 ]] || return 0

    mkdir -p .setup-logs
    log_path="$root_path/.setup-logs/server-$(date +%Y%m%d-%H%M%S)-$$.log"
    exec > >(tee -a "$log_path") 2>&1
    trap 'printf "Setup failed at line %s. See .setup-logs/; benchmark was not started.\n" "$LINENO" >&2' ERR
    # Reject an incompatible existing environment before expensive system or
    # toolkit installation. Never delete a caller's environment to repair it.
    if [[ -d .venv ]]; then
        [[ -x .venv/bin/python ]] || die '.venv exists but is not a Linux Python environment; use a separate server checkout'
        .venv/bin/python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "existing .venv needs Python 3.10-3.12; it has not been deleted"'
    fi
    if command -v "$setup_python" >/dev/null; then
        "$setup_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "use Python 3.10-3.12"'
    elif [[ "$setup_python" != /usr/bin/python3 || "$skip_system" == 1 ]]; then
        die "Python interpreter not found: $setup_python"
    fi
    if [[ "$skip_system" == 0 ]]; then
        if [[ "$EUID" != 0 ]]; then
            command -v sudo >/dev/null || die 'system packages need root or sudo; ask the administrator, then use --skip-system-packages'
            admin=(sudo)
            sudo -v
        fi
        step 'system libraries and build tools'
        "${admin[@]}" apt-get update
        "${admin[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential pkg-config git curl ca-certificates tmux \
            python3 python3-dev python3-venv libturbojpeg0-dev libopencv-dev libgl1 libglib2.0-0
        if [[ ! -x "${MPSAE_CUDA_HOME:-/usr/local/cuda-12.6}/bin/nvcc" ]]; then
            step 'CUDA Toolkit 12.6 from NVIDIA (no driver package)'
            temporary_dir="$(mktemp -d)"
            curl --fail --location --retry 3 \
                "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb" \
                --output "$temporary_dir/cuda-keyring.deb"
            "${admin[@]}" dpkg -i "$temporary_dir/cuda-keyring.deb"
            "${admin[@]}" apt-get update
            "${admin[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-toolkit-12-6
            rm -- "$temporary_dir/cuda-keyring.deb"
            rmdir -- "$temporary_dir"
        fi
    fi
    export CUDA_HOME="${MPSAE_CUDA_HOME:-/usr/local/cuda-12.6}"
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || die "CUDA 12.6 compiler missing at $CUDA_HOME/bin/nvcc; install the toolkit or set MPSAE_CUDA_HOME"
    "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.6,' || die 'CUDA_HOME must point to CUDA Toolkit 12.6'
    pkg-config --exists opencv4 libturbojpeg || die 'OpenCV/libturbojpeg development packages missing'
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TORCH_CUDA_ARCH_LIST=8.0
    export MAX_JOBS="${MAX_JOBS:-4}"
    export MPLBACKEND=Agg
    "$setup_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "use Python 3.10-3.12"'
    if [[ ! -d .venv ]]; then
        "$setup_python" -m venv .venv
    fi
    [[ -x .venv/bin/python ]] || die '.venv exists but is not a Linux Python environment; use a separate server checkout'
    .venv/bin/python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "existing .venv needs Python 3.10-3.12; it has not been deleted"'
    local venv_python="$root_path/.venv/bin/python"
    step 'Python packaging tools'
    "$venv_python" -m pip install --upgrade pip 'setuptools==80.9.0' 'wheel==0.45.1'
    step 'pinned A100 PyTorch'
    "$venv_python" -m pip install --index-url "$TORCH_INDEX" \
        "torch==$TORCH_VERSION+cu126" "torchvision==$TORCHVISION_VERSION+cu126"
    step 'all experiment dependencies, including FFCV and CuPy'
    "$venv_python" -m pip install --no-build-isolation -r requirements-server.txt
    "$venv_python" -m pip check

    step 'runtime environment'
    {
        printf '# Generated by setup_server.sh. No credentials.\n'
        printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
        printf 'export PATH="$CUDA_HOME/bin:$PATH"\n'
        printf 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        printf 'export TORCH_CUDA_ARCH_LIST=8.0\nexport MAX_JOBS="${MAX_JOBS:-4}"\nexport MPLBACKEND=Agg\n'
    } > scripts/env.sh
    step 'GPU, CuPy, FFCV pipeline and compiled retrieval checks'
    "$venv_python" -m mpsae.server
    "$venv_python" -m pip freeze > .venv/server-requirements.freeze.txt
    if [[ ! -e .env ]]; then
        cp -- .env.example .env
        chmod 600 .env
    fi
    printf '\nSetup complete. Log: %s\n' "$log_path"
    cat <<'DONE'

  source scripts/activate_server.sh
  # Fill HF_TOKEN in .env after accepting the ImageNet access conditions.
  python -m mpsae --config configs/thesis.yaml plan
  python -m mpsae --config configs/smoke.yaml --smoke run
  python -m mpsae --config configs/thesis.yaml run
DONE
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
