#!/usr/bin/env bash
# Native NVIDIA GPU setup: bash setup_server.sh
# Installs into .venv; never starts a benchmark or reads HF credentials.
set -Eeuo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
step() { printf '\n==> %s\n' "$*"; }

# Use the minimum tested toolkit covering all selected architectures. Keep
# package versions stable across devices; nvidia-smi's CUDA banner is not the
# installed toolkit. Driver floors cover the versioned apt packages' updates.
select_cuda_stack() {
    local driver_version="$1" capabilities="$2" requested="${3:-auto}"
    local capability required=126 minimum_driver
    [[ "$driver_version" =~ ^[0-9]+([.][0-9]+)*$ ]] || die "cannot parse NVIDIA driver: $driver_version"
    [[ -n "$capabilities" ]] || die 'no selected GPU compute capabilities'
    local -a selected_arches=()
    IFS=';' read -ra selected_arches <<< "$capabilities"
    for capability in "${selected_arches[@]}"; do
        case "$capability" in
            7.0|7.5|8.0|8.6|8.9|9.0) ;;
            10.0|12.0) (( required >= 128 )) || required=128 ;;
            10.3) required=129 ;;
            *) die "unsupported GPU capability $capability; supported: 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 10.3, 12.0" ;;
        esac
    done
    case "$requested" in
        auto) CUDA_FLAVOR=cu130 ;;
        12.6) CUDA_FLAVOR=cu126 ;;
        12.8) CUDA_FLAVOR=cu128 ;;
        12.9) CUDA_FLAVOR=cu129 ;;
        13.0) CUDA_FLAVOR=cu130 ;;
        *) die '--cuda must be auto, 13.0, 12.6, 12.8 or 12.9' ;;
    esac
    if [[ "$CUDA_FLAVOR" == cu130 && ";$capabilities;" == *";7.0;"* ]]; then
        die "CUDA 13 does not support Volta sm_70; use --cuda 12.6"
    fi
    (( ${CUDA_FLAVOR#cu} >= required )) || die "selected GPU needs CUDA ${required:0:2}.${required:2:1} or newer; --cuda $requested is too old"
    case "$CUDA_FLAVOR" in
        cu126) CUDA_VERSION=12.6; minimum_driver=560.35.05 ;;
        cu128) CUDA_VERSION=12.8; minimum_driver=570.124.06 ;;
        cu129) CUDA_VERSION=12.9; minimum_driver=575.57.08 ;;
        cu130) CUDA_VERSION=13.0; minimum_driver=580.65.06 ;;
    esac
    [[ "$(printf '%s\n' "$minimum_driver" "$driver_version" | sort -V | head -1)" == "$minimum_driver" ]] \
        || die "CUDA $CUDA_VERSION profile needs NVIDIA driver >= $minimum_driver; found $driver_version. Ask the server provider to update the driver."
    TORCH_VERSION=2.9.1
    TORCHVISION_VERSION=0.24.1
    TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_FLAVOR"
    TORCH_ARCHES="$(printf '%s\n' "${selected_arches[@]}" | sort -uV | paste -sd';' -)"
}

usage() {
    cat <<'HELP'
Usage: bash setup_server.sh [options]
  --dry-run               Inspect the server and print the plan without changes.
  --python PATH           Python 3.10, 3.11 or 3.12 for creating .venv.
  --cuda VERSION          auto (CUDA 13.0), 13.0, 12.6, 12.8 or 12.9.
  --skip-system-packages  Admin already installed the selected CUDA + Python development headers.
  --help                  Show this help.

Target: Ubuntu 22.04/24.04, x86_64 or ARM64/SBSA, NVIDIA GPUs including
A100, RTX A6000, RTX 4090/5090, H100/H200 and GB300.
Installs the selected CUDA Toolkit (toolkit only), PyTorch, direct JPEG extraction, all project
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
    local dry_run=0 skip_system=0 setup_python="" requested_cuda=auto
    local root_path log_path distro temporary_dir candidate gpu_info driver_version capabilities cpu_arch repo_arch vision_spec glib_package
    local -a admin=() gpu_query=()
    while (( $# )); do
        case "$1" in
            --dry-run) dry_run=1; shift ;;
            --skip-system-packages) skip_system=1; shift ;;
            --python) (( $# >= 2 )) || die '--python needs a path'; setup_python="$2"; shift 2 ;;
            --cuda) (( $# >= 2 )) || die '--cuda needs a version'; requested_cuda="$2"; shift 2 ;;
            --help|-h) usage; return ;;
            *) die "unknown option: $1 (use --help)" ;;
        esac
    done
    [[ "$(uname -s)" == Linux ]] || die 'run this installer on the Linux GPU server'
    cpu_arch="$(uname -m)"
    case "$cpu_arch" in
        x86_64) repo_arch=x86_64 ;;
        aarch64|arm64) repo_arch=sbsa ;;
        *) die "unsupported CPU architecture: $cpu_arch" ;;
    esac
    distro="$(ubuntu_repository)"
    glib_package=libglib2.0-0
    [[ "$distro" != ubuntu2404 ]] || glib_package=libglib2.0-0t64
    root_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    cd "$root_path"
    command -v nvidia-smi >/dev/null || die 'nvidia-smi missing; the server provider must install the NVIDIA driver'
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        [[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != -1 ]] || die 'CUDA_VISIBLE_DEVICES hides all GPUs'
        if [[ "$CUDA_VISIBLE_DEVICES" != *MIG-* ]]; then
            gpu_query=("--id=$CUDA_VISIBLE_DEVICES")
        fi
    fi
    gpu_info="$(nvidia-smi "${gpu_query[@]}" --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader)"
    printf '%s\n' "$gpu_info"
    capabilities="$(printf '%s\n' "$gpu_info" | cut -d, -f2 | tr -d ' ' | sort -uV | paste -sd';' -)"
    driver_version="$(printf '%s\n' "$gpu_info" | cut -d, -f4 | head -1 | tr -d ' ')"
    select_cuda_stack "$driver_version" "$capabilities" "$requested_cuda"
    if [[ -n "${MPSAE_TORCH_INDEX:-}" && "$MPSAE_TORCH_INDEX" != "$TORCH_INDEX" ]]; then
        die 'MPSAE_TORCH_INDEX conflicts with the selected GPU profile; unset it and use --cuda instead'
    fi
    if [[ -z "$setup_python" ]]; then
        # Prefer Ubuntu's interpreter so python3-dev/python3-venv match it.
        # An explicitly requested custom interpreter remains supported below.
        for candidate in /usr/bin/python3 python3.11 python3.12 python3.10 python3; do
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
    printf 'CPU: %s; NVIDIA repository: %s; CUDA architectures: %s\n' "$cpu_arch" "$repo_arch" "$TORCH_ARCHES"
    printf 'FFCV + CuPy: required; versions in constraints/server.txt\nSystem packages skipped: %s\n' "$skip_system"
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
            python3 python3-dev python3-venv libjpeg-dev libgl1 "$glib_package"
        if [[ ! -x "${MPSAE_CUDA_HOME:-/usr/local/cuda-$CUDA_VERSION}/bin/nvcc" ]]; then
            step "CUDA Toolkit $CUDA_VERSION from NVIDIA (no driver package)"
            temporary_dir="$(mktemp -d)"
            curl --fail --location --retry 3 \
                "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$repo_arch/cuda-keyring_1.1-1_all.deb" \
                --output "$temporary_dir/cuda-keyring.deb"
            "${admin[@]}" dpkg -i "$temporary_dir/cuda-keyring.deb"
            "${admin[@]}" apt-get update
            "${admin[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_VERSION/./-}"
            rm -- "$temporary_dir/cuda-keyring.deb"
            rmdir -- "$temporary_dir"
        fi
    fi
    export CUDA_HOME="${MPSAE_CUDA_HOME:-/usr/local/cuda-$CUDA_VERSION}"
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || die "CUDA $CUDA_VERSION compiler missing at $CUDA_HOME/bin/nvcc; install the toolkit or set MPSAE_CUDA_HOME"
    "$CUDA_HOME/bin/nvcc" --version | grep -Fq "release $CUDA_VERSION," || die "CUDA_HOME must point to CUDA Toolkit $CUDA_VERSION"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TORCH_CUDA_ARCH_LIST="$TORCH_ARCHES"
    export MPSAE_CUDA_VERSION="$CUDA_VERSION"
    export MAX_JOBS="${MAX_JOBS:-4}"
    export MPLBACKEND=Agg
    "$setup_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "use Python 3.10-3.12"'
    if [[ ! -d .venv ]]; then
        "$setup_python" -m venv .venv
    fi
    [[ -x .venv/bin/python ]] || die '.venv exists but is not a Linux Python environment; use a separate server checkout'
    .venv/bin/python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "existing .venv needs Python 3.10-3.12; it has not been deleted"'
    local venv_python="$root_path/.venv/bin/python"
    # A custom interpreter can coexist with Ubuntu's default Python. Installing
    # python3-dev alone does not provide that interpreter's extension headers.
    "$venv_python" -c 'import sys, sysconfig; from pathlib import Path; header = Path(sysconfig.get_path("include")) / "Python.h"; sys.exit(0 if header.is_file() else f"Missing {header}. Install python{sys.version_info.major}.{sys.version_info.minor}-dev (and the matching -venv package), or choose a Python with development headers using --python.")'
    step 'Python packaging tools'
    # Keep pip's wheel/download cache: reinstalling on a persistent server must
    # not fetch multi-GB CUDA wheels again. Versions remain pinned below.
    "$venv_python" -m pip install --upgrade pip 'setuptools==80.9.0' 'wheel==0.45.1'
    # torchvision depends on NumPy. Install the shared ABI first, so the initial
    # Torch install cannot leave NumPy 2 in a half-configured environment.
    "$venv_python" -m pip install -c constraints/server.txt numpy
    step 'pinned GPU-compatible PyTorch'
    # The official ARM64 torchvision wheels have no +cuXXX local suffix.
    vision_spec="torchvision==$TORCHVISION_VERSION+$CUDA_FLAVOR"
    [[ "$repo_arch" != sbsa ]] || vision_spec="torchvision==$TORCHVISION_VERSION"
    "$venv_python" -m pip install -c constraints/server.txt --index-url "$TORCH_INDEX" \
        "torch==$TORCH_VERSION+$CUDA_FLAVOR" "$vision_spec"
    if [[ "$repo_arch" == sbsa ]]; then
        "$venv_python" -m pip install --force-reinstall --no-deps --index-url "$TORCH_INDEX" "$vision_spec"
    fi
    step 'direct-extraction and experiment dependencies (no FFCV/CuPy required)'
    "$venv_python" -m pip install --no-build-isolation -r requirements-server.txt
    "$venv_python" -m pip check

    step 'runtime environment'
    {
        printf '# Generated by setup_server.sh. No credentials.\n'
        printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
        printf 'export PATH="$CUDA_HOME/bin:$PATH"\n'
        printf 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        printf 'export TORCH_CUDA_ARCH_LIST=%q\nexport MPSAE_CUDA_VERSION=%q\n' "$TORCH_ARCHES" "$CUDA_VERSION"
        printf 'export MAX_JOBS="${MAX_JOBS:-4}"\nexport MPLBACKEND=Agg\n'
    } > env.sh
    step 'GPU, direct JPEG pipeline and compiled retrieval checks'
    "$venv_python" -m mpsae.server --profile-output .venv/server-profile.json
    "$venv_python" -m pip freeze > .venv/server-requirements.freeze.txt
    if [[ ! -e .env ]]; then
        cp -- .env.example .env
        chmod 600 .env
    fi
    printf '\nSetup complete. Log: %s\n' "$log_path"
    cat <<'DONE'

  source activate_server.sh
  # Fill HF_TOKEN (read + write) and optionally WANDB_TOKEN in .env.
  # Defaults: private HF <account>/MPSAE; W&B project cvpr.
  bash all_in_one.sh --skip-setup
DONE
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
