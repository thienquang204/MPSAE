#!/usr/bin/env bash
# Evaluation wheels only: no toolkit, compiler, FFCV, CuPy or training setup.
set -Eeuo pipefail

scene_die() { printf 'error: %s\n' "$*" >&2; exit 1; }

scene_cuda_profile() {
    local driver="$1" capabilities="$2" flavor="$3" floor capability
    [[ "$driver" =~ ^[0-9]+([.][0-9]+)*$ ]] || scene_die 'invalid NVIDIA driver version'
    case "$flavor" in
        cu128) floor=570.124.06 ;;
        cu130) floor=580.65.06 ;;
        *) scene_die '--cuda must be cu128 or cu130' ;;
    esac
    [[ "$(printf '%s\n' "$floor" "$driver" | sort -V | head -1)" == "$floor" ]] \
        || scene_die "$flavor requires driver >= $floor; found $driver. Ask the provider to update the driver."
    [[ -n "$capabilities" ]] || scene_die 'no visible NVIDIA GPU'
    local -a arches=()
    IFS=';' read -ra arches <<< "$capabilities"
    for capability in "${arches[@]}"; do
        case "$capability" in
            7.5|8.0|8.6|8.9|9.0|10.0|12.0) ;;
            10.3) [[ "$flavor" == cu130 ]] || scene_die 'sm_103 requires --cuda cu130' ;;
            *) scene_die "unsupported compute capability $capability in this evaluation installer" ;;
        esac
    done
}

scene_setup_main() {
    local python=python3 venv=.venv-scene cuda=cu128 dry=0 root info driver caps
    while (( $# )); do
        case "$1" in
            --python|--venv|--cuda)
                (( $# >= 2 )) || scene_die "$1 needs a value"
                case "$1" in --python) python="$2";; --venv) venv="$2";; --cuda) cuda="$2";; esac
                shift 2 ;;
            --dry-run) dry=1; shift ;;
            --help|-h)
                printf 'Usage: bash setup_scene_server.sh [--python python3.11] [--venv PATH] [--cuda cu128|cu130] [--dry-run]\nLinux x86_64, Python 3.10-3.12, NVIDIA driver already installed. No root required.\n'
                return ;;
            *) scene_die "unknown setup option: $1" ;;
        esac
    done
    root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    cd "$root"
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || scene_die 'use a Linux x86_64 GPU server'
    command -v "$python" >/dev/null || scene_die 'Python missing; install python3 and python3-venv, or pass --python'
    "$python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "use Python 3.10-3.12 (or --python python3.11)"'
    command -v nvidia-smi >/dev/null || scene_die 'nvidia-smi missing; a GPU driver must be provided by the server host'
    local -a gpu_filter=()
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        [[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != -1 ]] || scene_die 'CUDA_VISIBLE_DEVICES hides all GPUs'
        [[ "$CUDA_VISIBLE_DEVICES" == *MIG-* ]] || gpu_filter=("--id=$CUDA_VISIBLE_DEVICES")
    fi
    info="$(nvidia-smi "${gpu_filter[@]}" --query-gpu=driver_version,compute_cap --format=csv,noheader,nounits)"
    driver="$(printf '%s\n' "$info" | cut -d, -f1 | head -1 | tr -d ' ')"
    caps="$(printf '%s\n' "$info" | cut -d, -f2 | tr -d ' ' | sort -u | paste -sd';' -)"
    scene_cuda_profile "$driver" "$caps" "$cuda"
    printf 'Scene environment: %s\nPython: %s\nTorch: 2.9.1; torchvision: 0.24.1; CUDA wheels: %s\n' "$venv" "$python" "$cuda"
    [[ "$dry" == 0 ]] || return
    if [[ -e "$venv" ]]; then
        [[ -x "$venv/bin/python" && -f "$venv/pyvenv.cfg" ]] || scene_die "$venv is not a Linux venv; nothing was deleted"
        "$venv/bin/python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "existing venv requires Python 3.10-3.12"'
    else
        "$python" -m venv "$venv" || scene_die 'venv creation failed; install python3-venv for your interpreter and retry'
    fi
    "$venv/bin/python" -m pip install --upgrade pip
    "$venv/bin/python" -m pip install --only-binary=:all: "torch==2.9.1+$cuda" "torchvision==0.24.1+$cuda" \
        --index-url "https://download.pytorch.org/whl/$cuda"
    "$venv/bin/python" -m pip install --only-binary=:all: -r requirements-scene.txt
    "$venv/bin/python" -m pip check
    "$venv/bin/python" -m mpsae.scene doctor --expected-cuda "${cuda:2:2}.${cuda:4:1}"
    printf 'Ready. Run: bash scene_only.sh\n'
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then scene_setup_main "$@"; fi
