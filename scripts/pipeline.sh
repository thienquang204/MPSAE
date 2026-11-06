#!/usr/bin/env bash
# All-in-one launcher. On a fresh GPU box the only thing you supply is .env.
#
#   cp .env.example .env && $EDITOR .env     # paste HF_TOKEN (and WANDB_API_KEY)
#   ./scripts/setup_server.sh                # once, installs the environment
#   ./scripts/pipeline.sh                    # everything: data -> embeddings -> 690 fits -> report
#
#   ./scripts/pipeline.sh --smoke            # 15 minutes, proves the plumbing end to end
#   ./scripts/pipeline.sh status             # what is done, what is left
#   ./scripts/pipeline.sh plan               # the experiment grid, no side effects
#   ./scripts/pipeline.sh stage embed        # one stage only
#   ./scripts/pipeline.sh -- --backbones ff2048 --methods csr,mpsae
#
# Safe to interrupt and re-run: every stage keeps a receipt and every fit
# checkpoints, so a restart continues instead of starting over.
set -Eeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${MPSAE_WORKSPACE:=$HERE/work}"
: "${MPSAE_ENV_FILE:=$HERE/.env}"
: "${MPSAE_PYTHON:=python3}"
# Keeps long runs from fragmenting the allocator across hundreds of fits.
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export MPSAE_WORKSPACE PYTORCH_CUDA_ALLOC_CONF

die() { printf '\033[38;5;203merror:\033[0m %s\n' "$*" >&2; exit 1; }
note() { printf '\033[38;5;245m%s\033[0m\n' "$*" >&2; }

if [[ -f "$HERE/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HERE/.venv/bin/activate"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    note "using conda environment $CONDA_PREFIX"
fi

command -v "$MPSAE_PYTHON" >/dev/null || die "$MPSAE_PYTHON not found; run scripts/setup_server.sh"

if [[ ! -f "$MPSAE_ENV_FILE" ]]; then
    die "missing $MPSAE_ENV_FILE
  cp .env.example .env
  then paste your Hugging Face token (it must have accepted the ImageNet terms at
  https://huggingface.co/datasets/ILSVRC/imagenet-1k and, to publish, have write scope)."
fi
if ! grep -Eq '^HF_TOKEN=.+' "$MPSAE_ENV_FILE"; then
    die "HF_TOKEN is empty in $MPSAE_ENV_FILE; ImageNet-1k is gated and cannot be downloaded without it"
fi

COMMAND="run"
case "${1:-}" in
    status|plan|report) COMMAND="$1"; shift ;;
    stage) COMMAND="stage"; shift ;;
    --help|-h) sed -n '2,20p' "$0"; exit 0 ;;
esac
[[ "${1:-}" == "--" ]] && shift

mkdir -p "$MPSAE_WORKSPACE"
note "workspace: $MPSAE_WORKSPACE"
note "logs:      $MPSAE_WORKSPACE/logs/"

# Mirror everything to a timestamped transcript as well as the structured log,
# so a crashed run still has its console output.
TRANSCRIPT="$MPSAE_WORKSPACE/logs/console-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$TRANSCRIPT")"

set -o pipefail
"$MPSAE_PYTHON" -m mpsae --env-file "$MPSAE_ENV_FILE" --workspace "$MPSAE_WORKSPACE" \
    "$COMMAND" "$@" 2>&1 | tee -a "$TRANSCRIPT"
status=${PIPESTATUS[0]}
note "transcript: $TRANSCRIPT"
exit "$status"
