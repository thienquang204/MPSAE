#!/usr/bin/env bash
# Native NVIDIA: install, run the configured experiments, back up to HF MPSAE.
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SKIP_SETUP=0
SETUP_ONLY=0
SKIP_EMBEDDINGS=0
SMOKE=0
HAVE_CONFIG=0
COMMAND=run
ARGS=()
SETUP_ARGS=()
while (( $# )); do
    case "$1" in
        --skip-setup) SKIP_SETUP=1; shift ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        --sync-only) COMMAND=sync; SKIP_SETUP=1; shift ;;
        --check) COMMAND=check; SKIP_SETUP=1; shift ;;
        --skip-embeddings) SKIP_EMBEDDINGS=1; shift ;;
        --cuda|--python)
            (( $# >= 2 )) || { printf '%s requires a value\n' "$1" >&2; exit 2; }
            SETUP_ARGS+=("$1" "$2"); shift 2 ;;
        --skip-system-packages) SETUP_ARGS+=("$1"); shift ;;
        --config|--env-file|--workspace|--device|--workers|--embed-batch-size)
            (( $# >= 2 )) || { printf '%s requires a value\n' "$1" >&2; exit 2; }
            [[ "$1" == --config ]] && HAVE_CONFIG=1
            ARGS+=("$1" "$2"); shift 2 ;;
        --smoke) SMOKE=1; ARGS+=("--smoke" "--config" "configs/smoke.yaml"); shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash all_in_one.sh [options]
  Default: detect GPU/CPU, install compatible dependencies, run thesis.yaml,
           upload private HF MPSAE
           artifacts and log to W&B project cvpr when WANDB_TOKEN is supplied.
  --skip-setup   Reuse the installed environment and resume the campaign.
  --setup-only   Install and verify the environment, then stop.
  --sync-only    Upload existing results without training.
  --check        Check/create the private repo and check W&B credentials.
  --smoke        Synthetic rehearsal; no remote uploads or W&B writes.
  --skip-embeddings
                 Do not run the embeddings step first (python -m mpsae.embeddings_hub
                 sync: pull each <dataset>_<backbone>_<dim> folder from the private
                 HF repo MPSAE-embeddings, or build and upload it once).
  --cuda auto|13.0|12.6|12.8|12.9 --python PATH --skip-system-packages
                 Forwarded to setup_server.sh for flexible installation.
  --config PATH --env-file PATH --workspace PATH --device cuda:0
Fill HF_TOKEN (write access) and optionally WANDB_TOKEN in .env before running.
HELP
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ "$SKIP_SETUP" == 1 && "${#SETUP_ARGS[@]}" != 0 ]]; then
    printf 'Installer options cannot be combined with --skip-setup, --sync-only or --check\n' >&2
    exit 2
fi
if [[ "$SETUP_ONLY" == 1 && "$SKIP_SETUP" == 1 ]]; then
    printf '%s\n' '--setup-only cannot be combined with --skip-setup, --sync-only or --check' >&2
    exit 2
fi
if [[ "$SKIP_SETUP" == 0 ]]; then
    bash "$HERE/setup_server.sh" "${SETUP_ARGS[@]}"
fi
[[ "$SETUP_ONLY" == 0 ]] || exit 0
# shellcheck disable=SC1091
source "$HERE/activate_server.sh"
if [[ "$COMMAND" == run && "$SMOKE" == 0 && "$SKIP_EMBEDDINGS" == 0 ]]; then
    # First step: every embedding the campaign reads is present locally, pulled
    # from the HF embeddings repo or built once, and linked where the campaign
    # looks, so the campaign does not download or extract it again.
    EMBED_ARGS=("${ARGS[@]}")
    [[ "$HAVE_CONFIG" == 1 ]] || EMBED_ARGS+=(--config configs/thesis.yaml)
    "$HERE/.venv/bin/python" -m mpsae.embeddings_hub sync "${EMBED_ARGS[@]}"
fi
exec "$HERE/.venv/bin/python" -m mpsae.remote "$COMMAND" "${ARGS[@]}"
