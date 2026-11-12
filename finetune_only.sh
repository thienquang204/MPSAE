#!/usr/bin/env bash
# Nine additional fits only. Installation is explicit, never repeated by default.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash finetune_only.sh [--setup] [fine-tuning run arguments]
  --setup  Install/verify the server environment first (new server only).
Defaults: 16 loader workers, 64-image extraction, JPEG packs, resumable download,
and private HF backup. Existing checkpoints resume; frozen source fits must exist.
MPSAE_WORKERS, MPSAE_EXTRACT_BATCH, MPSAE_WORKSPACE override runtime defaults.
Other arguments pass to python -m mpsae.finetune run; --config selects the recipe.
HELP
    exit 0
fi
if [[ "${1:-}" == --setup ]]; then
    shift
    bash setup_server.sh
fi
source activate_server.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND=Agg
exec python -m mpsae.finetune run \
    --workspace "${MPSAE_WORKSPACE:-work}" \
    --workers "${MPSAE_WORKERS:-16}" --extract-batch "${MPSAE_EXTRACT_BATCH:-64}" \
    --image-storage packed --download-images --backup "$@"
