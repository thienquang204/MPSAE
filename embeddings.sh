#!/usr/bin/env bash
# Embeddings only: make every <dataset>_<backbone>_<dim> folder present locally,
# pulling it from the private HF repository or building it once (download ->
# embed -> upload + verify -> delete raw). See mpsae/embeddings_hub.py.
#
#   bash embeddings.sh status
#   bash embeddings.sh sync
#   bash embeddings.sh sync --datasets imagenet,sop --backbones ff2048,rn18
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [[ -f "$HERE/activate_server.sh" && -x "$HERE/.venv/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "$HERE/activate_server.sh"
    PYTHON="$HERE/.venv/bin/python"
else
    PYTHON="${MPSAE_PYTHON:-python3}"
fi
exec "$PYTHON" -m mpsae.embeddings_hub "${@:-status}"
