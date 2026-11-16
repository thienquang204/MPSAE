#!/usr/bin/env bash
# One command: install on first use, then run the local segmentation grid.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-segmentation/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    bash "$ROOT/scripts/setup_segmentation_server.sh"
fi
exec "$PYTHON" -u "$ROOT/scripts/run_segmentation_server.py" \
    --workspace /workspace/segmentation \
    --datasets nyuv2 cityscapes \
    --ot-weights 0.5 1.0 2.0 "$@"
