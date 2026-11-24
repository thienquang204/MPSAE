#!/usr/bin/env bash
# Run from a cloned repository; no tmux and no Hugging Face uploads.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${MPSAE_PYTHON:-$ROOT/.venv-retrieval/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo 'Run bash scripts/setup_ablation79.sh first, or set MPSAE_PYTHON to your Python executable.' >&2
    exit 1
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec "$PYTHON" -u -m mpsae.benchmark.ablation79 "$@"
