#!/usr/bin/env bash
# Local-only ImageNet MPSAE mass ablation. No tmux or result publishing.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
if [[ -n "${MPSAE_PYTHON:-}" ]]; then
    PYTHON_BIN="$MPSAE_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    PYTHON_BIN=python
fi
exec "$PYTHON_BIN" -u -m mpsae.benchmark.mass_sweep "$@"
