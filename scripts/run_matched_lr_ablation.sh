#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${MPSAE_PYTHON:-$REPO_ROOT/.venv-retrieval/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
exec "$PYTHON_BIN" -u -m mpsae.benchmark.lr_ablation "$@"
