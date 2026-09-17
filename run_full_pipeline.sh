#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible host alias for the single ablation launcher.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "Starting the Matryoshka/CSR/MP-SAE ablation."
exec bash "$SCRIPT_DIR/run_all_experiments_docker.sh" "$@"
