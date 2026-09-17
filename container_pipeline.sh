#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible in-container alias for the single ablation entry point.
echo "Starting the Matryoshka/CSR/MP-SAE ablation."
exec /app/run_all_experiments_container.sh "$@"
