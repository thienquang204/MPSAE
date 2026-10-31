#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible in-container alias for the comparison entry point.
echo "Starting the configured ImageNet and frozen-transfer experiment suite."
exec /app/run_all_experiments_container.sh "$@"
