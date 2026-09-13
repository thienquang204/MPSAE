#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible host alias. New automation should use the explicit name.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "run_full_pipeline.sh is an alias for run_all_experiments_docker.sh."
exec bash "$SCRIPT_DIR/run_all_experiments_docker.sh" "$@"
