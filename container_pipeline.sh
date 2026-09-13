#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible in-container alias for older deployment commands.
echo "container_pipeline.sh is an alias for run_all_experiments_container.sh."
exec /app/run_all_experiments_container.sh "$@"
