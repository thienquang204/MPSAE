#!/usr/bin/env bash
set -Eeuo pipefail

# Host alias; the default scope is the ImageNet-only MMRSAE workflow.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "Starting configured scope (default: MMRSAE ImageNet only; environment overrides .env)."
exec bash "$SCRIPT_DIR/run_all_experiments_docker.sh" "$@"
