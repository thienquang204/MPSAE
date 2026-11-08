#!/usr/bin/env bash
# Migration helper for a full server. No Torch/NumPy/CUDA installation required.
# Only removes the released FFCV derivatives whose original JPEG split is whole.
# Stop any old FFCV job before running this command.
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec python3 -m mpsae.benchmark.storage "$@"
