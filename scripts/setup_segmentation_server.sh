#!/usr/bin/env bash
# Ubuntu/Linux + NVIDIA driver. Installs once; no training or artifact uploads.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v nvidia-smi >/dev/null || { echo 'Install/select an NVIDIA-driver-enabled server image first.' >&2; exit 1; }
nvidia-smi
if ! command -v uv >/dev/null; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        command -v curl >/dev/null || { echo 'Install curl first.' >&2; exit 1; }
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
ENV_DIR="$ROOT/.venv-segmentation"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    uv venv --python 3.11 --seed "$ENV_DIR"
fi
PYTHON="$ENV_DIR/bin/python"
uv pip install --python "$PYTHON" torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$PYTHON" -r requirements-patch-scene.txt
uv pip check --python "$PYTHON"
"$PYTHON" -m mpsae.patch_scene doctor --device cuda:0
printf '\nReady. Run:\n  %s scripts/run_segmentation_server.py --workspace /workspace/segmentation\n' "$PYTHON"
