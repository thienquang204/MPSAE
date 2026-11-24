#!/usr/bin/env bash
# Shared CUDA 12.8 environment for frozen-feature retrieval and VOC patch probes.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v nvidia-smi >/dev/null || { echo 'A working NVIDIA driver is required.' >&2; exit 1; }
nvidia-smi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
ENV_DIR="$ROOT/.venv-retrieval"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    uv venv --python 3.11 --seed "$ENV_DIR"
fi
uv pip install --python "$ENV_DIR/bin/python" torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ENV_DIR/bin/python" -r requirements-patch-scene.txt
uv pip check --python "$ENV_DIR/bin/python"
"$ENV_DIR/bin/python" -m mpsae.patch_scene doctor --device cuda:0
echo 'Ready: bash scripts/run_ablation79.sh --workspace /workspace/retrieval-studies --download-embeddings'
