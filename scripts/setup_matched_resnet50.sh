#!/usr/bin/env bash
# Cached features only; no image decoder, FFCV, torchvision, or result uploads.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
command -v nvidia-smi >/dev/null || { echo 'A working NVIDIA driver is required.' >&2; exit 1; }
nvidia-smi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
ENV_DIR="$REPO_ROOT/.venv-retrieval"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    uv venv --python 3.11 --seed "$ENV_DIR"
fi
uv pip install --python "$ENV_DIR/bin/python" torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ENV_DIR/bin/python" -r requirements-retrieval.txt
uv pip check --python "$ENV_DIR/bin/python"
"$ENV_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
echo 'Ready: bash scripts/run_matched_resnet50.sh --download-embeddings'
