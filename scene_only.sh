#!/usr/bin/env bash
# Fresh-server entry point. Always invokes scene bootstrap, never SAE training.
set -Eeuo pipefail
scene_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$scene_root"
scene_venv=.venv-scene
scene_setup=0
scene_setup_only=0
scene_python=python3
scene_cuda=cu128
scene_args=()
while (( $# )); do
    case "$1" in
        --setup) scene_setup=1; shift ;;
        --setup-only) scene_setup=1; scene_setup_only=1; shift ;;
        --python|--venv|--cuda)
            (( $# >= 2 )) || { printf 'error: %s needs a value\n' "$1" >&2; exit 2; }
            case "$1" in --python) scene_python="$2";; --venv) scene_venv="$2";; --cuda) scene_cuda="$2";; esac
            shift 2 ;;
        --help|-h)
            cat <<'HELP'
Usage: bash scene_only.sh --setup [evaluation options]
  --setup          Install the isolated evaluation environment, then run.
  --setup-only     Install/verify only; no checkpoints or datasets downloaded.
  --python PATH    Python 3.10-3.12 for setup (default python3).
  --venv PATH      Environment to use (default .venv-scene).
  --cuda FLAVOR    cu128 (default, RTX 5090 supported) or cu130.
Evaluation options are passed to python -m mpsae.scene bootstrap:
  --config configs/scene_voc.yaml --workspace /workspace/scene-v1
  --repo quangnguyenopt/MPSAE --revision main --prefix direct-v1
  --device cuda:0 --env-file .env --refresh-source --publish
Rerun without --setup to resume. HF_TOKEN or an existing HF login is supported.
HELP
            exit 0 ;;
        *) scene_args+=("$1"); shift ;;
    esac
done
if (( scene_setup )); then
    bash "$scene_root/setup_scene_server.sh" --python "$scene_python" --venv "$scene_venv" --cuda "$scene_cuda"
fi
(( scene_setup_only == 0 )) || exit 0
[[ -x "$scene_venv/bin/python" ]] || { printf 'Run first: bash scene_only.sh --setup\n' >&2; exit 1; }
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exec "$scene_venv/bin/python" -m mpsae.scene bootstrap "${scene_args[@]}"
