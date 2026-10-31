#!/usr/bin/env bash
set -Eeuo pipefail

# Standalone launcher, also called by the existing all-experiments entry point.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRANSFER_DATA_ROOT="${TRANSFER_DATA_ROOT:-$SCRIPT_DIR/data/transfer}"
TRANSFER_CACHE="${TRANSFER_CACHE:-$SCRIPT_DIR/data/transfer_cache}"
TRANSFER_OUTPUT="${TRANSFER_OUTPUT:-$SCRIPT_DIR/runs/frozen_transfer}"
TRANSFER_IMAGENET_ROOT="${TRANSFER_IMAGENET_ROOT:-$SCRIPT_DIR/data/imagenet1k}"

args=(
    --data-root "$TRANSFER_DATA_ROOT"
    --cache-dir "$TRANSFER_CACHE"
    --output-dir "$TRANSFER_OUTPUT"
    --imagenet-root "$TRANSFER_IMAGENET_ROOT"
    --imagenet-backend "${TRANSFER_IMAGENET_BACKEND:-hf}"
    --imagenet-dataset-id "${TRANSFER_IMAGENET_DATASET_ID:-ILSVRC/imagenet-1k}"
    --imagenet-revision "${TRANSFER_IMAGENET_REVISION:-main}"
    --tasks "${TRANSFER_TASKS:-detection,mask,pose}"
    --architectures "${TRANSFER_ARCHITECTURES:-resnet18,resnet50}"
    --methods "${TRANSFER_METHODS:-matryoshka,csr,mpsae,csrv2,mpsaev2}"
    --seeds "${TRANSFER_SEEDS:-0,1,2}"
    --device "${DEVICE:-cuda}"
    --workers "${TRANSFER_WORKERS:-4}"
    --feature-batch-size "${TRANSFER_FEATURE_BATCH_SIZE:-64}"
    --encode-batch-size "${TRANSFER_ENCODE_BATCH_SIZE:-1024}"
    --log-every "${TRANSFER_LOG_EVERY:-25}"
    --ot-microbatch "${OT_MICROBATCH:-32}"
    --mpsaev2-extra-epochs "${MPSAEV2_EXTRA_EPOCHS:-0}"
    --proposal-budget "${TRANSFER_PROPOSAL_BUDGET:-256}"
)
case "${TRANSFER_DOWNLOAD:-1}" in
    1) args+=(--download) ;;
    0) args+=(--no-download) ;;
    *) echo "TRANSFER_DOWNLOAD must be 0 or 1" >&2; exit 2 ;;
esac
case "${TRANSFER_VERIFY_IMAGES:-1}" in
    1) args+=(--data-verify-images) ;;
    0) args+=(--no-data-verify-images) ;;
    *) echo "TRANSFER_VERIFY_IMAGES must be 0 or 1" >&2; exit 2 ;;
esac
[[ -z "${TRANSFER_EDGE_MODEL:-}" ]] || args+=(--edge-model "$TRANSFER_EDGE_MODEL")
[[ -z "${TRANSFER_EDGE_MODEL_URL:-}" ]] || args+=(--edge-model-url "$TRANSFER_EDGE_MODEL_URL")
[[ -z "${TRANSFER_PASCAL3D_ROOT:-}" ]] || args+=(--pascal3d-root "$TRANSFER_PASCAL3D_ROOT")
[[ -z "${TRANSFER_PASCAL3D_URL:-}" ]] || args+=(--pascal3d-url "$TRANSFER_PASCAL3D_URL")
[[ -z "${TRANSFER_VOC2007_TRAINVAL_URL:-}" ]] || args+=(--voc2007-trainval-url "$TRANSFER_VOC2007_TRAINVAL_URL")
[[ -z "${TRANSFER_VOC2007_TEST_URL:-}" ]] || args+=(--voc2007-test-url "$TRANSFER_VOC2007_TEST_URL")
[[ -z "${TRANSFER_VOC2012_URL:-}" ]] || args+=(--voc2012-url "$TRANSFER_VOC2012_URL")
for argument in "$@"; do
    if [[ "$argument" == "--plan" ]]; then
        exec "$PYTHON_BIN" "$SCRIPT_DIR/transfer_experiments.py" "${args[@]}" "$@"
    fi
done
mkdir -p "$TRANSFER_OUTPUT/logs"
"$PYTHON_BIN" -u "$SCRIPT_DIR/transfer_experiments.py" "${args[@]}" "$@" \
    2>&1 | tee -a "$TRANSFER_OUTPUT/logs/pipeline.log"
