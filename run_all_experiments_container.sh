#!/usr/bin/env bash
set -Eeuo pipefail

# Matryoshka plus CSR/MPSAE v1 and v2 on ResNet-18/50.
DATA_ROOT="${DATA_ROOT:-/data/huggingface}"
OUTPUT_MOUNT="${OUTPUT_MOUNT:-/output}"
SUITE_ROOT="${SUITE_ROOT:-$OUTPUT_MOUNT/three_method_ablation}"
DATA_PREP="${DATA_PREP:-auto}"
IMAGENET_SPLITS="${IMAGENET_SPLITS:-train,validation}"
HF_DATASET_ID="${HF_DATASET_ID:-ILSVRC/imagenet-1k}"
HF_REVISION="${HF_REVISION:-main}"
WORKERS="${WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
AMP="${AMP:-1}"
CHANNELS_LAST="${CHANNELS_LAST:-1}"
HOST_UID="${HOST_UID:-0}"
HOST_GID="${HOST_GID:-0}"

# The experiment matrix is intentionally fixed to both requested architectures
# and all five arms. Normal resource/hyperparameter controls remain in .env.
BACKBONES="resnet18,resnet50"
BASE_EPOCHS="${ABLATION_EPOCHS:-10}"
MPSAEV2_EXTRA_EPOCHS="${MPSAEV2_EXTRA_EPOCHS:-${MPSAE_EXTRA_EPOCHS:-4}}"
BATCH_SIZE="${ABLATION_BATCH_SIZE:-1024}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-512}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_VAL="${MAX_VAL:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-8196}"
TRAIN_K="${TRAIN_K:-2}"
V1_TRAIN_K="${V1_TRAIN_K:-32}"
ANNEAL_START_K="${ANNEAL_START_K:-64}"
ANNEAL_FRACTION="${ANNEAL_FRACTION:-0.7}"
SPARSE_EXTRA_TOPK="${SPARSE_EXTRA_TOPK:-1,2,4}"
SPARSE_KNN_QUERY_BATCH="${SPARSE_KNN_QUERY_BATCH:-32}"
MRL_CLASSIFICATION_WEIGHT="${MRL_CLASSIFICATION_WEIGHT:-1.0}"
CSRV2_MAIN_RECON_WEIGHT="${CSRV2_MAIN_RECON_WEIGHT:-${CSR_MAIN_RECON_WEIGHT:-1.0}}"
CSRV2_MULTI_TOPK_RECON_WEIGHT="${CSRV2_MULTI_TOPK_RECON_WEIGHT:-${CSR_MULTI_TOPK_RECON_WEIGHT:-0.125}}"
CSRV2_AUX_RECON_WEIGHT="${CSRV2_AUX_RECON_WEIGHT:-${CSR_AUX_RECON_WEIGHT:-0.03125}}"
CSRV2_CONTRASTIVE_WEIGHT="${CSRV2_CONTRASTIVE_WEIGHT:-${CSR_CONTRASTIVE_WEIGHT:-1.0}}"
MPSAEV2_MAIN_RECON_WEIGHT="${MPSAEV2_MAIN_RECON_WEIGHT:-${MPSAE_MAIN_RECON_WEIGHT:-1.0}}"
MPSAEV2_NESTED_RECON_WEIGHT="${MPSAEV2_NESTED_RECON_WEIGHT:-${MPSAE_NESTED_RECON_WEIGHT:-0.125}}"
MPSAEV2_AUX_RECON_WEIGHT="${MPSAEV2_AUX_RECON_WEIGHT:-${MPSAE_AUX_RECON_WEIGHT:-0.03125}}"
MPSAEV2_MMPOT_WEIGHT="${MPSAEV2_MMPOT_WEIGHT:-${MPSAE_MMPOT_WEIGHT:-1.3}}"
MPSAEV2_LR="${MPSAEV2_LR:-${MPSAE_LR:-4e-5}}"
CSRV2_LR="${CSRV2_LR:-${CSR_LR:-1e-4}}"
MRL_LR="${MRL_LR:-1e-2}"
MRL_MOMENTUM="${MRL_MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
OT_MASS="${OT_MASS:-0.9}"
OT_ETA="${OT_ETA:-0.2}"
OT_ITERS="${OT_ITERS:-100}"
OT_MICROBATCH="${OT_MICROBATCH:-32}"
FAISS_GPU_DEVICE="${FAISS_GPU_DEVICE:-0}"
FAISS_TEMP_MEMORY_MIB="${FAISS_TEMP_MEMORY_MIB:-512}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"
FEATURE_CACHE="${FEATURE_CACHE:-/data/three_method_feature_cache}"
WEIGHTS_CACHE="${WEIGHTS_CACHE:-/data/torch}"

die() {
    echo "Error: $*" >&2
    exit 1
}

bool_flag() {
    local value="$1" enabled="$2" disabled="$3"
    case "$value" in
        1) printf '%s' "$enabled" ;;
        0) printf '%s' "$disabled" ;;
        *) die "expected 0 or 1, got '$value'" ;;
    esac
}

[[ "$MPSAEV2_EXTRA_EPOCHS" == "4" ]] \
    || die "MPSAEV2_EXTRA_EPOCHS is fixed at 4 for this ablation"
case "$SUITE_ROOT" in
    "$OUTPUT_MOUNT"|"$OUTPUT_MOUNT"/*) ;;
    *) die "SUITE_ROOT must be inside the persistent $OUTPUT_MOUNT mount" ;;
esac
if ! mountpoint -q "$OUTPUT_MOUNT" 2>/dev/null \
    && ! grep -qE "[[:space:]]${OUTPUT_MOUNT}[[:space:]]" /proc/self/mountinfo /proc/mounts 2>/dev/null; then
    die "$OUTPUT_MOUNT is not mounted; refusing to create disposable results"
fi

mkdir -p "$SUITE_ROOT/logs"
rm -f "$SUITE_ROOT/RUN_COMPLETE.json"

handoff_output() {
    local status=$?
    if [[ "$HOST_UID" != "0" || "$HOST_GID" != "0" ]]; then
        chown -R "${HOST_UID}:${HOST_GID}" "$SUITE_ROOT" 2>/dev/null || true
    fi
    chmod -R u+rwX,go+rX "$SUITE_ROOT" 2>/dev/null || true
    return "$status"
}
trap handoff_output EXIT

python - <<'PY'
import torch
import faiss

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; start Docker with NVIDIA GPU access")
print(f"CUDA training ready: {torch.cuda.get_device_name(0)}")
print("CPU FAISS retrieval ready")
PY

dataset_ready() {
    python - "$DATA_ROOT" "$IMAGENET_SPLITS" "$HF_DATASET_ID" "$HF_REVISION" <<'PY'
import json
import sys
from pathlib import Path

root, split_csv, dataset_id, revision = sys.argv[1:]
try:
    manifest = json.loads((Path(root) / "imagenet_download_manifest.json").read_text())
    requested = {item.strip() for item in split_csv.split(",") if item.strip()}
    splits = manifest.get("splits", {})
    valid = (
        manifest.get("validation_passed") is True
        and manifest.get("dataset_id") == dataset_id
        and manifest.get("revision") == revision
        and requested <= set(splits)
        and all(
            splits[name].get("validation_status") == "passed"
            and splits[name].get("fully_checked") is True
            for name in requested
        )
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

case "$DATA_PREP" in
    auto)
        if ! dataset_ready; then
            python /app/download_imagenet.py \
                --cache-dir "$DATA_ROOT" \
                --dataset-id "$HF_DATASET_ID" \
                --revision "$HF_REVISION" \
                --splits "$IMAGENET_SPLITS" \
                --check-samples all
        fi
        ;;
    always)
        python /app/download_imagenet.py \
            --cache-dir "$DATA_ROOT" \
            --dataset-id "$HF_DATASET_ID" \
            --revision "$HF_REVISION" \
            --splits "$IMAGENET_SPLITS" \
            --check-samples all
        ;;
    never) dataset_ready || die "DATA_PREP=never but the ImageNet cache is not validated" ;;
    *) die "DATA_PREP must be auto, always, or never" ;;
esac
dataset_ready || die "ImageNet preparation did not produce a valid manifest"

{
    echo "study=matryoshka_csr_mpsae_v1_v2_architecture_ablation"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "backbones=$BACKBONES"
    echo "methods=matryoshka,csr,mpsae,csrv2,mpsaev2"
    echo "base_epochs=$BASE_EPOCHS"
    echo "mpsaev2_extra_epochs=$MPSAEV2_EXTRA_EPOCHS"
    echo "anneal_start_k=$ANNEAL_START_K"
    echo "anneal_target_k=$TRAIN_K"
    echo "v1_fixed_train_k=$V1_TRAIN_K"
    echo "anneal_fraction=$ANNEAL_FRACTION"
    echo "resnet18_evaluation_budgets=8,16,32,64,128,256,512"
    echo "resnet50_evaluation_budgets=8,16,32,64,128,256,512,1024,2048"
    echo "sparse_extra_evaluation_budgets=$SPARSE_EXTRA_TOPK"
    echo "retrieval=unit_normalized_exact_l2"
    echo "mrl_classification_weight=$MRL_CLASSIFICATION_WEIGHT"
    echo "csrv2_main_recon_weight=$CSRV2_MAIN_RECON_WEIGHT"
    echo "csrv2_multi_topk_recon_weight=$CSRV2_MULTI_TOPK_RECON_WEIGHT"
    echo "csrv2_aux_recon_weight=$CSRV2_AUX_RECON_WEIGHT"
    echo "csrv2_contrastive_weight=$CSRV2_CONTRASTIVE_WEIGHT"
    echo "mpsaev2_main_recon_weight=$MPSAEV2_MAIN_RECON_WEIGHT"
    echo "mpsaev2_nested_recon_weight=$MPSAEV2_NESTED_RECON_WEIGHT"
    echo "mpsaev2_aux_recon_weight=$MPSAEV2_AUX_RECON_WEIGHT"
    echo "mpsaev2_mmpot_weight=$MPSAEV2_MMPOT_WEIGHT"
    echo "model_weights_saved=false"
    echo "dataset_root=$DATA_ROOT"
    echo "dataset_id=$HF_DATASET_ID"
    echo "dataset_revision=$HF_REVISION"
    echo "seed=$SEED"
} > "$SUITE_ROOT/ablation_run_manifest.txt"

amp_flag="$(bool_flag "$AMP" --amp --no-amp)"
channels_flag="$(bool_flag "$CHANNELS_LAST" --channels-last --no-channels-last)"
rebuild_flag="$(bool_flag "$REBUILD_CACHE" --rebuild-cache '')"

DATA_BACKEND=hf \
BACKBONES="$BACKBONES" \
CACHE_DIR="$FEATURE_CACHE" \
OUTPUT_DIR="$SUITE_ROOT" \
WEIGHTS_CACHE="$WEIGHTS_CACHE" \
INSTALL_DEPS=0 \
FAISS_GPU=0 \
AGGREGATE_RESULTS=1 \
/app/run_csr_vs_mmpot_imagenet.sh "$DATA_ROOT" \
    --hf-dataset-id "$HF_DATASET_ID" \
    --hf-revision "$HF_REVISION" \
    --method all \
    --epochs "$BASE_EPOCHS" \
    --mpsaev2-extra-epochs "$MPSAEV2_EXTRA_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --feature-batch-size "$FEATURE_BATCH_SIZE" \
    --workers "$WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --max-train "$MAX_TRAIN" \
    --max-val "$MAX_VAL" \
    --hidden-dim "$HIDDEN_DIM" \
    --train-k "$TRAIN_K" \
    --v1-train-k "$V1_TRAIN_K" \
    --anneal-start-k "$ANNEAL_START_K" \
    --anneal-fraction "$ANNEAL_FRACTION" \
    --sparse-extra-topk "$SPARSE_EXTRA_TOPK" \
    --sparse-knn-query-batch "$SPARSE_KNN_QUERY_BATCH" \
    --mrl-classification-weight "$MRL_CLASSIFICATION_WEIGHT" \
    --csrv2-main-recon-weight "$CSRV2_MAIN_RECON_WEIGHT" \
    --csrv2-multi-topk-recon-weight "$CSRV2_MULTI_TOPK_RECON_WEIGHT" \
    --csrv2-aux-recon-weight "$CSRV2_AUX_RECON_WEIGHT" \
    --csrv2-contrastive-weight "$CSRV2_CONTRASTIVE_WEIGHT" \
    --mpsaev2-main-recon-weight "$MPSAEV2_MAIN_RECON_WEIGHT" \
    --mpsaev2-nested-recon-weight "$MPSAEV2_NESTED_RECON_WEIGHT" \
    --mpsaev2-aux-recon-weight "$MPSAEV2_AUX_RECON_WEIGHT" \
    --mpsaev2-mmpot-weight "$MPSAEV2_MMPOT_WEIGHT" \
    --mpsaev2-lr "$MPSAEV2_LR" \
    --csrv2-lr "$CSRV2_LR" \
    --mrl-lr "$MRL_LR" \
    --mrl-momentum "$MRL_MOMENTUM" \
    --weight-decay "$WEIGHT_DECAY" \
    --ot-mass "$OT_MASS" \
    --ot-eta "$OT_ETA" \
    --ot-iters "$OT_ITERS" \
    --ot-microbatch "$OT_MICROBATCH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --faiss-gpu-device "$FAISS_GPU_DEVICE" \
    --faiss-temp-memory-mib "$FAISS_TEMP_MEMORY_MIB" \
    "$amp_flag" "$channels_flag" \
    ${rebuild_flag:+"$rebuild_flag"} \
    2>&1 | tee -a "$SUITE_ROOT/logs/three_method_ablation.log"

echo "Five-arm ablation complete: $SUITE_ROOT"
echo "Portable results: $SUITE_ROOT/imagenet_architecture_ablation_results.zip"
