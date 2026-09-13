#!/usr/bin/env bash
set -Eeuo pipefail

# Container-side entry point. The host-facing run_all_experiments_docker.sh
# builds one image and invokes this script once for the complete suite.

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Run every configured graduate-thesis experiment and create one result bundle.

Configuration is supplied through environment variables; start with
.env.example. Important variables are SUITE_EXPERIMENTS, DATA_PREP,
CSR_BACKBONES, CSR_EPOCHS, MMPOT_EPOCHS, and OUTPUT_ROOT.
EOF
    exit 0
fi

DATA_ROOT="${DATA_ROOT:-/data/huggingface}"
SUITE_ROOT="${SUITE_ROOT:-/output/all_experiments}"
OUTPUT_MOUNT="${OUTPUT_MOUNT:-/output}"
SUITE_EXPERIMENTS="${SUITE_EXPERIMENTS:-csr_vs_mpsae,mmpot_proxy,mmpot_true}"
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

# The default architecture ablation changes only the backbone between
# ResNet-18 and ResNet-50. Add swin_t explicitly for an extended study.
CSR_BACKBONES="${CSR_BACKBONES:-resnet18,resnet50}"
CSR_METHOD="${CSR_METHOD:-both}"
CSR_EPOCHS="${CSR_EPOCHS:-10}"
CSR_BATCH_SIZE="${CSR_BATCH_SIZE:-1024}"
CSR_FEATURE_BATCH_SIZE="${CSR_FEATURE_BATCH_SIZE:-512}"
CSR_MAX_TRAIN="${CSR_MAX_TRAIN:-0}"
CSR_MAX_VAL="${CSR_MAX_VAL:-0}"
CSR_HIDDEN_DIM="${CSR_HIDDEN_DIM:-0}"
CSR_TRAIN_K="${CSR_TRAIN_K:-32}"
CSR_TOPK="${CSR_TOPK:-8,16,32,64,128,256}"
CSR_LR="${CSR_LR:-4e-5}"
CSR_MRL_LR="${CSR_MRL_LR:-1e-2}"
CSR_MRL_MOMENTUM="${CSR_MRL_MOMENTUM:-0.9}"
CSR_WEIGHT_DECAY="${CSR_WEIGHT_DECAY:-1e-4}"
CSR_OT_MASS="${CSR_OT_MASS:-0.9}"
CSR_OT_ETA="${CSR_OT_ETA:-0.2}"
CSR_OT_ITERS="${CSR_OT_ITERS:-100}"
CSR_OT_MICROBATCH="${CSR_OT_MICROBATCH:-32}"
CSR_FAISS_GPU_DEVICE="${CSR_FAISS_GPU_DEVICE:-0}"
CSR_FAISS_TEMP_MEMORY_MIB="${CSR_FAISS_TEMP_MEMORY_MIB:-512}"
CSR_REBUILD_CACHE="${CSR_REBUILD_CACHE:-0}"
CSR_RESUME="${CSR_RESUME:-1}"
CSR_CACHE_DIR="${CSR_CACHE_DIR:-/data/csr_feature_cache}"
CSR_WEIGHTS_CACHE="${CSR_WEIGHTS_CACHE:-/data/torch}"

MMPOT_ARCHITECTURE="${MMPOT_ARCHITECTURE:-resnet50}"
MMPOT_METHOD="${MMPOT_METHOD:-both}"
MMPOT_EPOCHS="${MMPOT_EPOCHS:-5}"
MMPOT_BATCH_SIZE="${MMPOT_BATCH_SIZE:-256}"
MMPOT_BENCHMARKS="${MMPOT_BENCHMARKS:-head,linear}"
MMPOT_PROBE_EPOCHS="${MMPOT_PROBE_EPOCHS:-5}"
MMPOT_OPTIMIZERS="${MMPOT_OPTIMIZERS:-sgd}"
MMPOT_SGD_LR="${MMPOT_SGD_LR:-0.1}"
MMPOT_ADAM_LR="${MMPOT_ADAM_LR:-0.001}"
MMPOT_MOMENTUM="${MMPOT_MOMENTUM:-0.9}"
MMPOT_WEIGHT_DECAY="${MMPOT_WEIGHT_DECAY:-1e-4}"
MMPOT_OT_LAMBDA="${MMPOT_OT_LAMBDA:-0.5}"
MMPOT_OT_MASS="${MMPOT_OT_MASS:-0.8}"
MMPOT_OT_ETA="${MMPOT_OT_ETA:-0.1}"
MMPOT_OT_ITERS="${MMPOT_OT_ITERS:-50}"
MMPOT_OT_GRAD="${MMPOT_OT_GRAD:-envelope}"
MMPOT_OT_SOLVER_MODE="${MMPOT_OT_SOLVER_MODE:-cyclic}"
MMPOT_MAX_TRAIN_BATCHES="${MMPOT_MAX_TRAIN_BATCHES:-0}"
MMPOT_MAX_VAL_BATCHES="${MMPOT_MAX_VAL_BATCHES:-0}"
MMPOT_PRETRAINED="${MMPOT_PRETRAINED:-1}"
MMPOT_RESUME="${MMPOT_RESUME:-1}"

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

case "$SUITE_ROOT" in
    "$OUTPUT_MOUNT"|"$OUTPUT_MOUNT"/*) ;;
    *) die "SUITE_ROOT must be inside the persistent $OUTPUT_MOUNT mount" ;;
esac

if ! mountpoint -q "$OUTPUT_MOUNT" 2>/dev/null \
    && ! grep -qE "[[:space:]]${OUTPUT_MOUNT}[[:space:]]" /proc/self/mountinfo /proc/mounts 2>/dev/null; then
    die "$OUTPUT_MOUNT is not mounted; refusing to create disposable results"
fi

mkdir -p "$SUITE_ROOT/logs"
rm -f "$SUITE_ROOT/ALL_EXPERIMENTS_COMPLETE.json"

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
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("the image does not contain CUDA-enabled FAISS")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
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

IFS=',' read -r -a requested_experiments <<< "$SUITE_EXPERIMENTS"
experiments=()
for requested in "${requested_experiments[@]}"; do
    name="${requested//[[:space:]]/}"
    case "$name" in
        csr_vs_mpsae|mmpot_proxy|mmpot_true) experiments+=("$name") ;;
        "") ;;
        *) die "unsupported suite experiment '$name'" ;;
    esac
done
(( ${#experiments[@]} > 0 )) || die "SUITE_EXPERIMENTS is empty"

{
    echo "suite_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "experiments=$(IFS=,; echo "${experiments[*]}")"
    echo "dataset_root=$DATA_ROOT"
    echo "dataset_id=$HF_DATASET_ID"
    echo "dataset_revision=$HF_REVISION"
    echo "device=$DEVICE"
    echo "workers=$WORKERS"
    echo "prefetch_factor=$PREFETCH_FACTOR"
    echo "amp=$AMP"
    echo "channels_last=$CHANNELS_LAST"
    echo "csr_study=controlled_architecture_ablation"
    echo "csr_effect=mp_sae_top1_minus_matryoshka_top1"
    echo "csr_backbones=$CSR_BACKBONES"
    echo "csr_epochs=$CSR_EPOCHS"
    echo "csr_cache_dir=$CSR_CACHE_DIR"
    echo "mmpot_architecture=$MMPOT_ARCHITECTURE"
    echo "mmpot_epochs=$MMPOT_EPOCHS"
    echo "mmpot_optimizers=$MMPOT_OPTIMIZERS"
} > "$SUITE_ROOT/suite_run_manifest.txt"

run_csr() {
    local output="$SUITE_ROOT/csr_vs_mpsae"
    local amp_flag channels_flag rebuild_flag resume_flag
    amp_flag="$(bool_flag "$AMP" --amp --no-amp)"
    channels_flag="$(bool_flag "$CHANNELS_LAST" --channels-last --no-channels-last)"
    rebuild_flag="$(bool_flag "$CSR_REBUILD_CACHE" --rebuild-cache '')"
    resume_flag="$(bool_flag "$CSR_RESUME" --resume '')"
    [[ "$CSR_METHOD" == "both" ]] || die "CSR_METHOD must be 'both' for the architecture ablation"
    mkdir -p "$output"
    DATA_BACKEND=hf \
    BACKBONES="$CSR_BACKBONES" \
    CACHE_DIR="$CSR_CACHE_DIR" \
    OUTPUT_DIR="$output/results" \
    WEIGHTS_CACHE="$CSR_WEIGHTS_CACHE" \
    INSTALL_DEPS=0 \
    FAISS_GPU=1 \
    AGGREGATE_RESULTS=1 \
    /app/run_csr_vs_mmpot_imagenet.sh "$DATA_ROOT" \
        --hf-dataset-id "$HF_DATASET_ID" \
        --hf-revision "$HF_REVISION" \
        --method "$CSR_METHOD" \
        --epochs "$CSR_EPOCHS" \
        --batch-size "$CSR_BATCH_SIZE" \
        --feature-batch-size "$CSR_FEATURE_BATCH_SIZE" \
        --workers "$WORKERS" \
        --prefetch-factor "$PREFETCH_FACTOR" \
        --max-train "$CSR_MAX_TRAIN" \
        --max-val "$CSR_MAX_VAL" \
        --hidden-dim "$CSR_HIDDEN_DIM" \
        --train-k "$CSR_TRAIN_K" \
        --topk "$CSR_TOPK" \
        --lr "$CSR_LR" \
        --mrl-lr "$CSR_MRL_LR" \
        --mrl-momentum "$CSR_MRL_MOMENTUM" \
        --weight-decay "$CSR_WEIGHT_DECAY" \
        --ot-mass "$CSR_OT_MASS" \
        --ot-eta "$CSR_OT_ETA" \
        --ot-iters "$CSR_OT_ITERS" \
        --ot-microbatch "$CSR_OT_MICROBATCH" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --faiss-gpu-device "$CSR_FAISS_GPU_DEVICE" \
        --faiss-temp-memory-mib "$CSR_FAISS_TEMP_MEMORY_MIB" \
        "$amp_flag" "$channels_flag" \
        ${rebuild_flag:+"$rebuild_flag"} \
        ${resume_flag:+"$resume_flag"}
}

run_legacy() {
    local stage="$1" script="$2" marginals="$3"
    local stage_root="$SUITE_ROOT/$stage"
    local amp_flag channels_flag pretrained_flag resume_value
    amp_flag="$(bool_flag "$AMP" --amp --no-amp)"
    channels_flag="$(bool_flag "$CHANNELS_LAST" --channels-last '')"
    pretrained_flag="$(bool_flag "$MMPOT_PRETRAINED" --pretrained '')"
    resume_value=""
    [[ "$MMPOT_RESUME" == "1" ]] && resume_value=auto
    [[ "$MMPOT_RESUME" == "0" ]] || [[ "$MMPOT_RESUME" == "1" ]] \
        || die "MMPOT_RESUME must be 0 or 1"

    IFS=',' read -r -a optimizer_list <<< "$MMPOT_OPTIMIZERS"
    for optimizer_value in "${optimizer_list[@]}"; do
        optimizer="${optimizer_value//[[:space:]]/}"
        case "$optimizer" in
            sgd) optimizer_lr="$MMPOT_SGD_LR" ;;
            adam|adamw) optimizer_lr="$MMPOT_ADAM_LR" ;;
            *) die "unsupported MMPOT optimizer '$optimizer'" ;;
        esac
        output="$stage_root/$optimizer"
        mkdir -p "$output"
        command=(
            python "$script"
            --dataset imagenet
            --data-root "$DATA_ROOT"
            --hf-dataset-id "$HF_DATASET_ID"
            --hf-revision "$HF_REVISION"
            --require-validated-data
            --architecture "$MMPOT_ARCHITECTURE"
            --method "$MMPOT_METHOD"
            --epochs "$MMPOT_EPOCHS"
            --batch-size "$MMPOT_BATCH_SIZE"
            --workers "$WORKERS"
            --prefetch-factor "$PREFETCH_FACTOR"
            --optimizer "$optimizer"
            --lr "$optimizer_lr"
            --momentum "$MMPOT_MOMENTUM"
            --weight-decay "$MMPOT_WEIGHT_DECAY"
            --ot-lambda "$MMPOT_OT_LAMBDA"
            --ot-mass "$MMPOT_OT_MASS"
            --ot-eta "$MMPOT_OT_ETA"
            --ot-iters "$MMPOT_OT_ITERS"
            --ot-grad "$MMPOT_OT_GRAD"
            --benchmark "$MMPOT_BENCHMARKS"
            --probe-epochs "$MMPOT_PROBE_EPOCHS"
            --max-train-batches "$MMPOT_MAX_TRAIN_BATCHES"
            --max-val-batches "$MMPOT_MAX_VAL_BATCHES"
            --device "$DEVICE"
            --seed "$SEED"
            --output-dir "$output"
            "$amp_flag"
        )
        [[ -n "$channels_flag" ]] && command+=("$channels_flag")
        [[ -n "$pretrained_flag" ]] && command+=("$pretrained_flag")
        [[ -n "$resume_value" ]] && command+=(--resume "$resume_value")
        if [[ "$stage" == "mmpot_true" ]]; then
            command+=(
                --ot-marginals "$marginals"
                --ot-solver-mode "$MMPOT_OT_SOLVER_MODE"
            )
        fi
        "${command[@]}"
    done
}

for index in "${!experiments[@]}"; do
    experiment="${experiments[$index]}"
    log_path="$SUITE_ROOT/logs/${experiment}.log"
    echo "[$((index + 1))/${#experiments[@]}] Starting $experiment"
    case "$experiment" in
        csr_vs_mpsae) run_csr 2>&1 | tee -a "$log_path" ;;
        mmpot_proxy) run_legacy mmpot_proxy /app/matryoshka_mmpot_experiment.py 2 2>&1 | tee -a "$log_path" ;;
        mmpot_true) run_legacy mmpot_true /app/matryoshka_real_mmpot_experiment.py 3 2>&1 | tee -a "$log_path" ;;
    esac
done

experiment_csv="$(IFS=,; echo "${experiments[*]}")"
python /app/aggregate_all_experiments.py \
    "$SUITE_ROOT" \
    --experiments "$experiment_csv"

echo "All experiments completed. Portable bundle:"
echo "  $SUITE_ROOT/all_experiments_results.zip"
