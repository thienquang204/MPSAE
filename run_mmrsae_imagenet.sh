#!/usr/bin/env bash
set -Eeuo pipefail

# One manually configured MMRSAE fit per selected ImageNet backbone.
# This direct launcher also accepts an explicit ENV_FILE; the Docker
# launcher loads .env by default. Explicit CLI options appear last and override
# environment-derived values. No eval/string-built argument expansion.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_shell_common.sh"
load_experiment_env
select_experiment_python
wandb_mode="${WANDB_MODE:-}"
if [[ -z "$wandb_mode" ]]; then
    if [[ -n "${WANDB_API_KEY:-}" ]]; then wandb_mode=online; else wandb_mode=offline; fi
fi

args=(
    --backbones "${MMRSAE_BACKBONES:-resnet50,resnet18}"
    --k "${MMRSAE_K:-8}"
    --epochs "${MMRSAE_EPOCHS:-10}"
    --max-steps "${MMRSAE_MAX_STEPS:-500}"
    --batch-size "${MMRSAE_BATCH_SIZE:-1024}"
    --train-size "${MMRSAE_TRAIN_SIZE:-128000}"
    --dev-size "${MMRSAE_DEV_SIZE:-10000}"
    --anchor-count "${MMRSAE_ANCHOR_COUNT:-1024}"
    --evaluation "${MMRSAE_EVALUATION:-dev}"
    --seed "${MMRSAE_SEED:-1227}"
    --data-seed "${MMRSAE_DATA_SEED:-42}"
    --device "${MMRSAE_DEVICE:-cuda:0}"
    --data-backend "${MMRSAE_DATA_BACKEND:-hf}"
    --data-root "${MMRSAE_DATA_ROOT:-$SCRIPT_DIR/data/imagenet1k}"
    --cache-dir "${MMRSAE_CACHE:-$SCRIPT_DIR/data/mmrsae}"
    --output-dir "${MMRSAE_OUTPUT:-$SCRIPT_DIR/runs/mmrsae}"
    --wandb-mode "$wandb_mode"
    --wandb-project "${WANDB_PROJECT:-MMRSAE}"
)

# Unset optional settings use the runner defaults. Each variable maps to one
# argument, preserving spaces in paths and names without evaluating text.
optional_options=(
    MMRSAE_RUN_NAME:--run-name MMRSAE_HIDDEN_DIM:--hidden-dim
    MMRSAE_K_AUX:--k-aux MMRSAE_DEAD_STEPS:--dead-steps
    MMRSAE_LR:--lr MMRSAE_WEIGHT_DECAY:--weight-decay MMRSAE_ADAM_EPS:--adam-eps
    MMRSAE_RELATION_WEIGHT:--relation-weight MMRSAE_OT_WEIGHT:--ot-weight
    MMRSAE_MIXTURE_WEIGHT:--mixture-weight MMRSAE_REPAIR_RANK:--repair-rank
    MMRSAE_CLEAN_PROBABILITY:--clean-probability MMRSAE_ERASE_PROBABILITY:--erase-probability
    MMRSAE_TEACHER_TEMPERATURE:--teacher-temperature MMRSAE_STUDENT_TEMPERATURE:--student-temperature
    MMRSAE_BANK_REFRESH:--bank-refresh MMRSAE_CHECKPOINT_EVERY:--checkpoint-every
    MMRSAE_LOG_EVERY:--log-every MMRSAE_EVAL_EVERY:--eval-every
    MMRSAE_OT_MASS:--ot-mass MMRSAE_OT_ETA:--ot-eta MMRSAE_OT_ITERS:--ot-iters
    MMRSAE_OT_MAX_REFINE_ITERS:--ot-max-refine-iters MMRSAE_OT_TOL:--ot-tol
    MMRSAE_OT_MICROBATCH:--ot-microbatch MMRSAE_FEATURE_BATCH_SIZE:--feature-batch-size
    MMRSAE_CONTRASTIVE_WEIGHT:--contrastive-weight MMRSAE_CONTRASTIVE_TEMPERATURE:--contrastive-temperature
    MMRSAE_WORKERS:--workers MMRSAE_ENCODE_BATCH_SIZE:--encode-batch-size
    MMRSAE_QUERY_BATCH_SIZE:--query-batch-size MMRSAE_GALLERY_CHUNK_SIZE:--gallery-chunk-size
    MMRSAE_SCORE_BUFFER_MIB:--score-buffer-mib MMRSAE_GPU_MEMORY_FRACTION:--gpu-memory-fraction
    MMRSAE_FAISS_TEMP_MEMORY_MIB:--faiss-temp-memory-mib
    MMRSAE_RETRIEVAL_WARMUP:--retrieval-warmup MMRSAE_RETRIEVAL_REPEATS:--retrieval-repeats
    MMRSAE_MAX_VAL:--max-val
    MMRSAE_FF2048_TRAIN:--ff2048-train MMRSAE_FF2048_VAL:--ff2048-val
    MMRSAE_FF2048_REVISION:--ff2048-revision
    MMRSAE_FF2048_TRAIN_SHA256:--ff2048-train-sha256 MMRSAE_FF2048_VAL_SHA256:--ff2048-val-sha256
    MMRSAE_IMAGENET_REVISION:--imagenet-revision MMRSAE_PARENT_REVISION:--parent-revision
    WANDB_ENTITY:--wandb-entity WANDB_GROUP:--wandb-group
    WANDB_TAGS:--wandb-tags WANDB_NAME:--wandb-run-name
)
for mapping in "${optional_options[@]}"; do
    variable="${mapping%%:*}"
    option="${mapping#*:}"
    if [[ -n "${!variable:-}" ]]; then args+=("$option" "${!variable}"); fi
done

case "${MMRSAE_DOWNLOAD:-1}" in
    1) args+=(--download) ;;
    0) args+=(--no-download) ;;
    *) echo "MMRSAE_DOWNLOAD must be 0 or 1" >&2; exit 2 ;;
esac
case "${MMRSAE_MASK:-}" in
    1) args+=(--mask) ;;
    0) args+=(--no-mask) ;;
    '') ;;
    *) echo "MMRSAE_MASK must be empty, 0, or 1" >&2; exit 2 ;;
esac
case "${MMRSAE_TIMED_RETRIEVAL:-1}" in
    1) args+=(--timed-retrieval) ;;
    0) args+=(--no-timed-retrieval) ;;
    *) echo "MMRSAE_TIMED_RETRIEVAL must be 0 or 1" >&2; exit 2 ;;
esac
case "${MMRSAE_DENSE_REFERENCE:-0}" in
    1) args+=(--dense-reference) ;;
    0) ;;
    *) echo "MMRSAE_DENSE_REFERENCE must be 0 or 1" >&2; exit 2 ;;
esac

# MMRSAE_VARIANTS (comma-separated) runs several methods sequentially under
# identical settings; otherwise the single MMRSAE_VARIANT is used. A failed
# method is reported and the remaining methods still run.
IFS=',' read -r -a variants <<< "${MMRSAE_VARIANTS:-${MMRSAE_VARIANT:-mmrsae}}"
if (( ${#variants[@]} == 1 )); then
    exec "$python_bin" "$SCRIPT_DIR/mmrsae_imagenet.py" "${args[@]}" --variant "${variants[0]// /}" "$@"
fi
failed=()
for variant in "${variants[@]}"; do
    variant="${variant// /}"
    [[ -n "$variant" ]] || continue
    echo "=== MMRSAE method: $variant ===" >&2
    "$python_bin" "$SCRIPT_DIR/mmrsae_imagenet.py" "${args[@]}" --variant "$variant" "$@" || failed+=("$variant")
done
if (( ${#failed[@]} )); then
    echo "MMRSAE methods failed: ${failed[*]}" >&2
    exit 1
fi
