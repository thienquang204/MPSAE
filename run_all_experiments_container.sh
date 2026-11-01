#!/usr/bin/env bash
set -Eeuo pipefail

# MMRSAE runs ImageNet only. Historical suites remain explicit scopes.
EXPERIMENT_SCOPE="${EXPERIMENT_SCOPE:-mmrsae}"
OUTPUT_MOUNT="${OUTPUT_MOUNT:-/output}"
if [[ "$EXPERIMENT_SCOPE" == "mmrsae" ]]; then
    SUITE_ROOT="${SUITE_ROOT:-$OUTPUT_MOUNT/mmrsae}"
else
    SUITE_ROOT="${SUITE_ROOT:-$OUTPUT_MOUNT/three_method_ablation}"
fi
export TRANSFER_DATA_ROOT="${TRANSFER_DATA_ROOT:-/data/transfer}"
export TRANSFER_CACHE="${TRANSFER_CACHE:-/data/transfer_cache}"
export TRANSFER_OUTPUT="${TRANSFER_OUTPUT:-$SUITE_ROOT/frozen_transfer}"
export TRANSFER_IMAGENET_ROOT="${TRANSFER_IMAGENET_ROOT:-/data/imagenet1k}"
# Resolve traversal/symlinks before containment checks or recursive ownership
# handoff. These commands run inside the Linux container, not on the host.
OUTPUT_MOUNT="$(realpath -m -- "$OUTPUT_MOUNT")"
SUITE_ROOT="$(realpath -m -- "$SUITE_ROOT")"
export TRANSFER_OUTPUT="$(realpath -m -- "$TRANSFER_OUTPUT")"
case "$EXPERIMENT_SCOPE" in
    mmrsae|all|imagenet|transfer|repair) ;;
    *) echo "EXPERIMENT_SCOPE must be mmrsae, all, imagenet, transfer, or repair" >&2; exit 2 ;;
esac
if [[ "$EXPERIMENT_SCOPE" == "mmrsae" ]]; then
    case "$SUITE_ROOT" in
        "$OUTPUT_MOUNT"/*) ;;
        *) echo "MMRSAE SUITE_ROOT must be a child of the persistent output mount" >&2; exit 2 ;;
    esac
    export MMRSAE_OUTPUT="$(realpath -m -- "${MMRSAE_OUTPUT:-$SUITE_ROOT}")"
    case "$MMRSAE_OUTPUT" in
        "$SUITE_ROOT"|"$SUITE_ROOT"/*) ;;
        *) echo "MMRSAE_OUTPUT must remain inside SUITE_ROOT" >&2; exit 2 ;;
    esac
    export MMRSAE_CACHE="${MMRSAE_CACHE:-/data/mmrsae}"
    export MMRSAE_DATA_ROOT="${MMRSAE_DATA_ROOT:-/data/imagenet1k}"
    export MMRSAE_EMBEDDINGS="${MMRSAE_EMBEDDINGS:-/data/embeddings}"
    mountpoint -q "$OUTPUT_MOUNT" \
        || { echo "$OUTPUT_MOUNT must be mounted for persistent MMRSAE results" >&2; exit 2; }
    mmrsae_status=0
    bash /app/run_mmrsae_imagenet.sh "$@" || mmrsae_status=$?
    if [[ -d "$MMRSAE_OUTPUT" && ( "${HOST_UID:-0}" != "0" || "${HOST_GID:-0}" != "0" ) ]]; then
        chown -R "${HOST_UID:-0}:${HOST_GID:-0}" "$MMRSAE_OUTPUT" 2>/dev/null || true
    fi
    exit "$mmrsae_status"
fi
if [[ "$EXPERIMENT_SCOPE" == "repair" ]]; then
    source /app/experiment_shell_common.sh
    resolve_repair_stage "$@"
    set -- "${repair_cli_args[@]}"
    case "$SUITE_ROOT" in
        "$OUTPUT_MOUNT"/*) ;;
        *) echo "Repair SUITE_ROOT must be a child of the persistent output mount" >&2; exit 2 ;;
    esac
    export REPAIR_OUTPUT="$(realpath -m -- "${REPAIR_OUTPUT:-$SUITE_ROOT}")"
    case "$REPAIR_OUTPUT" in
        "$SUITE_ROOT"|"$SUITE_ROOT"/*) ;;
        *) echo "REPAIR_OUTPUT must remain inside SUITE_ROOT" >&2; exit 2 ;;
    esac
    export REPAIR_CACHE="${REPAIR_CACHE:-/data/mpsae_repair_cache}"
    export REPAIR_SPLITS="${REPAIR_SPLITS:-$REPAIR_OUTPUT/splits}"
    if [[ "$REPAIR_STAGE" != "plan" ]]; then
        mountpoint -q "$OUTPUT_MOUNT" \
            || { echo "$OUTPUT_MOUNT must be mounted for persistent research results" >&2; exit 2; }
    fi
    repair_status=0
    bash /app/run_mpsae_repair.sh "$@" || repair_status=$?
    if [[ -d "$REPAIR_OUTPUT" && ( "${HOST_UID:-0}" != "0" || "${HOST_GID:-0}" != "0" ) ]]; then
        chown -R "${HOST_UID:-0}:${HOST_GID:-0}" "$REPAIR_OUTPUT" 2>/dev/null || true
    fi
    exit "$repair_status"
fi
if [[ "$EXPERIMENT_SCOPE" == "transfer" ]]; then
    case "$SUITE_ROOT" in
        "$OUTPUT_MOUNT"|"$OUTPUT_MOUNT"/*) ;;
        *) echo "SUITE_ROOT must be within the persistent output mount" >&2; exit 2 ;;
    esac
    case "$TRANSFER_OUTPUT" in
        "$SUITE_ROOT"/*) ;;
        *) echo "TRANSFER_OUTPUT must be a child of SUITE_ROOT" >&2; exit 2 ;;
    esac
    mountpoint -q "$OUTPUT_MOUNT" \
        || { echo "$OUTPUT_MOUNT must be mounted for persistent results" >&2; exit 2; }
    transfer_status=0
    bash /app/run_transfer_experiments.sh "$@" || transfer_status=$?
    if [[ "${HOST_UID:-0}" != "0" || "${HOST_GID:-0}" != "0" ]]; then
        chown -R "${HOST_UID:-0}:${HOST_GID:-0}" "$TRANSFER_OUTPUT" 2>/dev/null || true
    fi
    exit "$transfer_status"
fi

# Paper-faithful FF2048 comparison by default. The Appendix E.1 ResNet-18
# study and legacy torchvision ablations are explicit alternative protocols.
EXPERIMENT_PROTOCOL="${EXPERIMENT_PROTOCOL:-paper_ff2048}"
case "$EXPERIMENT_PROTOCOL" in
    paper_ff2048)
        DEFAULT_DATA_ROOT=/data/csr_precompute_embeds
        DEFAULT_DATA_PREP=paper_embeddings
        DEFAULT_HF_DATASET_ID=W1nd-navigator/CSR-precompute-embeds
        DEFAULT_BACKBONES=resnet50
        DEFAULT_BATCH_SIZE=4096
        DEFAULT_FEATURE_BATCH_SIZE=512
        DEFAULT_CSR_GAMMA=0.1
        DEFAULT_STUDY_SEEDS=1227
        DEFAULT_V1_TRAIN_K=32
        ;;
    paper_rn18_study)
        DEFAULT_DATA_ROOT=/data/imagenet1k
        DEFAULT_DATA_PREP=auto
        DEFAULT_HF_DATASET_ID=ILSVRC/imagenet-1k
        DEFAULT_BACKBONES=resnet18
        DEFAULT_BATCH_SIZE=1024
        DEFAULT_FEATURE_BATCH_SIZE=256
        DEFAULT_CSR_GAMMA=1.0
        DEFAULT_STUDY_SEEDS=1227,1228,1229
        DEFAULT_V1_TRAIN_K=16
        ;;
    torchvision_ablation)
        DEFAULT_DATA_ROOT=/data/imagenet1k
        DEFAULT_DATA_PREP=auto
        DEFAULT_HF_DATASET_ID=ILSVRC/imagenet-1k
        DEFAULT_BACKBONES=resnet18,resnet50
        DEFAULT_BATCH_SIZE=4096
        DEFAULT_FEATURE_BATCH_SIZE=512
        DEFAULT_CSR_GAMMA=0.1
        DEFAULT_STUDY_SEEDS=1227
        DEFAULT_V1_TRAIN_K=32
        ;;
    *)
        echo "Error: unsupported EXPERIMENT_PROTOCOL '$EXPERIMENT_PROTOCOL'" >&2
        exit 1
        ;;
esac

DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
OUTPUT_MOUNT="${OUTPUT_MOUNT:-/output}"
SUITE_ROOT="${SUITE_ROOT:-$OUTPUT_MOUNT/three_method_ablation}"
DATA_PREP="${DATA_PREP:-$DEFAULT_DATA_PREP}"
IMAGENET_SPLITS="${IMAGENET_SPLITS:-train,validation}"
HF_DATASET_ID="${HF_DATASET_ID:-$DEFAULT_HF_DATASET_ID}"
HF_REVISION="${HF_REVISION:-main}"
FF2048_TRAIN_EMBEDDINGS="${FF2048_TRAIN_EMBEDDINGS:-$DATA_ROOT/FF2048_RN50_Embeds/1K_train_ff2048.npz}"
FF2048_VAL_EMBEDDINGS="${FF2048_VAL_EMBEDDINGS:-$DATA_ROOT/FF2048_RN50_Embeds/1K_val_ff2048.npz}"
WORKERS="${WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1227}"
STUDY_SEEDS="${STUDY_SEEDS:-$DEFAULT_STUDY_SEEDS}"
AMP="${AMP:-0}"
CHANNELS_LAST="${CHANNELS_LAST:-1}"
TF32="${TF32:-0}"
HOST_UID="${HOST_UID:-0}"
HOST_GID="${HOST_GID:-0}"

# Each strict paper protocol accepts exactly its documented backbone.
BACKBONES="${BACKBONES:-$DEFAULT_BACKBONES}"
BASE_EPOCHS="${ABLATION_EPOCHS:-10}"
MPSAEV2_EXTRA_EPOCHS="${MPSAEV2_EXTRA_EPOCHS:-${MPSAE_EXTRA_EPOCHS:-0}}"
BATCH_SIZE="${ABLATION_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-$DEFAULT_FEATURE_BATCH_SIZE}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_VAL="${MAX_VAL:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-8192}"
TRAIN_K="${TRAIN_K:-2}"
V1_TRAIN_K="${V1_TRAIN_K:-$DEFAULT_V1_TRAIN_K}"
ANNEAL_START_K="${ANNEAL_START_K:-64}"
ANNEAL_FRACTION="${ANNEAL_FRACTION:-0.7}"
SPARSE_EXTRA_TOPK="${SPARSE_EXTRA_TOPK:-}"
KNN_QUERY_BATCH="${KNN_QUERY_BATCH:-32}"
SPARSE_KNN_QUERY_BATCH="${SPARSE_KNN_QUERY_BATCH:-$KNN_QUERY_BATCH}"
SPARSE_GALLERY_CHUNK="${SPARSE_GALLERY_CHUNK:-65536}"
RETRIEVAL_WARMUP="${RETRIEVAL_WARMUP:-3}"
RETRIEVAL_REPEATS="${RETRIEVAL_REPEATS:-5}"
SPARSE_SCORE_BUFFER_MIB="${SPARSE_SCORE_BUFFER_MIB:-256}"
SPARSE_GPU_MEMORY_FRACTION="${SPARSE_GPU_MEMORY_FRACTION:-0.8}"
MRL_CLASSIFICATION_WEIGHT="${MRL_CLASSIFICATION_WEIGHT:-1.0}"
CSRV2_MAIN_RECON_WEIGHT="${CSRV2_MAIN_RECON_WEIGHT:-${CSR_MAIN_RECON_WEIGHT:-1.0}}"
CSRV2_MULTI_TOPK_RECON_WEIGHT="${CSRV2_MULTI_TOPK_RECON_WEIGHT:-${CSR_MULTI_TOPK_RECON_WEIGHT:-0.125}}"
CSRV2_AUX_RECON_WEIGHT="${CSRV2_AUX_RECON_WEIGHT:-${CSR_AUX_RECON_WEIGHT:-0.03125}}"
CSRV2_CONTRASTIVE_WEIGHT="${CSRV2_CONTRASTIVE_WEIGHT:-${CSR_CONTRASTIVE_WEIGHT:-$DEFAULT_CSR_GAMMA}}"
MPSAEV2_MAIN_RECON_WEIGHT="${MPSAEV2_MAIN_RECON_WEIGHT:-${MPSAE_MAIN_RECON_WEIGHT:-1.0}}"
MPSAEV2_NESTED_RECON_WEIGHT="${MPSAEV2_NESTED_RECON_WEIGHT:-${MPSAE_NESTED_RECON_WEIGHT:-0.125}}"
MPSAEV2_AUX_RECON_WEIGHT="${MPSAEV2_AUX_RECON_WEIGHT:-${MPSAE_AUX_RECON_WEIGHT:-0.03125}}"
MPSAEV2_MMPOT_WEIGHT="${MPSAEV2_MMPOT_WEIGHT:-${MPSAE_MMPOT_WEIGHT:-2.0}}"
MPSAEV2_LR="${MPSAEV2_LR:-${MPSAE_LR:-4e-5}}"
CSRV2_LR="${CSRV2_LR:-${CSR_LR:-4e-5}}"
MRL_LR="${MRL_LR:-0.475}"
MRL_MOMENTUM="${MRL_MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
ADAM_EPS="${ADAM_EPS:-6.25e-10}"
CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.2}"
OT_MASS="${OT_MASS:-0.9}"
OT_ETA="${OT_ETA:-0.2}"
OT_ITERS="${OT_ITERS:-100}"
OT_MICROBATCH="${OT_MICROBATCH:-32}"
FAISS_GPU_DEVICE="${FAISS_GPU_DEVICE:-0}"
FAISS_GPU="${FAISS_GPU:-1}"
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

[[ "$MPSAEV2_EXTRA_EPOCHS" =~ ^[0-9]+$ ]] \
    || die "MPSAEV2_EXTRA_EPOCHS must be a non-negative integer"
[[ "$STUDY_SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || die "STUDY_SEEDS must be comma-separated non-negative integers"
[[ "$FAISS_GPU" == "1" ]] \
    || die "GPU retrieval comparisons require FAISS_GPU=1; CPU fallback is disabled"
case "$EXPERIMENT_PROTOCOL" in
    paper_ff2048)
        [[ "$BACKBONES" == "resnet50" ]] \
            || die "paper_ff2048 requires BACKBONES=resnet50"
        ;;
    paper_rn18_study)
        [[ "$BACKBONES" == "resnet18" ]] \
            || die "paper_rn18_study requires BACKBONES=resnet18"
        ;;
esac
case "$SUITE_ROOT" in
    "$OUTPUT_MOUNT"|"$OUTPUT_MOUNT"/*) ;;
    *) die "SUITE_ROOT must be inside the persistent $OUTPUT_MOUNT mount" ;;
esac
case "$TRANSFER_OUTPUT" in
    "$SUITE_ROOT"/*) ;;
    *) die "TRANSFER_OUTPUT must be a child of SUITE_ROOT" ;;
esac
if ! mountpoint -q "$OUTPUT_MOUNT" 2>/dev/null \
    && ! grep -qE "[[:space:]]${OUTPUT_MOUNT}[[:space:]]" /proc/self/mountinfo /proc/mounts 2>/dev/null; then
    die "$OUTPUT_MOUNT is not mounted; refusing to create disposable results"
fi

mkdir -p "$SUITE_ROOT/logs"
rm -f "$SUITE_ROOT/RUN_COMPLETE.json"

handoff_output() {
    local status=$?
    if [[ "$status" != "0" ]]; then
        rm -f "$SUITE_ROOT/RUN_COMPLETE.json"
    fi
    if [[ "$HOST_UID" != "0" || "$HOST_GID" != "0" ]]; then
        chown -R "${HOST_UID}:${HOST_GID}" "$SUITE_ROOT" 2>/dev/null || true
    fi
    chmod -R u+rwX,go+rX "$SUITE_ROOT" 2>/dev/null || true
    return "$status"
}
trap handoff_output EXIT

python - "$FAISS_GPU_DEVICE" <<'PY'
import sys
import torch
import faiss

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; start Docker with NVIDIA GPU access")
device = int(sys.argv[1])
if not 0 <= device < torch.cuda.device_count():
    raise SystemExit(f"FAISS_GPU_DEVICE={device} is not a visible CUDA device")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("GPU retrieval requires CUDA-enabled FAISS; rebuild the project image")
print(f"GPU retrieval ready: cuda:{device} ({torch.cuda.get_device_name(device)})")
print("Sparse retrieval uses a CUDA inverted index compiled before timed search")
PY

# Build/check the sparse CUDA backend before any feature extraction or fitting.
python - "$FAISS_GPU_DEVICE" <<'PY'
import sys
from sparse_gpu_retrieval import preload_sparse_retrieval
preload_sparse_retrieval("cuda:" + sys.argv[1])
PY

if [[ "$EXPERIMENT_SCOPE" == "all" ]]; then
    # Validate all data after GPU preflight and before the ImageNet study.
    bash /app/run_transfer_experiments.sh --prepare-only
fi

imagenet_ready() {
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

paper_embeddings_ready() {
    [[ -s "$FF2048_TRAIN_EMBEDDINGS" && -s "$FF2048_VAL_EMBEDDINGS" ]]
}

download_paper_embeddings() {
    python - "$DATA_ROOT" "$HF_DATASET_ID" "$HF_REVISION" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

root, dataset_id, revision = sys.argv[1:]
Path(root).mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN")
for filename in (
    "FF2048_RN50_Embeds/1K_train_ff2048.npz",
    "FF2048_RN50_Embeds/1K_val_ff2048.npz",
):
    hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=root,
    )
PY
}

case "$DATA_PREP" in
    paper_embeddings)
        [[ "$EXPERIMENT_PROTOCOL" == "paper_ff2048" ]] \
            || die "DATA_PREP=paper_embeddings requires EXPERIMENT_PROTOCOL=paper_ff2048"
        paper_embeddings_ready || download_paper_embeddings
        ;;
    auto)
        if [[ "$EXPERIMENT_PROTOCOL" == "paper_ff2048" ]]; then
            paper_embeddings_ready || download_paper_embeddings
        elif ! imagenet_ready; then
            python /app/download_imagenet.py \
                --cache-dir "$DATA_ROOT" \
                --dataset-id "$HF_DATASET_ID" \
                --revision "$HF_REVISION" \
                --splits "$IMAGENET_SPLITS" \
                --check-samples all
        fi
        ;;
    always)
        if [[ "$EXPERIMENT_PROTOCOL" == "paper_ff2048" ]]; then
            download_paper_embeddings
        else
            python /app/download_imagenet.py \
                --cache-dir "$DATA_ROOT" \
                --dataset-id "$HF_DATASET_ID" \
                --revision "$HF_REVISION" \
                --splits "$IMAGENET_SPLITS" \
                --check-samples all
        fi
        ;;
    never)
        if [[ "$EXPERIMENT_PROTOCOL" == "paper_ff2048" ]]; then
            paper_embeddings_ready \
                || die "DATA_PREP=never but the FF2048 archives are missing"
        else
            imagenet_ready \
                || die "DATA_PREP=never but the ImageNet cache is not validated"
        fi
        ;;
    *) die "DATA_PREP must be paper_embeddings, auto, always, or never" ;;
esac
if [[ "$EXPERIMENT_PROTOCOL" == "paper_ff2048" ]]; then
    paper_embeddings_ready || die "FF2048 embedding preparation failed"
else
    imagenet_ready || die "ImageNet preparation did not produce a valid manifest"
fi

if [[ "$EXPERIMENT_PROTOCOL" == "paper_rn18_study" ]]; then
    APPLICABLE_METHODS=dense,csr,mpsae
else
    APPLICABLE_METHODS=matryoshka,csr,mpsae,csrv2,mpsaev2
fi

{
    echo "study=$EXPERIMENT_PROTOCOL"
    echo "protocol=$EXPERIMENT_PROTOCOL"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "backbones=$BACKBONES"
    echo "methods=$APPLICABLE_METHODS"
    echo "base_epochs=$BASE_EPOCHS"
    echo "mpsaev2_extra_epochs=$MPSAEV2_EXTRA_EPOCHS"
    echo "batch_size=$BATCH_SIZE"
    echo "feature_extraction_batch_size=$FEATURE_BATCH_SIZE"
    echo "hidden_dim=$HIDDEN_DIM"
    echo "anneal_start_k=$ANNEAL_START_K"
    echo "anneal_target_k=$TRAIN_K"
    echo "v1_fixed_train_k=$V1_TRAIN_K"
    echo "anneal_fraction=$ANNEAL_FRACTION"
    echo "resnet18_paper_study_budgets=8,16,64,256"
    echo "resnet18_legacy_ablation_budgets=8,16,32,64,128,256,512"
    echo "resnet50_evaluation_budgets=8,16,32,64,128,256,512,1024,2048"
    echo "sparse_extra_evaluation_budgets=$SPARSE_EXTRA_TOPK"
    echo "retrieval=unit_normalized_exact_l2"
    echo "dense_retrieval_backend=faiss_gpu_flat_l2"
    echo "sparse_retrieval_backend=cuda_inverted_index"
    echo "retrieval_device=cuda:$FAISS_GPU_DEVICE"
    echo "knn_query_batch=$KNN_QUERY_BATCH"
    echo "sparse_knn_query_batch=$SPARSE_KNN_QUERY_BATCH"
    echo "sparse_gallery_chunk=$SPARSE_GALLERY_CHUNK"
    echo "retrieval_warmup=$RETRIEVAL_WARMUP"
    echo "retrieval_repeats=$RETRIEVAL_REPEATS"
    echo "sparse_score_buffer_mib=$SPARSE_SCORE_BUFFER_MIB"
    echo "sparse_gpu_memory_fraction=$SPARSE_GPU_MEMORY_FRACTION"
    echo "retrieval_timing_scope=search_only_gpu_resident"
    echo "retrieval_query_metric=amortized_time_per_query_at_reported_batch_size"
    echo "mrl_classification_weight=$MRL_CLASSIFICATION_WEIGHT"
    echo "csrv2_main_recon_weight=$CSRV2_MAIN_RECON_WEIGHT"
    echo "csrv2_multi_topk_recon_weight=$CSRV2_MULTI_TOPK_RECON_WEIGHT"
    echo "csrv2_aux_recon_weight=$CSRV2_AUX_RECON_WEIGHT"
    echo "csrv2_contrastive_weight=$CSRV2_CONTRASTIVE_WEIGHT"
    echo "contrastive_temperature=$CONTRASTIVE_TEMPERATURE"
    echo "mpsaev2_main_recon_weight=$MPSAEV2_MAIN_RECON_WEIGHT"
    echo "mpsaev2_nested_recon_weight=$MPSAEV2_NESTED_RECON_WEIGHT"
    echo "mpsaev2_aux_recon_weight=$MPSAEV2_AUX_RECON_WEIGHT"
    echo "mpsaev2_mmpot_weight=$MPSAEV2_MMPOT_WEIGHT"
    echo "adam_eps=$ADAM_EPS"
    echo "weight_decay=$WEIGHT_DECAY"
    echo "model_weights_saved=false"
    echo "dataset_root=$DATA_ROOT"
    echo "dataset_id=$HF_DATASET_ID"
    echo "dataset_revision=$HF_REVISION"
    echo "ff2048_train_embeddings=$FF2048_TRAIN_EMBEDDINGS"
    echo "ff2048_val_embeddings=$FF2048_VAL_EMBEDDINGS"
    echo "seed=$SEED"
    echo "study_seeds=$STUDY_SEEDS"
    echo "tf32=$TF32"
} > "$SUITE_ROOT/ablation_run_manifest.txt"

amp_flag="$(bool_flag "$AMP" --amp --no-amp)"
channels_flag="$(bool_flag "$CHANNELS_LAST" --channels-last --no-channels-last)"
tf32_flag="$(bool_flag "$TF32" --tf32 --no-tf32)"
rebuild_flag="$(bool_flag "$REBUILD_CACHE" --rebuild-cache '')"
sparse_extra_args=()
if [[ -n "$SPARSE_EXTRA_TOPK" ]]; then
    sparse_extra_args=(--sparse-extra-topk "$SPARSE_EXTRA_TOPK")
fi

DATA_BACKEND=hf \
BACKBONES="$BACKBONES" \
CACHE_DIR="$FEATURE_CACHE" \
OUTPUT_DIR="$SUITE_ROOT" \
WEIGHTS_CACHE="$WEIGHTS_CACHE" \
INSTALL_DEPS=0 \
FAISS_GPU=1 \
AGGREGATE_RESULTS=1 \
/app/run_csr_vs_mmpot_imagenet.sh "$DATA_ROOT" \
    --protocol "$EXPERIMENT_PROTOCOL" \
    --ff2048-train-embeddings "$FF2048_TRAIN_EMBEDDINGS" \
    --ff2048-val-embeddings "$FF2048_VAL_EMBEDDINGS" \
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
    "${sparse_extra_args[@]}" \
    --knn-query-batch "$KNN_QUERY_BATCH" \
    --sparse-knn-query-batch "$SPARSE_KNN_QUERY_BATCH" \
    --sparse-gallery-chunk "$SPARSE_GALLERY_CHUNK" \
    --retrieval-warmup "$RETRIEVAL_WARMUP" \
    --retrieval-repeats "$RETRIEVAL_REPEATS" \
    --sparse-score-buffer-mib "$SPARSE_SCORE_BUFFER_MIB" \
    --sparse-gpu-memory-fraction "$SPARSE_GPU_MEMORY_FRACTION" \
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
    --adam-eps "$ADAM_EPS" \
    --contrastive-temperature "$CONTRASTIVE_TEMPERATURE" \
    --ot-mass "$OT_MASS" \
    --ot-eta "$OT_ETA" \
    --ot-iters "$OT_ITERS" \
    --ot-microbatch "$OT_MICROBATCH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --study-seeds "$STUDY_SEEDS" \
    --faiss-gpu-device "$FAISS_GPU_DEVICE" \
    --faiss-temp-memory-mib "$FAISS_TEMP_MEMORY_MIB" \
    "$amp_flag" "$channels_flag" "$tf32_flag" \
    ${rebuild_flag:+"$rebuild_flag"} \
    2>&1 | tee -a "$SUITE_ROOT/logs/three_method_ablation.log"

echo "Representation comparison complete: $SUITE_ROOT"
if [[ "$EXPERIMENT_SCOPE" == "all" ]]; then
    mv "$SUITE_ROOT/RUN_COMPLETE.json" "$SUITE_ROOT/IMAGENET_RUN_COMPLETE.json"
    bash /app/run_transfer_experiments.sh
    python - "$SUITE_ROOT" "$TRANSFER_OUTPUT" <<'PY'
import json
import os
import sys
from pathlib import Path
root, transfer = map(Path, sys.argv[1:])
legacy = json.loads((root / "IMAGENET_RUN_COMPLETE.json").read_text())
downstream = json.loads((transfer / "RUN_COMPLETE.json").read_text())
temporary = root / "RUN_COMPLETE.json.tmp"
temporary.write_text(json.dumps({"status": "complete", "scope": "all", "imagenet": legacy,
                                 "transfer": downstream}, indent=2))
os.replace(temporary, root / "RUN_COMPLETE.json")
PY
    echo "Frozen-transfer results: $TRANSFER_OUTPUT/transfer_results.zip"
fi
if [[ "$EXPERIMENT_PROTOCOL" == "paper_ff2048" ]]; then
    echo "Portable results: $SUITE_ROOT/imagenet_ff2048_comparison_results.zip"
elif [[ "$EXPERIMENT_PROTOCOL" == "paper_rn18_study" ]]; then
    echo "Portable results: $SUITE_ROOT/imagenet_rn18_csr_mpsae_study_results.zip"
else
    echo "Portable results: $SUITE_ROOT/imagenet_architecture_ablation_results.zip"
fi
