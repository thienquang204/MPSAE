#!/usr/bin/env bash
set -Eeuo pipefail

# The host-side command for the ImageNet-only MMRSAE workflow:
#   bash run_all_experiments_docker.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_shell_common.sh"
load_experiment_env "$SCRIPT_DIR/.env"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

EXPERIMENT_SCOPE="${EXPERIMENT_SCOPE:-mmrsae}"
case "$EXPERIMENT_SCOPE" in
    mmrsae|all|imagenet|transfer|repair) ;;
    *) echo "EXPERIMENT_SCOPE must be mmrsae, all, imagenet, transfer, or repair" >&2; exit 2 ;;
esac

# Planning is handled on the host and needs no container build or credentials.
# The same argument adapter is used later inside the container.
if [[ "$EXPERIMENT_SCOPE" == "mmrsae" ]]; then
    for requested_option in "$@"; do
        if [[ "$requested_option" == "--plan" ]]; then
            exec bash "$SCRIPT_DIR/run_mmrsae_imagenet.sh" "$@"
        fi
    done
fi

# Use the exact host configuration for both planning and container execution.
# A custom file outside the Docker build context is mounted read-only; its
# pathname is not forwarded as an unusable host path inside the container.
repair_mount_args=()
if [[ "$EXPERIMENT_SCOPE" == "repair" ]]; then
    resolve_repair_stage "$@"
    set -- "${repair_cli_args[@]}"
    # CLI configuration paths are host paths too. Resolve and mount them just
    # like REPAIR_CONFIG, rather than forwarding an inaccessible host pathname.
    repair_forward_args=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                [[ $# -ge 2 && "$2" != --* ]] \
                    || { echo "--config needs a host file path" >&2; exit 2; }
                REPAIR_CONFIG="$2"; shift 2 ;;
            --config=*) REPAIR_CONFIG="${1#*=}"; shift ;;
            *) repair_forward_args+=("$1"); shift ;;
        esac
    done
    set -- "${repair_forward_args[@]}"
    repair_host_config="${REPAIR_CONFIG:-$SCRIPT_DIR/experiments.json}"
    case "$repair_host_config" in
        /*) ;;
        *) repair_host_config="$SCRIPT_DIR/$repair_host_config" ;;
    esac
    [[ -f "$repair_host_config" ]] \
        || { echo "REPAIR_CONFIG must name an existing host file: $repair_host_config" >&2; exit 2; }
    repair_host_config="$(cd -- "$(dirname -- "$repair_host_config")" && pwd -P)/$(basename -- "$repair_host_config")"
    # A plan needs no image build, CUDA runtime, token, dataset, or output mount.
    if [[ "$REPAIR_STAGE" == "plan" ]]; then
        export REPAIR_CONFIG="$repair_host_config"
        exec bash "$SCRIPT_DIR/run_mpsae_repair.sh" "$@"
    fi
    repair_mount_args=(-v "$repair_host_config:/app/repair_configuration.json:ro")
fi

IMAGE_NAME="${IMAGE_NAME:-graduate-thesis-mmrsae:latest}"
DOCKERFILE="${DOCKERFILE:-$SCRIPT_DIR/Dockerfile.all-experiments}"
DATA_VOLUME="${DATA_VOLUME:-graduate-thesis-imagenet}"
DATA_HOST_DIR="${DATA_HOST_DIR:-}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/runs}"
if [[ "$EXPERIMENT_SCOPE" == "mmrsae" ]]; then
    SUITE_DIR="${SUITE_DIR:-mmrsae}"
elif [[ "$EXPERIMENT_SCOPE" == "repair" ]]; then
    SUITE_DIR="${SUITE_DIR:-mpsae_repair}"
else
    SUITE_DIR="${SUITE_DIR:-three_method_ablation}"
fi
SHM_SIZE="${SHM_SIZE:-16g}"
PULL_BASE_IMAGE="${PULL_BASE_IMAGE:-0}"

die() {
    echo "Error: $*" >&2
    exit 1
}

# A server can reuse its existing data directory directly. Keep the named
# volume default for existing installations; both appear as /data in Docker.
data_mount_args=(-v "$DATA_VOLUME:/data")
data_location="named volume $DATA_VOLUME -> /data"
if [[ -n "$DATA_HOST_DIR" ]]; then
    case "$DATA_HOST_DIR" in
        /*) ;;
        *) DATA_HOST_DIR="$SCRIPT_DIR/$DATA_HOST_DIR" ;;
    esac
    [[ -d "$DATA_HOST_DIR" ]] || die "DATA_HOST_DIR must be an existing host directory: $DATA_HOST_DIR"
    DATA_HOST_DIR="$(cd -- "$DATA_HOST_DIR" && pwd -P)"
    [[ "$DATA_HOST_DIR" != *","* ]] || die "DATA_HOST_DIR must not contain commas (Docker mount syntax)"
    data_mount_args=(--mount "type=bind,source=$DATA_HOST_DIR,target=/data")
    data_location="$DATA_HOST_DIR -> /data"
fi

if [[ "$EXPERIMENT_SCOPE" != "repair" ]]; then
    case "${HF_TOKEN:-}" in
        hf_replace*) die "replace the placeholder HF_TOKEN in $ENV_FILE" ;;
    esac
fi
case "$DOCKERFILE" in
    /*) ;;
    *) DOCKERFILE="$SCRIPT_DIR/$DOCKERFILE" ;;
esac
command -v docker >/dev/null 2>&1 || die "docker is not installed"
[[ -f "$DOCKERFILE" ]] || die "Dockerfile not found: $DOCKERFILE"
[[ "$SUITE_DIR" != /* && "$SUITE_DIR" != *".."* ]] \
    || die "SUITE_DIR must be a safe relative directory name"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    command -v sudo >/dev/null 2>&1 || die "cannot access Docker and sudo is unavailable"
    sudo docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
    # docker -e NAME reads the Docker client's environment. Preserve only the
    # named credentials across sudo; their values remain out of argv/logs.
    docker_cmd=(sudo --preserve-env=HF_TOKEN,WANDB_API_KEY docker)
fi

case "$OUTPUT_ROOT" in
    /*) ;;
    *) OUTPUT_ROOT="$SCRIPT_DIR/$OUTPUT_ROOT" ;;
esac
mkdir -p "$OUTPUT_ROOT/$SUITE_DIR"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
HOST_SUITE_ROOT="$OUTPUT_ROOT/$SUITE_DIR"
TRANSFER_OUTPUT="${TRANSFER_OUTPUT:-/output/$SUITE_DIR/frozen_transfer}"
if [[ "$EXPERIMENT_SCOPE" == "all" || "$EXPERIMENT_SCOPE" == "transfer" ]]; then
    case "$TRANSFER_OUTPUT" in
        "/output/$SUITE_DIR/"*) ;;
        *) die "TRANSFER_OUTPUT must be a child of /output/$SUITE_DIR" ;;
    esac
    [[ "$TRANSFER_OUTPUT" != *"/../"* && "$TRANSFER_OUTPUT" != */.. ]] \
        || die "TRANSFER_OUTPUT must not contain parent traversal"
fi
HOST_TRANSFER_ROOT="$OUTPUT_ROOT/${TRANSFER_OUTPUT#/output/}"
PIPELINE_LOG="$HOST_SUITE_ROOT/${EXPERIMENT_SCOPE}_docker.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "============================================================"
echo "Graduate thesis adaptive representations: scope=$EXPERIMENT_SCOPE"
echo "Image:        $IMAGE_NAME"
echo "Dockerfile:   $DOCKERFILE"
echo "Data source:  $data_location"
echo "Docker GPUs:  $DOCKER_GPUS"
echo "Host output:  $HOST_SUITE_ROOT"
echo "Started UTC:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

if [[ -z "$DATA_HOST_DIR" ]]; then
    if ! "${docker_cmd[@]}" volume inspect "$DATA_VOLUME" >/dev/null 2>&1; then
        "${docker_cmd[@]}" volume create "$DATA_VOLUME" >/dev/null
    fi
fi

build_args=(build --provenance=false --file "$DOCKERFILE" --tag "$IMAGE_NAME")
case "$PULL_BASE_IMAGE" in
    1) build_args+=(--pull) ;;
    0) ;;
    *) die "PULL_BASE_IMAGE must be 0 or 1" ;;
esac
build_args+=("$SCRIPT_DIR")
BUILDX_GIT_INFO=false "${docker_cmd[@]}" "${build_args[@]}"

docker_env_args=()
if [[ -f "$ENV_FILE" ]]; then
    docker_env_args+=(--env-file "$ENV_FILE")
elif [[ -n "${HF_TOKEN:-}" ]]; then
    docker_env_args+=(-e HF_TOKEN)
fi
# Explicitly exported/sourced options also work without an env file. Forward
# credentials by name so their values do not become Docker command arguments.
for suite_env_name in ${!TRANSFER_@} ${!REPAIR_@} ${!MMRSAE_@} ${!FF2048_@} \
    ${!MPSAE_@} ${!MPSAEV2_@} ${!CSR_@} ${!CSRV2_@} ${!MRL_@} \
    ${!OT_@} ${!FAISS_@} ${!SPARSE_@} ${!KNN_@} ${!RETRIEVAL_@} ${!ANNEAL_@} ${!ABLATION_@}; do
    docker_env_args+=(-e "$suite_env_name=${!suite_env_name}")
done
# Preserve the same precedence for the historical runner's scalar settings.
for runtime_name in EXPERIMENT_PROTOCOL DATA_ROOT DATA_PREP IMAGENET_SPLITS \
    HF_DATASET_ID HF_REVISION WORKERS PREFETCH_FACTOR DEVICE SEED STUDY_SEEDS \
    AMP CHANNELS_LAST TF32 BACKBONES FEATURE_BATCH_SIZE MAX_TRAIN MAX_VAL \
    HIDDEN_DIM TRAIN_K V1_TRAIN_K WEIGHT_DECAY ADAM_EPS CONTRASTIVE_TEMPERATURE \
    REBUILD_CACHE FEATURE_CACHE WEIGHTS_CACHE; do
    if [[ -v "$runtime_name" ]]; then
        docker_env_args+=(-e "$runtime_name=${!runtime_name}")
    fi
done
for credential_name in HF_TOKEN WANDB_API_KEY; do
    if [[ -v "$credential_name" ]]; then
        export "$credential_name"
        docker_env_args+=(-e "$credential_name")
    fi
done
for tracking_name in WANDB_MODE WANDB_ENTITY WANDB_PROJECT WANDB_GROUP WANDB_TAGS WANDB_NAME; do
    if [[ -v "$tracking_name" ]]; then
        docker_env_args+=(-e "$tracking_name=${!tracking_name}")
    fi
done
if [[ "$EXPERIMENT_SCOPE" == "repair" ]]; then
    # Override the host REPAIR_CONFIG after --env-file / forwarded variables.
    docker_env_args+=(-e REPAIR_CONFIG=/app/repair_configuration.json)
fi

run_status=0
"${docker_cmd[@]}" run --rm \
    --gpus "$DOCKER_GPUS" \
    --shm-size="$SHM_SIZE" \
    "${data_mount_args[@]}" \
    -v "$OUTPUT_ROOT:/output" \
    "${repair_mount_args[@]}" \
    "${docker_env_args[@]}" \
    -e EXPERIMENT_SCOPE="$EXPERIMENT_SCOPE" \
    -e TRANSFER_OUTPUT="$TRANSFER_OUTPUT" \
    -e SUITE_ROOT="/output/$SUITE_DIR" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "$IMAGE_NAME" "$@" || run_status=$?

if [[ "$run_status" == "0" ]]; then
    if [[ "$EXPERIMENT_SCOPE" == "mmrsae" ]]; then
        echo "MMRSAE ImageNet command finished: $HOST_SUITE_ROOT"
        echo "Per-backbone configurations, checkpoints, predictions, and W&B logs are stored there."
        echo "Full Docker log: $PIPELINE_LOG"
        exit 0
    fi
    if [[ "$EXPERIMENT_SCOPE" == "repair" ]]; then
        echo "Repair stage ${REPAIR_STAGE:-plan} finished: $HOST_SUITE_ROOT"
        echo "Review decisions.json and per-arm results before selecting a later stage."
        echo "Full Docker log: $PIPELINE_LOG"
        exit 0
    fi
    completion="$HOST_SUITE_ROOT/RUN_COMPLETE.json"
    case "${EXPERIMENT_PROTOCOL:-paper_ff2048}" in
        paper_ff2048)
            bundle="$HOST_SUITE_ROOT/imagenet_ff2048_comparison_results.zip"
            completion_label="FF2048 COMPARISON COMPLETE"
            ;;
        paper_rn18_study)
            bundle="$HOST_SUITE_ROOT/imagenet_rn18_csr_mpsae_study_results.zip"
            completion_label="RESNET-18 STUDY COMPLETE"
            ;;
        *)
            bundle="$HOST_SUITE_ROOT/imagenet_architecture_ablation_results.zip"
            completion_label="TORCHVISION ABLATION COMPLETE"
            ;;
    esac
    if [[ "$EXPERIMENT_SCOPE" == "transfer" ]]; then
        completion="$HOST_TRANSFER_ROOT/RUN_COMPLETE.json"
        bundle="$HOST_TRANSFER_ROOT/transfer_results.zip"
        completion_label="FROZEN TRANSFER COMPLETE"
    elif [[ "$EXPERIMENT_SCOPE" == "all" ]]; then
        [[ -s "$HOST_TRANSFER_ROOT/RUN_COMPLETE.json" ]] \
            || die "transfer completion marker missing"
        [[ -s "$HOST_TRANSFER_ROOT/transfer_results.zip" ]] \
            || die "transfer results bundle missing"
        completion_label="IMAGENET AND FROZEN TRANSFER COMPLETE"
        echo "Transfer results: $HOST_TRANSFER_ROOT/transfer_results.zip"
    fi
    [[ -s "$completion" ]] || die "container exited successfully but $completion is missing"
    [[ -s "$bundle" ]] || die "container exited successfully but $bundle is missing"
    echo "============================================================"
    echo "$completion_label"
    echo "Portable results: $bundle"
    echo "Completion file:  $completion"
    echo "Full Docker log:  $PIPELINE_LOG"
    echo "============================================================"
else
    echo "Suite failed with status $run_status. MMRSAE and transfer resume compatible caches/checkpoints; legacy ImageNet fits restart." >&2
    echo "Log: $PIPELINE_LOG" >&2
fi
exit "$run_status"
