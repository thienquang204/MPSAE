#!/usr/bin/env bash
set -Eeuo pipefail

# The only host-side command needed for the three-method ablation:
#   bash run_all_experiments_docker.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
case "$ENV_FILE" in
    /*) ;;
    *) ENV_FILE="$SCRIPT_DIR/$ENV_FILE" ;;
esac
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

IMAGE_NAME="${IMAGE_NAME:-graduate-thesis-three-method-ablation:latest}"
DOCKERFILE="${DOCKERFILE:-$SCRIPT_DIR/Dockerfile.all-experiments}"
DATA_VOLUME="${DATA_VOLUME:-graduate-thesis-imagenet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/runs}"
SUITE_DIR="${SUITE_DIR:-three_method_ablation}"
SHM_SIZE="${SHM_SIZE:-16g}"
PULL_BASE_IMAGE="${PULL_BASE_IMAGE:-0}"

die() {
    echo "Error: $*" >&2
    exit 1
}

case "${HF_TOKEN:-}" in
    hf_replace*) die "replace the placeholder HF_TOKEN in $ENV_FILE" ;;
esac
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
    docker_cmd=(sudo docker)
fi

case "$OUTPUT_ROOT" in
    /*) ;;
    *) OUTPUT_ROOT="$SCRIPT_DIR/$OUTPUT_ROOT" ;;
esac
mkdir -p "$OUTPUT_ROOT/$SUITE_DIR"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
HOST_SUITE_ROOT="$OUTPUT_ROOT/$SUITE_DIR"
PIPELINE_LOG="$HOST_SUITE_ROOT/three_method_ablation_docker.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "============================================================"
echo "Graduate thesis: Matryoshka / CSRv2 / MPSAEv2 ablation"
echo "Image:        $IMAGE_NAME"
echo "Dockerfile:   $DOCKERFILE"
echo "Data volume:  $DATA_VOLUME"
echo "Host output:  $HOST_SUITE_ROOT"
echo "Started UTC:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

if ! "${docker_cmd[@]}" volume inspect "$DATA_VOLUME" >/dev/null 2>&1; then
    "${docker_cmd[@]}" volume create "$DATA_VOLUME" >/dev/null
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

run_status=0
"${docker_cmd[@]}" run --rm \
    --gpus all \
    --shm-size="$SHM_SIZE" \
    -v "$DATA_VOLUME:/data" \
    -v "$OUTPUT_ROOT:/output" \
    "${docker_env_args[@]}" \
    -e SUITE_ROOT="/output/$SUITE_DIR" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "$IMAGE_NAME" || run_status=$?

if [[ "$run_status" == "0" ]]; then
    completion="$HOST_SUITE_ROOT/RUN_COMPLETE.json"
    bundle="$HOST_SUITE_ROOT/imagenet_architecture_ablation_results.zip"
    [[ -s "$completion" ]] || die "container exited successfully but $completion is missing"
    [[ -s "$bundle" ]] || die "container exited successfully but $bundle is missing"
    echo "============================================================"
    echo "THREE-METHOD ABLATION COMPLETE"
    echo "Portable results: $bundle"
    echo "Completion file:  $completion"
    echo "Full Docker log:  $PIPELINE_LOG"
    echo "============================================================"
else
    echo "Ablation failed with status $run_status; rerun this file to start a fresh training run while reusing data/features." >&2
    echo "Log: $PIPELINE_LOG" >&2
fi
exit "$run_status"
