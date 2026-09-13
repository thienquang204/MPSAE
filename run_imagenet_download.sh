#!/usr/bin/env bash
set -Eeuo pipefail

# Optional manual utility for the gated ImageNet downloader. The normal
# one-command workflow in run_all_experiments_docker.sh prepares this volume.
# This helper remains useful for downloading in the background ahead of time.
# Downloaded shards live in the Docker named volume and are reused after a
# network failure or container restart.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-start}"
CONTAINER_NAME="${CONTAINER_NAME:-imagenet-downloader}"
IMAGE_NAME="${IMAGE_NAME:-graduate-thesis-all-experiments:latest}"
DATA_VOLUME="${DATA_VOLUME:-graduate-thesis-imagenet}"
ENV_FILE="${ENV_FILE:-.env}"
CACHE_DIR="${CACHE_DIR:-/data/huggingface}"
RESTART_LIMIT="${RESTART_LIMIT:-20}"

die() {
    echo "Error: $*" >&2
    exit 1
}

container_exists() {
    docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker container inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == "true" ]]
}

show_status() {
    if ! container_exists; then
        echo "Container '$CONTAINER_NAME' does not exist."
        return 1
    fi
    docker ps -a --filter "name=^/${CONTAINER_NAME}$"
    docker container inspect \
        --format 'status={{.State.Status}} exit_code={{.State.ExitCode}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} restart_count={{.RestartCount}}' \
        "$CONTAINER_NAME"
}

start_downloader() {
    command -v docker >/dev/null 2>&1 || die "docker is not installed"
    docker info >/dev/null 2>&1 || die "Docker daemon is not running"
    [[ -f "$ENV_FILE" ]] || die "$ENV_FILE does not exist; create it with HF_TOKEN=hf_..."
    grep -q '^HF_TOKEN=hf_' "$ENV_FILE" || die "$ENV_FILE does not contain a valid-looking HF_TOKEN"
    docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || \
        die "Docker image '$IMAGE_NAME' does not exist; build it first"
    docker volume inspect "$DATA_VOLUME" >/dev/null 2>&1 || docker volume create "$DATA_VOLUME" >/dev/null

    if container_running; then
        echo "Downloader is already running."
        show_status
        echo "Follow progress with: $0 logs"
        return
    fi

    if container_exists; then
        local exit_code
        exit_code="$(docker container inspect --format '{{.State.ExitCode}}' "$CONTAINER_NAME")"
        if [[ "$exit_code" == "0" ]]; then
            echo "The previous downloader completed successfully."
            show_status
            echo "To explicitly download again, run: $0 restart"
            return
        fi
        echo "Removing failed downloader container; cached data remains in volume '$DATA_VOLUME'."
        docker rm "$CONTAINER_NAME" >/dev/null
    fi

    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart "on-failure:${RESTART_LIMIT}" \
        --env-file "$ENV_FILE" \
        -v "${DATA_VOLUME}:/data" \
        --entrypoint python \
        "$IMAGE_NAME" \
        /app/download_imagenet.py \
        --cache-dir "$CACHE_DIR" >/dev/null

    echo "ImageNet downloader started in the background."
    show_status
    echo
    echo "Progress: $0 logs"
    echo "Status:   $0 status"
    echo "Stop:     $0 stop"
}

case "$ACTION" in
    start)
        start_downloader
        ;;
    status)
        show_status
        ;;
    logs)
        container_exists || die "container '$CONTAINER_NAME' does not exist"
        docker logs --follow --tail 100 "$CONTAINER_NAME"
        ;;
    stop)
        if container_running; then
            docker stop --time 30 "$CONTAINER_NAME" >/dev/null
            echo "Downloader stopped. Cached data remains in volume '$DATA_VOLUME'."
        else
            echo "Downloader is not running."
        fi
        ;;
    restart)
        if container_exists; then
            docker rm --force "$CONTAINER_NAME" >/dev/null
        fi
        start_downloader
        ;;
    *)
        echo "Usage: $0 {start|status|logs|stop|restart}" >&2
        exit 2
        ;;
esac
