#!/usr/bin/env bash
set -Eeuo pipefail

# Explicit stages keep the inexpensive research screen separate from the
# original all-experiments launcher. No downloads or dependency installation.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_shell_common.sh"
# Direct execution reads only an explicitly selected ENV_FILE. Docker has
# already loaded its host file and forwards the resolved environment.
load_experiment_env
resolve_repair_stage "$@"
set -- "${repair_cli_args[@]}"
stage="$REPAIR_STAGE"

select_experiment_python
config="${REPAIR_CONFIG:-$SCRIPT_DIR/experiments.json}"
case "$config" in
    /*) ;;
    *) config="$SCRIPT_DIR/$config" ;;
esac
if [[ "$stage" == "plan" ]]; then
    plan_args=(--config "$config")
    if [[ -n "${REPAIR_CANDIDATE:-}" ]]; then plan_args+=(--candidate "$REPAIR_CANDIDATE"); fi
    exec "$python_bin" "$SCRIPT_DIR/make_plan.py" "${plan_args[@]}" "$@"
fi

output="${REPAIR_OUTPUT:-$SCRIPT_DIR/runs/mpsae_repair}"
cache="${REPAIR_CACHE:-$output/cache}"
splits="${REPAIR_SPLITS:-$output/splits}"
args=(--config "$config" --output-dir "$output"
      --cache-dir "$cache" --splits-dir "$splits"
      --device "${REPAIR_DEVICE:-cuda:0}")
train_npz="${REPAIR_TRAIN_NPZ:-${FF2048_TRAIN_EMBEDDINGS:-}}"
if [[ -n "$train_npz" ]]; then
    args+=(--train-npz "$train_npz")
fi
if [[ -n "${REPAIR_CANDIDATE:-}" ]]; then
    args+=(--candidate "$REPAIR_CANDIDATE")
fi
if [[ "$stage" == "confirm" && -n "${REPAIR_VAL_NPZ:-${FF2048_VAL_EMBEDDINGS:-}}" ]]; then
    args+=(--val-npz "${REPAIR_VAL_NPZ:-$FF2048_VAL_EMBEDDINGS}")
fi
if [[ "$stage" == "confirm" && -n "${REPAIR_VAL_SHA256:-}" ]]; then
    args+=(--val-sha256 "$REPAIR_VAL_SHA256")
fi
if [[ "$stage" == "pilot" ]]; then
    # One explicit initial budget: GPU preflight, config-derived splits, checks,
    # then shared W and C/S/N. Later gates and full confirmation stay separate.
    for pilot_stage in preflight prepare check triage; do
        "$python_bin" "$SCRIPT_DIR/mpsae_repair_experiment.py" \
            "${args[@]}" "$@" --stage "$pilot_stage"
    done
    exit 0
fi
exec "$python_bin" "$SCRIPT_DIR/mpsae_repair_experiment.py" "${args[@]}" "$@" --stage "$stage"
