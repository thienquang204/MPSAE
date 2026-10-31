#!/usr/bin/env bash
set -Eeuo pipefail

# Short server entry point. Experiment definitions remain in the existing runners.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_shell_common.sh"

usage() {
    cat <<'USAGE'
Usage: bash run_imagenet.sh [--env FILE] [--local] COMMAND [OPTIONS]

  init [repair|mmrsae]  Create settings without overwriting existing files.
                       Default: repair, the full_theory_detail.md study.
  plan                 Show the recipe/budget; no Docker build or training.
  preflight            Check the CUDA retrieval environment.
  prepare              Prepare inputs without fitting.
  run                  Run/resume the repair pilot or configured MMRSAE fit.
  logs                 Follow the saved console log (Ctrl-C stops viewing).

Repair also accepts: check, warm, triage, attribution, wide, timing, lock,
holdout, confirm, arm. Runner options go after the command.

Defaults: imagenet.env beside this script; Docker execution.
--local uses an already prepared Python/CUDA environment and local data paths.
Relative settings/output paths are resolved from this repository.

First use:
  bash run_imagenet.sh init
  # Edit imagenet.env (paths/GPU) and imagenet.json (trusted source pins).
  bash run_imagenet.sh plan
  bash run_imagenet.sh run
USAGE
}

die() { echo "Error: $*" >&2; exit 2; }

requested_execution=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "--env needs a file path"
            ENV_FILE="$2"; shift 2 ;;
        --env=*) ENV_FILE="${1#*=}"; [[ -n "$ENV_FILE" ]] || die "--env needs a file path"; shift ;;
        --local) requested_execution=local; shift ;;
        -h|--help|help) usage; exit 0 ;;
        --*) die "Put runner options after a command; use --help for usage" ;;
        *) break ;;
    esac
done
command_name="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$command_name" == help ]]; then usage; exit 0; fi
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/imagenet.env}"
case "$ENV_FILE" in /*) ;; *) ENV_FILE="$SCRIPT_DIR/$ENV_FILE" ;; esac

if [[ "$command_name" == init ]]; then
    [[ $# -le 1 ]] || die "init accepts only an optional repair or mmrsae profile"
    profile="${1:-repair}"
    case "$profile" in repair|mmrsae) ;; *) die "Profile must be repair or mmrsae" ;; esac
    [[ ! -e "$ENV_FILE" && ! -L "$ENV_FILE" ]] || die "Settings already exist: $ENV_FILE (edit them or use --env another.env)"
    mkdir -p -- "$(dirname -- "$ENV_FILE")"
    ENV_FILE="$(cd -- "$(dirname -- "$ENV_FILE")" && pwd -P)/$(basename -- "$ENV_FILE")"
    recipe_file="${ENV_FILE%.env}.json"
    [[ "$recipe_file" != "$ENV_FILE" ]] || recipe_file="$ENV_FILE.json"
    if [[ "$profile" == repair ]]; then
        [[ ! -e "$recipe_file" && ! -L "$recipe_file" ]] || die "Recipe already exists: $recipe_file (initialization will not overwrite it)"
    fi
    (
        umask 077
        set -o noclobber
        if [[ "$profile" == repair ]]; then
            cat "$SCRIPT_DIR/experiments.json" > "$recipe_file"
        fi
        {
            printf '# Created by run_imagenet.sh; edit this file once, then use plan/run/logs.\n'
            printf 'IMAGENET_EXECUTION=%s\n\n' "${requested_execution:-docker}"
            while IFS= read -r setting_line || [[ -n "$setting_line" ]]; do
                # Use a private recipe copy; preserve every other profile setting.
                if [[ "$profile" == repair && "$setting_line" == REPAIR_CONFIG=* ]]; then
                    printf 'REPAIR_CONFIG=%q\n' "$recipe_file"
                elif [[ "$requested_execution" == local && "$setting_line" == FF2048_TRAIN_EMBEDDINGS=* ]]; then
                    printf 'FF2048_TRAIN_EMBEDDINGS=%q\n' "$SCRIPT_DIR/data/csr_precompute_embeds/FF2048_RN50_Embeds/1K_train_ff2048.npz"
                elif [[ "$requested_execution" == local && "$setting_line" == FF2048_VAL_EMBEDDINGS=* ]]; then
                    printf 'FF2048_VAL_EMBEDDINGS=%q\n' "$SCRIPT_DIR/data/csr_precompute_embeds/FF2048_RN50_Embeds/1K_val_ff2048.npz"
                elif [[ "$requested_execution" == local && "$setting_line" == MMRSAE_DATA_ROOT=* ]]; then
                    printf 'MMRSAE_DATA_ROOT=%q\n' "$SCRIPT_DIR/data/imagenet1k"
                elif [[ "$requested_execution" == local && "$setting_line" == MMRSAE_CACHE=* ]]; then
                    printf 'MMRSAE_CACHE=%q\n' "$SCRIPT_DIR/data/mmrsae"
                else
                    printf '%s\n' "$setting_line"
                fi
            done < "$SCRIPT_DIR/$profile.env.example"
        } > "$ENV_FILE"
    )
    printf 'Created settings: %s\n' "$ENV_FILE"
    if [[ "$profile" == repair ]]; then
        printf 'Created recipe:   %s\nSet source.revision and source.train_sha256 from trusted release metadata.\n' "$recipe_file"
    fi
    echo "Edit data paths and GPU settings, then use plan and run with the same settings file."
    exit 0
fi

[[ -f "$ENV_FILE" ]] || die "Settings not found: $ENV_FILE. Start with: bash run_imagenet.sh init"
# The command selects the stage. Do not let a forwarded option turn a plan
# or preparation request into training in the lower-level stage adapter.
for runner_option in "$@"; do
    case "$runner_option" in
        --stage|--stage=*) die "Choose the stage as the command (for example: run_imagenet.sh triage), not --stage" ;;
    esac
done
load_experiment_env
case "${EXPERIMENT_SCOPE:-}" in
    repair|mmrsae) ;;
    *) die "This entry point accepts only EXPERIMENT_SCOPE=repair or mmrsae" ;;
esac
execution="${requested_execution:-${IMAGENET_EXECUTION:-docker}}"
case "$execution" in docker|local) ;; *) die "IMAGENET_EXECUTION must be docker or local" ;; esac

if [[ "$EXPERIMENT_SCOPE" == repair ]]; then
    suite_default=mpsae_repair
    case "$command_name" in
        run) runner_args=(pilot) ;;
        plan|preflight|prepare|pilot|check|warm|triage|attribution|wide|timing|lock|holdout|confirm|arm)
            runner_args=("$command_name") ;;
        logs) runner_args=() ;;
        *) die "Unknown repair command: $command_name (use --help)" ;;
    esac
    direct_runner=run_mpsae_repair.sh
else
    suite_default=mmrsae
    case "$command_name" in
        run|logs) runner_args=() ;;
        plan) runner_args=(--plan) ;;
        preflight) runner_args=(--preflight-only) ;;
        prepare) runner_args=(--prepare-only) ;;
        *) die "MMRSAE accepts plan, preflight, prepare, run, or logs" ;;
    esac
    direct_runner=run_mmrsae_imagenet.sh
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/runs}"
case "$OUTPUT_ROOT" in /*) ;; *) OUTPUT_ROOT="$SCRIPT_DIR/$OUTPUT_ROOT" ;; esac
SUITE_DIR="${SUITE_DIR:-$suite_default}"
[[ -n "$SUITE_DIR" && "$SUITE_DIR" != /* && "$SUITE_DIR" != *".."* ]] || die "SUITE_DIR must be a relative directory without parent traversal"
console_log="$OUTPUT_ROOT/$SUITE_DIR/${EXPERIMENT_SCOPE}_${execution}.log"
if [[ "$command_name" == logs ]]; then
    [[ $# == 0 ]] || die "logs does not accept runner options"
    [[ -f "$console_log" ]] || die "No saved log yet: $console_log. Run an execution stage first."
    exec tail -n 80 -F -- "$console_log"
fi

# Normalize relative runner paths consistently, including CLI overrides.
cd -- "$SCRIPT_DIR"
if [[ "$execution" == docker ]]; then
    exec bash "$SCRIPT_DIR/run_all_experiments_docker.sh" "${runner_args[@]}" "$@"
fi
export REPAIR_OUTPUT="${REPAIR_OUTPUT:-$OUTPUT_ROOT/$SUITE_DIR}"
export MMRSAE_OUTPUT="${MMRSAE_OUTPUT:-$OUTPUT_ROOT/$SUITE_DIR}"
# Planning leaves the filesystem alone. Executed local stages get the same
# append-only console logging that the Docker entry point already provides.
if [[ "$command_name" != plan ]]; then
    mkdir -p -- "$(dirname -- "$console_log")"
    exec > >(tee -a "$console_log") 2>&1
fi
exec bash "$SCRIPT_DIR/$direct_runner" "${runner_args[@]}" "$@"
