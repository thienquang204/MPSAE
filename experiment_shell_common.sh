#!/usr/bin/env bash
# Shared by host and direct entry points. Source this file; do not execute it.

select_experiment_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        command -v -- "$PYTHON" >/dev/null 2>&1 \
            || { echo "PYTHON executable not found: $PYTHON" >&2; return 2; }
        python_bin="$PYTHON"
    elif command -v python3 >/dev/null 2>&1; then
        python_bin=python3
    elif command -v python >/dev/null 2>&1; then
        python_bin=python
    else
        echo "Python is required for planning; install Python 3 or set PYTHON to its executable." >&2
        return 2
    fi
}

load_experiment_env() {
    local requested_file="${ENV_FILE:-${1:-}}"
    [[ -n "$requested_file" ]] || return 0
    case "$requested_file" in
        /*) ;;
        *) requested_file="$SCRIPT_DIR/$requested_file" ;;
    esac
    if [[ ! -f "$requested_file" ]]; then
        if [[ -n "${ENV_FILE:-}" ]]; then
            echo "ENV_FILE does not exist: $requested_file" >&2
            return 2
        fi
        return 0
    fi

    # Explicit environment assignments override file defaults, including a
    # command such as REPAIR_STAGE=pilot ENV_FILE=repair.env bash launcher.sh.
    # Restore through assignment, never eval, so spaces and credentials stay data.
    local variable declaration index had_allexport=0
    local -a saved_names=() saved_values=()
    while IFS= read -r variable; do
        declaration="$(declare -p "$variable" 2>/dev/null)" || continue
        # Read-only shell variables cannot be restored and are not settings.
        [[ "$declaration" =~ ^declare\ -[^[:space:]]*r ]] && continue
        saved_names+=("$variable")
        saved_values+=("${!variable}")
    done < <(compgen -e)
    [[ $- == *a* ]] && had_allexport=1
    set -a
    # ENV_FILE is a trusted Bash assignment file, just like the existing .env.
    # shellcheck disable=SC1090
    source "$requested_file"
    if [[ "$had_allexport" == 0 ]]; then set +a; fi
    for index in "${!saved_names[@]}"; do
        export "${saved_names[$index]}=${saved_values[$index]}"
    done
    export ENV_FILE="$requested_file"
}

resolve_repair_stage() {
    REPAIR_STAGE="${REPAIR_STAGE:-plan}"
    repair_cli_args=()
    if [[ $# -gt 0 && "$1" != --* ]]; then
        REPAIR_STAGE="$1"
        shift
    fi
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --stage)
                if [[ $# -lt 2 || "$2" == --* ]]; then
                    echo "--stage needs a value" >&2; return 2
                fi
                REPAIR_STAGE="$2"; shift 2 ;;
            --stage=*) REPAIR_STAGE="${1#*=}"; shift ;;
            *) repair_cli_args+=("$1"); shift ;;
        esac
    done
    case "$REPAIR_STAGE" in
        plan|preflight|prepare|pilot|check|warm|triage|attribution|wide|lock|holdout|confirm|timing|arm) ;;
        *) echo "Unknown repair stage: $REPAIR_STAGE" >&2; return 2 ;;
    esac
    export REPAIR_STAGE
}
