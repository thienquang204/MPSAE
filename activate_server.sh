#!/usr/bin/env bash
# Source this file to use the native server environment and CUDA compiler.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Use: source activate_server.sh\n' >&2
    exit 2
fi
_mpsae_server_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$_mpsae_server_root/.venv/bin/activate" || ! -f "$_mpsae_server_root/env.sh" ]]; then
    printf 'Server environment missing. Run: bash setup_server.sh\n' >&2
    unset _mpsae_server_root
    return 1
fi
source "$_mpsae_server_root/.venv/bin/activate"
source "$_mpsae_server_root/env.sh"
unset _mpsae_server_root
