#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export MEMORY_PRIOR_GUIDANCE_VERSION=v2
export SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"
exec bash "${SCRIPT_DIR}/run_libero10_prior_substep_batch8.sh" "$@"
