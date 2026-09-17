#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export MEMORY_PRIOR_GUIDANCE_VERSION=v3_prior_only
exec bash "${SCRIPT_DIR}/run_libero10_prior_anchor_bn_batch8_env16.sh" "$@"
