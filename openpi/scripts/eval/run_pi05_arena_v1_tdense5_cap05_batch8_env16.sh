#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ENV_WORKERS="${ENV_WORKERS:-16}"
export INFERENCE_BATCH_SIZE=8
export MEMORY_GUIDANCE_NORM_CAP=0.50
export MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.50
export RUN_ROOT="${RUN_ROOT:-${ROOT}/logs/robomemarena_extra8_pi05_v1_tdense5_cap05_batch8_env${ENV_WORKERS}_$(date -u +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/scripts/eval/run_pi05_arena_v1_dense5_budget1024_batch8.sh"
