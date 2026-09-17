#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MEMORY_STATIC_ACTION_MODE=observe
export MEMORY_STATIC_ACTION_ARM_DIM="${MEMORY_STATIC_ACTION_ARM_DIM:-6}"
export MEMORY_STATIC_ACTION_THRESHOLD="${MEMORY_STATIC_ACTION_THRESHOLD:-1e-8}"
export MEMORY_STATIC_ACTION_GATE_FRACTION="${MEMORY_STATIC_ACTION_GATE_FRACTION:-0.50}"
export ENV_WORKERS="${ENV_WORKERS:-16}"
export RUN_ROOT="${RUN_ROOT:-${ROOT}/logs/robomemarena_extra8_pi05_v1_sdense5_static_observe_batch8_env${ENV_WORKERS}_$(date -u +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/scripts/eval/run_pi05_arena_v1_dense5_single_budget1024_batch8.sh"
