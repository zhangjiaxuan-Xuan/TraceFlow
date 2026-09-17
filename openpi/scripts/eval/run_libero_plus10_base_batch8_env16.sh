#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OPENPI_ROOT}/logs/libero_plus10_full_pi05_base_batch8_env16_${RUN_ID}}"

# Keep the Base control identical to the full LIBERO-Plus guidance run except
# that all memory and guidance paths are disabled.
export GPU
export MODE=base
export EVAL_SCOPE=full
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
export INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-2000}"
export LIBERO_CLIENTS_PER_SUITE="${LIBERO_CLIENTS_PER_SUITE:-16}"
export LIBERO_SHARD_AXIS="${LIBERO_SHARD_AXIS:-tasks}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
export MAX_ENV_STEPS="${MAX_ENV_STEPS:-600}"
export NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
export SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
export SAVE_EPISODE_DATA="${SAVE_EPISODE_DATA:-0}"
export LOG_DIR

exec bash scripts/eval/run_libero_plus10_v1_bcpi_lowcost.sh
