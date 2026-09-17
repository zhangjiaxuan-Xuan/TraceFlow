#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FULL26_ROOT="${FULL26_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32}"
COUNTING_TASKS="6,7,8,9,10,15,16"

export SELF_ROOT="${FULL26_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data}"
export TASK_IDS="${TASK_IDS:-${COUNTING_TASKS}}"
export MEMORY_ALLOWED_TASK_IDS="${MEMORY_ALLOWED_TASK_IDS:-${COUNTING_TASKS}}"
export UPPER_GUIDANCE_ALLOWED_TASK_IDS="${UPPER_GUIDANCE_ALLOWED_TASK_IDS:-${COUNTING_TASKS}}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export SEED="${SEED:-50}"
export UPPER_GPU="${UPPER_GPU:-0}"
export LOWER_GPU="${LOWER_GPU:-1}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_extra8_history_knn_self.sh" "$@"
