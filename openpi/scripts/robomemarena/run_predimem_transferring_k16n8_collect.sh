#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_self_cl_k16n8_seed7/collector}"
resume=0
[[ -f "${RUN_ROOT}/run_config.json" ]] && resume=1
UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" SEED="${SEED:-7}" \
UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
ENV_WORKERS="${ENV_WORKERS:-96}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}" \
MEMORY_TOP_K=16 NEGATIVE_MEMORY_TOP_K=8 SAVE_VIDEO="${SAVE_VIDEO:-1}" \
RECORD_MEMORY_DATA=1 MEMORY_TRACE_LEVEL=full RUN_ROOT="${RUN_ROOT}" RESUME="${resume}" \
  bash "${ROOT}/scripts/robomemarena/run_predimem_arena_transferring_ablation_dual_gpu.sh" collect
