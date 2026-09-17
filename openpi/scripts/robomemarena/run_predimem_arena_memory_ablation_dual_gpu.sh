#!/usr/bin/env bash
set -euo pipefail

# One stable entry point for Arena memory ablations.  Admission values:
#   none    : no memory baseline
#   success : positive demonstration bank only
#   failure : negative failure bank only
#   both    : isolated positive and negative banks
ADMISSION="${1:?Usage: $0 none|success|failure|both}"
case "${ADMISSION}" in none|success|failure|both) ;; *)
  echo "Admission must be none, success, failure, or both." >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_IDS="${TASK_IDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}"
MODE="${MODE:-v1}"
HEAD_VARIANTS="${HEAD_VARIANTS:-fusion}"
UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
ENV_WORKERS="${ENV_WORKERS:-96}"
MEMORY_TOP_K="${MEMORY_TOP_K:-8}"
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-8}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
RECORD_MEMORY_DATA="${RECORD_MEMORY_DATA:-0}"
UPPER_GPU="${UPPER_GPU:-0}"
LOWER_GPU="${LOWER_GPU:-1}"

if [[ "${ADMISSION}" == "none" ]]; then
  # True no-memory baseline. The child runner supports this explicitly; do
  # not silently turn it into a positive-memory run.
  MEMORY_ADMISSION=none MODE="${MODE}" \
    AOSS_ROOT="${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}" \
    VLA_CKPT="${VLA_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}" \
    VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_extra8_reactive}" \
    ACTION_STATS="${ACTION_STATS:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch/assets/robomemarena/extra8_pi05_reactive/norm_stats.json}" \
    HEAD_VARIANTS="${HEAD_VARIANTS}" TASK_IDS="${TASK_IDS}" \
    EPISODES_PER_TASK="${EPISODES_PER_TASK}" UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE}" \
    LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE}" ENV_WORKERS="${ENV_WORKERS}" \
    UPPER_GPU="${UPPER_GPU}" LOWER_GPU="${LOWER_GPU}" SAVE_VIDEO="${SAVE_VIDEO}" \
    RECORD_MEMORY_DATA="${RECORD_MEMORY_DATA}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
    RUN_ROOT="${RUN_ROOT:-}" \
    bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" "${MODE}"
  exit $?
fi

MEMORY_ADMISSION="${ADMISSION}" \
MEMORY_TOP_K="${MEMORY_TOP_K}" NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K}" \
HEAD_VARIANTS="${HEAD_VARIANTS}" TASK_IDS="${TASK_IDS}" MODE="${MODE}" \
UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE}" \
ENV_WORKERS="${ENV_WORKERS}" UPPER_GPU="${UPPER_GPU}" LOWER_GPU="${LOWER_GPU}" \
EPISODES_PER_TASK="${EPISODES_PER_TASK}" SAVE_VIDEO="${SAVE_VIDEO}" \
RECORD_MEMORY_DATA="${RECORD_MEMORY_DATA}" \
bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" "${MODE}"
