#!/usr/bin/env bash
set -euo pipefail

ADMISSION="${1:?Usage: $0 collect|success|failure|both}"
case "${ADMISSION}" in collect|success|failure|both) ;; *) exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_BASE="${RUN_BASE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_memory_eval}"
POS_ROOT="${POS_ROOT:-${RUN_BASE}/../transferring_memory_success}"
NEG_ROOT="${NEG_ROOT:-${RUN_BASE}/../transferring_memory_failure}"

if [[ "${ADMISSION}" == "collect" ]]; then
  MEMORY_ADMISSION=success
  RECORD_MEMORY_DATA=1
  LABEL=collector_success
else
  MEMORY_ADMISSION="${ADMISSION}"
  RECORD_MEMORY_DATA="${RECORD_MEMORY_DATA:-1}"
  LABEL="${ADMISSION}"
fi

export AOSS_ROOT="${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}"
export VLA_CKPT="${VLA_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}"
export VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_extra8_reactive}"
export ACTION_STATS="${ACTION_STATS:-${VLA_CKPT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json}"
if [[ "${ADMISSION}" != "collect" ]]; then
  export POSITIVE_MEMORY_META_PATH="${POSITIVE_MEMORY_META_PATH:-${POS_ROOT}/gpm_memory_meta.pt}"
  export POSITIVE_FAISS_INDEX_PATH="${POSITIVE_FAISS_INDEX_PATH:-${POS_ROOT}/gpm_memory.index}"
  export POSITIVE_MEMORY_ACTIONS_PATH="${POSITIVE_MEMORY_ACTIONS_PATH:-${POS_ROOT}/gpm_memory_actions.npz}"
fi
export NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH:-${NEG_ROOT}/gpm_memory_meta.pt}"
export NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH:-${NEG_ROOT}/gpm_memory.index}"
export NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH:-${NEG_ROOT}/gpm_memory_actions.npz}"
export TASK_IDS="${TASK_IDS:-18,19,25,26}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-96}" HEAD_VARIANTS="${HEAD_VARIANTS:-fusion}"
export UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}"
export MEMORY_TOP_K="${MEMORY_TOP_K:-8}" NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-8}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}" RECORD_MEMORY_DATA
export RUN_ROOT="${RUN_ROOT:-${RUN_BASE}/${LABEL}_$(date -u +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_arena_memory_ablation_dual_gpu.sh" "${MEMORY_ADMISSION}"
