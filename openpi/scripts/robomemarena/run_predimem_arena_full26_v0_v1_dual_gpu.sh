#!/usr/bin/env bash
set -euo pipefail

# Runs one exact full-26-task positive-bank evaluation. V0 and V1 are
# intentionally separate invocations so they can be submitted independently
# and resumed without coupling their run state.
MODE="${1:?Usage: $0 v0|v1}"
case "${MODE}" in v0|v1) ;; *) echo "MODE must be v0 or v1" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON=(
  "AOSS_ROOT=${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32}"
  "VLA_CKPT=${VLA_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}"
  "VLA_CONFIG=${VLA_CONFIG:-pi05_robomemarena_all26_reactive}"
  "ACTION_STATS=${ACTION_STATS:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch/assets/robomemarena/all26_pi05_reactive/norm_stats.json}"
  "OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data}"
  "TASK_IDS=${TASK_IDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}"
  "EPISODES_PER_TASK=${EPISODES_PER_TASK:-50}"
  "SEED=${SEED:-7}"
  "MEMORY_ADMISSION=success" "MEMORY_TOP_K=${MEMORY_TOP_K:-8}"
  "MEMORY_META_BASENAME=${MEMORY_META_BASENAME:-gpm_memory_meta_dense_frame_v3.pt}"
  "MEMORY_ALIGNMENT_TAG=${MEMORY_ALIGNMENT_TAG:-dense_frame_v3}"
  "MEMORY_ACTION_ALIGNMENT=${MEMORY_ACTION_ALIGNMENT:-dense_frame_v3}"
  "MEMORY_EXACT_TASK_GATE=${MEMORY_EXACT_TASK_GATE:-1}"
  "HEAD_VARIANTS=fusion" "UPPER_BATCH_SIZE=${UPPER_BATCH_SIZE:-32}"
  "LOWER_BATCH_SIZE=${LOWER_BATCH_SIZE:-32}" "ENV_WORKERS=${ENV_WORKERS:-64}"
  "UPPER_GPU=${UPPER_GPU:-0}" "LOWER_GPU=${LOWER_GPU:-1}"
  "SAVE_VIDEO=${SAVE_VIDEO:-1}" "RECORD_MEMORY_DATA=${RECORD_MEMORY_DATA:-0}"
)

root="${RUN_ROOT:-${RUN_ROOT_BASE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval}/${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
env "${COMMON[@]}" MODE="${MODE}" RUN_ROOT="${root}" \
  bash "${ROOT}/scripts/robomemarena/run_predimem_arena_memory_ablation_dual_gpu.sh" success
