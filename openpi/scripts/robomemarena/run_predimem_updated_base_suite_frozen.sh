#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: $0 sequence|transferring|counting|occlusion}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ROOT}/../RoboMemArena"
LOCK="${ROOT}/configs/robomemarena/predimem_public_checkpoint_updated_benchmark_v1.json"

case "${SUITE}" in
  sequence) TASK_IDS="1,2,3,22" ;;
  transferring) TASK_IDS="18,19,25,26" ;;
  counting) TASK_IDS="6,7,8,9,10,15,16" ;;
  occlusion) TASK_IDS="4,5,11,12,13,14,17,20,21,23,24" ;;
  *) echo "Unknown suite: ${SUITE}" >&2; exit 2 ;;
esac

OPENPI_PY="/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python"
cd "${ROOT}"
"${OPENPI_PY}" -m optimus_eval.verify_predimem_frozen_protocol \
  --lock "${LOCK}" --arena-root "${ARENA_ROOT}"

export MODE="base"
export TASK_IDS
export HEAD_VARIANTS="base"
export MEMORY_ADMISSION="none"
export MEMORY_ALLOWED_TASK_IDS=""
export EPISODES_PER_TASK="51"
export SEED="50"
export REPLAN_STEPS="10"
export MAX_STEPS="2500"
export NUM_STEPS_WAIT="10"
export POST_GOAL_STEPS="200"
export VLM_INTERVAL="5"
export N_RECENT="5"
export K_MAX="0"
export D_MERGE="6"
export UPPER_GPU="0"
export LOWER_GPU="1"
export UPPER_BATCH_SIZE="32"
export LOWER_BATCH_SIZE="32"
export ENV_WORKERS="64"
export SAVE_VIDEO="1"
export RECORD_MEMORY_DATA="0"
export MEMORY_TRACE_LEVEL="light"
export PREDIMEM_COMPONENT_TIMING="1"
export AOSS_ROOT="/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32"
export VLA_CKPT="/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch"
export VLA_CONFIG="pi05_robomemarena_all26_reactive"
export ACTION_STATS="${VLA_CKPT}/assets/robomemarena/all26_pi05_reactive/norm_stats.json"
export VLM_CKPT="/path/to/local/data/robomemarena/models/PrediMem/vlm_tasks1to26_ckpt74500"
export OPENPI_DATA_HOME="/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data"
export MEMORY_ALIGNMENT_TAG="frozen-updated-benchmark-v1"
export PROTOCOL_LOCK_FILE="${LOCK}"
export RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_frozen_base/${SUITE}_base_seed50_ep51_$(date -u +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" base
