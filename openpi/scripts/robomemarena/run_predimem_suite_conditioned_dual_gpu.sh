#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: $0 sequence|transferring|counting|occlusion}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "${SUITE}" in
  sequence)
    TASK_IDS="1,2,3,22"
    MEMORY_SCOPE_TASK_IDS="${TASK_IDS}"
    MODE="v0"
    DEFAULT_HEAD_VARIANTS="upper"
    GATE_PATH=""
    DEFAULT_AOSS_ROOT="/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32"
    DEFAULT_SEED="50"
    ;;
  transferring)
    TASK_IDS="18,19,25,26"
    MEMORY_SCOPE_TASK_IDS="${TASK_IDS}"
    MODE="v1"
    DEFAULT_HEAD_VARIANTS="fusion"
    GATE_PATH=""
    DEFAULT_AOSS_ROOT="/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32"
    DEFAULT_SEED="50"
    ;;
  counting)
    TASK_IDS="6,7,8,9,10,15,16"
    MEMORY_SCOPE_TASK_IDS="${TASK_IDS}"
    MODE="v1"
    DEFAULT_HEAD_VARIANTS="fusion"
    GATE_PATH="${ROOT}/configs/robomemarena/arena_counting_v1_1_gate.json"
    DEFAULT_AOSS_ROOT="/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32"
    DEFAULT_SEED="50"
    ;;
  occlusion)
    TASK_IDS="4,5,11,12,13,14,17,20,21,23,24"
    MEMORY_SCOPE_TASK_IDS="${TASK_IDS}"
    MODE="v1"
    DEFAULT_HEAD_VARIANTS="fusion"
    GATE_PATH="${ROOT}/configs/robomemarena/arena_occlusion_v1_1_gate.json"
    DEFAULT_AOSS_ROOT="/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32"
    DEFAULT_SEED="50"
    ;;
  *)
    echo "Unknown suite: ${SUITE}" >&2
    exit 2
    ;;
esac

if [[ "${MEMORY_GUIDANCE_DISABLE_SUITE_GATE:-0}" == "1" ]]; then
  GATE_PATH=""
fi

HEAD_VARIANTS="${HEAD_VARIANTS:-${DEFAULT_HEAD_VARIANTS}}"

UPPER_GUIDANCE_ENABLED="${UPPER_GUIDANCE_ENABLED:-0}"
UPPER_GUIDANCE_HEAD="${UPPER_GUIDANCE_HEAD:-${DEFAULT_AOSS_ROOT}/heads/upper/best.pt}"
UPPER_GUIDANCE_MANIFEST="${UPPER_GUIDANCE_MANIFEST:-${DEFAULT_AOSS_ROOT}/features/upper_stride10.jsonl}"
UPPER_GUIDANCE_FEATURES="${UPPER_GUIDANCE_FEATURES:-${DEFAULT_AOSS_ROOT}/features/upper_stride10/upper_features.npy}"
UPPER_GUIDANCE_TOP_K="${UPPER_GUIDANCE_TOP_K:-16}"
UPPER_GUIDANCE_TEMPERATURE="${UPPER_GUIDANCE_TEMPERATURE:-0.07}"
UPPER_GUIDANCE_MIN_TASK_PURITY="${UPPER_GUIDANCE_MIN_TASK_PURITY:-0.5}"
UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE="${UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE:-0.8}"
UPPER_GUIDANCE_INTERPOLATION="${UPPER_GUIDANCE_INTERPOLATION:-0.8}"
UPPER_GUIDANCE_BANK_PER_PRIMITIVE="${UPPER_GUIDANCE_BANK_PER_PRIMITIVE:-16}"
UPPER_GUIDANCE_BANK_SEED="${UPPER_GUIDANCE_BANK_SEED:-17}"
UPPER_GUIDANCE_DEVICE="${UPPER_GUIDANCE_DEVICE:-cpu}"

if [[ -n "${GATE_PATH}" && ! -f "${GATE_PATH}" ]]; then
  echo "Missing suite gate: ${GATE_PATH}" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-${RUN_ROOT_BASE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_suite_conditioned}/${SUITE}_${MODE}_seed${SEED:-${DEFAULT_SEED}}_$(date -u +%Y%m%d_%H%M%S)}"

export AOSS_ROOT="${AOSS_ROOT:-${DEFAULT_AOSS_ROOT}}"
export VLA_CKPT="${VLA_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}"
export VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_all26_reactive}"
export ACTION_STATS="${ACTION_STATS:-${VLA_CKPT}/assets/robomemarena/all26_pi05_reactive/norm_stats.json}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data}"
export TASK_IDS MODE HEAD_VARIANTS RUN_ROOT
export MEMORY_ALLOWED_TASK_IDS="${MEMORY_SCOPE_TASK_IDS}"
export MEMORY_ADMISSION="success"
export MEMORY_TOP_K="${MEMORY_TOP_K:-8}"
export MEMORY_META_BASENAME="${MEMORY_META_BASENAME:-gpm_memory_meta_dense_frame_v3.pt}"
export MEMORY_ALIGNMENT_TAG="${MEMORY_ALIGNMENT_TAG:-dense_frame_v3-suite-selected}"
export MEMORY_ACTION_ALIGNMENT="${MEMORY_ACTION_ALIGNMENT:-dense_frame_v3}"
export MEMORY_EXACT_TASK_GATE="${MEMORY_EXACT_TASK_GATE:-1}"
export MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}"
export MEMORY_GUIDANCE_TOTAL_NORM_CAP="${MEMORY_GUIDANCE_TOTAL_NORM_CAP:-0.20}"
export MEMORY_GUIDANCE_SUITE_GATE_PATH="${GATE_PATH}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-51}"
export SEED="${SEED:-${DEFAULT_SEED}}"
export UPPER_GPU="${UPPER_GPU:-0}"
export LOWER_GPU="${LOWER_GPU:-1}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export RECORD_MEMORY_DATA="${RECORD_MEMORY_DATA:-0}"
export MEMORY_TRACE_LEVEL="${MEMORY_TRACE_LEVEL:-light}"
export PREDIMEM_COMPONENT_TIMING="${PREDIMEM_COMPONENT_TIMING:-1}"
export UPPER_GUIDANCE_ENABLED="${UPPER_GUIDANCE_ENABLED}"
export UPPER_GUIDANCE_HEAD="${UPPER_GUIDANCE_HEAD}"
export UPPER_GUIDANCE_MANIFEST="${UPPER_GUIDANCE_MANIFEST}"
export UPPER_GUIDANCE_FEATURES="${UPPER_GUIDANCE_FEATURES}"
export UPPER_GUIDANCE_TOP_K="${UPPER_GUIDANCE_TOP_K}"
export UPPER_GUIDANCE_TEMPERATURE="${UPPER_GUIDANCE_TEMPERATURE}"
export UPPER_GUIDANCE_MIN_TASK_PURITY="${UPPER_GUIDANCE_MIN_TASK_PURITY}"
export UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE="${UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE}"
export UPPER_GUIDANCE_INTERPOLATION="${UPPER_GUIDANCE_INTERPOLATION}"
export UPPER_GUIDANCE_BANK_PER_PRIMITIVE="${UPPER_GUIDANCE_BANK_PER_PRIMITIVE}"
export UPPER_GUIDANCE_BANK_SEED="${UPPER_GUIDANCE_BANK_SEED}"
export UPPER_GUIDANCE_DEVICE="${UPPER_GUIDANCE_DEVICE}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" "${MODE}"
