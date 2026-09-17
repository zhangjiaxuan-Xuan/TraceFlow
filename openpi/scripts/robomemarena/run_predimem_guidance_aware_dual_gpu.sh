#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}"
ARM="${1:?Usage: $0 lower|upper|both}"
case "${ARM}" in lower|upper|both) ;; *) echo "ARM must be lower, upper, or both" >&2; exit 2 ;; esac
BASE_UPPER_CKPT="${BASE_UPPER_CKPT:-/path/to/local/data/robomemarena/models/PrediMem/vlm_tasks1to26_ckpt74500}"
BASE_LOWER_CKPT="${BASE_LOWER_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}"
UPPER_AWARE_CKPT="${UPPER_AWARE_CKPT:-}"
LOWER_AWARE_CKPT="${LOWER_AWARE_CKPT:-}"
if [[ "${ARM}" == upper || "${ARM}" == both ]]; then
  [[ -n "${UPPER_AWARE_CKPT}" ]] || { echo "UPPER_AWARE_CKPT is required for ${ARM}" >&2; exit 2; }
else
  UPPER_AWARE_CKPT="${BASE_UPPER_CKPT}"
fi
if [[ "${ARM}" == lower || "${ARM}" == both ]]; then
  [[ -n "${LOWER_AWARE_CKPT}" ]] || { echo "LOWER_AWARE_CKPT is required for ${ARM}" >&2; exit 2; }
else
  LOWER_AWARE_CKPT="${BASE_LOWER_CKPT}"
fi

for path in \
  "${UPPER_AWARE_CKPT}/config.json" \
  "${UPPER_AWARE_CKPT}/model.safetensors" \
  "${LOWER_AWARE_CKPT}/model.safetensors" \
  "${SELF_ROOT}/heads/lower/best.pt" \
  "${SELF_ROOT}/memory/lower/gpm_memory.index" \
  "${SELF_ROOT}/memory/shared/gpm_memory_actions.npz" \
  "${SELF_ROOT}/features/upper_stride10.jsonl" \
  "${SELF_ROOT}/features/upper_stride10/upper_features.npy" \
  "${SELF_ROOT}/heads/upper/best.pt"; do
  [[ -f "${path}" ]] || { echo "Missing guidance-aware evaluation artifact: ${path}" >&2; exit 2; }
done

export AOSS_ROOT="${SELF_ROOT}"
export VLM_CKPT="${UPPER_AWARE_CKPT}"
export VLA_CKPT="${LOWER_AWARE_CKPT}"
export HEAD_VARIANTS="lower"
export MEMORY_META_BASENAME="gpm_memory_meta_dense_frame_v3.pt"
export MEMORY_ALIGNMENT_TAG="aware-fixed-tdense5-stage-frame-v3"
export MEMORY_GUIDANCE_DISABLE_SUITE_GATE=1
export MEMORY_GUIDANCE_NORM_CAP=0.20
export MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20
export UPPER_GUIDANCE_ENABLED=1
export UPPER_GUIDANCE_HEAD="${SELF_ROOT}/heads/upper/best.pt"
export UPPER_GUIDANCE_MANIFEST="${SELF_ROOT}/features/upper_stride10.jsonl"
export UPPER_GUIDANCE_FEATURES="${SELF_ROOT}/features/upper_stride10/upper_features.npy"
export UPPER_GUIDANCE_TOP_K=16
export UPPER_GUIDANCE_TEMPERATURE=0.07
export UPPER_GUIDANCE_MIN_TASK_PURITY=0.5
export UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE=0.8
export UPPER_GUIDANCE_INTERPOLATION=0.8
export UPPER_GUIDANCE_BANK_PER_PRIMITIVE=16
export UPPER_GUIDANCE_BANK_SEED=17
export UPPER_GUIDANCE_DEVICE="${UPPER_GUIDANCE_DEVICE:-cpu}"
export REPLAN_STEPS=10 VLM_INTERVAL=5
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-8}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-8}"
export ENV_WORKERS="${ENV_WORKERS:-16}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-51}"
export SEED="${SEED:-50}"
export RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/guidance_aware_v2/eval/${ARM}_aware_v1_tdense5_$(date -u +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" v1
