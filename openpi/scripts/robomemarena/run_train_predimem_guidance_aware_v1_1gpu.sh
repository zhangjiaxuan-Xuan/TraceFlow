#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"
HEAD_ROOT="${HEAD_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5_balanced_v2_4096/predimem_4096_fp32}"
FEATURE_ROOT="${FEATURE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32/features}"
CACHE_ROOT="${CACHE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/guidance_aware_v1/predimem_full26_fusion4096_topk16}"
EXP_NAME="${EXP_NAME:-adapter_fusion_v1_seed7}"
CHECKPOINT_DIR="/path/to/storage/datasets/robotics/RoboMemArena/derived/guidance_aware_v1/checkpoints/predimem_all26_guidance_aware_v1/${EXP_NAME}"
DATASET_ROOT="${HF_LEROBOT_HOME:-/path/to/local/data/huggingface/lerobot}/robomemarena/all26_pi05_reactive"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/path/to/local/data/huggingface/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

required=(
  "${PYTHON}"
  "${FEATURE_ROOT}/anchors_stride5.jsonl"
  "${FEATURE_ROOT}/lower_tdense5/pooled_prefix.npy"
  "${FEATURE_ROOT}/upper_dense5_held/upper_features.npy"
  "${FEATURE_ROOT}/upper_dense5_held/upper_age.npy"
  "${HEAD_ROOT}/heads/fusion/best.pt"
  "${HEAD_ROOT}/memory/fusion/gpm_memory.index"
  "${HEAD_ROOT}/memory/shared/gpm_memory_actions.npz"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing required artifact: ${path}" >&2; exit 2; }
done

if [[ ! -f "${CACHE_ROOT}/manifest.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/robomemarena/build_guidance_training_cache.py" \
    --manifest "${FEATURE_ROOT}/anchors_stride5.jsonl" \
    --lower-features "${FEATURE_ROOT}/lower_tdense5/pooled_prefix.npy" \
    --upper-features "${FEATURE_ROOT}/upper_dense5_held/upper_features.npy" \
    --upper-age "${FEATURE_ROOT}/upper_dense5_held/upper_age.npy" \
    --head "${HEAD_ROOT}/heads/fusion/best.pt" \
    --index "${HEAD_ROOT}/memory/fusion/gpm_memory.index" \
    --actions "${HEAD_ROOT}/memory/shared/gpm_memory_actions.npz" \
    --output "${CACHE_ROOT}" --top-k 16 --search-k "${CACHE_SEARCH_K:-2048}" \
    --batch-size "${CACHE_BUILD_BATCH_SIZE:-4096}" --device cuda:0
fi

if [[ "${CACHE_ONLY:-0}" == "1" ]]; then
  echo "Guidance cache complete: ${CACHE_ROOT}"
  exit 0
fi

[[ -f "${DATASET_ROOT}/meta/info.json" ]] || { echo "Missing LeRobot dataset: ${DATASET_ROOT}" >&2; exit 2; }

export OPENPI_TRAIN_LOG="${CACHE_ROOT}/train_${EXP_NAME}.log"
export OPENPI_METRICS_LOG="${CACHE_ROOT}/metrics_${EXP_NAME}.jsonl"

args=(
  "${ROOT}/scripts/train_pytorch.py" predimem_all26_guidance_aware_v1
  --exp-name "${EXP_NAME}"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  args+=(--resume)
elif [[ -d "${CHECKPOINT_DIR}" ]] && [[ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Checkpoint directory is non-empty; set RESUME=1 or choose a new EXP_NAME: ${CHECKPOINT_DIR}" >&2
  exit 2
else
  args+=(--overwrite)
fi
exec "${PYTHON}" "${args[@]}"
