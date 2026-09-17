#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"
SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}"
CACHE_ROOT="${CACHE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/guidance_aware_v1/predimem_full26_lower_tdense5_topk16}"
DATASET_ROOT="${HF_LEROBOT_HOME:-/path/to/local/data/huggingface/lerobot}/robomemarena/all26_pi05_subtask"
EXP_NAME="${EXP_NAME:-lower_fullft_v1_tdense5_seed42}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/path/to/local/data/huggingface/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${SELF_ROOT}/runtime/openpi_data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MANIFEST="${SELF_ROOT}/features/anchors_stride5.jsonl"
LOWER_FEATURES="${SELF_ROOT}/features/lower_tdense5/pooled_prefix.npy"
UPPER_FEATURES="${SELF_ROOT}/features/upper_dense5_held/upper_features.npy"
UPPER_AGE="${SELF_ROOT}/features/upper_dense5_held/upper_age.npy"
HEAD="${SELF_ROOT}/heads/lower/best.pt"
INDEX="${SELF_ROOT}/memory/lower/gpm_memory.index"
ACTIONS="${SELF_ROOT}/memory/shared/gpm_memory_actions.npz"

for path in "${PYTHON}" "${MANIFEST}" "${LOWER_FEATURES}" "${UPPER_FEATURES}" "${UPPER_AGE}" "${HEAD}" "${INDEX}" "${ACTIONS}"; do
  [[ -f "${path}" ]] || { echo "Missing required Tdense5 artifact: ${path}" >&2; exit 2; }
done

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/robomemarena/convert_robomemarena_subtasks_to_lerobot.py"
fi

if [[ ! -f "${CACHE_ROOT}/manifest.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/robomemarena/build_guidance_training_cache.py" \
    --manifest "${MANIFEST}" --lower-features "${LOWER_FEATURES}" \
    --upper-features "${UPPER_FEATURES}" --upper-age "${UPPER_AGE}" \
    --head "${HEAD}" --index "${INDEX}" --actions "${ACTIONS}" \
    --output "${CACHE_ROOT}" --top-k 16 --search-k "${CACHE_SEARCH_K:-2048}" \
    --batch-size "${CACHE_BUILD_BATCH_SIZE:-4096}" --device cuda:0
fi

CONFIG=predimem_all26_subtask_guidance_aware_v1_full
NORM_STATS="${SELF_ROOT}/../../predimem_dual_tower/checkpoints/vla_alltask_pytorch/assets/robomemarena/all26_pi05_reactive/norm_stats.json"
NORM_STATS="$(realpath -m "${NORM_STATS}")"
[[ -f "${NORM_STATS}" ]] || { echo "Missing official all26 normalization stats: ${NORM_STATS}" >&2; exit 2; }

mode=(--overwrite)
[[ "${RESUME:-0}" == 1 ]] && mode=(--resume)
export OPENPI_TRAIN_LOG="${CACHE_ROOT}/train_${EXP_NAME}.log"
export OPENPI_METRICS_LOG="${CACHE_ROOT}/metrics_${EXP_NAME}.jsonl"
exec "${PYTHON}" "${ROOT}/scripts/train_pytorch.py" "${CONFIG}" --exp-name "${EXP_NAME}" "${mode[@]}"
