#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/path/to/user/miniforge3/envs/predimem-vlm/bin/python}"
GPU="${GPU:-0}"
SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32}"
CHECKPOINT="${VLM_CKPT:-/path/to/local/data/robomemarena/models/PrediMem/vlm_tasks1to26_ckpt74500}"
OUTPUT="${OUTPUT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/guidance_aware_v2/checkpoints/predimem_upper_knnlm_fullft}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${ROOT}/../RoboMemArena:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

required=(
  "${PYTHON}" "${CHECKPOINT}/config.json" "${CHECKPOINT}/model.safetensors"
  "${SELF_ROOT}/features/upper_stride10.jsonl"
  "${SELF_ROOT}/features/upper_stride10/upper_features.npy"
  "${SELF_ROOT}/heads/upper/best.pt"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing required Upper-aware artifact: ${path}" >&2; exit 2; }
done

args=(
  "${ROOT}/scripts/robomemarena/train_predimem_upper_guidance_aware.py"
  --checkpoint "${CHECKPOINT}"
  --manifest "${SELF_ROOT}/features/upper_stride10.jsonl"
  --head "${SELF_ROOT}/heads/upper/best.pt"
  --upper-features "${SELF_ROOT}/features/upper_stride10/upper_features.npy"
  --output "${OUTPUT}"
  --batch-size "${BATCH_SIZE:-1}"
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-16}"
  --workers "${WORKERS:-16}"
  --steps "${STEPS:-10000}"
  --save-every "${SAVE_EVERY:-1000}"
  --learning-rate "${LEARNING_RATE:-2e-5}"
  --seed "${SEED:-42}"
)
[[ -n "${MAX_ITEMS:-}" ]] && args+=(--max-items "${MAX_ITEMS}")
exec "${PYTHON}" "${args[@]}"
