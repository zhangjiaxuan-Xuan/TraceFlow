#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ARENA_ROOT:-${ROOT}/../RoboMemArena}"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
VLM_PY="${VLM_PY:-/path/to/user/miniforge3/envs/predimem-vlm/bin/python}"
GPU="${GPU:-0}"

SOURCE_ROOT="${SOURCE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v3/predimem_2048_fp32}"
MANIFEST="${MANIFEST:-${SOURCE_ROOT}/features/anchors16.jsonl}"
VLA_CKPT="${VLA_CKPT:-${SOURCE_ROOT}/checkpoints/vla_alltask_pytorch}"
VLM_CKPT="${VLM_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/models/PrediMem/vlm_tasks1to26_ckpt74500}"
TASK_CONFIG="${TASK_CONFIG:-${ARENA_ROOT}/evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json}"
UPPER_DIR="${UPPER_DIR:-${OUTPUT_ROOT}/features/upper}"
LOWER_DIR="${LOWER_DIR:-${OUTPUT_ROOT}/features/lower}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/train_joint_heads.log}"

for path in \
  "${OPENPI_PY}" "${VLM_PY}" "${MANIFEST}" \
  "${VLA_CKPT}/model.safetensors" "${VLM_CKPT}/config.json" "${TASK_CONFIG}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed: PrediMem joint upper->subtask->lower training output=${OUTPUT_ROOT}"
  exit 0
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/packages/openpi-client/src:${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export OPENPI_TORCH_COMPILE=0
export TOKENIZERS_PARALLELISM=false

echo "[$(date -Is)] Joint head pipeline start/resume: output=${OUTPUT_ROOT} gpu=${GPU}"
echo "[$(date -Is)] Stage 1/3: causal Upper feature and subtask cache"

"${VLM_PY}" "${ROOT}/scripts/robomemarena/cache_predimem_upper_features.py" \
  --manifest "${MANIFEST}" \
  --checkpoint "${VLM_CKPT}" \
  --task-config "${TASK_CONFIG}" \
  --output-dir "${UPPER_DIR}" \
  --device cuda:0 \
  --batch-size "${UPPER_BATCH_SIZE:-2}" \
  --workers "${DATA_WORKERS:-16}" \
  --n-recent "${N_RECENT:-5}" \
  --merge-distance "${D_MERGE:-6}" \
  --keyframe-max "${K_MAX:-0}" \
  --max-new-tokens "${UPPER_MAX_NEW_TOKENS:-256}" \
  --feature-dim 4096 \
  --generate-subtasks

echo "[$(date -Is)] Stage 1/3 complete"
echo "[$(date -Is)] Stage 2/3: Upper-conditioned Lower feature cache"

"${OPENPI_PY}" "${ROOT}/scripts/memory/cache_prior_head_features.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${LOWER_DIR}" \
  --policy-dir "${VLA_CKPT}" \
  --config-name pi05_robomemarena_extra8_reactive \
  --device cuda \
  --batch-size "${LOWER_BATCH_SIZE:-16}" \
  --feature-dim 2048 \
  --prompt-overrides "${UPPER_DIR}/subtasks.jsonl"

echo "[$(date -Is)] Stage 2/3 complete"
echo "[$(date -Is)] Stage 3/3: train lower, upper, and fusion retrieval heads"

"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py" \
  --manifest "${MANIFEST}" \
  --lower-features "${LOWER_DIR}/pooled_prefix.npy" \
  --upper-features "${UPPER_DIR}/upper_features.npy" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --variants lower,upper,fusion \
  --hidden-dim 1024 \
  --out-dim 2048 \
  --bank-budget 16 \
  --bank-dtype fp32 \
  --require-joint-conditioning \
  --resume \
  --epochs "${HEAD_EPOCHS:-30}" \
  --steps-per-epoch "${HEAD_STEPS_PER_EPOCH:-100}" \
  --batch-size "${HEAD_BATCH_SIZE:-512}"

echo "[$(date -Is)] PrediMem joint-conditioned 2048D heads complete: ${OUTPUT_ROOT}"
