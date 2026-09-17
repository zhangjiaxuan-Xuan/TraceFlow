#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"

SOURCE_ROOT="${SOURCE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v2/pi05_finetuned}"
MANIFEST="${MANIFEST:-${SOURCE_ROOT}/features/anchors16.jsonl}"
UPPER_FEATURES="${UPPER_FEATURES:-${SOURCE_ROOT}/features/upper/upper_features.npy}"
POLICY_DIR="${POLICY_DIR:-${ROOT}/checkpoints/pi05_robomemarena_extra8_reactive/extra8_reactive_fullft_seed42_finite_loader/30000_pytorch}"
CONFIG_NAME="${CONFIG_NAME:-pi05_robomemarena_extra8_reactive}"
FEATURE_DIR="${FEATURE_DIR:-${OUTPUT_ROOT}/features/lower}"

for path in \
  "${OPENPI_PY}" "${MANIFEST}" "${UPPER_FEATURES}" "${POLICY_DIR}/model.safetensors"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OPENPI_TORCH_COMPILE=0

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed: Pi self-head source=${POLICY_DIR} output=${OUTPUT_ROOT}"
  exit 0
fi

mkdir -p "${FEATURE_DIR}"

"${OPENPI_PY}" "${ROOT}/scripts/memory/cache_prior_head_features.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${FEATURE_DIR}" \
  --policy-dir "${POLICY_DIR}" \
  --config-name "${CONFIG_NAME}" \
  --device cuda \
  --batch-size "${FEATURE_BATCH_SIZE:-16}" \
  --feature-dim 2048

common=(
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py"
  --manifest "${MANIFEST}"
  --lower-features "${FEATURE_DIR}/pooled_prefix.npy"
  --upper-features "${UPPER_FEATURES}"
  --device cuda:0
  --variants lower
  --hidden-dim 1024
  --bank-budget 16
  --bank-dtype fp32
  --resume
  --epochs "${HEAD_EPOCHS:-30}"
  --steps-per-epoch "${HEAD_STEPS_PER_EPOCH:-100}"
  --batch-size "${HEAD_BATCH_SIZE:-512}"
)

"${common[@]}" \
  --out-dim 2048 \
  --output-root "${OUTPUT_ROOT}/original2048_fp32"

"${common[@]}" \
  --out-dim 256 \
  --output-root "${OUTPUT_ROOT}/capacity256_fp32"

echo "Pi0.5 self-model retrieval heads complete: ${OUTPUT_ROOT}"
