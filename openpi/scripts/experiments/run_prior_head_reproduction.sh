#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/local/data/huggingface/hub/datasets--yifengzhu-hf--LIBERO-datasets}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
RUN_ROOT="${RUN_ROOT:-${OPENPI_ROOT}/artifacts/prior_head_reproduction}"
STAGE="${STAGE:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
export OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-0}"

MANIFEST="${RUN_ROOT}/manifest.jsonl"
FEATURE_DIR="${RUN_ROOT}/features"
HEAD_DIR="${RUN_ROOT}/head_task_file"
MEMORY_DIR="${RUN_ROOT}/memory_task_file"

cd "${OPENPI_ROOT}"
mkdir -p "${RUN_ROOT}"

run_manifest() {
  if [[ ! -f "${MANIFEST}" ]]; then
    "${PYTHON_BIN}" scripts/data/prepare_prior_head_manifest.py \
      --dataset-root "${DATASET_ROOT}" \
      --output "${MANIFEST}"
  fi
}

run_cache() {
  "${PYTHON_BIN}" scripts/memory/cache_prior_head_features.py \
    --manifest "${MANIFEST}" \
    --output-dir "${FEATURE_DIR}" \
    --policy-dir "${POLICY_DIR}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}"
}

run_train() {
  "${PYTHON_BIN}" scripts/memory/train_prior_head.py \
    --manifest "${MANIFEST}" \
    --feature-dir "${FEATURE_DIR}" \
    --output-dir "${HEAD_DIR}" \
    --device "${DEVICE}"
}

run_memory() {
  if [[ -f "${MEMORY_DIR}/gpm_memory.index" && \
        -f "${MEMORY_DIR}/gpm_memory_meta.pt" && \
        -f "${MEMORY_DIR}/gpm_memory_actions.npz" && \
        -f "${MEMORY_DIR}/build_summary.json" ]]; then
    echo "Memory artifacts already complete: ${MEMORY_DIR}"
    return
  fi
  "${PYTHON_BIN}" scripts/memory/build_prior_head_memory.py \
    --manifest "${MANIFEST}" \
    --feature-dir "${FEATURE_DIR}" \
    --checkpoint "${HEAD_DIR}/last.pt" \
    --output-dir "${MEMORY_DIR}" \
    --device "${DEVICE}"
}

case "${STAGE}" in
  manifest) run_manifest ;;
  cache) run_manifest; run_cache ;;
  train) run_train ;;
  memory) run_memory ;;
  all) run_manifest; run_cache; run_train; run_memory ;;
  *) echo "Unknown STAGE=${STAGE}; expected manifest|cache|train|memory|all" >&2; exit 2 ;;
esac
