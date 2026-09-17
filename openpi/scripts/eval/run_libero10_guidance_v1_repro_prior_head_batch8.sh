#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
REPRO_HEAD_VARIANT="${REPRO_HEAD_VARIANT:-steps1000}"

case "${REPRO_HEAD_VARIANT}" in
  steps1000)
    HEAD_DIR="${OPENPI_ROOT}/artifacts/prior_head_reproduction/head_task_file"
    MEMORY_DIR="${OPENPI_ROOT}/artifacts/prior_head_reproduction/memory_task_file"
    SOURCE_TAG="repro_prior_steps1000"
    ;;
  epochs20)
    HEAD_DIR="${OPENPI_ROOT}/artifacts/prior_head_reproduction/head_task_file_20epochs"
    MEMORY_DIR="${OPENPI_ROOT}/artifacts/prior_head_reproduction/memory_task_file_20epochs"
    SOURCE_TAG="repro_prior_epochs20"
    ;;
  *)
    echo "REPRO_HEAD_VARIANT must be steps1000 or epochs20, got: ${REPRO_HEAD_VARIANT}" >&2
    exit 2
    ;;
esac

if [[ "${MEMORY_VARIANT:-success}" != "success" ]]; then
  echo "Reproduced Prior Head validation currently requires MEMORY_VARIANT=success." >&2
  echo "The existing failure FAISS index belongs to the released-head embedding space." >&2
  exit 2
fi

export OPENPI_ROOT
export MEMORY_VARIANT=success
export MEMORY_GUIDANCE_VERSION=v1
export MEMORY_SOURCE_TAG="${MEMORY_SOURCE_TAG:-${SOURCE_TAG}}"
export TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${HEAD_DIR}/last.pt}"
export MEMORY_META_PATH="${MEMORY_META_PATH:-${MEMORY_DIR}/gpm_memory_meta.pt}"
export FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${MEMORY_DIR}/gpm_memory.index}"
export MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-${MEMORY_DIR}/gpm_memory_actions.npz}"

exec bash "${SCRIPT_DIR}/run_libero10_guidance_only_batch8.sh" "$@"
