#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
CL_GUIDANCE_VERSION="${CL_GUIDANCE_VERSION:-v0}"
case "${CL_GUIDANCE_VERSION}" in
  v0) BASE_EVAL_SCRIPT="${SCRIPT_DIR}/../eval/run_libero10_guidance_only_v0_batch8.sh" ;;
  v1) BASE_EVAL_SCRIPT="${SCRIPT_DIR}/../eval/run_libero10_guidance_only_batch8.sh" ;;
  *) echo "CL_GUIDANCE_VERSION must be v0 or v1." >&2; exit 2 ;;
esac

if [[ -z "${CL_BANK_DIR:-}" ]]; then
  echo "CL_BANK_DIR is required." >&2
  exit 2
fi
if [[ ! -d "${CL_BANK_DIR}" ]]; then
  echo "CL_BANK_DIR is not a directory: ${CL_BANK_DIR}" >&2
  exit 1
fi
CL_BANK_DIR="$(cd -- "${CL_BANK_DIR}" >/dev/null 2>&1 && pwd)"

if [[ -z "${CL_GROUP_TAG:-}" || ! "${CL_GROUP_TAG}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "CL_GROUP_TAG is required and must contain only letters, digits, underscores, or hyphens." >&2
  exit 2
fi
case "${CL_ADMISSION:-}" in
  success|failure|both) ;;
  *) echo "CL_ADMISSION must be success, failure, or both." >&2; exit 2 ;;
esac

TOPK_SELECTION="${TOPK_SELECTION:?TOPK_SELECTION is required}"
TOPK_SELECTION_SHA256="${TOPK_SELECTION_SHA256:?TOPK_SELECTION_SHA256 is required}"
IFS=$'\t' read -r SELECTED_POSITIVE_TOP_K SELECTED_NEGATIVE_TOP_K _ < <(
  "${PYTHON}" -m openpi.experiments.topk_contract \
    --path "${TOPK_SELECTION}" --sha256 "${TOPK_SELECTION_SHA256}" --consumer pi --field tsv
)

require_fixed_value() {
  local name="$1"
  local expected="$2"
  local actual="${!name-}"
  if [[ -n "${actual}" && "${actual}" != "${expected}" ]]; then
    echo "${name} is fixed to ${expected} for CL V0 evaluation; got: ${actual}" >&2
    exit 2
  fi
  printf -v "${name}" '%s' "${expected}"
  export "${name}"
}

if [[ "${CL_ADMISSION}" == "success" ]]; then
  require_fixed_value MEMORY_VARIANT success
else
  require_fixed_value MEMORY_VARIANT success_fail
fi
require_fixed_value MEMORY_GUIDANCE_VERSION "${CL_GUIDANCE_VERSION}"
require_fixed_value BATCH_SIZE 8
require_fixed_value MEMORY_TOP_K "${SELECTED_POSITIVE_TOP_K}"
require_fixed_value NEGATIVE_MEMORY_TOP_K "${SELECTED_NEGATIVE_TOP_K}"
require_fixed_value NEGATIVE_MEMORY_MIN_SIMILARITY -1.0
require_fixed_value NEGATIVE_MEMORY_MIN_CONFIDENCE 0.0

MEMORY_META_PATH="${CL_BANK_DIR}/positive/gpm_memory_meta.pt"
FAISS_INDEX_PATH="${CL_BANK_DIR}/positive/gpm_memory.index"
MEMORY_ACTIONS_PATH="${CL_BANK_DIR}/positive/gpm_memory_actions.npz"
NEGATIVE_MEMORY_META_PATH="${CL_BANK_DIR}/negative/gpm_negative_memory_meta.pt"
NEGATIVE_FAISS_INDEX_PATH="${CL_BANK_DIR}/negative/gpm_negative_memory.index"
NEGATIVE_MEMORY_ACTIONS_PATH="${CL_BANK_DIR}/negative/gpm_negative_memory_actions.npz"
readonly MEMORY_META_PATH FAISS_INDEX_PATH MEMORY_ACTIONS_PATH
readonly NEGATIVE_MEMORY_META_PATH NEGATIVE_FAISS_INDEX_PATH NEGATIVE_MEMORY_ACTIONS_PATH
export MEMORY_META_PATH FAISS_INDEX_PATH MEMORY_ACTIONS_PATH
export NEGATIVE_MEMORY_META_PATH NEGATIVE_FAISS_INDEX_PATH NEGATIVE_MEMORY_ACTIONS_PATH

required_artifacts=(
  "${MEMORY_META_PATH}"
  "${FAISS_INDEX_PATH}"
  "${MEMORY_ACTIONS_PATH}"
  "${NEGATIVE_MEMORY_META_PATH}"
  "${NEGATIVE_FAISS_INDEX_PATH}"
  "${NEGATIVE_MEMORY_ACTIONS_PATH}"
)
for path in "${required_artifacts[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing CL bank artifact: ${path}" >&2
    exit 1
  fi
done

for summary_path in "${CL_BANK_DIR}/summary.json" "${CL_BANK_DIR}/build_summary.json"; do
  if [[ -f "${summary_path}" ]]; then
    python3 - "${summary_path}" "${CL_GROUP_TAG}" "${CL_ADMISSION}" <<'PY'
import json
import sys

path, expected_group, expected_admission = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    summary = json.load(stream)
if not isinstance(summary, dict):
    raise SystemExit(f"CL bank summary must contain a JSON object: {path}")

for field in ("group", "group_tag"):
    if field in summary and str(summary[field]) != expected_group:
        raise SystemExit(
            f"CL bank summary {field} mismatch: expected {expected_group}, got {summary[field]}"
        )
if "admission" in summary and str(summary["admission"]) != expected_admission:
    raise SystemExit(
        "CL bank summary admission mismatch: "
        f"expected {expected_admission}, got {summary['admission']}"
    )
PY
  fi
done

GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SEED="${SEED:-7}"
RESUME="${RESUME:-0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-100}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_EPISODE_DATA="${SAVE_EPISODE_DATA:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${OPENPI_ROOT}/logs/cl_${CL_GUIDANCE_VERSION}/${CL_GROUP_TAG}/${CL_ADMISSION}/seed_${SEED}/libero10_guidance_only_${CL_GUIDANCE_VERSION}_success_fail_batch8_${RUN_ID}}"
MEMORY_SOURCE_TAG="cl_${CL_GUIDANCE_VERSION}_${CL_GROUP_TAG}_${CL_ADMISSION}"
export GPU PORT SEED RESUME NUM_TRIALS_PER_TASK SAVE_VIDEOS SAVE_EPISODE_DATA RUN_ID RUN_ROOT
export MEMORY_SOURCE_TAG
export TOPK_SELECTION TOPK_SELECTION_SHA256

case "${PREFLIGHT_ONLY:-0}" in
  0|1) ;;
  *) echo "PREFLIGHT_ONLY must be 0 or 1." >&2; exit 2 ;;
esac

PREFLIGHT_ONLY=1 bash "${BASE_EVAL_SCRIPT}" "$@"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "CL ${CL_GUIDANCE_VERSION} preflight passed: group=${CL_GROUP_TAG} admission=${CL_ADMISSION} bank=${CL_BANK_DIR}"
  exit 0
fi

PREFLIGHT_ONLY=0 exec bash "${BASE_EVAL_SCRIPT}" "$@"
