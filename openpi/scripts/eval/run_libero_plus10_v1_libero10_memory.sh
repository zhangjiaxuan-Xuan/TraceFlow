#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

PLUS_ROOT="${LIBERO_PLUS_ROOT:-${OPENPI_ROOT}/third_party/LIBERO-plus}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
AOSS_ROOT="${AOSS_ROOT:-/path/to/storage/datasets/robotics/LIBERO}"
SOURCE_POSITIVE="${SOURCE_POSITIVE:-${AOSS_ROOT}/derived/bcpi_positive_topk_ablation_v3_batch16_env32/selected_four_suites_s50k16_canonical_v1/capacity_50/positive}"
SOURCE_NEGATIVE="${SOURCE_NEGATIVE:-${AOSS_ROOT}/derived/bcpi_nfailure_four_suites_v1/bank_b_plus_cpi_success_cpi_n_failure/negative}"
LIBERO10_BANK="${LIBERO10_BANK:-${AOSS_ROOT}/derived/libero10_only_v1_b50_nfailure}"

[[ -f "${PLUS_ROOT}/libero/libero/benchmark/task_classification.json" ]] || { echo "Missing LIBERO-plus checkout" >&2; exit 2; }
[[ -d "${PLUS_ROOT}/libero/libero/assets" ]] || { echo "Missing LIBERO-plus assets" >&2; exit 2; }
grep -Fq '"bddl_file_name": str(task_bddl_file)' "${OPENPI_ROOT}/examples/libero/main.py" || {
  echo "Evaluator is missing the LIBERO-plus Path-to-str compatibility fix: ${OPENPI_ROOT}/examples/libero/main.py" >&2
  exit 2
}
echo "evaluator_sha256=$(sha256sum "${OPENPI_ROOT}/examples/libero/main.py" | awk '{print $1}')"

for path in \
  "${SOURCE_POSITIVE}/gpm_memory_meta.pt" "${SOURCE_POSITIVE}/gpm_memory.index" "${SOURCE_POSITIVE}/gpm_memory_actions.npz" \
  "${SOURCE_NEGATIVE}/gpm_negative_memory_meta.pt" "${SOURCE_NEGATIVE}/gpm_negative_memory.index" "${SOURCE_NEGATIVE}/gpm_negative_memory_actions.npz"; do
  [[ -f "${path}" ]] || { echo "Missing source bank artifact: ${path}" >&2; exit 2; }
done

if [[ ! -f "${LIBERO10_BANK}/positive/slice_summary.json" || ! -f "${LIBERO10_BANK}/negative/slice_summary.json" ]]; then
  "${OPENPI_PYTHON}" scripts/memory/slice_libero10_memory_banks.py \
    --positive-source "${SOURCE_POSITIVE}" \
    --negative-source "${SOURCE_NEGATIVE}" \
    --output-root "${LIBERO10_BANK}"
fi

export BANK_ROOT="${LIBERO10_BANK}"
export POSITIVE_BANK="${LIBERO10_BANK}/positive"
export MODE=guidance
export EVAL_SCOPE=full
export MEMORY_BANK_SCOPE=libero10
export LIBERO_PLUS_ROOT="${PLUS_ROOT}"
export LIBERO_SOURCE_ROOT="${PLUS_ROOT}"
export MEMORY_TOP_K=16
export NEGATIVE_MEMORY_TOP_K=8
export MEMORY_GUIDANCE_TIME_VERSION=v1
export MEMORY_GUIDANCE_NORM_CAP=0.20
export MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20
export NEGATIVE_MEMORY_MIN_SIMILARITY=-1.0
export NEGATIVE_MEMORY_MIN_CONFIDENCE=0.0
export NEGATIVE_GUIDANCE_BETA=0.10
export NEGATIVE_GUIDANCE_NORM_CAP=0.10
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-${OPENPI_ROOT}/logs/libero_plus10_full_v1_libero10_memory_${RUN_ID}}"

# Reuse the tested selector/evaluator while replacing only the memory roots.
exec bash scripts/eval/run_libero_plus10_v1_bcpi_lowcost.sh
