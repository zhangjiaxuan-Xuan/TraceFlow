#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/real_diagnostics_libero10_${RUN_ID}}"
TASK_IDS_CSV="${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-0}"

COLLECT_LOG_DIR="${RUN_ROOT}/eval"
OBS_DIR="${RUN_ROOT}/observations"
DIAG_ROOT="${RUN_ROOT}/per_observation"
SUMMARY_DIR="${RUN_ROOT}/summary"

mkdir -p "${RUN_ROOT}" "${OBS_DIR}" "${DIAG_ROOT}" "${SUMMARY_DIR}"

echo "[1/3] Collecting real libero_10 observations"
POLICY_DIR="${POLICY_DIR}" \
LOG_DIR="${COLLECT_LOG_DIR}" \
RESULTS_TXT="${COLLECT_LOG_DIR}/results.txt" \
DUMP_OBSERVATION_DIR="${OBS_DIR}" \
TASK_IDS_CSV="${TASK_IDS_CSV}" \
NUM_TRIALS_PER_TASK=1 \
SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
bash scripts/eval/run_libero_eval.sh libero_10

mapfile -t OBS_FILES < <(find "${OBS_DIR}" -name '*.npz' -type f | sort)
if [[ "${#OBS_FILES[@]}" -eq 0 ]]; then
  echo "No observation npz files were collected under ${OBS_DIR}" >&2
  exit 1
fi
if [[ "${MAX_OBSERVATIONS}" -gt 0 && "${#OBS_FILES[@]}" -gt "${MAX_OBSERVATIONS}" ]]; then
  OBS_FILES=("${OBS_FILES[@]:0:${MAX_OBSERVATIONS}}")
fi
printf "%s\n" "${OBS_FILES[@]}" > "${RUN_ROOT}/observation_files.txt"
echo "Collected ${#OBS_FILES[@]} observation files"

echo "[2/3] Running batched memory/action diagnostics with one model load"
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"${OPENPI_PYTHON}" scripts/analysis/diagnose_memory_action_similarity.py \
  --policy-dir "${POLICY_DIR}" \
  --observation-list "${RUN_ROOT}/observation_files.txt" \
  --output-dir "${DIAG_ROOT}"

echo "[3/3] Aggregating diagnostics and rendering plots"
"${OPENPI_PYTHON}" scripts/analysis/analyze_observation_diagnostics.py \
  --diagnostics-root "${DIAG_ROOT}" \
  --output-dir "${SUMMARY_DIR}"

echo "Done"
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY_DIR}"
