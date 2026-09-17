#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/memory_guidance_debug_${RUN_ID}}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-10}"
SEED="${SEED:-7}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/.mplconfig}"
MEMORY_TOP_K="${MEMORY_TOP_K:-8}"
MEMORY_GUIDANCE_LAMBDA_MAX="${MEMORY_GUIDANCE_LAMBDA_MAX:-0.20}"
MEMORY_GUIDANCE_T_CUT="${MEMORY_GUIDANCE_T_CUT:-0.30}"
MEMORY_GUIDANCE_SIGMA="${MEMORY_GUIDANCE_SIGMA:-0.30}"
MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}"

EVAL_DIR="${RUN_ROOT}/eval"
OBS_DIR="${RUN_ROOT}/observations"
TRACE_DIR="${RUN_ROOT}/guidance_traces"
ANALYSIS_DIR="${RUN_ROOT}/analysis"
if [[ -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to mix data into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}" "${OBS_DIR}" "${TRACE_DIR}" "${ANALYSIS_DIR}"
mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

"${OPENPI_PYTHON}" - "${RUN_ROOT}/collection_config.json" "${POLICY_DIR}" "${TASK_ID}" "${NUM_TRIALS}" "${SEED}" "${MEMORY_TOP_K}" "${MEMORY_GUIDANCE_LAMBDA_MAX}" "${MEMORY_GUIDANCE_T_CUT}" "${MEMORY_GUIDANCE_SIGMA}" "${MEMORY_GUIDANCE_NORM_CAP}" <<'PY'
import json
from pathlib import Path
import sys

path, policy_dir, task_id, num_trials, seed, top_k, lambda_max, t_cut, sigma, norm_cap = sys.argv[1:]
payload = {
    "schema_version": 1,
    "suite": "libero_10",
    "task_id": int(task_id),
    "num_trials": int(num_trials),
    "seed": int(seed),
    "policy_dir": policy_dir,
    "mode": "gpm_guidance_only_fixed_10_no_lcm",
    "guidance": {
        "top_k": int(top_k),
        "lambda_max": float(lambda_max),
        "t_cut": float(t_cut),
        "sigma": float(sigma),
        "norm_cap": float(norm_cap),
    },
}
Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

POLICY_DIR="${POLICY_DIR}" \
LOG_DIR="${EVAL_DIR}" \
RESULTS_TXT="${EVAL_DIR}/results.txt" \
DUMP_OBSERVATION_DIR="${OBS_DIR}" \
MEMORY_GUIDANCE_TRACE_DIR="${TRACE_DIR}" \
TASK_IDS_CSV="${TASK_ID}" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
SEED="${SEED}" \
USE_MEMORY=1 \
MEMORY_GUIDANCE_ONLY=1 \
USE_LCM=0 \
MEMORY_GUIDANCE_NUM_STEPS=10 \
MEMORY_TOP_K="${MEMORY_TOP_K}" \
MEMORY_GUIDANCE_LAMBDA_MAX="${MEMORY_GUIDANCE_LAMBDA_MAX}" \
MEMORY_GUIDANCE_T_CUT="${MEMORY_GUIDANCE_T_CUT}" \
MEMORY_GUIDANCE_SIGMA="${MEMORY_GUIDANCE_SIGMA}" \
MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP}" \
DEBUG_MEMORY=1 \
SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
bash scripts/eval/run_libero_eval.sh libero_10

"${OPENPI_PYTHON}" scripts/analysis/analyze_memory_guidance_traces.py \
  --trace-dir "${TRACE_DIR}" \
  --output-dir "${ANALYSIS_DIR}" \
  --observations-dir "${OBS_DIR}" \
  --eval-log "${EVAL_DIR}/libero_10.jsonl" \
  --expected-episodes "${NUM_TRIALS}"

find "${OBS_DIR}" -type f -name '*.npz' | sort > "${RUN_ROOT}/observation_files.txt"
find "${TRACE_DIR}" -type f -name '*.npz' | sort > "${RUN_ROOT}/trace_files.txt"

echo "Completed reusable one-task guidance collection: ${RUN_ROOT}"
echo "Evaluation: ${EVAL_DIR}/results.txt"
echo "Trace summary: ${ANALYSIS_DIR}/summary.json"
echo "Visualization: ${ANALYSIS_DIR}/guidance_dynamics.png"
