#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/libero10_guidance_500ep_${RUN_ID}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-7}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
TASK_IDS_CSV="${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
TASK_COUNT="$(awk -F, '{print NF}' <<<"${TASK_IDS_CSV}")"
EXPECTED_EPISODES="$((TASK_COUNT * NUM_TRIALS_PER_TASK))"
MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/.mplconfig}"

MEMORY_TOP_K="${MEMORY_TOP_K:-8}"
MEMORY_GUIDANCE_LAMBDA_MAX="${MEMORY_GUIDANCE_LAMBDA_MAX:-0.20}"
MEMORY_GUIDANCE_T_CUT="${MEMORY_GUIDANCE_T_CUT:-0.30}"
MEMORY_GUIDANCE_SIGMA="${MEMORY_GUIDANCE_SIGMA:-0.30}"
MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}"
TRAJECTORY_IMAGE_SIZE="${TRAJECTORY_IMAGE_SIZE:-128}"

EVAL_DIR="${RUN_ROOT}/eval"
OBS_DIR="${RUN_ROOT}/policy_observations"
TRACE_DIR="${RUN_ROOT}/guidance_traces"
EPISODE_DATA_ROOT="${RUN_ROOT}/episode_data"
ANALYSIS_DIR="${RUN_ROOT}/analysis"
INDEX_DIR="${RUN_ROOT}/indexes"

if [[ -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to mix data into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${EVAL_DIR}" "${OBS_DIR}" "${TRACE_DIR}" "${EPISODE_DATA_ROOT}" "${ANALYSIS_DIR}" "${INDEX_DIR}" "${MPLCONFIGDIR}"
export MPLCONFIGDIR

"${OPENPI_PYTHON}" - "${RUN_ROOT}/collection_config.json" "${POLICY_DIR}" "${SEED}" "${NUM_TRIALS_PER_TASK}" "${TASK_IDS_CSV}" "${TRAJECTORY_IMAGE_SIZE}" "${MEMORY_TOP_K}" "${MEMORY_GUIDANCE_LAMBDA_MAX}" "${MEMORY_GUIDANCE_T_CUT}" "${MEMORY_GUIDANCE_SIGMA}" "${MEMORY_GUIDANCE_NORM_CAP}" <<'PY'
import json
from pathlib import Path
import sys

path, policy_dir, seed, trials, task_ids, image_size, top_k, lambda_max, t_cut, sigma, norm_cap = sys.argv[1:]
payload = {
    "schema_version": 1,
    "suite": "libero_10",
    "task_ids": [int(x) for x in task_ids.split(",")],
    "episodes_per_task": int(trials),
    "seed": int(seed),
    "policy_dir": policy_dir,
    "mode": "gpm_guidance_only_fixed_10_no_lcm",
    "save_videos": True,
    "episode_data_mode": "all",
    "trajectory_format": "libero-eval-trajectory-v1",
    "trajectory_image_size": int(image_size),
    "guidance": {
        "top_k": int(top_k),
        "lambda_max": float(lambda_max),
        "t_cut": float(t_cut),
        "sigma": float(sigma),
        "norm_cap": float(norm_cap),
        "num_steps": 10,
        "use_lcm": False,
    },
}
Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

echo "[1/3] Running LIBERO-10: 10 tasks x ${NUM_TRIALS_PER_TASK} episodes"
POLICY_DIR="${POLICY_DIR}" \
LOG_DIR="${EVAL_DIR}" \
RESULTS_TXT="${EVAL_DIR}/results.txt" \
VIDEO_ROOT="${EVAL_DIR}/videos" \
DUMP_OBSERVATION_DIR="${OBS_DIR}" \
MEMORY_GUIDANCE_TRACE_DIR="${TRACE_DIR}" \
EPISODE_DATA_ROOT="${EPISODE_DATA_ROOT}" \
EPISODE_DATA_MODE=all \
TRAJECTORY_IMAGE_SIZE="${TRAJECTORY_IMAGE_SIZE}" \
TASK_IDS_CSV="${TASK_IDS_CSV}" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
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
SAVE_VIDEOS=1 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
bash scripts/eval/run_libero_eval.sh libero_10

echo "[2/3] Validating and indexing videos and trajectories"
"${OPENPI_PYTHON}" scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${EVAL_DIR}/libero_10.jsonl" \
  --output-dir "${INDEX_DIR}" \
  --expected-tasks "${TASK_COUNT}" \
  --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
  --require-videos \
  --require-trajectories

echo "[3/3] Aggregating guidance traces and rendering plots"
MPLBACKEND=Agg "${OPENPI_PYTHON}" scripts/analysis/analyze_memory_guidance_traces.py \
  --trace-dir "${TRACE_DIR}" \
  --output-dir "${ANALYSIS_DIR}" \
  --observations-dir "${OBS_DIR}" \
  --eval-log "${EVAL_DIR}/libero_10.jsonl" \
  --expected-episodes "${EXPECTED_EPISODES}"

echo "Completed: ${RUN_ROOT}"
echo "Failures: ${INDEX_DIR}/FAILED_episodes.tsv"
echo "Per-task summary: ${INDEX_DIR}/task_summary.tsv"
echo "Guidance plots: ${ANALYSIS_DIR}/guidance_dynamics.png and guidance_effects.png"
