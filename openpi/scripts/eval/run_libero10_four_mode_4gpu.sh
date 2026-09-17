#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8200}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/libero10_four_mode_4gpu_${RUN_ID}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
TASK_IDS_CSV="${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
SEED="${SEED:-7}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_EPISODE_DATA="${SAVE_EPISODE_DATA:-1}"
TRAJECTORY_IMAGE_SIZE="${TRAJECTORY_IMAGE_SIZE:-128}"
NEGATIVE_GUIDANCE_NORM_CAP="${NEGATIVE_GUIDANCE_NORM_CAP:-0.10}"
NEGATIVE_MEMORY_MIN_SIMILARITY="${NEGATIVE_MEMORY_MIN_SIMILARITY:-0.975}"
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-4}"
NEGATIVE_MEMORY_MIN_CONFIDENCE="${NEGATIVE_MEMORY_MIN_CONFIDENCE:-0.75}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
MEMORY_META_PATH="${MEMORY_META_PATH:-${OPENPI_ROOT}/memory/gpm_memory_meta.pt}"
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${OPENPI_ROOT}/memory/gpm_memory.index}"
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-${OPENPI_ROOT}/memory/gpm_memory_actions.npz}"
NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory_meta.pt}"
NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory.index}"
NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory_actions.npz}"

IFS=',' read -r -a GPUS <<<"${GPUS_CSV}"
MODES=(base optimus guidance_pos guidance_pos_neg)
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPUS_CSV must contain exactly four GPU ids." >&2
  exit 1
fi
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPUS_CSV must contain four distinct GPU ids." >&2
  exit 1
fi
if [[ "${NUM_TRIALS_PER_TASK}" -le 0 ]]; then
  echo "NUM_TRIALS_PER_TASK must be positive." >&2
  exit 1
fi
if [[ -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to mix data into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
  exit 1
fi
for executable in "${OPENPI_PYTHON}" "${LIBERO_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Missing executable Python: ${executable}" >&2
    exit 1
  fi
done
for required_path in \
  "${POLICY_DIR}" \
  "${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json" \
  "${MEMORY_META_PATH}" "${FAISS_INDEX_PATH}" "${MEMORY_ACTIONS_PATH}" \
  "${NEGATIVE_MEMORY_META_PATH}" "${NEGATIVE_FAISS_INDEX_PATH}" "${NEGATIVE_MEMORY_ACTIONS_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done
for i in "${!MODES[@]}"; do
  port="$((BASE_PORT + i))"
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1; then
    exec 3<&- || true
    exec 3>&- || true
    echo "Port ${port} is already in use." >&2
    exit 1
  fi
done
"${OPENPI_PYTHON}" -c "import faiss, torch, tyro; import sys; sys.path.insert(0, '${OPENPI_ROOT}/src'); import openpi" >/dev/null
preflight_numba_cache="${TMPDIR:-/tmp}/optimusvla_preflight_${USER:-user}_$$/numba"
mkdir -p "${preflight_numba_cache}"
for gpu in "${GPUS[@]}"; do
  PYTHONPATH="${OPENPI_ROOT}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}" \
  NUMBA_CACHE_DIR="${preflight_numba_cache}/gpu${gpu}" \
  CUDA_VISIBLE_DEVICES="${GPUS_CSV}" \
  MUJOCO_EGL_DEVICE_ID="${gpu}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
    "${LIBERO_PYTHON}" scripts/eval/probe_libero_egl.py
done
rm -rf "${preflight_numba_cache%/numba}"
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight passed: GPUs=${GPUS_CSV} ports=${BASE_PORT}-$((BASE_PORT + 3)) run_root=${RUN_ROOT}"
  exit 0
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/pids"
cat >"${RUN_ROOT}/run_config.json" <<EOF
{
  "modes": ["base", "optimus", "guidance_pos", "guidance_pos_neg"],
  "gpus": "${GPUS_CSV}",
  "base_port": ${BASE_PORT},
  "episodes_per_task": ${NUM_TRIALS_PER_TASK},
  "task_ids": "${TASK_IDS_CSV}",
  "seed": ${SEED},
  "negative_gate": {
    "hard_task_match": false,
    "min_similarity": ${NEGATIVE_MEMORY_MIN_SIMILARITY},
    "top_k": ${NEGATIVE_MEMORY_TOP_K},
    "allow_empty": true,
    "beta": 0.10,
    "norm_cap": ${NEGATIVE_GUIDANCE_NORM_CAP}
  }
}
EOF

PIDS=()
cleanup() {
  local code=$?
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  exit "${code}"
}
trap cleanup INT TERM EXIT

run_mode() {
  local mode="$1"
  local gpu="$2"
  local port="$3"
  local mode_root="${RUN_ROOT}/${mode}"
  local use_memory=1 use_lcm=0 guidance_only=0 use_negative=0
  case "${mode}" in
    base) use_memory=0 ;;
    optimus) use_lcm=1 ;;
    guidance_pos) guidance_only=1 ;;
    guidance_pos_neg) guidance_only=1; use_negative=1 ;;
    *) echo "Unknown mode: ${mode}" >&2; return 2 ;;
  esac
  mkdir -p "${mode_root}/eval"
  local episode_data_root=""
  if [[ "${SAVE_EPISODE_DATA}" == "1" ]]; then
    episode_data_root="${mode_root}/episode_data"
  fi
  # Keep all render GPUs visible so EGL IDs remain physical device indices.
  # Model servers remain isolated by SERVER_CUDA_VISIBLE_DEVICES.
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SERVER_CUDA_VISIBLE_DEVICES="${gpu}" \
  CLIENT_CUDA_VISIBLE_DEVICES="${GPUS_CSV}" \
  MUJOCO_EGL_DEVICE_ID="${gpu}" \
  LIBERO_CONFIG_PATH="${mode_root}/libero_config" \
  NUMBA_CACHE_DIR="${mode_root}/cache/numba" \
  MPLCONFIGDIR="${mode_root}/cache/matplotlib" \
  PORT="${port}" \
  POLICY_DIR="${POLICY_DIR}" \
  LOG_DIR="${mode_root}/eval" \
  RESULTS_TXT="${mode_root}/eval/results.txt" \
  VIDEO_ROOT="${mode_root}/eval/videos" \
  EPISODE_DATA_ROOT="${episode_data_root}" \
  EPISODE_DATA_MODE=all \
  TRAJECTORY_IMAGE_SIZE="${TRAJECTORY_IMAGE_SIZE}" \
  TASK_IDS_CSV="${TASK_IDS_CSV}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
  SEED="${SEED}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" \
  USE_MEMORY="${use_memory}" \
  USE_LCM="${use_lcm}" \
  MEMORY_GUIDANCE_ONLY="${guidance_only}" \
  USE_NEGATIVE_GUIDANCE="${use_negative}" \
  NEGATIVE_MEMORY_MIN_SIMILARITY="${NEGATIVE_MEMORY_MIN_SIMILARITY}" \
  NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K}" \
  NEGATIVE_MEMORY_MIN_CONFIDENCE="${NEGATIVE_MEMORY_MIN_CONFIDENCE}" \
  NEGATIVE_GUIDANCE_NORM_CAP="${NEGATIVE_GUIDANCE_NORM_CAP}" \
  MEMORY_META_PATH="${MEMORY_META_PATH}" \
  FAISS_INDEX_PATH="${FAISS_INDEX_PATH}" \
  MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH}" \
  NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH}" \
  NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH}" \
  NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH}" \
  LIBERO_PYTHON="${LIBERO_PYTHON}" \
  bash scripts/eval/run_libero_eval.sh libero_10

  local validate_args=()
  if [[ "${SAVE_VIDEOS}" == "1" ]]; then validate_args+=(--require-videos); fi
  if [[ "${SAVE_EPISODE_DATA}" == "1" ]]; then validate_args+=(--require-trajectories); fi
  "${OPENPI_PYTHON}" scripts/analysis/summarize_libero_eval_run.py \
    --eval-log "${mode_root}/eval/libero_10.jsonl" \
    --output-dir "${mode_root}/indexes" \
    --expected-tasks 10 \
    --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
    "${validate_args[@]}"
}

for i in "${!MODES[@]}"; do
  mode="${MODES[$i]}"
  gpu="${GPUS[$i]}"
  port="$((BASE_PORT + i))"
  (
    run_mode "${mode}" "${gpu}" "${port}"
  ) >"${RUN_ROOT}/logs/${mode}_gpu${gpu}_port${port}.log" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  echo "${pid}" >"${RUN_ROOT}/pids/${mode}.pid"
  echo "Started mode=${mode} gpu=${gpu} port=${port} pid=${pid}"
done

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "Finished mode=${MODES[$i]}"
  else
    echo "Failed mode=${MODES[$i]}" >&2
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${OPENPI_PYTHON}" scripts/analysis/summarize_four_mode_eval.py --run-root "${RUN_ROOT}"
echo "Comparison: ${RUN_ROOT}/comparison.tsv"
echo "Paired analysis: ${RUN_ROOT}/paired_vs_base.tsv"
trap - INT TERM EXIT
