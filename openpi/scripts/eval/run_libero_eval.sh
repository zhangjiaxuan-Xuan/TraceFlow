#!/usr/bin/env bash

set -u -o pipefail

if [[ "${OPTIMUS_RUN_LIBERO_EVAL_SNAPSHOT:-0}" != "1" ]]; then
  original_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
  export OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${original_script_dir}/../.." >/dev/null 2>&1 && pwd)}"
  script_snapshot="${TMPDIR:-/tmp}/optimus_run_libero_eval_${$}.sh"
  cp -- "${BASH_SOURCE[0]}" "${script_snapshot}"
  export OPTIMUS_RUN_LIBERO_EVAL_SNAPSHOT=1
  exec bash "${script_snapshot}" "$@"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-}"
ACTION_NORM_STATS_PATH="${ACTION_NORM_STATS_PATH:-}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/local/data/openpi}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/optimusvla_libero_config}"
LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-${OPENPI_ROOT}/third_party/libero}"

SERVER_CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
CLIENT_CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-0}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-180}"
EXTERNAL_POLICY_SERVER="${EXTERNAL_POLICY_SERVER:-0}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"
INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-5.0}"
INFERENCE_BATCH_GROUP_SIZE="${INFERENCE_BATCH_GROUP_SIZE:-0}"
LIBERO_CLIENTS_PER_SUITE="${LIBERO_CLIENTS_PER_SUITE:-1}"
LIBERO_SHARD_AXIS="${LIBERO_SHARD_AXIS:-episodes}"
if [[ "${INFERENCE_BATCH_SIZE}" -le 0 || "${LIBERO_CLIENTS_PER_SUITE}" -le 0 || "${INFERENCE_BATCH_GROUP_SIZE}" -lt 0 ]]; then
  echo "INFERENCE_BATCH_SIZE and LIBERO_CLIENTS_PER_SUITE must be positive." >&2
  exit 1
fi
if [[ "${INFERENCE_BATCH_GROUP_SIZE}" -gt 0 && "${INFERENCE_BATCH_GROUP_SIZE}" -ne "${INFERENCE_BATCH_SIZE}" ]]; then
  echo "INFERENCE_BATCH_GROUP_SIZE must equal INFERENCE_BATCH_SIZE when enabled." >&2
  exit 1
fi
if [[ "${LIBERO_SHARD_AXIS}" != "episodes" && "${LIBERO_SHARD_AXIS}" != "tasks" ]]; then
  echo "LIBERO_SHARD_AXIS must be episodes or tasks, got ${LIBERO_SHARD_AXIS}." >&2
  exit 1
fi
if [[ "${LIBERO_SHARD_AXIS}" == "tasks" && -z "${TASK_IDS_CSV:-}" ]]; then
  echo "LIBERO_SHARD_AXIS=tasks requires TASK_IDS_CSV." >&2
  exit 1
fi

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
EPISODE_START="${EPISODE_START:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-0}"
SEED="${SEED:-7}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
DUMP_OBSERVATION_NPZ="${DUMP_OBSERVATION_NPZ:-}"
DUMP_OBSERVATION_DIR="${DUMP_OBSERVATION_DIR:-}"
TASK_IDS_CSV="${TASK_IDS_CSV:-}"
EPISODE_DATA_ROOT="${EPISODE_DATA_ROOT:-}"
EPISODE_DATA_MODE="${EPISODE_DATA_MODE:-all}"
TRAJECTORY_IMAGE_SIZE="${TRAJECTORY_IMAGE_SIZE:-128}"
FAIL_ON_EPISODE_ERROR="${FAIL_ON_EPISODE_ERROR:-0}"
MUJOCO_BACKEND="${MUJOCO_GL:-egl}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/optimusvla_numba_cache}"
NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/optimusvla_mpl_cache}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-0}"
LOG_DIR="${LOG_DIR:-logs/libero_eval_${RUN_ID}}"
VIDEO_ROOT="${VIDEO_ROOT:-${LOG_DIR}/videos}"
RESULTS_TXT="${RESULTS_TXT:-${LOG_DIR}/results.txt}"

MEMORY_TOP_K="${MEMORY_TOP_K:-8}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
MEMORY_META_PATH="${MEMORY_META_PATH:-${OPENPI_ROOT}/memory/gpm_memory_meta.pt}"
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${OPENPI_ROOT}/memory/gpm_memory.index}"
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-${OPENPI_ROOT}/memory/gpm_memory_actions.npz}"
MEMORY_REFRESH_EVERY="${MEMORY_REFRESH_EVERY:-1}"
USE_MEMORY="${USE_MEMORY:-1}"
DEBUG_MEMORY="${DEBUG_MEMORY:-0}"
ALIGN_MODE="${ALIGN_MODE:-hybrid}"
MIXTURE_MODE="${MIXTURE_MODE:-gaussian}"
TEMPERATURE="${TEMPERATURE:-10.0}"
SIGMA_MIN="${SIGMA_MIN:-0.05}"
NOISE_MIN="${NOISE_MIN:-0.20}"
NOISE_MAX="${NOISE_MAX:-1.00}"
NFE_MIN="${NFE_MIN:-1}"
NFE_MAX="${NFE_MAX:-10}"
NFE_FLOOR="${NFE_FLOOR:-${NFE_MIN}}"
USE_LCM="${USE_LCM:-1}"
LCM_SCALE="${LCM_SCALE:-0.10}"
USE_MEMORY_GUIDANCE="${USE_MEMORY_GUIDANCE:-0}"
MEMORY_GUIDANCE_ONLY="${MEMORY_GUIDANCE_ONLY:-0}"
MEMORY_GUIDANCE_TIME_VERSION="${MEMORY_GUIDANCE_TIME_VERSION:-v1}"
MEMORY_PRIOR_SUBSTEP_GUIDANCE="${MEMORY_PRIOR_SUBSTEP_GUIDANCE:-0}"
MEMORY_PRIOR_GUIDANCE_VERSION="${MEMORY_PRIOR_GUIDANCE_VERSION:-v1}"
MEMORY_PRIOR_GUIDANCE_FINAL_SCALE="${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE:-0.01}"
MEMORY_JOINT_FAILURE_PRIOR="${MEMORY_JOINT_FAILURE_PRIOR:-0.50}"
MEMORY_GUIDANCE_V2_MAGNITUDE_CAP="${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP:-0.10}"
MEMORY_GUIDANCE_V05_DYNAMIC_NFE="${MEMORY_GUIDANCE_V05_DYNAMIC_NFE:-0}"
MEMORY_GUIDANCE_V05_FINE_RATIO="${MEMORY_GUIDANCE_V05_FINE_RATIO:-1.20}"
MEMORY_GUIDANCE_V05_FINE_SCALE="${MEMORY_GUIDANCE_V05_FINE_SCALE:-0.20}"
MEMORY_GUIDANCE_V05_FINE_CONFIRM_STEPS="${MEMORY_GUIDANCE_V05_FINE_CONFIRM_STEPS:-2}"
MEMORY_GUIDANCE_V05_DYNAMIC_GUIDANCE_SCALE="${MEMORY_GUIDANCE_V05_DYNAMIC_GUIDANCE_SCALE:-0.50}"
MEMORY_GUIDANCE_NUM_STEPS="${MEMORY_GUIDANCE_NUM_STEPS:-10}"
MEMORY_GUIDANCE_LAMBDA_MAX="${MEMORY_GUIDANCE_LAMBDA_MAX:-0.20}"
MEMORY_GUIDANCE_T_CUT="${MEMORY_GUIDANCE_T_CUT:-0.30}"
MEMORY_GUIDANCE_SIGMA="${MEMORY_GUIDANCE_SIGMA:-0.30}"
MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}"
MEMORY_GUIDANCE_MIN_SIMILARITY="${MEMORY_GUIDANCE_MIN_SIMILARITY:--1.0}"
MEMORY_GUIDANCE_TRACE_DIR="${MEMORY_GUIDANCE_TRACE_DIR:-}"
MEMORY_GUIDANCE_TRACE_LEVEL="${MEMORY_GUIDANCE_TRACE_LEVEL:-full}"
USE_NEGATIVE_GUIDANCE="${USE_NEGATIVE_GUIDANCE:-0}"
NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory_meta.pt}"
NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory.index}"
NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory_actions.npz}"
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-4}"
NEGATIVE_MEMORY_MIN_SIMILARITY="${NEGATIVE_MEMORY_MIN_SIMILARITY:-0.975}"
NEGATIVE_MEMORY_MIN_CONFIDENCE="${NEGATIVE_MEMORY_MIN_CONFIDENCE:-0.75}"
NEGATIVE_GUIDANCE_BETA="${NEGATIVE_GUIDANCE_BETA:-0.10}"
NEGATIVE_GUIDANCE_SIGMA="${NEGATIVE_GUIDANCE_SIGMA:-0.30}"
NEGATIVE_GUIDANCE_NORM_CAP="${NEGATIVE_GUIDANCE_NORM_CAP:-0.10}"
MEMORY_GUIDANCE_TOTAL_NORM_CAP="${MEMORY_GUIDANCE_TOTAL_NORM_CAP:-0.20}"

if [[ "$#" -gt 0 ]]; then
  SUITES=("$@")
else
  SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

server_pid=""
client_pids=()
client_names=()

is_true() {
  case "${1}" in
    1 | true | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

validate_gpu_contract() {
  if [[ "${MUJOCO_BACKEND}" != "egl" ]]; then
    return 0
  fi
  if [[ ! "${MUJOCO_EGL_DEVICE_ID}" =~ ^[0-9]+$ ]]; then
    echo "MUJOCO_EGL_DEVICE_ID must be a numeric EGL device index, got: ${MUJOCO_EGL_DEVICE_ID}" >&2
    return 1
  fi
  if [[ "${MUJOCO_EGL_DEVICE_ID}" == "0" ]]; then
    return 0
  fi
  case ",${CLIENT_CUDA_VISIBLE_DEVICES}," in
    *",${MUJOCO_EGL_DEVICE_ID},"*) ;;
    *)
      echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} is not present in CLIENT_CUDA_VISIBLE_DEVICES=${CLIENT_CUDA_VISIBLE_DEVICES}." >&2
      return 1
      ;;
  esac
}

wait_for_server() {
  local waited=0
  while ((waited < SERVER_WAIT_SECONDS)); do
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "Policy server exited before it became ready. See ${server_log}" >&2
      tail -n 80 "${server_log}" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 1 "http://${HOST}:${PORT}/healthz" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "Timed out waiting for policy server at ${HOST}:${PORT}. See ${server_log}" >&2
  tail -n 80 "${server_log}" >&2 || true
  return 1
}

wait_for_external_server() {
  local waited=0
  while ((waited < SERVER_WAIT_SECONDS)); do
    if curl --silent --show-error --fail --max-time 1 "http://${HOST}:${PORT}/healthz" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "Timed out waiting for external policy server at ${HOST}:${PORT}." >&2
  [[ -n "${SERVER_LOG_PATH}" ]] && tail -n 80 "${SERVER_LOG_PATH}" >&2 || true
  return 1
}

require_free_server_port() {
  if "${OPENPI_PYTHON}" - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    occupied = sock.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0
raise SystemExit(0 if occupied else 1)
PY
  then
    echo "Refusing to start: ${HOST}:${PORT} is already occupied by another server." >&2
    return 1
  fi
}

cleanup() {
  local pid
  for pid in "${client_pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  if [[ -n "${server_pid:-}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

cd "${OPENPI_ROOT}" || exit 1

validate_gpu_contract || exit 1
echo "GPU contract: server=${SERVER_CUDA_VISIBLE_DEVICES} client=${CLIENT_CUDA_VISIBLE_DEVICES} egl=${MUJOCO_EGL_DEVICE_ID}"

if [[ -z "${POLICY_DIR}" ]]; then
  echo "Set POLICY_DIR to the pi0.5 PyTorch checkpoint directory." >&2
  exit 1
fi

if [[ -z "${ACTION_NORM_STATS_PATH}" ]]; then
  ACTION_NORM_STATS_PATH="${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json"
fi

if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "Missing OpenPI Python: ${OPENPI_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "Missing LIBERO Python: ${LIBERO_PYTHON}" >&2
  exit 1
fi

worker_log_root="${LOG_DIR}/workers/${RUN_ID}"
worker_stdout_root="${LOG_DIR}/stdout/${RUN_ID}"
mkdir -p "${worker_stdout_root}" "${worker_log_root}"
if is_true "${SAVE_VIDEOS}"; then
  mkdir -p "${VIDEO_ROOT}"
fi
mkdir -p "${LIBERO_CONFIG_PATH}" "${LIBERO_CONFIG_PATH}/datasets"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  cat >"${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${LIBERO_SOURCE_ROOT}/libero/libero
bddl_files: ${LIBERO_SOURCE_ROOT}/libero/libero/./bddl_files
init_states: ${LIBERO_SOURCE_ROOT}/libero/libero/./init_files
datasets: ${LIBERO_CONFIG_PATH}/datasets
assets: ${LIBERO_SOURCE_ROOT}/libero/libero/./assets
EOF
fi
server_log="${SERVER_LOG_PATH:-${LOG_DIR}/server.log}"
status_file="${LOG_DIR}/suite_status.tsv"
: >"${status_file}"

PYTHONPATH_VALUE="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${LIBERO_SOURCE_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"
fi

server_cmd=(
  "${OPENPI_PYTHON}" scripts/serve_policy.py
  --env LIBERO
  --port "${PORT}"
  --seed "${SEED}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"
  --inference-batch-wait-ms "${INFERENCE_BATCH_WAIT_MS}"
  --inference-batch-group-size "${INFERENCE_BATCH_GROUP_SIZE}"
)

if is_true "${USE_MEMORY}"; then
  server_cmd+=(
    --use-memory
    --task-head-ckpt "${TASK_HEAD_CKPT}"
    --action-norm-stats-path "${ACTION_NORM_STATS_PATH}"
    --action-use-quantile-norm
    --memory-meta-path "${MEMORY_META_PATH}"
    --faiss-index-path "${FAISS_INDEX_PATH}"
    --memory-actions-path "${MEMORY_ACTIONS_PATH}"
    --memory-top-k "${MEMORY_TOP_K}"
    --memory-refresh-every "${MEMORY_REFRESH_EVERY}"
    --align-mode "${ALIGN_MODE}"
    --mixture-mode "${MIXTURE_MODE}"
    --temperature "${TEMPERATURE}"
    --sigma-min "${SIGMA_MIN}"
    --noise-min "${NOISE_MIN}"
    --noise-max "${NOISE_MAX}"
    --nfe-min "${NFE_MIN}"
    --nfe-max "${NFE_MAX}"
    --nfe-floor "${NFE_FLOOR}"
  )
  if is_true "${DEBUG_MEMORY}"; then
    server_cmd+=(--debug-memory)
  fi

  if is_true "${USE_MEMORY_GUIDANCE}" && ! is_true "${MEMORY_GUIDANCE_ONLY}"; then
    server_cmd+=(
      --use-memory-guidance
      --memory-guidance-num-steps "${MEMORY_GUIDANCE_NUM_STEPS}"
      --memory-guidance-lambda-max "${MEMORY_GUIDANCE_LAMBDA_MAX}"
      --memory-guidance-t-cut "${MEMORY_GUIDANCE_T_CUT}"
      --memory-guidance-sigma "${MEMORY_GUIDANCE_SIGMA}"
      --memory-guidance-norm-cap "${MEMORY_GUIDANCE_NORM_CAP}"
      --memory-guidance-min-similarity "${MEMORY_GUIDANCE_MIN_SIMILARITY}"
    )
  fi

  if is_true "${MEMORY_GUIDANCE_ONLY}"; then
    server_cmd+=(
      --memory-guidance-only
      --memory-guidance-time-version "${MEMORY_GUIDANCE_TIME_VERSION}"
      --memory-guidance-num-steps "${MEMORY_GUIDANCE_NUM_STEPS}"
      --memory-guidance-v2-magnitude-cap "${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP}"
      --memory-guidance-v05-fine-ratio "${MEMORY_GUIDANCE_V05_FINE_RATIO}"
      --memory-guidance-v05-fine-scale "${MEMORY_GUIDANCE_V05_FINE_SCALE}"
      --memory-guidance-v05-fine-confirm-steps "${MEMORY_GUIDANCE_V05_FINE_CONFIRM_STEPS}"
      --memory-guidance-v05-dynamic-guidance-scale "${MEMORY_GUIDANCE_V05_DYNAMIC_GUIDANCE_SCALE}"
      --memory-guidance-lambda-max "${MEMORY_GUIDANCE_LAMBDA_MAX}"
      --memory-guidance-t-cut "${MEMORY_GUIDANCE_T_CUT}"
      --memory-guidance-sigma "${MEMORY_GUIDANCE_SIGMA}"
      --memory-guidance-norm-cap "${MEMORY_GUIDANCE_NORM_CAP}"
      --memory-guidance-min-similarity "${MEMORY_GUIDANCE_MIN_SIMILARITY}"
    )
    if is_true "${MEMORY_GUIDANCE_V05_DYNAMIC_NFE}"; then
      server_cmd+=(--memory-guidance-v05-dynamic-nfe)
    fi
  fi

  if is_true "${MEMORY_PRIOR_SUBSTEP_GUIDANCE}"; then
    server_cmd+=(
      --memory-prior-substep-guidance
      --memory-prior-guidance-version "${MEMORY_PRIOR_GUIDANCE_VERSION}"
      --memory-prior-guidance-final-scale "${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE}"
      --memory-joint-failure-prior "${MEMORY_JOINT_FAILURE_PRIOR}"
      --memory-guidance-v2-magnitude-cap "${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP}"
      --memory-guidance-lambda-max "${MEMORY_GUIDANCE_LAMBDA_MAX}"
      --memory-guidance-t-cut "${MEMORY_GUIDANCE_T_CUT}"
      --memory-guidance-sigma "${MEMORY_GUIDANCE_SIGMA}"
      --memory-guidance-norm-cap "${MEMORY_GUIDANCE_NORM_CAP}"
      --memory-guidance-min-similarity "${MEMORY_GUIDANCE_MIN_SIMILARITY}"
    )
  fi

  if [[ -n "${MEMORY_GUIDANCE_TRACE_DIR}" ]]; then
    server_cmd+=(
      --memory-guidance-trace-dir "${MEMORY_GUIDANCE_TRACE_DIR}"
      --memory-guidance-trace-level "${MEMORY_GUIDANCE_TRACE_LEVEL}"
    )
  fi

  if is_true "${USE_NEGATIVE_GUIDANCE}"; then
    server_cmd+=(
      --use-negative-guidance
      --negative-memory-meta-path "${NEGATIVE_MEMORY_META_PATH}"
      --negative-faiss-index-path "${NEGATIVE_FAISS_INDEX_PATH}"
      --negative-memory-actions-path "${NEGATIVE_MEMORY_ACTIONS_PATH}"
      --negative-memory-top-k "${NEGATIVE_MEMORY_TOP_K}"
      --negative-memory-min-similarity "${NEGATIVE_MEMORY_MIN_SIMILARITY}"
      --negative-memory-min-confidence "${NEGATIVE_MEMORY_MIN_CONFIDENCE}"
      --negative-guidance-beta "${NEGATIVE_GUIDANCE_BETA}"
      --negative-guidance-sigma "${NEGATIVE_GUIDANCE_SIGMA}"
      --negative-guidance-norm-cap "${NEGATIVE_GUIDANCE_NORM_CAP}"
      --memory-guidance-total-norm-cap "${MEMORY_GUIDANCE_TOTAL_NORM_CAP}"
    )
  fi

  if is_true "${USE_LCM}" && ! is_true "${MEMORY_GUIDANCE_ONLY}" && ! is_true "${MEMORY_PRIOR_SUBSTEP_GUIDANCE}"; then
    server_cmd+=(--use-lcm --lcm-scale "${LCM_SCALE}")
  fi
else
  server_cmd+=(--no-use-memory)
fi

server_cmd+=(policy:checkpoint --policy.config "${POLICY_CONFIG}" --policy.dir "${POLICY_DIR}")

if is_true "${EXTERNAL_POLICY_SERVER}"; then
  [[ -n "${SERVER_LOG_PATH}" ]] || {
    echo "EXTERNAL_POLICY_SERVER=1 requires SERVER_LOG_PATH." >&2
    exit 1
  }
  wait_for_external_server || exit 1
  echo "Using persistent policy server at ${HOST}:${PORT} log=${server_log}"
else
  require_free_server_port || exit 1
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
  OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE}" \
  PYTHONPATH="${PYTHONPATH_VALUE}" \
  CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES}" \
    "${server_cmd[@]}" >"${server_log}" 2>&1 &
  server_pid="$!"

  echo "Started policy server: pid=${server_pid} log=${server_log}"
  wait_for_server || exit 1
  echo "Policy server is ready at ${HOST}:${PORT}"
fi

for suite in "${SUITES[@]}"; do
  jsonl_file="${LOG_DIR}/${suite}.jsonl"
  if is_true "${RESUME}"; then
    if [[ ! -f "${jsonl_file}" ]]; then
      echo "RESUME requires an existing combined log: ${jsonl_file}" >&2
      exit 1
    fi
    "${OPENPI_PYTHON}" scripts/data/merge_libero_jsonl.py \
      --output "${jsonl_file}" \
      --worker-root "${LOG_DIR}/workers" \
      --pattern "${suite}_worker*.jsonl"
  else
    : >"${jsonl_file}"
  fi
  for worker_id in $(seq 0 $((LIBERO_CLIENTS_PER_SUITE - 1))); do
    stdout_file="${worker_stdout_root}/${suite}_worker${worker_id}.log"
    worker_jsonl_file="${worker_log_root}/${suite}_worker${worker_id}.jsonl"
    episode_indices=""
    worker_task_ids="${TASK_IDS_CSV}"
    if [[ "${LIBERO_SHARD_AXIS}" == "tasks" ]]; then
      worker_task_ids=""
      IFS=',' read -r -a selected_task_ids <<<"${TASK_IDS_CSV}"
      for task_position in "${!selected_task_ids[@]}"; do
        if (( task_position % LIBERO_CLIENTS_PER_SUITE == worker_id )); then
          task_id="${selected_task_ids[task_position]//[[:space:]]/}"
          [[ -n "${task_id}" ]] && worker_task_ids+="${worker_task_ids:+,}${task_id}"
        fi
      done
      [[ -n "${worker_task_ids}" ]] || continue
      for ((episode_idx=EPISODE_START; episode_idx<EPISODE_START+NUM_TRIALS_PER_TASK; episode_idx++)); do
        episode_indices+="${episode_indices:+,}${episode_idx}"
      done
    else
      for ((episode_idx=EPISODE_START+worker_id; episode_idx<EPISODE_START+NUM_TRIALS_PER_TASK; episode_idx+=LIBERO_CLIENTS_PER_SUITE)); do
        episode_indices+="${episode_indices:+,}${episode_idx}"
      done
      [[ -n "${episode_indices}" ]] || continue
    fi
    client_cmd=(
    "${LIBERO_PYTHON}" examples/libero/main.py
    --args.host "${HOST}"
    --args.port "${PORT}"
    --args.task-suite-name "${suite}"
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"
    --args.episode-start "${EPISODE_START}"
    --args.replan-steps "${REPLAN_STEPS}"
    --args.num-steps-wait "${NUM_STEPS_WAIT}"
    --args.max-env-steps "${MAX_ENV_STEPS}"
    --args.resize-size "${RESIZE_SIZE}"
    --args.video-root "${VIDEO_ROOT}"
    --args.seed "${SEED}"
    --args.log-file "${worker_jsonl_file}"
    --args.episode-indices-csv "${episode_indices}"
    --args.environment-id "${suite}-worker${worker_id}"
  )
  if is_true "${RESUME}"; then
    client_cmd+=(
      --args.resume-log-file "${jsonl_file}"
      --args.resume-worker-id "$([[ "${LIBERO_SHARD_AXIS}" == "tasks" ]] && echo 0 || echo "${worker_id}")"
      --args.resume-num-workers "$([[ "${LIBERO_SHARD_AXIS}" == "tasks" ]] && echo 1 || echo "${LIBERO_CLIENTS_PER_SUITE}")"
    )
  fi
  if [[ -n "${DUMP_OBSERVATION_NPZ}" ]]; then
    client_cmd+=(--args.dump-observation-npz "${DUMP_OBSERVATION_NPZ}")
  fi
  if [[ -n "${DUMP_OBSERVATION_DIR}" ]]; then
    client_cmd+=(--args.dump-observation-dir "${DUMP_OBSERVATION_DIR}")
  fi
  if [[ -n "${worker_task_ids}" ]]; then
    client_cmd+=(--args.task-ids-csv "${worker_task_ids}")
  fi
  if [[ -n "${EPISODE_DATA_ROOT}" ]]; then
    client_cmd+=(
      --args.episode-data-root "${EPISODE_DATA_ROOT}"
      --args.episode-data-mode "${EPISODE_DATA_MODE}"
      --args.trajectory-image-size "${TRAJECTORY_IMAGE_SIZE}"
    )
  fi
  if is_true "${SAVE_VIDEOS}"; then
    client_cmd+=(--args.save-videos)
  fi
  if is_true "${FAIL_ON_EPISODE_ERROR}"; then
    client_cmd+=(--args.fail-on-episode-error)
  fi
  (
    export PYTHONPATH="${PYTHONPATH_VALUE}"
    export LIBERO_CONFIG_PATH
    export MUJOCO_GL="${MUJOCO_BACKEND}"
    export PYOPENGL_PLATFORM
    export MUJOCO_EGL_DEVICE_ID
    export NUMBA_CACHE_DIR
    export NUMBA_DISABLE_JIT
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export MPLCONFIGDIR
    export CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES}"
    "${client_cmd[@]}"
  ) >"${stdout_file}" 2>&1 &
  client_pid="$!"
  client_pids+=("${client_pid}")
    client_names+=("${suite}:worker${worker_id}")
    echo "Started ${suite} worker=${worker_id}: pid=${client_pid} log=${worker_jsonl_file} stdout=${stdout_file}"
  done
done

status=0
for i in "${!client_pids[@]}"; do
  if wait "${client_pids[$i]}"; then
    exit_code=0
    echo "Finished ${client_names[$i]}"
  else
    exit_code="$?"
    echo "Failed ${client_names[$i]} with exit code ${exit_code}" >&2
    status=1
  fi
  printf "%s\t%s\n" "${client_names[$i]}" "${exit_code}" >>"${status_file}"
done

# Workers never share a writable JSONL. Merge atomically after every writer exits.
for suite in "${SUITES[@]}"; do
  jsonl_file="${LOG_DIR}/${suite}.jsonl"
  if ! is_true "${RESUME}"; then
    : >"${jsonl_file}"
  fi
  "${OPENPI_PYTHON}" scripts/data/merge_libero_jsonl.py \
    --output "${jsonl_file}" \
    --worker-root "${LOG_DIR}/workers" \
    --pattern "${suite}_worker*.jsonl"
done

# Collapse worker statuses to one suite status for the report.
: >"${status_file}.suites"
for suite in "${SUITES[@]}"; do
  suite_code=0
  while IFS=$'\t' read -r worker_name worker_code; do
    if [[ "${worker_name}" == "${suite}:"* && "${worker_code}" != "0" ]]; then suite_code=1; fi
  done <"${status_file}"
  printf "%s\t%s\n" "${suite}" "${suite_code}" >>"${status_file}.suites"
done
mv "${status_file}.suites" "${status_file}"

if "${OPENPI_PYTHON}" - "${RESULTS_TXT}" "${LOG_DIR}" "${server_log}" "${status_file}" "${SUITES[@]}" <<'PY'
import datetime
import json
from pathlib import Path
import sys

results_path = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
server_log = sys.argv[3]
status_path = Path(sys.argv[4])
suites = sys.argv[5:]

suite_status = {}
if status_path.exists():
    for line in status_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            suite_status[fields[0]] = fields[1]


def load_run_summary(path):
    episodes = {}
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "episode_result":
                key = (int(record["task_id"]), int(record["episode_idx"]))
                episodes[key] = bool(record.get("success", False))
    if not episodes:
        return None
    successes = sum(episodes.values())
    return {
        "total_episodes": len(episodes),
        "total_successes": successes,
        "total_success_rate": successes / len(episodes),
    }


total_episodes = 0
total_successes = 0
summary_rows = []
for suite in suites:
    jsonl_path = log_dir / f"{suite}.jsonl"
    stdout_path = log_dir / "stdout" / f"{suite}.log"
    summary = load_run_summary(jsonl_path)
    exit_code = suite_status.get(suite, "unknown")
    if summary is None:
        summary_rows.append((suite, exit_code, None, None, None, jsonl_path, stdout_path))
        continue
    episodes = int(summary.get("total_episodes", 0))
    successes = int(summary.get("total_successes", 0))
    rate = float(summary.get("total_success_rate", 0.0))
    total_episodes += episodes
    total_successes += successes
    summary_rows.append((suite, exit_code, episodes, successes, rate, jsonl_path, stdout_path))

results_path.parent.mkdir(parents=True, exist_ok=True)
with results_path.open("w", encoding="utf-8") as f:
    f.write("OptimusVLA LIBERO Evaluation Results\n")
    f.write(f"generated_at: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"log_dir: {log_dir.as_posix()}\n")
    f.write(f"server_log: {server_log}\n\n")
    f.write("suite\texit_code\tepisodes\tsuccesses\tsuccess_rate\tjsonl\tstdout\n")
    for suite, exit_code, episodes, successes, rate, jsonl_path, stdout_path in summary_rows:
        if rate is None:
            f.write(
                f"{suite}\t{exit_code}\tNA\tNA\tNA\t"
                f"{jsonl_path.as_posix()}\t{stdout_path.as_posix()}\n"
            )
        else:
            f.write(
                f"{suite}\t{exit_code}\t{episodes}\t{successes}\t{rate:.4f}\t"
                f"{jsonl_path.as_posix()}\t{stdout_path.as_posix()}\n"
            )
    if total_episodes > 0:
        overall_rate = total_successes / total_episodes
        f.write(
            f"\noverall_success_rate: {overall_rate:.4f} "
            f"({total_successes}/{total_episodes})\n"
        )
PY
then
  echo "Wrote evaluation results to ${RESULTS_TXT}"
else
  echo "Failed to write evaluation results to ${RESULTS_TXT}" >&2
  status=1
fi

cleanup
trap - EXIT
exit "${status}"
