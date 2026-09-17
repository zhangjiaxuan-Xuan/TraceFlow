#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8000}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
SUITES_CSV="${SUITES_CSV:-libero_spatial,libero_object,libero_goal,libero_10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/libero_eval_multigpu_${RUN_ID}}"
STOP_ON_EXIT="${STOP_ON_EXIT:-1}"
BACKGROUND="${BACKGROUND:-0}"

POLICY_DIR="${POLICY_DIR:-}"
if [[ -z "${POLICY_DIR}" ]]; then
  echo "Set POLICY_DIR to the pi0.5 PyTorch checkpoint directory." >&2
  exit 1
fi

if [[ ! -f "${POLICY_DIR}/model.safetensors" ]]; then
  echo "Missing policy weights: ${POLICY_DIR}/model.safetensors" >&2
  exit 1
fi

if [[ ! -f "${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json" ]]; then
  echo "Missing policy norm stats: ${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IFS=',' read -r -a SUITES <<< "${SUITES_CSV}"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPUS_CSV did not contain any GPU ids." >&2
  exit 1
fi
if [[ "${#SUITES[@]}" -eq 0 ]]; then
  echo "SUITES_CSV did not contain any suites." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/pids"
ORCH_LOG="${RUN_ROOT}/orchestrator.log"
RESULTS_TSV="${RUN_ROOT}/results.tsv"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${ORCH_LOG}"
}

wait_for_child() {
  local pid="$1"
  if wait "${pid}"; then
    return 0
  fi
  return "$?"
}

PIDS=()

cleanup() {
  local code=$?
  if [[ "${STOP_ON_EXIT}" == "1" ]]; then
    for pid in "${PIDS[@]:-}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
      fi
    done
  fi
  exit "${code}"
}
trap cleanup INT TERM EXIT

cat > "${RUN_ROOT}/launch.env" <<EOF
OPENPI_ROOT=${OPENPI_ROOT}
POLICY_DIR=${POLICY_DIR}
GPUS_CSV=${GPUS_CSV}
SUITES_CSV=${SUITES_CSV}
BASE_PORT=${BASE_PORT}
HOST=${HOST}
RUN_ROOT=${RUN_ROOT}
STOP_ON_EXIT=${STOP_ON_EXIT}
BACKGROUND=${BACKGROUND}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
REPLAN_STEPS=${REPLAN_STEPS:-10}
USE_LCM=${USE_LCM:-1}
LCM_SCALE=${LCM_SCALE:-0.10}
SAVE_VIDEOS=${SAVE_VIDEOS:-0}
VIDEO_ROOT=${VIDEO_ROOT:-${RUN_ROOT}/videos}
EOF

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "openpi_root": "${OPENPI_ROOT}",
  "policy_dir": "${POLICY_DIR}",
  "gpus_csv": "${GPUS_CSV}",
  "suites_csv": "${SUITES_CSV}",
  "base_port": ${BASE_PORT},
  "host": "${HOST}",
  "run_root": "${RUN_ROOT}",
  "num_trials_per_task": "${NUM_TRIALS_PER_TASK:-50}",
  "replan_steps": "${REPLAN_STEPS:-10}",
  "use_lcm": "${USE_LCM:-1}",
  "lcm_scale": "${LCM_SCALE:-0.10}",
  "save_videos": "${SAVE_VIDEOS:-0}",
  "video_root": "${VIDEO_ROOT:-${RUN_ROOT}/videos}"
}
EOF

start_suite() {
  local suite_index="$1"
  local suite="$2"
  local gpu="${GPUS[$((suite_index % ${#GPUS[@]}))]}"
  local port=$((BASE_PORT + suite_index))
  local suite_log_dir="${RUN_ROOT}/${suite}"
  local stdout_log="${RUN_ROOT}/logs/${suite}_gpu${gpu}_port${port}.log"

  mkdir -p "${suite_log_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export SERVER_CUDA_VISIBLE_DEVICES="0"
    export CLIENT_CUDA_VISIBLE_DEVICES="0"
    export HOST
    export PORT="${port}"
    export POLICY_DIR
    export LOG_DIR="${suite_log_dir}"
    export RESULTS_TXT="${suite_log_dir}/results.txt"
    export RUN_ID="${suite}"
    export SAVE_VIDEOS
    export VIDEO_ROOT="${VIDEO_ROOT:-${RUN_ROOT}/videos/${suite}}"
    exec "${SCRIPT_DIR}/run_libero_eval.sh" "${suite}"
  ) >"${stdout_log}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  echo "${pid}" > "${RUN_ROOT}/pids/${suite}.pid"
  log "started suite=${suite} gpu=${gpu} port=${port} pid=${pid} log=${stdout_log}"
}

start_all() {
  log "starting ${#SUITES[@]} suites on gpus=${GPUS_CSV} base_port=${BASE_PORT}"
  local i
  for i in "${!SUITES[@]}"; do
    start_suite "${i}" "${SUITES[$i]}"
  done
}

monitor_all() {
  : > "${RESULTS_TSV}"
  printf "suite\tpid\texit_code\tlog_dir\tresults\n" >> "${RESULTS_TSV}"

  local status=0
  local i
  for i in "${!SUITES[@]}"; do
    local suite="${SUITES[$i]}"
    local pid="${PIDS[$i]}"
    local exit_code=0
    if wait_for_child "${pid}"; then
      exit_code=0
      log "finished suite=${suite} pid=${pid}"
    else
      exit_code="$?"
      status=1
      log "failed suite=${suite} pid=${pid} exit_code=${exit_code}"
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${suite}" \
      "${pid}" \
      "${exit_code}" \
      "${RUN_ROOT}/${suite}" \
      "${RUN_ROOT}/${suite}/results.txt" >> "${RESULTS_TSV}"
  done
  return "${status}"
}

run_foreground() {
  start_all
  monitor_all
  log "multigpu eval finished results=${RESULTS_TSV}"
}

if [[ "${BACKGROUND}" != "1" ]]; then
  run_foreground
  trap - INT TERM EXIT
  exit 0
fi

if command -v setsid >/dev/null 2>&1; then
  setsid env \
    BACKGROUND=0 \
    RUN_ROOT="${RUN_ROOT}" \
    GPUS_CSV="${GPUS_CSV}" \
    SUITES_CSV="${SUITES_CSV}" \
    BASE_PORT="${BASE_PORT}" \
    HOST="${HOST}" \
    POLICY_DIR="${POLICY_DIR}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}" \
    REPLAN_STEPS="${REPLAN_STEPS:-10}" \
    USE_LCM="${USE_LCM:-1}" \
    LCM_SCALE="${LCM_SCALE:-0.10}" \
    SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
    VIDEO_ROOT="${VIDEO_ROOT:-${RUN_ROOT}/videos}" \
    bash -lc "cd '${OPENPI_ROOT}' && '${SCRIPT_DIR}/run_libero_eval_multigpu.sh'" \
    > "${RUN_ROOT}/nohup.log" 2>&1 &
else
  nohup env \
    BACKGROUND=0 \
    RUN_ROOT="${RUN_ROOT}" \
    GPUS_CSV="${GPUS_CSV}" \
    SUITES_CSV="${SUITES_CSV}" \
    BASE_PORT="${BASE_PORT}" \
    HOST="${HOST}" \
    POLICY_DIR="${POLICY_DIR}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}" \
    REPLAN_STEPS="${REPLAN_STEPS:-10}" \
    USE_LCM="${USE_LCM:-1}" \
    LCM_SCALE="${LCM_SCALE:-0.10}" \
    SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
    VIDEO_ROOT="${VIDEO_ROOT:-${RUN_ROOT}/videos}" \
    bash -lc "cd '${OPENPI_ROOT}' && '${SCRIPT_DIR}/run_libero_eval_multigpu.sh'" \
    > "${RUN_ROOT}/nohup.log" 2>&1 &
fi
pid=$!
echo "${pid}" > "${RUN_ROOT}/orchestrator.pid"

log "started background multigpu eval pid=${pid} run_root=${RUN_ROOT} log=${RUN_ROOT}/nohup.log"
trap - INT TERM EXIT
