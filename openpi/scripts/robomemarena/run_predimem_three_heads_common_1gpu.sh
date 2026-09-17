#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?Usage: $0 v0|v3}
case "${MODE}" in v0|v3) ;; *) echo "MODE must be v0 or v3" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ROOT}/../RoboMemArena"
OPENPI_PY=${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}
VLM_PY=${VLM_PY:-/path/to/user/miniforge3/envs/predimem-vlm/bin/python}
GPU=${GPU:-0}
PORT=${PORT:-8320}
SEED=${SEED:-7}
NUM_TRIALS=${NUM_TRIALS:-50}
TASKS_JSON=${TASKS_JSON:-'[1,2,3,18,19,22,25,26]'}
HEAD_VARIANTS=${HEAD_VARIANTS:-"lower upper fusion"}

AOSS_ROOT=${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}
VLM_CKPT=${VLM_CKPT:-/path/to/local/data/robomemarena/models/PrediMem/vlm_tasks1to26_ckpt74500}
VLA_CKPT=${VLA_CKPT:-${AOSS_ROOT}/checkpoints/vla_alltask_pytorch}
ACTION_STATS=${ACTION_STATS:-${VLA_CKPT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${AOSS_ROOT}/eval/${MODE}_${RUN_ID}}
LIBERO_SRC=${LIBERO_SRC:-${ROOT}/third_party/libero}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${AOSS_ROOT}/runtime/openpi_data}
OPENPI_TORCH_COMPILE=${OPENPI_TORCH_COMPILE:-0}
LOCAL_PROXY=${LOCAL_PROXY:-http://127.0.0.1:7897}

required=("${OPENPI_PY}" "${VLM_PY}" "${VLA_CKPT}/model.safetensors" "${VLM_CKPT}/config.json" "${ACTION_STATS}")
for path in "${required[@]}"; do [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }; done
for variant in ${HEAD_VARIANTS}; do
  case "${variant}" in lower|upper|fusion) ;; *) echo "Invalid head variant: ${variant}" >&2; exit 2 ;; esac
  for path in \
    "${AOSS_ROOT}/heads/${variant}/best.pt" \
    "${AOSS_ROOT}/memory/${variant}/gpm_memory_meta.pt" \
    "${AOSS_ROOT}/memory/${variant}/gpm_memory.index"; do
    [[ -f "${path}" ]] || { echo "Missing trained artifact: ${path}" >&2; exit 2; }
  done
done
[[ -f "${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz" ]] || { echo "Missing shared action store" >&2; exit 2; }
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed: mode=${MODE} tasks=${TASKS_JSON} trials_per_task=${NUM_TRIALS} gpu=${GPU} root=${AOSS_ROOT}"
  exit 0
fi

mkdir -p "${RUN_ROOT}"
mkdir -p "${OPENPI_DATA_HOME}"
export OPENPI_ROOT="${ROOT}"
export OPENPI_INFERENCE_ROOT="${ROOT}"
export TARGET_LIBERO_PATH="${LIBERO_SRC}"
export PYTHONPATH="${LIBERO_SRC}:${ARENA_ROOT}/evaluation_benchmark/openpi_minimal_runtime:${ROOT}/packages/openpi-client/src:${ROOT}/packages/openpi/src:${ROOT}:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME OPENPI_TORCH_COMPILE
if [[ -n "${LOCAL_PROXY}" ]]; then
  export HTTP_PROXY="${LOCAL_PROXY}" HTTPS_PROXY="${LOCAL_PROXY}" ALL_PROXY="${LOCAL_PROXY}"
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
fi
export TASK_CONFIG="${ARENA_ROOT}/evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json"
export VLM_CKPT TASKS_JSON NUM_TRIALS SEED
export HOST=127.0.0.1 VLM_DEVICE=cuda:0 ASYNC_VLM=1 VLM_INTERVAL=${VLM_INTERVAL:-5}
export N_RECENT=5 VLM_USE_WRIST=1 VLM_USE_KEYFRAME_MEMORY=1
export MAX_STEPS=${MAX_STEPS:-2500} REPLAN_STEPS=${REPLAN_STEPS:-10} NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}

for variant in ${HEAD_VARIANTS}; do
  OUT_ROOT="${RUN_ROOT}/${variant}"
  if [[ "${RESUME:-0}" == "1" && -f "${OUT_ROOT}/summary.json" ]]; then
    echo "[skip] completed ${MODE}/${variant}: ${OUT_ROOT}"
    continue
  fi
  if [[ -d "${OUT_ROOT}" && -n "$(find "${OUT_ROOT}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite partial run: ${OUT_ROOT}; use a new RUN_ID" >&2
    exit 2
  fi
  mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/videos" "${OUT_ROOT}/libero_config"
  mkdir -p "${OUT_ROOT}/cache/numba" "${OUT_ROOT}/cache/matplotlib"
  cat >"${OUT_ROOT}/libero_config/config.yaml" <<EOF
benchmark_root: ${ROOT}/third_party/libero/libero/libero
bddl_files: ${ROOT}/third_party/libero/libero/libero/bddl_files
init_states: ${ROOT}/third_party/libero/libero/libero/init_files
datasets: ${OUT_ROOT}/libero_config/datasets
assets: ${ROOT}/third_party/libero/libero/libero/assets
EOF
  mkdir -p "${OUT_ROOT}/libero_config/datasets"
  export OUT_ROOT VIDEO_DIR="${OUT_ROOT}/videos"
  export SUMMARY_JSON="${OUT_ROOT}/summary.json" SUMMARY_TSV="${OUT_ROOT}/summary.tsv"
  export PROMPT_TRACE_TSV="${OUT_ROOT}/prompt_trace.tsv" PORT
  export LIBERO_CONFIG_PATH="${OUT_ROOT}/libero_config"
  export NUMBA_CACHE_DIR="${OUT_ROOT}/cache/numba"
  export MPLCONFIGDIR="${OUT_ROOT}/cache/matplotlib"
  SERVER_LOG="${OUT_ROOT}/logs/server.log"
  EVAL_LOG="${OUT_ROOT}/logs/eval.log"

  server=(
    "${OPENPI_PY}" "${ROOT}/scripts/serve_policy.py"
    --env LIBERO --port "${PORT}" --seed "${SEED}"
    --use-memory
    --task-head-ckpt "${AOSS_ROOT}/heads/${variant}/best.pt"
    --memory-meta-path "${AOSS_ROOT}/memory/${variant}/gpm_memory_meta.pt"
    --faiss-index-path "${AOSS_ROOT}/memory/${variant}/gpm_memory.index"
    --memory-actions-path "${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz"
    --action-norm-stats-path "${ACTION_STATS}" --action-use-quantile-norm
    --memory-top-k "${MEMORY_TOP_K:-8}" --memory-progress-window "${MEMORY_PROGRESS_WINDOW:-0.20}"
    --align-mode hybrid --mixture-mode gaussian --nfe-min 1 --nfe-max 10
  )
  if [[ "${MODE}" == "v0" ]]; then
    server+=(--memory-guidance-only --memory-guidance-time-version v0 --memory-guidance-num-steps 10)
  else
    server+=(
      --memory-prior-substep-guidance --memory-prior-guidance-version v3_prior_decay
      --memory-guidance-v2-magnitude-cap 0
      --memory-guidance-lambda-max 0.20 --memory-prior-guidance-final-scale 0.01
    )
  fi
  server+=(policy:checkpoint --policy.config pi05_robomemarena_extra8_reactive --policy.dir "${VLA_CKPT}")

  cleanup() {
    [[ -n "${SERVER_PID:-}" ]] && kill "${SERVER_PID}" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM
  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${server[@]}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  ready=0
  for _ in $(seq 1 300); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 200 "${SERVER_LOG}" >&2
      exit 1
    fi
    if "${OPENPI_PY}" -c "import socket;s=socket.create_connection(('127.0.0.1',${PORT}),1);s.close()" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" == "1" ]] || { tail -n 200 "${SERVER_LOG}" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID=0 \
    "${VLM_PY}" "${ARENA_ROOT}/evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py" \
    2>&1 | tee "${EVAL_LOG}"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  unset SERVER_PID
  PORT=$((PORT + 1))
done

echo "Completed ${MODE} three-head evaluation: ${RUN_ROOT}"
