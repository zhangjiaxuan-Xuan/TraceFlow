#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

MODE="${MODE:?MODE must be base, success, failure, or both}"
case "${MODE}" in base|success|failure|both) ;; *) echo "Invalid MODE=${MODE}" >&2; exit 2 ;; esac
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
BANK_ROOT="${BANK_ROOT:-}"
GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SEED="${SEED:-7}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-100}"
ENV_WORKERS="${ENV_WORKERS:-32}"
SHARDS_PER_TASK="${SHARDS_PER_TASK:-4}"
FEATURE_IMAGE_SIZE="${FEATURE_IMAGE_SIZE:-128}"
TASK_IDS_CSV="${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-0}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
EPISODE_DATA_MODE="${EPISODE_DATA_MODE:-all}"
SMOKE="${SMOKE:-0}"
ENVIRONMENT_ID_PREFIX="${ENVIRONMENT_ID_PREFIX:-selfcl-round}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/local/data/openpi}"
LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-${OPENPI_ROOT}/third_party/libero}"
ACTION_NORM_STATS_PATH="${ACTION_NORM_STATS_PATH:-${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json}"
PYTHONPATH_VALUE="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${LIBERO_SOURCE_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"; fi
export PYTHONPATH_VALUE

[[ "${RUN_ROOT}" == /* ]] || { echo "RUN_ROOT must be an absolute path: ${RUN_ROOT}" >&2; exit 2; }
if [[ "${SMOKE}" != "1" && "${EPISODES_PER_TASK}" -ne 50 ]]; then
  echo "This protocol fixes 50 episodes per task outside SMOKE=1." >&2
  exit 2
fi
case "${EPISODE_DATA_MODE}" in all|successes|failures|none) ;; *) echo "Invalid EPISODE_DATA_MODE=${EPISODE_DATA_MODE}" >&2; exit 2 ;; esac
[[ "${INFERENCE_BATCH_SIZE}" -gt 0 && "${ENV_WORKERS}" -gt 0 && "${SHARDS_PER_TASK}" -gt 0 ]] || exit 2
for path in "${OPENPI_PYTHON}" "${LIBERO_PYTHON}" "${POLICY_DIR}/model.safetensors" \
  "${TASK_HEAD_CKPT}" "${ACTION_NORM_STATS_PATH}" \
  "${OPENPI_ROOT}/optimus_eval/libero_dynamic_episode_scheduler.py" \
  "${OPENPI_ROOT}/scripts/eval/wait_for_websocket.py"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

positive_args=()
negative_args=()
if [[ "${MODE}" != "base" ]]; then
  [[ -n "${BANK_ROOT}" && -f "${BANK_ROOT}/build_summary.json" ]] || {
    echo "MODE=${MODE} requires a completed BANK_ROOT" >&2; exit 2;
  }
fi
if [[ "${MODE}" == "success" || "${MODE}" == "both" ]]; then
  positive_args=(
    --memory-meta-path "${BANK_ROOT}/positive/gpm_memory_meta.pt"
    --faiss-index-path "${BANK_ROOT}/positive/gpm_memory.index"
    --memory-actions-path "${BANK_ROOT}/positive/gpm_memory_actions.npz"
    --memory-top-k 16
  )
fi
if [[ "${MODE}" == "failure" || "${MODE}" == "both" ]]; then
  negative_args=(
    --use-negative-guidance
    --negative-memory-meta-path "${BANK_ROOT}/negative/gpm_negative_memory_meta.pt"
    --negative-faiss-index-path "${BANK_ROOT}/negative/gpm_negative_memory.index"
    --negative-memory-actions-path "${BANK_ROOT}/negative/gpm_negative_memory_actions.npz"
    --negative-memory-top-k 8
    --negative-memory-min-similarity -1.0
    --negative-memory-min-confidence 0.0
    --negative-guidance-beta 0.10
    --negative-guidance-sigma 0.30
    --negative-guidance-norm-cap 0.10
  )
fi

# Create the round root before the private LIBERO runtime bootstrap. On AOSS,
# creating the nested runtime path before its run root exists can fail without
# leaving a useful round log behind.
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/cache/numba" "${RUN_ROOT}/cache/matplotlib" \
  "${RUN_ROOT}/libero_config/datasets" "${RUN_ROOT}/episode_data"
source "${OPENPI_ROOT}/scripts/eval/bootstrap_libero_runtime.sh"
cat >"${RUN_ROOT}/libero_config/config.yaml" <<EOF
benchmark_root: ${LIBERO_SOURCE_ROOT}/libero/libero
bddl_files: ${LIBERO_SOURCE_ROOT}/libero/libero/bddl_files
init_states: ${LIBERO_SOURCE_ROOT}/libero/libero/init_files
datasets: ${RUN_ROOT}/libero_config/datasets
assets: ${LIBERO_SOURCE_ROOT}/libero/libero/assets
EOF

server_cmd=(
  "${OPENPI_PYTHON}" scripts/serve_policy.py --env LIBERO --port "${PORT}" --seed "${SEED}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" --inference-batch-wait-ms "${INFERENCE_BATCH_WAIT_MS}"
  policy:checkpoint --policy.config pi05_libero --policy.dir "${POLICY_DIR}"
)
if [[ "${MODE}" == "base" ]]; then
  server_cmd=(
    "${OPENPI_PYTHON}" scripts/serve_policy.py --env LIBERO --port "${PORT}" --seed "${SEED}"
    --inference-batch-size "${INFERENCE_BATCH_SIZE}" --inference-batch-wait-ms "${INFERENCE_BATCH_WAIT_MS}"
    --no-use-memory policy:checkpoint --policy.config pi05_libero --policy.dir "${POLICY_DIR}"
  )
else
  positive_mode_args=()
  if [[ "${MODE}" == "failure" ]]; then positive_mode_args+=(--no-use-positive-memory); fi
  server_cmd=(
    "${OPENPI_PYTHON}" scripts/serve_policy.py --env LIBERO --port "${PORT}" --seed "${SEED}"
    --inference-batch-size "${INFERENCE_BATCH_SIZE}" --inference-batch-wait-ms "${INFERENCE_BATCH_WAIT_MS}"
    --use-memory --task-head-ckpt "${TASK_HEAD_CKPT}"
    --action-norm-stats-path "${ACTION_NORM_STATS_PATH}" --action-use-quantile-norm
    --memory-guidance-only --memory-guidance-time-version v1 --memory-guidance-num-steps 10
    --memory-guidance-lambda-max 0.20 --memory-guidance-t-cut 0.30
    --memory-guidance-sigma 0.30 --memory-guidance-norm-cap 0.20
    --memory-guidance-min-similarity -1.0 --memory-guidance-total-norm-cap 0.20
    "${positive_mode_args[@]}"
    "${positive_args[@]}" "${negative_args[@]}"
    policy:checkpoint --policy.config pi05_libero --policy.dir "${POLICY_DIR}"
  )
fi

server_pid=""
scheduler_pid=""
cleanup() {
  if [[ -n "${scheduler_pid}" ]] && kill -0 "${scheduler_pid}" 2>/dev/null; then kill "${scheduler_pid}" 2>/dev/null || true; wait "${scheduler_pid}" 2>/dev/null || true; fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" OPENPI_TORCH_COMPILE=0 CUDA_VISIBLE_DEVICES="${GPU}" \
  "${server_cmd[@]}" >"${RUN_ROOT}/server.log" 2>&1 &
server_pid="$!"
"${OPENPI_PYTHON}" scripts/eval/wait_for_websocket.py --host 127.0.0.1 --port "${PORT}" --timeout 900 --pid "${server_pid}"

scheduler_args=(
  "${OPENPI_PYTHON}" -m optimus_eval.libero_dynamic_episode_scheduler
  --openpi-root "${OPENPI_ROOT}" --libero-python "${LIBERO_PYTHON}" --run-root "${RUN_ROOT}"
  --host 127.0.0.1 --port "${PORT}" --num-workers "${ENV_WORKERS}"
  --shards-per-task "${SHARDS_PER_TASK}" --max-shard-retries 2
  --episodes-per-task "${EPISODES_PER_TASK}" --episode-start 0 --seed "${SEED}"
  --task-ids-csv "${TASK_IDS_CSV}" --max-env-steps "${MAX_ENV_STEPS}"
  --episode-data-mode "${EPISODE_DATA_MODE}"
  --trajectory-image-size "${FEATURE_IMAGE_SIZE}" --environment-id-prefix "${ENVIRONMENT_ID_PREFIX}"
)
if [[ "${EPISODE_DATA_MODE}" != "none" ]]; then
  scheduler_args+=(--episode-data-root "${RUN_ROOT}/episode_data")
fi
if [[ "${SAVE_VIDEOS}" == "1" ]]; then scheduler_args+=(--save-videos); else scheduler_args+=(--no-save-videos); fi
PYTHONPATH="${PYTHONPATH_VALUE}" \
LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" NUMBA_DISABLE_JIT=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" CUDA_VISIBLE_DEVICES="${GPU}" \
  "${scheduler_args[@]}" >"${RUN_ROOT}/scheduler.log" 2>&1 &
scheduler_pid="$!"
wait "${scheduler_pid}"
scheduler_pid=""
echo "Self-CL round complete: mode=${MODE} root=${RUN_ROOT}"
