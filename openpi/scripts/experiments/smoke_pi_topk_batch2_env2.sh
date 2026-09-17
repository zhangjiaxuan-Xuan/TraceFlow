#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-${OPENPI_ROOT}/third_party/libero}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
ABLATION_ROOT="${ABLATION_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_positive_topk_ablation_v1}"
POSITIVE_ROOT="${ABLATION_ROOT}/banks/s01_fmax/positive"
NEGATIVE_ROOT="${ABLATION_ROOT}/banks/fixed_cpi_n_fmax/negative"
GPU="${GPU:-0}"
PORT="${PORT:-8299}"
RUN_ROOT="${RUN_ROOT:-${TMPDIR:-/tmp}/bcpi_batch2_env2_smoke_$$}"

cd "${OPENPI_ROOT}"
PYTHONPATH_VALUE="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${LIBERO_SOURCE_ROOT}"
export PYTHONPATH="${PYTHONPATH_VALUE}"
source "${OPENPI_ROOT}/scripts/eval/bootstrap_libero_runtime.sh"

for path in \
  "${PYTHON}" \
  "${LIBERO_PYTHON}" \
  "${LIBERO_SOURCE_ROOT}/libero/libero/__init__.py" \
  "${POLICY_DIR}/model.safetensors" \
  "${TASK_HEAD_CKPT}" \
  "${POSITIVE_ROOT}/gpm_memory_meta.pt" \
  "${POSITIVE_ROOT}/gpm_memory.index" \
  "${POSITIVE_ROOT}/gpm_memory_actions.npz" \
  "${NEGATIVE_ROOT}/gpm_negative_memory_meta.pt" \
  "${NEGATIVE_ROOT}/gpm_negative_memory.index" \
  "${NEGATIVE_ROOT}/gpm_negative_memory_actions.npz"; do
  [[ -f "${path}" ]] || { echo "Missing smoke artifact: ${path}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}/libero_config/datasets" "${RUN_ROOT}/cache/numba" "${RUN_ROOT}/cache/matplotlib"
cat >"${RUN_ROOT}/libero_config/config.yaml" <<EOF
benchmark_root: ${LIBERO_SOURCE_ROOT}/libero/libero
bddl_files: ${LIBERO_SOURCE_ROOT}/libero/libero/bddl_files
init_states: ${LIBERO_SOURCE_ROOT}/libero/libero/init_files
datasets: ${RUN_ROOT}/libero_config/datasets
assets: ${LIBERO_SOURCE_ROOT}/libero/libero/assets
EOF

"${LIBERO_PYTHON}" scripts/eval/patch_robosuite_egl.py

server_pid=""
client_a_pid=""
client_b_pid=""
cleanup() {
  for pid in "${client_a_pid}" "${client_b_pid}" "${server_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OPENPI_DATA_HOME=/path/to/local/data/openpi \
OPENPI_TORCH_COMPILE=0 \
CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PYTHON}" scripts/serve_policy.py \
    --env LIBERO \
    --port "${PORT}" \
    --seed 7 \
    --inference-batch-size 2 \
    --inference-batch-wait-ms 100 \
    --use-memory \
    --task-head-ckpt "${TASK_HEAD_CKPT}" \
    --action-norm-stats-path "${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json" \
    --action-use-quantile-norm \
    --memory-meta-path "${POSITIVE_ROOT}/gpm_memory_meta.pt" \
    --faiss-index-path "${POSITIVE_ROOT}/gpm_memory.index" \
    --memory-actions-path "${POSITIVE_ROOT}/gpm_memory_actions.npz" \
    --memory-top-k 8 \
    --memory-refresh-every 1 \
    --align-mode hybrid \
    --mixture-mode gaussian \
    --memory-guidance-only \
    --memory-guidance-time-version v1 \
    --memory-guidance-num-steps 10 \
    --memory-guidance-lambda-max 0.20 \
    --memory-guidance-t-cut 0.30 \
    --memory-guidance-sigma 0.30 \
    --memory-guidance-norm-cap 0.20 \
    --memory-guidance-min-similarity -1.0 \
    --use-negative-guidance \
    --negative-memory-meta-path "${NEGATIVE_ROOT}/gpm_negative_memory_meta.pt" \
    --negative-faiss-index-path "${NEGATIVE_ROOT}/gpm_negative_memory.index" \
    --negative-memory-actions-path "${NEGATIVE_ROOT}/gpm_negative_memory_actions.npz" \
    --negative-memory-top-k 8 \
    --negative-memory-min-similarity -1.0 \
    --negative-memory-min-confidence 0.0 \
    --negative-guidance-beta 0.10 \
    --negative-guidance-sigma 0.30 \
    --negative-guidance-norm-cap 0.10 \
    --memory-guidance-total-norm-cap 0.20 \
    --memory-guidance-trace-dir "${RUN_ROOT}/traces" \
    --memory-guidance-trace-level light \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "${POLICY_DIR}" >"${RUN_ROOT}/server.log" 2>&1 &
server_pid="$!"

"${PYTHON}" scripts/eval/wait_for_websocket.py \
  --host 127.0.0.1 --port "${PORT}" --timeout 900 --pid "${server_pid}"

run_client() {
  local task_id="$1"
  local log_file="${RUN_ROOT}/task_${task_id}.jsonl"
  PYTHONPATH="${PYTHONPATH_VALUE}" \
  LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
  NUMBA_DISABLE_JIT=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" \
  MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID=0 \
  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${LIBERO_PYTHON}" examples/libero/main.py \
      --args.host 127.0.0.1 \
      --args.port "${PORT}" \
      --args.task-suite-name libero_10 \
      --args.task-ids-csv "${task_id}" \
      --args.num-trials-per-task 1 \
      --args.episode-start 900 \
      --args.episode-indices-csv 900 \
      --args.replan-steps 10 \
      --args.num-steps-wait 10 \
      --args.max-env-steps 20 \
      --args.resize-size 224 \
      --args.seed 7 \
      --args.log-file "${log_file}" \
      --args.environment-id "smoke-task-${task_id}" \
      --args.fail-on-episode-error >"${RUN_ROOT}/task_${task_id}.log" 2>&1
}

run_client 0 &
client_a_pid="$!"
run_client 1 &
client_b_pid="$!"
wait "${client_a_pid}"
client_a_pid=""
wait "${client_b_pid}"
client_b_pid=""

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for task_id in (0, 1):
    rows = [json.loads(line) for line in (root / f"task_{task_id}.jsonl").read_text().splitlines()]
    episodes = [row for row in rows if row.get("event") == "episode_result"]
    if len(episodes) != 1 or episodes[0].get("error"):
        raise SystemExit(f"Invalid smoke result for task {task_id}: {episodes}")
    if int(episodes[0].get("policy_calls", 0)) < 1:
        raise SystemExit(f"No policy inference completed for task {task_id}")
print(f"Pi top-k batch2/env2 smoke passed: {root}")
PY
