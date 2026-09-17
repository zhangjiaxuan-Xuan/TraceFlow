#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"
LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-${OPENPI_ROOT}/third_party/libero}"
[[ -f "${LIBERO_SOURCE_ROOT}/libero/libero/__init__.py" ]] || {
  echo "Invalid LIBERO_SOURCE_ROOT: ${LIBERO_SOURCE_ROOT}" >&2
  exit 1
}
PYTHONPATH_VALUE="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${LIBERO_SOURCE_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"; fi
export PYTHONPATH="${PYTHONPATH_VALUE}"

PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/local/data/openpi}"
POLICY_DIR="${POLICY_DIR:?POLICY_DIR is required}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:?TASK_HEAD_CKPT is required}"
MEMORY_META_PATH="${MEMORY_META_PATH:?MEMORY_META_PATH is required}"
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:?FAISS_INDEX_PATH is required}"
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:?MEMORY_ACTIONS_PATH is required}"
NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH:?NEGATIVE_MEMORY_META_PATH is required}"
NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH:?NEGATIVE_FAISS_INDEX_PATH is required}"
NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH:?NEGATIVE_MEMORY_ACTIONS_PATH is required}"
MEMORY_TOP_K="${MEMORY_TOP_K:?MEMORY_TOP_K is required}"
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:?NEGATIVE_MEMORY_TOP_K is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SEED="${SEED:-7}"
RESUME="${RESUME:-auto}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
ACTION_NORM_STATS_PATH="${ACTION_NORM_STATS_PATH:-${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json}"

[[ "${MEMORY_TOP_K}" =~ ^(1|8|16|32)$ ]] || { echo "Invalid MEMORY_TOP_K=${MEMORY_TOP_K}" >&2; exit 2; }
[[ "${NEGATIVE_MEMORY_TOP_K}" == "8" ]] || { echo "NEGATIVE_MEMORY_TOP_K must be 8." >&2; exit 2; }
[[ "${SEED}" == "7" ]] || { echo "SEED must be 7." >&2; exit 2; }

for path in \
  "${PYTHON}" \
  "${LIBERO_PYTHON}" \
  "${POLICY_DIR}/model.safetensors" \
  "${TASK_HEAD_CKPT}" \
  "${ACTION_NORM_STATS_PATH}" \
  "${MEMORY_META_PATH}" \
  "${FAISS_INDEX_PATH}" \
  "${MEMORY_ACTIONS_PATH}" \
  "${NEGATIVE_MEMORY_META_PATH}" \
  "${NEGATIVE_FAISS_INDEX_PATH}" \
  "${NEGATIVE_MEMORY_ACTIONS_PATH}" \
  "${OPENPI_ROOT}/optimus_eval/libero_dynamic_episode_scheduler.py" \
  "${OPENPI_ROOT}/scripts/eval/wait_for_websocket.py"; do
  [[ -f "${path}" ]] || { echo "Missing required artifact: ${path}" >&2; exit 1; }
done

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  "${PYTHON}" -m optimus_eval.libero_dynamic_episode_scheduler \
    --openpi-root "${OPENPI_ROOT}" \
    --libero-python "${LIBERO_PYTHON}" \
    --run-root "${RUN_ROOT}" \
    --port "${PORT}" \
    --num-workers 32 \
    --shards-per-task 4 \
    --episodes-per-task 50 \
    --episode-start 100 \
    --preflight-only
  echo "Pi top-k batch16/env32 candidate preflight passed."
  exit 0
fi

if [[ "${RESUME}" == "auto" ]]; then
  if [[ -f "${RUN_ROOT}/run_config.json" ]]; then RESUME=1; else RESUME=0; fi
fi
meaningful_run_entry=""
if [[ -d "${RUN_ROOT}" ]]; then
  meaningful_run_entry="$(
    find "${RUN_ROOT}" -mindepth 1 \
      \( -path "${RUN_ROOT}/runtime" -o -path "${RUN_ROOT}/runtime/*" \) -prune \
      -o -print -quit
  )"
fi
if [[ "${RESUME}" == "0" ]] && [[ -n "${meaningful_run_entry}" ]]; then
  echo "Refusing non-empty RUN_ROOT without resume: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ "${RESUME}" == "1" ]] && [[ ! -f "${RUN_ROOT}/run_config.json" ]]; then
  echo "Resume requires ${RUN_ROOT}/run_config.json" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/cache/numba" "${RUN_ROOT}/cache/matplotlib" "${RUN_ROOT}/libero_config/datasets"
"${PYTHON}" - "${RUN_ROOT}/run_config.json" "${POLICY_DIR}" "${TASK_HEAD_CKPT}" \
  "${MEMORY_META_PATH}" "${NEGATIVE_MEMORY_META_PATH}" "${MEMORY_TOP_K}" \
  "${NEGATIVE_MEMORY_TOP_K}" "${RESUME}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

output = Path(sys.argv[1])
expected = {
    "protocol": "pi_v1_bcpi_positive_topk_libero10_batch16_env32_v3",
    "policy_dir": str(Path(sys.argv[2]).resolve()),
    "task_head_sha256": digest(sys.argv[3]),
    "positive_meta_sha256": digest(sys.argv[4]),
    "negative_meta_sha256": digest(sys.argv[5]),
    "positive_top_k": int(sys.argv[6]),
    "negative_top_k": int(sys.argv[7]),
    "suite": "libero_10",
    "tasks": 10,
    "episode_start": 100,
    "episodes_per_task": 50,
    "policy_batch_size": 16,
    "policy_batch_wait_ms": 100,
    "environment_workers": 32,
    "shards_per_task": 4,
    "max_shard_retries": 2,
    "dynamic_assignment": True,
    "guidance": "V1 positive+negative fixed10",
    "save_videos": False,
}
resume = sys.argv[8] == "1"
if output.exists():
    actual = json.loads(output.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit(f"Resume configuration mismatch:\nexpected={expected}\nactual={actual}")
elif resume:
    raise SystemExit(f"Resume requested without run config: {output}")
else:
    output.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

source "${OPENPI_ROOT}/scripts/eval/bootstrap_libero_runtime.sh"

cat >"${RUN_ROOT}/libero_config/config.yaml" <<EOF
benchmark_root: ${LIBERO_SOURCE_ROOT}/libero/libero
bddl_files: ${LIBERO_SOURCE_ROOT}/libero/libero/bddl_files
init_states: ${LIBERO_SOURCE_ROOT}/libero/libero/init_files
datasets: ${RUN_ROOT}/libero_config/datasets
assets: ${LIBERO_SOURCE_ROOT}/libero/libero/assets
EOF

PYTHONPATH="${PYTHONPATH_VALUE}" \
LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
NUMBA_DISABLE_JIT=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
  "${LIBERO_PYTHON}" - <<'PY'
import yaml
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_10"]()
states = suite.get_task_init_states(0)
if suite.n_tasks != 10 or tuple(states.shape) != (50, 123):
    raise SystemExit(f"Unexpected LIBERO runtime inventory: tasks={suite.n_tasks} states={states.shape}")
print("LIBERO runtime audit passed: yaml, benchmark, environment, init states")
PY

if ! "${PYTHON}" - "${PORT}" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.settimeout(0.5)
    occupied = sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
raise SystemExit(1 if occupied else 0)
PY
then
  echo "Port ${PORT} is already occupied." >&2
  exit 2
fi

server_cmd=(
  "${PYTHON}" scripts/serve_policy.py
  --env LIBERO
  --port "${PORT}"
  --seed 7
  --inference-batch-size 16
  --inference-batch-wait-ms 100
  --use-memory
  --task-head-ckpt "${TASK_HEAD_CKPT}"
  --action-norm-stats-path "${ACTION_NORM_STATS_PATH}"
  --action-use-quantile-norm
  --memory-meta-path "${MEMORY_META_PATH}"
  --faiss-index-path "${FAISS_INDEX_PATH}"
  --memory-actions-path "${MEMORY_ACTIONS_PATH}"
  --memory-top-k "${MEMORY_TOP_K}"
  --memory-refresh-every 1
  --align-mode hybrid
  --mixture-mode gaussian
  --memory-guidance-only
  --memory-guidance-time-version v1
  --memory-guidance-num-steps 10
  --memory-guidance-lambda-max 0.20
  --memory-guidance-t-cut 0.30
  --memory-guidance-sigma 0.30
  --memory-guidance-norm-cap 0.20
  --memory-guidance-min-similarity -1.0
  --use-negative-guidance
  --negative-memory-meta-path "${NEGATIVE_MEMORY_META_PATH}"
  --negative-faiss-index-path "${NEGATIVE_FAISS_INDEX_PATH}"
  --negative-memory-actions-path "${NEGATIVE_MEMORY_ACTIONS_PATH}"
  --negative-memory-top-k 8
  --negative-memory-min-similarity -1.0
  --negative-memory-min-confidence 0.0
  --negative-guidance-beta 0.10
  --negative-guidance-sigma 0.30
  --negative-guidance-norm-cap 0.10
  --memory-guidance-total-norm-cap 0.20
  --memory-guidance-trace-dir "${RUN_ROOT}/traces"
  --memory-guidance-trace-level light
  policy:checkpoint
  --policy.config pi05_libero
  --policy.dir "${POLICY_DIR}"
)

server_pid=""
scheduler_pid=""
cleanup() {
  if [[ -n "${scheduler_pid}" ]] && kill -0 "${scheduler_pid}" 2>/dev/null; then
    kill "${scheduler_pid}" 2>/dev/null || true
    wait "${scheduler_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
OPENPI_TORCH_COMPILE=0 \
CUDA_VISIBLE_DEVICES="${GPU}" \
  "${server_cmd[@]}" >"${RUN_ROOT}/server.log" 2>&1 &
server_pid="$!"

"${PYTHON}" scripts/eval/wait_for_websocket.py \
  --host 127.0.0.1 --port "${PORT}" --timeout 900 --pid "${server_pid}"

PYTHONPATH="${PYTHONPATH_VALUE}" \
LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" \
NUMBA_DISABLE_JIT=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PYTHON}" -m optimus_eval.libero_dynamic_episode_scheduler \
    --openpi-root "${OPENPI_ROOT}" \
    --libero-python "${LIBERO_PYTHON}" \
    --run-root "${RUN_ROOT}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --num-workers 32 \
    --shards-per-task 4 \
    --max-shard-retries 2 \
    --episodes-per-task 50 \
    --episode-start 100 \
    --seed 7 \
    --no-save-videos >"${RUN_ROOT}/scheduler.log" 2>&1 &
scheduler_pid="$!"
wait "${scheduler_pid}"
scheduler_pid=""

echo "Pi V1 B+C-pi top-k candidate complete: ${RUN_ROOT}/eval/results.txt"
