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

MODE="${MODE:?MODE must be positive or both}"
case "${MODE}" in
  positive) USE_NEGATIVE=0; MODE_TAG=positive ;;
  both) USE_NEGATIVE=1; MODE_TAG=positive_cpi_n_failure ;;
  *) echo "MODE must be positive or both, got ${MODE}" >&2; exit 2 ;;
esac

OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/local/data/openpi}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
C_PI_SOURCE_ROOT="${C_PI_SOURCE_ROOT:-/path/to/local/CVPR26-OptimusVLA/openpi}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
BCPI_ARTIFACT_ROOT="${BCPI_ARTIFACT_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_nfailure_four_suites_v1}"
BANK_ROOT="${BANK_ROOT:-${BCPI_ARTIFACT_ROOT}/bank_b_plus_cpi_success_cpi_n_failure}"
MEMORY_META_PATH="${BANK_ROOT}/positive/gpm_memory_meta.pt"
FAISS_INDEX_PATH="${BANK_ROOT}/positive/gpm_memory.index"
MEMORY_ACTIONS_PATH="${BANK_ROOT}/positive/gpm_memory_actions.npz"
NEGATIVE_MEMORY_META_PATH="${BANK_ROOT}/negative/gpm_negative_memory_meta.pt"
NEGATIVE_FAISS_INDEX_PATH="${BANK_ROOT}/negative/gpm_negative_memory.index"
NEGATIVE_MEMORY_ACTIONS_PATH="${BANK_ROOT}/negative/gpm_negative_memory_actions.npz"
POSITIVE_SELECTION_PATH="${POSITIVE_SELECTION_PATH:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_positive_topk_ablation_v3_batch16_env32/selection/positive_selection.json}"
SELECTED_FOUR_SUITE_ROOT="${SELECTED_FOUR_SUITE_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_positive_topk_ablation_v3_batch16_env32/selected_four_suites}"
POSITIVE_CAPACITY_OVERRIDE="${POSITIVE_CAPACITY_OVERRIDE:-}"
POSITIVE_TOP_K_OVERRIDE="${POSITIVE_TOP_K_OVERRIDE:-0}"
SELECTION_AUDITED="${SELECTION_AUDITED:-0}"
MEMORY_TOP_K=8
POSITIVE_CAPACITY_PER_TASK=max
ACTIVE_POSITIVE_ITEMS=11834
ACTION_NORM_STATS_PATH="${ACTION_NORM_STATS_PATH:-${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json}"

GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SEED="${SEED:-7}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
ENV_WORKERS="${ENV_WORKERS:-32}"
INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-100}"
MAX_TASK_RETRIES="${MAX_TASK_RETRIES:-2}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
RESUME="${RESUME:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"

if [[ "${USE_NEGATIVE}" == "1" ]] && [[ "${PREFLIGHT_ONLY}" != "1" ]]; then
  [[ "${SELECTION_AUDITED}" == "1" ]] || {
    echo "Final four-suite +/- evaluation requires SELECTION_AUDITED=1 after inspecting ${POSITIVE_SELECTION_PATH}" >&2
    exit 2
  }
  [[ -f "${POSITIVE_SELECTION_PATH}" ]] || {
    echo "Missing completed positive ablation selection: ${POSITIVE_SELECTION_PATH}" >&2
    exit 2
  }
fi

if [[ "${RESUME}" == "1" ]] && [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RESUME=1 requires an explicit RUN_ROOT." >&2
  exit 2
fi
RUN_ROOT="${RUN_ROOT:-${OPENPI_ROOT}/logs/pi_v1_bcpi_${MODE_TAG}_four_suites_batch16_env32_${RUN_ID}}"

[[ "${INFERENCE_BATCH_SIZE}" -eq 16 ]] || { echo "INFERENCE_BATCH_SIZE must be 16." >&2; exit 2; }
[[ "${ENV_WORKERS}" -eq 32 ]] || { echo "ENV_WORKERS must be 32." >&2; exit 2; }
[[ "${EPISODES_PER_TASK}" -gt 0 ]] || { echo "EPISODES_PER_TASK must be positive." >&2; exit 2; }

for path in \
  "${OPENPI_PYTHON}" \
  "${LIBERO_PYTHON}" \
  "${POLICY_DIR}/model.safetensors" \
  "${TASK_HEAD_CKPT}" \
  "${ACTION_NORM_STATS_PATH}" \
  "${OPENPI_ROOT}/optimus_eval/libero_dynamic_task_scheduler.py" \
  "${OPENPI_ROOT}/scripts/eval/wait_for_websocket.py"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
POLICY_DIR="${POLICY_DIR}" \
TASK_HEAD_CKPT="${TASK_HEAD_CKPT}" \
C_PI_SOURCE_ROOT="${C_PI_SOURCE_ROOT}" \
BCPI_ARTIFACT_ROOT="${BCPI_ARTIFACT_ROOT}" \
PREFLIGHT_ONLY="${PREFLIGHT_ONLY}" \
  bash scripts/experiments/prepare_pi_v1_bcpi_four_suites_bank.sh

materialize_args=(
  --selection "${POSITIVE_SELECTION_PATH}"
  --source-positive "${BANK_ROOT}/positive"
  --output-root "${SELECTED_FOUR_SUITE_ROOT}"
)
if [[ -n "${POSITIVE_CAPACITY_OVERRIDE}" || "${POSITIVE_TOP_K_OVERRIDE}" != "0" ]]; then
  materialize_args+=(
    --capacity-per-task "${POSITIVE_CAPACITY_OVERRIDE}"
    --top-k "${POSITIVE_TOP_K_OVERRIDE}"
  )
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  if [[ "${USE_NEGATIVE}" == "1" ]]; then
    "${OPENPI_PYTHON}" scripts/experiments/materialize_bcpi_selected_four_suites.py \
      "${materialize_args[@]}" --dry-run
  fi
  "${OPENPI_PYTHON}" -m optimus_eval.libero_dynamic_task_scheduler \
    --openpi-root "${OPENPI_ROOT}" \
    --libero-python "${LIBERO_PYTHON}" \
    --run-root "${RUN_ROOT}" \
    --port "${PORT}" \
    --num-workers "${ENV_WORKERS}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --seed "${SEED}" \
    --preflight-only
  echo "Pi V1 B+C-pi ${MODE} four-suite preflight passed."
  exit 0
fi

"${LIBERO_PYTHON}" scripts/eval/patch_robosuite_egl.py

if [[ "${USE_NEGATIVE}" == "1" ]]; then
  "${OPENPI_PYTHON}" scripts/experiments/materialize_bcpi_selected_four_suites.py "${materialize_args[@]}"
  read -r MEMORY_META_PATH FAISS_INDEX_PATH MEMORY_ACTIONS_PATH MEMORY_TOP_K POSITIVE_CAPACITY_PER_TASK ACTIVE_POSITIVE_ITEMS < <(
    "${OPENPI_PYTHON}" - "${SELECTED_FOUR_SUITE_ROOT}/selected_runtime.json" <<'PY'
import json
from pathlib import Path
import sys
runtime = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(runtime["positive_bank"])
print(
    root / "gpm_memory_meta.pt",
    root / "gpm_memory.index",
    root / "gpm_memory_actions.npz",
    runtime["positive_top_k"],
    runtime["positive_capacity_per_task"],
    runtime["positive_items"],
)
PY
  )
fi

for path in \
  "${BANK_ROOT}/bank_identity.json" \
  "${MEMORY_META_PATH}" \
  "${FAISS_INDEX_PATH}" \
  "${MEMORY_ACTIONS_PATH}" \
  "${NEGATIVE_MEMORY_META_PATH}" \
  "${NEGATIVE_FAISS_INDEX_PATH}" \
  "${NEGATIVE_MEMORY_ACTIONS_PATH}"; do
  [[ -f "${path}" ]] || { echo "Missing prepared B+C-pi bank artifact: ${path}" >&2; exit 1; }
done

if [[ "${USE_NEGATIVE}" == "1" ]] && [[ "${POSITIVE_CAPACITY_PER_TASK}" == "50" ]] && [[ "${MEMORY_TOP_K}" == "16" ]]; then
  "${OPENPI_PYTHON}" - "${MEMORY_META_PATH}" "${NEGATIVE_MEMORY_META_PATH}" <<'PY'
from collections import Counter
from pathlib import Path
import sys

import torch

positive = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
negative = torch.load(Path(sys.argv[2]), map_location="cpu", weights_only=False)

positive_tasks = Counter((str(item["suite"]), str(item["task_name"])) for item in positive)
positive_suites = Counter(str(item["suite"]) for item in positive)
if len(positive) != 2000 or len(positive_tasks) != 40 or set(positive_tasks.values()) != {50}:
    raise SystemExit(
        "Final positive bank must be the audited B-only 50/task slice: "
        f"items={len(positive)} tasks={len(positive_tasks)} capacities={sorted(set(positive_tasks.values()))}"
    )
if positive_suites != Counter(
    {"libero_spatial": 500, "libero_object": 500, "libero_goal": 500, "libero_10": 500}
):
    raise SystemExit(f"Unexpected final positive suite inventory: {positive_suites}")
if any(item.get("provenance") for item in positive):
    raise SystemExit("Final positive bank contains non-B provenance; expected B-only 50/task")

n_failures = sum(
    (item.get("provenance") or {}).get("source_family") == "new_pi"
    for item in negative
)
cpi_failures = sum(
    (item.get("provenance") or {}).get("collection_group") == "C-pi"
    for item in negative
)
if len(negative) != 613 or n_failures != 497 or cpi_failures != 116:
    raise SystemExit(
        "Final negative bank must be N_failure497 + C_pi_failure116: "
        f"items={len(negative)} N={n_failures} C_pi={cpi_failures}"
    )
if n_failures + cpi_failures != len(negative):
    raise SystemExit("Final negative bank contains an unregistered source")

print(
    "Final memory contract verified: positive=B-only 50/task (2000), k+=16; "
    "negative=N497+C_pi116 (613), k-=8"
)
PY
fi

if [[ "${RESUME}" == "1" ]]; then
  [[ -f "${RUN_ROOT}/run_config.json" ]] || {
    echo "RESUME=1 requires ${RUN_ROOT}/run_config.json" >&2
    exit 2
  }
elif [[ -d "${RUN_ROOT}" ]] && [[ -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to mix results into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/cache/numba" "${RUN_ROOT}/cache/matplotlib" "${RUN_ROOT}/libero_config/datasets"
RUN_CONFIG="${RUN_ROOT}/run_config.json"
"${OPENPI_PYTHON}" - "${RUN_CONFIG}" "${MODE}" "${POLICY_DIR}" "${BANK_ROOT}" \
  "${SEED}" "${EPISODES_PER_TASK}" "${INFERENCE_BATCH_SIZE}" "${ENV_WORKERS}" \
  "${SAVE_VIDEOS}" "${RESUME}" "${MEMORY_TOP_K}" "${POSITIVE_CAPACITY_PER_TASK}" \
  "${POSITIVE_SELECTION_PATH}" "${ACTIVE_POSITIVE_ITEMS}" "${INFERENCE_BATCH_WAIT_MS}" \
  "${MAX_TASK_RETRIES}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = {
    "protocol": "pi_v1_bcpi_nfailure_four_suites_dynamic_v2",
    "mode": sys.argv[2],
    "positive_memory": (
        "B6500 + all C-pi successes"
        if sys.argv[12] == "max"
        else f"B6500 deterministic first {sys.argv[12]} trajectories per canonical task"
    ),
    "negative_memory": "all C-pi failures + all N failures" if sys.argv[2] == "both" else "disabled",
    "positive_source_items": 11834,
    "active_positive_items": int(sys.argv[14]),
    "negative_items_available": 613,
    "policy_dir": str(Path(sys.argv[3]).resolve()),
    "bank_root": str(Path(sys.argv[4]).resolve()),
    "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    "tasks": 40,
    "seed": int(sys.argv[5]),
    "episodes_per_task": int(sys.argv[6]),
    "inference_batch_size": int(sys.argv[7]),
    "inference_batch_wait_ms": int(sys.argv[15]),
    "environment_workers": int(sys.argv[8]),
    "max_task_retries": int(sys.argv[16]),
    "torch_compile": False,
    "websocket_keepalive_disabled": True,
    "dynamic_task_assignment": True,
    "positive_top_k": int(sys.argv[11]),
    "positive_capacity_per_task": sys.argv[12] if sys.argv[12] == "max" else int(sys.argv[12]),
    "positive_selection": str(Path(sys.argv[13]).resolve()) if sys.argv[2] == "both" else "",
    "negative_top_k": 8,
    "guidance": "V1 capped additive",
    "save_videos": sys.argv[9] == "1",
}
resume = sys.argv[10] == "1"
if path.exists():
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit(f"Resume configuration mismatch:\nexpected={expected}\nactual={actual}")
elif resume:
    raise SystemExit(f"Resume requested without run config: {path}")
else:
    path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
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

if ! "${OPENPI_PYTHON}" - "${PORT}" <<'PY'
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
  "${OPENPI_PYTHON}" scripts/serve_policy.py
  --env LIBERO
  --port "${PORT}"
  --seed "${SEED}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"
  --inference-batch-wait-ms "${INFERENCE_BATCH_WAIT_MS}"
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
)
if [[ "${USE_NEGATIVE}" == "1" ]]; then
  server_cmd+=(
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
  )
fi
server_cmd+=(policy:checkpoint --policy.config pi05_libero --policy.dir "${POLICY_DIR}")

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

"${OPENPI_PYTHON}" scripts/eval/wait_for_websocket.py \
  --host 127.0.0.1 --port "${PORT}" --timeout 900 --pid "${server_pid}"

scheduler_args=(
  --openpi-root "${OPENPI_ROOT}"
  --libero-python "${LIBERO_PYTHON}"
  --run-root "${RUN_ROOT}"
  --host 127.0.0.1
  --port "${PORT}"
  --num-workers "${ENV_WORKERS}"
  --episodes-per-task "${EPISODES_PER_TASK}"
  --max-task-retries "${MAX_TASK_RETRIES}"
  --seed "${SEED}"
)
if [[ "${SAVE_VIDEOS}" == "1" ]]; then scheduler_args+=(--save-videos); else scheduler_args+=(--no-save-videos); fi
if [[ "${RESUME}" == "1" ]]; then scheduler_args+=(--resume); fi

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
  "${OPENPI_PYTHON}" -m optimus_eval.libero_dynamic_task_scheduler \
    "${scheduler_args[@]}" >"${RUN_ROOT}/scheduler.log" 2>&1 &
scheduler_pid="$!"
wait "${scheduler_pid}"
scheduler_pid=""

echo "Pi V1 B+C-pi ${MODE} four-suite evaluation complete: ${RUN_ROOT}/eval/results.txt"
