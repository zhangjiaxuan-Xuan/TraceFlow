#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

PLUS_ROOT="${LIBERO_PLUS_ROOT:-${OPENPI_ROOT}/third_party/LIBERO-plus}"
[[ -f "${PLUS_ROOT}/libero/libero/benchmark/task_classification.json" ]] || {
  echo "Missing LIBERO-plus checkout: ${PLUS_ROOT}" >&2
  exit 2
}
[[ -d "${PLUS_ROOT}/libero/libero/assets" ]] || {
  echo "Missing LIBERO-plus assets at ${PLUS_ROOT}/libero/libero/assets. Extract assets.zip before evaluation." >&2
  exit 2
}

OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OPENPI_ROOT}/logs/libero_plus10_v1_bcpi_lowcost_${RUN_ID}}"
TASK_SELECTION="${TASK_SELECTION:-${LOG_DIR}/task_selection.json}"
TASKS_PER_CATEGORY="${TASKS_PER_CATEGORY:-10}"
EVAL_SCOPE="${EVAL_SCOPE:-stratified}"
SEED="${SEED:-7}"
MODE="${MODE:-guidance}"
GPU="${GPU:-0}"

mkdir -p "${LOG_DIR}"
selector_args=(
  --plus-root "${PLUS_ROOT}"
  --suite libero_10
  --seed "${SEED}"
  --output "${TASK_SELECTION}"
)
if [[ -n "${PLUS_TASK_IDS_CSV:-}" ]]; then
  selector_args+=(--task-ids-csv "${PLUS_TASK_IDS_CSV}")
else
  case "${EVAL_SCOPE}" in
    full) selector_args+=(--all-tasks) ;;
    stratified) selector_args+=(--per-category "${TASKS_PER_CATEGORY}") ;;
    *) echo "EVAL_SCOPE must be full or stratified, got ${EVAL_SCOPE}" >&2; exit 2 ;;
  esac
fi
TASK_IDS_CSV="$(${OPENPI_PYTHON} scripts/eval/select_libero_plus_tasks.py "${selector_args[@]}")"

case "${MODE}" in
  guidance)
    USE_MEMORY=1
    MEMORY_GUIDANCE_ONLY=1
    USE_MEMORY_GUIDANCE=0
    USE_NEGATIVE_GUIDANCE=1
    ;;
  base)
    USE_MEMORY=0
    MEMORY_GUIDANCE_ONLY=0
    USE_MEMORY_GUIDANCE=0
    USE_NEGATIVE_GUIDANCE=0
    ;;
  *) echo "MODE must be guidance or base, got ${MODE}" >&2; exit 2 ;;
esac
export USE_MEMORY MEMORY_GUIDANCE_ONLY USE_MEMORY_GUIDANCE USE_NEGATIVE_GUIDANCE

POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
ACTION_NORM_STATS_PATH="${ACTION_NORM_STATS_PATH:-${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json}"
BANK_ROOT="${BANK_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_nfailure_four_suites_v1/bank_b_plus_cpi_success_cpi_n_failure}"
POSITIVE_BANK="${POSITIVE_BANK:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_positive_topk_ablation_v3_batch16_env32/selected_four_suites_s50k16_canonical_v1/capacity_50/positive}"

export OPENPI_ROOT LIBERO_SOURCE_ROOT="${PLUS_ROOT}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LOG_DIR}/libero_config}"
export POLICY_DIR TASK_HEAD_CKPT ACTION_NORM_STATS_PATH
export MEMORY_META_PATH="${POSITIVE_BANK}/gpm_memory_meta.pt"
export FAISS_INDEX_PATH="${POSITIVE_BANK}/gpm_memory.index"
export MEMORY_ACTIONS_PATH="${POSITIVE_BANK}/gpm_memory_actions.npz"
export NEGATIVE_MEMORY_META_PATH="${BANK_ROOT}/negative/gpm_negative_memory_meta.pt"
export NEGATIVE_FAISS_INDEX_PATH="${BANK_ROOT}/negative/gpm_negative_memory.index"
export NEGATIVE_MEMORY_ACTIONS_PATH="${BANK_ROOT}/negative/gpm_negative_memory_actions.npz"
export MEMORY_TOP_K=16
export NEGATIVE_MEMORY_TOP_K=8
export MEMORY_GUIDANCE_TIME_VERSION=v1
export MEMORY_GUIDANCE_NORM_CAP=0.20
export MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20
export NEGATIVE_MEMORY_MIN_SIMILARITY=-1.0
export NEGATIVE_MEMORY_MIN_CONFIDENCE=0.0
export NEGATIVE_GUIDANCE_BETA=0.10
export NEGATIVE_GUIDANCE_NORM_CAP=0.10
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
export MAX_ENV_STEPS="${MAX_ENV_STEPS:-600}"
export NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
export INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-2000}"
export INFERENCE_BATCH_GROUP_SIZE="${INFERENCE_BATCH_GROUP_SIZE:-${INFERENCE_BATCH_SIZE}}"
export LIBERO_CLIENTS_PER_SUITE="${LIBERO_CLIENTS_PER_SUITE:-16}"
export LIBERO_SHARD_AXIS="${LIBERO_SHARD_AXIS:-tasks}"
export SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
export TASK_IDS_CSV
export LOG_DIR
export RESUME="${RESUME:-0}"
export SERVER_CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-${GPU}}"
export CLIENT_CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES:-${GPU}}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

for path in "${POLICY_DIR}/model.safetensors" "${ACTION_NORM_STATS_PATH}"; do
  [[ -f "${path}" ]] || { echo "Missing required model file: ${path}" >&2; exit 2; }
done
if [[ "${MODE}" == guidance ]]; then
  for path in "${TASK_HEAD_CKPT}" "${MEMORY_META_PATH}" "${FAISS_INDEX_PATH}" \
    "${MEMORY_ACTIONS_PATH}" "${NEGATIVE_MEMORY_META_PATH}" \
    "${NEGATIVE_FAISS_INDEX_PATH}" "${NEGATIVE_MEMORY_ACTIONS_PATH}"; do
    [[ -f "${path}" ]] || { echo "Missing guidance artifact: ${path}" >&2; exit 2; }
  done
  "${OPENPI_PYTHON}" - "${MEMORY_META_PATH}" "${NEGATIVE_MEMORY_META_PATH}" "${MEMORY_BANK_SCOPE:-four_suites}" <<'PY'
from collections import Counter
from pathlib import Path
import sys
import torch

positive = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
negative = torch.load(Path(sys.argv[2]), map_location="cpu", weights_only=False)
scope = sys.argv[3]
positive_tasks = Counter((str(x.get("suite")), str(x.get("task_name"))) for x in positive)
positive_suites = Counter(str(x.get("suite")) for x in positive)
negative_sources = Counter((x.get("provenance") or {}).get("source_family", "unknown") for x in negative)
def suite_of(item):
    provenance = item.get("provenance") or {}
    return str(item.get("suite") or provenance.get("suite") or provenance.get("task_suite") or "")
if scope == "libero10":
    if len(positive) != 500 or len(positive_tasks) != 10 or set(positive_tasks.values()) != {50}:
        raise SystemExit(f"Invalid LIBERO-10 B50 positive bank: items={len(positive)} tasks={len(positive_tasks)}")
    if positive_suites != Counter({"libero_10": 500}):
        raise SystemExit(f"Invalid LIBERO-10 positive suite inventory: {positive_suites}")
    negative_suites = Counter(suite_of(x) for x in negative)
    if set(negative_suites) != {"libero_10"} or not negative:
        raise SystemExit(f"Invalid LIBERO-10 negative bank suite inventory: {negative_suites}")
    print(f"Memory contract passed: LIBERO-10-only positive=B50/task ({len(positive)}), k+=16; negative={len(negative)}, k-=8")
else:
    if len(positive) != 2000 or len(positive_tasks) != 40 or set(positive_tasks.values()) != {50}:
        raise SystemExit(f"Invalid B50 positive bank: items={len(positive)} tasks={len(positive_tasks)}")
    if positive_suites != Counter({"libero_spatial": 500, "libero_object": 500, "libero_goal": 500, "libero_10": 500}):
        raise SystemExit(f"Invalid B50 suite inventory: {positive_suites}")
    if len(negative) != 613 or negative_sources["new_pi"] != 497:
        raise SystemExit(f"Invalid N+C_pi negative bank: items={len(negative)} sources={negative_sources}")
    print("Memory contract passed: positive=B-only 50/task (2000), k+=16; negative=N497+C_pi116 (613), k-=8")
PY
fi

echo "LIBERO-plus libero_10 low-cost evaluation"
TASK_COUNT="$(${OPENPI_PYTHON} - "${TASK_SELECTION}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["task_count"])
PY
)"
if [[ -z "${PLUS_TASK_IDS_CSV:-}" && "${EVAL_SCOPE}" == "full" && "${TASK_COUNT}" -ne 2519 ]]; then
  echo "Full LIBERO-10-plus evaluation requires exactly 2519 tasks, got ${TASK_COUNT}." >&2
  exit 2
fi
echo "mode=${MODE} eval_scope=${EVAL_SCOPE} tasks=${TASK_COUNT} seed=${SEED} max_steps=${MAX_ENV_STEPS}"
echo "policy_batch=${INFERENCE_BATCH_SIZE} batch_wait_ms=${INFERENCE_BATCH_WAIT_MS} batch_group_size=${INFERENCE_BATCH_GROUP_SIZE} env_clients=${LIBERO_CLIENTS_PER_SUITE} shard_axis=${LIBERO_SHARD_AXIS}"
echo "task_selection=${TASK_SELECTION} log_dir=${LOG_DIR}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight-only validation passed."
  exit 0
fi

if [[ "${MODE}" == base ]]; then
  unset MEMORY_META_PATH FAISS_INDEX_PATH MEMORY_ACTIONS_PATH \
    NEGATIVE_MEMORY_META_PATH NEGATIVE_FAISS_INDEX_PATH NEGATIVE_MEMORY_ACTIONS_PATH
fi

"${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}" scripts/eval/patch_robosuite_egl.py

exec bash scripts/eval/run_libero_eval.sh libero_10
