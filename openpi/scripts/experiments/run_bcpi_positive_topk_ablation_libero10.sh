#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
SOURCE_ROOT="${C_PI_SOURCE_ROOT:-/path/to/local/CVPR26-OptimusVLA/openpi}"
LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-${OPENPI_ROOT}/third_party/libero}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
BCPI_ARTIFACT_ROOT="${BCPI_ARTIFACT_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_nfailure_four_suites_v1}"
SOURCE_BANK="${BCPI_ARTIFACT_ROOT}/bank_b_plus_cpi_success_cpi_n_failure"
ABLATION_ROOT="${ABLATION_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_positive_topk_ablation_v3_batch16_env32}"
GPU_POOL_CSV="${GPU_POOL_CSV:-0}"
BASE_PORT="${BASE_PORT:-8200}"
PREP_GPU="${PREP_GPU:-${GPU_POOL_CSV%%,*}}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

cd "${OPENPI_ROOT}"
[[ -f "${LIBERO_SOURCE_ROOT}/libero/libero/__init__.py" ]] || {
  echo "Invalid LIBERO_SOURCE_ROOT: ${LIBERO_SOURCE_ROOT}" >&2
  exit 1
}
export LIBERO_SOURCE_ROOT
PYTHONPATH_VALUE="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${LIBERO_SOURCE_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"; fi
export PYTHONPATH="${PYTHONPATH_VALUE}"

if [[ -f "${SOURCE_BANK}/build_summary.json" && -f "${SOURCE_BANK}/bank_identity.json" ]]; then
  echo "Using verified B+C-pi+N bank: ${SOURCE_BANK}"
else
  C_PI_SOURCE_ROOT="${SOURCE_ROOT}" \
  TASK_HEAD_CKPT="${TASK_HEAD_CKPT}" \
  BCPI_ARTIFACT_ROOT="${BCPI_ARTIFACT_ROOT}" \
  PREFLIGHT_ONLY="${PREFLIGHT_ONLY}" \
  CUDA_VISIBLE_DEVICES="${PREP_GPU}" \
    bash scripts/experiments/prepare_pi_v1_bcpi_four_suites_bank.sh
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  "${PYTHON}" - <<'PY'
from scripts.experiments.run_bcpi_positive_topk_ablation import CANDIDATES
expected = ((1, 8), (10, 1), (10, 8), (50, 8), (50, 16), (50, 32), ("max", 8), ("max", 16), ("max", 32))
if CANDIDATES != expected:
    raise SystemExit(f"Unexpected B+C-pi positive candidates: {CANDIDATES}")
print(f"B+C-pi positive ablation preflight passed: candidates={len(CANDIDATES)} suite=libero_10")
PY
  "${PYTHON}" -m optimus_eval.libero_dynamic_episode_scheduler \
    --openpi-root "${OPENPI_ROOT}" \
    --libero-python "${LIBERO_PYTHON}" \
    --run-root "${TMPDIR:-/tmp}/bcpi_positive_topk_scheduler_preflight" \
    --port "${BASE_PORT}" \
    --num-workers 32 \
    --shards-per-task 4 \
    --episodes-per-task 50 \
    --episode-start 100 \
    --preflight-only
  exit 0
fi

"${LIBERO_PYTHON}" scripts/eval/patch_robosuite_egl.py

"${PYTHON}" scripts/experiments/prepare_bcpi_positive_topk_ablation.py \
  --source-bank "${SOURCE_BANK}" \
  --output-root "${ABLATION_ROOT}"

"${PYTHON}" scripts/experiments/run_bcpi_positive_topk_ablation.py \
  --openpi-root "${OPENPI_ROOT}" \
  --artifact-root "${ABLATION_ROOT}" \
  --policy-dir "${POLICY_DIR}" \
  --task-head "${TASK_HEAD_CKPT}" \
  --python "${PYTHON}" \
  --gpu-pool "${GPU_POOL_CSV}" \
  --base-port "${BASE_PORT}"

echo "Ablation complete. Audit before final evaluation: ${ABLATION_ROOT}/selection/positive_selection.json"
