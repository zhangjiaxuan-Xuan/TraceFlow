#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${OPENPI_ROOT}/.." >/dev/null 2>&1 && pwd)"

POLICY_CONSUMER="${POLICY_CONSUMER:?POLICY_CONSUMER is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
HEAD_PATH="${HEAD_PATH:?HEAD_PATH is required}"
POSITIVE_META="${POSITIVE_META:?POSITIVE_META is required}"
POSITIVE_INDEX="${POSITIVE_INDEX:?POSITIVE_INDEX is required}"
POSITIVE_ACTIONS="${POSITIVE_ACTIONS:?POSITIVE_ACTIONS is required}"
NEGATIVE_META="${NEGATIVE_META:?NEGATIVE_META is required}"
NEGATIVE_INDEX="${NEGATIVE_INDEX:?NEGATIVE_INDEX is required}"
NEGATIVE_ACTIONS="${NEGATIVE_ACTIONS:?NEGATIVE_ACTIONS is required}"
POSITIVE_TOP_K="${POSITIVE_TOP_K:?POSITIVE_TOP_K is required}"
NEGATIVE_TOP_K="${NEGATIVE_TOP_K:?NEGATIVE_TOP_K is required}"
GPU="${GPU:-0}"
PORT="${PORT:-8200}"
RESUME="${RESUME:-1}"

cd "${OPENPI_ROOT}"

[[ "${POSITIVE_TOP_K}" =~ ^[1-9][0-9]*$ ]] || { echo "positive top-k must be positive" >&2; exit 2; }
[[ "${NEGATIVE_TOP_K}" =~ ^[1-9][0-9]*$ ]] || { echo "negative top-k must be positive" >&2; exit 2; }
for path in "${HEAD_PATH}" "${POSITIVE_META}" "${POSITIVE_INDEX}" "${POSITIVE_ACTIONS}" \
  "${NEGATIVE_META}" "${NEGATIVE_INDEX}" "${NEGATIVE_ACTIONS}"; do
  [[ -f "${path}" ]] || { echo "Missing fixed match artifact: ${path}" >&2; exit 1; }
done

case "${POLICY_CONSUMER}" in
  pi)
    mkdir -p "${RUN_ROOT}/eval" "${RUN_ROOT}/indexes"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    SERVER_CUDA_VISIBLE_DEVICES="${GPU}" \
    CLIENT_CUDA_VISIBLE_DEVICES="${GPU}" \
    MUJOCO_EGL_DEVICE_ID=0 \
    LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
    NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" \
    MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
    PORT="${PORT}" \
    OPENPI_PYTHON="${POLICY_PYTHON}" \
    POLICY_DIR="${CHECKPOINT_DIR}" \
    TASK_HEAD_CKPT="${HEAD_PATH}" \
    MEMORY_META_PATH="${POSITIVE_META}" \
    FAISS_INDEX_PATH="${POSITIVE_INDEX}" \
    MEMORY_ACTIONS_PATH="${POSITIVE_ACTIONS}" \
    NEGATIVE_MEMORY_META_PATH="${NEGATIVE_META}" \
    NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_INDEX}" \
    NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_ACTIONS}" \
    LOG_DIR="${RUN_ROOT}/eval" \
    RESULTS_TXT="${RUN_ROOT}/eval/results.txt" \
    VIDEO_ROOT="${RUN_ROOT}/videos" \
    EPISODE_DATA_ROOT="" \
    TASK_IDS_CSV="0,1,2,3,4,5,6,7,8,9" \
    NUM_TRIALS_PER_TASK=100 EPISODE_START=0 SEED=7 SAVE_VIDEOS=0 RESUME="${RESUME}" \
    INFERENCE_BATCH_SIZE=8 INFERENCE_BATCH_WAIT_MS=20 LIBERO_CLIENTS_PER_SUITE=8 \
    USE_MEMORY=1 MEMORY_REFRESH_EVERY=1 MEMORY_TOP_K="${POSITIVE_TOP_K}" \
    USE_LCM=0 USE_MEMORY_GUIDANCE=0 MEMORY_GUIDANCE_ONLY=1 MEMORY_GUIDANCE_TIME_VERSION=v0 \
    MEMORY_PRIOR_SUBSTEP_GUIDANCE=0 MEMORY_GUIDANCE_NUM_STEPS=10 \
    MEMORY_GUIDANCE_LAMBDA_MAX=0.20 MEMORY_GUIDANCE_T_CUT=0.30 MEMORY_GUIDANCE_SIGMA=0.30 \
    MEMORY_GUIDANCE_NORM_CAP=0.20 MEMORY_GUIDANCE_MIN_SIMILARITY=-1.0 \
    MEMORY_GUIDANCE_TRACE_DIR="${RUN_ROOT}/traces" MEMORY_GUIDANCE_TRACE_LEVEL=light \
    USE_NEGATIVE_GUIDANCE=1 NEGATIVE_MEMORY_MIN_SIMILARITY=-1.0 NEGATIVE_MEMORY_MIN_CONFIDENCE=0.0 \
    NEGATIVE_MEMORY_TOP_K="${NEGATIVE_TOP_K}" NEGATIVE_GUIDANCE_BETA=0.10 \
    NEGATIVE_GUIDANCE_SIGMA=0.30 NEGATIVE_GUIDANCE_NORM_CAP=0.10 MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20 \
    bash "${OPENPI_ROOT}/scripts/eval/run_libero_eval.sh" libero_10
    ;;
  smol)
    SMOL_ROOT="${REPO_ROOT}/smolvla"
    SMOL_PYTHON="${POLICY_PYTHON:-/path/to/envs/.venv/bin/python}"
    smol_resume_flag="--no-resume"
    [[ "${RESUME}" == "1" ]] && smol_resume_flag="--resume"
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
    PYTHONPATH="${SMOL_ROOT}/src:${OPENPI_ROOT}/src:${OPENPI_ROOT}/third_party/libero:${PYTHONPATH:-}" \
    "${SMOL_PYTHON}" "${SMOL_ROOT}/scripts/eval_libero10.py" \
      --checkpoint "${CHECKPOINT_DIR}" \
      --bank "${BANK_ROOT:?BANK_ROOT is required for Smol}" \
      --head "${HEAD_PATH}" \
      --version v0 \
      --output-dir "${RUN_ROOT}" \
      --seed 7 --episode-start 0 --episodes-per-task 100 --batch-size 8 \
      --positive-top-k "${POSITIVE_TOP_K}" --negative-top-k "${NEGATIVE_TOP_K}" \
      --negative-guidance --no-save-videos --no-save-trajectories "${smol_resume_flag}"
    ;;
  *) echo "POLICY_CONSUMER must be pi or smol" >&2; exit 2 ;;
esac
