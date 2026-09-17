#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
MEMORY_META_PATH="${MEMORY_META_PATH:-${OPENPI_ROOT}/memory/gpm_memory_meta.pt}"
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${OPENPI_ROOT}/memory/gpm_memory.index}"
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-${OPENPI_ROOT}/memory/gpm_memory_actions.npz}"
LCM_CKPT="${LCM_CKPT:-${OPENPI_ROOT}/checkpoints/lcm.pt}"
RUN_ROOT="${RUN_ROOT:-${OPENPI_ROOT}/logs/pi_prior_four_suites_seed7_$(date -u +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SEED="${SEED:-7}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
RESUME="${RESUME:-0}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

for path in "${POLICY_DIR}/model.safetensors" "${TASK_HEAD_CKPT}" "${MEMORY_META_PATH}" \
  "${FAISS_INDEX_PATH}" "${MEMORY_ACTIONS_PATH}" "${LCM_CKPT}"; do
  [[ -f "${path}" ]] || { echo "Missing prior artifact: ${path}" >&2; exit 1; }
done

CUDA_VISIBLE_DEVICES="${GPU}" \
SERVER_CUDA_VISIBLE_DEVICES="${GPU}" \
CLIENT_CUDA_VISIBLE_DEVICES="${GPU}" \
MUJOCO_EGL_DEVICE_ID=0 \
LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" \
MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
PORT="${PORT}" \
POLICY_DIR="${POLICY_DIR}" \
TASK_HEAD_CKPT="${TASK_HEAD_CKPT}" \
MEMORY_META_PATH="${MEMORY_META_PATH}" \
FAISS_INDEX_PATH="${FAISS_INDEX_PATH}" \
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH}" \
LCM_CKPT="${LCM_CKPT}" \
LOG_DIR="${RUN_ROOT}/eval" \
RESULTS_TXT="${RUN_ROOT}/eval/results.txt" \
VIDEO_ROOT="${RUN_ROOT}/eval/videos" \
EPISODE_DATA_ROOT="" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
SEED="${SEED}" \
SAVE_VIDEOS="${SAVE_VIDEOS}" \
RESUME="${RESUME}" \
INFERENCE_BATCH_SIZE=8 \
INFERENCE_BATCH_WAIT_MS=20 \
LIBERO_CLIENTS_PER_SUITE=8 \
USE_MEMORY=1 \
MEMORY_TOP_K=8 \
USE_LCM=1 \
MEMORY_GUIDANCE_ONLY=0 \
MEMORY_PRIOR_SUBSTEP_GUIDANCE=0 \
USE_NEGATIVE_GUIDANCE=0 \
bash scripts/eval/run_libero_eval.sh "${SUITES[@]}"

echo "Pi prior four-suite evaluation complete: ${RUN_ROOT}"
