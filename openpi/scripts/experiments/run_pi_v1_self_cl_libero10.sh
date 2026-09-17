#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/pi_v1_self_cl_libero10_seed7}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
save_video_arg="--no-save-videos"
if [[ "${SAVE_VIDEOS:-0}" == "1" ]]; then save_video_arg="--save-videos"; fi

exec "${PYTHON}" -m optimus_eval.libero_self_cl \
  --openpi-root "${OPENPI_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --policy-dir "${POLICY_DIR}" \
  --task-head-checkpoint "${TASK_HEAD_CKPT}" \
  --python "${PYTHON}" \
  --libero-python "${LIBERO_PYTHON}" \
  --round-script "${OPENPI_ROOT}/scripts/eval/run_pi_v1_self_cl_round.sh" \
  --branch "${BRANCH:-all_branches}" \
  --rounds "${ROUNDS:-10}" \
  --episodes-per-task 50 \
  --task-ids-csv "${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}" \
  --max-env-steps "${MAX_ENV_STEPS:-0}" \
  --seed "${SEED:-7}" \
  --gpu "${GPU:-0}" \
  --port "${PORT:-8200}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE:-16}" \
  --env-workers "${ENV_WORKERS:-32}" \
  --shards-per-task "${SHARDS_PER_TASK:-4}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE:-32}" \
  --trajectory-image-size "${TRAJECTORY_IMAGE_SIZE:-128}" \
  "${save_video_arg}" \
  "${@}"
