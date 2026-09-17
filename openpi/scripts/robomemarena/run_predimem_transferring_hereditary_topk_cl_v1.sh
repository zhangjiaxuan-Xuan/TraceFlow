#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
args=()
[[ "${PREFLIGHT_ONLY:-0}" == "1" ]] && args+=(--preflight-only)
cd "${ROOT}"
exec "${PYTHON}" -m optimus_eval.predimem_hereditary_topk_cl \
  --upper-gpu "${UPPER_GPU:-0}" \
  --lower-gpu "${LOWER_GPU:-1}" \
  --upper-batch-size "${UPPER_BATCH_SIZE:-32}" \
  --lower-batch-size "${LOWER_BATCH_SIZE:-32}" \
  --env-workers "${ENV_WORKERS:-64}" \
  --bank-workers "${BANK_WORKERS:-32}" \
  --episodes-per-task "${EPISODES_PER_TASK:-51}" \
  --output-root "${CAMPAIGN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_hereditary_topk_cl_v2_async_bank_seedblocks}" \
  "${args[@]}"
