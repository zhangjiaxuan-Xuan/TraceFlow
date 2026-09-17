#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
HEAD="${HEAD:-${OPENPI_ROOT}/artifacts/prior_head_reproduction/head_task_file/best.pt}"
B_BANK="${B_BANK:-${OPENPI_ROOT}/artifacts/cl_data_energy/B_reencoded/positive}"
C_PI_MANIFEST="${C_PI_MANIFEST:-${OPENPI_ROOT}/logs/data_collection/C-pi_four_suites_seed7/natural_attempts.jsonl}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${OPENPI_ROOT}/artifacts/pi_topk_v1_seed7_v2}"
DEVICE="${DEVICE:-cuda}"

cd "${OPENPI_ROOT}"
"${LIBERO_PYTHON}" scripts/eval/patch_robosuite_egl.py

"${PYTHON}" scripts/experiments/prepare_pi_topk_banks.py \
  --openpi-root "${OPENPI_ROOT}" \
  --python "${PYTHON}" \
  --policy-dir "${POLICY_DIR}" \
  --head "${HEAD}" \
  --b-bank "${B_BANK}" \
  --c-pi-manifest "${C_PI_MANIFEST}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --device "${DEVICE}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE:-8}" \
  --gpu-pool "${GPU_POOL_CSV:-0,1}" \
  --base-port "${BASE_PORT:-8200}"

"${PYTHON}" scripts/experiments/run_topk_campaign.py \
  --manifest "${ARTIFACT_ROOT}/pi_topk_campaign_manifest.json"
