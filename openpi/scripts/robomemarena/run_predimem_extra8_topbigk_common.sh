#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: $0 sequence_v0|transferring_v1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTRA8_ROOT="${EXTRA8_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}"
CAMPAIGN_BASE="${CAMPAIGN_BASE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/extra8_topbigk_20260814}"
TOPK_VALUES="${TOPK_VALUES:-8,16,32,50,100}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-26}"
SEED="${SEED:-50}"

[[ "${EPISODES_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || {
  echo "EPISODES_PER_TASK must be a positive integer" >&2; exit 2;
}
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 2; }

case "${SUITE}" in
  sequence_v0)
    mode=v0
    head=upper
    tasks=1,2,3,22
    campaign="sequence_upper_v0_ep${EPISODES_PER_TASK}_seed${SEED}"
    ;;
  transferring_v1)
    mode=v1
    head=fusion
    tasks=18,19,25,26
    campaign="transferring_fusion_v1_ep${EPISODES_PER_TASK}_seed${SEED}"
    ;;
  *)
    echo "Unknown top-big-k suite: ${SUITE}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a topk_array <<<"${TOPK_VALUES}"
[[ "${#topk_array[@]}" -gt 0 ]] || { echo "TOPK_VALUES must not be empty" >&2; exit 2; }
for topk in "${topk_array[@]}"; do
  [[ "${topk}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid positive top-k: ${topk}" >&2; exit 2; }
done

campaign_root="${CAMPAIGN_ROOT:-${CAMPAIGN_BASE}/${campaign}}"
mkdir -p "${campaign_root}"
printf '%s\n' \
  "suite=${SUITE}" \
  "mode=${mode}" \
  "head=${head}" \
  "tasks=${tasks}" \
  "episodes_per_task=${EPISODES_PER_TASK}" \
  "seed_start=${SEED}" \
  "seed_end=$((SEED + EPISODES_PER_TASK - 1))" \
  "topk_values=${TOPK_VALUES}" \
  "extra8_root=${EXTRA8_ROOT}" >"${campaign_root}/protocol.txt"

for topk in "${topk_array[@]}"; do
  run_root="${campaign_root}/topk_${topk}"
  resume=0
  [[ -f "${run_root}/run_config.json" ]] && resume=1
  if [[ -f "${run_root}/${head}/results.txt" ]]; then
    echo "[skip] completed ${SUITE} topk=${topk}: ${run_root}"
    continue
  fi
  echo "[topbigk] suite=${SUITE} topk=${topk} run_root=${run_root} resume=${resume}"
  PREDIMEM_MODE="${mode}" \
  AOSS_ROOT="${EXTRA8_ROOT}" SELF_ROOT="${EXTRA8_ROOT}" \
  HEAD_VARIANTS="${head}" TASK_IDS="${tasks}" \
  MEMORY_ALLOWED_TASK_IDS="${tasks}" MEMORY_ADMISSION=success \
  MEMORY_TOP_K="${topk}" NEGATIVE_MEMORY_TOP_K=8 \
  EPISODES_PER_TASK="${EPISODES_PER_TASK}" SEED="${SEED}" \
  UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" \
  UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
  ENV_WORKERS="${ENV_WORKERS:-64}" \
  SAVE_VIDEO="${SAVE_VIDEO:-1}" RECORD_MEMORY_DATA=1 MEMORY_TRACE_LEVEL=light \
  RUN_ROOT="${run_root}" RESUME="${resume}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
    bash "${ROOT}/scripts/robomemarena/run_predimem_v1_tdense5_common_dual_gpu.sh"
done

echo "CAMPAIGN_ROOT=${campaign_root}"
