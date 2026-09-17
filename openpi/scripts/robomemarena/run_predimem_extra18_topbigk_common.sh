#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: $0 counting_v1_1|occlusion_v1_1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FULL26_ROOT="${FULL26_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32}"
CAMPAIGN_BASE="${CAMPAIGN_BASE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/extra18_topbigk_20260815}"
TOPK_VALUES="${TOPK_VALUES:-8,16,32,50,100}"
EXTRA18_TASKS="4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,23,24"
VLA_CKPT="${VLA_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data}"

case "${SUITE}" in
  counting_v1_1)
    tasks="6,7,8,9,10,15,16"
    gate_path="${ROOT}/configs/robomemarena/arena_counting_v1_1_gate.json"
    campaign="counting_fusion_v1_1_ep26_seed50"
    ;;
  occlusion_v1_1)
    tasks="4,5,11,12,13,14,17,20,21,23,24"
    gate_path="${ROOT}/configs/robomemarena/arena_occlusion_v1_1_gate.json"
    campaign="occlusion_fusion_v1_1_ep26_seed50"
    ;;
  *)
    echo "Unknown Extra-18 top-big-k suite: ${SUITE}" >&2
    exit 2
    ;;
esac

[[ -f "${gate_path}" ]] || { echo "Missing V1.1 gate: ${gate_path}" >&2; exit 2; }
IFS=',' read -r -a topk_array <<<"${TOPK_VALUES}"
[[ "${#topk_array[@]}" -gt 0 ]] || { echo "TOPK_VALUES must not be empty" >&2; exit 2; }
for topk in "${topk_array[@]}"; do
  [[ "${topk}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid positive top-k: ${topk}" >&2; exit 2; }
done

campaign_root="${CAMPAIGN_ROOT:-${CAMPAIGN_BASE}/${campaign}}"
mkdir -p "${campaign_root}"
printf '%s\n' \
  "suite=${SUITE}" \
  "mode=v1" \
  "head=fusion" \
  "evaluation_tasks=${tasks}" \
  "memory_scope_tasks=${EXTRA18_TASKS}" \
  "episodes_per_task=26" \
  "seed_start=50" \
  "topk_values=${TOPK_VALUES}" \
  "full26_self_root=${FULL26_ROOT}" \
  "memory_recording=disabled" >"${campaign_root}/protocol.txt"

for topk in "${topk_array[@]}"; do
  run_root="${campaign_root}/topk_${topk}"
  resume=0
  [[ -f "${run_root}/run_config.json" ]] && resume=1
  if [[ -f "${run_root}/fusion/results.txt" ]]; then
    echo "[skip] completed ${SUITE} topk=${topk}: ${run_root}"
    continue
  fi
  echo "[extra18-topbigk] suite=${SUITE} topk=${topk} run_root=${run_root} resume=${resume}"
  PREDIMEM_MODE=v1 \
  AOSS_ROOT="${FULL26_ROOT}" SELF_ROOT="${FULL26_ROOT}" \
  VLA_CKPT="${VLA_CKPT}" VLA_CONFIG=pi05_robomemarena_all26_reactive \
  ACTION_STATS="${VLA_CKPT}/assets/robomemarena/all26_pi05_reactive/norm_stats.json" \
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
  HEAD_VARIANTS=fusion TASK_IDS="${tasks}" \
  MEMORY_ALLOWED_TASK_IDS="${EXTRA18_TASKS}" MEMORY_ADMISSION=success \
  MEMORY_TOP_K="${topk}" NEGATIVE_MEMORY_TOP_K=8 \
  MEMORY_GUIDANCE_SUITE_GATE_PATH="${gate_path}" \
  EPISODES_PER_TASK=26 SEED=50 \
  UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" \
  UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
  ENV_WORKERS="${ENV_WORKERS:-64}" \
  SAVE_VIDEO="${SAVE_VIDEO:-1}" RECORD_MEMORY_DATA=0 MEMORY_TRACE_LEVEL=light \
  RUN_ROOT="${run_root}" RESUME="${resume}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
    bash "${ROOT}/scripts/robomemarena/run_predimem_v1_tdense5_common_dual_gpu.sh"
done

echo "CAMPAIGN_ROOT=${campaign_root}"
