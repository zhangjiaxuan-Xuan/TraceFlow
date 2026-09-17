#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:?Usage: $0 success|failure|both}"
case "${GROUP}" in success|failure|both) ;; *) echo "GROUP must be success, failure, or both" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
AOSS_ROOT="${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}"
PREDIMEM_RUNTIME_ROOT="${PREDIMEM_RUNTIME_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}"
VLA_CKPT="${VLA_CKPT:-${PREDIMEM_RUNTIME_ROOT}/checkpoints/vla_alltask_pytorch}"
VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_extra8_reactive}"
ACTION_STATS="${ACTION_STATS:-${VLA_CKPT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${PREDIMEM_RUNTIME_ROOT}/runtime/openpi_data}"
CAMPAIGN_BASE="${CAMPAIGN_BASE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_inherited_self_cl_v1_seed7}"
CL_ROOT="${CAMPAIGN_BASE}/k8n8"
RATIO_ROOT="${RATIO_ROOT:-${CL_ROOT}/delayed_asymmetric_topk/${GROUP}}"
RATIO_BANK_ROOT="${RATIO_BANK_ROOT:-${CL_ROOT}/delayed_asymmetric_topk/shared_banks/${GROUP}}"
FIXED_HEAD="${FIXED_HEAD:-${AOSS_ROOT}/heads/fusion/best.pt}"
FIXED_META="${FIXED_META:-${AOSS_ROOT}/memory/fusion/gpm_memory_meta_dense_frame_v3.pt}"
FIXED_INDEX="${FIXED_INDEX:-${AOSS_ROOT}/memory/fusion/gpm_memory.index}"
FIXED_ACTIONS="${FIXED_ACTIONS:-${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz}"
START_ROUND="${START_ROUND:-1}"
END_ROUND="${END_ROUND:-10}"
BANK_WORKERS="${BANK_WORKERS:-32}"
TASK_IDS="${TASK_IDS:-18,19,25,26}"
MEMORY_ALLOWED_TASK_IDS="${MEMORY_ALLOWED_TASK_IDS:-18,19,25,26}"
RATIO_CANDIDATES="${RATIO_CANDIDATES:-p16_n8,p8_n16}"

[[ "${START_ROUND}" =~ ^[0-9]+$ && "${END_ROUND}" =~ ^[0-9]+$ ]] || exit 2
(( START_ROUND >= 1 && END_ROUND >= START_ROUND && END_ROUND <= 10 )) || {
  echo "Require 1 <= START_ROUND <= END_ROUND <= 10" >&2; exit 2;
}

for path in "${OPENPI_PY}" "${FIXED_HEAD}" "${FIXED_META}" "${FIXED_INDEX}" "${FIXED_ACTIONS}" \
  "${VLA_CKPT}/model.safetensors" "${ACTION_STATS}" \
  "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done

collector="${COLLECTOR_ROOT:-${CL_ROOT}/collector/fusion}"
[[ -f "${collector}/memory_records/index.jsonl" ]] || {
  echo "Missing completed collector: ${collector}" >&2; exit 2;
}

run_eval() {
  local bank_label="$1" name="$2" positive_k="$3" negative_k="$4"
  local positive_meta="$5" positive_index="$6" positive_actions="$7" negative_bank="$8"
  local run_root="${RATIO_ROOT}/${bank_label}/${name}"
  if [[ -f "${run_root}/fusion/results.txt" ]]; then
    echo "[skip] completed ratio evaluation: ${run_root}"
    return 0
  fi
  local resume=0
  [[ -f "${run_root}/run_config.json" ]] && resume=1
  printf '[ratio] group=%s bank=%s k+=%s k-=%s resume=%s record_memory_data=0 run=%s\n' \
    "${GROUP}" "${bank_label}" "${positive_k}" "${negative_k}" "${resume}" "${run_root}"
  env \
    AOSS_ROOT="${AOSS_ROOT}" VLA_CKPT="${VLA_CKPT}" VLA_CONFIG="${VLA_CONFIG}" \
    ACTION_STATS="${ACTION_STATS}" OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
    HEAD_VARIANTS=fusion TASK_IDS="${TASK_IDS}" MEMORY_ALLOWED_TASK_IDS="${MEMORY_ALLOWED_TASK_IDS}" \
    MEMORY_ADMISSION=both MEMORY_TOP_K="${positive_k}" NEGATIVE_MEMORY_TOP_K="${negative_k}" \
    POSITIVE_MEMORY_META_PATH="${positive_meta}" \
    POSITIVE_FAISS_INDEX_PATH="${positive_index}" \
    POSITIVE_MEMORY_ACTIONS_PATH="${positive_actions}" \
    NEGATIVE_MEMORY_META_PATH="${negative_bank}/gpm_memory_meta.pt" \
    NEGATIVE_FAISS_INDEX_PATH="${negative_bank}/gpm_memory.index" \
    NEGATIVE_MEMORY_ACTIONS_PATH="${negative_bank}/gpm_memory_actions.npz" \
    TASK_HEAD_CKPT="${FIXED_HEAD}" MEMORY_ALIGNMENT_TAG=dense-frame-v3-delayed-asymmetric-topk-v1 \
    MEMORY_GUIDANCE_NORM_CAP=0.20 MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20 \
    NEGATIVE_MEMORY_MIN_SIMILARITY="${NEGATIVE_MEMORY_MIN_SIMILARITY:-0.975}" \
    NEGATIVE_MEMORY_MIN_CONFIDENCE="${NEGATIVE_MEMORY_MIN_CONFIDENCE:-0.75}" \
    UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" \
    UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
    ENV_WORKERS="${ENV_WORKERS:-64}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}" \
    SEED="${SEED:-7}" SAVE_VIDEO="${SAVE_VIDEO:-1}" RECORD_MEMORY_DATA=0 \
    RUN_ROOT="${run_root}" RESUME="${resume}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
    bash "${ROOT}/scripts/robomemarena/run_predimem_arena_memory_ablation_dual_gpu.sh" both
}

run_candidates() {
  local bank_label="$1" positive_meta="$2" positive_index="$3" positive_actions="$4"
  local negative_bank="$5"
  IFS=',' read -r -a candidates <<<"${RATIO_CANDIDATES}"
  for candidate in "${candidates[@]}"; do
    case "${candidate}" in
      p8_n8) run_eval "${bank_label}" p8_n8 8 8 "${positive_meta}" "${positive_index}" "${positive_actions}" "${negative_bank}" ;;
      p16_n8) run_eval "${bank_label}" p16_n8 16 8 "${positive_meta}" "${positive_index}" "${positive_actions}" "${negative_bank}" ;;
      p8_n16) run_eval "${bank_label}" p8_n16 8 16 "${positive_meta}" "${positive_index}" "${positive_actions}" "${negative_bank}" ;;
      *) echo "Unknown ratio candidate: ${candidate}" >&2; exit 2 ;;
    esac
  done
}

for ((round=START_ROUND; round<=END_ROUND; round++)); do
  canonical_bank_root="${CL_ROOT}/branches/${GROUP}/banks/input_$(printf 'round_%02d' "${round}")"
  positive_meta="${FIXED_META}"
  positive_index="${FIXED_INDEX}"
  positive_actions="${FIXED_ACTIONS}"
  if [[ "${GROUP}" != "failure" ]]; then
    positive_bank="${canonical_bank_root}/positive_with_fixed_extra8"
    [[ -f "${positive_bank}/provenance.json" ]] || {
      echo "Missing frozen positive bank for round ${round}: ${positive_bank}" >&2; exit 3;
    }
    positive_meta="${positive_bank}/gpm_memory_meta.pt"
    positive_index="${positive_bank}/gpm_memory.index"
    positive_actions="${positive_bank}/gpm_memory_actions.npz"
  fi

  negative_bank="${RATIO_BANK_ROOT}/input_$(printf '%02d' "${round}")/online_failure"
  if [[ ( "${GROUP}" == "both" || "${GROUP}" == "failure" ) \
    && -f "${canonical_bank_root}/online_failure/provenance.json" ]]; then
    negative_bank="${canonical_bank_root}/online_failure"
  elif [[ ! -f "${negative_bank}/provenance.json" ]]; then
    command -v flock >/dev/null || { echo "flock is required for concurrent ratio bank preparation" >&2; exit 2; }
    mkdir -p "$(dirname "${negative_bank}")"
    lock_path="${negative_bank}.lock"
    exec {bank_lock_fd}>"${lock_path}"
    flock "${bank_lock_fd}"
    if [[ ! -f "${negative_bank}/provenance.json" ]]; then
      source_args=(--run-root "${collector}")
      for ((source_round=1; source_round<round; source_round++)); do
        source_args+=(--run-root "${CL_ROOT}/branches/${GROUP}/$(printf 'round_%02d' "${source_round}")/fusion")
      done
      "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_predimem_recorded_bank.py" \
        "${source_args[@]}" --label failure --output "${negative_bank}" \
        --workers "${BANK_WORKERS}" --alignment dense_frame_v3
    fi
    flock -u "${bank_lock_fd}"
    exec {bank_lock_fd}>&-
  fi

  [[ -f "${negative_bank}/provenance.json" ]] || {
    echo "Missing frozen negative bank for round ${round}: ${negative_bank}" >&2; exit 3;
  }
  run_candidates "round_$(printf '%02d' "${round}")" \
    "${positive_meta}" "${positive_index}" "${positive_actions}" "${negative_bank}"
done

echo "Delayed asymmetric top-k sweep complete: ${RATIO_ROOT}"
