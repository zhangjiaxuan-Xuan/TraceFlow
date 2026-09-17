#!/usr/bin/env bash
set -euo pipefail

POSITIVE_TOP_K="${1:?Usage: $0 8|16 collect|success|failure|both}"
GROUP="${2:?Usage: $0 8|16 collect|success|failure|both}"
case "${POSITIVE_TOP_K}" in 8|16) ;; *) echo "positive top-k must be 8 or 16" >&2; exit 2 ;; esac
case "${GROUP}" in collect|success|failure|both) ;; *) echo "invalid group: ${GROUP}" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
AOSS_ROOT="${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}"
PREDIMEM_RUNTIME_ROOT="${PREDIMEM_RUNTIME_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}"
VLA_CKPT="${VLA_CKPT:-${PREDIMEM_RUNTIME_ROOT}/checkpoints/vla_alltask_pytorch}"
VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_extra8_reactive}"
ACTION_STATS="${ACTION_STATS:-${VLA_CKPT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${PREDIMEM_RUNTIME_ROOT}/runtime/openpi_data}"
CAMPAIGN_BASE="${CAMPAIGN_BASE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_inherited_self_cl_v1_seed7}"
CAMPAIGN_ROOT="${CAMPAIGN_BASE}/k${POSITIVE_TOP_K}n8"
FIXED_META="${FIXED_META:-${AOSS_ROOT}/memory/fusion/gpm_memory_meta_dense_frame_v3.pt}"
FIXED_INDEX="${FIXED_INDEX:-${AOSS_ROOT}/memory/fusion/gpm_memory.index}"
FIXED_ACTIONS="${FIXED_ACTIONS:-${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz}"
FIXED_HEAD="${FIXED_HEAD:-${AOSS_ROOT}/heads/fusion/best.pt}"
COLLECTOR_ROOT="${COLLECTOR_ROOT:-${CAMPAIGN_ROOT}/collector/fusion}"
TASK_IDS="${TASK_IDS:-18,19,25,26}"
MEMORY_ALLOWED_TASK_IDS="${MEMORY_ALLOWED_TASK_IDS:-18,19,25,26}"
ROUNDS="${ROUNDS:-3}"
START_ROUND="${START_ROUND:-1}"
BANK_WORKERS="${BANK_WORKERS:-32}"

[[ "${START_ROUND}" =~ ^[0-9]+$ && "${ROUNDS}" =~ ^[0-9]+$ ]] || {
  echo "START_ROUND and ROUNDS must be positive integers" >&2; exit 2;
}
(( START_ROUND >= 1 && ROUNDS >= START_ROUND )) || {
  echo "Require 1 <= START_ROUND <= ROUNDS" >&2; exit 2;
}

for path in "${OPENPI_PY}" "${FIXED_META}" "${FIXED_INDEX}" "${FIXED_ACTIONS}" "${FIXED_HEAD}" \
  "${VLA_CKPT}/model.safetensors" "${ACTION_STATS}" \
  "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done

run_eval() {
  local admission="$1" run_root="$2" pos_meta="$3" pos_index="$4" pos_actions="$5"
  local neg_meta="${6:-}" neg_index="${7:-}" neg_actions="${8:-}"
  if [[ "${PREFLIGHT_ONLY:-0}" != "1" && -f "${run_root}/fusion/results.txt" \
    && -f "${run_root}/fusion/memory_records/index.jsonl" ]]; then
    echo "[skip] completed evaluation: ${run_root}"
    return 0
  fi
  local resume=0
  [[ -f "${run_root}/run_config.json" ]] && resume=1
  local env_args=(
    AOSS_ROOT="${AOSS_ROOT}"
    VLA_CKPT="${VLA_CKPT}" VLA_CONFIG="${VLA_CONFIG}" ACTION_STATS="${ACTION_STATS}"
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME}"
    HEAD_VARIANTS=fusion TASK_IDS="${TASK_IDS}" MEMORY_ALLOWED_TASK_IDS="${MEMORY_ALLOWED_TASK_IDS}"
    MEMORY_ADMISSION="${admission}" MEMORY_TOP_K="${POSITIVE_TOP_K}" NEGATIVE_MEMORY_TOP_K=8
    NEGATIVE_MEMORY_MIN_SIMILARITY="${NEGATIVE_MEMORY_MIN_SIMILARITY:-0.975}"
    NEGATIVE_MEMORY_MIN_CONFIDENCE="${NEGATIVE_MEMORY_MIN_CONFIDENCE:-0.75}"
    POSITIVE_MEMORY_META_PATH="${pos_meta}" POSITIVE_FAISS_INDEX_PATH="${pos_index}"
    POSITIVE_MEMORY_ACTIONS_PATH="${pos_actions}" TASK_HEAD_CKPT="${FIXED_HEAD}"
    MEMORY_ALIGNMENT_TAG=dense-frame-v3-inherited-self-cl-v1
    UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}"
    UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
    ENV_WORKERS="${ENV_WORKERS:-64}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
    SEED="${SEED:-7}" SAVE_VIDEO="${SAVE_VIDEO:-1}"
    RECORD_MEMORY_DATA=1 MEMORY_TRACE_LEVEL=bank
    TRACE_LOCAL_WORKERS="${TRACE_LOCAL_WORKERS:-8}" TRACE_TRANSFER_WORKERS="${TRACE_TRANSFER_WORKERS:-8}"
    TRACE_TRANSFER_BATCH_SIZE="${TRACE_TRANSFER_BATCH_SIZE:-1024}"
    RUN_ROOT="${run_root}" RESUME="${resume}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
  )
  if [[ "${admission}" == "both" ]]; then
    env_args+=(
      NEGATIVE_MEMORY_META_PATH="${neg_meta}" NEGATIVE_FAISS_INDEX_PATH="${neg_index}"
      NEGATIVE_MEMORY_ACTIONS_PATH="${neg_actions}"
    )
  fi
  env "${env_args[@]}" bash "${ROOT}/scripts/robomemarena/run_predimem_arena_memory_ablation_dual_gpu.sh" "${admission}"
}

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  if [[ "${GROUP}" == "collect" ]]; then
    run_eval success "${CAMPAIGN_ROOT}/collector" "${FIXED_META}" "${FIXED_INDEX}" "${FIXED_ACTIONS}"
  else
    [[ -f "${COLLECTOR_ROOT}/memory_records/index.jsonl" ]] || {
      echo "Collector must finish first: ${COLLECTOR_ROOT}" >&2; exit 2;
    }
    for ((round=1; round<START_ROUND; round++)); do
      prior_round="${CAMPAIGN_ROOT}/branches/${GROUP}/$(printf 'round_%02d' "${round}")/fusion"
      [[ -f "${prior_round}/results.txt" && -f "${prior_round}/memory_records/index.jsonl" ]] || {
        echo "Cannot continue: incomplete prior round ${prior_round}" >&2; exit 3;
      }
    done
    printf 'Static branch preflight passed: k+=%s k-=8 group=%s fixed_bank=%s campaign=%s\n' \
      "${POSITIVE_TOP_K}" "${GROUP}" "${FIXED_META}" "${CAMPAIGN_ROOT}"
  fi
  exit 0
fi

if [[ "${GROUP}" == "collect" ]]; then
  run_eval success "${CAMPAIGN_ROOT}/collector" "${FIXED_META}" "${FIXED_INDEX}" "${FIXED_ACTIONS}"
  exit 0
fi

[[ -f "${COLLECTOR_ROOT}/memory_records/index.jsonl" ]] || {
  echo "Collector must finish first: ${COLLECTOR_ROOT}" >&2; exit 2;
}

sources=("${COLLECTOR_ROOT}")
for ((round=1; round<START_ROUND; round++)); do
  prior_round="${CAMPAIGN_ROOT}/branches/${GROUP}/$(printf 'round_%02d' "${round}")/fusion"
  [[ -f "${prior_round}/results.txt" && -f "${prior_round}/memory_records/index.jsonl" ]] || {
    echo "Cannot continue: incomplete prior round ${prior_round}" >&2; exit 3;
  }
  sources+=("${prior_round}")
done

has_records() {
  local label="$1" source
  for source in "${sources[@]}"; do
    [[ -s "${source}/memory_records/${label}_index.jsonl" ]] && return 0
  done
  return 1
}

for ((round=START_ROUND; round<=ROUNDS; round++)); do
  round_name="$(printf 'round_%02d' "${round}")"
  round_root="${CAMPAIGN_ROOT}/branches/${GROUP}/${round_name}"
  bank_root="${CAMPAIGN_ROOT}/branches/${GROUP}/banks/input_${round_name}"
  online_success="${bank_root}/online_success"
  online_failure="${bank_root}/online_failure"
  composed_positive="${bank_root}/positive_with_fixed_extra8"
  source_args=()
  for source in "${sources[@]}"; do source_args+=(--run-root "${source}"); done

  positive_meta="${FIXED_META}"
  positive_index="${FIXED_INDEX}"
  positive_actions="${FIXED_ACTIONS}"
  if [[ "${GROUP}" == "success" || "${GROUP}" == "both" ]]; then
    if has_records success; then
      if [[ ! -f "${online_success}/provenance.json" ]]; then
        "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_predimem_recorded_bank.py" \
          "${source_args[@]}" --label success --output "${online_success}" \
          --workers "${BANK_WORKERS}" --alignment dense_frame_v3
      fi
      if [[ ! -f "${composed_positive}/provenance.json" ]]; then
        "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/merge_predimem_positive_banks.py" \
          --left-meta "${FIXED_META}" --left-index "${FIXED_INDEX}" --left-actions "${FIXED_ACTIONS}" \
          --right-bank "${online_success}" --output "${composed_positive}" \
          --left-name immutable_extra8_demonstrations
      fi
      positive_meta="${composed_positive}/gpm_memory_meta.pt"
      positive_index="${composed_positive}/gpm_memory.index"
      positive_actions="${composed_positive}/gpm_memory_actions.npz"
    else
      echo "[empty] no admitted success records through ${round_name}; retaining fixed positive bank"
    fi
  fi

  if [[ "${GROUP}" == "failure" || "${GROUP}" == "both" ]]; then
    if has_records failure; then
      if [[ ! -f "${online_failure}/provenance.json" ]]; then
        "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_predimem_recorded_bank.py" \
          "${source_args[@]}" --label failure --output "${online_failure}" \
          --workers "${BANK_WORKERS}" --alignment dense_frame_v3
      fi
      run_eval both "${round_root}" "${positive_meta}" "${positive_index}" "${positive_actions}" \
        "${online_failure}/gpm_memory_meta.pt" "${online_failure}/gpm_memory.index" \
        "${online_failure}/gpm_memory_actions.npz"
    else
      echo "[empty] no admitted failure records through ${round_name}; running without negative memory"
      run_eval success "${round_root}" "${positive_meta}" "${positive_index}" "${positive_actions}"
    fi
  else
    run_eval success "${round_root}" "${positive_meta}" "${positive_index}" "${positive_actions}"
  fi

  [[ -f "${round_root}/fusion/memory_records/index.jsonl" ]] || {
    echo "Round did not finish: ${round_root}" >&2; exit 3;
  }
  sources+=("${round_root}/fusion")
done
