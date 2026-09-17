#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:?Usage: $0 success|failure|both}"
case "${BRANCH}" in success|failure|both) ;; *) echo "Invalid branch: ${BRANCH}" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
COLLECTOR_ROOT="${COLLECTOR_ROOT:?COLLECTOR_ROOT must point to the collector Fusion directory}"
FIRST_ROUND_ROOT="${FIRST_ROUND_ROOT:-}"
START_ROUND="${START_ROUND:-1}"
END_ROUND="${END_ROUND:-3}"
WORKERS="${WORKERS:-16}"
POSITIVE_TOP_K="${POSITIVE_TOP_K:-8}"
NEGATIVE_TOP_K="${NEGATIVE_TOP_K:-8}"

[[ -f "${COLLECTOR_ROOT}/memory_records/index.jsonl" ]] || { echo "Incomplete collector: ${COLLECTOR_ROOT}" >&2; exit 2; }
[[ "${START_ROUND}" -ge 1 && "${END_ROUND}" -ge "${START_ROUND}" ]] || exit 2
mkdir -p "${CAMPAIGN_ROOT}/branches/${BRANCH}"

sources=("${COLLECTOR_ROOT}")
if [[ -n "${FIRST_ROUND_ROOT}" ]]; then
  [[ -f "${FIRST_ROUND_ROOT}/memory_records/index.jsonl" ]] || { echo "Incomplete first round: ${FIRST_ROUND_ROOT}" >&2; exit 2; }
  sources+=("${FIRST_ROUND_ROOT}")
fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf 'Preflight passed: branch=%s rounds=%s..%s +k=%s -k=%s sources=%s\n' \
    "${BRANCH}" "${START_ROUND}" "${END_ROUND}" "${POSITIVE_TOP_K}" "${NEGATIVE_TOP_K}" "${#sources[@]}"
  exit 0
fi

for ((round=START_ROUND; round<=END_ROUND; round++)); do
  round_name="$(printf 'round_%02d' "${round}")"
  round_root="${CAMPAIGN_ROOT}/branches/${BRANCH}/${round_name}"
  bank_root="${CAMPAIGN_ROOT}/branches/${BRANCH}/banks/input_${round_name}"
  positive_root="${bank_root}/positive"
  negative_root="${bank_root}/negative"
  source_args=()
  for source in "${sources[@]}"; do source_args+=(--run-root "${source}"); done

  if [[ "${BRANCH}" != "failure" && ! -f "${positive_root}/provenance.json" ]]; then
    "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_predimem_recorded_bank.py" \
      "${source_args[@]}" --label success --output "${positive_root}" --workers "${WORKERS}"
  fi
  if [[ "${BRANCH}" != "success" && ! -f "${negative_root}/provenance.json" ]]; then
    "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_predimem_recorded_bank.py" \
      "${source_args[@]}" --label failure --output "${negative_root}" --workers "${WORKERS}"
  fi

  resume=0
  [[ -f "${round_root}/run_config.json" ]] && resume=1
  POS_ROOT="${positive_root}" NEG_ROOT="${negative_root}" \
  POSITIVE_MEMORY_META_PATH="${positive_root}/gpm_memory_meta.pt" \
  POSITIVE_FAISS_INDEX_PATH="${positive_root}/gpm_memory.index" \
  POSITIVE_MEMORY_ACTIONS_PATH="${positive_root}/gpm_memory_actions.npz" \
  NEGATIVE_MEMORY_META_PATH="${negative_root}/gpm_memory_meta.pt" \
  NEGATIVE_FAISS_INDEX_PATH="${negative_root}/gpm_memory.index" \
  NEGATIVE_MEMORY_ACTIONS_PATH="${negative_root}/gpm_memory_actions.npz" \
  MEMORY_TOP_K="${POSITIVE_TOP_K}" NEGATIVE_MEMORY_TOP_K="${NEGATIVE_TOP_K}" \
  UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" \
  SEED="${SEED:-7}" UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
  ENV_WORKERS="${ENV_WORKERS:-96}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}" \
  SAVE_VIDEO="${SAVE_VIDEO:-1}" RECORD_MEMORY_DATA=1 MEMORY_TRACE_LEVEL=full \
  RUN_ROOT="${round_root}" RESUME="${resume}" \
    bash "${ROOT}/scripts/robomemarena/run_predimem_arena_transferring_ablation_dual_gpu.sh" "${BRANCH}"

  [[ -f "${round_root}/fusion/memory_records/index.jsonl" ]] || { echo "Round did not finish: ${round_root}" >&2; exit 3; }
  sources+=("${round_root}/fusion")
done
