#!/usr/bin/env bash
set -uo pipefail

GROUP=${1:?Usage: $0 group1|group2}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUND_ROOT="${ROUND_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32/eval/lower_rescue_round1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

case "${GROUP}" in
  group1)
    samples=(sample0 sample1 sample2)
    run_names=(sample0_rescue0_c2_m0.05_r3_seed50 sample1_rescue1_c3_m0.08_r3_seed50 sample2_rescue1_c2_m0.05_r3_seed50)
    port_base=9100 ;;
  group2)
    samples=(sample3 sample4 sample5)
    run_names=(sample3_rescue1_c1_m0.08_r3_seed50 sample4_rescue1_c2_m0.03_r4_seed50 sample5_rescue1_c1_m0.03_r4_seed50)
    port_base=9200 ;;
  *) echo "Unknown group: ${GROUP}" >&2; exit 2 ;;
esac

failed=()
for index in "${!samples[@]}"; do
  sample="${samples[index]}"; run_root="${ROUND_ROOT}/${run_names[index]}"; result="${run_root}/fusion/results.txt"
  [[ -s "${result}" ]] && { echo "[lower-rescue:${GROUP}] ${sample}: complete, skipping"; continue; }
  completed=0
  for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    resume=0; [[ -f "${run_root}/run_config.json" ]] && resume=1
    upper_port=$((port_base + index * 10 + attempt * 2)); lower_port=$((upper_port + 1))
    echo "[lower-rescue:${GROUP}] ${sample}: attempt=${attempt}/${MAX_ATTEMPTS} resume=${resume}"
    if env RUN_ROOT="${run_root}" RESUME="${resume}" UPPER_PORT="${upper_port}" LOWER_PORT="${lower_port}" \
      UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" \
      UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
      ENV_WORKERS="${ENV_WORKERS:-32}" \
      bash "${ROOT}/scripts/robomemarena/run_predimem_counting_lower_rescue_sample.sh" "${sample}"; then
      [[ -s "${result}" ]] && { completed=1; break; }
    fi
  done
  [[ "${completed}" == "1" ]] || failed+=("${sample}")
done

[[ "${#failed[@]}" == "0" ]] || { echo "[lower-rescue:${GROUP}] incomplete: ${failed[*]}" >&2; exit 1; }
echo "[lower-rescue:${GROUP}] all samples complete"
