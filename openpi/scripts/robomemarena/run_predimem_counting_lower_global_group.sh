#!/usr/bin/env bash
set -euo pipefail

GROUP=${1:?Usage: $0 group1|group2}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUND_ROOT="${ROUND_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32/eval/lower_global_round2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

case "${GROUP}" in
  group1)
    samples=(sample0 sample1 sample2)
    run_names=(sample0_shadow_p0.35_s0.45_m0.03_seed50 sample1_control_p0.65_s0.70_m0.10_seed50 sample2_control_p0.55_s0.62_m0.07_seed50)
    port_base=9300 ;;
  group2)
    samples=(sample3 sample4 sample5)
    run_names=(sample3_control_p0.45_s0.55_m0.05_seed50 sample4_control_p0.35_s0.45_m0.03_seed50 sample5_control_p0.25_s0.35_m0.00_seed50)
    port_base=9400 ;;
  *) echo "Unknown group: ${GROUP}" >&2; exit 2 ;;
esac

failed=()
for index in "${!samples[@]}"; do
  sample="${samples[index]}"
  run_root="${ROUND_ROOT}/${run_names[index]}"
  result="${run_root}/fusion/results.txt"
  if [[ -s "${result}" ]]; then
    echo "[lower-global:${GROUP}] ${sample}: complete, skipping"
    continue
  fi
  completed=0
  for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    resume=0
    [[ -f "${run_root}/run_config.json" ]] && resume=1
    upper_port=$((port_base + index * 10 + attempt * 2))
    lower_port=$((upper_port + 1))
    echo "[lower-global:${GROUP}] ${sample}: attempt=${attempt}/${MAX_ATTEMPTS} resume=${resume}"
    if env RUN_ROOT="${run_root}" RESUME="${resume}" UPPER_PORT="${upper_port}" LOWER_PORT="${lower_port}" \
      UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" \
      UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
      ENV_WORKERS="${ENV_WORKERS:-32}" \
      bash "${ROOT}/scripts/robomemarena/run_predimem_counting_lower_global_sample.sh" "${sample}"; then
      if [[ -s "${result}" ]]; then
        completed=1
        break
      fi
    fi
  done
  [[ "${completed}" == "1" ]] || failed+=("${sample}")
done

if [[ "${#failed[@]}" != "0" ]]; then
  echo "[lower-global:${GROUP}] incomplete: ${failed[*]}" >&2
  exit 1
fi
echo "[lower-global:${GROUP}] all samples complete"
