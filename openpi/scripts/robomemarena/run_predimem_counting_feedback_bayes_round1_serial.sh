#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUND_ROOT="${ROUND_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32/eval/feedback_bayes_round1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

samples=(sample0 sample1 sample2 sample3 sample4 sample5)
run_names=(
  sample0_h4_c2_s0.02_p0.10_r2_seed50
  sample1_h3_c1_s0.04_p0.20_r1_seed50
  sample2_h3_c2_s0.01_p0.05_r1_seed50
  sample3_h6_c2_s0.02_p0.10_r3_seed50
  sample4_h6_c3_s0.04_p0.20_r2_seed50
  sample5_h6_c2_s0.01_p0.05_r2_seed50
)

failed=()
for index in "${!samples[@]}"; do
  sample="${samples[index]}"
  run_root="${ROUND_ROOT}/${run_names[index]}"
  result="${run_root}/fusion/results.txt"

  if [[ -s "${result}" ]]; then
    echo "[bayes-round1] ${sample}: complete, skipping"
    continue
  fi

  completed=0
  for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    resume=0
    [[ -f "${run_root}/run_config.json" ]] && resume=1
    upper_port=$((8830 + index * 10 + attempt * 2))
    lower_port=$((upper_port + 1))
    echo "[bayes-round1] ${sample}: attempt=${attempt}/${MAX_ATTEMPTS} resume=${resume} ports=${upper_port}/${lower_port}"

    if env \
      RUN_ROOT="${run_root}" RESUME="${resume}" \
      UPPER_PORT="${upper_port}" LOWER_PORT="${lower_port}" \
      UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" \
      UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" \
      LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
      ENV_WORKERS="${ENV_WORKERS:-64}" \
      bash "${ROOT}/scripts/robomemarena/run_predimem_counting_feedback_bayes_round1.sh" "${sample}"; then
      if [[ -s "${result}" ]]; then
        completed=1
        echo "[bayes-round1] ${sample}: complete"
        break
      fi
      echo "[bayes-round1] ${sample}: command exited cleanly but result is missing"
    else
      status=$?
      echo "[bayes-round1] ${sample}: attempt ${attempt} failed with status ${status}"
    fi
  done

  [[ "${completed}" == "1" ]] || failed+=("${sample}")
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "[bayes-round1] incomplete samples: ${failed[*]}" >&2
  exit 1
fi
echo "[bayes-round1] all samples complete"
