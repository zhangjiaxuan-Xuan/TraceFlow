#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/analysis}"
TASK_METRICS="${TASK_METRICS:-${ANALYSIS_ROOT}/suite_discreteness_gpu_20260812/task_metrics.csv}"
ALIGNMENT_METRICS="${ALIGNMENT_METRICS:-${ANALYSIS_ROOT}/action_directional_compatibility_gpu_20260812/task_compatibility.csv}"
ALIGNMENT_COLUMN="${ALIGNMENT_COLUMN:-behavior_motion_weighted_loo_cosine}"
V0_AGGREGATE="${V0_AGGREGATE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_timed/v0_k16_batch32_env64_seed50_trials51_20260811_040843/fusion/aggregate.json}"
V1_AGGREGATE="${V1_AGGREGATE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_timed/v1_k16_batch32_env64_seed50_trials51_20260811_035629/fusion/aggregate.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/docs/report/data/predimem_two_metric_gate_v1}"
FIT_DEVICE="${FIT_DEVICE:-cpu}"
FIT_L2="${FIT_L2:-0.005}"
V1_CAP="${V1_CAP:-0.2}"
V0_CAP="${V0_CAP:-1.0}"
BASE_CUTOFF="${BASE_CUTOFF:-0.3}"
MAX_CUTOFF="${MAX_CUTOFF:-0.6}"

for path in "${PYTHON}" "${TASK_METRICS}" "${ALIGNMENT_METRICS}" "${V0_AGGREGATE}" "${V1_AGGREGATE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required fit input: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
exec "${PYTHON}" -m optimus_eval.fit_predimem_two_metric_gate \
  --task-metrics "${TASK_METRICS}" \
  --alignment "${ALIGNMENT_METRICS}" \
  --alignment-column "${ALIGNMENT_COLUMN}" \
  --v0-aggregate "${V0_AGGREGATE}" \
  --v1-aggregate "${V1_AGGREGATE}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${FIT_DEVICE}" \
  --l2 "${FIT_L2}" \
  --v1-cap "${V1_CAP}" \
  --v0-cap "${V0_CAP}" \
  --base-cutoff "${BASE_CUTOFF}" \
  --max-cutoff "${MAX_CUTOFF}"
