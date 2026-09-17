#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AOSS="/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5"
PYTHON="${PYTHON:-/path/to/user/miniforge3/envs/optimusvla-analysis/bin/python}"
OUTPUT="${OUTPUT:-${ROOT}/docs/report/artifacts/upper_knnlm_temporal_gate_20260818}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memguidance-matplotlib}"

cd "${ROOT}"
exec "${PYTHON}" -m optimus_eval.optimize_upper_knnlm_temporal_gate \
  --manifest "${AOSS}/predimem_2048_fp32/features/anchors_stride5.jsonl" \
  --base-episodes "${AOSS}/eval_frozen_base/counting_base_seed50_ep51_20260814_033923/base/episodes.tsv" \
  --stateless-run "${AOSS}/eval_upper_knnlm_only/counting_tdense5_upper_only_seed50_20260818_031331" \
  --history-run "${AOSS}/eval_upper_knnlm_history/counting_tdense5_upper_history_seed50_20260818_053645" \
  --output-dir "${OUTPUT}" \
  --candidates "${CANDIDATES:-768}" \
  --seed "${SEARCH_SEED:-17}"
