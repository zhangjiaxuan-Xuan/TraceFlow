#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
POSITIVE_SELECTION_PATH="${POSITIVE_SELECTION_PATH:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_positive_topk_ablation_v3_batch16_env32/selection/positive_selection.json}" \
SELECTED_FOUR_SUITE_ROOT="${SELECTED_FOUR_SUITE_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_positive_topk_ablation_v3_batch16_env32/selected_four_suites_s50k16_canonical_v1}" \
POSITIVE_CAPACITY_OVERRIDE=50 \
POSITIVE_TOP_K_OVERRIDE=16 \
SELECTION_AUDITED=1 \
MODE=both \
  exec bash "${SCRIPT_DIR}/run_pi_v1_bcpi_four_suites_dynamic.sh" "$@"
