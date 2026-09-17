#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_self_cl_k8_seed7}" \
COLLECTOR_ROOT="${COLLECTOR_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_memory_eval/collector_success_20260808_030251/fusion}" \
FIRST_ROUND_ROOT="${FIRST_ROUND_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_memory_eval/both_20260808_093304/fusion}" \
START_ROUND=2 END_ROUND=3 POSITIVE_TOP_K=8 NEGATIVE_TOP_K=8 \
  bash "${ROOT}/scripts/robomemarena/run_predimem_transferring_cl_branch.sh" both
