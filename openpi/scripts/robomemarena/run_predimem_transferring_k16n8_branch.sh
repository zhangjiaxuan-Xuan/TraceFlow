#!/usr/bin/env bash
set -euo pipefail
BRANCH="${1:?Usage: $0 success|failure|both}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_self_cl_k16n8_seed7}" \
COLLECTOR_ROOT="${COLLECTOR_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_self_cl_k16n8_seed7/collector/fusion}" \
START_ROUND=1 END_ROUND=3 POSITIVE_TOP_K=16 NEGATIVE_TOP_K=8 \
  bash "${ROOT}/scripts/robomemarena/run_predimem_transferring_cl_branch.sh" "${BRANCH}"
