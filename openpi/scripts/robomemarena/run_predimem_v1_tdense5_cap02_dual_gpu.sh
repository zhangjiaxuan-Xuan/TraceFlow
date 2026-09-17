#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MEMORY_GUIDANCE_NORM_CAP=0.20
export MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20
export PREDIMEM_MODE=v1
exec bash "${ROOT}/scripts/robomemarena/run_predimem_v1_tdense5_common_dual_gpu.sh"
