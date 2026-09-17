#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PREDIMEM_MODE=v0
# V0 does not apply a cap; these values only keep the shared contract explicit.
export MEMORY_GUIDANCE_NORM_CAP=0.20
export MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20
exec bash "${ROOT}/scripts/robomemarena/run_predimem_v1_tdense5_common_dual_gpu.sh"
