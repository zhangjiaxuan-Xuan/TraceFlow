#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/robomemarena/run_predimem_suite_conditioned_dual_gpu.sh" occlusion
