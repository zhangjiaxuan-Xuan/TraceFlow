#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
SCOPE="${EXPERIMENT_SCOPE:-formal}"
GROUP="${MEMORY_GROUP:-}"

case "${SCOPE}:${GROUP}" in
  formal:N|formal:S|special:N) ;;
  *) echo "Allowed evaluation targets: formal:N/S or special:N" >&2; exit 2 ;;
esac

BANK="${OPENPI_ROOT}/artifacts/cl_data_energy/standalone_banks/${SCOPE}/${GROUP}"
SUMMARY="${BANK}/build_summary.json"
[[ -f "${SUMMARY}" ]] || { echo "Missing completed bank: ${SUMMARY}" >&2; exit 1; }
python3 - "${SUMMARY}" "${SCOPE}" "${GROUP}" <<'PY'
import json
from pathlib import Path
import sys

summary_path, scope, group = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected = {
    ("formal", "N"): (9503, 497),
    ("formal", "S"): (1493, 1507),
    ("special", "N"): (1010, 427),
}[(scope, group)]
actual = (
    int(summary["output"]["positive"]["items"]),
    int(summary["output"]["negative"]["items"]),
)
if actual != expected or int(summary["baseline_items"]["positive"]) or int(summary["baseline_items"]["negative"]):
    raise SystemExit(
        f"Standalone bank mismatch for {scope}:{group}: expected={expected}, actual={actual}, "
        f"baseline={summary['baseline_items']}"
    )
print(f"Bank composition verified: {scope}:{group} success/failure={actual}")
PY
export CL_GUIDANCE_VERSION=v0
export CL_BANK_DIR="${BANK}"
export CL_GROUP_TAG="${GROUP}"
export CL_ADMISSION=both
export RUN_ROOT="${RUN_ROOT:-${OPENPI_ROOT}/logs/stage2/${SCOPE}/pi_v0_${GROUP}_seed${SEED:-7}_${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}}"

exec bash "${SCRIPT_DIR}/run_cl_v0_eval.sh" "$@"
