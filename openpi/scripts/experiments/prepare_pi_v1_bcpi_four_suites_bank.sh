#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHONPATH_VALUE="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/third_party/libero"
if [[ -n "${PYTHONPATH:-}" ]]; then PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"; fi
export PYTHONPATH="${PYTHONPATH_VALUE}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
SOURCE_ROOT="${C_PI_SOURCE_ROOT:-/path/to/local/CVPR26-OptimusVLA/openpi}"
C_PI_MANIFEST="${C_PI_MANIFEST:-${SOURCE_ROOT}/logs/data_collection/C-pi_four_suites_seed7/natural_attempts.jsonl}"
N_MANIFEST="${N_FAILURE_SOURCE_MANIFEST:-${SOURCE_ROOT}/artifacts/cl_data_energy/manifests/new_pi.jsonl}"
B_POSITIVE="${B_POSITIVE_ROOT:-${SOURCE_ROOT}/artifacts/cl_data_energy/B_reencoded/positive}"
ARTIFACT_ROOT="${BCPI_ARTIFACT_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/bcpi_nfailure_four_suites_v1}"
SOURCE_MANIFEST="${ARTIFACT_ROOT}/manifests/c_pi_all_plus_n_failures.jsonl"
FEATURE_DIR="${ARTIFACT_ROOT}/features/c_pi_all_plus_n_failures"
BANK_ROOT="${ARTIFACT_ROOT}/bank_b_plus_cpi_success_cpi_n_failure"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${SOURCE_ROOT}/checkpoints/gpm_task_head.pt}"
DEVICE="${DEVICE:-cuda}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
FEATURE_MIN_BATCH_SIZE="${FEATURE_MIN_BATCH_SIZE:-8}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

command -v flock >/dev/null 2>&1 || { echo "Missing required command: flock" >&2; exit 1; }

for path in \
  "${PYTHON}" \
  "${C_PI_MANIFEST}" \
  "${N_MANIFEST}" \
  "${POLICY_DIR}/model.safetensors" \
  "${TASK_HEAD_CKPT}" \
  "${B_POSITIVE}/gpm_memory_meta.pt" \
  "${B_POSITIVE}/gpm_memory.index" \
  "${B_POSITIVE}/gpm_memory_actions.npz" \
  "${OPENPI_ROOT}/scripts/memory/cache_prior_head_features.py" \
  "${OPENPI_ROOT}/scripts/memory/build_cl_memory_bank.py"; do
  require_file "${path}"
done

"${PYTHON}" - "${C_PI_MANIFEST}" "${N_MANIFEST}" "${B_POSITIVE}/gpm_memory_meta.pt" <<'PY'
import json
from collections import Counter
from pathlib import Path
import sys
import torch

c_manifest = Path(sys.argv[1])
n_manifest = Path(sys.argv[2])
c_rows = [json.loads(line) for line in c_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
n_rows = [json.loads(line) for line in n_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
c_outcomes = Counter(bool(row["success"]) for row in c_rows)
c_suites = Counter(str(row["suite"]) for row in c_rows)
expected_suites = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
if len(c_rows) != 5450 or c_outcomes != Counter({True: 5334, False: 116}):
    raise SystemExit(f"Unexpected frozen C-pi inventory: rows={len(c_rows)} outcomes={c_outcomes}")
if set(c_suites) != expected_suites:
    raise SystemExit(f"Unexpected C-pi suites: {c_suites}")
if any(row.get("collection_group") != "C-pi" or row.get("producer") != "pi0.5-libero" for row in c_rows):
    raise SystemExit("C-pi manifest contains a foreign collection group or producer")
n_failures = [row for row in n_rows if not bool(row["success"])]
if len(n_rows) != 10000 or len(n_failures) != 497:
    raise SystemExit(f"Unexpected frozen N inventory: rows={len(n_rows)} failures={len(n_failures)}")
if any(
    row.get("suite") != "libero_10"
    or row.get("source_family") != "new_pi"
    or row.get("source_model") != "pi0.5"
    or row.get("source_format") != "eval_hdf5"
    for row in n_failures
):
    raise SystemExit("N failure manifest contains an unexpected suite, source family, model, or format")
c_paths = {str(Path(row["trajectory_path"]).resolve()) for row in c_rows if not bool(row["success"])}
n_paths = {str(Path(row["trajectory_path"]).resolve()) for row in n_failures}
if c_paths & n_paths:
    raise SystemExit(f"C-pi and N failure trajectories overlap: {len(c_paths & n_paths)} paths")
if len(n_paths) != len(n_failures):
    raise SystemExit("N failure manifest contains duplicate trajectory paths")
if any(not Path(path).is_file() for path in n_paths):
    raise SystemExit("At least one frozen N failure trajectory is missing")
c_action_ids = {str(row["action_id"]) for row in c_rows if not bool(row["success"])}
n_action_ids = {str(row["action_id"]) for row in n_failures}
if c_action_ids & n_action_ids:
    raise SystemExit(f"C-pi and N failure action IDs overlap: {len(c_action_ids & n_action_ids)} IDs")
baseline = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
if len(baseline) != 6500:
    raise SystemExit(f"Expected B positive baseline with 6500 items, found {len(baseline)}")
print(
    "Frozen four-suite inventory passed: "
    f"B=6500 C-positive=5334 C-failure=116 N-failure={len(n_failures)} "
    f"suites={dict(c_suites)}"
)
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "B+C-pi+N-failure preparation preflight passed: output=${BANK_ROOT}"
  exit 0
fi

mkdir -p "${ARTIFACT_ROOT}"
exec 9>"${ARTIFACT_ROOT}/.prepare.lock"
flock 9

"${PYTHON}" - "${C_PI_MANIFEST}" "${N_MANIFEST}" "${SOURCE_MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

c_source = Path(sys.argv[1])
n_source = Path(sys.argv[2])
output = Path(sys.argv[3])
c_rows = [json.loads(line) for line in c_source.read_text(encoding="utf-8").splitlines() if line.strip()]
n_rows = [json.loads(line) for line in n_source.read_text(encoding="utf-8").splitlines() if line.strip()]
n_failures = [row for row in n_rows if not bool(row["success"])]
rows = c_rows + n_failures
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(".jsonl.tmp")
with temporary.open("w", encoding="utf-8") as stream:
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
temporary.replace(output)
print(f"Materialized four-suite source manifest: rows={len(rows)} output={output}")
PY

"${PYTHON}" "${OPENPI_ROOT}/scripts/memory/cache_prior_head_features.py" \
  --manifest "${SOURCE_MANIFEST}" \
  --output-dir "${FEATURE_DIR}" \
  --policy-dir "${POLICY_DIR}" \
  --config-name pi05_libero \
  --device "${DEVICE}" \
  --batch-size "${FEATURE_BATCH_SIZE}" \
  --min-batch-size "${FEATURE_MIN_BATCH_SIZE}" \
  --auto-batch

if [[ ! -f "${BANK_ROOT}/build_summary.json" ]]; then
  if [[ -d "${BANK_ROOT}" ]] && [[ -n "$(find "${BANK_ROOT}" -mindepth 1 -print -quit)" ]]; then
    interrupted="${BANK_ROOT}.interrupted_$(date -u +%Y%m%d_%H%M%S)_$$"
    mv "${BANK_ROOT}" "${interrupted}"
    echo "Preserved interrupted bank at ${interrupted}" >&2
  fi
  "${PYTHON}" "${OPENPI_ROOT}/scripts/memory/build_cl_memory_bank.py" \
    --group B_C_pi_Nfailure_four_suites \
    --manifest "${SOURCE_MANIFEST}" \
    --feature-dir "${FEATURE_DIR}" \
    --checkpoint "${TASK_HEAD_CKPT}" \
    --baseline-positive-meta "${B_POSITIVE}/gpm_memory_meta.pt" \
    --baseline-positive-index "${B_POSITIVE}/gpm_memory.index" \
    --baseline-positive-actions "${B_POSITIVE}/gpm_memory_actions.npz" \
    --output-dir "${BANK_ROOT}" \
    --admission both \
    --device "${DEVICE}"
fi

"${PYTHON}" - "${BANK_ROOT}" "${SOURCE_MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import faiss
import torch

root = Path(sys.argv[1])
source = Path(sys.argv[2])
summary = json.loads((root / "build_summary.json").read_text(encoding="utf-8"))
positive = torch.load(root / "positive/gpm_memory_meta.pt", map_location="cpu", weights_only=False)
negative = torch.load(root / "negative/gpm_negative_memory_meta.pt", map_location="cpu", weights_only=False)
positive_index = faiss.read_index(str(root / "positive/gpm_memory.index"))
negative_index = faiss.read_index(str(root / "negative/gpm_negative_memory.index"))
if len(positive) != 11834 or len(negative) != 613:
    raise SystemExit(f"Unexpected four-suite bank sizes: positive={len(positive)} negative={len(negative)}")
if int(positive_index.ntotal) != len(positive) or int(negative_index.ntotal) != len(negative):
    raise SystemExit("B+C-pi metadata and FAISS lengths differ")
if summary.get("baseline_items", {}).get("positive") != 6500:
    raise SystemExit("B+C-pi bank does not have the expected B baseline")
if summary.get("admitted_items") != {"positive": 5334, "negative": 613}:
    raise SystemExit(f"Unexpected admitted inventory: {summary.get('admitted_items')}")
identity = {
    "schema": "pi_v1_bcpi_nfailure_four_suites_bank_v1",
    "source_manifest": str(source.resolve()),
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "positive_items": len(positive),
    "negative_items": len(negative),
    "positive_contract": "B6500 + all C-pi successes",
    "negative_contract": "all C-pi failures + all N failures, physically isolated",
}
(root / "bank_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
print(f"B+C-pi+N-failure bank verified: positive={len(positive)} negative={len(negative)} root={root}")
PY

echo "B+C-pi+N-failure four-suite bank ready: ${BANK_ROOT}"
