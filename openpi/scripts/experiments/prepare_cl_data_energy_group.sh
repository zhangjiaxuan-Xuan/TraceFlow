#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GROUP="${GROUP:-}"
ADMISSION="${ADMISSION:-both}"
DEVICE="${DEVICE:-cuda}"
ARTIFACT_ROOT="${CL_ARTIFACT_ROOT:-${OPENPI_ROOT}/artifacts/cl_data_energy}"
SMOL_MANIFEST="${SMOL_MANIFEST:-/path/to/local/Mem/runs/collections/smolvla_libero10_3000ep/manifest.jsonl}"

case "${GROUP}" in
  BN|BS|BNS|ALL) ;;
  *) echo "GROUP must be one of BN, BS, BNS, or ALL." >&2; exit 2 ;;
esac
case "${ADMISSION}" in
  success|failure|both) ;;
  *) echo "ADMISSION must be success, failure, or both." >&2; exit 2 ;;
esac
if [[ ! -x "${PYTHON}" ]]; then
  echo "Conda Python is not executable: ${PYTHON}" >&2
  exit 1
fi

HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
B_MANIFEST="${OPENPI_ROOT}/artifacts/prior_head_reproduction/manifest.jsonl"
B_FEATURES="${OPENPI_ROOT}/artifacts/prior_head_reproduction/features"
B_POSITIVE="${ARTIFACT_ROOT}/B_reencoded/positive"
N_MANIFEST="${ARTIFACT_ROOT}/manifests/new_pi.jsonl"
S_MANIFEST="${ARTIFACT_ROOT}/manifests/smol.jsonl"
N_FEATURES="${ARTIFACT_ROOT}/features/new_pi"
S_FEATURES="${ARTIFACT_ROOT}/features/smol"

default_pi_runs=(
  "${OPENPI_ROOT}/logs/libero10_pi05_base_batch8_cl_seed17"
  "${OPENPI_ROOT}/logs/libero10_pi05_base_batch8_cl_seed27"
  "${OPENPI_ROOT}/logs/libero10_guidance_only_v0_success_batch8_cl_seed17"
  "${OPENPI_ROOT}/logs/libero10_guidance_only_v0_success_batch8_cl_seed27"
  "${OPENPI_ROOT}/logs/libero10_guidance_only_v0_success_fail_batch8_cl_seed17"
  "${OPENPI_ROOT}/logs/libero10_guidance_only_v0_success_fail_batch8_cl_seed27"
  "${OPENPI_ROOT}/logs/libero10_guidance_only_v1_success_batch8_cl_seed17"
  "${OPENPI_ROOT}/logs/libero10_guidance_only_v1_success_batch8_cl_seed27"
  "${OPENPI_ROOT}/logs/libero10_guidance_only_v1_success_fail_batch8_cl_seed17"
  "${OPENPI_ROOT}/logs/libero10_guidance_only_v1_success_fail_batch8_cl_seed27"
)
if [[ -n "${PI_RUN_ROOTS:-}" ]]; then
  IFS=: read -r -a pi_runs <<<"${PI_RUN_ROOTS}"
else
  pi_runs=("${default_pi_runs[@]}")
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

positive_args() {
  local root="$1"
  printf '%s\n' \
    "--baseline-positive-meta=${root}/gpm_memory_meta.pt" \
    "--baseline-positive-index=${root}/gpm_memory.index" \
    "--baseline-positive-actions=${root}/gpm_memory_actions.npz"
}

negative_args() {
  local root="$1"
  printf '%s\n' \
    "--baseline-negative-meta=${root}/gpm_negative_memory_meta.pt" \
    "--baseline-negative-index=${root}/gpm_negative_memory.index" \
    "--baseline-negative-actions=${root}/gpm_negative_memory_actions.npz"
}

verify_bank() {
  local root="$1"
  require_file "${root}/positive/gpm_memory_meta.pt"
  require_file "${root}/positive/gpm_memory.index"
  require_file "${root}/positive/gpm_memory_actions.npz"
  require_file "${root}/negative/gpm_negative_memory_meta.pt"
  require_file "${root}/negative/gpm_negative_memory.index"
  require_file "${root}/negative/gpm_negative_memory_actions.npz"
  require_file "${root}/build_summary.json"
}

verify_cl_identity() {
  local root="$1"
  local group="$2"
  local admission="$3"
  "${PYTHON}" - "${root}" "${group}" "${admission}" <<'PY'
import json
from pathlib import Path
import sys
import faiss
import torch

root, group, admission = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
summary = json.loads((root / "build_summary.json").read_text(encoding="utf-8"))
if (summary.get("group"), summary.get("admission")) != (group, admission):
    raise SystemExit(f"CL bank identity mismatch: {summary.get('group')}/{summary.get('admission')}")
negative = torch.load(root / "negative/gpm_negative_memory_meta.pt", map_location="cpu", weights_only=False)
index = faiss.read_index(str(root / "negative/gpm_negative_memory.index"))
if len(negative) != int(index.ntotal) or len(negative) != int(summary["output"]["negative"]["items"]):
    raise SystemExit("Negative metadata/index/summary counts differ")
allowed = {"BN": {"new_pi"}, "BS": {"smol"}, "BNS": {"new_pi", "smol"}}[group]
for item in negative:
    provenance = item.get("provenance", {})
    if provenance.get("source_family") not in allowed:
        raise SystemExit(f"Non-CL failure memory detected: {item.get('action_id')}")
    if float(item.get("failure_confidence", -1.0)) != 1.0:
        raise SystemExit(f"Failure confidence is not neutralized: {item.get('action_id')}")
if group in {"BN", "BS"} and int(summary["baseline_items"]["negative"]) != 0:
    raise SystemExit("CL parent negative bank must be empty")
if admission == "success" and negative:
    raise SystemExit("Success-only admission produced failure memories")
print(f"CL bank identity passed: {group}/{admission}, failures={len(negative)}")
PY
}

prepare_b() {
  require_file "${HEAD_CHECKPOINT}"
  require_file "${B_MANIFEST}"
  require_file "${B_FEATURES}/pooled_prefix.npy"
  require_file "${B_FEATURES}/completed.npy"
  if [[ ! -f "${B_POSITIVE}/build_summary.json" ]]; then
    "${PYTHON}" "${OPENPI_ROOT}/scripts/memory/build_prior_head_memory.py" \
      --manifest "${B_MANIFEST}" \
      --feature-dir "${B_FEATURES}" \
      --checkpoint "${HEAD_CHECKPOINT}" \
      --output-dir "${B_POSITIVE}" \
      --device "${DEVICE}"
  fi
  require_file "${B_POSITIVE}/gpm_memory_meta.pt"
  require_file "${B_POSITIVE}/gpm_memory.index"
  require_file "${B_POSITIVE}/gpm_memory_actions.npz"
}

prepare_manifest_n() {
  if [[ ! -f "${N_MANIFEST}" ]]; then
    local args=()
    local run
    for run in "${pi_runs[@]}"; do
      args+=(--pi-run-root "${run}")
    done
    "${PYTHON}" "${OPENPI_ROOT}/scripts/data/prepare_cl_memory_manifest.py" \
      "${args[@]}" --output "${N_MANIFEST}" --outcome all
  fi
}

prepare_manifest_s() {
  require_file "${SMOL_MANIFEST}"
  if [[ ! -f "${S_MANIFEST}" ]]; then
    "${PYTHON}" "${OPENPI_ROOT}/scripts/data/prepare_cl_memory_manifest.py" \
      --smol-manifest "${SMOL_MANIFEST}" --output "${S_MANIFEST}" --outcome all \
      --smol-workers "${SMOL_VALIDATION_WORKERS:-8}"
  fi
}

cache_features() {
  local manifest="$1"
  local output="$2"
  "${PYTHON}" "${OPENPI_ROOT}/scripts/memory/cache_prior_head_features.py" \
    --manifest "${manifest}" \
    --output-dir "${output}" \
    --policy-dir "${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}" \
    --config-name pi05_libero \
    --device "${DEVICE}" \
    --batch-size "${FEATURE_BATCH_SIZE:-8}"
}

build_bank() {
  local group="$1"
  local manifest="$2"
  local features="$3"
  local baseline_root="$4"
  local output="${ARTIFACT_ROOT}/banks/${group}/${ADMISSION}"
  if [[ -f "${output}/build_summary.json" ]]; then
    verify_bank "${output}"
    verify_cl_identity "${output}" "${group}" "${ADMISSION}"
    echo "Validated existing bank: ${output}"
    return
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    local quarantine="${output}.interrupted_$(date +%Y%m%d_%H%M%S)_$$"
    mv "${output}" "${quarantine}"
    echo "Quarantined interrupted bank build: ${quarantine}" >&2
  fi
  mapfile -t pos_args < <(positive_args "${baseline_root}/positive")
  neg_args=()
  if [[ "${baseline_root}" != "${ARTIFACT_ROOT}/B_reencoded" && "${ADMISSION}" != "success" ]]; then
    mapfile -t neg_args < <(negative_args "${baseline_root}/negative")
  fi
  "${PYTHON}" "${OPENPI_ROOT}/scripts/memory/build_cl_memory_bank.py" \
    --group "${group}" \
    --manifest "${manifest}" \
    --feature-dir "${features}" \
    --checkpoint "${HEAD_CHECKPOINT}" \
    "${pos_args[@]}" "${neg_args[@]}" \
    --output-dir "${output}" \
    --admission "${ADMISSION}" \
    --device "${DEVICE}"
  verify_bank "${output}"
  verify_cl_identity "${output}" "${group}" "${ADMISSION}"
}

cd "${OPENPI_ROOT}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  require_file "${HEAD_CHECKPOINT}"
  require_file "${B_MANIFEST}"
  require_file "${B_FEATURES}/pooled_prefix.npy"
  require_file "${B_FEATURES}/completed.npy"
  require_file "${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}/model.safetensors"
  if [[ "${GROUP}" == "BN" || "${GROUP}" == "BNS" || "${GROUP}" == "ALL" ]]; then
    for run in "${pi_runs[@]}"; do require_file "${run}/indexes/all_episodes.tsv"; done
  fi
  if [[ "${GROUP}" == "BS" || "${GROUP}" == "BNS" || "${GROUP}" == "ALL" ]]; then
    require_file "${SMOL_MANIFEST}"
  fi
  echo "CL preparation preflight passed: group=${GROUP} admission=${ADMISSION} python=${PYTHON}"
  exit 0
fi

prepare_b

if [[ "${GROUP}" == "BN" || "${GROUP}" == "BNS" || "${GROUP}" == "ALL" ]]; then
  prepare_manifest_n
  cache_features "${N_MANIFEST}" "${N_FEATURES}"
  build_bank BN "${N_MANIFEST}" "${N_FEATURES}" "${ARTIFACT_ROOT}/B_reencoded"
fi
if [[ "${GROUP}" == "BS" || "${GROUP}" == "BNS" || "${GROUP}" == "ALL" ]]; then
  prepare_manifest_s
  cache_features "${S_MANIFEST}" "${S_FEATURES}"
fi
if [[ "${GROUP}" == "BS" || "${GROUP}" == "ALL" ]]; then
  build_bank BS "${S_MANIFEST}" "${S_FEATURES}" "${ARTIFACT_ROOT}/B_reencoded"
fi
if [[ "${GROUP}" == "BNS" || "${GROUP}" == "ALL" ]]; then
  build_bank BNS "${S_MANIFEST}" "${S_FEATURES}" "${ARTIFACT_ROOT}/banks/BN/${ADMISSION}"
fi

if [[ "${GROUP}" == "ALL" ]]; then
  for ready_group in BN BS BNS; do
    final_bank="${ARTIFACT_ROOT}/banks/${ready_group}/${ADMISSION}"
    verify_bank "${final_bank}"
    verify_cl_identity "${final_bank}" "${ready_group}" "${ADMISSION}"
  done
  echo "All CL data-energy banks are ready in order: BN -> BS -> BNS"
else
  FINAL_BANK="${ARTIFACT_ROOT}/banks/${GROUP}/${ADMISSION}"
  verify_bank "${FINAL_BANK}"
  verify_cl_identity "${FINAL_BANK}" "${GROUP}" "${ADMISSION}"
  echo "CL data-energy bank ready: group=${GROUP} admission=${ADMISSION} path=${FINAL_BANK}"
fi
