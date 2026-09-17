#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_PY="${OPENPI_PY:-python}"
SELF_ROOT="${SELF_ROOT:?SELF_ROOT must point to a verified TraceFlow Arena asset group}"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT must contain the converted official PrediMem VLA}"
export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src:${ROOT}:${PYTHONPATH:-}"
CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}"
MODE="${PREDIMEM_MODE:-v1}"
case "${MODE}" in
  v0|v1) ;;
  *) echo "PREDIMEM_MODE must be v0 or v1, got ${MODE}" >&2; exit 2 ;;
esac

case "${CAP}" in
  0.2|0.20|0.200|0.5|0.50|0.500) ;;
  *) echo "PrediMem Tdense5 V1 cap must be 0.20 or 0.50, got ${CAP}" >&2; exit 2 ;;
esac

HEAD_VARIANTS="${HEAD_VARIANTS:-lower upper fusion}"
MEMORY_META_BASENAME="${MEMORY_META_BASENAME:-gpm_memory_meta.pt}"
for variant in ${HEAD_VARIANTS}; do
  for path in \
    "${SELF_ROOT}/heads/${variant}/best.pt" \
    "${SELF_ROOT}/memory/${variant}/gpm_memory.index" \
    "${SELF_ROOT}/memory/${variant}/${MEMORY_META_BASENAME}"; do
    [[ -f "${path}" ]] || { echo "Missing PrediMem Tdense5 artifact: ${path}" >&2; exit 2; }
  done
done
for path in "${SELF_ROOT}/memory/shared/gpm_memory_actions.npz"; do
  [[ -f "${path}" ]] || { echo "Missing PrediMem Tdense5 artifact: ${path}" >&2; exit 2; }
done

"${OPENPI_PY}" - "${SELF_ROOT}" "${ROOT}" ${HEAD_VARIANTS} <<'PY'
import inspect
from pathlib import Path
import sys
import torch
from openpi.task_head.memory_init import MemoryInitProvider

root = Path(sys.argv[1])
runtime_root = Path(sys.argv[2]).resolve()
variants = sys.argv[3:]
provider_source = Path(inspect.getfile(MemoryInitProvider)).resolve()
assert provider_source.is_relative_to(runtime_root / "src"), provider_source
assert "action_alignment" in inspect.signature(MemoryInitProvider).parameters
for variant in variants:
    checkpoint = torch.load(root / "heads" / variant / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["head_type"] == "dual_tower"
    assert checkpoint["variant"] == variant
    assert int(checkpoint["lower_dim"]) == 6144
    assert int(checkpoint["upper_dim"]) == 4096
    assert int(checkpoint["hidden"]) == 1024
    assert int(checkpoint["out_dim"]) == 2048
    assert int(checkpoint["temporal_window"]) == 3
    assert list(checkpoint["temporal_offsets"]) == [-20, -10, 0]
    assert bool(checkpoint["joint_conditioned"])
    assert checkpoint["lower_conditioning_protocol"] == "upper_generated_subtask_v1"
    assert checkpoint["upper_feature_protocol"] == "predimem_trajectory_ordered_runtime_generate_subtask_hidden_v4"
    assert checkpoint.get("lower_producer_policy_dir")
    assert checkpoint.get("subtask_records_sha256")
print(f"Verified producer-matched joint Tdense5 heads: {variants}")
PY

REPLAN_STEPS="${REPLAN_STEPS:-10}"
VLM_INTERVAL="${VLM_INTERVAL:-5}"
if (( REPLAN_STEPS != 10 || REPLAN_STEPS % VLM_INTERVAL != 0 )); then
  echo "Tdense5 frequency contract requires REPLAN_STEPS=10 and VLM_INTERVAL dividing 10" >&2
  exit 2
fi

cap_tag="$("${OPENPI_PY}" -c 'import sys; print(f"{float(sys.argv[1]):.1f}".replace(".", ""))' "${CAP}")"
export ROOT
export ARENA_ROOT="${ARENA_ROOT:-${ROOT}/../RoboMemArena}"
export AOSS_ROOT="${SELF_ROOT}"
export HEAD_VARIANTS
export VLA_CKPT="${VLA_CKPT:-${SOURCE_ROOT}/checkpoints/vla_alltask_pytorch}"
export VLM_CKPT="${VLM_CKPT:?VLM_CKPT must point to the official PrediMem Upper checkpoint}"
export MEMORY_META_BASENAME
export MEMORY_ALIGNMENT_TAG="joint-tdense5-stage-frame-v3-upper-hold10"
export MEMORY_GUIDANCE_NORM_CAP="${CAP}"
export MEMORY_GUIDANCE_TOTAL_NORM_CAP="${CAP}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-16}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-16}"
export ENV_WORKERS="${ENV_WORKERS:-32}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export SEED="${SEED:-50}"
export REPLAN_STEPS VLM_INTERVAL
if [[ "${MODE}" == "v0" ]]; then
  run_tag="v0"
else
  run_tag="v1_cap${cap_tag}"
fi
export RUN_ROOT="${RUN_ROOT:-${SELF_ROOT}/eval/${run_tag}_batch${LOWER_BATCH_SIZE}_env${ENV_WORKERS}_seed${SEED}_$(date -u +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" "${MODE}"
