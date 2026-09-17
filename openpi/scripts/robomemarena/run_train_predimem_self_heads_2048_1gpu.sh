#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"

SOURCE_ROOT="${SOURCE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v2/predimem_2048_fp32}"
MANIFEST="${MANIFEST:-${SOURCE_ROOT}/features/anchors16.jsonl}"
LOWER_FEATURES="${LOWER_FEATURES:-${SOURCE_ROOT}/features/lower/pooled_prefix.npy}"
UPPER_FEATURES="${UPPER_FEATURES:-${SOURCE_ROOT}/features/upper/upper_features.npy}"
VLA_CKPT="${VLA_CKPT:-${SOURCE_ROOT}/checkpoints/vla_alltask_pytorch}"

for path in \
  "${OPENPI_PY}" "${MANIFEST}" "${LOWER_FEATURES}" "${UPPER_FEATURES}" \
  "${VLA_CKPT}/model.safetensors"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

"${OPENPI_PY}" - "${LOWER_FEATURES}" "${VLA_CKPT}" <<'PY'
import json
from pathlib import Path
import sys

state = json.loads((Path(sys.argv[1]).parent / "cache_state.json").read_text())
if Path(state["policy_dir"]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit(
        f"PrediMem lower feature/model mismatch: {state['policy_dir']} != {sys.argv[2]}"
    )
print(f"Verified PrediMem self-model lower features: {state['policy_dir']}")
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed: PrediMem 2048D self-head output=${OUTPUT_ROOT}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py" \
  --manifest "${MANIFEST}" \
  --lower-features "${LOWER_FEATURES}" \
  --upper-features "${UPPER_FEATURES}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --variants lower,upper,fusion \
  --hidden-dim 1024 \
  --out-dim 2048 \
  --bank-budget 16 \
  --bank-dtype fp32 \
  --resume \
  --epochs "${HEAD_EPOCHS:-30}" \
  --steps-per-epoch "${HEAD_STEPS_PER_EPOCH:-100}" \
  --batch-size "${HEAD_BATCH_SIZE:-512}"

echo "PrediMem self-model 2048D heads complete: ${OUTPUT_ROOT}"
