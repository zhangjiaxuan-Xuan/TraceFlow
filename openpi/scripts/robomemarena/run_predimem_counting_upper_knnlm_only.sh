#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ROOT}/../RoboMemArena"
LOCK="${ROOT}/configs/robomemarena/predimem_public_checkpoint_updated_benchmark_v1.json"
OPENPI_PY="/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python"
AOSS_ROOT="/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32"
VLA_CKPT="/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch"

UPPER_HEAD="${AOSS_ROOT}/heads/upper/best.pt"
UPPER_MANIFEST="${AOSS_ROOT}/features/anchors_stride5.jsonl"
UPPER_FEATURES="${AOSS_ROOT}/features/upper_dense5_held/upper_features.npy"

cd "${ROOT}"
"${OPENPI_PY}" -m optimus_eval.verify_predimem_frozen_protocol \
  --lock "${LOCK}" --arena-root "${ARENA_ROOT}"

"${OPENPI_PY}" - "${UPPER_HEAD}" "${UPPER_MANIFEST}" "${UPPER_FEATURES}" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np
import torch

head_path, manifest_path, feature_path = map(Path, sys.argv[1:])
for path in (head_path, manifest_path, feature_path):
    if not path.is_file():
        raise FileNotFoundError(path)

checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
if checkpoint.get("variant") != "upper":
    raise RuntimeError(f"Expected Tdense5 Upper head, got {checkpoint.get('variant')!r}")

features = np.load(feature_path, mmap_mode="r")
with manifest_path.open("r", encoding="utf-8") as stream:
    first = json.loads(next(line for line in stream if line.strip()))
if features.ndim != 2 or int(first["row_index"]) >= features.shape[0]:
    raise RuntimeError(
        f"Tdense5 manifest/features mismatch: first_row={first.get('row_index')} features={features.shape}"
    )
print(
    "Verified Upper-only Tdense5 contract: "
    f"variant=upper hidden={checkpoint['hidden']} out={checkpoint['out_dim']} "
    f"features={features.shape} manifest={manifest_path.name}"
)
PY

# The Lower policy is the frozen public PrediMem base. No action memory path is
# loaded; Upper token guidance is the only treatment relative to the baseline.
export MODE="base"
export TASK_IDS="${TASK_IDS:-6,7,8,9,10,15,16}"
export HEAD_VARIANTS="base"
export MEMORY_ADMISSION="none"
export MEMORY_ALLOWED_TASK_IDS=""
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-51}"
export SEED="${SEED:-50}"
export REPLAN_STEPS="10"
export MAX_STEPS="${MAX_STEPS:-2500}"
export NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
export POST_GOAL_STEPS="${POST_GOAL_STEPS:-200}"
export VLM_INTERVAL="5"
export N_RECENT="5"
export K_MAX="0"
export D_MERGE="6"
export UPPER_GPU="${UPPER_GPU:-0}"
export LOWER_GPU="${LOWER_GPU:-1}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export RECORD_MEMORY_DATA="0"
export MEMORY_TRACE_LEVEL="light"
export PREDIMEM_COMPONENT_TIMING="1"

export AOSS_ROOT
export VLA_CKPT
export VLA_CONFIG="pi05_robomemarena_all26_reactive"
export ACTION_STATS="${VLA_CKPT}/assets/robomemarena/all26_pi05_reactive/norm_stats.json"
export VLM_CKPT="/path/to/local/data/robomemarena/models/PrediMem/vlm_tasks1to26_ckpt74500"
export OPENPI_DATA_HOME="/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data"
export MEMORY_ALIGNMENT_TAG="frozen-updated-benchmark-v1+upper-knnlm-tdense5-only"
export PROTOCOL_LOCK_FILE="${LOCK}"

export UPPER_GUIDANCE_ENABLED="1"
export UPPER_GUIDANCE_HEAD="${UPPER_HEAD}"
export UPPER_GUIDANCE_MANIFEST="${UPPER_MANIFEST}"
export UPPER_GUIDANCE_FEATURES="${UPPER_FEATURES}"
export UPPER_GUIDANCE_TOP_K="${UPPER_GUIDANCE_TOP_K:-16}"
export UPPER_GUIDANCE_TEMPERATURE="${UPPER_GUIDANCE_TEMPERATURE:-0.07}"
export UPPER_GUIDANCE_MIN_TASK_PURITY="${UPPER_GUIDANCE_MIN_TASK_PURITY:-0.5}"
export UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE="${UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE:-0.8}"
export UPPER_GUIDANCE_INTERPOLATION="${UPPER_GUIDANCE_INTERPOLATION:-0.8}"
export UPPER_GUIDANCE_BANK_PER_PRIMITIVE="${UPPER_GUIDANCE_BANK_PER_PRIMITIVE:-16}"
export UPPER_GUIDANCE_BANK_SEED="${UPPER_GUIDANCE_BANK_SEED:-17}"
export UPPER_GUIDANCE_DEVICE="${UPPER_GUIDANCE_DEVICE:-cpu}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
export RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_upper_knnlm_only/counting_tdense5_upper_only_seed${SEED}_${STAMP}}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" base
