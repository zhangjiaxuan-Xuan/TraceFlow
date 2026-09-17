#!/usr/bin/env bash
set -euo pipefail

# Retrain only the deployable Full26 Fusion retrieval head. Producer feature
# caches remain read-only; all new checkpoints and banks use a separate root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-2048}"
HEAD_OUT_DIM="${HEAD_OUT_DIM:-4096}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5_balanced_v2_4096/predimem_4096_fp32}"
PYTHON="${PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"

MANIFEST="${SOURCE_ROOT}/features/anchors_stride5.jsonl"
LOWER_DIR="${SOURCE_ROOT}/features/lower_tdense5"
UPPER_DIR="${SOURCE_ROOT}/features/upper_dense5_held"

for path in \
  "${PYTHON}" "${MANIFEST}" \
  "${LOWER_DIR}/pooled_prefix.npy" "${LOWER_DIR}/cache_state.json" \
  "${UPPER_DIR}/upper_features.npy" "${UPPER_DIR}/upper_age.npy" \
  "${UPPER_DIR}/subtasks.jsonl" "${UPPER_DIR}/upper_cache_state.json"; do
  [[ -f "${path}" ]] || { echo "Missing required Full26 cache artifact: ${path}" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  "${PYTHON}" - "${MANIFEST}" "${LOWER_DIR}/cache_state.json" "${UPPER_DIR}/upper_cache_state.json" <<'PY'
import json
import pathlib
import sys

manifest, lower_state_path, upper_state_path = map(pathlib.Path, sys.argv[1:])
lower = json.loads(lower_state_path.read_text())
upper = json.loads(upper_state_path.read_text())
assert pathlib.Path(lower["manifest"]).resolve() == manifest.resolve(), lower["manifest"]
assert pathlib.Path(upper["manifest"]).resolve() == manifest.resolve(), upper["manifest"]
assert lower["manifest_sha256"] == upper["manifest_sha256"]
assert int(lower["rows"]) == int(upper["rows"]) == 563091
assert lower["conditioning_protocol"] == "upper_generated_subtask_v1"
assert lower["extraction_protocol"] == "cl_prefix_temporal_cat_v1"
assert int(lower["temporal_window"]) == 3
assert bool(upper["complete"])
print(f"Balanced Full26 preflight passed: rows={lower['rows']} manifest={lower['manifest_sha256']}")
PY
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/logs"
exec > >(tee -a "${OUTPUT_ROOT}/logs/train_balanced_fusion.log") 2>&1

"${PYTHON}" - "${MANIFEST}" "${LOWER_DIR}/cache_state.json" "${UPPER_DIR}/upper_cache_state.json" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest, lower_state_path, upper_state_path = map(pathlib.Path, sys.argv[1:])
digest = hashlib.sha256()
with manifest.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
expected = digest.hexdigest()
lower = json.loads(lower_state_path.read_text())
upper = json.loads(upper_state_path.read_text())
assert lower["manifest_sha256"] == expected, (lower["manifest_sha256"], expected)
assert lower["conditioning_protocol"] == "upper_generated_subtask_v1", lower
assert lower["extraction_protocol"] == "cl_prefix_temporal_cat_v1", lower
assert int(lower["temporal_window"]) == 3, lower
assert upper["manifest_sha256"] == expected, (upper["manifest_sha256"], expected)
assert bool(upper["complete"]), upper
print(f"Validated immutable Full26 producer cache: rows={lower['rows']} manifest={expected}", flush=True)
PY

"${PYTHON}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py" \
  --manifest "${MANIFEST}" \
  --lower-features "${LOWER_DIR}/pooled_prefix.npy" \
  --upper-features "${UPPER_DIR}/upper_features.npy" \
  --upper-age "${UPPER_DIR}/upper_age.npy" \
  --output-root "${OUTPUT_ROOT}" --device cuda:0 --variants fusion \
  --hidden-dim "${HEAD_HIDDEN_DIM}" --out-dim "${HEAD_OUT_DIM}" --bank-dtype fp32 \
  --sampling-strategy task_stage_progress_pairs \
  --progress-coordinate stage --positive-progress-radius 0.025 \
  --progress-recall-tolerance 0.04 --require-cross-trajectory-positive \
  --temporal-regression-weight 0.5 --temporal-regression-scale 0.04 \
  --deploy-all-anchors --skip-compression-study --require-joint-conditioning --resume \
  --retrieval-device cuda:0 \
  --retrieval-query-batch-size "${RETRIEVAL_QUERY_BATCH_SIZE:-4096}" \
  --projection-batch-size "${PROJECTION_BATCH_SIZE:-8192}" \
  --epochs "${EPOCHS:-12}" --steps-per-epoch "${STEPS_PER_EPOCH:-200}" \
  --batch-size "${HEAD_BATCH_SIZE:-1024}" --seed "${HEAD_SEED:-7}"

"${PYTHON}" "${ROOT}/scripts/robomemarena/build_anchor_aligned_memory_meta.py" \
  --manifest "${MANIFEST}" \
  --source-meta "${OUTPUT_ROOT}/memory/fusion/gpm_memory_meta.pt" \
  --output-meta "${OUTPUT_ROOT}/memory/fusion/gpm_memory_meta_dense_frame_v3.pt" \
  --protocol dense_frame_v3 --overwrite

echo "Balanced Full26 Fusion head complete: ${OUTPUT_ROOT}"
