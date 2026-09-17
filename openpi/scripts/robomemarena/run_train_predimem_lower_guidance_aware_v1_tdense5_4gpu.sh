#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32}"
CACHE_ROOT="/path/to/storage/datasets/robotics/RoboMemArena/derived/guidance_aware_v1/predimem_full26_lower_tdense5_topk8"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/path/to/storage/datasets/robotics/RoboMemArena/derived/lerobot}"
DATASET_ROOT="${HF_LEROBOT_HOME}/robomemarena/all26_pi05_subtask"
DATASET_COMPLETE="${HF_LEROBOT_HOME}/robomemarena/all26_pi05_subtask_conversion_summary.json"
CONFIG="predimem_all26_subtask_guidance_aware_v1_full"
EXP_NAME="${EXP_NAME:-lower_fullft_v1_tdense5_topk8_official_batch128_seed42}"
CHECKPOINT_ROOT="/path/to/storage/datasets/robotics/RoboMemArena/derived/guidance_aware_v2/checkpoints/${CONFIG}/${EXP_NAME}"
BASE_JAX="${BASE_JAX:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_base}"
BASE_PT="/path/to/storage/datasets/robotics/RoboMemArena/derived/guidance_aware_v2/checkpoints/pi05_base_pytorch"

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#gpu_ids[@]}" -ne 4 ]] || [[ "$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "Expected four distinct CUDA devices, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -f "${BASE_JAX}/params/_METADATA" ]] || { echo "Missing official pi05_base: ${BASE_JAX}" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES HF_LEROBOT_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/local/data/huggingface/datasets_cache/robomemarena_all26_subtask}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${SELF_ROOT}/runtime/openpi_data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${CACHE_ROOT}/launcher_logs"
LAUNCH_LOG="${CACHE_ROOT}/launcher_logs/launch_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
echo "Launcher log: ${LAUNCH_LOG}"
on_exit() {
  status=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launcher exit status=${status}"
}
trap on_exit EXIT

MANIFEST="${SELF_ROOT}/features/anchors_stride5.jsonl"
LOWER_FEATURES="${SELF_ROOT}/features/lower_tdense5/pooled_prefix.npy"
UPPER_FEATURES="${SELF_ROOT}/features/upper_dense5_held/upper_features.npy"
UPPER_AGE="${SELF_ROOT}/features/upper_dense5_held/upper_age.npy"
HEAD="${SELF_ROOT}/heads/lower/best.pt"
INDEX="${SELF_ROOT}/memory/lower/gpm_memory.index"
ACTIONS="${SELF_ROOT}/memory/shared/gpm_memory_actions.npz"
for path in "${MANIFEST}" "${LOWER_FEATURES}" "${UPPER_FEATURES}" "${UPPER_AGE}" "${HEAD}" "${INDEX}" "${ACTIONS}"; do
  [[ -f "${path}" ]] || { echo "Missing required Tdense5 artifact: ${path}" >&2; exit 2; }
done

"${PYTHON}" - "${MANIFEST}" "${LOWER_FEATURES}" "${UPPER_FEATURES}" "${UPPER_AGE}" "${HEAD}" <<'PY'
from pathlib import Path
import sys

import numpy as np
import torch

manifest, lower_path, upper_path, age_path, head_path = map(Path, sys.argv[1:])
expected_rows = 563_091
expected_shapes = {
    lower_path: (expected_rows, 6_144),
    upper_path: (expected_rows, 4_096),
    age_path: (expected_rows,),
}
for path, expected in expected_shapes.items():
    actual = np.load(path, mmap_mode="r", allow_pickle=False).shape
    if actual != expected:
        raise SystemExit(f"Full26 feature mismatch: {path} has {actual}, expected {expected}")
head = torch.load(head_path, map_location="cpu", weights_only=False)
spec = (head.get("variant"), head.get("lower_dim"), head.get("hidden"), head.get("out_dim"))
if spec != ("lower", 6_144, 1_024, 2_048):
    raise SystemExit(f"Expected default Lower 2048D retrieval head, got {spec}")
print("Verified full26 producer: 563091 Tdense5 anchors, default Lower 2048D head")
PY

# Independent preparation jobs run concurrently. Formal DDP starts only after
# all durable artifacts pass their completion checks.
prep_pids=()
prep_names=()
if [[ ! -f "${BASE_PT}/model.safetensors" ]]; then
  (
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${PYTHON}" \
      "${ROOT}/examples/convert_jax_model_to_pytorch.py" \
      --checkpoint-dir "${BASE_JAX}" --config-name "${CONFIG}" \
      --output-path "${BASE_PT}" --precision bfloat16
  ) &
  prep_pids+=("$!"); prep_names+=("pi05-base-conversion")
fi
if [[ ! -f "${CACHE_ROOT}/manifest.json" ]]; then
  (
    CUDA_VISIBLE_DEVICES="${gpu_ids[1]}" "${PYTHON}" \
      "${ROOT}/scripts/robomemarena/build_guidance_training_cache.py" \
      --manifest "${MANIFEST}" --lower-features "${LOWER_FEATURES}" \
      --upper-features "${UPPER_FEATURES}" --upper-age "${UPPER_AGE}" \
      --head "${HEAD}" --index "${INDEX}" --actions "${ACTIONS}" \
      --output "${CACHE_ROOT}" --top-k 8 --search-k "${CACHE_SEARCH_K:-2048}" \
      --batch-size "${CACHE_BUILD_BATCH_SIZE:-8192}" --device cuda:0
  ) &
  prep_pids+=("$!"); prep_names+=("tdense5-guidance-cache")
fi
if [[ ! -f "${DATASET_COMPLETE}" ]]; then
  if [[ -d "${DATASET_ROOT}" ]]; then
    stale="${DATASET_ROOT}.incomplete.$(date -u +%Y%m%d_%H%M%S)"
    mv "${DATASET_ROOT}" "${stale}"
    echo "Moved incomplete dataset aside: ${stale}"
  fi
  (
    "${PYTHON}" "${ROOT}/scripts/robomemarena/convert_robomemarena_subtasks_to_lerobot.py" \
      --lerobot-home "${HF_LEROBOT_HOME}" \
      --read-workers "${DATA_READ_WORKERS:-32}" \
      --image-writer-threads "${IMAGE_WRITER_THREADS:-32}" \
      --image-writer-processes "${IMAGE_WRITER_PROCESSES:-8}"
  ) &
  prep_pids+=("$!"); prep_names+=("subtask-dataset")
fi

prep_failed=0
for index in "${!prep_pids[@]}"; do
  echo "Waiting for ${prep_names[index]} pid=${prep_pids[index]}"
  if ! wait "${prep_pids[index]}"; then
    echo "Preparation failed: ${prep_names[index]}" >&2
    prep_failed=1
  else
    echo "Preparation completed: ${prep_names[index]}"
  fi
done
(( prep_failed == 0 )) || exit 1

[[ -f "${BASE_PT}/model.safetensors" ]] || { echo "Base conversion incomplete: ${BASE_PT}" >&2; exit 2; }
[[ -f "${CACHE_ROOT}/manifest.json" ]] || { echo "Guidance cache incomplete: ${CACHE_ROOT}" >&2; exit 2; }
[[ -f "${DATASET_COMPLETE}" ]] || { echo "Dataset conversion incomplete: ${DATASET_ROOT}" >&2; exit 2; }

# A completed marker alone is insufficient: an older Extra8 cache has the
# same file layout but covers only episodes 0..799. Freeze the full26 cache
# provenance and cardinality before any expensive model initialization.
"${PYTHON}" - "${CACHE_ROOT}" "${MANIFEST}" "${HEAD}" "${INDEX}" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

cache, source_manifest, head, index = map(Path, sys.argv[1:])
payload = json.loads((cache / "manifest.json").read_text())
expected = {
    "rows": 563_091,
    "source_rows": 563_091,
    "top_k": 8,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"Guidance cache mismatch: {key}={payload.get(key)!r}, expected {value!r}")
for key, path in (("source_manifest", source_manifest), ("head", head), ("index", index)):
    if Path(payload.get(key, "")).resolve() != path.resolve():
        raise SystemExit(f"Guidance cache provenance mismatch: {key}={payload.get(key)!r}, expected {path}")
episodes = np.load(cache / "episode_index.npy", mmap_mode="r", allow_pickle=False)
frames = np.load(cache / "frame_index.npy", mmap_mode="r", allow_pickle=False)
if len(episodes) != expected["rows"] or len(frames) != expected["rows"]:
    raise SystemExit("Guidance cache address arrays have the wrong row count")
unique = np.unique(episodes)
if len(unique) != 2_600 or int(unique[0]) != 0 or int(unique[-1]) != 2_599:
    raise SystemExit(
        f"Guidance cache must cover episodes 0..2599 exactly; got count={len(unique)} "
        f"range={int(unique[0]) if len(unique) else None}..{int(unique[-1]) if len(unique) else None}"
    )
print("Verified full26 Guidance cache: 563091 anchors, 2600 episodes, Lower 2048D, TopK8")
PY

# Build the expensive Parquet -> Arrow cache once. Without this gate every DDP
# rank can independently scan all 2600 episodes before the first train step.
ARROW_MARKER="${DATASET_ROOT}/meta/arrow_cache_ready.json"
if ! "${PYTHON}" - "${ARROW_MARKER}" "${HF_DATASETS_CACHE}" <<'PY'
import json
from pathlib import Path
import sys

marker = Path(sys.argv[1])
cache = Path(sys.argv[2]).resolve()
if not marker.is_file() or not cache.is_dir():
    raise SystemExit(1)
payload = json.loads(marker.read_text())
if (
    payload.get("rows") != 2_800_257
    or payload.get("episodes") != 2_600
    or Path(payload.get("cache_dir", "")).resolve() != cache
    or not any(cache.rglob("*.arrow"))
):
    raise SystemExit(1)
print(f"Using persistent Arrow cache: {cache}")
PY
then
  mkdir -p "${HF_DATASETS_CACHE}"
  "${PYTHON}" "${ROOT}/scripts/data/prewarm_lerobot_parquet_cache.py" \
    --dataset-root "${DATASET_ROOT}" --marker "${ARROW_MARKER}" \
    --expected-rows 2800257 --expected-episodes 2600 \
    --workers "${ARROW_CACHE_WORKERS:-64}"
fi

# Bypass HuggingFace's unstable data_dir fingerprint on object-storage
# listings and directly mmap one validated, complete Arrow split.
export OPENPI_LEROBOT_ARROW_DIR="${OPENPI_LEROBOT_ARROW_DIR:-$("${PYTHON}" - "${HF_DATASETS_CACHE}" <<'PY'
import json
from pathlib import Path
import sys

matches = []
for info_path in Path(sys.argv[1]).rglob("dataset_info.json"):
    info = json.loads(info_path.read_text())
    if info.get("splits", {}).get("train", {}).get("num_examples") != 2_800_257:
        continue
    arrow_files = sorted(info_path.parent.glob("*.arrow"))
    if arrow_files:
        matches.append((len(arrow_files), info_path.parent))
if not matches:
    raise SystemExit("No complete 2,800,257-row Arrow cache found")
matches.sort(key=lambda item: (-item[0], str(item[1])))
print(matches[0][1])
PY
)}"
echo "Using direct Arrow dataset: ${OPENPI_LEROBOT_ARROW_DIR}"

NORM_STATS="${SELF_ROOT}/../../predimem_dual_tower/checkpoints/vla_alltask_pytorch/assets/robomemarena/all26_pi05_reactive/norm_stats.json"
NORM_STATS="$(realpath -m "${NORM_STATS}")"
[[ -f "${NORM_STATS}" ]] || { echo "Missing official all26 normalization stats: ${NORM_STATS}" >&2; exit 2; }

"${PYTHON}" - "${CONFIG}" "${CACHE_ROOT}" "${BASE_PT}" <<'PY'
import json
from pathlib import Path
import sys
from openpi.training.config import get_config

config = get_config(sys.argv[1])
cache = Path(sys.argv[2])
base = Path(sys.argv[3])
data_config = config.data.create(config.assets_dirs, config.model)
expected = {
    "batch_size": 128,
    "gradient_accumulation_steps": 16,
    "num_workers": 0,
    "num_train_steps": 40_000,
    "save_interval": 5_000,
    "ema_decay": 0.999,
}
for field, value in expected.items():
    actual = getattr(config, field)
    if actual != value:
        raise SystemExit(f"Official protocol mismatch: {field}={actual!r}, expected {value!r}")
guidance = config.guidance_aware
if not guidance.enabled or guidance.training_mode != "full_model" or guidance.version != "v1":
    raise SystemExit(f"Invalid guidance-aware mode: {guidance}")
if guidance.norm_cap != 0.2 or guidance.cache_path != str(cache):
    raise SystemExit("Lower Guidance must be fixed V1/Tdense5 cap=0.2")
if Path(config.pytorch_weight_path).resolve() != base.resolve() or not (base / "model.safetensors").is_file():
    raise SystemExit(f"Training must initialize from converted official pi05_base, got {config.pytorch_weight_path}")
if data_config.asset_id != "robomemarena/all26_pi05_reactive" or data_config.norm_stats is None:
    raise SystemExit("Subtask prompts must reuse the official all26 reactive numeric normalization stats")
if data_config.lerobot_root != (
    "/path/to/storage/datasets/robotics/RoboMemArena/derived/lerobot/robomemarena/all26_pi05_subtask"
):
    raise SystemExit(f"Training dataset root is not frozen to the complete AOSS dataset: {data_config.lerobot_root}")
schedule = config.lr_schedule
if (schedule.warmup_steps, schedule.peak_lr, schedule.decay_steps, schedule.decay_lr) != (
    10_000, 5e-5, 1_000_000, 5e-5
):
    raise SystemExit(f"Official LR protocol mismatch: {schedule}")
manifest = json.loads((cache / "manifest.json").read_text())
if manifest["top_k"] != 8 or "heads/lower/best.pt" not in manifest["head"]:
    raise SystemExit(f"Cache is not Lower-only Tdense5 TopK8: {manifest}")
print("Verified: 4 GPUs x micro-batch 2 x accumulation 16 = global batch 128; only Guidance changes the loss")
PY

mode=(--overwrite)
if [[ "${RESUME:-0}" == 1 ]]; then
  [[ -d "${CHECKPOINT_ROOT}" ]] || { echo "Cannot resume missing checkpoint root: ${CHECKPOINT_ROOT}" >&2; exit 2; }
  mode=(--resume)
elif [[ -e "${CHECKPOINT_ROOT}" ]] && [[ -n "$(find "${CHECKPOINT_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Checkpoint root is non-empty; set RESUME=1 or change EXP_NAME: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi

export OPENPI_TRAIN_LOG="${CACHE_ROOT}/train_${EXP_NAME}.log"
export OPENPI_METRICS_LOG="${CACHE_ROOT}/metrics_${EXP_NAME}.jsonl"
export OPENPI_SERIALIZE_DDP_DATASET_INIT="${OPENPI_SERIALIZE_DDP_DATASET_INIT:-1}"
export OPENPI_EARLY_CHECKPOINT_STEP="${OPENPI_EARLY_CHECKPOINT_STEP:-100}"
export ARROW_NUM_THREADS="${ARROW_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="false"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
TORCHRUN_LOG_DIR="${CACHE_ROOT}/torchrun/${EXP_NAME}_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${TORCHRUN_LOG_DIR}"
echo "Per-rank torchrun logs: ${TORCHRUN_LOG_DIR}"
exec "${PYTHON}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  --log-dir "${TORCHRUN_LOG_DIR}" --redirects 3 --tee 3 \
  "${ROOT}/scripts/train_pytorch.py" "${CONFIG}" --exp-name "${EXP_NAME}" "${mode[@]}"
