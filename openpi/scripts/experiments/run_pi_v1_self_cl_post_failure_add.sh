#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/pi_v1_self_cl_libero10_seed7}"
POST_ROOT="${POST_ROOT:-/path/to/storage/datasets/robotics/LIBERO/derived/pi_v1_self_cl_delayed_failure_add_seed7}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
GPU="${GPU:-0}"
PORT="${PORT:-8210}"
SEED="${SEED:-7}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
ENV_WORKERS="${ENV_WORKERS:-32}"
SHARDS_PER_TASK="${SHARDS_PER_TASK:-4}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
FINAL_ADD="${FINAL_ADD:-1}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"

if [[ "${SMOKE}" == "1" ]]; then
  EPISODES_PER_TASK=1
  ENV_WORKERS=2
  SHARDS_PER_TASK=1
  INFERENCE_BATCH_SIZE=2
  FINAL_ADD=0
fi

if [[ "${SMOKE}" != "1" && "${EPISODES_PER_TASK}" -ne 50 ]]; then
  echo "This protocol fixes 50 episodes per task outside SMOKE=1." >&2
  exit 2
fi
[[ -f "${POLICY_DIR}/model.safetensors" && -f "${TASK_HEAD_CKPT}" ]] || exit 1
[[ -d "${SOURCE_ROOT}/branches/success_only" ]] || { echo "Missing completed Success-only campaign: ${SOURCE_ROOT}" >&2; exit 1; }
if [[ -e "${POST_ROOT}" && "${RESUME}" != "1" ]]; then
  echo "Refusing to mix data into non-empty POST_ROOT: ${POST_ROOT}" >&2
  echo "Use RESUME=1 to continue an interrupted campaign." >&2
  exit 2
fi
mkdir -p "${POST_ROOT}"
exec > >(tee -a "${POST_ROOT}/orchestrator.log") 2>&1
echo "post-failure-add start: source=${SOURCE_ROOT} output=${POST_ROOT} gpu=${GPU} envs=${ENV_WORKERS} batch=${INFERENCE_BATCH_SIZE}"

make_manifest() {
  local last_source_round="$1"
  local manifest="$2"
  "${PYTHON}" - "${SOURCE_ROOT}" "${last_source_round}" "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]) / "branches" / "success_only"
last_round = int(sys.argv[2])
output = Path(sys.argv[3])
rows = []
for round_index in range(1, last_round + 1):
    path = source / f"round_{round_index:02d}" / "manifests" / "all_outcomes.jsonl"
    current = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(current) != 500:
        raise RuntimeError(f"expected 500 outcomes in {path}, found {len(current)}")
    rows.extend(current)
if not rows:
    raise RuntimeError("Delayed-failure manifest must contain at least one completed source round")
with output.open("w", encoding="utf-8") as stream:
    for row in rows:
        row = dict(row)
        row["source_branch"] = "success_only_delayed_failure_add"
        row["source_campaign"] = "pi_v1_self_cl_libero10_seed7"
        row["source_round"] = int(row["source_round"])
        row["admission_reason"] = "success_or_failure_from_past_success_only_rounds"
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

build_bank() {
  local source_round="$1"
  local root="$2"
  local manifest="${root}/manifest.jsonl"
  if [[ "${RESUME}" == "1" && -f "${root}/bank/build_summary.json" ]]; then
    echo "Resume: bank already complete, skipping ${root}"
    return
  fi
  mkdir -p "${root}"
  make_manifest "${source_round}" "${manifest}"
  OPENPI_ROOT="${OPENPI_ROOT}" PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/third_party/libero" \
    "${PYTHON}" scripts/memory/cache_prior_head_features.py \
    --manifest "${manifest}" --output-dir "${root}/features" \
    --policy-dir "${POLICY_DIR}" --config-name pi05_libero --device cuda \
    --batch-size "${FEATURE_BATCH_SIZE}" --min-batch-size 1 --auto-batch \
    2>&1 | tee "${root}/feature_cache.log"
  OPENPI_ROOT="${OPENPI_ROOT}" PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/third_party/libero" \
    "${PYTHON}" scripts/memory/build_cl_memory_bank.py \
    --group "success_only_delayed_failure_r${source_round}" --manifest "${manifest}" \
    --feature-dir "${root}/features" --checkpoint "${TASK_HEAD_CKPT}" \
    --output-dir "${root}/bank" --admission both --device cuda --overwrite \
    2>&1 | tee "${root}/bank_build.log"
}

run_round() {
  local eval_round="$1"
  local bank_root="$2"
  local run_root="${POST_ROOT}/eval_round_${eval_round}"
  if [[ "${RESUME}" == "1" && -f "${run_root}/eval/results.txt" ]] && \
     rg -q "^episodes: 500$" "${run_root}/eval/results.txt"; then
    echo "Resume: evaluation round ${eval_round} already complete, skipping ${run_root}"
    return
  fi
  mkdir -p "${run_root}"
  GPU="${GPU}" PORT="$((PORT + eval_round))" MODE="both" RUN_ROOT="${run_root}" BANK_ROOT="${bank_root}/bank" \
    SEED="${SEED}" EPISODES_PER_TASK="${EPISODES_PER_TASK}" ENV_WORKERS="${ENV_WORKERS}" \
    SHARDS_PER_TASK="${SHARDS_PER_TASK}" INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" EPISODE_DATA_MODE="none" POLICY_DIR="${POLICY_DIR}" \
    TASK_HEAD_CKPT="${TASK_HEAD_CKPT}" OPENPI_PYTHON="${PYTHON}" LIBERO_PYTHON="${LIBERO_PYTHON}" \
    ENVIRONMENT_ID_PREFIX="delayed-failure-r${eval_round}" SMOKE="${SMOKE}" \
    bash scripts/eval/run_pi_v1_self_cl_round.sh
}

# Evaluation round r sees only source rounds 1..r-1. Source round r is never
# available to its own evaluation, preserving the temporal information boundary.
END_EVAL_ROUND=10
if [[ "${SMOKE}" == "1" ]]; then END_EVAL_ROUND=2; fi
for eval_round in $(seq 2 "${END_EVAL_ROUND}"); do
  source_last_round=$((eval_round - 1))
  bank_root="${POST_ROOT}/bank_before_round_${eval_round}"
  build_bank "${source_last_round}" "${bank_root}"
  run_round "${eval_round}" "${bank_root}"
done

if [[ "${FINAL_ADD}" == "1" ]]; then
  final_root="${POST_ROOT}/final_add_all_rounds"
  build_bank 10 "${final_root}"
  run_round 11 "${final_root}"
fi

echo "Delayed failure-add validation complete: ${POST_ROOT}"
