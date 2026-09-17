#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SEED="${SEED:-7}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/libero130_guidance_v0_basic6500_batch8_${RUN_ID}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_EPISODE_DATA="${SAVE_EPISODE_DATA:-0}"
RESUME="${RESUME:-0}"

POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
MEMORY_META_PATH="${MEMORY_META_PATH:-${OPENPI_ROOT}/memory/gpm_memory_meta.pt}"
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${OPENPI_ROOT}/memory/gpm_memory.index}"
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-${OPENPI_ROOT}/memory/gpm_memory_actions.npz}"

SUITES=(libero_spatial libero_object libero_goal libero_10 libero_90)
declare -A SUITE_TASK_COUNTS=(
  [libero_spatial]=10
  [libero_object]=10
  [libero_goal]=10
  [libero_10]=10
  [libero_90]=90
)

is_true() {
  case "${1}" in
    1 | true | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${BATCH_SIZE}" -ne 8 ]]; then
  echo "This entrypoint requires BATCH_SIZE=8." >&2
  exit 2
fi
if [[ "${NUM_TRIALS_PER_TASK}" -lt "${BATCH_SIZE}" ]]; then
  echo "NUM_TRIALS_PER_TASK must be at least ${BATCH_SIZE}." >&2
  exit 2
fi
if [[ ! "${NUM_TRIALS_PER_TASK}" =~ ^[0-9]+$ || "${NUM_TRIALS_PER_TASK}" -le 0 ]]; then
  echo "NUM_TRIALS_PER_TASK must be a positive integer." >&2
  exit 2
fi

required_files=(
  "${POLICY_DIR}/model.safetensors"
  "${TASK_HEAD_CKPT}"
  "${MEMORY_META_PATH}"
  "${FAISS_INDEX_PATH}"
  "${MEMORY_ACTIONS_PATH}"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done
if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "Missing OpenPI Python: ${OPENPI_PYTHON}" >&2
  exit 1
fi

memory_items="$("${OPENPI_PYTHON}" - "${MEMORY_META_PATH}" <<'PY'
import sys
import torch

metadata = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(len(metadata))
PY
)"
if [[ "${memory_items}" -ne 6500 ]]; then
  echo "basic6500 requires exactly 6500 memory trajectories, found ${memory_items}." >&2
  exit 1
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed: V0 basic6500, suites=${SUITES[*]}, tasks=130, episodes_per_task=${NUM_TRIALS_PER_TASK}, batch=8"
  exit 0
fi

if is_true "${RESUME}"; then
  if [[ ! -f "${RUN_ROOT}/run_config.json" ]]; then
    if [[ -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
      echo "RESUME=1 found non-empty data without run_config.json: ${RUN_ROOT}" >&2
      exit 1
    fi
    echo "RESUME=1 requested for an empty run root; starting a fresh run: ${RUN_ROOT}"
    RESUME=0
  fi
elif [[ -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to mix data into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/indexes" "${RUN_ROOT}/suites"
"${OPENPI_PYTHON}" - \
  "${RUN_ROOT}/run_config.json" "${POLICY_DIR}" "${TASK_HEAD_CKPT}" \
  "${MEMORY_META_PATH}" "${FAISS_INDEX_PATH}" "${MEMORY_ACTIONS_PATH}" \
  "${SEED}" "${NUM_TRIALS_PER_TASK}" "${SAVE_VIDEOS}" "${SAVE_EPISODE_DATA}" "${RESUME}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = {
    "benchmark": "LIBERO-130",
    "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    "suite_task_counts": [10, 10, 10, 10, 90],
    "total_tasks": 130,
    "reporting_groups": {"libero_long": ["libero_10", "libero_90"]},
    "runtime": "guidance_v0_full_interval_direct_unbounded",
    "memory_variant": "basic6500_success",
    "memory_items": 6500,
    "batch_size": 8,
    "policy_dir": sys.argv[2],
    "task_head_ckpt": sys.argv[3],
    "memory_meta_path": sys.argv[4],
    "faiss_index_path": sys.argv[5],
    "memory_actions_path": sys.argv[6],
    "seed": int(sys.argv[7]),
    "episodes_per_task": int(sys.argv[8]),
    "save_videos": sys.argv[9] == "1",
    "save_episode_data": sys.argv[10] == "1",
}
if path.exists():
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit(f"Resume configuration mismatch:\nexpected={expected}\nactual={actual}")
else:
    path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
PY

validate_args=()
if is_true "${SAVE_VIDEOS}"; then validate_args+=(--require-videos); fi
if is_true "${SAVE_EPISODE_DATA}"; then validate_args+=(--require-trajectories); fi

for suite in "${SUITES[@]}"; do
  task_count="${SUITE_TASK_COUNTS[${suite}]}"
  task_ids_csv="$(seq -s, 0 $((task_count - 1)))"
  suite_root="${RUN_ROOT}/suites/${suite}"
  eval_log="${suite_root}/eval/${suite}.jsonl"
  suite_resume=0

  if [[ -f "${eval_log}" ]]; then
    suite_resume=1
    if "${OPENPI_PYTHON}" scripts/analysis/summarize_libero_eval_run.py \
      --eval-log "${eval_log}" \
      --output-dir "${suite_root}/indexes" \
      --expected-tasks "${task_count}" \
      --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
      "${validate_args[@]}" >/dev/null 2>&1; then
      echo "Suite already complete; skipping ${suite}"
      continue
    fi
    if ! is_true "${RESUME}"; then
      echo "Found partial suite data but RESUME is not enabled: ${suite_root}" >&2
      exit 1
    fi
  fi

  episode_data_root=""
  if is_true "${SAVE_EPISODE_DATA}"; then
    episode_data_root="${RUN_ROOT}/episode_data"
  fi

  echo "Starting ${suite}: tasks=${task_count} episodes_per_task=${NUM_TRIALS_PER_TASK} resume=${suite_resume}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  SERVER_CUDA_VISIBLE_DEVICES="${GPU}" \
  CLIENT_CUDA_VISIBLE_DEVICES="${GPU}" \
  MUJOCO_EGL_DEVICE_ID=0 \
  LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
  NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" \
  MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
  PORT="${PORT}" \
  POLICY_DIR="${POLICY_DIR}" \
  TASK_HEAD_CKPT="${TASK_HEAD_CKPT}" \
  MEMORY_META_PATH="${MEMORY_META_PATH}" \
  FAISS_INDEX_PATH="${FAISS_INDEX_PATH}" \
  MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH}" \
  LOG_DIR="${suite_root}/eval" \
  RESULTS_TXT="${suite_root}/eval/results.txt" \
  VIDEO_ROOT="${RUN_ROOT}/videos" \
  EPISODE_DATA_ROOT="${episode_data_root}" \
  EPISODE_DATA_MODE=all \
  TASK_IDS_CSV="${task_ids_csv}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
  SEED="${SEED}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" \
  RESUME="${suite_resume}" \
  RUN_ID="${RUN_ID}_${suite}" \
  INFERENCE_BATCH_SIZE=8 \
  INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-20}" \
  LIBERO_CLIENTS_PER_SUITE=8 \
  USE_MEMORY=1 \
  USE_LCM=0 \
  USE_MEMORY_GUIDANCE=0 \
  MEMORY_GUIDANCE_ONLY=1 \
  MEMORY_GUIDANCE_TIME_VERSION=v0 \
  MEMORY_PRIOR_SUBSTEP_GUIDANCE=0 \
  MEMORY_GUIDANCE_NUM_STEPS=10 \
  MEMORY_GUIDANCE_LAMBDA_MAX="${MEMORY_GUIDANCE_LAMBDA_MAX:-0.20}" \
  MEMORY_GUIDANCE_T_CUT="${MEMORY_GUIDANCE_T_CUT:-0.30}" \
  MEMORY_GUIDANCE_SIGMA="${MEMORY_GUIDANCE_SIGMA:-0.30}" \
  MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}" \
  MEMORY_GUIDANCE_MIN_SIMILARITY="${MEMORY_GUIDANCE_MIN_SIMILARITY:--1.0}" \
  USE_NEGATIVE_GUIDANCE=0 \
    bash scripts/eval/run_libero_eval.sh "${suite}"

  "${OPENPI_PYTHON}" scripts/analysis/summarize_libero_eval_run.py \
    --eval-log "${eval_log}" \
    --output-dir "${suite_root}/indexes" \
    --expected-tasks "${task_count}" \
    --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
    "${validate_args[@]}" >/dev/null
  echo "Finished ${suite}"
  sleep 2
done

"${OPENPI_PYTHON}" - "${RUN_ROOT}" "${NUM_TRIALS_PER_TASK}" <<'PY'
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
episodes_per_task = int(sys.argv[2])
suites = [
    ("libero_spatial", 10),
    ("libero_object", 10),
    ("libero_goal", 10),
    ("libero_10", 10),
    ("libero_90", 90),
]
suite_rows = []
task_rows = []
total_episodes = 0
total_successes = 0
for suite, expected_tasks in suites:
    summary_path = root / "suites" / suite / "indexes" / "run_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing suite summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["tasks"]) != expected_tasks:
        raise SystemExit(f"{suite}: expected {expected_tasks} tasks, found {summary['tasks']}")
    expected_episodes = expected_tasks * episodes_per_task
    if int(summary["episodes"]) != expected_episodes:
        raise SystemExit(f"{suite}: expected {expected_episodes} episodes, found {summary['episodes']}")
    episodes = int(summary["episodes"])
    successes = int(summary["successes"])
    suite_rows.append({
        "suite": suite,
        "tasks": expected_tasks,
        "episodes": episodes,
        "successes": successes,
        "failures": episodes - successes,
        "success_rate": successes / episodes,
    })
    for row in summary["task_summary"]:
        task_rows.append({"suite": suite, **row})
    total_episodes += episodes
    total_successes += successes

if len(task_rows) != 130:
    raise SystemExit(f"Expected 130 task rows, found {len(task_rows)}")
expected_total = 130 * episodes_per_task
if total_episodes != expected_total:
    raise SystemExit(f"Expected {expected_total} total episodes, found {total_episodes}")

suite_by_name = {row["suite"]: row for row in suite_rows}
long_components = [suite_by_name["libero_10"], suite_by_name["libero_90"]]
long_episodes = sum(int(row["episodes"]) for row in long_components)
long_successes = sum(int(row["successes"]) for row in long_components)
long_summary = {
    "scope": "libero_long",
    "components": ["libero_10", "libero_90"],
    "tasks": 100,
    "episodes": long_episodes,
    "successes": long_successes,
    "failures": long_episodes - long_successes,
    "success_rate": long_successes / long_episodes,
}
if long_episodes != 100 * episodes_per_task:
    raise SystemExit(f"Expected {100 * episodes_per_task} LIBERO-LONG episodes, found {long_episodes}")

index_dir = root / "indexes"
index_dir.mkdir(parents=True, exist_ok=True)
with (index_dir / "suite_summary.tsv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(suite_rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(suite_rows)
with (index_dir / "task_summary.tsv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(task_rows)
for suite in ("libero_10", "libero_90"):
    rows = [row for row in task_rows if row["suite"] == suite]
    with (index_dir / f"{suite}_task_summary.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

long_rows = [
    {
        "scope": row["suite"],
        "tasks": row["tasks"],
        "episodes": row["episodes"],
        "successes": row["successes"],
        "failures": row["failures"],
        "success_rate": row["success_rate"],
    }
    for row in long_components
]
long_rows.append({key: value for key, value in long_summary.items() if key != "components"})
with (index_dir / "long_summary.tsv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(long_rows)

aggregate = {
    "benchmark": "LIBERO-130",
    "tasks": 130,
    "episodes_per_task": episodes_per_task,
    "episodes": total_episodes,
    "successes": total_successes,
    "failures": total_episodes - total_successes,
    "success_rate": total_successes / total_episodes,
    "suite_summary": suite_rows,
    "long_summary": long_summary,
}
(index_dir / "run_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
with (root / "results.txt").open("w", encoding="utf-8") as handle:
    handle.write("OptimusVLA V0 basic6500 LIBERO-130 Evaluation\n\n")
    handle.write("suite\ttasks\tepisodes\tsuccesses\tsuccess_rate\n")
    for row in suite_rows:
        handle.write(
            f"{row['suite']}\t{row['tasks']}\t{row['episodes']}\t"
            f"{row['successes']}\t{row['success_rate']:.4f}\n"
        )
    handle.write(
        f"libero_long\t{long_summary['tasks']}\t{long_summary['episodes']}\t"
        f"{long_summary['successes']}\t{long_summary['success_rate']:.4f}\n"
    )
    handle.write(
        f"\noverall_success_rate: {aggregate['success_rate']:.4f} "
        f"({total_successes}/{total_episodes})\n"
    )
print(json.dumps(aggregate, indent=2))
PY

echo "LIBERO-130 V0 basic6500 evaluation complete: ${RUN_ROOT}"
echo "Overall results: ${RUN_ROOT}/results.txt"
echo "Suite summary: ${RUN_ROOT}/indexes/suite_summary.tsv"
echo "Task summary: ${RUN_ROOT}/indexes/task_summary.tsv"
echo "LIBERO-LONG summary: ${RUN_ROOT}/indexes/long_summary.tsv"
echo "LIBERO-10 tasks: ${RUN_ROOT}/indexes/libero_10_task_summary.tsv"
echo "LIBERO-90 tasks: ${RUN_ROOT}/indexes/libero_90_task_summary.tsv"
