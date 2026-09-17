#!/usr/bin/env bash
set -euo pipefail

if [[ "${PI_ARENA_RUN_SNAPSHOT:-0}" != "1" ]]; then
  source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  export ROOT="${ROOT:-${source_root}}"
  export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "${source_root}/../RoboMemArena/evaluation_benchmark" && pwd)}"
  snapshot_path="$(mktemp /tmp/pi05_robomemarena_extra8.XXXXXX.sh)"
  cp "$0" "${snapshot_path}"
  chmod +x "${snapshot_path}"
  export PI_ARENA_RUN_SNAPSHOT=1
  exec bash "${snapshot_path}" "$@"
fi

ROOT="${ROOT:?ROOT must be exported by the snapshot launcher}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:?BENCHMARK_ROOT must be exported by the snapshot launcher}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/.venvs/libero/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/pi05_robomemarena_extra8_reactive/extra8_reactive_fullft_seed42_finite_loader/30000_pytorch}"
CONFIG_NAME="${CONFIG_NAME:-pi05_robomemarena_extra8_reactive}"
MEMORY_MODE="${MEMORY_MODE:-none}"
case "${MEMORY_MODE}" in
  none) MEMORY_TAG="" ;;
  v0) MEMORY_TAG="_memory_v0_lower" ;;
  v1) MEMORY_TAG="_memory_v1_lower" ;;
  v3) MEMORY_TAG="_memory_v3_lower" ;;
  v3_prior_only) MEMORY_TAG="_memory_v3_prior_only_lower" ;;
  v3_prior_decay) MEMORY_TAG="_memory_v3_prior_decay_lower" ;;
  v3_1) MEMORY_TAG="_memory_v3_1_lower" ;;
  v3_re) MEMORY_TAG="_memory_v3_re_lower" ;;
  *) echo "MEMORY_MODE must be none, v0, v1, v3, v3_prior_only, v3_prior_decay, v3_1, or v3_re; got ${MEMORY_MODE}" >&2; exit 2 ;;
esac

GPU="${GPU:-0}"
PORT="${PORT:-8700}"
TASK_IDS="${TASK_IDS:-1,2,3,18,19,22,25,26}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
SEED="${SEED:-50}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-100}"
OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-0}"
ENV_WORKERS="${ENV_WORKERS:-32}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_STEPS="${MAX_STEPS:-2500}"
POST_GOAL_STEPS="${POST_GOAL_STEPS:-200}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
RESUME="${RESUME:-0}"
RESUME_ALLOW_ENV_WORKERS_OVERRIDE="${RESUME_ALLOW_ENV_WORKERS_OVERRIDE:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
WORKER_TIMEOUT_SECONDS="${WORKER_TIMEOUT_SECONDS:-172800}"
ENV_INIT_RETRIES="${ENV_INIT_RETRIES:-30}"
AOSS_ROOT="${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${AOSS_ROOT}/heads/lower/best.pt}"
MEMORY_META_PATH="${MEMORY_META_PATH:-${AOSS_ROOT}/memory/lower/gpm_memory_meta.pt}"
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${AOSS_ROOT}/memory/lower/gpm_memory.index}"
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz}"
ACTION_NORM_STATS="${ACTION_NORM_STATS:-${CHECKPOINT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json}"
MEMORY_TOP_K="${MEMORY_TOP_K:-8}"
MEMORY_PROGRESS_WINDOW="${MEMORY_PROGRESS_WINDOW:-0.20}"
MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}"
MEMORY_GUIDANCE_TOTAL_NORM_CAP="${MEMORY_GUIDANCE_TOTAL_NORM_CAP:-${MEMORY_GUIDANCE_NORM_CAP}}"
MEMORY_ACTION_ALIGNMENT="${MEMORY_ACTION_ALIGNMENT:-auto}"
MEMORY_PROVENANCE_TAG="${MEMORY_PROVENANCE_TAG:-lower-cross-checkpoint}"
MEMORY_STATIC_ACTION_MODE="${MEMORY_STATIC_ACTION_MODE:-off}"
MEMORY_STATIC_ACTION_ARM_DIM="${MEMORY_STATIC_ACTION_ARM_DIM:-6}"
MEMORY_STATIC_ACTION_THRESHOLD="${MEMORY_STATIC_ACTION_THRESHOLD:-1e-8}"
MEMORY_STATIC_ACTION_GATE_FRACTION="${MEMORY_STATIC_ACTION_GATE_FRACTION:-0.50}"
case "${MEMORY_STATIC_ACTION_MODE}" in
  off|observe|gate) ;;
  *) echo "MEMORY_STATIC_ACTION_MODE must be off, observe, or gate." >&2; exit 2 ;;
esac
if [[ "${MEMORY_STATIC_ACTION_MODE}" != "off" && "${MEMORY_MODE}" != "v1" ]]; then
  echo "Static-action testing currently requires MEMORY_MODE=v1." >&2
  exit 2
fi
if [[ "${MEMORY_ACTION_ALIGNMENT}" =~ ^(continuous_frame_v2|dense_frame_v3)$ && "${REPLAN_STEPS}" -ne 10 ]]; then
  echo "${MEMORY_ACTION_ALIGNMENT} requires REPLAN_STEPS=10 for temporal offsets [-20,-10,0]." >&2
  exit 2
fi
if [[ "${MEMORY_MODE}" =~ ^(v3_prior_only|v3_prior_decay|v3_1|v3_re)$ ]]; then
  NFE_FLOOR="${NFE_FLOOR:-3}"
  if (( NFE_FLOOR < 3 )); then
    echo "Arena ${MEMORY_MODE} requires NFE_FLOOR >= 3, got ${NFE_FLOOR}." >&2
    exit 2
  fi
else
  NFE_FLOOR="${NFE_FLOOR:-1}"
fi

if [[ "${RESUME}" == "1" && -z "${RUN_ROOT:-}" ]]; then
  echo "RESUME=1 requires an explicit RUN_ROOT." >&2
  exit 2
fi
RUN_ROOT="${RUN_ROOT:-${ROOT}/logs/robomemarena_extra8_pi05_finetuned_ckpt30000${MEMORY_TAG}_batch8_$(date -u +%Y%m%d_%H%M%S)}"

for path in \
  "${CHECKPOINT}/model.safetensors" \
  "${CHECKPOINT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json" \
  "${BENCHMARK_ROOT}/scripts/eval_common.py" \
  "${BENCHMARK_ROOT}/scripts/task2_26_reference_stage.py"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
if [[ "${MEMORY_MODE}" != "none" ]]; then
  for path in \
    "${TASK_HEAD_CKPT}" \
    "${MEMORY_META_PATH}" \
    "${FAISS_INDEX_PATH}" \
    "${MEMORY_ACTIONS_PATH}" \
    "${ACTION_NORM_STATS}"; do
    [[ -f "${path}" ]] || { echo "Missing required memory file: ${path}" >&2; exit 2; }
  done
fi
[[ -x "${OPENPI_PYTHON}" ]] || { echo "Missing OpenPI Python: ${OPENPI_PYTHON}" >&2; exit 2; }
[[ -x "${LIBERO_PYTHON}" ]] || { echo "Missing LIBERO Python: ${LIBERO_PYTHON}" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout is required." >&2; exit 2; }

IFS=',' read -r -a task_ids_array <<<"${TASK_IDS}"
task_count="${#task_ids_array[@]}"
[[ "${task_count}" -gt 0 ]] || { echo "At least one task ID is required." >&2; exit 2; }
[[ "${INFERENCE_BATCH_SIZE}" =~ ^[0-9]+$ \
  && "${INFERENCE_BATCH_SIZE}" -ge 1 \
  && "${INFERENCE_BATCH_SIZE}" -le 64 ]] || {
  echo "INFERENCE_BATCH_SIZE must be an integer in [1, 64]." >&2
  exit 2
}
[[ "${ENV_WORKERS}" -ge "${task_count}" ]] || {
  echo "ENV_WORKERS must be at least the number of tasks (${task_count})." >&2
  exit 2
}
[[ "$((ENV_WORKERS % task_count))" -eq 0 ]] || {
  echo "ENV_WORKERS must be divisible by the number of tasks (${task_count})." >&2
  exit 2
}
case "${OPENPI_TORCH_COMPILE,,}" in
  0|false|no|off) ;;
  *) echo "OPENPI_TORCH_COMPILE must remain disabled for dynamic Arena batches." >&2; exit 2 ;;
esac
workers_per_task="$((ENV_WORKERS / task_count))"

protocol="RoboMemArena-Pi05-training-aligned-relative-action+state8+flipud-both-cameras+chunk10+eager-dynamic-batch+full-documented-task22"
protocol="${protocol}+env${ENV_WORKERS}+batch${INFERENCE_BATCH_SIZE}-deadline${INFERENCE_BATCH_WAIT_MS}ms"
if [[ "${MEMORY_MODE}" != "none" ]]; then
  protocol="${protocol}+memory-${MEMORY_MODE}-${MEMORY_PROVENANCE_TAG}+${MEMORY_ACTION_ALIGNMENT}"
  if [[ "${MEMORY_STATIC_ACTION_MODE}" != "off" ]]; then
    protocol="${protocol}+static-${MEMORY_STATIC_ACTION_MODE}-fraction${MEMORY_STATIC_ACTION_GATE_FRACTION}"
  fi
  if [[ "${MEMORY_MODE}" == "v3_re" ]]; then
    protocol="${protocol}+active-row-mask+nfe-floor${NFE_FLOOR}+coupled-prior-noise"
  elif [[ "${MEMORY_MODE}" =~ ^(v3_prior_only|v3_prior_decay)$ ]]; then
    protocol="${protocol}+nfe-floor${NFE_FLOOR}+coupled-prior-noise"
  elif [[ "${MEMORY_MODE}" == "v3_1" ]]; then
    protocol="${protocol}+nfe-floor${NFE_FLOOR}+coupled-prior-noise+guidance-cap0.5"
  elif [[ "${MEMORY_MODE}" == "v1" \
    && ( "${MEMORY_GUIDANCE_NORM_CAP}" != "0.20" \
      || "${MEMORY_GUIDANCE_TOTAL_NORM_CAP}" != "0.20" ) ]]; then
    protocol="${protocol}+guidance-cap${MEMORY_GUIDANCE_NORM_CAP}-total${MEMORY_GUIDANCE_TOTAL_NORM_CAP}"
  fi
fi
if [[ "${RESUME}" == "1" ]]; then
  [[ -f "${RUN_ROOT}/run_config.json" ]] || {
    echo "RESUME=1 requires ${RUN_ROOT}/run_config.json" >&2
    exit 2
  }
  "${OPENPI_PYTHON}" -c \
    'import datetime,json,re,sys; stored=json.load(open(sys.argv[1])); expected=dict(task_ids=sys.argv[2],episodes=int(sys.argv[3]),seed=int(sys.argv[4]),checkpoint=sys.argv[5],config=sys.argv[6],env_workers=int(sys.argv[7]),batch_size=int(sys.argv[8]),batch_wait_ms=float(sys.argv[9]),replan_steps=int(sys.argv[10]),num_steps_wait=int(sys.argv[11]),max_steps=int(sys.argv[12]),post_goal_steps=int(sys.argv[13]),save_video=int(sys.argv[14]),protocol=sys.argv[15]); allow=sys.argv[16]=="1"; comparable=lambda d:{**d,"env_workers":None,"protocol":re.sub(r"\+env\d+", "+env*",d["protocol"])} if allow else d; assert comparable(stored)==comparable(expected), f"Resume config mismatch: stored={stored}, requested={expected}"; allow and stored["env_workers"]!=expected["env_workers"] and open(sys.argv[17],"a").write(json.dumps(dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),field="env_workers",stored=stored["env_workers"],requested=expected["env_workers"]))+"\n")' \
    "${RUN_ROOT}/run_config.json" "${TASK_IDS}" "${EPISODES_PER_TASK}" "${SEED}" \
    "${CHECKPOINT}" "${CONFIG_NAME}" "${ENV_WORKERS}" "${INFERENCE_BATCH_SIZE}" \
    "${INFERENCE_BATCH_WAIT_MS}" "${REPLAN_STEPS}" "${NUM_STEPS_WAIT}" "${MAX_STEPS}" \
    "${POST_GOAL_STEPS}" "${SAVE_VIDEO}" "${protocol}" \
    "${RESUME_ALLOW_ENV_WORKERS_OVERRIDE}" "${RUN_ROOT}/resume_runtime_overrides.jsonl"
else
  if [[ -e "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to mix results into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
    exit 2
  fi
  mkdir -p "${RUN_ROOT}"
  "${OPENPI_PYTHON}" -c \
    'import json,sys; json.dump(dict(task_ids=sys.argv[2],episodes=int(sys.argv[3]),seed=int(sys.argv[4]),checkpoint=sys.argv[5],config=sys.argv[6],env_workers=int(sys.argv[7]),batch_size=int(sys.argv[8]),batch_wait_ms=float(sys.argv[9]),replan_steps=int(sys.argv[10]),num_steps_wait=int(sys.argv[11]),max_steps=int(sys.argv[12]),post_goal_steps=int(sys.argv[13]),save_video=int(sys.argv[14]),protocol=sys.argv[15]),open(sys.argv[1],"w"),indent=2)' \
    "${RUN_ROOT}/run_config.json" "${TASK_IDS}" "${EPISODES_PER_TASK}" "${SEED}" \
    "${CHECKPOINT}" "${CONFIG_NAME}" "${ENV_WORKERS}" "${INFERENCE_BATCH_SIZE}" \
    "${INFERENCE_BATCH_WAIT_MS}" "${REPLAN_STEPS}" "${NUM_STEPS_WAIT}" "${MAX_STEPS}" \
    "${POST_GOAL_STEPS}" "${SAVE_VIDEO}" "${protocol}"
fi
if [[ "${MEMORY_MODE}" != "none" ]]; then
  memory_config="${RUN_ROOT}/memory_config.json"
  if [[ "${RESUME}" == "1" ]]; then
    [[ -f "${memory_config}" ]] || { echo "RESUME=1 requires ${memory_config}" >&2; exit 2; }
    "${OPENPI_PYTHON}" -c \
      'import json,sys; stored=json.load(open(sys.argv[1])); expected=dict(memory_mode=sys.argv[2],task_head_ckpt=sys.argv[3],memory_meta_path=sys.argv[4],faiss_index_path=sys.argv[5],memory_actions_path=sys.argv[6],action_norm_stats=sys.argv[7],memory_top_k=int(sys.argv[8]),memory_progress_window=float(sys.argv[9]),memory_action_alignment=sys.argv[10],static_action_mode=sys.argv[11],static_action_arm_dim=int(sys.argv[12]),static_action_threshold=float(sys.argv[13]),static_action_gate_fraction=float(sys.argv[14])); assert stored == expected, f"Memory resume config mismatch: stored={stored}, requested={expected}"' \
      "${memory_config}" "${MEMORY_MODE}" "${TASK_HEAD_CKPT}" "${MEMORY_META_PATH}" \
      "${FAISS_INDEX_PATH}" "${MEMORY_ACTIONS_PATH}" "${ACTION_NORM_STATS}" \
      "${MEMORY_TOP_K}" "${MEMORY_PROGRESS_WINDOW}" "${MEMORY_ACTION_ALIGNMENT}" \
      "${MEMORY_STATIC_ACTION_MODE}" "${MEMORY_STATIC_ACTION_ARM_DIM}" \
      "${MEMORY_STATIC_ACTION_THRESHOLD}" "${MEMORY_STATIC_ACTION_GATE_FRACTION}"
  else
    "${OPENPI_PYTHON}" -c \
      'import json,sys; json.dump(dict(memory_mode=sys.argv[2],task_head_ckpt=sys.argv[3],memory_meta_path=sys.argv[4],faiss_index_path=sys.argv[5],memory_actions_path=sys.argv[6],action_norm_stats=sys.argv[7],memory_top_k=int(sys.argv[8]),memory_progress_window=float(sys.argv[9]),memory_action_alignment=sys.argv[10],static_action_mode=sys.argv[11],static_action_arm_dim=int(sys.argv[12]),static_action_threshold=float(sys.argv[13]),static_action_gate_fraction=float(sys.argv[14])),open(sys.argv[1],"w"),indent=2)' \
      "${memory_config}" "${MEMORY_MODE}" "${TASK_HEAD_CKPT}" "${MEMORY_META_PATH}" \
      "${FAISS_INDEX_PATH}" "${MEMORY_ACTIONS_PATH}" "${ACTION_NORM_STATS}" \
      "${MEMORY_TOP_K}" "${MEMORY_PROGRESS_WINDOW}" "${MEMORY_ACTION_ALIGNMENT}" \
      "${MEMORY_STATIC_ACTION_MODE}" "${MEMORY_STATIC_ACTION_ARM_DIM}" \
      "${MEMORY_STATIC_ACTION_THRESHOLD}" "${MEMORY_STATIC_ACTION_GATE_FRACTION}"
  fi
fi

LIBERO_CONFIG_DIR="${RUN_ROOT}/libero_config"
mkdir -p "${RUN_ROOT}/stdout" "${LIBERO_CONFIG_DIR}/datasets"
cat >"${LIBERO_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${BENCHMARK_ROOT}/libero_fork/libero
bddl_files: ${BENCHMARK_ROOT}/libero_fork/libero/bddl_files
init_states: ${BENCHMARK_ROOT}/libero_fork/libero/init_files
datasets: ${LIBERO_CONFIG_DIR}/datasets
assets: ${BENCHMARK_ROOT}/libero_fork/libero/assets
EOF

PYTHONPATH="${BENCHMARK_ROOT}/scripts:${BENCHMARK_ROOT}/libero_fork:${ROOT}" \
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}" \
  "${LIBERO_PYTHON}" -c \
  'from libero import get_libero_path; assert all(get_libero_path(k) for k in ("benchmark_root","bddl_files","init_states","datasets","assets")); print("LIBERO preflight passed.")'

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Pi0.5 RoboMemArena evaluation preflight completed successfully."
  exit 0
fi

if [[ "${MEMORY_STATIC_ACTION_MODE}" != "off" ]]; then
  export MEMORY_STATIC_ACTION_STATS_PATH="${MEMORY_STATIC_ACTION_STATS_PATH:-${RUN_ROOT}/static_action_batches.jsonl}"
fi

"${OPENPI_PYTHON}" -c \
  'import socket,sys; s=socket.socket(); s.settimeout(0.5); occupied=s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0; s.close(); raise SystemExit(1 if occupied else 0)' \
  "${PORT}" || { echo "Port ${PORT} is already occupied; choose another PORT." >&2; exit 2; }

cd "${ROOT}"
server_pid=""
worker_pids=()
cleanup() {
  for pid in "${worker_pids[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -KILL "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'echo "[launcher] received SIGINT"; exit 130' INT
trap 'echo "[launcher] received SIGTERM"; exit 143' TERM

server=(
  "${OPENPI_PYTHON}" scripts/serve_policy.py
  --env LIBERO \
  --port "${PORT}" \
  --seed "${SEED}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --inference-batch-wait-ms "${INFERENCE_BATCH_WAIT_MS}"
)
if [[ "${MEMORY_MODE}" == "none" ]]; then
  server+=(--no-use-memory)
else
  server+=(
    --use-memory
    --task-head-ckpt "${TASK_HEAD_CKPT}"
    --memory-meta-path "${MEMORY_META_PATH}"
    --faiss-index-path "${FAISS_INDEX_PATH}"
    --memory-actions-path "${MEMORY_ACTIONS_PATH}"
    --action-norm-stats-path "${ACTION_NORM_STATS}"
    --action-use-quantile-norm
    --memory-top-k "${MEMORY_TOP_K}"
    --memory-progress-window "${MEMORY_PROGRESS_WINDOW}"
    --memory-action-alignment "${MEMORY_ACTION_ALIGNMENT}"
    --memory-static-action-mode "${MEMORY_STATIC_ACTION_MODE}"
    --memory-static-action-arm-dim "${MEMORY_STATIC_ACTION_ARM_DIM}"
    --memory-static-action-threshold "${MEMORY_STATIC_ACTION_THRESHOLD}"
    --memory-static-action-gate-fraction "${MEMORY_STATIC_ACTION_GATE_FRACTION}"
    --align-mode hybrid
    --mixture-mode gaussian
    --nfe-min 1
    --nfe-max 10
    --nfe-floor "${NFE_FLOOR}"
  )
  if [[ "${MEMORY_MODE}" == "v0" ]]; then
    server+=(
      --memory-guidance-only
      --memory-guidance-time-version v0
      --memory-guidance-num-steps 10
    )
  elif [[ "${MEMORY_MODE}" == "v1" ]]; then
    server+=(
      --memory-guidance-only
      --memory-guidance-time-version v1
      --memory-guidance-num-steps 10
      --memory-guidance-norm-cap "${MEMORY_GUIDANCE_NORM_CAP}"
      --memory-guidance-total-norm-cap "${MEMORY_GUIDANCE_TOTAL_NORM_CAP}"
    )
  elif [[ "${MEMORY_MODE}" == "v3" || "${MEMORY_MODE}" == "v3_prior_decay" ]]; then
    server+=(
      --memory-prior-substep-guidance
      --memory-prior-guidance-version v3_prior_decay
      --memory-guidance-v2-magnitude-cap 0
      --memory-guidance-lambda-max 0.20
      --memory-prior-guidance-final-scale 0.01
    )
  elif [[ "${MEMORY_MODE}" == "v3_prior_only" ]]; then
    server+=(
      --memory-prior-substep-guidance
      --memory-prior-guidance-version v3_prior_only
      --memory-guidance-v2-magnitude-cap 0
      --memory-guidance-lambda-max 0.20
      --memory-prior-guidance-final-scale 0.01
    )
  elif [[ "${MEMORY_MODE}" == "v3_1" ]]; then
    server+=(
      --memory-prior-substep-guidance
      --memory-prior-guidance-version v3_1_prior_decay
      --memory-guidance-v2-magnitude-cap 0
      --memory-guidance-lambda-max 0.20
      --memory-prior-guidance-final-scale 0.01
      --memory-prior-guidance-norm-cap 0.50
    )
  else
    server+=(
      --memory-prior-substep-guidance
      --memory-prior-guidance-version v3_re_prior_decay
      --memory-guidance-v2-magnitude-cap 0
      --memory-guidance-lambda-max 0.20
      --memory-prior-guidance-final-scale 0.01
    )
  fi
fi
server+=(
  policy:checkpoint
  --policy.config "${CONFIG_NAME}"
  --policy.dir "${CHECKPOINT}"
)
NFE_STATS_PATH=""
if [[ "${MEMORY_MODE}" =~ ^(v3|v3_prior_only|v3_prior_decay|v3_1|v3_re)$ ]]; then
  NFE_STATS_PATH="${RUN_ROOT}/nfe_stats.jsonl"
fi
PREDIMEM_NFE_STATS_PATH="${NFE_STATS_PATH}" \
CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE}" \
PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}" \
  "${server[@]}" >"${RUN_ROOT}/server.log" 2>&1 &
server_pid=$!

if ! "${OPENPI_PYTHON}" scripts/eval/wait_for_websocket.py \
  --port "${PORT}" --pid "${server_pid}" \
  --timeout "${SERVER_START_TIMEOUT_SECONDS}" >"${RUN_ROOT}/server_ready.log" 2>&1; then
  echo "Pi policy server failed to become ready; see ${RUN_ROOT}/server.log" >&2
  exit 3
fi
kill -0 "${server_pid}" 2>/dev/null || {
  echo "Pi policy server exited after startup; see ${RUN_ROOT}/server.log" >&2
  exit 3
}

video_arg="--save-video"
[[ "${SAVE_VIDEO}" == "1" ]] || video_arg="--no-save-video"
for task_id in "${task_ids_array[@]}"; do
  task_id="${task_id//[[:space:]]/}"
  mkdir -p "${RUN_ROOT}/task${task_id}"
  PYTHONPATH="${BENCHMARK_ROOT}/scripts:${BENCHMARK_ROOT}/libero_fork:${ROOT}" \
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}" \
  MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
    timeout --signal=TERM --kill-after=60 "${WORKER_TIMEOUT_SECONDS}" \
    "${LIBERO_PYTHON}" -m optimus_eval.robomemarena_pi_worker \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --task-id "${task_id}" \
    --num-workers "${workers_per_task}" \
    --episodes "${EPISODES_PER_TASK}" \
    --seed "${SEED}" \
    --server-port "${PORT}" \
    --output-dir "${RUN_ROOT}/task${task_id}" \
    --replan-steps "${REPLAN_STEPS}" \
    --num-steps-wait "${NUM_STEPS_WAIT}" \
    --max-steps "${MAX_STEPS}" \
    --post-goal-steps "${POST_GOAL_STEPS}" \
    --env-init-retries "${ENV_INIT_RETRIES}" \
    "${video_arg}" >"${RUN_ROOT}/stdout/task${task_id}.log" 2>&1 &
  worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
  wait "${pid}" || failed=1
done
if [[ "${failed}" != "0" ]]; then
  echo "At least one Arena worker failed; inspect ${RUN_ROOT}/stdout and resume with the same RUN_ROOT." >&2
  exit 5
fi

"${OPENPI_PYTHON}" -m optimus_eval.summarize_robomemarena "${RUN_ROOT}" \
  --task-ids "${TASK_IDS}" \
  --episodes-per-task "${EPISODES_PER_TASK}" | tee "${RUN_ROOT}/results.txt"
if [[ "${MEMORY_MODE}" =~ ^(v3|v3_prior_only|v3_prior_decay|v3_1|v3_re)$ ]]; then
  "${OPENPI_PYTHON}" -m optimus_eval.summarize_predimem_nfe "${RUN_ROOT}" \
    --task-ids "${TASK_IDS}"
fi
if [[ "${MEMORY_STATIC_ACTION_MODE}" != "off" ]]; then
  "${OPENPI_PYTHON}" -m optimus_eval.summarize_static_action_gate "${RUN_ROOT}" \
    | tee "${RUN_ROOT}/static_action_results.txt"
fi
"${OPENPI_PYTHON}" -m optimus_eval.summarize_memory_ablation "${RUN_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
