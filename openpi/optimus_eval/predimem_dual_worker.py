from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import tempfile
from typing import Any

import numpy as np


def _load_reference(benchmark_root: Path):
    reference = benchmark_root / "async_vlm26_reference"
    runtime = benchmark_root / "openpi_minimal_runtime"
    scripts = benchmark_root / "scripts"
    for path in (reference, runtime, scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import eval_fullvlm26_async_vlm_vla as reference_eval

    return reference_eval


def _trace_keys(trace_root: Path | None) -> set[tuple[int, int, int]] | None:
    if trace_root is None:
        return None
    manifest = trace_root / "manifest.jsonl"
    if not manifest.is_file():
        return set()
    keys = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        keys.add((int(row["task_id"]), int(row["episode_idx"]), int(row["policy_call_idx"])))
    return keys


def _read_completed(output_dir: Path, trace_keys: set[tuple[int, int, int]] | None = None) -> set[int]:
    completed: set[int] = set()
    for path in sorted(output_dir.glob("worker_*.jsonl")):
        rows = []
        changed = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if trace_keys is not None:
                    task_id = int(row["task_id"])
                    episode = int(row["episode"])
                    expected = {
                        (task_id, episode, call) for call in range(len(row.get("lower_timing", [])))
                    }
                    if not expected or not expected.issubset(trace_keys):
                        changed = True
                        logging.warning(
                            "Requeueing task=%d episode=%d because staged traces are incomplete (%d/%d)",
                            task_id,
                            episode,
                            len(expected & trace_keys),
                            len(expected),
                        )
                        continue
                rows.append(row)
                completed.add(int(row["episode"]))
        if changed:
            _atomic_rewrite_jsonl(path, rows)
    return completed


def _build_pending_by_task(
    task_ids: list[int],
    episodes: int,
    completed_by_task: dict[int, set[int]],
) -> dict[int, list[tuple[int, int]]]:
    """Group pending episodes by task while preserving resume filtering."""
    pending: dict[int, list[tuple[int, int]]] = {}
    for task_id in task_ids:
        completed = completed_by_task.get(task_id, set())
        pending[int(task_id)] = [
            (int(task_id), episode)
            for episode in range(int(episodes))
            if episode not in completed
        ]
    return pending


def _task_probe_order(task_ids: list[int], worker_id: int, active_task_id: int | None) -> list[int]:
    """Prioritize environment reuse, then worker affinity, then work stealing."""
    preferred_index = int(worker_id) % len(task_ids)
    rotated = task_ids[preferred_index:] + task_ids[:preferred_index]
    if active_task_id in rotated:
        rotated.remove(active_task_id)
        rotated.insert(0, int(active_task_id))
    return rotated


def _claim_next_item(
    task_queues: dict[int, Any],
    task_ids: list[int],
    worker_id: int,
    active_task_id: int | None,
) -> tuple[int, int] | None:
    for candidate_task_id in _task_probe_order(task_ids, worker_id, active_task_id):
        try:
            return task_queues[candidate_task_id].get_nowait()
        except queue.Empty:
            continue
    return None


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _atomic_rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            tmp_path = Path(handle.name)
            for row in rows:
                handle.write((json.dumps(row, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _prune_pending_memory_traces(
    output_root: Path, pending: list[tuple[int, int]], trace_root: Path | None = None
) -> int:
    trace_root = trace_root or output_root / "memory_records" / "traces"
    if not pending or not trace_root.is_dir():
        return 0
    pending_set = set(pending)
    manifest_path = trace_root / "manifest.jsonl"
    kept_rows = []
    removed_paths: set[Path] = set()
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row.get("task_id", -1)), int(row.get("episode_idx", -1)))
            if key in pending_set:
                removed_paths.add(trace_root / str(row["trace"]))
            else:
                kept_rows.append(row)
    for task_id, episode in pending_set:
        pattern = f"robomemarena_task_{task_id:03d}_episode_{episode:03d}_call_*"
        removed_paths.update(trace_root.glob(pattern))
    removed = 0
    for path in removed_paths:
        if path.is_file():
            path.unlink()
            removed += 1
    if manifest_path.is_file():
        _atomic_rewrite_jsonl(manifest_path, kept_rows)
    if removed:
        logging.warning("Removed %d orphan traces for %d pending episodes", removed, len(pending_set))
    return removed


def _close_env_safely(env: Any) -> None:
    if env is None:
        return
    try:
        env.close()
    except AttributeError as exc:
        logging.warning("Ignoring close failure from partially initialized environment: %s", exc)


def _create_env_with_retries(env_cls, kwargs: dict[str, Any], *, task_id: int, max_attempts: int):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return env_cls(**kwargs)
        except Exception as exc:
            if type(exc).__name__ != "RandomizationError":
                raise
            last_error = exc
            logging.warning(
                "task=%d environment randomization attempt=%d/%d failed: %s",
                task_id,
                attempt,
                max_attempts,
                exc,
            )
    raise RuntimeError(
        f"Task {task_id} environment initialization failed after {max_attempts} attempts: {last_error}"
    )


def _completion_observing_specs(stage_specs: list[Any], planner: Any) -> list[Any]:
    """Mirror official stage checks into the planner without changing checks."""
    wrapped = []
    for stage, spec in enumerate(stage_specs):
        original = spec.check_fn

        def check_fn(env, state, stage_start, *, _original=original, _stage=stage):
            completed = bool(_original(env, state, stage_start))
            if completed:
                planner.notify_stage_completed(_stage, int(state.get("step_idx", -1)))
            return completed

        wrapped.append(type(spec)(name=spec.name, check_fn=check_fn))
    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--upper-host", default="127.0.0.1")
    parser.add_argument("--upper-port", type=int, default=8330)
    parser.add_argument("--pi-host", default="127.0.0.1")
    parser.add_argument("--pi-port", type=int, default=8331)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--n-recent", type=int, default=5)
    parser.add_argument("--vlm-interval", type=int, default=5)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--post-goal-steps", type=int, default=200)
    parser.add_argument("--env-init-retries", type=int, default=20)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--lower-probe-cross-env-batch", action="store_true")
    args = parser.parse_args()
    task_ids = [int(value.strip()) for value in args.task_ids.split(",") if value.strip()]
    if not task_ids or any(not 1 <= task_id <= 26 for task_id in task_ids):
        raise ValueError("task-ids must be a non-empty comma-separated subset of [1, 26]")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task-ids must not contain duplicates")
    if min(args.num_workers, args.episodes, args.n_recent, args.vlm_interval, args.env_init_retries) < 1:
        raise ValueError("workers, episodes, n-recent, vlm-interval, and env-init-retries must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    trace_keys = _trace_keys(args.trace_root)
    completed_by_task: dict[int, set[int]] = {}
    for task_id in task_ids:
        task_root = args.output_root / f"task{task_id}"
        task_root.mkdir(parents=True, exist_ok=True)
        (task_root / "videos").mkdir(exist_ok=True)
        completed_by_task[task_id] = _read_completed(task_root, trace_keys)
    pending_by_task = _build_pending_by_task(task_ids, args.episodes, completed_by_task)
    pending = [item for task_id in task_ids for item in pending_by_task[task_id]]
    _prune_pending_memory_traces(args.output_root, pending, args.trace_root)
    reference_eval = _load_reference(args.benchmark_root)
    task_infos = reference_eval.load_task_infos(args.task_config)
    reference_eval.patch_env_resolution()

    runtime_args = reference_eval.BaseArgs()
    runtime_args.host = args.pi_host
    runtime_args.port = args.pi_port
    runtime_args.resize_size = 256
    runtime_args.replan_steps = args.replan_steps
    runtime_args.num_steps_wait = args.num_steps_wait
    runtime_args.max_steps = args.max_steps
    runtime_args.n_recent = args.n_recent
    runtime_args.async_vlm = True
    runtime_args.vlm_interval = args.vlm_interval
    runtime_args.vlm_queue_size = 1
    runtime_args.vlm_input_profile = "fullvlm_256"
    runtime_args.vlm_use_wrist = True
    runtime_args.vlm_use_keyframe_memory = True
    reference_eval._apply_vlm_input_profile(runtime_args)

    context = mp.get_context("fork")
    manager = context.Manager()
    task_queues = {task_id: manager.Queue() for task_id in task_ids}
    for task_id in task_ids:
        for item in pending_by_task[task_id]:
            task_queues[task_id].put(item)
    affinity_workers = {
        task_id: sum(1 for worker_id in range(args.num_workers) if task_ids[worker_id % len(task_ids)] == task_id)
        for task_id in task_ids
    }
    logging.info(
        "scheduler=task_affinity_work_stealing workers=%d affinity=%s pending=%s",
        args.num_workers,
        affinity_workers,
        {task_id: len(pending_by_task[task_id]) for task_id in task_ids},
    )

    def run_dynamic_worker(worker_id: int) -> None:
        from optimus_eval.predimem_contextual_pi_client import ContextualPiClient
        from optimus_eval.predimem_remote_planner import RemotePrediMemPlanner

        environment_id = f"{args.variant}-envslot{worker_id}"
        raw_client = reference_eval.StableWebsocketClientPolicy(
            args.pi_host,
            args.pi_port,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=30.0,
        )
        pi_client = ContextualPiClient(
            raw_client,
            environment_id,
            cross_environment_probe_batch=args.lower_probe_cross_env_batch,
        )
        env = None
        active_task_id = None
        task_runtime = None
        try:
            while True:
                item = _claim_next_item(task_queues, task_ids, worker_id, active_task_id)
                if item is None:
                    break
                task_id, episode = item
                if active_task_id != task_id:
                    _close_env_safely(env)
                    env = None
                    bddl_path = reference_eval.ec._resolve_bddl_path(task_id)
                    env_cls = reference_eval.ec._get_env_class()
                    env = _create_env_with_retries(
                        env_cls,
                        {
                            "bddl_file_name": str(bddl_path),
                            "camera_heights": 480,
                            "camera_widths": 640,
                            "ignore_done": True,
                            "reward_shaping": True,
                            "control_freq": 20,
                            "initialization_noise": None,
                        },
                        task_id=task_id,
                        max_attempts=args.env_init_retries,
                    )
                    counting_pour = reference_eval.stage_eval._is_counting_pour_task(task_id)
                    task_runtime = (
                        reference_eval._task_specs(task_id),
                        {} if counting_pour else reference_eval.ec._build_goal_monitor_dict(bddl_path),
                        reference_eval._goal_override_check(task_id),
                    )
                    active_task_id = task_id
                assert env is not None and task_runtime is not None
                stage_specs, goal_monitor, goal_override = task_runtime
                task_root = args.output_root / f"task{task_id}"
                output_path = task_root / f"worker_{worker_id:02d}.jsonl"
                episode_seed = args.seed + episode
                reference_eval._seed_everywhere(episode_seed)
                try:
                    env.seed(episode_seed)
                except AttributeError:
                    pass
                pi_client.set_episode_context(
                    task_id=task_id,
                    episode_idx=episode,
                    seed=episode_seed,
                )
                run_dir = task_root / f"ep{episode:04d}"
                logger = reference_eval.make_episode_logger(run_dir)
                planner = RemotePrediMemPlanner(
                    host=args.upper_host,
                    port=args.upper_port,
                    environment_id=environment_id,
                    task_info=task_infos[task_id],
                    lower_feature_provider=pi_client.latest_lower_retrieval_feature,
                    lower_probe_provider=pi_client.latest_lower_probe_results,
                    lower_probe_candidate_consumer=pi_client.set_lower_probe_candidates,
                )
                planner.reset_episode(run_dir=run_dir, logger=logger)
                try:
                    episode_stage_specs = stage_specs
                    if os.environ.get("WILL_GUIDANCE_ORACLE_COMPLETION", "0") == "1":
                        episode_stage_specs = _completion_observing_specs(stage_specs, planner)
                    stage_pct, stage_done, _, diagnostics, replay, replay_wrist = (
                        reference_eval.run_episode_async_stateful(
                            task_id=task_id,
                            env=env,
                            client=pi_client,
                            planner=planner,
                            args=runtime_args,
                            stage_specs=episode_stage_specs,
                            goal_monitor_dict=goal_monitor,
                            goal_check_override=goal_override,
                            vlm_camera_pose=None,
                            logger=logger,
                            fail_on_extra_pour=True,
                            extra_pour_monitor_steps=30,
                            post_goal_steps=args.post_goal_steps,
                        )
                    )
                    if planner.last_error is not None:
                        raise RuntimeError(f"Upper VLM request failed: {planner.last_error}")
                finally:
                    planner.close()

                base_name = reference_eval.ec.get_video_basename(
                    task_id,
                    episode,
                    episode_seed,
                    diagnostics["stage_success"],
                )
                if args.save_video and replay:
                    reference_eval._write_video(task_root / "videos" / f"{base_name}.mp4", replay, fps=10)
                if args.save_video and replay_wrist:
                    reference_eval._write_video(
                        task_root / "videos" / f"{base_name}_wrist.mp4",
                        replay_wrist,
                        fps=10,
                    )
                record = {
                    "episode": episode,
                    "ep": episode,
                    "seed": episode_seed,
                    "task_id": task_id,
                    "TSR": 100.0 if diagnostics["stage_success"] else 0.0,
                    "CSR": float(stage_pct),
                    "stage_done": stage_done,
                    "failure_reason": diagnostics["failure_reason"],
                    "video_dir": str(task_root / "videos"),
                    "upper_timing": planner.timing_samples(),
                    "upper_guidance": planner.guidance_samples(),
                    "lower_timing": pi_client.timing_samples(),
                }
                _append_jsonl(output_path, record)
                logging.info(
                    "task=%d worker=%d episode=%d TSR=%.1f CSR=%.1f",
                    task_id,
                    worker_id,
                    episode,
                    record["TSR"],
                    record["CSR"],
                )
        finally:
            _close_env_safely(env)
            pi_client.close()

    children = [
        context.Process(target=run_dynamic_worker, args=(worker_id,), name=f"predimem-envslot{worker_id}")
        for worker_id in range(args.num_workers)
    ]
    for child in children:
        child.start()

    def terminate_children(signum, unused_frame) -> None:
        del unused_frame
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=5)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, terminate_children)
    signal.signal(signal.SIGTERM, terminate_children)
    failed = False
    for child in children:
        child.join()
        failed = failed or child.exitcode != 0
    missing = {
        task_id: sorted(set(range(args.episodes)) - _read_completed(args.output_root / f"task{task_id}"))
        for task_id in task_ids
    }
    missing = {task_id: episodes for task_id, episodes in missing.items() if episodes}
    if failed or missing:
        raise SystemExit(
            f"Dynamic Arena pool incomplete: child_failed={failed}, missing_episodes={missing}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
