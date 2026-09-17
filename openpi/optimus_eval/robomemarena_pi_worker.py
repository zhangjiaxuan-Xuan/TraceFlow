from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
import signal
import sys
from typing import Any

import numpy as np


def _load_benchmark(benchmark_root: Path):
    for path in (benchmark_root / "scripts", benchmark_root / "libero_fork"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import eval_common as common
    import eval_task1_only as task1
    import eval_tasks2_26 as tasks

    return common, task1, tasks


def _restore_full_task22_protocol(tasks) -> None:
    import task2_26_reference_stage as stages

    original_task_specs = tasks._task_specs

    def task_specs(task_id: int):
        specs = original_task_specs(task_id)
        if task_id != 22:
            return specs
        return [
            *specs,
            stages.StageSpec(
                "04_Place_Tomato_Aside",
                stages._near_fixed_position(
                    "tomato_sauce_1",
                    np.array([0.0, -0.2, 0.50], dtype=np.float32),
                    0.20,
                    0.20,
                ),
            ),
            stages.StageSpec("05_Open_Microwave", stages._microwave_open(0.30)),
            stages.StageSpec("06_Place_Cookies_Microwave", stages._in_microwave("cookies_1")),
        ]

    tasks._task_specs = task_specs


def _read_completed(output_dir: Path) -> set[int]:
    completed: set[int] = set()
    for path in sorted(output_dir.glob("worker_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(int(json.loads(line)["episode"]))
    return completed


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _run_one(*, task_id, episode, seed, adapter, common, task1, tasks, args):
    adapter.set_episode_context(
        task_id=task_id,
        episode_idx=episode,
        seed=seed + episode,
    )
    result = None
    last_randomization_error = None
    for attempt in range(1, args.env_init_retries + 1):
        try:
            kwargs = {
                "task_id": task_id,
                "num_trials_per_task": 1,
                "adapter": adapter,
                "resize_size": 256,
                "replan_steps": args.replan_steps,
                "num_steps_wait": args.num_steps_wait,
                "max_steps": args.max_steps,
                "post_goal_steps": args.post_goal_steps,
                "video_out_path": str(args.output_dir / "videos"),
                "seed": seed + episode,
            }
            if task_id == 1:
                result = common.run_eval(
                    **kwargs,
                    stage_checks=task1.STAGE_CHECKS,
                    seed_everywhere_fn=lambda value: np.random.seed(value),
                )
            else:
                result = tasks.run_eval_task(
                    **kwargs,
                    fail_on_extra_pour=True,
                    extra_pour_monitor_steps=30,
                )
            break
        except Exception as exc:
            if type(exc).__name__ != "RandomizationError":
                raise
            last_randomization_error = exc
            logging.warning(
                "task=%d episode=%d randomization attempt=%d/%d failed: %s",
                task_id,
                episode,
                attempt,
                args.env_init_retries,
                exc,
            )
    if result is None:
        raise RuntimeError(
            f"Task {task_id} episode {episode} initialization failed: {last_randomization_error}"
        )
    if adapter.last_error is not None:
        raise RuntimeError(f"Policy request failed: {adapter.last_error}")

    record = dict(result["episodes"][0])
    record.update(
        episode=episode,
        ep=episode,
        task_id=task_id,
        prompt=result["prompt"],
        video_dir=result["video_dir"],
        action_generation_timing=adapter.timing_samples(),
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8700)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--post-goal-steps", type=int, default=200)
    parser.add_argument("--env-init-retries", type=int, default=30)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not 1 <= args.task_id <= 26:
        raise ValueError("task-id must be in [1, 26]")
    if args.episodes < 1 or args.num_workers < 1 or args.env_init_retries < 1:
        raise ValueError("episodes, num-workers and env-init-retries must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [ep for ep in range(args.episodes) if ep not in _read_completed(args.output_dir)]
    common, task1, tasks = _load_benchmark(args.benchmark_root)
    _restore_full_task22_protocol(tasks)
    if args.task_id == 1:
        common.patch_env_resolution(480, 640)
    else:
        tasks._patch_env_resolution()
    if not args.save_video:
        common.imageio.mimwrite = lambda *unused_args, **unused_kwargs: None
        tasks.imageio.mimwrite = lambda *unused_args, **unused_kwargs: None

    from optimus_eval.robomemarena_pi_adapter import PiRoboMemArenaAdapter

    def run_shard(worker_id: int) -> None:
        output_path = args.output_dir / f"worker_{worker_id:02d}.jsonl"
        assigned = [
            episode
            for pending_index, episode in enumerate(pending)
            if pending_index % args.num_workers == worker_id
        ]
        if not assigned:
            return
        adapter = PiRoboMemArenaAdapter(
            host=args.server_host,
            port=args.server_port,
            environment_id=f"task{args.task_id}-worker{worker_id}",
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
        )
        try:
            # Establish every websocket before MuJoCo initialization. The policy
            # server runs GPU inference synchronously and cannot accept a late
            # handshake while another environment is inside inference.
            adapter.connect()
            for episode in assigned:
                record = _run_one(
                    task_id=args.task_id,
                    episode=episode,
                    seed=args.seed,
                    adapter=adapter,
                    common=common,
                    task1=task1,
                    tasks=tasks,
                    args=args,
                )
                _append_jsonl(output_path, record)
                logging.info(
                    "task=%d worker=%d episode=%d seed=%d TSR=%.1f CSR=%.1f",
                    args.task_id,
                    worker_id,
                    episode,
                    args.seed + episode,
                    float(record["TSR"]),
                    float(record["CSR"]),
                )
        finally:
            adapter.close()

    context = mp.get_context("fork")
    children = [
        context.Process(target=run_shard, args=(worker_id,), name=f"pi-task{args.task_id}-env{worker_id}")
        for worker_id in range(args.num_workers)
    ]
    for child in children:
        child.start()

    def terminate_children(signum, unused_frame) -> None:
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

    missing = sorted(set(range(args.episodes)) - _read_completed(args.output_dir))
    if failed or missing:
        raise SystemExit(
            f"Task {args.task_id} incomplete: child_failed={failed}, missing_episodes={missing}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
