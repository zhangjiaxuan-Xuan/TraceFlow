from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
from typing import Any


SUITE = "libero_10"
TASK_IDS = tuple(range(10))


def _absolute_executable_path(path: Path) -> Path:
    # Resolving a venv launcher follows its symlink to system Python and drops the venv.
    return Path(os.path.abspath(path))


def _records(path: Path, task_id: int) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
            if row.get("event") != "episode_result":
                continue
            if row.get("task_suite") != SUITE or int(row.get("task_id", -1)) != task_id:
                raise RuntimeError(f"Task identity mismatch at {path}:{line_number}")
            episode = int(row["episode_idx"])
            previous = records.get(episode)
            if previous is not None and not previous.get("error"):
                raise RuntimeError(f"Duplicate completed episode {task_id}/{episode} in {path}")
            records[episode] = row
    return records


def _partition(episode_start: int, episodes_per_task: int, shards_per_task: int) -> list[tuple[int, ...]]:
    episodes = tuple(range(episode_start, episode_start + episodes_per_task))
    shards = [episodes[shard::shards_per_task] for shard in range(shards_per_task)]
    return [shard for shard in shards if shard]


def _client_command(
    args: argparse.Namespace,
    task_id: int,
    shard_id: int,
    episodes: tuple[int, ...],
) -> list[str]:
    task_log = args.run_root / "tasks" / f"task_{task_id:03d}.jsonl"
    command = [
        str(args.libero_python),
        str(args.openpi_root / "examples/libero/main.py"),
        "--args.host",
        args.host,
        "--args.port",
        str(args.port),
        "--args.task-suite-name",
        SUITE,
        "--args.task-ids-csv",
        str(task_id),
        "--args.num-trials-per-task",
        str(args.episodes_per_task),
        "--args.episode-start",
        str(args.episode_start),
        "--args.episode-indices-csv",
        ",".join(str(episode) for episode in episodes),
        "--args.replan-steps",
        str(args.replan_steps),
        "--args.num-steps-wait",
        str(args.num_steps_wait),
        "--args.resize-size",
        str(args.resize_size),
        "--args.video-root",
        str(args.run_root / "videos"),
        "--args.seed",
        str(args.seed),
        "--args.log-file",
        str(task_log),
        "--args.environment-id",
        f"{args.environment_id_prefix}-task{task_id}-shard{shard_id}",
        "--args.fail-on-episode-error",
    ]
    if args.max_env_steps > 0:
        command.extend(["--args.max-env-steps", str(args.max_env_steps)])
    if args.save_videos:
        command.append("--args.save-videos")
    if args.episode_data_root:
        command.extend(
            [
                "--args.episode-data-root",
                str(args.episode_data_root),
                "--args.episode-data-mode",
                args.episode_data_mode,
                "--args.trajectory-image-size",
                str(args.trajectory_image_size),
            ]
        )
    if task_log.exists():
        command.extend(
            [
                "--args.resume-log-file",
                str(task_log),
                "--args.resume-worker-id",
                "0",
                "--args.resume-num-workers",
                "1",
            ]
        )
    return command


def _write_results(args: argparse.Namespace) -> None:
    expected = set(range(args.episode_start, args.episode_start + args.episodes_per_task))
    all_records: dict[tuple[int, int], dict[str, Any]] = {}
    for task_id in args.task_ids:
        task_log = args.run_root / "tasks" / f"task_{task_id:03d}.jsonl"
        records = _records(task_log, task_id)
        if set(records) != expected:
            raise RuntimeError(
                f"Incomplete task {task_id}: found={len(records)} expected={args.episodes_per_task}"
            )
        errors = {episode: row["error"] for episode, row in records.items() if row.get("error")}
        if errors:
            raise RuntimeError(f"Task {task_id} contains episode errors: {errors}")
        all_records.update({(task_id, episode): row for episode, row in records.items()})
    eval_root = args.run_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    output = eval_root / f"{SUITE}.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for key in sorted(all_records):
            stream.write(json.dumps(all_records[key], ensure_ascii=False) + "\n")
    temporary.replace(output)
    successes = sum(bool(row["success"]) for row in all_records.values())
    total = len(all_records)
    (eval_root / "results.txt").write_text(
        "\n".join(
            [
                "OptimusVLA LIBERO-10 Dynamic Episode-Shard Results",
                f"episodes: {total}",
                f"successes: {successes}",
                f"success_rate: {successes / total:.4f}",
                f"workers: {args.num_workers}",
                f"shards_per_task: {args.shards_per_task}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--libero-python", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--shards-per-task", type=int, default=4)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--task-ids-csv", default=",".join(str(task_id) for task_id in TASK_IDS))
    parser.add_argument("--max-env-steps", type=int, default=0)
    parser.add_argument("--episode-start", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--max-shard-retries", type=int, default=2)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episode-data-root", type=Path)
    parser.add_argument(
        "--episode-data-mode",
        choices=("all", "successes", "failures", "none"),
        default="none",
    )
    parser.add_argument("--trajectory-image-size", type=int, default=128)
    parser.add_argument("--environment-id-prefix", default="libero10-eval")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    args.openpi_root = args.openpi_root.resolve()
    args.libero_python = _absolute_executable_path(args.libero_python)
    args.run_root = args.run_root.resolve()
    try:
        args.task_ids = tuple(int(value) for value in args.task_ids_csv.split(","))
    except ValueError as error:
        raise ValueError("task-ids-csv must contain comma-separated integers") from error
    if not args.task_ids or len(args.task_ids) != len(set(args.task_ids)) or not set(args.task_ids).issubset(TASK_IDS):
        raise ValueError(f"task-ids-csv must contain unique ids from {TASK_IDS}")
    if args.episode_data_root is not None:
        args.episode_data_root = args.episode_data_root.resolve()
    if min(args.num_workers, args.shards_per_task, args.episodes_per_task) < 1:
        raise ValueError("num-workers, shards-per-task, and episodes-per-task must be positive")
    if args.max_shard_retries < 0:
        raise ValueError("max-shard-retries must be non-negative")
    if args.trajectory_image_size <= 0:
        raise ValueError("trajectory-image-size must be positive")
    if args.max_env_steps < 0:
        raise ValueError("max-env-steps must be non-negative")
    if not args.environment_id_prefix.strip():
        raise ValueError("environment-id-prefix must not be empty")
    for path in (args.libero_python, args.openpi_root / "examples/libero/main.py"):
        if not path.is_file():
            raise FileNotFoundError(path)

    partitions = _partition(args.episode_start, args.episodes_per_task, args.shards_per_task)
    jobs: list[tuple[int, int, tuple[int, ...]]] = []
    for task_id in args.task_ids:
        task_log = args.run_root / "tasks" / f"task_{task_id:03d}.jsonl"
        completed = {
            episode
            for episode, row in _records(task_log, task_id).items()
            if not row.get("error")
        }
        for shard_id, shard in enumerate(partitions):
            missing = tuple(episode for episode in shard if episode not in completed)
            if missing:
                jobs.append((task_id, shard_id, missing))
    if args.preflight_only:
        print(
            f"Dynamic episode scheduler preflight passed: suite={SUITE} jobs={len(jobs)} "
            f"tasks={args.task_ids_csv} workers={args.num_workers} shards_per_task={args.shards_per_task}",
            flush=True,
        )
        return

    args.run_root.mkdir(parents=True, exist_ok=True)
    pending = [(*job, 0) for job in jobs]
    active: dict[int, tuple[subprocess.Popen[bytes], int, int, int, tuple[int, ...], int, Any]] = {}
    free_slots = list(range(args.num_workers))

    def terminate_active() -> None:
        for process, *_rest in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, *_middle, stream in active.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stream.close()

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while pending or active:
            while pending and free_slots:
                slot = free_slots.pop(0)
                task_id, shard_id, episodes, attempt = pending.pop(0)
                stdout_dir = args.run_root / "stdout"
                stdout_dir.mkdir(parents=True, exist_ok=True)
                stream = (stdout_dir / f"task_{task_id:03d}_shard_{shard_id:02d}.log").open("wb")
                process = subprocess.Popen(
                    _client_command(args, task_id, shard_id, episodes),
                    cwd=args.openpi_root,
                    env=os.environ.copy(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[process.pid] = (process, slot, task_id, shard_id, episodes, attempt, stream)
                print(
                    f"Started slot={slot} pid={process.pid} task={task_id} shard={shard_id} "
                    f"episodes={len(episodes)} attempt={attempt + 1}/{args.max_shard_retries + 1}",
                    flush=True,
                )
            pid, wait_status = os.wait()
            if pid not in active:
                continue
            process, slot, task_id, shard_id, episodes, attempt, stream = active.pop(pid)
            stream.close()
            free_slots.append(slot)
            free_slots.sort()
            exit_code = os.waitstatus_to_exitcode(wait_status)
            process.returncode = exit_code
            if exit_code != 0:
                if attempt < args.max_shard_retries:
                    pending.insert(0, (task_id, shard_id, episodes, attempt + 1))
                    print(
                        f"Retrying task={task_id} shard={shard_id} after exit={exit_code}; "
                        f"next_attempt={attempt + 2}/{args.max_shard_retries + 1}",
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    f"Episode shard failed after {attempt + 1} attempts: "
                    f"task={task_id} shard={shard_id} exit={exit_code}; "
                    f"see {args.run_root / 'stdout' / f'task_{task_id:03d}_shard_{shard_id:02d}.log'}"
                )
            records = _records(args.run_root / "tasks" / f"task_{task_id:03d}.jsonl", task_id)
            incomplete = [episode for episode in episodes if episode not in records or records[episode].get("error")]
            if incomplete:
                raise RuntimeError(f"Shard exited successfully but is incomplete: task={task_id} episodes={incomplete}")
            print(f"Finished slot={slot} task={task_id} shard={shard_id}", flush=True)
    except BaseException:
        terminate_active()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    _write_results(args)
    print(f"Dynamic LIBERO-10 evaluation complete: {args.run_root / 'eval/results.txt'}", flush=True)


if __name__ == "__main__":
    main()
