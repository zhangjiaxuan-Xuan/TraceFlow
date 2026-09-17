from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _absolute_executable_path(path: Path) -> Path:
    # Resolving a venv launcher follows its symlink to system Python and drops the venv.
    return Path(os.path.abspath(path))


def _episode_records(path: Path, suite: str, task_id: int) -> dict[int, dict[str, Any]]:
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
            if row.get("task_suite") != suite or int(row.get("task_id", -1)) != task_id:
                raise RuntimeError(f"Task identity mismatch in {path}:{line_number}")
            episode = int(row["episode_idx"])
            previous = records.get(episode)
            if previous is not None and not str(previous.get("error", "")):
                raise RuntimeError(f"Duplicate completed episode {suite}/{task_id}/{episode} in {path}")
            records[episode] = row
    return records


def _task_complete(path: Path, suite: str, task_id: int, episodes: int) -> bool:
    records = _episode_records(path, suite, task_id)
    expected = set(range(episodes))
    return set(records) == expected and all(not str(row.get("error", "")) for row in records.values())


def _client_command(args: argparse.Namespace, suite: str, task_id: int, slot: int) -> list[str]:
    task_log = args.run_root / "tasks" / suite / f"task_{task_id:03d}.jsonl"
    command = [
        str(args.libero_python),
        str(args.openpi_root / "examples/libero/main.py"),
        "--args.host",
        args.host,
        "--args.port",
        str(args.port),
        "--args.task-suite-name",
        suite,
        "--args.task-ids-csv",
        str(task_id),
        "--args.num-trials-per-task",
        str(args.episodes_per_task),
        "--args.episode-start",
        "0",
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
        "--args.episode-indices-csv",
        ",".join(str(index) for index in range(args.episodes_per_task)),
        "--args.environment-id",
        f"slot{slot}-{suite}-task{task_id}",
        "--args.fail-on-episode-error",
    ]
    if args.save_videos:
        command.append("--args.save-videos")
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
    eval_root = args.run_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    total_episodes = 0
    total_successes = 0
    lines = [
        "OptimusVLA LIBERO Dynamic Evaluation Results",
        f"run_root: {args.run_root}",
        "",
        "suite\tepisodes\tsuccesses\tsuccess_rate",
    ]
    for suite in SUITES:
        suite_records: dict[tuple[int, int], dict[str, Any]] = {}
        for task_id in range(10):
            task_path = args.run_root / "tasks" / suite / f"task_{task_id:03d}.jsonl"
            records = _episode_records(task_path, suite, task_id)
            if set(records) != set(range(args.episodes_per_task)):
                raise RuntimeError(
                    f"Incomplete task after scheduler exit: {suite}/{task_id} "
                    f"has {len(records)}/{args.episodes_per_task}"
                )
            for episode, row in records.items():
                if str(row.get("error", "")):
                    raise RuntimeError(f"Episode error in {suite}/{task_id}/{episode}: {row['error']}")
                suite_records[(task_id, episode)] = row
        output = eval_root / f"{suite}.jsonl"
        temporary = output.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for key in sorted(suite_records):
                stream.write(json.dumps(suite_records[key], ensure_ascii=False) + "\n")
        temporary.replace(output)
        successes = sum(bool(row.get("success", False)) for row in suite_records.values())
        episodes = len(suite_records)
        total_episodes += episodes
        total_successes += successes
        lines.append(f"{suite}\t{episodes}\t{successes}\t{successes / episodes:.4f}")
    lines.extend(
        [
            "",
            f"overall_success_rate: {total_successes / total_episodes:.4f} "
            f"({total_successes}/{total_episodes})",
        ]
    )
    (eval_root / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--libero-python", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--max-task-retries", type=int, default=2)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    args.openpi_root = args.openpi_root.resolve()
    args.libero_python = _absolute_executable_path(args.libero_python)
    args.run_root = args.run_root.resolve()
    if args.num_workers < 1 or args.episodes_per_task < 1:
        raise ValueError("num-workers and episodes-per-task must be positive")
    if args.max_task_retries < 0:
        raise ValueError("max-task-retries must be non-negative")
    for path in (args.libero_python, args.openpi_root / "examples/libero/main.py"):
        if not path.is_file():
            raise FileNotFoundError(path)

    jobs = [(suite, task_id) for suite in SUITES for task_id in range(10)]
    pending = []
    for suite, task_id in jobs:
        task_log = args.run_root / "tasks" / suite / f"task_{task_id:03d}.jsonl"
        if _task_complete(task_log, suite, task_id, args.episodes_per_task):
            continue
        if task_log.exists() and not args.resume:
            raise RuntimeError(f"Partial task log requires --resume: {task_log}")
        pending.append((suite, task_id, 0))
    if args.preflight_only:
        print(
            f"Dynamic scheduler preflight passed: jobs={len(jobs)} pending={len(pending)} "
            f"workers={args.num_workers}",
            flush=True,
        )
        return

    args.run_root.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[subprocess.Popen[bytes], int, str, int, int, Any]] = {}
    free_slots = list(range(args.num_workers))

    def terminate_active() -> None:
        for process, _, _, _, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, _, _, _, _, stream in active.values():
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
                suite, task_id, attempt = pending.pop(0)
                task_dir = args.run_root / "tasks" / suite
                stdout_dir = args.run_root / "stdout" / suite
                task_dir.mkdir(parents=True, exist_ok=True)
                stdout_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = stdout_dir / f"task_{task_id:03d}.log"
                stream = stdout_path.open("ab")
                command = _client_command(args, suite, task_id, slot)
                process = subprocess.Popen(
                    command,
                    cwd=args.openpi_root,
                    env=os.environ.copy(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[process.pid] = (process, slot, suite, task_id, attempt, stream)
                print(
                    f"Started slot={slot} pid={process.pid} job={suite}/task{task_id} "
                    f"attempt={attempt + 1}/{args.max_task_retries + 1}",
                    flush=True,
                )
            pid, wait_status = os.wait()
            if pid not in active:
                continue
            process, slot, suite, task_id, attempt, stream = active.pop(pid)
            stream.close()
            free_slots.append(slot)
            free_slots.sort()
            exit_code = os.waitstatus_to_exitcode(wait_status)
            process.returncode = exit_code
            if exit_code != 0:
                if attempt < args.max_task_retries:
                    pending.insert(0, (suite, task_id, attempt + 1))
                    print(
                        f"Retrying {suite}/task{task_id} after exit={exit_code}; "
                        f"next_attempt={attempt + 2}/{args.max_task_retries + 1}",
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    f"Dynamic task failed after {attempt + 1} attempts: "
                    f"{suite}/task{task_id} exit={exit_code}; "
                    f"see {args.run_root / 'stdout' / suite / f'task_{task_id:03d}.log'}"
                )
            task_log = args.run_root / "tasks" / suite / f"task_{task_id:03d}.jsonl"
            if not _task_complete(task_log, suite, task_id, args.episodes_per_task):
                raise RuntimeError(f"Task exited successfully but is incomplete: {suite}/task{task_id}")
            print(f"Finished slot={slot} job={suite}/task{task_id}", flush=True)
    except BaseException:
        terminate_active()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    _write_results(args)
    print(f"Dynamic four-suite evaluation complete: {args.run_root / 'eval/results.txt'}", flush=True)


if __name__ == "__main__":
    main()
