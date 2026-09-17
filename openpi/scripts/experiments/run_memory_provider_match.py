#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
from typing import Any

OPENPI_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = OPENPI_ROOT.parent
sys.path.insert(0, str(OPENPI_ROOT / "src"))

from openpi.experiments.memory_provider_match import POLICIES  # noqa: E402
from openpi.experiments.memory_provider_match import PROVIDERS  # noqa: E402
from openpi.experiments.memory_provider_match import MatchExperiment  # noqa: E402
from openpi.experiments.memory_provider_match import MatchProtocolError  # noqa: E402
from openpi.experiments.memory_provider_match import job_identity  # noqa: E402
from openpi.experiments.memory_provider_match import load_experiment  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Job:
    consumer: str
    provider: str
    run_root: Path
    identity: dict[str, Any]
    env: dict[str, str]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MatchProtocolError(f"missing completed eval log: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatchProtocolError(f"invalid JSON at {path}:{line_number}") from exc
            if (
                isinstance(row, dict)
                and row.get("event", "episode_result") == "episode_result"
                and "success" in row
                and "task_id" in row
            ):
                rows.append(row)
    return rows


def validate_completion(job: Job) -> dict[str, Any]:
    log = job.run_root / ("eval/libero_10.jsonl" if job.consumer == "pi" else "episodes.jsonl")
    rows = _records(log)
    identities: set[tuple[int, int]] = set()
    successes = 0
    per_task = dict.fromkeys(range(10), 0)
    for row in rows:
        success = row.get("success")
        if not isinstance(success, bool):
            raise MatchProtocolError(f"success must be JSON bool in {log}")
        task = row.get("task_id")
        episode = row.get("episode_idx", row.get("episode_ix"))
        if isinstance(task, bool) or not isinstance(task, int) or task not in per_task:
            raise MatchProtocolError(f"invalid LIBERO-10 task_id in {log}: {task!r}")
        if isinstance(episode, bool) or not isinstance(episode, int) or not 0 <= episode < 100:
            raise MatchProtocolError(f"invalid episode index in {log}: {episode!r}")
        identity = (task, episode)
        if identity in identities:
            raise MatchProtocolError(f"duplicate episode identity in {log}: {identity}")
        identities.add(identity)
        per_task[task] += 1
        successes += int(success)
        if job.consumer == "smol":
            expected_seed = 7 + task * 10_000 + episode
            if row.get("seed") != expected_seed:
                raise MatchProtocolError(
                    f"Smol seed mismatch for {identity}: expected {expected_seed}, got {row.get('seed')}"
                )
        elif row.get("seed") != 7:
            raise MatchProtocolError(f"Pi seed mismatch for {identity}: expected 7, got {row.get('seed')}")
        suite = row.get("task_suite", row.get("suite"))
        if suite != "libero_10":
            raise MatchProtocolError(f"suite mismatch for {identity}: expected libero_10, got {suite!r}")
    if len(identities) != 1000 or set(per_task.values()) != {100}:
        raise MatchProtocolError(f"incomplete LIBERO-10 result in {log}: rows={len(identities)}, per_task={per_task}")
    return {"episodes": 1000, "successes": successes, "success_rate": successes / 1000, "log": str(log)}


def build_jobs(experiment: MatchExperiment) -> list[Job]:
    jobs = []
    for consumer in POLICIES:
        policy = experiment.policies[consumer]
        for provider in PROVIDERS:
            bank = experiment.providers[provider][consumer]
            run_root = experiment.run_root / consumer / provider
            selection = policy.selection
            env = {
                "POLICY_CONSUMER": consumer,
                "MEMORY_PROVIDER": provider,
                "RUN_ROOT": str(run_root),
                "CHECKPOINT_DIR": str(policy.checkpoint_dir),
                "HEAD_PATH": str(policy.head.path),
                "POLICY_PYTHON": policy.python,
                "BANK_ROOT": str(bank.root),
                "POSITIVE_META": str(bank.positive["meta"].path),
                "POSITIVE_INDEX": str(bank.positive["index"].path),
                "POSITIVE_ACTIONS": str(bank.positive["actions"].path),
                "NEGATIVE_META": str(bank.negative["meta"].path),
                "NEGATIVE_INDEX": str(bank.negative["index"].path),
                "NEGATIVE_ACTIONS": str(bank.negative["actions"].path),
                "POSITIVE_TOP_K": str(selection.positive_top_k),
                "NEGATIVE_TOP_K": str(selection.negative_top_k),
            }
            jobs.append(Job(consumer, provider, run_root, job_identity(experiment, consumer, provider), env))
    return jobs


def _preflight(job: Job, *, resume: bool) -> str:
    identity_path = job.run_root / "match.identity.json"
    if identity_path.is_file():
        actual = json.loads(identity_path.read_text(encoding="utf-8"))
        if actual != job.identity:
            raise MatchProtocolError(f"resume identity mismatch: {identity_path}")
        if not resume:
            raise MatchProtocolError(f"completed job requires resume=true: {identity_path}")
        validate_completion(job)
        return "skip"
    run_identity_path = job.run_root / "match.run_identity.json"
    if run_identity_path.is_file():
        actual = json.loads(run_identity_path.read_text(encoding="utf-8"))
        if actual != job.identity:
            raise MatchProtocolError(f"partial-run identity mismatch: {run_identity_path}")
        if not resume:
            raise MatchProtocolError(f"partial job requires resume=true: {run_identity_path}")
        return "resume"
    if job.run_root.exists() and any(job.run_root.iterdir()):
        raise MatchProtocolError(f"refusing non-empty run root without identity: {job.run_root}")
    return "run"


def run_campaign(experiment: MatchExperiment, *, dry_run: bool) -> dict[str, Any]:
    jobs = build_jobs(experiment)
    states = {f"{job.consumer}:{job.provider}": _preflight(job, resume=experiment.resume) for job in jobs}
    if dry_run:
        return {"status": "dry-run", "jobs": states, "gpu_pool": list(experiment.gpu_pool)}
    pending: queue.Queue[Job] = queue.Queue()
    results: dict[str, Any] = {}
    for job in jobs:
        key = f"{job.consumer}:{job.provider}"
        if states[key] == "skip":
            results[key] = {"status": "skipped", **validate_completion(job)}
        else:
            pending.put(job)
    stop = threading.Event()
    processes: dict[str, subprocess.Popen[str]] = {}
    lock = threading.Lock()

    def terminate() -> None:
        stop.set()
        with lock:
            active = list(processes.values())
        for process in active:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)

    def worker(gpu: str, port: int) -> None:
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            key = f"{job.consumer}:{job.provider}"
            log = experiment.run_root / "stdout" / f"{job.consumer}__{job.provider}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            job.run_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(job.run_root / "match.run_identity.json", job.identity)
            resume_log = job.run_root / ("eval/libero_10.jsonl" if job.consumer == "pi" else "episodes.jsonl")
            env = {
                **os.environ,
                **job.env,
                "GPU": gpu,
                "PORT": str(port),
                "RESUME": "1" if states[key] == "resume" and resume_log.is_file() else "0",
            }
            command = ("bash", str(OPENPI_ROOT / "scripts/experiments/run_memory_provider_match.sh"))
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with lock:
                    processes[key] = process
                returncode = process.wait()
                with lock:
                    processes.pop(key, None)
            if returncode:
                with lock:
                    results[key] = {"status": "failed", "returncode": returncode, "stdout": str(log)}
                terminate()
                pending.task_done()
                return
            try:
                summary = validate_completion(job)
                _atomic_json(job.run_root / "match.identity.json", job.identity)
                with lock:
                    results[key] = {"status": "completed", **summary, "stdout": str(log)}
            except Exception as exc:
                with lock:
                    results[key] = {"status": "failed", "error": str(exc), "stdout": str(log)}
                terminate()
            pending.task_done()

    previous: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        terminate()
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, handle_signal)
    try:
        threads = [
            threading.Thread(target=worker, args=(gpu, port), daemon=True)
            for gpu, port in zip(experiment.gpu_pool, experiment.ports, strict=True)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        terminate()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    status = (
        "completed"
        if len(results) == 4 and all(row["status"] in {"completed", "skipped"} for row in results.values())
        else "failed"
    )
    summary = {"schema_version": 1, "status": status, "manifest_sha256": experiment.manifest_sha256, "results": results}
    _atomic_json(experiment.run_root / "match_campaign_summary.json", summary)
    if status != "completed":
        raise MatchProtocolError(f"memory-provider match campaign failed: {results}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the four Pi/Smol x Pi-mem/Smol-mem LIBERO-10 matches")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run_campaign(load_experiment(args.manifest), dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
