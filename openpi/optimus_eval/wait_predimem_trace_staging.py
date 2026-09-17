from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time


def _manifest_keys(path: Path) -> set[tuple[int, int, int]]:
    keys = set()
    if not path.is_file():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        keys.add((int(row["task_id"]), int(row["episode_idx"]), int(row["policy_call_idx"])))
    return keys


def _expected_keys(run_root: Path) -> set[tuple[int, int, int]]:
    keys = set()
    for path in sorted(run_root.glob("task*/worker_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = int(row["task_id"])
            episode = int(row["episode"])
            keys.update((task_id, episode, call) for call in range(len(row.get("lower_timing", []))))
    return keys


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--producer-pid", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    expected = _expected_keys(args.run_root)
    if not expected:
        raise RuntimeError(f"No completed lower policy calls found in {args.run_root}")
    deadline = time.monotonic() + args.timeout_seconds
    waiter = threading.Event()
    while True:
        if (args.trace_root / "writer_errors.jsonl").is_file():
            raise RuntimeError(f"Trace writer failed: {args.trace_root / 'writer_errors.jsonl'}")
        actual = _manifest_keys(args.trace_root / "manifest.jsonl")
        missing = expected - actual
        if not missing:
            if actual != expected:
                extras = sorted(actual - expected)
                raise RuntimeError(f"Unexpected staged traces: {extras[:10]}")
            marker = {
                "schema_version": 1,
                "trace_count": len(expected),
                "run_root": str(args.run_root),
                "producer_pid": args.producer_pid,
                "completed_at_unix": time.time(),
            }
            _atomic_json(args.trace_root / "PRODUCER_COMPLETE.json", marker)
            print(json.dumps(marker, sort_keys=True), flush=True)
            return
        try:
            os.kill(args.producer_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError(f"Trace producer exited with {len(missing)} pending records") from exc
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {len(missing)} staged traces")
        waiter.wait(0.1)


if __name__ == "__main__":
    main()
