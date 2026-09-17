#!/usr/bin/env python3
"""Atomically merge a LIBERO combined JSONL with per-worker logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def merge_jsonl(output: Path, worker_root: Path, pattern: str = "*.jsonl") -> tuple[int, int, int]:
    sources = ([output] if output.is_file() else []) + sorted(worker_root.rglob(pattern))
    unique_lines: list[str] = []
    seen: set[str] = set()
    duplicate_count = 0
    errored_episode_count = 0
    for source in sources:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {source}:{line_number}") from error
                if record.get("event") == "episode_result" and record.get("error"):
                    errored_episode_count += 1
                    continue
                if line in seen:
                    duplicate_count += 1
                    continue
                seen.add(line)
                unique_lines.append(line)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            for line in unique_lines:
                temp.write(line + "\n")
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, output)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return len(unique_lines), duplicate_count, errored_episode_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.jsonl")
    args = parser.parse_args()
    records, duplicates, errored = merge_jsonl(args.output, args.worker_root, args.pattern)
    print(
        f"Merged {records} unique records into {args.output} "
        f"(ignored {duplicates} duplicates, excluded {errored} errored episodes)"
    )


if __name__ == "__main__":
    main()
