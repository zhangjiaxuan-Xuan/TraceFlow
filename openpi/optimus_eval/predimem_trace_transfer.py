from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def _trace_sizes(directory: Path, *, label: str) -> dict[str, int]:
    sizes = {}
    started = time.monotonic()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith(".npz") and entry.is_file(follow_symlinks=False):
                sizes[entry.name] = entry.stat(follow_symlinks=False).st_size
                if len(sizes) % 4096 == 0:
                    print(
                        f"[trace-transfer] scanning={label} files={len(sizes)} "
                        f"elapsed_seconds={time.monotonic() - started:.1f}",
                        flush=True,
                    )
    print(
        f"[trace-transfer] scanned={label} files={len(sizes)} "
        f"elapsed_seconds={time.monotonic() - started:.1f}",
        flush=True,
    )
    return sizes


def _append_many(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError(f"Short manifest append: {path}")
    finally:
        os.close(fd)


def _copy_one(source: Path, destination: Path, expected_size: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".transfer", delete=False
        ) as stream:
            temporary = Path(stream.name)
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream, length=8 * 1024 * 1024)
            stream.flush()
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(f"Transferred size mismatch: {source}")
        os.replace(temporary, destination)
        return expected_size
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_manifest(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < args.workers or args.poll_seconds <= 0:
        raise ValueError("workers/poll-seconds must be positive and batch-size >= workers")
    args.source.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.source / "SAFE_TO_DELETE.json").unlink(missing_ok=True)
    (args.destination / "TRANSFER_COMPLETE.json").unlink(missing_ok=True)
    destination_manifest = args.destination / "manifest.jsonl"
    transferred = {str(row["trace"]): row for row in _rows(destination_manifest)}
    destination_sizes = _trace_sizes(args.destination, label="destination-startup")
    transferred = {
        name: row
        for name, row in transferred.items()
        if destination_sizes.get(name) == int(row.get("trace_size", -1))
    }
    waiter = threading.Event()
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="trace-transfer") as executor:
        while True:
            if (args.source / "writer_errors.jsonl").is_file():
                raise RuntimeError(f"Local trace writer reported errors: {args.source / 'writer_errors.jsonl'}")
            source_rows = _rows(args.source / "manifest.jsonl")
            pending = []
            for row in source_rows:
                name = str(row["trace"])
                previous = transferred.get(name)
                if (
                    previous is None
                    or int(previous.get("trace_size", -1)) != int(row["trace_size"])
                    or int(previous.get("trace_source_mtime_ns", -1))
                    != int(row.get("trace_source_mtime_ns", -2))
                ):
                    pending.append(row)
            pending = pending[: args.batch_size]
            futures = {
                executor.submit(
                    _copy_one,
                    args.source / str(row["trace"]),
                    args.destination / str(row["trace"]),
                    int(row["trace_size"]),
                ): row
                for row in pending
            }
            completed_rows = []
            for future in as_completed(futures):
                row = futures[future]
                try:
                    future.result()
                except FileNotFoundError:
                    # A worker-pool retry atomically rewrites the source
                    # manifest and removes that episode's stale traces. The
                    # transfer may already hold the previous manifest
                    # snapshot; refresh on the next loop instead of killing
                    # an otherwise recoverable evaluation.
                    print(
                        f"[trace-transfer] source-invalidated trace={row['trace']}",
                        flush=True,
                    )
                    continue
                completed_rows.append(row)
                transferred[str(row["trace"])] = row
                destination_sizes[str(row["trace"])] = int(row["trace_size"])
            _append_many(destination_manifest, completed_rows)
            if completed_rows:
                print(
                    f"[trace-transfer] copied_batch={len(completed_rows)} "
                    f"transferred={len(transferred)} source_rows={len(source_rows)}",
                    flush=True,
                )
            producer_marker = args.source / "PRODUCER_COMPLETE.json"
            if producer_marker.is_file() and not pending:
                producer = json.loads(producer_marker.read_text(encoding="utf-8"))
                expected = int(producer["trace_count"])
                source_rows = _rows(args.source / "manifest.jsonl")
                if len(source_rows) == expected:
                    source_names = {str(row["trace"]) for row in source_rows}
                    destination_sizes = _trace_sizes(
                        args.destination, label="destination-final-verification"
                    )
                    incomplete = []
                    for row in source_rows:
                        name = str(row["trace"])
                        if destination_sizes.get(name) != int(row["trace_size"]):
                            incomplete.append(str(row["trace"]))
                    if incomplete:
                        # AOSS rename visibility can lag behind close. The next
                        # loop rechecks and recopies only files still missing.
                        for name in incomplete:
                            transferred.pop(name, None)
                        print(
                            f"Destination verification pending: {len(incomplete)} trace(s)",
                            flush=True,
                        )
                        waiter.wait(args.poll_seconds)
                        continue
                    for name in set(destination_sizes) - source_names:
                        (args.destination / name).unlink(missing_ok=True)
                    for temporary in args.destination.glob(".*.transfer"):
                        temporary.unlink(missing_ok=True)
                    _atomic_manifest(destination_manifest, source_rows)
                    transferred = {str(row["trace"]): row for row in source_rows}
                    complete = {
                        "schema_version": 1,
                        "trace_count": expected,
                        "source": str(args.source),
                        "destination": str(args.destination),
                        "workers": args.workers,
                        "batch_size": args.batch_size,
                        "completed_at_unix": time.time(),
                    }
                    _atomic_json(args.destination / "TRANSFER_COMPLETE.json", complete)
                    _atomic_json(args.source / "SAFE_TO_DELETE.json", complete)
                    print(json.dumps(complete, sort_keys=True), flush=True)
                    return
                if len(source_rows) > expected:
                    raise RuntimeError(
                        f"Trace count exceeds producer contract: source={len(source_rows)} expected={expected}"
                    )
            waiter.wait(args.poll_seconds)


if __name__ == "__main__":
    main()
