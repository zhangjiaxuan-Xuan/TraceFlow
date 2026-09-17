from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np


class AsyncTraceWriter:
    """Write already-materialized trace arrays to fast local storage."""

    def __init__(self, trace_dir: Path, *, workers: int = 8) -> None:
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trace-stage")
        self._lock = threading.Lock()
        self._submitted: set[str] = set()

    def submit(self, stem: str, arrays: dict[str, np.ndarray], record: dict) -> None:
        with self._lock:
            if stem in self._submitted:
                raise FileExistsError(f"Duplicate guidance trace submission: {stem}")
            self._submitted.add(stem)
        final_path = self.trace_dir / f"{stem}.npz"
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing guidance trace: {final_path}")
        self._executor.submit(self._write, final_path, arrays, record)

    def _write(self, final_path: Path, arrays: dict[str, np.ndarray], record: dict) -> None:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.trace_dir,
                prefix=f".{final_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                tmp_path = Path(stream.name)
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, final_path)
            stat = final_path.stat()
            manifest_record = {
                **record,
                "trace": final_path.name,
                "trace_size": int(stat.st_size),
                "trace_source_mtime_ns": int(stat.st_mtime_ns),
            }
            payload = (json.dumps(manifest_record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            fd = os.open(self.trace_dir / "manifest.jsonl", os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                written = os.write(fd, payload)
                if written != len(payload):
                    raise OSError(f"Short trace manifest append: {written}/{len(payload)}")
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            error_path = self.trace_dir / "writer_errors.jsonl"
            payload = json.dumps({"trace": final_path.name}, sort_keys=True) + "\n"
            with error_path.open("a", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            raise


def get_async_trace_writer(model: object, trace_dir: Path) -> AsyncTraceWriter:
    writer = getattr(model, "_async_trace_writer", None)
    if writer is None:
        workers = int(getattr(model, "memory_guidance_trace_workers", 8))
        writer = AsyncTraceWriter(trace_dir, workers=workers)
        setattr(model, "_async_trace_writer", writer)
    elif writer.trace_dir != trace_dir:
        raise RuntimeError(f"Trace directory changed during one policy process: {writer.trace_dir} -> {trace_dir}")
    return writer
