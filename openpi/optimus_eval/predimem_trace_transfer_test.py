from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _write_source(source: Path, count: int) -> None:
    source.mkdir(parents=True)
    rows = []
    for index in range(count):
        path = source / f"trace_{index:03d}.npz"
        path.write_bytes((f"trace-{index}" * 100).encode())
        rows.append(
            {
                "trace": path.name,
                "trace_size": path.stat().st_size,
                "trace_source_mtime_ns": path.stat().st_mtime_ns,
                "task_id": 18,
                "episode_idx": index,
                "policy_call_idx": 0,
            }
        )
    (source / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (source / "PRODUCER_COMPLETE.json").write_text(
        json.dumps({"trace_count": count}), encoding="utf-8"
    )


def test_eight_worker_transfer_is_complete_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, 24)
    command = [
        sys.executable,
        "-m",
        "optimus_eval.predimem_trace_transfer",
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--workers",
        "8",
        "--batch-size",
        "8",
    ]
    subprocess.run(command, check=True)
    (destination / "trace_001.npz").unlink()
    changed = source / "trace_000.npz"
    changed.write_bytes(b"X" * changed.stat().st_size)
    source_rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines()]
    source_rows[0]["trace_source_mtime_ns"] = changed.stat().st_mtime_ns
    (source / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )
    subprocess.run(command, check=True)
    transferred = [json.loads(line) for line in (destination / "manifest.jsonl").read_text().splitlines()]
    assert len(transferred) == 24
    assert len({row["trace"] for row in transferred}) == 24
    assert (destination / "trace_000.npz").read_bytes() == changed.read_bytes()
    assert (destination / "trace_001.npz").read_bytes() == (source / "trace_001.npz").read_bytes()
    assert (destination / "TRANSFER_COMPLETE.json").is_file()
    safe = json.loads((source / "SAFE_TO_DELETE.json").read_text())
    assert safe["trace_count"] == 24
