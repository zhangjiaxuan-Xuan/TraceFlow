from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "openpi" / "scripts" / "data" / "merge_libero_jsonl.py"
SPEC = importlib.util.spec_from_file_location("merge_libero_jsonl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_merge_is_atomic_deduplicated_and_excludes_errored_episodes(tmp_path: Path) -> None:
    output = tmp_path / "combined.jsonl"
    workers = tmp_path / "workers"
    workers.mkdir()
    start = {"event": "process_start", "pid": 1}
    success = {"event": "episode_result", "task_id": 0, "episode_idx": 0, "success": True, "error": ""}
    errored = {"event": "episode_result", "task_id": 0, "episode_idx": 1, "success": False, "error": "boom"}
    output.write_text(json.dumps(start) + "\n", encoding="utf-8")
    (workers / "suite_worker0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (start, success, errored)) + "\n", encoding="utf-8"
    )

    records, duplicates, errors = MODULE.merge_jsonl(output, workers, "suite_worker*.jsonl")

    assert (records, duplicates, errors) == (2, 1, 1)
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [start, success]
