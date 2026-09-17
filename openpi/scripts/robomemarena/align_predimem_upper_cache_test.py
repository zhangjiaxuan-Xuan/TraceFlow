from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from scripts.robomemarena import align_predimem_upper_cache


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_causal_hold_never_uses_future_upper_feature(tmp_path: Path, monkeypatch) -> None:
    source_manifest = tmp_path / "source.jsonl"
    target_manifest = tmp_path / "target.jsonl"
    source_cache = tmp_path / "source_cache"
    output = tmp_path / "output"
    source_cache.mkdir()
    source_rows = [
        {"row_index": 0, "action_id": "trajectory", "global_frame_index": 0},
        {"row_index": 1, "action_id": "trajectory", "global_frame_index": 10},
    ]
    target_rows = [
        {"row_index": 0, "action_id": "trajectory", "global_frame_index": 0},
        {"row_index": 1, "action_id": "trajectory", "global_frame_index": 5},
        {"row_index": 2, "action_id": "trajectory", "global_frame_index": 10},
    ]
    _write_jsonl(source_manifest, source_rows)
    _write_jsonl(target_manifest, target_rows)
    np.save(source_cache / "upper_features.npy", np.asarray([[1, 0], [0, 1]], dtype=np.float16))
    _write_jsonl(
        source_cache / "subtasks.jsonl",
        [
            {"row_index": 0, "subtask": "first"},
            {"row_index": 1, "subtask": "second"},
        ],
    )
    digest = hashlib.sha256((source_cache / "subtasks.jsonl").read_bytes()).hexdigest()
    (source_cache / "upper_cache_state.json").write_text(
        json.dumps(
            {
                "complete": True,
                "protocol": "predimem_trajectory_ordered_runtime_generate_subtask_hidden_v4",
                "subtask_records_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "align_predimem_upper_cache.py",
            "--source-manifest",
            str(source_manifest),
            "--target-manifest",
            str(target_manifest),
            "--source-cache-dir",
            str(source_cache),
            "--output-dir",
            str(output),
            "--age-normalizer-steps",
            "20",
        ],
    )

    align_predimem_upper_cache.main()

    np.testing.assert_array_equal(
        np.load(output / "upper_features.npy"),
        np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float16),
    )
    np.testing.assert_allclose(np.load(output / "upper_age.npy"), [0.0, 0.25, 0.0])
    records = [json.loads(line) for line in (output / "subtasks.jsonl").read_text().splitlines()]
    assert [record["subtask"] for record in records] == ["first", "first", "second"]
    assert [record["held_source_row_index"] for record in records] == [0, 0, 1]
    np.testing.assert_array_equal(np.load(output / "upper_available.npy"), [True, True, True])


def test_causal_hold_marks_frames_before_first_upper_update_unavailable(tmp_path: Path, monkeypatch) -> None:
    source_manifest = tmp_path / "source.jsonl"
    target_manifest = tmp_path / "target.jsonl"
    source_cache = tmp_path / "source_cache"
    output = tmp_path / "output"
    source_cache.mkdir()
    _write_jsonl(source_manifest, [{"row_index": 0, "action_id": "t", "global_frame_index": 10}])
    _write_jsonl(
        target_manifest,
        [
            {"row_index": 0, "action_id": "t", "global_frame_index": 0, "prompt": "task"},
            {"row_index": 1, "action_id": "t", "global_frame_index": 10, "prompt": "task"},
        ],
    )
    np.save(source_cache / "upper_features.npy", np.asarray([[3, 4]], dtype=np.float16))
    _write_jsonl(source_cache / "subtasks.jsonl", [{"row_index": 0, "subtask": "move"}])
    (source_cache / "upper_cache_state.json").write_text(
        json.dumps({"complete": True, "protocol": "traceflow_three_view_runtime_generate_subtask_hidden_v1"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "align_predimem_upper_cache.py",
            "--source-manifest", str(source_manifest),
            "--target-manifest", str(target_manifest),
            "--source-cache-dir", str(source_cache),
            "--output-dir", str(output),
        ],
    )

    align_predimem_upper_cache.main()

    np.testing.assert_array_equal(
        np.load(output / "upper_features.npy"),
        np.asarray([[0, 0], [3, 4]], dtype=np.float16),
    )
    np.testing.assert_array_equal(np.load(output / "upper_available.npy"), [False, True])
    records = [json.loads(line) for line in (output / "subtasks.jsonl").read_text().splitlines()]
    assert [record["upper_available"] for record in records] == [False, True]
