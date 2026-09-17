from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).with_name("merge_predimem_feature_cache.py")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.mark.parametrize(
    ("kind", "feature_name", "completed_name"),
    [("upper", "upper_features.npy", "upper_completed.npy"), ("lower", "pooled_prefix.npy", "completed.npy")],
)
def test_merge_uses_producer_completion_protocol_and_full_order(
    tmp_path: Path, kind: str, feature_name: str, completed_name: str
) -> None:
    base_manifest = tmp_path / "base.jsonl"
    extra_manifest = tmp_path / "extra.jsonl"
    full_manifest = tmp_path / "full.jsonl"
    base_rows = [{"action_id": "a", "global_frame_index": 5}]
    extra_rows = [
        {"action_id": "b", "global_frame_index": 0},
        {"action_id": "a", "global_frame_index": 0},
    ]
    full_rows = [extra_rows[1], base_rows[0], extra_rows[0]]
    _write_jsonl(base_manifest, base_rows)
    _write_jsonl(extra_manifest, extra_rows)
    _write_jsonl(full_manifest, full_rows)

    base_cache, extra_cache, output_cache = (tmp_path / name for name in ("base", "extra", "output"))
    base_cache.mkdir()
    extra_cache.mkdir()
    np.save(base_cache / feature_name, np.asarray([[5.0, 50.0]], dtype=np.float32))
    np.save(extra_cache / feature_name, np.asarray([[10.0, 100.0], [0.0, 0.0]], dtype=np.float32))
    np.save(base_cache / completed_name, np.ones(1, dtype=np.bool_))
    np.save(extra_cache / completed_name, np.ones(2, dtype=np.bool_))

    if kind == "upper":
        state = {"rows": 1, "complete": True}
        (base_cache / "upper_cache_state.json").write_text(json.dumps(state), encoding="utf-8")
        (extra_cache / "upper_cache_state.json").write_text(json.dumps({**state, "rows": 2}), encoding="utf-8")
        _write_jsonl(base_cache / "subtasks.jsonl", [{"row_index": 0, "subtask": "a5"}])
        _write_jsonl(
            extra_cache / "subtasks.jsonl",
            [{"row_index": 0, "subtask": "b0"}, {"row_index": 1, "subtask": "a0"}],
        )
    else:
        (base_cache / "cache_state.json").write_text(
            json.dumps({"rows": 1, "config_name": "extra8"}), encoding="utf-8"
        )
        (extra_cache / "cache_state.json").write_text(
            json.dumps({"rows": 2, "config_name": "all26"}), encoding="utf-8"
        )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--kind",
            kind,
            "--full-manifest",
            str(full_manifest),
            "--base-manifest",
            str(base_manifest),
            "--base-cache",
            str(base_cache),
            "--extra-manifest",
            str(extra_manifest),
            "--extra-cache",
            str(extra_cache),
            "--output-cache",
            str(output_cache),
            *( ["--config-name", "all26"] if kind == "lower" else [] ),
        ],
        check=True,
    )

    np.testing.assert_array_equal(
        np.load(output_cache / feature_name),
        np.asarray([[0.0, 0.0], [5.0, 50.0], [10.0, 100.0]], dtype=np.float32),
    )
    assert np.load(output_cache / completed_name).all()
    assert not (output_cache / ("completed.npy" if kind == "upper" else "upper_completed.npy")).exists()
    if kind == "upper":
        merged_subtasks = [json.loads(line) for line in (output_cache / "subtasks.jsonl").read_text().splitlines()]
        assert [(row["row_index"], row["subtask"]) for row in merged_subtasks] == [
            (0, "a0"),
            (1, "a5"),
            (2, "b0"),
        ]
    else:
        state = json.loads((output_cache / "cache_state.json").read_text())
        assert state["config_name"] == "all26"
        assert state["source_config_names"] == ["all26", "extra8"]
