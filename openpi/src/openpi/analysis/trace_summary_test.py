from __future__ import annotations

import json
from pathlib import Path

import pytest

from openpi.analysis.trace_summary import summarize_trace_directory


def test_summarize_trace_directory_joins_episode_outcome(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace = {
        "schema_version": 2,
        "trace_level": "light",
        "task_suite": "libero_10",
        "task_id": 2,
        "episode_idx": 3,
        "policy_call_idx": 0,
        "task_name": "put mug away",
        "progress": 0.5,
        "positive_retrieval": {
            "task_names": ["put mug away", "other task"],
            "scores": [0.9, 0.7],
            "weights": [0.75, 0.25],
        },
        "negative_retrieval": {"task_names": [], "scores": [], "weights": []},
        "steps": [
            {
                "v_base": {"norm": [2.0], "cosine_to_v_base": [1.0]},
                "guidance_positive": {"norm": [0.4], "cosine_to_v_base": [0.5]},
                "guidance_negative": {"norm": [0.0], "cosine_to_v_base": [0.0]},
                "guidance_final": {"norm": [0.4], "cosine_to_v_base": [0.5]},
            }
        ],
    }
    (trace_dir / "one.light.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        json.dumps(
            {
                "event": "episode_result",
                "task_suite": "libero_10",
                "task_id": 2,
                "episode_idx": 3,
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = summarize_trace_directory(trace_dir, eval_log, condition="v0", memory_group="N")
    assert len(rows) == 1
    assert rows[0]["success"] == 1
    assert rows[0]["purity"] == 0.5
    assert rows[0]["margin"] == pytest.approx(0.2)
    assert rows[0]["guidance_norm_ratio"] == pytest.approx(0.2)
