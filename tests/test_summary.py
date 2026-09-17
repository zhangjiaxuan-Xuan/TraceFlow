from __future__ import annotations

import json
from pathlib import Path

from traceflow.summary import summarize_libero


def test_summary_aggregates_independent_suites(tmp_path: Path) -> None:
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        directory = tmp_path / suite / "eval"
        directory.mkdir(parents=True)
        (directory / f"{suite}.jsonl").write_text(
            "\n".join(json.dumps({"success": value}) for value in (True, False)) + "\n", encoding="utf-8"
        )
    output = tmp_path / "summary.json"
    result = summarize_libero(tmp_path, output)
    assert result["successes"] == 4
    assert result["episodes"] == 8
    assert "not one 1966/2000 run" in result["warning"]
    assert output.is_file()


def test_summary_prefers_combined_episode_results(tmp_path: Path) -> None:
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        directory = tmp_path / suite / "eval"
        worker_directory = directory / "workers" / "run"
        worker_directory.mkdir(parents=True)
        episode = {
            "event": "episode_result",
            "task_suite": suite,
            "task_id": 0,
            "episode_idx": 0,
            "seed": 7,
            "success": True,
        }
        video = {**episode, "event": "episode_video"}
        text = "\n".join(json.dumps(row) for row in (episode, video)) + "\n"
        (directory / f"{suite}.jsonl").write_text(text, encoding="utf-8")
        (worker_directory / f"{suite}_worker0.jsonl").write_text(text, encoding="utf-8")

    result = summarize_libero(tmp_path, tmp_path / "summary.json")

    assert result["successes"] == 4
    assert result["episodes"] == 4
