from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimus_eval import libero_dynamic_task_scheduler as scheduler


def test_absolute_executable_path_preserves_venv_launcher(tmp_path: Path) -> None:
    system_python = tmp_path / "system-python"
    system_python.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(system_python)

    assert scheduler._absolute_executable_path(venv_python) == venv_python


def _args(run_root: Path, episodes: int = 2) -> argparse.Namespace:
    return argparse.Namespace(run_root=run_root, episodes_per_task=episodes)


def _write_task(path: Path, suite: str, task_id: int, episodes: int, *, error: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for episode in range(episodes):
            stream.write(
                json.dumps(
                    {
                        "event": "episode_result",
                        "task_suite": suite,
                        "task_id": task_id,
                        "episode_idx": episode,
                        "success": episode % 2 == 0,
                        "error": error if episode == episodes - 1 else "",
                    }
                )
                + "\n"
            )


def test_task_complete_rejects_missing_duplicate_and_error(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    _write_task(path, "libero_10", 3, 2)
    assert scheduler._task_complete(path, "libero_10", 3, 2)

    _write_task(path, "libero_10", 3, 2, error="failed")
    assert not scheduler._task_complete(path, "libero_10", 3, 2)

    path.write_text(
        json.dumps(
            {
                "event": "episode_result",
                "task_suite": "libero_10",
                "task_id": 3,
                "episode_idx": 0,
                "success": True,
                "error": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert not scheduler._task_complete(path, "libero_10", 3, 2)


def test_episode_records_accepts_error_then_success(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    failed = {
        "event": "episode_result",
        "task_suite": "libero_10",
        "task_id": 3,
        "episode_idx": 0,
        "success": False,
        "error": "temporary failure",
    }
    succeeded = {**failed, "success": True, "error": ""}
    path.write_text(
        json.dumps(failed) + "\n" + json.dumps(succeeded) + "\n",
        encoding="utf-8",
    )

    records = scheduler._episode_records(path, "libero_10", 3)
    assert records[0]["success"] is True
    assert records[0]["error"] == ""


def test_write_results_merges_all_tasks_and_suites(tmp_path: Path) -> None:
    episodes = 2
    for suite in scheduler.SUITES:
        for task_id in range(10):
            _write_task(
                tmp_path / "tasks" / suite / f"task_{task_id:03d}.jsonl",
                suite,
                task_id,
                episodes,
            )

    scheduler._write_results(_args(tmp_path, episodes))

    results = (tmp_path / "eval/results.txt").read_text(encoding="utf-8")
    assert "overall_success_rate: 0.5000 (40/80)" in results
    for suite in scheduler.SUITES:
        merged = (tmp_path / "eval" / f"{suite}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(merged) == 20
