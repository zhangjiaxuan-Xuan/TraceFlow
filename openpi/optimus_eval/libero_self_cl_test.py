from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from optimus_eval.libero_self_cl import cumulative_rows
from optimus_eval.libero_self_cl import absolute_executable_path
from optimus_eval.libero_self_cl import evaluation_complete
from optimus_eval.libero_self_cl import materialize_round_manifest


def _trajectory(path: Path, *, success: bool, prompt: str = "do the task") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
        demo.attrs["success"] = success
        demo.attrs["language_instruction"] = prompt


def _eval_log(root: Path, *, episodes_per_task: int = 1) -> Path:
    path = root / "eval/libero_10.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as stream:
        for task_id in range(10):
            for episode_idx in range(episodes_per_task):
                success = task_id % 2 == 0
                trajectory = root / "episode_data" / f"t{task_id}_e{episode_idx}.hdf5"
                _trajectory(trajectory, success=success, prompt=f"task {task_id}")
                stream.write(
                    json.dumps(
                        {
                            "event": "episode_result",
                            "task_suite": "libero_10",
                            "task_id": task_id,
                            "episode_idx": episode_idx,
                            "seed": 7,
                            "success": success,
                            "error": "",
                            "trajectory_path": str(trajectory),
                        }
                    )
                    + "\n"
                )
    return path


def test_round_manifest_and_three_admission_lines_are_isolated(tmp_path: Path) -> None:
    eval_log = _eval_log(tmp_path)
    assert evaluation_complete(eval_log, 1)
    manifests = []
    for round_index in (1, 2):
        output = tmp_path / f"round_{round_index}.jsonl"
        rows = materialize_round_manifest(
            eval_log=eval_log,
            output=output,
            branch="all",
            round_index=round_index,
            episodes_per_task=1,
        )
        assert len(rows) == 10
        manifests.append(output)

    success = cumulative_rows(branch="success_only", round_manifests=manifests)
    failure = cumulative_rows(branch="failure_only", round_manifests=manifests)
    both = cumulative_rows(branch="all", round_manifests=manifests)
    assert (len(success), len(failure), len(both)) == (10, 10, 20)
    assert all(row["success"] for row in success)
    assert all(not row["success"] for row in failure)
    assert {row["source_round"] for row in both} == {1, 2}
    assert not ({row["action_id"] for row in success} & {row["action_id"] for row in failure})


def test_incomplete_or_error_eval_is_not_committed(tmp_path: Path) -> None:
    eval_log = _eval_log(tmp_path)
    rows = [json.loads(line) for line in eval_log.read_text(encoding="utf-8").splitlines()]
    rows[-1]["error"] = "RuntimeError: failed"
    eval_log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert not evaluation_complete(eval_log, 1)


def test_smoke_task_subset_is_complete_and_materialized(tmp_path: Path) -> None:
    eval_log = _eval_log(tmp_path)
    rows = [json.loads(line) for line in eval_log.read_text(encoding="utf-8").splitlines()]
    eval_log.write_text(
        "\n".join(json.dumps(row) for row in rows if row["task_id"] == 0) + "\n",
        encoding="utf-8",
    )

    assert evaluation_complete(eval_log, 1, (0,))
    output = tmp_path / "task0_manifest.jsonl"
    manifest = materialize_round_manifest(
        eval_log=eval_log,
        output=output,
        branch="all",
        round_index=1,
        episodes_per_task=1,
        task_ids=(0,),
    )
    assert len(manifest) == 1
    assert manifest[0]["task_id"] == 0


def test_virtualenv_executable_symlink_is_not_resolved(tmp_path: Path) -> None:
    system_python = tmp_path / "python3.10"
    system_python.touch()
    launcher = tmp_path / "venv/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(system_python)

    assert absolute_executable_path(launcher) == launcher.absolute()
    assert absolute_executable_path(launcher) != launcher.resolve()
