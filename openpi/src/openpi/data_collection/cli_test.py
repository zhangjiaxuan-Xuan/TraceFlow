from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from openpi.data_collection.cli import _canonicalize
from openpi.data_collection.cli import _commands_by_gpu
from openpi.data_collection.cli import _gpu_pool
from openpi.data_collection.outcome_deficit import CollectionSpec
from openpi.data_collection.outcome_deficit import OutcomeQuota
from openpi.data_collection.outcome_deficit import PlanJob
from openpi.data_collection.outcome_deficit import select_collection_group_failures
from openpi.data_collection.outcome_deficit import select_collection_group_records


def test_canonicalize_eval_hdf5_is_memory_builder_compatible(tmp_path: Path) -> None:
    trajectory = tmp_path / "episode.hdf5"
    with h5py.File(trajectory, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["language_instruction"] = "put the mug away"
        demo.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))
    args = argparse.Namespace(
        producer="pi0.5-libero",
        checkpoint="checkpoint",
        checkpoint_digest="a" * 64,
        seed=7,
        action_contract="libero-relative-ee-7d-v1",
        source_run="run-1",
        collection_group="C-pi",
    )
    row = _canonicalize(
        args,
        {
            "task_suite": "libero_10",
            "task_id": 3,
            "episode_idx": 12,
            "seed": 19,
            "success": False,
            "trajectory_path": str(trajectory),
        },
    )
    assert row["source_format"] == "eval_hdf5"
    assert row["frame_index"] == 0
    assert row["prompt"] == "put the mug away"
    assert row["trajectory_length"] == 4
    assert row["outcome"] == "failure"
    assert row["success"] is False
    assert row["action_id"] == row["trajectory_id"]
    assert row["collection_group"] == "C-pi"


def test_c_failure_selection_is_exact_group_and_has_no_task_cap() -> None:
    def row(index: int, *, group: str, outcome: str = "failure", producer: str = "pi0.5-libero") -> dict:
        return {
            "schema": "optimus-outcome-attempt-v1",
            "trajectory_id": f"traj_{index}",
            "producer": producer,
            "checkpoint_digest": "a" * 64,
            "suite": "libero_10",
            "task_key": "libero_10/task_000",
            "task_id": 0,
            "seed": 7 + index,
            "episode_idx": index,
            "outcome": outcome,
            "trajectory_path": f"/tmp/{index}.hdf5",
            "trajectory_sha256": "b" * 64,
            "action_contract": "libero-relative-ee-7d-v1",
            "collection_group": group,
        }

    records = [*(row(index, group="C-pi") for index in range(8)), row(20, group="B"), row(21, group="C-pi", outcome="success")]
    selected, summary = select_collection_group_failures(records, "C-pi")

    assert len(selected) == 8
    assert all(item["collection_group"] == "C-pi" and item["outcome"] == "failure" for item in selected)
    assert summary["selection"] == "exact_group_only"
    assert summary["capacity_per_task"] == "max"
    assert summary["counts_by_task"] == {"libero_10/task_000": 8}

    capped, capped_summary = select_collection_group_records(records, "C-pi", "failure", 5)
    assert len(capped) == 5
    assert capped_summary["capacity_per_task"] == "5"

    fallback_records = [
        row(index, group="C-pi")
        for index in range(3)
    ] + [
        {**row(index, group="C-pi"), "task_key": "libero_10/task_001", "task_id": 1}
        for index in range(5, 10)
    ]
    selected, fallback_summary = select_collection_group_records(
        fallback_records, "C-pi", "failure", 5, fallback_to_max=True
    )
    assert len(selected) == 8
    assert fallback_summary["fallback_to_max_tasks"] == ["libero_10/task_000"]
    assert fallback_summary["counts_by_task"] == {
        "libero_10/task_000": 3,
        "libero_10/task_001": 5,
    }


def test_c_group_capacity_contract_rejects_cross_sign_levels() -> None:
    with np.testing.assert_raises_regex(ValueError, "Unsupported failure capacity"):
        select_collection_group_records([], "C-pi", "failure", 10)
    with np.testing.assert_raises_regex(ValueError, "Unsupported success capacity"):
        select_collection_group_records([], "C-smol", "success", 5)
    with np.testing.assert_raises_regex(ValueError, "Unsupported failure capacity"):
        select_collection_group_records([], "C-pi", "failure", 50)
    with np.testing.assert_raises_regex(ValueError, "Unsupported failure capacity"):
        select_collection_group_records([], "C-smol", "failure", None)


def test_gpu_pool_partitions_jobs_into_serial_per_gpu_queues(tmp_path: Path) -> None:
    args = argparse.Namespace(
        root=tmp_path,
        checkpoint="checkpoint",
        python="python",
        gpu="9",
        gpus="0,1,2,3",
        runner_template="run --gpu {gpu} --suite {suite} --task {task_id} --count {attempt_count} --seed {seed} --output {output_dir}",
    )
    spec = CollectionSpec(
        producer="smolvla",
        checkpoint_digest="a" * 64,
        suites=("libero_10",),
        quota=OutcomeQuota(0, 100),
        max_attempts_per_task=5000,
        batch_attempts=50,
    )
    jobs = [PlanJob("libero_10", task_id, 0, 50, 7 + task_id, 0, 0, 0, 100, "pending") for task_id in range(10)]

    queues = _commands_by_gpu(args, spec, jobs)

    assert _gpu_pool(args) == ("0", "1", "2", "3")
    assert sum(job.attempt_count for job in jobs) == 500
    assert [len(commands) for commands in queues.values()] == [3, 3, 2, 2]
    for gpu, commands in queues.items():
        assert all(f"--gpu {gpu}" in command for command in commands)


def test_pi_command_adapter_uses_suite_specific_log(tmp_path: Path) -> None:
    args = argparse.Namespace(
        root=tmp_path,
        checkpoint="checkpoint",
        python="python",
        gpu="0",
        gpus="",
        runner_template="",
    )
    spec = CollectionSpec(
        producer="pi0.5-libero",
        checkpoint_digest="a" * 64,
        suites=("libero_goal",),
        quota=OutcomeQuota(50, 10),
        max_attempts_per_task=5000,
        batch_attempts=50,
    )
    job = PlanJob("libero_goal", 2, 50, 50, 7, 25, 5, 25, 5, "pending")

    command = _commands_by_gpu(args, spec, [job])["0"][0]

    assert "scripts/eval/run_libero_eval.sh libero_goal" in command
    assert "libero_goal.jsonl" in command
    assert "TASK_IDS_CSV=2" in command
    assert "EPISODE_START=50" in command


def test_pi_command_adapter_can_reuse_persistent_server(tmp_path: Path) -> None:
    args = argparse.Namespace(
        root=tmp_path,
        checkpoint="checkpoint",
        python="python",
        gpu="0",
        gpus="",
        runner_template="",
        external_policy_server=True,
        port=8300,
        server_log_path=str(tmp_path / "persistent_server.log"),
    )
    spec = CollectionSpec(
        producer="pi0.5-libero",
        checkpoint_digest="a" * 64,
        suites=("libero_10",),
        quota=OutcomeQuota(0, 5),
        max_attempts_per_task=5000,
        batch_attempts=50,
    )
    job = PlanJob("libero_10", 0, 0, 50, 7, 0, 0, 0, 5, "pending")

    command = _commands_by_gpu(args, spec, [job])["0"][0]

    assert "EXTERNAL_POLICY_SERVER=1" in command
    assert "SERVER_LOG_PATH=" in command
    assert "PORT=8300" in command
    assert "scripts/eval/run_libero_eval.sh libero_10" in command


def test_gpu_pool_rejects_duplicate_slots() -> None:
    args = argparse.Namespace(gpu="0", gpus="0,1,0")
    with np.testing.assert_raises_regex(ValueError, "unique"):
        _gpu_pool(args)
