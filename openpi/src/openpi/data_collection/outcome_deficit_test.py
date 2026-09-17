from __future__ import annotations

import copy

import pytest

from openpi.data_collection.outcome_deficit import CollectionSpec
from openpi.data_collection.outcome_deficit import OutcomeQuota
from openpi.data_collection.outcome_deficit import build_plan
from openpi.data_collection.outcome_deficit import expand_suites
from openpi.data_collection.outcome_deficit import merge_records


def _spec(*, quota: tuple[int, int] = (2, 2), cap: int = 5) -> CollectionSpec:
    return CollectionSpec(
        producer="pi0.5-libero",
        checkpoint_digest="abc",
        suites=("libero_10",),
        quota=OutcomeQuota(*quota),
        max_attempts_per_task=cap,
        batch_attempts=3,
    )


def _row(episode: int, *, success: bool, task: int = 0) -> dict:
    return {
        "schema": "optimus-outcome-attempt-v1",
        "trajectory_id": f"t{task}-{episode}",
        "producer": "pi0.5-libero",
        "checkpoint_digest": "abc",
        "suite": "libero_10",
        "task_id": task,
        "seed": 7 + episode,
        "episode_idx": episode,
        "outcome": "success" if success else "failure",
        "trajectory_path": f"/tmp/t{task}-{episode}.h5",
        "trajectory_sha256": "f" * 64,
        "action_contract": "libero-relative-ee-7d-v1",
    }


def test_gap_calculation_and_completion() -> None:
    jobs = build_plan(
        _spec(),
        [
            _row(0, success=True),
            _row(1, success=False),
            _row(2, success=True),
            _row(3, success=False),
        ],
    )
    assert jobs[0].status == "complete"
    assert jobs[0].attempt_count == 0
    assert jobs[1].success_deficit == 2
    assert jobs[1].failure_deficit == 2
    assert jobs[1].attempt_count == 3


def test_resume_deduplicates_identical_and_rejects_conflict() -> None:
    row = _row(0, success=True)
    assert merge_records([row], [copy.deepcopy(row)]) == [row]
    conflict = copy.deepcopy(row)
    conflict["outcome"] = "failure"
    with pytest.raises(ValueError, match="Conflicting resumed attempt"):
        merge_records([row], [conflict])


def test_long_expands_to_reported_ten_task_condition() -> None:
    assert expand_suites(["long"]) == ("libero_10",)
    assert expand_suites(["spatial,object", "goal", "long"]) == (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    )


def test_pi_expands_to_four_ten_task_suites() -> None:
    assert expand_suites(["pi"]) == (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    )


def test_hard_cap_marks_incomplete_without_synthetic_records() -> None:
    records = [_row(index, success=index == 0) for index in range(5)]
    job = build_plan(_spec(quota=(2, 2), cap=5), records)[0]
    assert job.status == "incomplete"
    assert job.attempt_count == 0
    assert job.success_deficit == 1


def test_fixed_50_attempts_per_task_round_finishes_before_quota_replan() -> None:
    spec = CollectionSpec(
        producer="pi0.5-libero",
        checkpoint_digest="abc",
        suites=("libero_10",),
        quota=OutcomeQuota(0, 10),
        max_attempts_per_task=5000,
        batch_attempts=50,
    )
    initial = build_plan(spec, [])[0]
    assert initial.status == "pending"
    assert initial.attempt_count == 50

    completed_round = [_row(index, success=index >= 10) for index in range(50)]
    after_round = build_plan(spec, completed_round)[0]
    assert after_round.status == "complete"
    assert after_round.attempt_count == 0


def test_complete_suite_round_keeps_all_ten_tasks_in_the_current_group() -> None:
    spec = CollectionSpec(
        producer="pi0.5-libero",
        checkpoint_digest="abc",
        suites=("libero_10",),
        quota=OutcomeQuota(0, 1),
        max_attempts_per_task=5000,
        batch_attempts=50,
        complete_suite_rounds=True,
    )
    records = [_row(0, success=False, task=0)]

    jobs = build_plan(spec, records)

    assert len(jobs) == 10
    assert all(job.status == "pending" for job in jobs)
    assert all(job.attempt_count == 50 for job in jobs)
    assert sum(job.attempt_count for job in jobs) == 500
    assert jobs[0].failure_deficit == 0


def test_complete_long_round_covers_one_hundred_tasks() -> None:
    spec = CollectionSpec(
        producer="pi0.5-libero",
        checkpoint_digest="abc",
        suites=("libero_10", "libero_90"),
        quota=OutcomeQuota(50, 10),
        max_attempts_per_task=5000,
        batch_attempts=50,
        complete_suite_rounds=True,
    )

    jobs = build_plan(spec, [])

    assert len(jobs) == 100
    assert all(job.status == "pending" for job in jobs)
    assert all(job.attempt_count == 50 for job in jobs)
    assert sum(job.attempt_count for job in jobs) == 5000
