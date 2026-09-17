from __future__ import annotations

import queue
import json

import pytest

from optimus_eval.predimem_dual_worker import _claim_next_item
from optimus_eval.predimem_dual_worker import _close_env_safely
from optimus_eval.predimem_dual_worker import _build_pending_by_task
from optimus_eval.predimem_dual_worker import _create_env_with_retries
from optimus_eval.predimem_dual_worker import _prune_pending_memory_traces
from optimus_eval.predimem_dual_worker import _read_completed
from optimus_eval.predimem_dual_worker import _task_probe_order


class RandomizationError(Exception):
    pass


def test_create_env_retries_randomization_errors() -> None:
    attempts = 0

    class Env:
        def __init__(self, **kwargs):
            nonlocal attempts
            attempts += 1
            self.kwargs = kwargs
            if attempts < 3:
                raise RandomizationError("placement failed")

    env = _create_env_with_retries(Env, {"value": 7}, task_id=25, max_attempts=3)

    assert attempts == 3
    assert env.kwargs == {"value": 7}


def test_create_env_does_not_retry_unrelated_errors() -> None:
    class Env:
        def __init__(self, **unused_kwargs):
            raise ValueError("invalid configuration")

    with pytest.raises(ValueError, match="invalid configuration"):
        _create_env_with_retries(Env, {}, task_id=1, max_attempts=20)


def test_create_env_reports_exhausted_randomization_retries() -> None:
    class Env:
        def __init__(self, **unused_kwargs):
            raise RandomizationError("placement failed")

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        _create_env_with_retries(Env, {}, task_id=26, max_attempts=2)


def test_close_env_tolerates_partially_initialized_wrapper() -> None:
    class Env:
        def close(self):
            raise AttributeError("missing env")

    _close_env_safely(Env())


def test_prunes_only_pending_episode_traces(tmp_path) -> None:
    trace_root = tmp_path / "memory_records" / "traces"
    trace_root.mkdir(parents=True)
    stale = trace_root / "robomemarena_task_026_episode_009_call_0000.npz"
    keep = trace_root / "robomemarena_task_026_episode_010_call_0000.npz"
    orphan = trace_root / "robomemarena_task_026_episode_009_call_0001.npz"
    for path in (stale, keep, orphan):
        path.write_bytes(b"trace")
    rows = [
        {"trace": stale.name, "task_id": 26, "episode_idx": 9, "policy_call_idx": 0},
        {"trace": keep.name, "task_id": 26, "episode_idx": 10, "policy_call_idx": 0},
    ]
    (trace_root / "manifest.jsonl").write_text(
        "".join(__import__("json").dumps(row) + "\n" for row in rows)
    )

    removed = _prune_pending_memory_traces(tmp_path, [(26, 9)])

    assert removed == 2
    assert not stale.exists()
    assert not orphan.exists()
    assert keep.exists()
    assert "episode_010" in (trace_root / "manifest.jsonl").read_text()


def test_pending_queues_preserve_resume_per_task() -> None:
    pending = _build_pending_by_task(
        [6, 7, 8],
        3,
        {6: {1}, 7: set(), 8: {0, 2}},
    )

    assert pending == {
        6: [(6, 0), (6, 2)],
        7: [(7, 0), (7, 1), (7, 2)],
        8: [(8, 1)],
    }


def test_resume_requeues_episode_with_incomplete_staged_calls(tmp_path) -> None:
    task_root = tmp_path / "task18"
    task_root.mkdir()
    path = task_root / "worker_00.jsonl"
    rows = [
        {"task_id": 18, "episode": 0, "lower_timing": [{}, {}]},
        {"task_id": 18, "episode": 1, "lower_timing": [{}]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    trace_keys = {(18, 0, 0), (18, 1, 0)}

    completed = _read_completed(task_root, trace_keys)

    assert completed == {1}
    remaining = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["episode"] for row in remaining] == [1]


def test_task_probe_order_preserves_active_env_then_steals() -> None:
    assert _task_probe_order([6, 7, 8], worker_id=4, active_task_id=None) == [7, 8, 6]
    assert _task_probe_order([6, 7, 8], worker_id=4, active_task_id=8) == [8, 7, 6]


def test_worker_reuses_active_task_then_steals() -> None:
    queues = {6: queue.Queue(), 7: queue.Queue(), 8: queue.Queue()}
    queues[6].put((6, 2))
    queues[7].put((7, 1))

    assert _claim_next_item(queues, [6, 7, 8], worker_id=1, active_task_id=6) == (6, 2)
    assert _claim_next_item(queues, [6, 7, 8], worker_id=1, active_task_id=6) == (7, 1)
    assert _claim_next_item(queues, [6, 7, 8], worker_id=1, active_task_id=7) is None
