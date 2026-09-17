from types import SimpleNamespace

import numpy as np

from optimus_eval.predimem_upper_server import BatchedUpperPlanner
from optimus_eval.predimem_upper_server import PlannerState
from optimus_eval.predimem_upper_server import TaskInfo


def _planner() -> BatchedUpperPlanner:
    planner = BatchedUpperPlanner.__new__(BatchedUpperPlanner)
    planner.upper_guidance = SimpleNamespace(min_subtask_confidence=0.8, interpolation=0.8)
    planner.upper_guidance_history_enabled = True
    planner.upper_guidance_history_lock_observations = 2
    planner.upper_guidance_history_advance_confirmations = 2
    planner.upper_guidance_history_max_rollback = 0.10
    planner.upper_guidance_history_max_advance = 0.35
    return planner


def _state(stage: int = 1, progress: float = 0.4) -> PlannerState:
    return PlannerState(
        task=TaskInfo(6, "task", "task", "scene"),
        current_subtask="task",
        guidance_locked_stage=stage,
        guidance_progress=progress,
    )


def _temporal_planner() -> BatchedUpperPlanner:
    planner = BatchedUpperPlanner.__new__(BatchedUpperPlanner)
    planner.upper_guidance = SimpleNamespace(interpolation=0.8)
    planner.upper_guidance_temporal_posterior = 0.75
    planner.upper_guidance_temporal_purity = 0.35
    planner.upper_guidance_temporal_evidence_decay = 0.95
    planner.upper_guidance_temporal_advance_evidence = 0.45
    planner.upper_guidance_temporal_same_stage_budget = 2
    planner.upper_guidance_temporal_max_rollback = 0.15
    planner.upper_guidance_temporal_max_advance = 0.40
    planner.upper_guidance_feedback_mode = "off"
    planner.upper_guidance_feedback_horizon = 4
    planner.upper_guidance_feedback_confirmations = 2
    planner.upper_guidance_feedback_similarity_tolerance = 0.02
    planner.upper_guidance_feedback_purity_tolerance = 0.10
    planner.upper_guidance_feedback_max_reinforcements = 2
    planner.upper_guidance_feedback_lower_rescue_enabled = False
    planner.upper_guidance_feedback_lower_rescue_confirmations = 2
    planner.upper_guidance_feedback_lower_rescue_min_margin = 0.05
    return planner


def _guidance(
    stage: int,
    progress: float,
    *,
    applied: bool = False,
    purity: float = 0.25,
    candidate: str = "candidate",
    lower_probe_best: str | None = None,
    lower_probe_margin: float = 0.0,
):
    return (
        [(candidate, 0.9)],
        0.0 if not applied else 0.8,
        {
            "enabled": True,
            "applied": applied,
            "reason": "gate-fail" if not applied else "",
            "same_task_confidence": 0.9,
            "same_task_purity": purity,
            "candidate_stage": stage,
            "candidate_progress": progress,
            "candidate": candidate,
            "same_task_similarity": 0.8,
            "lower_probe_available": lower_probe_best is not None,
            "lower_probe_best_candidate": lower_probe_best or "",
            "lower_probe_margin": lower_probe_margin,
        },
    )


def test_history_recovers_high_confidence_same_stage_candidate() -> None:
    candidates, interpolation, meta = _planner()._history_gate(_state(), _guidance(1, 0.45))
    assert candidates
    assert interpolation == 0.8
    assert meta["applied"] is True
    assert meta["history_recovered"] is True
    assert meta["reason"] == "history-recovered-purity"


def test_history_rejects_regression_and_large_jump() -> None:
    planner = _planner()
    _, interpolation, meta = planner._history_gate(_state(), _guidance(0, 0.3))
    assert interpolation == 0.0
    assert meta["reason"] == "history-stage-regression"

    _, interpolation, meta = planner._history_gate(_state(), _guidance(3, 0.6))
    assert interpolation == 0.0
    assert meta["reason"] == "history-stage-jump"


def test_next_stage_requires_two_consecutive_confirmations() -> None:
    planner = _planner()
    state = _state()
    _, first_interpolation, first = planner._history_gate(state, _guidance(2, 0.6))
    _, second_interpolation, second = planner._history_gate(state, _guidance(2, 0.6))
    assert first_interpolation == 0.0
    assert first["reason"] == "history-next-stage-pending"
    assert second_interpolation == 0.8
    assert second["history_stage_advanced"] is True
    assert state.guidance_locked_stage == 2


def test_two_native_outputs_acquire_initial_lock() -> None:
    planner = _planner()
    planner.upper_guidance = SimpleNamespace(
        lookup_stage=lambda task_id, subtask: 0 if task_id == 6 and subtask == "pick" else None,
        task_ids=np.asarray([6, 6]),
        stage_indices=np.asarray([0, 0]),
        anchor_progress=np.asarray([0.0, 0.1], dtype=np.float32),
    )
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    first = {"applied": False}
    second = {"applied": False}
    planner._observe_native_subtask(state, "pick", first, 0)
    planner._observe_native_subtask(state, "pick", second, 5)
    assert state.guidance_locked_stage == 0
    assert state.guidance_progress == 0.0
    assert second["history_lock_acquired"] is True


def test_duplicate_native_step_does_not_acquire_lock() -> None:
    planner = _planner()
    planner.upper_guidance = SimpleNamespace(
        lookup_stage=lambda task_id, subtask: 0,
        task_ids=np.asarray([6]),
        stage_indices=np.asarray([0]),
        anchor_progress=np.asarray([0.0], dtype=np.float32),
    )
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    first = {"applied": False}
    duplicate = {"applied": False}
    planner._observe_native_subtask(state, "pick", first, 0)
    planner._observe_native_subtask(state, "pick", duplicate, 0)
    assert state.guidance_locked_stage is None
    assert duplicate["history_native_observation"] == "duplicate-step"


def test_temporal_gate_limits_same_stage_repetition() -> None:
    planner = _temporal_planner()
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    results = [planner._temporal_gate(state, _guidance(0, 0.1, purity=0.5)) for _ in range(3)]
    assert [result[1] for result in results] == [0.8, 0.8, 0.0]
    assert results[-1][2]["reason"] == "temporal-budget-or-evidence"


def test_temporal_gate_uses_decayed_evidence_to_advance() -> None:
    planner = _temporal_planner()
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    first = planner._temporal_gate(state, _guidance(1, 0.3))
    second = planner._temporal_gate(state, _guidance(1, 0.3))
    assert first[1] == 0.0
    assert second[1] == 0.8
    assert second[2]["temporal_stage_advanced"] is True
    assert state.guidance_temporal_stage == 1


def test_temporal_gate_rejects_stage_regression_and_jump() -> None:
    planner = _temporal_planner()
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    state.guidance_temporal_stage = 1
    state.guidance_temporal_progress = 0.4
    regression = planner._temporal_gate(state, _guidance(0, 0.3))
    jump = planner._temporal_gate(state, _guidance(3, 0.6))
    assert regression[2]["reason"] == "temporal-stage-regression"
    assert jump[2]["reason"] == "temporal-stage-jump"


def test_feedback_shadow_records_observation_without_extending_budget() -> None:
    planner = _temporal_planner()
    planner.upper_guidance_feedback_mode = "shadow"
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    results = [planner._temporal_gate(state, _guidance(0, 0.1, purity=0.5)) for _ in range(3)]
    assert [result[1] for result in results] == [0.8, 0.8, 0.0]
    assert results[-1][2]["feedback_confirmed"] is True
    assert results[-1][2]["feedback_retrieval_supported"] is True


def test_feedback_control_extends_supported_stage_then_expires() -> None:
    planner = _temporal_planner()
    planner.upper_guidance_feedback_mode = "control"
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    results = [planner._temporal_gate(state, _guidance(0, 0.1, purity=0.5)) for _ in range(5)]
    assert [result[1] for result in results] == [0.8, 0.8, 0.8, 0.8, 0.0]
    assert results[2][2]["feedback_reinforced"] is True
    assert results[3][2]["feedback_reinforced"] is True


def test_feedback_control_requires_observation_release_before_stage_switch() -> None:
    planner = _temporal_planner()
    planner.upper_guidance_feedback_mode = "control"
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    planner._temporal_gate(state, _guidance(0, 0.1, purity=0.5))
    first_next = planner._temporal_gate(state, _guidance(1, 0.3, purity=0.5))
    second_next = planner._temporal_gate(state, _guidance(1, 0.3, purity=0.5))
    assert first_next[2]["reason"] == "feedback-current-stage-hold"
    assert second_next[1] == 0.8
    assert second_next[2]["temporal_stage_advanced"] is True


def test_lower_probe_rescues_previous_target_when_retrieval_moves_away() -> None:
    planner = _temporal_planner()
    planner.upper_guidance_feedback_mode = "control"
    planner.upper_guidance_feedback_lower_rescue_enabled = True
    state = PlannerState(TaskInfo(6, "task", "task", "scene"), "task")
    state.guidance_temporal_same_used = 2
    state.guidance_temporal_previous_candidate = 0
    state.guidance_feedback_target = "pick wine bottle"
    state.guidance_feedback_stage = 0
    state.guidance_feedback_progress = 0.1
    state.guidance_feedback_baseline_similarity = 0.8
    state.guidance_feedback_baseline_purity = 0.5
    state.guidance_feedback_baseline_confidence = 0.9

    first = planner._temporal_gate(
        state,
        _guidance(
            0,
            0.2,
            purity=0.5,
            candidate="pour wine into mug 1st",
            lower_probe_best="pick wine bottle",
            lower_probe_margin=0.1,
        ),
    )
    second = planner._temporal_gate(
        state,
        _guidance(
            0,
            0.2,
            purity=0.5,
            candidate="pour wine into mug 1st",
            lower_probe_best="pick wine bottle",
            lower_probe_margin=0.1,
        ),
    )

    assert first[2]["feedback_lower_rescue_confirmed"] is False
    assert second[0][0][0] == "pick wine bottle"
    assert second[2]["feedback_lower_rescue_applied"] is True
    assert second[2]["feedback_reinforced"] is True
