from __future__ import annotations

import numpy as np

from optimus_eval.trajectory_quality_audit import (
    Trajectory,
    analyze_mixed_dataset,
    analyze_task,
)


def _trajectory(episode: str, actions: np.ndarray, boundary: int | None = None) -> Trajectory:
    states = np.cumsum(np.pad(actions[:, :2], ((0, 0), (0, 1))), axis=0)
    return Trajectory(
        dataset="synthetic",
        task_id=0,
        task_name="synthetic task",
        episode_id=episode,
        actions=actions,
        states=states,
        boundaries=() if boundary is None else (boundary,),
    )


def test_behavior_alignment_detects_inserted_pause() -> None:
    first = np.tile(np.array([[1.0, 0.0, 0.0]]), (30, 1))
    second = np.tile(np.array([[0.0, 1.0, 0.0]]), (30, 1))
    pause = np.zeros((30, 3))
    trajectories = [
        _trajectory("a", np.concatenate([first, second])),
        _trajectory("b", np.concatenate([first, pause, second])),
    ]

    result = analyze_task(trajectories)

    assert result["action_dispersion"]["behavior_alignment_gain"] > 0.25
    assert result["progress"]["time_warp_std_at_behavior_progress_mean"] > 0.02


def test_boundary_stationary_excess_detects_subtask_gap() -> None:
    moving = np.tile(np.array([[1.0, 0.0, 0.0]]), (40, 1))
    pause = np.zeros((20, 3))
    actions = np.concatenate([moving, pause, moving])
    trajectories = [_trajectory(str(index), actions, boundary=50) for index in range(3)]

    result = analyze_task(trajectories)

    assert result["boundaries"]["stationary_excess"] > 0.5
    assert result["boundaries"]["pause_length_mean"] == 20.0


def test_identical_trajectories_have_zero_progress_spread() -> None:
    actions = np.tile(np.array([[0.5, 0.25, 0.0]]), (60, 1))
    trajectories = [_trajectory(str(index), actions) for index in range(3)]

    result = analyze_task(trajectories)

    assert result["progress"]["time_warp_std_at_behavior_progress_mean"] < 1e-12
    assert result["progress"]["same_absolute_step_behavior_progress_variance_mean"] < 1e-12


def test_zero_action_while_state_moves_is_not_strict_idle() -> None:
    actions = np.concatenate(
        [np.zeros((20, 3)), np.tile(np.array([[1.0, 0.0, 0.0]]), (20, 1))]
    )
    states = np.stack([np.arange(40), np.zeros(40), np.zeros(40)], axis=1).astype(np.float64)
    trajectories = [
        Trajectory("synthetic", 0, "moving state", str(index), actions, states)
        for index in range(3)
    ]

    result = analyze_task(trajectories)

    assert result["smoothness"]["exact_zero_while_state_moving_ratio"] == 0.5
    assert result["smoothness"]["exact_zero_while_state_moving_steps"] == 20.0
    assert result["smoothness"]["exact_zero_while_state_moving_steps_per_segment"] == 20.0
    assert result["smoothness"]["exact_zero_while_state_moving_replan_windows_any"] == 2.0
    assert result["smoothness"]["exact_zero_while_state_moving_replan_windows_majority"] == 2.0
    assert result["smoothness"]["idle_ratio"] == 0.0


def test_mixed_analysis_keeps_task_variance_separate_from_quality() -> None:
    task_zero = np.tile(np.array([[1.0, 0.0, 0.0]]), (60, 1))
    task_one = np.tile(np.array([[0.0, 1.0, 0.0]]), (60, 1))
    trajectories = [
        _trajectory("a", task_zero),
        _trajectory("b", task_zero),
        Trajectory("synthetic", 1, "other", "c", task_one, np.cumsum(task_one, axis=0)),
        Trajectory("synthetic", 1, "other", "d", task_one, np.cumsum(task_one, axis=0)),
    ]

    result = analyze_mixed_dataset(trajectories)

    assert result["action_time"]["between_task_fraction_mean"] > 0.99
    assert result["action_time"]["task_nearest_centroid_accuracy"] == 1.0
    assert "not a quality score" in result["interpretation"]
