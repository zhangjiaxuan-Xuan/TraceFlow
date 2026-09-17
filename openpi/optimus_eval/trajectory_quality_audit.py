from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Trajectory:
    dataset: str
    task_id: int
    task_name: str
    episode_id: str
    actions: np.ndarray
    states: np.ndarray
    boundaries: tuple[int, ...] = ()


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()) if array.size else float("nan")


def _safe_p95(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, 95)) if array.size else float("nan")


def _resample(values: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if len(values) == 1:
        return np.repeat(values, len(target), axis=0)
    keep = np.r_[source[1:] > source[:-1], True]
    source_unique = source[keep]
    values_unique = values[keep]
    if len(source_unique) == 1:
        return np.repeat(values_unique, len(target), axis=0)
    return np.stack(
        [np.interp(target, source_unique, values_unique[:, dim]) for dim in range(values.shape[1])],
        axis=1,
    )


def _behavior_progress(actions: np.ndarray) -> np.ndarray:
    arm = np.asarray(actions[:, :-1] if actions.shape[1] > 1 else actions, dtype=np.float64)
    effort = np.linalg.norm(arm, axis=1)
    cumulative = np.cumsum(effort)
    cumulative -= cumulative[0]
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(actions), dtype=np.float64)
    return cumulative / total


def _normalized_dispersion(stack: np.ndarray) -> tuple[float, np.ndarray]:
    centroid = stack.mean(axis=0, keepdims=True)
    curve = np.mean(np.square(stack - centroid), axis=(0, 2))
    pooled = stack.reshape(-1, stack.shape[-1])
    denominator = float(np.mean(np.square(pooled - pooled.mean(axis=0, keepdims=True))))
    ratio_curve = curve / max(denominator, 1e-12)
    return float(ratio_curve.mean()), ratio_curve


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _run_containing(mask: np.ndarray, center: int) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not len(mask):
        return 0
    center = int(np.clip(center, 0, len(mask) - 1))
    if not mask[center]:
        return 0
    left = center
    right = center
    while left > 0 and mask[left - 1]:
        left -= 1
    while right + 1 < len(mask) and mask[right + 1]:
        right += 1
    return right - left + 1


def analyze_task(trajectories: list[Trajectory], grid_size: int = 101) -> dict[str, Any]:
    if len(trajectories) < 2:
        raise ValueError("At least two trajectories are required per task")
    task_id = trajectories[0].task_id
    if any(item.task_id != task_id for item in trajectories):
        raise ValueError("analyze_task received mixed task IDs")

    target = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    action_dim = min(item.actions.shape[1] for item in trajectories)
    state_dim = min(item.states.shape[1] for item in trajectories)
    arm_dim = max(1, action_dim - 1)
    actions = [np.asarray(item.actions[:, :action_dim], dtype=np.float64) for item in trajectories]
    states = [np.asarray(item.states[:, :state_dim], dtype=np.float64) for item in trajectories]
    behavior = [_behavior_progress(value) for value in actions]
    time_axes = [np.linspace(0.0, 1.0, len(value), dtype=np.float64) for value in actions]

    action_time = np.stack(
        [_resample(value[:, :arm_dim], time, target) for value, time in zip(actions, time_axes, strict=True)]
    )
    action_behavior = np.stack(
        [_resample(value[:, :arm_dim], progress, target) for value, progress in zip(actions, behavior, strict=True)]
    )
    state_time = np.stack(
        [_resample(value, time, target) for value, time in zip(states, time_axes, strict=True)]
    )
    state_behavior = np.stack(
        [_resample(value, progress, target) for value, progress in zip(states, behavior, strict=True)]
    )
    behavior_at_time = np.stack(
        [np.interp(target, time, progress) for time, progress in zip(time_axes, behavior, strict=True)]
    )
    time_at_behavior = np.stack(
        [np.interp(target, progress, time) for time, progress in zip(time_axes, behavior, strict=True)]
    )

    action_time_ratio, action_time_curve = _normalized_dispersion(action_time)
    action_behavior_ratio, action_behavior_curve = _normalized_dispersion(action_behavior)
    state_time_ratio, state_time_curve = _normalized_dispersion(state_time)
    state_behavior_ratio, state_behavior_curve = _normalized_dispersion(state_behavior)

    min_length = min(len(value) for value in actions)
    same_step_progress = np.stack([value[:min_length] for value in behavior])
    same_step_progress_var = np.var(same_step_progress, axis=0)

    all_speed = np.concatenate([np.linalg.norm(value[:, :arm_dim], axis=1) for value in actions])
    stationary_threshold = max(1e-8, 0.05 * float(np.percentile(all_speed, 90)))
    pooled_state = np.concatenate(states, axis=0)
    state_scale = np.std(pooled_state, axis=0)
    state_scale[state_scale < 1e-6] = 1.0
    pooled_state_speed = np.concatenate(
        [np.linalg.norm(np.diff(value / state_scale, axis=0), axis=1) for value in states]
    )
    state_stationary_threshold = max(
        1e-8, 0.05 * float(np.percentile(pooled_state_speed, 90))
    )

    trajectory_metrics: list[dict[str, float]] = []
    boundary_stationary: list[float] = []
    boundary_pause_lengths: list[float] = []
    boundary_action_jumps: list[float] = []
    boundary_state_jumps: list[float] = []
    pre_boundary_stationary: list[float] = []
    post_boundary_stationary: list[float] = []
    nonboundary_stationary: list[float] = []
    normalized_boundaries: defaultdict[int, list[float]] = defaultdict(list)
    segment_stationary_profiles: list[np.ndarray] = []
    segment_idle_profiles: list[np.ndarray] = []
    pre_boundary_idle: list[float] = []
    post_boundary_idle: list[float] = []

    for item, action, state in zip(trajectories, actions, states, strict=True):
        arm = action[:, :arm_dim]
        speed = np.linalg.norm(arm, axis=1)
        acceleration = np.linalg.norm(np.diff(arm, axis=0), axis=1)
        jerk = np.linalg.norm(np.diff(arm, n=2, axis=0), axis=1)
        state_velocity = np.linalg.norm(np.diff(state / state_scale, axis=0), axis=1)
        state_velocity_per_frame = np.r_[state_velocity[0] if len(state_velocity) else 0.0, state_velocity]
        stationary = speed <= stationary_threshold
        exact_zero_arm = speed <= 1e-8
        state_moving = state_velocity_per_frame > state_stationary_threshold
        idle = stationary & ~state_moving
        stationary_while_state_moving = stationary & state_moving
        exact_zero_while_state_moving = exact_zero_arm & state_moving
        interior = stationary.copy()
        edge = max(1, int(round(0.05 * len(stationary))))
        interior[:edge] = False
        interior[-edge:] = False
        nonboundary_mask = np.ones(len(stationary), dtype=bool)
        for boundary_index, boundary in enumerate(item.boundaries, start=1):
            boundary = int(boundary)
            normalized_boundaries[boundary_index].append(boundary / max(len(action) - 1, 1))
            left = max(0, boundary - 15)
            right = min(len(stationary), boundary + 16)
            boundary_stationary.append(float(stationary[left:right].mean()))
            boundary_pause_lengths.append(float(_run_containing(stationary, boundary)))
            pre_boundary_stationary.append(float(stationary[max(0, boundary - 60) : boundary].mean()))
            post_boundary_stationary.append(float(stationary[boundary : min(len(stationary), boundary + 60)].mean()))
            pre_boundary_idle.append(float(idle[max(0, boundary - 60) : boundary].mean()))
            post_boundary_idle.append(float(idle[boundary : min(len(idle), boundary + 60)].mean()))
            nonboundary_mask[max(0, boundary - 30) : min(len(stationary), boundary + 31)] = False
            if 0 < boundary < len(action):
                boundary_action_jumps.append(
                    float(np.linalg.norm(arm[boundary] - arm[boundary - 1]))
                )
                boundary_state_jumps.append(
                    float(np.linalg.norm((state[boundary] - state[boundary - 1]) / state_scale))
                )
        if nonboundary_mask.any():
            nonboundary_stationary.append(float(stationary[nonboundary_mask].mean()))
        segment_edges = [0, *item.boundaries, len(stationary)]
        idle_steps_by_segment = []
        stationary_moving_steps_by_segment = []
        exact_zero_moving_steps_by_segment = []
        for left, right in zip(segment_edges[:-1], segment_edges[1:], strict=True):
            segment = stationary[int(left) : int(right)].astype(np.float64)[:, None]
            idle_segment = idle[int(left) : int(right)].astype(np.float64)[:, None]
            idle_steps_by_segment.append(int(idle[int(left) : int(right)].sum()))
            stationary_moving_steps_by_segment.append(
                int(stationary_while_state_moving[int(left) : int(right)].sum())
            )
            exact_zero_moving_steps_by_segment.append(
                int(exact_zero_while_state_moving[int(left) : int(right)].sum())
            )
            if len(segment):
                segment_stationary_profiles.append(
                    _resample(
                        segment,
                        np.linspace(0.0, 1.0, len(segment)),
                        target,
                    )[:, 0]
                )
                segment_idle_profiles.append(
                    _resample(
                        idle_segment,
                        np.linspace(0.0, 1.0, len(idle_segment)),
                        target,
                    )[:, 0]
                )
        speed_p90 = max(float(np.percentile(speed, 90)), 1e-12)

        def replan_windows(mask: np.ndarray, *, majority: bool = False) -> int:
            count = 0
            for start in range(0, len(mask), 10):
                window = mask[start : start + 10]
                count += int(window.mean() >= 0.5 if majority else window.any())
            return count

        trajectory_metrics.append(
            {
                "length": float(len(action)),
                "segment_count": float(len(segment_edges) - 1),
                "speed_mean": float(speed.mean()),
                "speed_variance": float(speed.var()),
                "speed_cv": float(speed.std() / max(speed.mean(), 1e-12)),
                "acceleration_rms": float(np.sqrt(np.mean(np.square(acceleration)))) if len(acceleration) else 0.0,
                "jerk_rms": float(np.sqrt(np.mean(np.square(jerk)))) if len(jerk) else 0.0,
                "acceleration_rms_relative": (
                    float(np.sqrt(np.mean(np.square(acceleration))) / speed_p90)
                    if len(acceleration)
                    else 0.0
                ),
                "jerk_rms_relative": (
                    float(np.sqrt(np.mean(np.square(jerk))) / speed_p90)
                    if len(jerk)
                    else 0.0
                ),
                "state_velocity_variance": float(state_velocity.var()) if len(state_velocity) else 0.0,
                "stationary_ratio": float(stationary.mean()),
                "exact_zero_arm_action_ratio": float(exact_zero_arm.mean()),
                "idle_ratio": float(idle.mean()),
                "stationary_while_state_moving_ratio": float(stationary_while_state_moving.mean()),
                "exact_zero_while_state_moving_ratio": float(exact_zero_while_state_moving.mean()),
                "idle_steps": float(idle.sum()),
                "exact_zero_arm_action_steps": float(exact_zero_arm.sum()),
                "stationary_while_state_moving_steps": float(stationary_while_state_moving.sum()),
                "exact_zero_while_state_moving_steps": float(exact_zero_while_state_moving.sum()),
                "idle_steps_per_segment": float(np.mean(idle_steps_by_segment)),
                "stationary_while_state_moving_steps_per_segment": float(
                    np.mean(stationary_moving_steps_by_segment)
                ),
                "exact_zero_while_state_moving_steps_per_segment": float(
                    np.mean(exact_zero_moving_steps_by_segment)
                ),
                "idle_replan_windows_any": float(replan_windows(idle)),
                "idle_replan_windows_majority": float(replan_windows(idle, majority=True)),
                "exact_zero_while_state_moving_replan_windows_any": float(
                    replan_windows(exact_zero_while_state_moving)
                ),
                "exact_zero_while_state_moving_replan_windows_majority": float(
                    replan_windows(exact_zero_while_state_moving, majority=True)
                ),
                "interior_stationary_ratio": float(interior.sum() / max(len(interior) - 2 * edge, 1)),
                "longest_stationary_run": float(_longest_true_run(stationary)),
                "longest_idle_run": float(_longest_true_run(idle)),
            }
        )

    def metric(name: str) -> float:
        return _safe_mean(row[name] for row in trajectory_metrics)

    boundary_position_std = [float(np.std(values)) for values in normalized_boundaries.values()]
    segment_stationary_profile = (
        np.mean(segment_stationary_profiles, axis=0)
        if segment_stationary_profiles
        else np.full(grid_size, np.nan)
    )
    segment_idle_profile = (
        np.mean(segment_idle_profiles, axis=0)
        if segment_idle_profiles
        else np.full(grid_size, np.nan)
    )
    action_alignment_gain = 1.0 - action_behavior_ratio / max(action_time_ratio, 1e-12)
    state_alignment_gain = 1.0 - state_behavior_ratio / max(state_time_ratio, 1e-12)
    return {
        "dataset": trajectories[0].dataset,
        "task_id": task_id,
        "task_name": trajectories[0].task_name,
        "episodes": len(trajectories),
        "trajectory_length_mean": metric("length"),
        "trajectory_length_std": float(np.std([row["length"] for row in trajectory_metrics])),
        "trajectory_length_cv": float(
            np.std([row["length"] for row in trajectory_metrics])
            / max(np.mean([row["length"] for row in trajectory_metrics]), 1e-12)
        ),
        "progress": {
            "time_vs_behavior_mae": float(np.mean(np.abs(behavior_at_time - target[None, :]))),
            "behavior_progress_std_at_normalized_time_mean": float(np.std(behavior_at_time, axis=0).mean()),
            "behavior_progress_std_at_normalized_time_p95": float(np.percentile(np.std(behavior_at_time, axis=0), 95)),
            "time_warp_std_at_behavior_progress_mean": float(np.std(time_at_behavior, axis=0).mean()),
            "time_warp_std_at_behavior_progress_p95": float(np.percentile(np.std(time_at_behavior, axis=0), 95)),
            "same_absolute_step_behavior_progress_variance_mean": float(same_step_progress_var.mean()),
            "same_absolute_step_behavior_progress_variance_p95": float(np.percentile(same_step_progress_var, 95)),
            "stage_boundary_normalized_position_std_mean": _safe_mean(boundary_position_std),
            "stage_boundary_normalized_position_std_p95": _safe_p95(boundary_position_std),
        },
        "action_dispersion": {
            "normalized_time": action_time_ratio,
            "behavior_progress": action_behavior_ratio,
            "behavior_alignment_gain": action_alignment_gain,
        },
        "state_dispersion": {
            "normalized_time": state_time_ratio,
            "behavior_progress": state_behavior_ratio,
            "behavior_alignment_gain": state_alignment_gain,
        },
        "smoothness": {
            "segment_count": metric("segment_count"),
            "speed_mean": metric("speed_mean"),
            "speed_variance": metric("speed_variance"),
            "speed_cv": metric("speed_cv"),
            "acceleration_rms": metric("acceleration_rms"),
            "jerk_rms": metric("jerk_rms"),
            "acceleration_rms_relative": metric("acceleration_rms_relative"),
            "jerk_rms_relative": metric("jerk_rms_relative"),
            "state_velocity_variance": metric("state_velocity_variance"),
            "stationary_ratio": metric("stationary_ratio"),
            "exact_zero_arm_action_ratio": metric("exact_zero_arm_action_ratio"),
            "idle_ratio": metric("idle_ratio"),
            "stationary_while_state_moving_ratio": metric("stationary_while_state_moving_ratio"),
            "exact_zero_while_state_moving_ratio": metric("exact_zero_while_state_moving_ratio"),
            "idle_steps": metric("idle_steps"),
            "exact_zero_arm_action_steps": metric("exact_zero_arm_action_steps"),
            "stationary_while_state_moving_steps": metric("stationary_while_state_moving_steps"),
            "exact_zero_while_state_moving_steps": metric("exact_zero_while_state_moving_steps"),
            "idle_steps_per_segment": metric("idle_steps_per_segment"),
            "stationary_while_state_moving_steps_per_segment": metric(
                "stationary_while_state_moving_steps_per_segment"
            ),
            "exact_zero_while_state_moving_steps_per_segment": metric(
                "exact_zero_while_state_moving_steps_per_segment"
            ),
            "idle_replan_windows_any": metric("idle_replan_windows_any"),
            "idle_replan_windows_majority": metric("idle_replan_windows_majority"),
            "exact_zero_while_state_moving_replan_windows_any": metric(
                "exact_zero_while_state_moving_replan_windows_any"
            ),
            "exact_zero_while_state_moving_replan_windows_majority": metric(
                "exact_zero_while_state_moving_replan_windows_majority"
            ),
            "interior_stationary_ratio": metric("interior_stationary_ratio"),
            "longest_stationary_run": metric("longest_stationary_run"),
            "longest_idle_run": metric("longest_idle_run"),
        },
        "boundaries": {
            "count": int(sum(len(item.boundaries) for item in trajectories)),
            "stationary_ratio": _safe_mean(boundary_stationary),
            "nonboundary_stationary_ratio": _safe_mean(nonboundary_stationary),
            "stationary_excess": (
                _safe_mean(boundary_stationary) - _safe_mean(nonboundary_stationary)
                if boundary_stationary and nonboundary_stationary
                else float("nan")
            ),
            "pause_length_mean": _safe_mean(boundary_pause_lengths),
            "pause_length_p95": _safe_p95(boundary_pause_lengths),
            "pre_boundary_stationary_ratio_60": _safe_mean(pre_boundary_stationary),
            "post_boundary_stationary_ratio_60": _safe_mean(post_boundary_stationary),
            "pre_boundary_idle_ratio_60": _safe_mean(pre_boundary_idle),
            "post_boundary_idle_ratio_60": _safe_mean(post_boundary_idle),
            "segment_start_stationary_ratio": float(np.nanmean(segment_stationary_profile[:11])),
            "segment_middle_stationary_ratio": float(np.nanmean(segment_stationary_profile[11:90])),
            "segment_end_stationary_ratio": float(np.nanmean(segment_stationary_profile[90:])),
            "segment_edge_stationary_excess": float(
                0.5
                * (
                    np.nanmean(segment_stationary_profile[:11])
                    + np.nanmean(segment_stationary_profile[90:])
                )
                - np.nanmean(segment_stationary_profile[11:90])
            ),
            "segment_start_idle_ratio": float(np.nanmean(segment_idle_profile[:11])),
            "segment_middle_idle_ratio": float(np.nanmean(segment_idle_profile[11:90])),
            "segment_end_idle_ratio": float(np.nanmean(segment_idle_profile[90:])),
            "action_jump_mean": _safe_mean(boundary_action_jumps),
            "state_jump_mean_normalized": _safe_mean(boundary_state_jumps),
        },
        "curves": {
            "progress_grid": target.tolist(),
            "action_dispersion_normalized_time": action_time_curve.tolist(),
            "action_dispersion_behavior_progress": action_behavior_curve.tolist(),
            "state_dispersion_normalized_time": state_time_curve.tolist(),
            "state_dispersion_behavior_progress": state_behavior_curve.tolist(),
            "behavior_progress_std_at_normalized_time": np.std(behavior_at_time, axis=0).tolist(),
            "time_warp_std_at_behavior_progress": np.std(time_at_behavior, axis=0).tolist(),
            "segment_stationary_profile": segment_stationary_profile.tolist(),
            "segment_idle_profile": segment_idle_profile.tolist(),
        },
    }


def load_libero(
    root: Path,
    task_ids: set[int],
    episodes_per_task: int,
) -> list[Trajectory]:
    tasks_table = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    task_names = {
        int(row["task_index"]): str(row["__index_level_0__"]) for row in tasks_table
    }
    grouped: defaultdict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    counts: defaultdict[int, set[int]] = defaultdict(set)
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(
            path,
            columns=["observation.state", "action", "frame_index", "episode_index", "task_index"],
        ).to_pylist()
        for row in table:
            task_id = int(row["task_index"])
            episode_id = int(row["episode_index"])
            if task_id not in task_ids:
                continue
            if episode_id not in counts[task_id] and len(counts[task_id]) >= episodes_per_task:
                continue
            counts[task_id].add(episode_id)
            grouped[(task_id, episode_id)].append(
                (
                    int(row["frame_index"]),
                    np.asarray(row["action"], dtype=np.float64),
                    np.asarray(row["observation.state"], dtype=np.float64),
                )
            )
        if all(len(counts[task_id]) >= episodes_per_task for task_id in task_ids):
            break
    missing = {task_id: len(counts[task_id]) for task_id in task_ids if len(counts[task_id]) < episodes_per_task}
    if missing:
        raise RuntimeError(f"Insufficient LIBERO episodes: {missing}")
    trajectories = []
    for (task_id, episode_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row[0])
        trajectories.append(
            Trajectory(
                dataset="LIBERO-10",
                task_id=task_id,
                task_name=task_names[task_id],
                episode_id=str(episode_id),
                actions=np.stack([row[1] for row in rows]),
                states=np.stack([row[2] for row in rows]),
            )
        )
    return trajectories


def _load_arena_trajectory(record: dict[str, Any]) -> Trajectory:
    actions = []
    states = []
    boundaries = []
    offset = 0
    for path_string in record["segment_paths"]:
        with h5py.File(path_string, "r") as handle:
            demo = handle["data/demo_0"]
            action = np.asarray(demo["actions"], dtype=np.float64)
            obs = demo["obs"]
            if "ee_states" in obs:
                state = np.asarray(obs["ee_states"], dtype=np.float64)
            else:
                state = np.concatenate(
                    [np.asarray(obs["ee_pos"]), np.asarray(obs["ee_ori"])], axis=1
                )
            if "gripper_states" in obs:
                gripper = np.asarray(obs["gripper_states"], dtype=np.float64).mean(axis=1, keepdims=True)
                state = np.concatenate([state, gripper], axis=1)
        if actions:
            boundaries.append(offset)
        actions.append(action)
        states.append(state)
        offset += len(action)
    return Trajectory(
        dataset="RoboMemArena-Extra8",
        task_id=int(record["task_id"]),
        task_name=str(record["prompt"]),
        episode_id=str(record["action_id"]),
        actions=np.concatenate(actions, axis=0),
        states=np.concatenate(states, axis=0),
        boundaries=tuple(boundaries),
    )


def load_arena(
    manifest: Path,
    task_ids: set[int],
    episodes_per_task: int,
    workers: int,
) -> list[Trajectory]:
    records: defaultdict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task_id = int(row["task_id"])
            if task_id in task_ids:
                records[task_id].setdefault(str(row["action_id"]), row)
    selected = []
    for task_id in sorted(task_ids):
        candidates = [records[task_id][key] for key in sorted(records[task_id])]
        if len(candidates) < episodes_per_task:
            raise RuntimeError(
                f"Task {task_id} has only {len(candidates)} Arena trajectories"
            )
        selected.extend(candidates[:episodes_per_task])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_load_arena_trajectory, selected))


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": record["dataset"],
        "task_id": record["task_id"],
        "episodes": record["episodes"],
        "length_mean": record["trajectory_length_mean"],
        "length_cv": record["trajectory_length_cv"],
        "segment_count": record["smoothness"]["segment_count"],
        "time_behavior_mae": record["progress"]["time_vs_behavior_mae"],
        "time_warp_std": record["progress"]["time_warp_std_at_behavior_progress_mean"],
        "same_step_progress_var": record["progress"]["same_absolute_step_behavior_progress_variance_mean"],
        "action_dispersion_time": record["action_dispersion"]["normalized_time"],
        "action_dispersion_behavior": record["action_dispersion"]["behavior_progress"],
        "action_alignment_gain": record["action_dispersion"]["behavior_alignment_gain"],
        "state_alignment_gain": record["state_dispersion"]["behavior_alignment_gain"],
        "speed_cv": record["smoothness"]["speed_cv"],
        "acceleration_rms": record["smoothness"]["acceleration_rms"],
        "jerk_rms": record["smoothness"]["jerk_rms"],
        "acceleration_rms_relative": record["smoothness"]["acceleration_rms_relative"],
        "jerk_rms_relative": record["smoothness"]["jerk_rms_relative"],
        "stationary_ratio": record["smoothness"]["stationary_ratio"],
        "exact_zero_arm_action_ratio": record["smoothness"]["exact_zero_arm_action_ratio"],
        "idle_ratio": record["smoothness"]["idle_ratio"],
        "stationary_while_state_moving_ratio": record["smoothness"]["stationary_while_state_moving_ratio"],
        "exact_zero_while_state_moving_ratio": record["smoothness"]["exact_zero_while_state_moving_ratio"],
        "idle_steps": record["smoothness"]["idle_steps"],
        "exact_zero_arm_action_steps": record["smoothness"]["exact_zero_arm_action_steps"],
        "stationary_while_state_moving_steps": record["smoothness"]["stationary_while_state_moving_steps"],
        "exact_zero_while_state_moving_steps": record["smoothness"]["exact_zero_while_state_moving_steps"],
        "idle_steps_per_segment": record["smoothness"]["idle_steps_per_segment"],
        "stationary_while_state_moving_steps_per_segment": record["smoothness"]["stationary_while_state_moving_steps_per_segment"],
        "exact_zero_while_state_moving_steps_per_segment": record["smoothness"]["exact_zero_while_state_moving_steps_per_segment"],
        "idle_replan_windows_any": record["smoothness"]["idle_replan_windows_any"],
        "idle_replan_windows_majority": record["smoothness"]["idle_replan_windows_majority"],
        "exact_zero_while_state_moving_replan_windows_any": record["smoothness"]["exact_zero_while_state_moving_replan_windows_any"],
        "exact_zero_while_state_moving_replan_windows_majority": record["smoothness"]["exact_zero_while_state_moving_replan_windows_majority"],
        "longest_stationary_run": record["smoothness"]["longest_stationary_run"],
        "longest_idle_run": record["smoothness"]["longest_idle_run"],
        "boundary_stationary_excess": record["boundaries"]["stationary_excess"],
        "boundary_pause_length": record["boundaries"]["pause_length_mean"],
        "pre_boundary_stationary": record["boundaries"]["pre_boundary_stationary_ratio_60"],
        "post_boundary_stationary": record["boundaries"]["post_boundary_stationary_ratio_60"],
        "pre_boundary_idle": record["boundaries"]["pre_boundary_idle_ratio_60"],
        "post_boundary_idle": record["boundaries"]["post_boundary_idle_ratio_60"],
        "segment_start_stationary": record["boundaries"]["segment_start_stationary_ratio"],
        "segment_middle_stationary": record["boundaries"]["segment_middle_stationary_ratio"],
        "segment_end_stationary": record["boundaries"]["segment_end_stationary_ratio"],
        "segment_edge_stationary_excess": record["boundaries"]["segment_edge_stationary_excess"],
        "segment_start_idle": record["boundaries"]["segment_start_idle_ratio"],
        "segment_middle_idle": record["boundaries"]["segment_middle_idle_ratio"],
        "segment_end_idle": record["boundaries"]["segment_end_idle_ratio"],
        "boundary_state_jump": record["boundaries"]["state_jump_mean_normalized"],
    }


def _dataset_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    flat = [_flatten_record(record) for record in records]
    numeric = [key for key in flat[0] if key not in {"dataset", "task_id"}]
    result = {"tasks": len(records), "dataset": records[0]["dataset"]}
    for key in numeric:
        values = np.asarray([row[key] for row in flat], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = float(finite.mean()) if len(finite) else float("nan")
    return result


def analyze_mixed_dataset(
    trajectories: list[Trajectory], grid_size: int = 101
) -> dict[str, Any]:
    """Measure task-conditioned structure without treating task mixture as noise."""
    task_ids = sorted({item.task_id for item in trajectories})
    if len(task_ids) < 2:
        raise ValueError("Mixed analysis requires at least two tasks")
    target = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    action_dim = min(item.actions.shape[1] for item in trajectories)
    arm_dim = max(1, action_dim - 1)
    state_dim = min(item.states.shape[1] for item in trajectories)
    labels = np.asarray([item.task_id for item in trajectories], dtype=np.int64)
    action_time = []
    action_behavior = []
    state_time = []
    for item in trajectories:
        action = np.asarray(item.actions[:, :action_dim], dtype=np.float64)
        state = np.asarray(item.states[:, :state_dim], dtype=np.float64)
        time = np.linspace(0.0, 1.0, len(action), dtype=np.float64)
        behavior = _behavior_progress(action)
        action_time.append(_resample(action[:, :arm_dim], time, target))
        action_behavior.append(_resample(action[:, :arm_dim], behavior, target))
        state_time.append(_resample(state, time, target))
    action_time_array = np.stack(action_time)
    action_behavior_array = np.stack(action_behavior)
    state_time_array = np.stack(state_time)

    def decomposition(stack: np.ndarray) -> tuple[float, float, list[float]]:
        global_mean = stack.mean(axis=0, keepdims=True)
        total_curve = np.mean(np.square(stack - global_mean), axis=(0, 2))
        within_sum = np.zeros(grid_size, dtype=np.float64)
        for task_id in task_ids:
            subset = stack[labels == task_id]
            within_sum += len(subset) * np.mean(
                np.square(subset - subset.mean(axis=0, keepdims=True)), axis=(0, 2)
            )
        within_curve = within_sum / len(stack)
        between_fraction = np.clip(
            1.0 - within_curve / np.maximum(total_curve, 1e-12), 0.0, 1.0
        )
        return (
            float(within_curve.mean()),
            float(total_curve.mean()),
            between_fraction.tolist(),
        )

    def nearest_task_accuracy(stack: np.ndarray) -> float:
        flattened = stack.reshape(len(stack), -1)
        scale = np.std(flattened, axis=0)
        scale[scale < 1e-6] = 1.0
        correct = 0
        for index, row in enumerate(flattened):
            distances = {}
            for task_id in task_ids:
                mask = labels == task_id
                mask[index] = False
                candidates = flattened[mask]
                if not len(candidates):
                    continue
                centroid = candidates.mean(axis=0)
                distances[task_id] = float(np.mean(np.square((row - centroid) / scale)))
            correct += int(min(distances, key=distances.get) == int(labels[index]))
        return float(correct / len(flattened))

    action_time_within, action_time_total, action_time_between = decomposition(
        action_time_array
    )
    action_behavior_within, action_behavior_total, action_behavior_between = decomposition(
        action_behavior_array
    )
    state_time_within, state_time_total, state_time_between = decomposition(state_time_array)
    return {
        "dataset": trajectories[0].dataset,
        "tasks": len(task_ids),
        "episodes": len(trajectories),
        "task_ids": task_ids,
        "action_time": {
            "within_task_variance": action_time_within,
            "total_mixed_variance": action_time_total,
            "between_task_fraction_mean": float(np.mean(action_time_between)),
            "task_nearest_centroid_accuracy": nearest_task_accuracy(action_time_array),
        },
        "action_behavior": {
            "within_task_variance": action_behavior_within,
            "total_mixed_variance": action_behavior_total,
            "between_task_fraction_mean": float(np.mean(action_behavior_between)),
            "task_nearest_centroid_accuracy": nearest_task_accuracy(action_behavior_array),
        },
        "state_time": {
            "within_task_variance": state_time_within,
            "total_mixed_variance": state_time_total,
            "between_task_fraction_mean": float(np.mean(state_time_between)),
            "task_nearest_centroid_accuracy": nearest_task_accuracy(state_time_array),
        },
        "curves": {
            "progress_grid": target.tolist(),
            "action_time_between_task_fraction": action_time_between,
            "action_behavior_between_task_fraction": action_behavior_between,
            "state_time_between_task_fraction": state_time_between,
        },
        "interpretation": (
            "Total mixed variance is not a quality score. Between-task fraction and nearest-centroid "
            "accuracy measure task structure; per-task records remain the quality measurements."
        ),
    }


def _write_plots(
    records: list[dict[str, Any]], mixed: list[dict[str, Any]], output_dir: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["dataset"]].append(record)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for dataset, items in grouped.items():
        grid = np.asarray(items[0]["curves"]["progress_grid"])
        action_time = np.mean(
            [item["curves"]["action_dispersion_normalized_time"] for item in items], axis=0
        )
        action_behavior = np.mean(
            [item["curves"]["action_dispersion_behavior_progress"] for item in items], axis=0
        )
        warp = np.mean(
            [item["curves"]["time_warp_std_at_behavior_progress"] for item in items], axis=0
        )
        axes[0].plot(grid, action_time, label=f"{dataset}: time")
        axes[0].plot(grid, action_behavior, linestyle="--", label=f"{dataset}: behavior")
        axes[1].plot(grid, warp, label=dataset)
    axes[0].set(title="Cross-episode action dispersion", xlabel="Progress", ylabel="Normalized variance")
    axes[1].set(title="Time-warp spread", xlabel="Behavior progress", ylabel="Std of normalized time")
    labels = list(grouped)
    x = np.arange(len(labels))
    width = 0.25
    for offset, (key, label) in enumerate(
        [("speed_cv", "Speed CV"), ("stationary_ratio", "Stationary ratio"), ("time_behavior_mae", "Time-behavior MAE")]
    ):
        axes[2].bar(
            x + (offset - 1) * width,
            [np.mean([_flatten_record(item)[key] for item in grouped[name]]) for name in labels],
            width,
            label=label,
        )
    axes[2].set(title="Trajectory quality summary", xticks=x, xticklabels=labels)
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_quality_overview.png", dpi=180)
    plt.close(fig)

    arena = grouped.get("RoboMemArena-Extra8", [])
    if arena:
        fig, axis = plt.subplots(figsize=(10, 4.5))
        labels = [str(item["task_id"]) for item in arena]
        axis.bar(
            np.arange(len(arena)),
            [item["boundaries"]["stationary_excess"] for item in arena],
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set(
            title="Arena stationary excess around official subtask boundaries",
            xlabel="Task",
            ylabel="Boundary stationary ratio - non-boundary ratio",
            xticks=np.arange(len(arena)),
            xticklabels=labels,
        )
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "arena_boundary_stationary_excess.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    task_labels = [f"{item['dataset'].split('-')[0]}:{item['task_id']}" for item in records]
    axes[0].bar(
        np.arange(len(records)),
        [item["action_dispersion"]["behavior_alignment_gain"] for item in records],
    )
    axes[0].set_xticks(np.arange(len(records)), task_labels, rotation=70, ha="right")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set(title="Per-task action alignment gain", xlabel="Task", ylabel="1 - behavior/time dispersion")
    for item in mixed:
        grid = np.asarray(item["curves"]["progress_grid"])
        axes[1].plot(
            grid,
            item["curves"]["action_time_between_task_fraction"],
            label=f"{item['dataset']}: time",
        )
        axes[1].plot(
            grid,
            item["curves"]["action_behavior_between_task_fraction"],
            linestyle="--",
            label=f"{item['dataset']}: behavior",
        )
    axes[1].set(title="Mixed-task action separability", xlabel="Progress", ylabel="Between-task variance fraction")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    axes[0].set_yscale("symlog", linthresh=0.1)
    fig.tight_layout()
    fig.savefig(output_dir / "single_and_mixed_task_analysis.png", dpi=180)
    plt.close(fig)

    if arena:
        fig, axis = plt.subplots(figsize=(10, 4.5))
        grid = np.asarray(arena[0]["curves"]["progress_grid"])
        for item in arena:
            axis.plot(
                grid,
                item["curves"]["segment_idle_profile"],
                label=f"Task {item['task_id']}",
            )
        axis.set(
            title="Strict idle probability within official Arena subtasks",
            xlabel="Relative position inside subtask",
            ylabel="Idle probability",
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "arena_subtask_stationary_profile.png", dpi=180)
        plt.close(fig)


def _write_markdown(
    records: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    mixed: list[dict[str, Any]],
    output: Path,
) -> None:
    lines = [
        "# LIBERO vs RoboMemArena Trajectory Quality Audit",
        "",
        "Behavior progress is cumulative arm-action path length. Dispersion values are normalized by pooled task variance. Strict idle requires both low arm action and low robot-state velocity.",
        "Cross-task stationary exposure must use the absolute columns below: raw control steps, raw steps per official subtask, and affected non-overlapping 10-step Pi replanning windows. Ratios are dataset-composition descriptors only.",
        "",
        "## Dataset aggregates",
        "",
        "| Dataset | Tasks | Length | Time/behavior MAE | Time-warp std | Same-step progress var | Action align gain | Speed CV | Stationary | Exact-zero arm | Idle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['dataset']} | {row['tasks']} | {row['length_mean']:.1f} | "
            f"{row['time_behavior_mae']:.4f} | {row['time_warp_std']:.4f} | "
            f"{row['same_step_progress_var']:.6f} | {row['action_alignment_gain']:.3f} | "
            f"{row['speed_cv']:.3f} | {row['stationary_ratio']:.3f} | "
            f"{row['exact_zero_arm_action_ratio']:.3f} | {row['idle_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-task results",
            "",
            "| Dataset | Task | Episodes | Length | Subtasks | Idle steps | Zero/state-moving steps | Zero/state-moving/subtask | Affected 10-step windows | Time-warp std | Align gain | Speed CV |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in records:
        row = _flatten_record(item)
        lines.append(
            f"| {row['dataset']} | {row['task_id']} | {row['episodes']} | {row['length_mean']:.1f} | "
            f"{row['segment_count']:.0f} | {row['idle_steps']:.1f} | "
            f"{row['exact_zero_while_state_moving_steps']:.1f} | "
            f"{row['exact_zero_while_state_moving_steps_per_segment']:.1f} | "
            f"{row['exact_zero_while_state_moving_replan_windows_any']:.1f} | "
            f"{row['time_warp_std']:.4f} | {row['action_alignment_gain']:.3f} | "
            f"{row['speed_cv']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Mixed-task structure",
            "",
            "Mixed total variance is not interpreted as data quality. This section measures task-conditioned structure.",
            "",
            "| Dataset | Episodes | Action between fraction (time) | Action task accuracy (time) | Action between fraction (behavior) | Action task accuracy (behavior) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in mixed:
        lines.append(
            f"| {item['dataset']} | {item['episodes']} | "
            f"{item['action_time']['between_task_fraction_mean']:.3f} | "
            f"{item['action_time']['task_nearest_centroid_accuracy']:.3f} | "
            f"{item['action_behavior']['between_task_fraction_mean']:.3f} | "
            f"{item['action_behavior']['task_nearest_centroid_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Large positive action alignment gain means normalized time is a poor action-progress coordinate.",
            "- Large time-warp spread means different episodes reach the same behavior progress at different normalized times.",
            "- Positive Arena boundary stationary excess supports the subtask-gap hypothesis.",
            "- Large normalized boundary state jumps indicate discontinuous segment concatenation rather than only a pause.",
            "- Low offline retrieval error cannot guarantee low action dispersion or online action compatibility.",
            "",
            "![Overview](trajectory_quality_overview.png)",
            "",
            "![Arena boundary audit](arena_boundary_stationary_excess.png)",
            "",
            "![Arena subtask stationary profile](arena_subtask_stationary_profile.png)",
            "",
            "![Single and mixed task analysis](single_and_mixed_task_analysis.png)",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LIBERO and Arena trajectory quality.")
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=Path("/path/to/local/data/huggingface/lerobot/HuggingFaceVLA/libero"),
    )
    parser.add_argument(
        "--arena-manifest",
        type=Path,
        default=Path(
            "/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v4_dense10/"
            "pi05_finetuned/features/anchors_stride10.jsonl"
        ),
    )
    parser.add_argument("--libero-task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--arena-task-ids", default="1,2,3,18,19,22,25,26")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes_per_task < 2:
        raise ValueError("episodes-per-task must be at least 2")

    libero_ids = {int(value) for value in args.libero_task_ids.split(",") if value.strip()}
    arena_ids = {int(value) for value in args.arena_task_ids.split(",") if value.strip()}
    trajectories = load_libero(args.libero_root, libero_ids, args.episodes_per_task)
    trajectories.extend(
        load_arena(args.arena_manifest, arena_ids, args.episodes_per_task, args.workers)
    )
    grouped: defaultdict[tuple[str, int], list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[(trajectory.dataset, trajectory.task_id)].append(trajectory)
    records = [
        analyze_task(items, grid_size=args.grid_size)
        for _, items in sorted(grouped.items(), key=lambda item: item[0])
    ]
    by_dataset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectories_by_dataset: defaultdict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        trajectories_by_dataset[trajectory.dataset].append(trajectory)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    aggregates = [_dataset_aggregate(items) for _, items in sorted(by_dataset.items())]
    mixed = [
        analyze_mixed_dataset(items, grid_size=args.grid_size)
        for _, items in sorted(trajectories_by_dataset.items())
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "libero_arena_trajectory_quality_v1",
        "episodes_per_task": args.episodes_per_task,
        "definitions": {
            "behavior_progress": "Cumulative arm-action path length normalized to [0,1].",
            "action_alignment_gain": "1 - behavior-aligned dispersion / time-aligned dispersion.",
            "stationary": "Arm action norm <= 5% of the task-pooled P90 arm action norm.",
            "strict_idle": "Stationary arm action and robot-state speed <= 5% of task-pooled P90 state speed.",
            "exact_zero_arm": "Arm-action L2 norm <= 1e-8; independent of dataset-relative thresholds.",
            "stationary_while_state_moving": "Stationary arm action while normalized robot-state speed exceeds 5% of task-pooled P90.",
            "exact_zero_while_state_moving": "Exact-zero arm action while normalized robot-state speed exceeds 5% of task-pooled P90.",
            "absolute_step_metrics": "Mean raw simulator/control steps per complete trajectory; no trajectory-length normalization.",
            "per_segment_step_metrics": "Mean raw steps per official subtask segment.",
            "replan_window_metrics": "Count of non-overlapping 10-step Pi action/replanning windows intersecting the condition.",
            "boundary_window": "+/-15 frames around official Arena segment boundaries.",
        },
        "aggregates": aggregates,
        "mixed_task_analysis": mixed,
        "tasks": records,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "task_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = [_flatten_record(record) for record in records]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_plots(records, mixed, args.output_dir)
    _write_markdown(records, aggregates, mixed, args.output_dir / "report.md")
    print(json.dumps(aggregates, indent=2, allow_nan=True))
    print(f"Wrote trajectory quality audit to {args.output_dir}")


if __name__ == "__main__":
    main()
