#!/usr/bin/env python3
"""Measure task-local action-direction compatibility in Arena demonstrations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from optimus_eval.arena_suite_discreteness import _resample_gpu
from optimus_eval.trajectory_quality_audit import Trajectory, _behavior_progress, load_arena


def _chunks(values: torch.Tensor, horizon: int) -> torch.Tensor:
    padded = torch.cat([values, values[-1:].expand(horizon - 1, -1)], dim=0)
    return padded.unfold(0, horizon, 1).transpose(1, 2).flatten(1)


def _resampled_chunks(
    item: Trajectory,
    mean: torch.Tensor,
    scale: torch.Tensor,
    grid: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = mean.device
    action = torch.as_tensor(item.actions[:, : mean.numel()], dtype=torch.float32, device=device)
    action = (action - mean) / scale
    time = torch.linspace(0.0, 1.0, len(action), device=device)
    behavior = torch.as_tensor(_behavior_progress(item.actions), dtype=torch.float32, device=device)
    behavior = torch.maximum(behavior, torch.arange(len(behavior), device=device) * 1e-8)
    behavior = behavior / behavior[-1].clamp_min(1e-8)
    time_row = _resample_gpu(action, time, grid)
    behavior_row = _resample_gpu(action, behavior, grid)
    return _chunks(time_row, horizon), _chunks(behavior_row, horizon)


def _compatibility(chunks: torch.Tensor) -> dict[str, float]:
    # [episodes, progress, chunk_dim]. Compare each trajectory with a mean that
    # excludes itself, preventing a trajectory from validating its own action.
    episodes = chunks.shape[0]
    loo_mean = (chunks.sum(0, keepdim=True) - chunks) / max(1, episodes - 1)
    chunk_norm = chunks.norm(dim=2)
    mean_norm = loo_mean.norm(dim=2)
    valid = (chunk_norm > 1e-4) & (mean_norm > 1e-4)
    cosine = (chunks * loo_mean).sum(2) / (chunk_norm * mean_norm).clamp_min(1e-12)
    cosine_valid = cosine[valid]
    unit = chunks / chunk_norm.clamp_min(1e-12).unsqueeze(2)
    coherence = unit.mean(0).norm(dim=1)
    motion_weight = chunk_norm / chunk_norm.sum(1, keepdim=True).clamp_min(1e-12)
    weighted_cosine = (cosine.clamp(-1.0, 1.0) * motion_weight).sum(1).mean()
    return {
        "loo_action_cosine_mean": float(cosine_valid.mean()),
        "loo_action_cosine_p10": float(torch.quantile(cosine_valid, 0.10)),
        "loo_action_cosine_positive_fraction": float((cosine_valid > 0).float().mean()),
        "motion_weighted_loo_cosine": float(weighted_cosine),
        "directional_coherence_mean": float(coherence.mean()),
        "directional_coherence_p10": float(torch.quantile(coherence, 0.10)),
        "valid_fraction": float(valid.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=29)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    task_config = json.loads(args.task_config.read_text())
    suites = {int(row["task_id"]): str(row["suite"]) for row in task_config["tasks"]}
    trajectories = load_arena(args.manifest, set(suites), args.episodes_per_task, args.workers)
    arm_dim = min(item.actions.shape[1] - 1 for item in trajectories)
    pooled = np.concatenate([item.actions[:, :arm_dim] for item in trajectories])
    mean = torch.as_tensor(pooled.mean(0), dtype=torch.float32, device=device)
    scale = torch.as_tensor(pooled.std(0), dtype=torch.float32, device=device).clamp_min(1e-6)
    grid = torch.linspace(0.0, 1.0, args.grid_size, device=device)
    by_task: defaultdict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for item in trajectories:
        by_task[item.task_id].append(_resampled_chunks(item, mean, scale, grid, args.horizon))

    records = []
    for task_id in sorted(suites):
        time_chunks = torch.stack([row[0] for row in by_task[task_id]])
        behavior_chunks = torch.stack([row[1] for row in by_task[task_id]])
        record = {
            "task_id": task_id,
            "suite": suites[task_id],
            "episodes": len(time_chunks),
        }
        record.update({f"time_{key}": value for key, value in _compatibility(time_chunks).items()})
        record.update({f"behavior_{key}": value for key, value in _compatibility(behavior_chunks).items()})
        records.append(record)

    suite_records = []
    for suite in sorted(set(suites.values())):
        selected = [row for row in records if row["suite"] == suite]
        suite_record = {"suite": suite, "tasks": len(selected)}
        for key in records[0]:
            if key not in ("task_id", "suite", "episodes"):
                suite_record[key] = float(np.mean([row[key] for row in selected]))
        suite_records.append(suite_record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("task_compatibility.csv", records), ("suite_compatibility.csv", suite_records)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "protocol": "arena_leave_one_trajectory_out_action_directional_compatibility_v1",
        "device": torch.cuda.get_device_name(device),
        "episodes_per_task": args.episodes_per_task,
        "grid_size": args.grid_size,
        "action_chunk_horizon": args.horizon,
        "suites": suite_records,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
