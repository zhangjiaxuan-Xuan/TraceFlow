#!/usr/bin/env python3
"""Compute step-normalized, full-episode PrediMem guidance alignment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace-dir", type=Path)
    source.add_argument("--record-root", type=Path)
    parser.add_argument("--task-metrics", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--expected-tasks", type=int, default=26)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _scalar(value: object) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected one trace sample, found shape {array.shape}")
    return float(array[0])


def _call_geometry(record: dict) -> tuple[float, float, float, int]:
    alignments: list[float] = []
    cap_ratios: list[float] = []
    rotations: list[float] = []
    for step in record["steps"]:
        base_norm = _scalar(step["v_base"]["norm"])
        guidance_norm = _scalar(step["guidance_final"]["norm"])
        if base_norm <= 1e-12 or guidance_norm <= 1e-12:
            continue
        cosine = float(np.clip(_scalar(step["guidance_final"]["cosine_to_v_base"]), -1.0, 1.0))
        actual_cosine = float(np.clip(_scalar(step["v_actual"]["cosine_to_v_base"]), -1.0, 1.0))
        alignments.append(cosine)
        cap_ratios.append(guidance_norm / base_norm)
        rotations.append(float(np.degrees(np.arccos(actual_cosine))))
    if not alignments:
        return 0.0, 0.0, 0.0, 0
    return (
        float(np.mean(alignments)),
        float(np.mean(cap_ratios)),
        float(np.mean(rotations)),
        len(alignments),
    )


def _full_call_geometry(path: Path) -> tuple[float, float, float, int]:
    with np.load(path, allow_pickle=False) as trace:
        base = np.asarray(trace["v_base"], dtype=np.float64).reshape(-1, np.prod(trace["v_base"].shape[-2:]))
        guidance = np.asarray(trace["guidance_clipped"], dtype=np.float64).reshape(
            -1, np.prod(trace["guidance_clipped"].shape[-2:])
        )
    base_norm = np.linalg.norm(base, axis=1)
    guidance_norm = np.linalg.norm(guidance, axis=1)
    active = (base_norm > 1e-12) & (guidance_norm > 1e-12)
    if not active.any():
        return 0.0, 0.0, 0.0, 0
    cosine = np.sum(base * guidance, axis=1) / np.maximum(base_norm * guidance_norm, 1e-12)
    actual = base + guidance
    actual_norm = np.linalg.norm(actual, axis=1)
    actual_cosine = np.sum(base * actual, axis=1) / np.maximum(base_norm * actual_norm, 1e-12)
    return (
        float(np.mean(cosine[active])),
        float(np.mean((guidance_norm / np.maximum(base_norm, 1e-12))[active])),
        float(np.mean(np.degrees(np.arccos(np.clip(actual_cosine[active], -1.0, 1.0))))),
        int(active.sum()),
    )


def _normalized_curve(calls: list[dict], key: str, grid: np.ndarray) -> np.ndarray:
    calls = sorted(calls, key=lambda row: row["policy_call_idx"])
    call_indices = np.asarray([row["policy_call_idx"] for row in calls], dtype=np.float64)
    if len(np.unique(call_indices)) != len(call_indices):
        raise ValueError("Duplicate policy_call_idx in one episode")
    if len(calls) == 1:
        return np.full_like(grid, float(calls[0][key]))
    progress = (call_indices - call_indices[0]) / (call_indices[-1] - call_indices[0])
    return np.interp(grid, progress, np.asarray([row[key] for row in calls], dtype=np.float64))


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("grid-size must be at least 2")
    task_meta = {
        int(row["task_id"]): row for row in csv.DictReader(args.task_metrics.open())
    }
    episodes: dict[tuple[int, int], list[dict]] = defaultdict(list)
    if args.trace_dir is not None:
        trace_paths = sorted(args.trace_dir.glob("*.light.jsonl"))
        if not trace_paths:
            raise FileNotFoundError(f"No light traces found in {args.trace_dir}")
        for path in trace_paths:
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError(f"Expected one JSON record in {path}, found {len(lines)}")
            record = json.loads(lines[0])
            alignment, cap_ratio, rotation, active_steps = _call_geometry(record)
            episodes[(int(record["task_id"]), int(record["episode_idx"]))].append(
                {
                    "policy_call_idx": int(record["policy_call_idx"]),
                    "alignment": alignment,
                    "effective_cap": cap_ratio,
                    "rotation_deg": rotation,
                    "active_steps": active_steps,
                }
            )
        source_path = args.trace_dir
        trace_count = len(trace_paths)
    else:
        index_path = args.record_root / "index.jsonl"
        index_rows = [
            json.loads(line) for line in index_path.read_text().splitlines() if line.strip()
        ]
        for record in index_rows:
            trace_path = args.record_root.parent / str(record["trace_path"])
            alignment, cap_ratio, rotation, active_steps = _full_call_geometry(trace_path)
            episodes[(int(record["task_id"]), int(record["episode_idx"]))].append(
                {
                    "policy_call_idx": int(record["policy_call_idx"]),
                    "alignment": alignment,
                    "effective_cap": cap_ratio,
                    "rotation_deg": rotation,
                    "active_steps": active_steps,
                }
            )
        source_path = args.record_root
        trace_count = len(index_rows)

    task_ids = sorted({task_id for task_id, _ in episodes})
    if len(task_ids) != args.expected_tasks:
        raise RuntimeError(
            f"Expected {args.expected_tasks} tasks, found {len(task_ids)}: {task_ids}"
        )
    missing_meta = sorted(set(task_ids) - set(task_meta))
    if missing_meta:
        raise RuntimeError(f"Missing task metadata for {missing_meta}")

    grid = np.linspace(0.0, 1.0, args.grid_size, dtype=np.float64)
    episode_rows: list[dict] = []
    curves: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for key, calls in sorted(episodes.items()):
        if any(row["active_steps"] == 0 for row in calls):
            raise RuntimeError(f"Episode {key} contains a call without active V0 guidance")
        episode_curves = {
            metric: _normalized_curve(calls, metric, grid)
            for metric in ("alignment", "effective_cap", "rotation_deg")
        }
        curves[key] = episode_curves
        episode_rows.append(
            {
                "task_id": key[0],
                "episode_idx": key[1],
                "suite": task_meta[key[0]]["suite"],
                "policy_calls": len(calls),
                "alignment_auc": float(np.trapezoid(episode_curves["alignment"], grid)),
                "alignment_min": float(episode_curves["alignment"].min()),
                "alignment_q10": float(np.quantile(episode_curves["alignment"], 0.10)),
                "effective_cap_auc": float(np.trapezoid(episode_curves["effective_cap"], grid)),
                "rotation_auc_deg": float(np.trapezoid(episode_curves["rotation_deg"], grid)),
            }
        )

    task_rows: list[dict] = []
    task_curves: dict[int, np.ndarray] = {}
    for task_id in task_ids:
        selected = [row for row in episode_rows if row["task_id"] == task_id]
        selected_curves = [
            curves[(task_id, int(row["episode_idx"]))]["alignment"] for row in selected
        ]
        task_curve = np.mean(selected_curves, axis=0)
        task_curves[task_id] = task_curve
        task_rows.append(
            {
                "task_id": task_id,
                "suite": task_meta[task_id]["suite"],
                "episodes": len(selected),
                "policy_calls_mean": float(np.mean([row["policy_calls"] for row in selected])),
                "full_length_alignment": float(np.trapezoid(task_curve, grid)),
                "alignment_min": float(task_curve.min()),
                "alignment_q10": float(np.quantile(task_curve, 0.10)),
                "effective_cap": float(np.mean([row["effective_cap_auc"] for row in selected])),
                "rotation_deg": float(np.mean([row["rotation_auc_deg"] for row in selected])),
            }
        )

    suite_rows: list[dict] = []
    for suite in dict.fromkeys(row["suite"] for row in task_rows):
        selected = [row for row in task_rows if row["suite"] == suite]
        suite_curve = np.mean([task_curves[int(row["task_id"])] for row in selected], axis=0)
        suite_rows.append(
            {
                "suite": suite,
                "tasks": len(selected),
                "full_length_alignment": float(np.trapezoid(suite_curve, grid)),
                "alignment_min": float(suite_curve.min()),
                "alignment_q10": float(np.quantile(suite_curve, 0.10)),
                "policy_calls_mean": float(np.mean([row["policy_calls_mean"] for row in selected])),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("episode_alignment.csv", episode_rows),
        ("task_alignment.csv", task_rows),
        ("suite_alignment.csv", suite_rows),
    ):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "alignment_curves.npz",
        normalized_progress=grid,
        task_ids=np.asarray(task_ids, dtype=np.int64),
        task_alignment=np.stack([task_curves[task_id] for task_id in task_ids]),
    )
    summary = {
        "protocol": "predimem_full_episode_step_normalized_alignment_v1",
        "trace_source": str(source_path),
        "trace_files": trace_count,
        "tasks": len(task_ids),
        "episodes": len(episode_rows),
        "normalization": "Each completed episode is linearly resampled by its own policy-call length to [0, 1].",
        "alignment": "Mean cosine(g_final, v_base) over active V0 denoising steps, then normalized-trajectory AUC.",
        "suites": suite_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
