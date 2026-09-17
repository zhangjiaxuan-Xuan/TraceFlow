from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from openpi.task_head.memory_init import PackedActionStore, _interpolate_local_peak


def _metrics(errors: list[float]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mae": float(np.mean(np.abs(values))),
        "p95": float(np.percentile(np.abs(values), 95)),
        "max": float(np.max(np.abs(values))),
    }


def _action_metrics(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    difference = prediction - reference
    rmse = float(np.sqrt(np.mean(difference**2, dtype=np.float64)))
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(prediction))
    cosine = float(np.dot(reference.ravel(), prediction.ravel()) / max(denominator, 1e-12))
    return rmse, cosine


def _slice(actions: np.ndarray, frame: int, horizon: int) -> np.ndarray:
    frame = int(np.clip(frame, 0, len(actions) - 1))
    block = actions[frame : frame + horizon]
    if len(block) < horizon:
        block = np.concatenate((block, np.repeat(block[-1:], horizon - len(block), axis=0)))
    return block


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate sparse-anchor frame interpolation against held-out true Arena frames."
    )
    parser.add_argument("--memory-meta", type=Path, required=True)
    parser.add_argument("--memory-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strides", default="2,4")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-trajectories", type=int, default=0)
    args = parser.parse_args()

    metadata = torch.load(args.memory_meta, map_location="cpu", weights_only=False)
    action_store = PackedActionStore(str(args.memory_actions))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in metadata:
        grouped[str(entry["action_id"])].append(entry)
    trajectory_ids = sorted(grouped)
    if args.max_trajectories > 0:
        trajectory_ids = trajectory_ids[: args.max_trajectories]

    report: dict[str, object] = {
        "protocol": "continuous_frame_v2_leave_anchor_out_v1",
        "memory_meta": str(args.memory_meta.resolve()),
        "memory_actions": str(args.memory_actions.resolve()),
        "horizon": int(args.horizon),
        "trajectories": len(trajectory_ids),
        "strides": {},
    }
    for stride in [int(value) for value in args.strides.split(",") if value.strip()]:
        if stride < 2:
            raise ValueError("Each anchor subsampling stride must be at least 2")
        buckets: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for action_id in trajectory_ids:
            rows = sorted(grouped[action_id], key=lambda row: int(row["anchor_frame"]))
            frames = np.asarray([int(row["anchor_frame"]) for row in rows], dtype=np.int64)
            embeddings = np.stack(
                [torch.as_tensor(row["task_emb"], dtype=torch.float32).numpy() for row in rows]
            )
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
            retained = np.zeros(len(rows), dtype=bool)
            retained[::stride] = True
            retained[-1] = True
            held_out = np.flatnonzero(~retained)
            if not len(held_out):
                continue
            retained_frames = frames[retained]
            retained_embeddings = embeddings[retained]
            actions = action_store.get(action_id)
            task_id = str(rows[0].get("task_id", "unknown"))

            for query_index in held_out:
                scores = retained_embeddings @ embeddings[query_index]
                nearest_offset = int(np.argmax(scores))
                nearest_frame = int(retained_frames[nearest_offset])
                continuous_frame = int(
                    np.rint(_interpolate_local_peak(retained_frames, scores))
                )
                true_frame = int(frames[query_index])
                reference = _slice(actions, true_frame, args.horizon)
                nearest = _slice(actions, nearest_frame, args.horizon)
                continuous = _slice(actions, continuous_frame, args.horizon)
                nearest_rmse, nearest_cosine = _action_metrics(reference, nearest)
                continuous_rmse, continuous_cosine = _action_metrics(reference, continuous)

                for bucket_name in ("all", f"task_{task_id}"):
                    bucket = buckets[bucket_name]
                    bucket["nearest_frame_error"].append(nearest_frame - true_frame)
                    bucket["continuous_frame_error"].append(continuous_frame - true_frame)
                    max_frame = max(len(actions) - 1, 1)
                    bucket["nearest_progress_error"].append(
                        (nearest_frame - true_frame) / max_frame
                    )
                    bucket["continuous_progress_error"].append(
                        (continuous_frame - true_frame) / max_frame
                    )
                    bucket["nearest_action_rmse"].append(nearest_rmse)
                    bucket["continuous_action_rmse"].append(continuous_rmse)
                    bucket["nearest_action_cosine"].append(nearest_cosine)
                    bucket["continuous_action_cosine"].append(continuous_cosine)

        stride_report: dict[str, object] = {}
        for bucket_name, values in sorted(buckets.items()):
            stride_report[bucket_name] = {
                "nearest_frame": _metrics(values["nearest_frame_error"]),
                "continuous_frame": _metrics(values["continuous_frame_error"]),
                "nearest_progress": _metrics(values["nearest_progress_error"]),
                "continuous_progress": _metrics(values["continuous_progress_error"]),
                "nearest_action_rmse_mean": float(np.mean(values["nearest_action_rmse"])),
                "continuous_action_rmse_mean": float(np.mean(values["continuous_action_rmse"])),
                "nearest_action_cosine_mean": float(np.mean(values["nearest_action_cosine"])),
                "continuous_action_cosine_mean": float(np.mean(values["continuous_action_cosine"])),
            }
        report["strides"][str(stride)] = stride_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["strides"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
