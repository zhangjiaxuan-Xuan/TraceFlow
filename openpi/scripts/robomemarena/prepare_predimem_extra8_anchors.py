from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import re

import h5py
import numpy as np


DEFAULT_TASK_IDS = (1, 2, 3, 18, 19, 22, 25, 26)
FILE_RE = re.compile(r"_(?P<order>\d+)_seed(?P<seed>\d+)_task(?P<task>\d+)\.hdf5$")


def _inspect(path: str) -> tuple[str, int]:
    with h5py.File(path, "r") as data:
        return path, int(data["data/demo_0/actions"].shape[0])


def _prompts(path: Path) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("arena_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prompts from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.TASK_PROMPTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--task-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors-per-trajectory", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=0)
    parser.add_argument("--history-offsets", default="0")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--task-ids",
        default=",".join(str(value) for value in DEFAULT_TASK_IDS),
        help="Comma-separated task IDs to include; each task/seed group is kept ordered.",
    )
    parser.add_argument(
        "--expected-trajectories",
        type=int,
        default=0,
        help="Optional exact trajectory count. 0 derives it from the selection.",
    )
    args = parser.parse_args()
    task_ids = tuple(dict.fromkeys(int(value) for value in args.task_ids.split(",") if value.strip()))
    if not task_ids or any(value <= 0 for value in task_ids):
        raise ValueError("task-ids must contain positive integers")
    if args.anchors_per_trajectory <= 0:
        raise ValueError("anchors-per-trajectory must be positive")
    if args.anchor_stride < 0:
        raise ValueError("anchor-stride cannot be negative")
    history_offsets = tuple(int(value) for value in args.history_offsets.split(",") if value.strip())
    if not history_offsets or history_offsets[-1] != 0 or tuple(sorted(set(history_offsets))) != history_offsets:
        raise ValueError("history-offsets must be unique, increasing, and end at 0")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    groups: dict[tuple[int, int], list[tuple[int, Path]]] = {}
    for item in selection["files"]:
        relative = Path(item["path"])
        match = FILE_RE.search(relative.name)
        if match is None:
            raise ValueError(f"Unexpected dataset filename: {relative}")
        task, seed, order = (int(match.group(name)) for name in ("task", "seed", "order"))
        if task not in task_ids:
            continue
        path = args.raw_root / relative
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise RuntimeError(f"Missing or incomplete source file: {path}")
        groups.setdefault((task, seed), []).append((order, path))
    if sorted({task for task, _ in groups}) != sorted(task_ids):
        raise RuntimeError(f"Selection does not contain exactly the requested tasks: {task_ids}")
    if args.expected_trajectories and len(groups) != args.expected_trajectories:
        raise RuntimeError(
            f"Expected {args.expected_trajectories} trajectories, got {len(groups)}"
        )
    for chunks in groups.values():
        chunks.sort()

    all_paths = [str(path) for chunks in groups.values() for _, path in chunks]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        lengths = dict(pool.map(_inspect, all_paths, chunksize=8))
    prompts = _prompts(args.task_prompts)
    task_config = {
        int(row["task_id"]): row
        for row in json.loads(args.task_config.read_text(encoding="utf-8"))["tasks"]
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    rows = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for episode_index, ((task, seed), chunks) in enumerate(sorted(groups.items())):
            segment_lengths = [lengths[str(path)] for _, path in chunks]
            offsets = np.cumsum([0, *segment_lengths])
            total = int(offsets[-1])
            if args.anchor_stride > 0:
                anchor_frames = np.arange(0, total, args.anchor_stride, dtype=np.int64)
                if int(anchor_frames[-1]) != total - 1:
                    anchor_frames = np.concatenate(
                        (anchor_frames, np.asarray([total - 1], dtype=np.int64))
                    )
            else:
                anchor_frames = np.unique(
                    np.rint(np.linspace(0, total - 1, args.anchors_per_trajectory)).astype(np.int64)
                )
            action_id = f"task{task:02d}_seed{seed:04d}"
            segment_paths = [str(path.resolve()) for _, path in chunks]
            for anchor_rank, global_frame in enumerate(anchor_frames.tolist()):
                stage = int(np.searchsorted(offsets[1:], global_frame, side="right"))
                local_frame = int(global_frame - offsets[stage])
                stage_length = int(segment_lengths[stage])
                temporal_context = []
                for history_offset in history_offsets:
                    history_global_frame = max(0, global_frame + history_offset)
                    history_stage = int(np.searchsorted(offsets[1:], history_global_frame, side="right"))
                    temporal_context.append(
                        {
                            "offset": history_offset,
                            "global_frame_index": history_global_frame,
                            "stage_index": history_stage,
                            "frame_index": int(history_global_frame - offsets[history_stage]),
                            "trajectory_path": segment_paths[history_stage],
                        }
                    )
                config = task_config[task]
                row = {
                    "row_index": rows,
                    "episode_index": episode_index,
                    "task_id": task,
                    "task_index": task_ids.index(task),
                    "suite": str(config["suite"]),
                    "seed": seed,
                    "stage_index": stage,
                    "stage_start_frame": int(offsets[stage]),
                    "stage_length": stage_length,
                    "stage_frame_index": local_frame,
                    "stage_progress": float(local_frame / max(stage_length - 1, 1)),
                    "anchor_rank": anchor_rank,
                    "anchor_progress": float(global_frame / max(total - 1, 1)),
                    "frame_index": local_frame,
                    "global_frame_index": global_frame,
                    "trajectory_length": total,
                    "trajectory_path": segment_paths[stage],
                    "segment_paths": segment_paths,
                    "segment_lengths": segment_lengths,
                    "temporal_context": temporal_context,
                    "demo": "demo_0",
                    "source_format": "official_hdf5",
                    "prompt": prompts[f"task{task}"],
                    "task_block": str(config["task_block"]),
                    "scene_description": str(config.get("scene_description", "")),
                    "action_id": action_id,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                rows += 1
    temporary.replace(args.output)
    summary = {
        "tasks": list(task_ids),
        "trajectories": len(groups),
        "anchors": rows,
        "anchors_per_trajectory": args.anchors_per_trajectory,
        "anchor_stride": args.anchor_stride,
        "history_offsets": list(history_offsets),
        "workers": args.workers,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
