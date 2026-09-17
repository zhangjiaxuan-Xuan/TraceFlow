from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import functools
import os
from pathlib import Path
import threading
from typing import Any

import h5py
import numpy as np

from openpi.task_head.npz_stream import NpzStreamReader


@functools.lru_cache(maxsize=4)
def _create_traceflow_dataset(root: str, repo_id: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id,
        root=Path(root),
        video_backend=os.environ.get("OPENPI_LEROBOT_VIDEO_BACKEND", "pyav"),
    )


_TRACEFLOW_DATASET_LOCK = threading.Lock()


def _traceflow_dataset(root: str, repo_id: str):
    # lru_cache may execute duplicate misses concurrently; dataset construction is expensive.
    with _TRACEFLOW_DATASET_LOCK:
        return _create_traceflow_dataset(root, repo_id)


def _format_observation(
    row: dict[str, Any],
    image: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
) -> dict[str, Any]:
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": row["prompt"],
    }


def _load_traceflow_observation(dataset, row: dict[str, Any], root: str) -> dict[str, Any]:
    dataset_index = int(row["dataset_index"])
    if dataset_index < 0 or dataset_index >= len(dataset):
        raise IndexError(f"dataset_index={dataset_index} is invalid for {root}")
    sample = dataset[dataset_index]
    actual_episode = int(sample["episode_index"])
    actual_frame = int(sample["frame_index"])
    if actual_episode != int(row["episode_index"]) or actual_frame != int(row["frame_index"]):
        raise RuntimeError(
            "TraceFlow manifest/data mismatch: "
            f"expected episode={row['episode_index']} frame={row['frame_index']}, "
            f"got episode={actual_episode} frame={actual_frame}"
        )
    return {
        "images": {
            "cam_high": np.asarray(sample["observation.images.cam_high"]),
            "cam_left_wrist": np.asarray(sample["observation.images.cam_left_wrist"]),
            "cam_right_wrist": np.asarray(sample["observation.images.cam_right_wrist"]),
        },
        "state": np.asarray(sample["observation.state"], dtype=np.float32),
        "prompt": row["prompt"],
    }


def _load_traceflow_observation_group(
    dataset,
    items: list[tuple[int, dict[str, Any]]],
    root: str,
    loader_workers: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Decode all requested timestamps once per camera for one episode."""
    from lerobot.common.datasets.video_utils import decode_video_frames

    dataset_indices = [int(row["dataset_index"]) for _, row in items]
    if min(dataset_indices) < 0 or max(dataset_indices) >= len(dataset):
        raise IndexError(f"dataset indices are invalid for {root}: {dataset_indices}")
    samples = dataset.hf_dataset.select(dataset_indices)
    actual_episodes = [int(value) for value in samples["episode_index"]]
    actual_frames = [int(value) for value in samples["frame_index"]]
    expected_episodes = [int(row["episode_index"]) for _, row in items]
    expected_frames = [int(row["frame_index"]) for _, row in items]
    if actual_episodes != expected_episodes or actual_frames != expected_frames:
        raise RuntimeError(
            "TraceFlow manifest/data batch mismatch: "
            f"expected episodes={expected_episodes} frames={expected_frames}, "
            f"got episodes={actual_episodes} frames={actual_frames}"
        )
    episodes = set(actual_episodes)
    if len(episodes) != 1:
        raise RuntimeError(f"A TraceFlow video batch must belong to one episode, got {episodes}")
    episode = actual_episodes[0]
    timestamps = [float(value) for value in samples["timestamp"]]

    def decode_camera(camera: str) -> tuple[str, np.ndarray]:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode, camera)
        frames = decode_video_frames(
            video_path,
            timestamps,
            dataset.tolerance_s,
            dataset.video_backend,
        )
        return camera, np.asarray(frames)

    cameras = tuple(dataset.meta.video_keys)
    if loader_workers > 1 and len(cameras) > 1:
        with ThreadPoolExecutor(
            max_workers=min(loader_workers, len(cameras)),
            thread_name_prefix="traceflow-camera",
        ) as executor:
            decoded = dict(executor.map(decode_camera, cameras))
    else:
        decoded = dict(map(decode_camera, cameras))

    required_cameras = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    missing = set(required_cameras).difference(decoded)
    if missing:
        raise KeyError(f"TraceFlow dataset is missing required cameras: {sorted(missing)}")
    states = samples["observation.state"]
    loaded = []
    for position, (result_index, row) in enumerate(items):
        loaded.append(
            (
                result_index,
                {
                    "images": {
                        "cam_high": decoded[required_cameras[0]][position],
                        "cam_left_wrist": decoded[required_cameras[1]][position],
                        "cam_right_wrist": decoded[required_cameras[2]][position],
                    },
                    "state": np.asarray(states[position], dtype=np.float32),
                    "prompt": row["prompt"],
                },
            )
        )
    return loaded


def load_cl_observations(
    rows: list[dict[str, Any]], *, loader_workers: int = 0
) -> list[dict[str, Any]]:
    """Load trajectory frames while opening each backing file once per batch."""
    if loader_workers < 0:
        raise ValueError("loader_workers cannot be negative")
    results: list[dict[str, Any] | None] = [None] * len(rows)
    groups: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        source_format = str(row.get("source_format", "official_hdf5"))
        path = row.get("trajectory_path", row.get("task_file"))
        if not path:
            raise KeyError(f"{source_format} feature rows require trajectory_path or task_file")
        demo_name = str(row.get("demo", "demo_0"))
        groups[(source_format, str(path), demo_name)].append((index, row))

    for (source_format, path, demo_name), items in groups.items():
        if source_format in {"official_hdf5", "eval_hdf5"}:
            with h5py.File(path, "r") as data_file:
                demo = data_file["data"][demo_name]
                length = int(demo["actions"].shape[0])
                for index, row in items:
                    frame = int(row["frame_index"])
                    if frame < 0 or frame >= length:
                        raise IndexError(f"frame_index={frame} is invalid for {path}:{demo_name}")
                    image = np.ascontiguousarray(demo["obs/agentview_rgb"][frame][::-1, ::-1])
                    wrist = np.ascontiguousarray(demo["obs/eye_in_hand_rgb"][frame][::-1, ::-1])
                    state = np.concatenate(
                        [
                            np.asarray(demo["obs/ee_states"][frame]),
                            np.asarray(demo["obs/gripper_states"][frame]),
                        ]
                    ).astype(np.float32)
                    results[index] = _format_observation(row, image, wrist, state)
        elif source_format == "smol_npz":
            with NpzStreamReader(path) as data:
                image_header = data.header("observation.images.image")
                wrist_header = data.header("observation.images.image2")
                state_header = data.header("observation.state")
                if (
                    wrist_header.shape[0] != image_header.shape[0]
                    or state_header.shape[0] != image_header.shape[0]
                ):
                    raise RuntimeError(f"SmolVLA observation arrays have different lengths: {path}")
                for index, row in items:
                    frame = int(row["frame_index"])
                    if frame < 0 or frame >= int(image_header.shape[0]):
                        raise IndexError(f"frame_index={frame} is invalid for {path}")
                    image = np.ascontiguousarray(
                        data.frame("observation.images.image", frame), dtype=np.uint8
                    )
                    wrist = np.ascontiguousarray(
                        data.frame("observation.images.image2", frame), dtype=np.uint8
                    )
                    state = np.asarray(data.frame("observation.state", frame), dtype=np.float32)
                    if state.shape != (8,):
                        raise RuntimeError(
                            f"Expected SmolVLA state [8], got {state.shape} from {path}"
                        )
                    results[index] = _format_observation(row, image, wrist, state)
        elif source_format == "lerobot_traceflow":
            roots = {str(row["dataset_root"]) for _, row in items}
            repo_ids = {str(row["repo_id"]) for _, row in items}
            if len(roots) != 1 or len(repo_ids) != 1:
                raise RuntimeError("A TraceFlow observation group must use one dataset root and repo id")
            root = next(iter(roots))
            dataset = _traceflow_dataset(root, next(iter(repo_ids)))
            if len(items) > 1 and hasattr(dataset, "hf_dataset"):
                for index, observation in _load_traceflow_observation_group(
                    dataset, items, root, loader_workers
                ):
                    results[index] = observation
            else:
                for index, row in items:
                    results[index] = _load_traceflow_observation(dataset, row, root)
        else:
            raise ValueError(f"Unsupported source_format={source_format!r}")

    if any(result is None for result in results):
        raise RuntimeError("Batch observation loading left an unfilled row")
    return [result for result in results if result is not None]


def load_cl_observation(row: dict[str, Any]) -> dict[str, Any]:
    """Load one trajectory frame in the online Pi0.5 observation convention."""
    return load_cl_observations([row])[0]
