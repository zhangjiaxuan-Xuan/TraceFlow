"""Build the PrediMem lower-policy dataset with ground-truth subtask prompts.

Episodes retain the complete task/seed trajectory and therefore the same
episode/frame addressing as the reactive dataset and Tdense5 guidance cache.
Only the per-frame language label changes: each source HDF5 segment contributes
its own canonical primitive instead of inheriting the long-task prompt.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Iterator

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import tyro

import sys

DATA_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_SCRIPTS))
from convert_robomemarena_extra8_to_lerobot import _group_manifest_files  # noqa: E402
from convert_robomemarena_extra8_to_lerobot import _iter_chunk_frames  # noqa: E402


DEFAULT_MANIFEST = pathlib.Path(
    "/path/to/storage/datasets/robotics/RoboMemArena/raw/selection_sequence-occlusion-counting-transferring_all_seeds.json"
)
DEFAULT_RAW_ROOT = pathlib.Path("/path/to/storage/datasets/robotics/RoboMemArena/raw")
DEFAULT_LEROBOT_HOME = pathlib.Path("/path/to/local/data/huggingface/lerobot")


def primitive_from_path(path: pathlib.Path) -> str:
    stem = re.sub(r"_\d+_seed\d+_task\d+$", "", path.stem)
    primitive = " ".join(stem.replace("_", " ").replace("-", " ").lower().split())
    if not primitive:
        raise ValueError(f"Cannot infer primitive from {path}")
    return primitive


def load_episode(
    item: tuple[tuple[int, int], list[tuple[int, pathlib.Path]]],
) -> tuple[tuple[int, int], list[tuple[str, dict]]]:
    key, chunks = item
    frames: list[tuple[str, dict]] = []
    for _, segment in chunks:
        primitive = primitive_from_path(segment)
        frames.extend((primitive, frame) for frame in _iter_chunk_frames(segment))
    if not frames:
        raise ValueError(f"Empty trajectory task={key[0]} seed={key[1]}")
    return key, frames


def ordered_prefetch(
    items: list[tuple[tuple[int, int], list[tuple[int, pathlib.Path]]]], workers: int,
) -> Iterator[tuple[tuple[int, int], list[tuple[str, dict]]]]:
    """Bounded parallel HDF5 reads without changing deterministic episode order."""
    if workers <= 1:
        yield from map(load_episode, items)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arena-hdf5") as pool:
        pending: dict[int, Future] = {}
        submitted = 0
        while submitted < min(workers, len(items)):
            pending[submitted] = pool.submit(load_episode, items[submitted])
            submitted += 1
        for index in range(len(items)):
            yield pending.pop(index).result()
            if submitted < len(items):
                pending[submitted] = pool.submit(load_episode, items[submitted])
                submitted += 1


def main(
    manifest: pathlib.Path = DEFAULT_MANIFEST,
    raw_root: pathlib.Path = DEFAULT_RAW_ROOT,
    lerobot_home: pathlib.Path = DEFAULT_LEROBOT_HOME,
    repo_id: str = "robomemarena/all26_pi05_subtask",
    *,
    expected_tasks: int = 26,
    expected_episodes: int = 2600,
    max_episodes: int | None = None,
    overwrite: bool = False,
    read_workers: int = 32,
    image_writer_threads: int = 32,
    image_writer_processes: int = 8,
) -> None:
    groups = _group_manifest_files(manifest, raw_root)
    episode_keys = sorted(groups)
    task_ids = sorted({task for task, _ in episode_keys})
    if len(task_ids) != expected_tasks or len(episode_keys) != expected_episodes:
        raise ValueError(
            f"Dataset cardinality mismatch: tasks={len(task_ids)}/{expected_tasks}, "
            f"episodes={len(episode_keys)}/{expected_episodes}"
        )
    if max_episodes is not None:
        episode_keys = episode_keys[:max_episodes]

    output = lerobot_home / repo_id
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        robot_type="panda",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    primitive_counts: dict[str, int] = {}
    work = [(key, groups[key]) for key in episode_keys]
    for (task_id, seed), frames in tqdm(
        ordered_prefetch(work, read_workers), total=len(work), desc="Converting subtask-labelled trajectories"
    ):
        for primitive, frame in frames:
            dataset.add_frame({**frame, "task": primitive})
        for _, segment in groups[(task_id, seed)]:
            primitive = primitive_from_path(segment)
            primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
        dataset.save_episode()

    summary = {
        "schema": "predimem_lower_ground_truth_subtask_v1",
        "repo_id": repo_id,
        "manifest": str(manifest.resolve()),
        "episodes": len(episode_keys),
        "task_ids": task_ids,
        "primitive_segment_counts": primitive_counts,
        "addressing": "complete task/seed episodes; source subtask prompt per frame",
        "read_workers": read_workers,
        "image_writer_threads": image_writer_threads,
        "image_writer_processes": image_writer_processes,
    }
    (output.parent / f"{output.name}_conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    tyro.cli(main)
