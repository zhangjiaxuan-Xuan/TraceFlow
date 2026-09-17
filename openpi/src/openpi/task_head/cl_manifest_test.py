from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/data/prepare_cl_memory_manifest.py"
SPEC = importlib.util.spec_from_file_location("prepare_cl_memory_manifest", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_manifest = MODULE.build_manifest
main = MODULE.main


def _write_pi_episode(path: Path, *, episode_idx: int, success: bool, seed: int = 17) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = 2
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        data.attrs.update(
            format="libero-eval-trajectory-v1", num_demos=1, task_suite="libero_10", task_id=3, total=length
        )
        demo.attrs.update(
            task_suite="libero_10",
            task_id=3,
            episode_idx=episode_idx,
            language_instruction="pick up the bowl",
            seed=seed,
            num_samples=length,
            success=success,
        )
        arrays = {
            "states": np.zeros((length, 10)),
            "actions": np.zeros((length, 7), dtype=np.float32),
            "rewards": np.zeros(length),
            "dones": np.array([0, 1], dtype=np.uint8),
            "terminals": np.array([0, 1], dtype=np.uint8),
            "timeouts": np.zeros(length, dtype=np.uint8),
            "env_steps": np.arange(length),
            "policy_call_idx": np.arange(length),
            "action_chunk_offset": np.arange(length),
            "robot_states": np.zeros((length, 10)),
        }
        for key, value in arrays.items():
            demo.create_dataset(key, data=value)
        obs.create_dataset("agentview_rgb", data=np.zeros((length, 2, 2, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((length, 2, 2, 3), dtype=np.uint8))
        obs.create_dataset("gripper_states", data=np.zeros((length, 2)))
        obs.create_dataset("joint_states", data=np.zeros((length, 7)))
        obs.create_dataset("ee_states", data=np.zeros((length, 6)))
        obs.create_dataset("ee_pos", data=np.zeros((length, 3)))
        obs.create_dataset("ee_ori", data=np.zeros((length, 3)))


def _write_pi_run(root: Path) -> tuple[Path, Path]:
    success_path = root / "episodes/success.hdf5"
    failure_path = root / "episodes/failure.hdf5"
    _write_pi_episode(success_path, episode_idx=0, success=True)
    _write_pi_episode(failure_path, episode_idx=1, success=False)
    index = root / "indexes/all_episodes.tsv"
    index.parent.mkdir(parents=True)
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_suite", "task_id", "episode_idx", "success", "trajectory_path"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_suite": "libero_10",
                "task_id": 3,
                "episode_idx": 0,
                "success": "True",
                "trajectory_path": "episodes/success.hdf5",
            }
        )
        writer.writerow(
            {
                "task_suite": "libero_10",
                "task_id": 3,
                "episode_idx": 1,
                "success": "False",
                "trajectory_path": "episodes/failure.hdf5",
            }
        )
    return success_path, failure_path


def _write_smol(manifest: Path, *, success: bool = False) -> Path:
    trajectory = manifest.parent / "smol_episode.npz"
    length = 2
    np.savez_compressed(
        trajectory,
        language=np.array("put the mug down"),
        **{
            "observation.images.image": np.zeros((length, 2, 2, 3), dtype=np.uint8),
            "observation.images.image2": np.ones((length, 2, 2, 3), dtype=np.uint8),
            "observation.state": np.zeros((length, 8), dtype=np.float32),
            "actions": np.zeros((length, 7), dtype=np.float32),
            "timestep": np.arange(length, dtype=np.int32),
            "task_id": np.array(4, dtype=np.int32),
            "episode_index": np.array(2, dtype=np.int32),
            "seed": np.array(7002, dtype=np.int64),
            "suite": np.array("libero_10"),
            "success": np.array(success),
            "rewards": np.zeros(length, dtype=np.float32),
            "dones": np.array([0, 1], dtype=np.uint8),
        },
    )
    row = {
        "path": str(trajectory),
        "format": "mem_smolvla_libero_npz_v2",
        "suite": "libero_10",
        "group_seed": 7,
        "seed": 7002,
        "task_id": 4,
        "episode_index": 2,
        "prompt": "put the mug down",
        "success": success,
        "trajectory_length": length,
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return trajectory


def test_build_combined_manifest_schema_and_stable_ids(tmp_path: Path) -> None:
    pi_root = tmp_path / "libero10_pi05_base_seed17"
    _write_pi_run(pi_root)
    smol_manifest = tmp_path / "smol/manifest.jsonl"
    smol_manifest.parent.mkdir()
    _write_smol(smol_manifest)

    rows = build_manifest(pi_run_roots=[pi_root], smol_manifest=smol_manifest, outcome="all", smol_workers=2)
    again = build_manifest(pi_run_roots=[pi_root], smol_manifest=smol_manifest, outcome="all", smol_workers=2)
    required = {
        "index",
        "action_id",
        "source_format",
        "trajectory_path",
        "source_family",
        "source_model",
        "source_run",
        "producer",
        "seed",
        "group_seed",
        "suite",
        "task_id",
        "episode_idx",
        "success",
        "prompt",
        "frame_index",
        "trajectory_length",
        "image_transform",
        "split",
    }
    assert len(rows) == 3
    assert all(required <= row.keys() for row in rows)
    assert [row["action_id"] for row in rows] == [row["action_id"] for row in again]
    assert len({row["action_id"] for row in rows}) == 3
    assert all(Path(row["trajectory_path"]).is_absolute() for row in rows)
    assert {row["source_format"] for row in rows} == {"eval_hdf5", "smol_npz"}
    assert rows[0]["image_transform"] == "flip_height_width_then_policy_preprocess"
    assert rows[-1]["image_transform"] == "already_flipped_then_policy_preprocess"
    assert all(row["frame_index"] == 0 and row["split"] == "all" for row in rows)


@pytest.mark.parametrize(("outcome", "expected"), [("success", [True]), ("failure", [False, False])])
def test_outcome_filters(tmp_path: Path, outcome: str, expected: list[bool]) -> None:
    pi_root = tmp_path / "run"
    _write_pi_run(pi_root)
    smol_manifest = tmp_path / "smol/manifest.jsonl"
    smol_manifest.parent.mkdir()
    _write_smol(smol_manifest)
    rows = build_manifest(pi_run_roots=[pi_root], smol_manifest=smol_manifest, outcome=outcome)
    assert [row["success"] for row in rows] == expected
    assert [row["index"] for row in rows] == list(range(len(rows)))


def test_rejects_duplicate_identity_and_source_path(tmp_path: Path) -> None:
    pi_root = tmp_path / "run"
    _write_pi_run(pi_root)
    with pytest.raises(RuntimeError, match="Duplicate trajectory identity"):
        build_manifest(pi_run_roots=[pi_root, pi_root], smol_manifest=None, outcome="all")

    other_root = tmp_path / "other_run"
    other_index = other_root / "indexes/all_episodes.tsv"
    other_index.parent.mkdir(parents=True)
    shared_path = (pi_root / "episodes/success.hdf5").resolve()
    with other_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_suite", "task_id", "episode_idx", "success", "trajectory_path"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_suite": "libero_10",
                "task_id": 3,
                "episode_idx": 0,
                "success": "True",
                "trajectory_path": shared_path,
            }
        )
    with pytest.raises(RuntimeError, match="Duplicate trajectory source path"):
        build_manifest(pi_run_roots=[pi_root, other_root], smol_manifest=None, outcome="all")


def test_rejects_missing_file_and_empty_filter(tmp_path: Path) -> None:
    pi_root = tmp_path / "run"
    _, failure_path = _write_pi_run(pi_root)
    failure_path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing trajectory"):
        build_manifest(pi_run_roots=[pi_root], smol_manifest=None, outcome="all")

    smol_manifest = tmp_path / "smol/manifest.jsonl"
    smol_manifest.parent.mkdir()
    _write_smol(smol_manifest, success=False)
    with pytest.raises(RuntimeError, match="empty manifest"):
        build_manifest(pi_run_roots=[], smol_manifest=smol_manifest, outcome="success")


def test_cli_writes_atomic_outputs_and_requires_overwrite(tmp_path: Path) -> None:
    pi_root = tmp_path / "run"
    _write_pi_run(pi_root)
    output = tmp_path / "output/manifest.jsonl"
    args = ["--pi-run-root", str(pi_root), "--output", str(output)]
    main(args)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert len(rows) == summary["trajectories"] == 2
    assert summary["successes"] == summary["failures"] == 1
    with pytest.raises(FileExistsError, match="--overwrite"):
        main(args)
    main([*args, "--overwrite"])
