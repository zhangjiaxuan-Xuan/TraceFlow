from __future__ import annotations

import h5py
import json
import numpy as np
import torch

from scripts.memory import cache_prior_head_features as temporal_cache
from scripts.memory.cache_prior_head_features import _temporal_layout
from scripts.memory.cache_prior_head_features import _temporal_observation_row
from openpi.task_head import cl_features
from openpi.task_head.cl_features import load_cl_observation
from openpi.task_head.cl_features import load_cl_observations


def test_load_eval_hdf5_rotates_raw_images(tmp_path) -> None:
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        obs = demo.create_group("obs")
        demo.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float32))
        image = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
        obs.create_dataset("agentview_rgb", data=image)
        obs.create_dataset("eye_in_hand_rgb", data=image + 20)
        obs.create_dataset("ee_states", data=np.zeros((1, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.ones((1, 2), dtype=np.float32))

    result = load_cl_observation(
        {
            "source_format": "eval_hdf5",
            "trajectory_path": str(path),
            "frame_index": 0,
            "prompt": "task",
        }
    )
    np.testing.assert_array_equal(result["observation/image"], image[0, ::-1, ::-1])
    assert result["observation/state"].shape == (8,)


def test_batch_hdf5_loader_opens_shared_file_once(tmp_path, monkeypatch) -> None:
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        obs = demo.create_group("obs")
        demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
        image = np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)
        obs.create_dataset("agentview_rgb", data=image)
        obs.create_dataset("eye_in_hand_rgb", data=image + 20)
        obs.create_dataset("ee_states", data=np.zeros((3, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.ones((3, 2), dtype=np.float32))

    real_file = h5py.File
    opens = []

    def tracked_file(*args, **kwargs):
        opens.append(args[0])
        return real_file(*args, **kwargs)

    monkeypatch.setattr(cl_features.h5py, "File", tracked_file)
    result = load_cl_observations(
        [
            {"trajectory_path": str(path), "frame_index": frame, "prompt": "task"}
            for frame in (0, 1, 2)
        ]
    )
    assert len(result) == 3
    assert opens == [str(path)]


def test_load_smol_npz_streams_one_frame_without_numpy_npz_inflation(tmp_path, monkeypatch) -> None:
    path = tmp_path / "episode.npz"
    image = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    np.savez_compressed(
        path,
        **{
            "observation.images.image": image,
            "observation.images.image2": image + 20,
            "observation.state": np.zeros((2, 8), dtype=np.float32),
            "actions": np.zeros((2, 7), dtype=np.float32),
        },
    )
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("np.load used")))
    result = load_cl_observation(
        {
            "source_format": "smol_npz",
            "trajectory_path": str(path),
            "frame_index": 1,
            "prompt": "task",
        }
    )
    np.testing.assert_array_equal(result["observation/image"], image[1])
    assert result["observation/state"].shape == (8,)


def test_load_traceflow_lerobot_preserves_three_cameras_and_identity(tmp_path, monkeypatch) -> None:
    image = np.arange(18, dtype=np.float32).reshape(3, 2, 3)

    class FakeDataset:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            assert index == 2
            return {
                "episode_index": np.asarray(1),
                "frame_index": np.asarray(0),
                "observation.images.cam_high": image,
                "observation.images.cam_left_wrist": image + 20,
                "observation.images.cam_right_wrist": image + 40,
                "observation.state": np.zeros(14, dtype=np.float32),
            }

    monkeypatch.setattr(cl_features, "_traceflow_dataset", lambda root, repo_id: FakeDataset())
    result = load_cl_observation(
        {
            "source_format": "lerobot_traceflow",
            "trajectory_path": str(tmp_path / "episode.parquet"),
            "dataset_root": str(tmp_path),
            "repo_id": "traceflow/test",
            "dataset_index": 2,
            "episode_index": 1,
            "frame_index": 0,
            "prompt": "task",
        }
    )
    assert set(result["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    np.testing.assert_array_equal(result["images"]["cam_right_wrist"], image + 40)
    assert result["state"].shape == (14,)


def test_parallel_traceflow_loader_preserves_requested_order(tmp_path, monkeypatch) -> None:
    image = np.zeros((3, 2, 3), dtype=np.float32)

    class FakeDataset:
        def __len__(self):
            return 8

        def __getitem__(self, index):
            return {
                "episode_index": np.asarray(0),
                "frame_index": np.asarray(index),
                "observation.images.cam_high": image + index,
                "observation.images.cam_left_wrist": image + index,
                "observation.images.cam_right_wrist": image + index,
                "observation.state": np.full(14, index, dtype=np.float32),
            }

    monkeypatch.setattr(cl_features, "_traceflow_dataset", lambda root, repo_id: FakeDataset())
    rows = [
        {
            "source_format": "lerobot_traceflow",
            "trajectory_path": str(tmp_path / "episode.parquet"),
            "dataset_root": str(tmp_path),
            "repo_id": "traceflow/test",
            "dataset_index": frame,
            "episode_index": 0,
            "frame_index": frame,
            "prompt": "task",
        }
        for frame in (5, 1, 7, 2)
    ]
    result = load_cl_observations(rows, loader_workers=4)
    assert [int(item["state"][0]) for item in result] == [5, 1, 7, 2]


def test_traceflow_batch_loader_decodes_each_camera_once(tmp_path, monkeypatch) -> None:
    camera_keys = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )

    class FakeHfDataset:
        def select(self, indices):
            return {
                "episode_index": [0 for _ in indices],
                "frame_index": list(indices),
                "timestamp": [index / 50 for index in indices],
                "observation.state": [np.full(14, index, dtype=np.float32) for index in indices],
            }

    class FakeMeta:
        video_keys = camera_keys

        @staticmethod
        def get_video_file_path(_episode, camera):
            return f"{camera}.mp4"

    class FakeDataset:
        hf_dataset = FakeHfDataset()
        meta = FakeMeta()
        root = tmp_path
        tolerance_s = 0.0001
        video_backend = "pyav"

        def __len__(self):
            return 8

    calls = []

    def fake_decode(path, timestamps, tolerance, backend):
        calls.append((str(path), tuple(timestamps), tolerance, backend))
        return torch.stack(
            [torch.full((3, 2, 3), timestamp, dtype=torch.float32) for timestamp in timestamps]
        )

    from lerobot.common.datasets import video_utils

    monkeypatch.setattr(cl_features, "_traceflow_dataset", lambda root, repo_id: FakeDataset())
    monkeypatch.setattr(video_utils, "decode_video_frames", fake_decode)
    rows = [
        {
            "source_format": "lerobot_traceflow",
            "trajectory_path": str(tmp_path / "episode.parquet"),
            "dataset_root": str(tmp_path),
            "repo_id": "traceflow/test",
            "dataset_index": frame,
            "episode_index": 0,
            "frame_index": frame,
            "prompt": "task",
        }
        for frame in (5, 1, 7, 2)
    ]
    result = load_cl_observations(rows, loader_workers=3)

    assert len(calls) == 3
    assert all(call[1] == (0.1, 0.02, 0.14, 0.04) for call in calls)
    assert [int(item["state"][0]) for item in result] == [5, 1, 7, 2]
    np.testing.assert_allclose(result[2]["images"]["cam_high"], 0.14)


def test_temporal_feature_rows_use_ordered_real_trajectory_frames() -> None:
    row = {
        "trajectory_path": "current.hdf5",
        "frame_index": 30,
        "temporal_context": [
            {"offset": -20, "trajectory_path": "first.hdf5", "frame_index": 10},
            {"offset": -10, "trajectory_path": "first.hdf5", "frame_index": 20},
            {"offset": 0, "trajectory_path": "current.hdf5", "frame_index": 30},
        ],
    }

    offsets, window = _temporal_layout([row])
    assert offsets == (-20, -10, 0)
    assert window == 3
    assert [
        (
            _temporal_observation_row(row, index)["trajectory_path"],
            _temporal_observation_row(row, index)["frame_index"],
        )
        for index in range(window)
    ] == [
        ("first.hdf5", 10),
        ("first.hdf5", 20),
        ("current.hdf5", 30),
    ]


def test_temporal_traceflow_row_updates_global_dataset_index() -> None:
    row = {
        "trajectory_path": "episode.parquet",
        "frame_index": 30,
        "dataset_index": 1030,
        "temporal_context": [
            {
                "offset": -10,
                "trajectory_path": "episode.parquet",
                "frame_index": 20,
                "dataset_index": 1020,
            },
            {
                "offset": 0,
                "trajectory_path": "episode.parquet",
                "frame_index": 30,
                "dataset_index": 1030,
            },
        ],
    }
    assert _temporal_observation_row(row, 0)["dataset_index"] == 1020


def test_temporal_feature_cache_concatenates_each_history_slot(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for row_index, frame in enumerate((30, 40)):
        rows.append(
            {
                "row_index": row_index,
                "prompt": "task",
                "frame_index": frame,
                "trajectory_path": "current.hdf5",
                "temporal_context": [
                    {"offset": -20, "trajectory_path": "episode.hdf5", "frame_index": frame - 20},
                    {"offset": -10, "trajectory_path": "episode.hdf5", "frame_index": frame - 10},
                    {"offset": 0, "trajectory_path": "episode.hdf5", "frame_index": frame},
                ],
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    monkeypatch.setattr(temporal_cache.training_config, "get_config", lambda _name: object())
    monkeypatch.setattr(
        temporal_cache.policy_config,
        "create_trained_policy",
        lambda *_args, **_kwargs: object(),
    )
    extraction_batches = []

    monkeypatch.setattr(
        temporal_cache,
        "load_cl_observations",
        lambda requested, loader_workers=0: requested,
    )

    def fake_extract(
        _policy,
        batch,
        _device,
        *,
        inference_batch_size=0,
        auto_batch=False,
        min_batch_size=1,
    ):
        assert inference_batch_size == 0
        assert not auto_batch
        assert min_batch_size == 1
        extraction_batches.append([row["frame_index"] for row in batch])
        return np.asarray(
            [[row["frame_index"], -row["frame_index"]] for row in batch],
            dtype=np.float32,
        )

    monkeypatch.setattr(temporal_cache, "_extract_observations", fake_extract)
    output = tmp_path / "features"
    temporal_cache.main(
        temporal_cache.Args(
            manifest=str(manifest),
            output_dir=str(output),
            policy_dir=str(tmp_path / "policy"),
            config_name="test",
            device="cpu",
            batch_size=2,
            feature_dim=2,
            require_cuda=False,
        )
    )

    features = np.load(output / "pooled_prefix.npy")
    assert features.shape == (2, 6)
    np.testing.assert_array_equal(
        features[0],
        np.asarray([10, -10, 20, -20, 30, -30], dtype=np.float32),
    )
    # Shared history frames are extracted once and then scattered back to anchors.
    assert extraction_batches == [[10, 20, 30, 40]]
    state = json.loads((output / "cache_state.json").read_text(encoding="utf-8"))
    assert state["feature_dim"] == 6
    assert state["temporal_window"] == 3
    assert state["temporal_offsets"] == [-20, -10, 0]


def test_feature_loading_batch_is_decoupled_from_inference_microbatch(monkeypatch) -> None:
    rows = [{"frame_index": index} for index in range(11)]
    monkeypatch.setattr(
        temporal_cache,
        "load_cl_observations",
        lambda requested, loader_workers=0: requested,
    )
    microbatches = []

    def fake_extract(_policy, observations, _device):
        microbatches.append([row["frame_index"] for row in observations])
        return np.asarray([[row["frame_index"]] for row in observations], dtype=np.float32)

    monkeypatch.setattr(temporal_cache, "_extract_loaded_batch", fake_extract)
    features = temporal_cache._extract_batch(
        object(),
        rows,
        "cpu",
        loader_workers=4,
        inference_batch_size=4,
    )

    assert microbatches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    np.testing.assert_array_equal(features[:, 0], np.arange(11, dtype=np.float32))


def test_trajectory_chunks_preserve_resume_order_and_boundaries() -> None:
    rows = [
        {"action_id": action_id}
        for action_id in ("episode-a", "episode-a", "episode-a", "episode-b", "episode-b")
    ]
    chunks = temporal_cache._pending_chunks(
        np.asarray([1, 2, 3, 4], dtype=np.int64),
        rows,
        max_batch_size=1024,
        trajectory_batches=True,
    )
    assert [chunk.tolist() for chunk in chunks] == [[1, 2], [3, 4]]


def test_prefetch_pipeline_yields_batches_in_manifest_order(monkeypatch) -> None:
    def fake_prepare(_rows, indices, _window, _workers):
        return temporal_cache._PreparedFeatureBatch(
            indices=indices,
            source_rows=[],
            inverse=np.asarray([], dtype=np.int64),
            observations=[],
            requested_frames=0,
        )

    monkeypatch.setattr(temporal_cache, "_prepare_feature_batch", fake_prepare)
    chunks = [np.asarray([index], dtype=np.int64) for index in range(8)]
    prepared = list(temporal_cache._prefetched_batches([], chunks, 3, 3, 4))
    assert [int(batch.indices[0]) for batch in prepared] == list(range(8))
