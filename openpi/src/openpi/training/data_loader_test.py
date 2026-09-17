import dataclasses

import jax
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_video_query_retries_transient_container_error(monkeypatch):
    calls = 0

    def decode(video_path, query_timestamps, tolerance_s, backend):
        nonlocal calls
        del video_path, query_timestamps, tolerance_s, backend
        calls += 1
        if calls < 3:
            raise RuntimeError("Invalid data found when processing input: moov atom not found")
        return "decoded"

    monkeypatch.setenv("OPENPI_VIDEO_READ_ATTEMPTS", "3")
    monkeypatch.setenv("OPENPI_VIDEO_READ_RETRY_DELAY_S", "0")
    monkeypatch.setattr(_data_loader.lerobot_video_utils, "decode_video_frames", decode)

    assert _data_loader._decode_video_frames_with_retry(
        _data_loader.pathlib.Path("video.mp4"),
        [0.0],
        tolerance_s=0.01,
        backend="pyav",
        episode_index=267,
    ) == "decoded"
    assert calls == 3


def test_video_query_does_not_hide_permanent_error(monkeypatch):
    calls = 0

    def decode(video_path, query_timestamps, tolerance_s, backend):
        nonlocal calls
        del video_path, query_timestamps, tolerance_s, backend
        calls += 1
        raise ValueError("timestamp alignment is invalid")

    monkeypatch.setenv("OPENPI_VIDEO_READ_ATTEMPTS", "5")
    monkeypatch.setattr(_data_loader.lerobot_video_utils, "decode_video_frames", decode)

    with pytest.raises(ValueError, match="timestamp alignment"):
        _data_loader._decode_video_frames_with_retry(
            _data_loader.pathlib.Path("video.mp4"),
            [0.0],
            tolerance_s=0.01,
            backend="pyav",
            episode_index=267,
        )
    assert calls == 1


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_strict_resume_random_sampler_matches_uninterrupted_stream():
    dataset = list(range(23))
    batch_size = 4
    seed = 42
    start_batch = 7
    expected_batches = 8

    generator = torch.Generator().manual_seed(seed)
    uninterrupted = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    uninterrupted_iter = iter(uninterrupted)
    expected = []
    for batch_index in range(start_batch + expected_batches):
        try:
            batch = next(uninterrupted_iter)
        except StopIteration:
            uninterrupted_iter = iter(uninterrupted)
            batch = next(uninterrupted_iter)
        if batch_index >= start_batch:
            expected.append(batch.tolist())

    for num_workers in (0, 2):
        resumed_sampler = _data_loader._StrictResumeRandomSampler(
            dataset,
            batch_size=batch_size,
            seed=seed,
            start_batch=start_batch,
        )
        resumed = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=resumed_sampler,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=num_workers,
            multiprocessing_context="spawn" if num_workers else None,
        )
        resumed_iter = iter(resumed)
        actual = []
        for _ in range(expected_batches):
            try:
                batch = next(resumed_iter)
            except StopIteration:
                resumed_iter = iter(resumed)
                batch = next(resumed_iter)
            actual.append(batch.tolist())

        assert actual == expected


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
