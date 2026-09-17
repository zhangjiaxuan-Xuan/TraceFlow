from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest
import torch

from openpi.task_head.memory_init import ActionMemorySession, MemoryInitProvider, _interpolate_local_peak


def test_static_action_diagnostics_use_raw_absolute_steps() -> None:
    session = object.__new__(ActionMemorySession)
    session.raw_blocks_all = [
        np.zeros((1, 10, 7), dtype=np.float32),
        np.ones((1, 10, 7), dtype=np.float32),
    ]
    session.weights = np.asarray([0.75, 0.25], dtype=np.float32)

    diagnostics = session.static_action_diagnostics(
        0.0,
        arm_dim=6,
        action_threshold=1e-8,
        gate_fraction=0.5,
    )

    assert diagnostics["block_static_fractions"] == [1.0, 0.0]
    assert diagnostics["weighted_static_fraction"] == pytest.approx(0.75)
    assert diagnostics["static_steps"] == 10
    assert diagnostics["total_steps"] == 20
    assert diagnostics["gate"] is True


def test_anchor_keys_share_actions_and_query_deduplicates(tmp_path: Path) -> None:
    keys = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    faiss.normalize_L2(keys)
    index = faiss.IndexFlatIP(2)
    index.add(keys)
    faiss.write_index(index, str(tmp_path / "memory.index"))
    metadata = [
        {"task_emb": torch.from_numpy(keys[0]), "action_id": "a", "anchor_progress": 0.0},
        {"task_emb": torch.from_numpy(keys[1]), "action_id": "a", "anchor_progress": 0.5},
        {"task_emb": torch.from_numpy(keys[2]), "action_id": "b", "anchor_progress": 0.5},
        {"task_emb": torch.from_numpy(keys[3]), "action_id": "c", "anchor_progress": 1.0},
    ]
    torch.save(metadata, tmp_path / "memory.pt")
    actions = np.arange(18, dtype=np.float32).reshape(6, 3)
    with (tmp_path / "actions.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            actions=actions,
            offsets=np.asarray([0, 2, 4, 6]),
            ids=np.asarray(["a", "b", "c"]),
        )
    provider = MemoryInitProvider(
        str(tmp_path / "memory.pt"),
        str(tmp_path / "memory.index"),
        memory_actions_path=str(tmp_path / "actions.npz"),
        progress_window=0.2,
        sigma_min=0.01,
        device="cpu",
    )
    _, _, info = provider.query(torch.tensor([1.0, 0.0]), k=2, action_horizon=2, progress=0.5)
    assert info["k"] == 2
    assert {source["action_id"] for source in info["sources"]} == {"a", "b"}


@pytest.mark.parametrize("retrieval_backend", ["cpu"])
def test_allowed_task_ids_isolate_retrieval(tmp_path: Path, retrieval_backend: str) -> None:
    keys = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    faiss.normalize_L2(keys)
    index = faiss.IndexFlatIP(2)
    index.add(keys)
    faiss.write_index(index, str(tmp_path / "memory.index"))
    torch.save(
        [
            {"task_emb": torch.from_numpy(keys[0]), "task_id": 1, "action_id": "task1"},
            {"task_emb": torch.from_numpy(keys[1]), "task_id": 2, "action_id": "task2"},
            {"task_emb": torch.from_numpy(keys[2]), "task_id": 2, "action_id": "task2b"},
        ],
        tmp_path / "memory.pt",
    )
    np.savez_compressed(
        tmp_path / "actions.npz",
        actions=np.arange(12, dtype=np.float32).reshape(6, 2),
        offsets=np.asarray([0, 2, 4, 6]),
        ids=np.asarray(["task1", "task2", "task2b"]),
    )
    provider = MemoryInitProvider(
        str(tmp_path / "memory.pt"),
        str(tmp_path / "memory.index"),
        memory_actions_path=str(tmp_path / "actions.npz"),
        allowed_task_ids={2},
        retrieval_backend=retrieval_backend,
        device="cpu",
    )

    _, _, info = provider.query(torch.tensor([1.0, 0.0]), k=1, action_horizon=2)

    assert info["sources"][0]["action_id"] == "task2"
    assert provider.allowed_task_ids == frozenset({2})


def test_nfe_floor_couples_steps_and_prior_noise() -> None:
    provider = object.__new__(MemoryInitProvider)
    provider.noise_low = 0.2
    provider.noise_high = 1.0
    provider.nfe_min = 1
    provider.nfe_max = 10
    provider.nfe_floor = 3

    assert provider._raw_nfe_continuous_from_similarity(1.0) == 1.0
    assert provider._nfe_continuous_from_similarity(1.0) == 3.0
    assert provider._nfe_from_similarity(1.0) == 3
    np.testing.assert_allclose(
        provider._lambda_from_similarity(1.0),
        0.2 + (3.0 - 1.0) / (10.0 - 1.0) * (1.0 - 0.2),
        rtol=0.0,
        atol=1e-12,
    )

    similarity_for_nfe_four = 1.0 / 3.0
    np.testing.assert_allclose(
        provider._nfe_continuous_from_similarity(similarity_for_nfe_four),
        4.0,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        provider._lambda_from_similarity(similarity_for_nfe_four),
        0.2 + (4.0 - 1.0) / (10.0 - 1.0) * (1.0 - 0.2),
        rtol=0.0,
        atol=1e-12,
    )


def test_local_peak_interpolation_recovers_sub_anchor_frame() -> None:
    frames = np.asarray([0, 40, 100], dtype=np.float32)
    target = 63.0
    scores = -((frames - target) / 100.0) ** 2

    assert _interpolate_local_peak(frames, scores) == pytest.approx(target, abs=1e-5)


def test_continuous_frame_alignment_tracks_dense_actions_without_regression(tmp_path: Path) -> None:
    angles = np.asarray([-0.4, 0.0, 0.4], dtype=np.float32)
    keys = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(keys)
    faiss.write_index(index, str(tmp_path / "memory.index"))
    metadata = [
        {
            "task_emb": torch.from_numpy(keys[index]),
            "task_id": 1,
            "action_id": "trajectory",
            "anchor_progress": frame / 20.0,
            "anchor_frame": frame,
            "length": 21,
        }
        for index, frame in enumerate((0, 10, 20))
    ]
    torch.save(metadata, tmp_path / "memory.pt")
    actions = np.stack((np.arange(21), np.arange(21) + 100), axis=1).astype(np.float32)
    np.savez_compressed(
        tmp_path / "actions.npz",
        actions=actions,
        offsets=np.asarray([0, 21]),
        ids=np.asarray(["trajectory"]),
    )
    provider = MemoryInitProvider(
        str(tmp_path / "memory.pt"),
        str(tmp_path / "memory.index"),
        memory_actions_path=str(tmp_path / "actions.npz"),
        action_alignment="continuous_frame_v2",
        device="cpu",
    )
    query_angle = 0.15
    query = torch.tensor([np.cos(query_angle), np.sin(query_angle)], dtype=torch.float32)
    session = ActionMemorySession(provider, query, k=1, H=3, progress=0.0)
    blocks, _, debug = session.guidance_tensors(progress=0.0, device="cpu")

    first_frame = int(debug["window_starts"][0])
    assert 10 < first_frame < 20
    np.testing.assert_array_equal(blocks[0].numpy(), actions[first_frame : first_frame + 3])
    assert debug["action_alignment"] == "continuous_frame_v2"
    assert debug["progress_source"] == "retrieval_continuous_frame"

    exact_anchor = ActionMemorySession(provider, torch.from_numpy(keys[1]), k=1, H=3)
    _, _, exact_debug = exact_anchor.guidance_tensors(progress=0.0, device="cpu")
    assert exact_debug["window_starts"] == [10]

    earlier_query = torch.tensor([np.cos(-0.3), np.sin(-0.3)], dtype=torch.float32)
    assert session.maybe_refresh(earlier_query, step_idx=1, refresh_every=1, k=1)
    held_blocks, _, held_debug = session.guidance_tensors(progress=0.0, device="cpu")
    held_frame = int(held_debug["window_starts"][0])
    assert held_frame == first_frame
    np.testing.assert_array_equal(held_blocks[0].numpy(), actions[held_frame : held_frame + 3])


def test_dense_frame_alignment_uses_real_anchor_and_local_forward_tracking(tmp_path: Path) -> None:
    angles = np.linspace(-0.8, 0.8, 5, dtype=np.float32)
    keys = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(keys)
    faiss.write_index(index, str(tmp_path / "memory.index"))
    frames = (0, 10, 20, 30, 40)
    metadata = [
        {
            "task_emb": torch.from_numpy(keys[index]),
            "task_id": 1,
            "action_id": "trajectory",
            "anchor_progress": frame / 40.0,
            "anchor_frame": frame,
            "length": 41,
        }
        for index, frame in enumerate(frames)
    ]
    torch.save(metadata, tmp_path / "memory.pt")
    actions = np.stack((np.arange(41), np.arange(41) + 100), axis=1).astype(np.float32)
    np.savez_compressed(
        tmp_path / "actions.npz",
        actions=actions,
        offsets=np.asarray([0, 41]),
        ids=np.asarray(["trajectory"]),
    )
    provider = MemoryInitProvider(
        str(tmp_path / "memory.pt"),
        str(tmp_path / "memory.index"),
        memory_actions_path=str(tmp_path / "actions.npz"),
        action_alignment="dense_frame_v3",
        device="cpu",
    )

    between = torch.tensor([np.cos(0.18), np.sin(0.18)], dtype=torch.float32)
    session = ActionMemorySession(provider, between, k=1, H=3)
    _, _, initial = session.guidance_tensors(progress=0.0, device="cpu")
    assert initial["window_starts"] == [20]
    assert initial["progress_source"] == "retrieval_dense_frame_v3"

    backward = torch.from_numpy(keys[0])
    assert session.maybe_refresh(
        backward,
        step_idx=1,
        refresh_every=1,
        k=1,
        advance_steps=10,
    )
    blocks, _, refreshed = session.guidance_tensors(progress=0.0, device="cpu")
    assert refreshed["window_starts"] == [30]
    np.testing.assert_array_equal(blocks[0].numpy(), actions[30:33])


def test_anchor_aligned_query_uses_visual_progress_and_exact_forward_actions(tmp_path: Path) -> None:
    keys = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.1, 0.99],
            [0.99, 0.1],
        ],
        dtype=np.float32,
    )
    faiss.normalize_L2(keys)
    index = faiss.IndexFlatIP(2)
    index.add(keys)
    faiss.write_index(index, str(tmp_path / "memory.index"))
    metadata = [
        {
            "task_emb": torch.from_numpy(keys[0]),
            "task_id": 1,
            "action_id": "a",
            "anchor_progress": 0.0,
            "anchor_frame": 0,
            "length": 6,
        },
        {
            "task_emb": torch.from_numpy(keys[1]),
            "task_id": 1,
            "action_id": "a",
            "anchor_progress": 0.8,
            "anchor_frame": 4,
            "length": 6,
        },
        {
            "task_emb": torch.from_numpy(keys[2]),
            "task_id": 1,
            "action_id": "b",
            "anchor_progress": 0.0,
            "anchor_frame": 0,
            "length": 6,
        },
        {
            "task_emb": torch.from_numpy(keys[3]),
            "task_id": 1,
            "action_id": "b",
            "anchor_progress": 0.8,
            "anchor_frame": 3,
            "length": 6,
        },
    ]
    torch.save(metadata, tmp_path / "memory.pt")
    action_a = np.arange(12, dtype=np.float32).reshape(6, 2)
    action_b = 100.0 + np.arange(12, dtype=np.float32).reshape(6, 2)
    with (tmp_path / "actions.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            actions=np.concatenate((action_a, action_b)),
            offsets=np.asarray([0, 6, 12]),
            ids=np.asarray(["a", "b"]),
        )
    provider = MemoryInitProvider(
        str(tmp_path / "memory.pt"),
        str(tmp_path / "memory.index"),
        memory_actions_path=str(tmp_path / "actions.npz"),
        progress_window=0.2,
        device="cpu",
    )

    # The intentionally wrong client clock must not pull retrieval back to 0.
    session = ActionMemorySession(provider, torch.tensor([1.0, 0.0]), k=2, H=3, progress=0.0)
    blocks, _, debug = session.guidance_tensors(progress=0.0, device="cpu")

    assert provider.anchor_aligned
    assert debug["action_alignment"] == "anchor_forward_v1"
    assert debug["progress_source"] == "retrieval_anchor"
    assert debug["retrieval_progress"] == pytest.approx(0.8)
    np.testing.assert_array_equal(blocks[0].numpy(), np.stack((action_a[4], action_a[5], action_a[5])))
    np.testing.assert_array_equal(blocks[1].numpy(), action_b[3:6])

    assert session.maybe_refresh(
        torch.tensor([0.0, 1.0]),
        step_idx=1,
        refresh_every=1,
        k=2,
    )
    refreshed, _, refreshed_debug = session.guidance_tensors(progress=1.0, device="cpu")
    assert refreshed_debug["retrieval_progress"] == pytest.approx(0.0)
    np.testing.assert_array_equal(refreshed[0].numpy(), action_a[:3])
    np.testing.assert_array_equal(refreshed[1].numpy(), action_b[:3])


def test_legacy_memory_keeps_progress_indexing(tmp_path: Path) -> None:
    keys = np.asarray([[1.0, 0.0]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(keys)
    faiss.write_index(index, str(tmp_path / "memory.index"))
    torch.save(
        [
            {
                "task_emb": torch.from_numpy(keys[0]),
                "action_id": "a",
                "anchor_progress": 0.5,
                "length": 6,
                "chunk_meta": {"chunk_len": 2, "stride": 1, "T": 6},
            }
        ],
        tmp_path / "memory.pt",
    )
    actions = np.arange(12, dtype=np.float32).reshape(6, 2)
    with (tmp_path / "actions.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            actions=actions,
            offsets=np.asarray([0, 6]),
            ids=np.asarray(["a"]),
        )
    provider = MemoryInitProvider(
        str(tmp_path / "memory.pt"),
        str(tmp_path / "memory.index"),
        memory_actions_path=str(tmp_path / "actions.npz"),
        device="cpu",
    )
    session = ActionMemorySession(provider, torch.tensor([1.0, 0.0]), k=1, H=2, progress=0.0)
    start, _, _ = session._chosen_blocks(0.0)
    end, _, _ = session._chosen_blocks(1.0)

    assert not provider.anchor_aligned
    np.testing.assert_array_equal(start[0], actions[:2])
    np.testing.assert_array_equal(end[0], actions[4:6])


def test_search_batch_can_enforce_a_per_query_task_gate(tmp_path: Path) -> None:
    keys = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0]], dtype=np.float32
    )
    faiss.normalize_L2(keys)
    index = faiss.IndexFlatIP(2)
    index.add(keys)
    faiss.write_index(index, str(tmp_path / "memory.index"))
    torch.save(
        [
            {"task_emb": torch.from_numpy(keys[0]), "task_id": 1, "action_id": "a"},
            {"task_emb": torch.from_numpy(keys[1]), "task_id": 1, "action_id": "b"},
            {"task_emb": torch.from_numpy(keys[2]), "task_id": 2, "action_id": "c"},
            {"task_emb": torch.from_numpy(keys[3]), "task_id": 2, "action_id": "d"},
        ],
        tmp_path / "memory.pt",
    )
    provider = MemoryInitProvider(
        str(tmp_path / "memory.pt"),
        str(tmp_path / "memory.index"),
        device="cpu",
    )

    results = provider.search_batch(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        k=1,
        required_task_ids=[1, 2],
    )

    assert int(results[0][1][0]) in {0, 1}
    assert int(results[1][1][0]) in {2, 3}
