# ruff: noqa: SLF001

import json
from types import SimpleNamespace
import threading

import numpy as np
import torch

import openpi.models_pytorch.pi0_pytorch as pi0_module
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.task_head.dual_tower_head import DualTowerRetrievalHead


def _model(tmp_path, *, level: str = "light") -> PI0Pytorch:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_trace_dir=str(tmp_path),
        memory_guidance_trace_level=level,
        memory_provider=SimpleNamespace(memory=[{"task_name": "pick up cup"}, {"task_name": "open drawer"}]),
        _trace_context={"task_id": 3, "task_name": "libero task", "episode_idx": 4, "policy_call_idx": 5},
    )
    return model


def test_light_step_summary_records_norms_and_cosines_without_vectors() -> None:
    base = torch.tensor([[[3.0, 4.0]]])
    positive = torch.tensor([[[0.0, 2.0]]])
    negative = torch.tensor([[[-1.0, 0.0]]])
    final = positive + negative

    record = PI0Pytorch._summarize_memory_guidance_step(
        v_base=base,
        guidance_positive=positive,
        guidance_negative=negative,
        guidance_final=final,
        time=0.8,
        step_index=2,
    )

    assert record["step_index"] == 2
    assert record["v_base"]["norm"] == [5.0]
    assert np.isclose(record["guidance_positive"]["cosine_to_v_base"][0], 0.8)
    assert np.isclose(record["guidance_negative"]["cosine_to_v_base"][0], -0.6)
    assert set(record) == {
        "time",
        "step_index",
        "v_base",
        "guidance_positive",
        "guidance_negative",
        "guidance_final",
        "v_actual",
    }


def test_light_trace_is_compact_and_contains_positive_and_negative_retrieval(tmp_path) -> None:
    model = _model(tmp_path)
    step = PI0Pytorch._summarize_memory_guidance_step(
        v_base=torch.ones((1, 2, 2)),
        guidance_positive=torch.full((1, 2, 2), 0.2),
        guidance_negative=torch.full((1, 2, 2), -0.1),
        guidance_final=torch.full((1, 2, 2), 0.1),
        time=1.0,
        step_index=0,
    )

    model._write_memory_guidance_trace(
        trace_x=[],
        trace_steps=[step],
        blocks=torch.randn((2, 10, 32)),
        weights=torch.tensor([0.75, 0.25]),
        retrieval_debug={
            "scores": [0.99, 0.98],
            "memory_indices": [1, 0],
            "window_starts": [7, 9],
        },
        negative_weights=torch.tensor([1.0]),
        negative_debug={
            "reason": "ok",
            "scores": [0.976],
            "memory_indices": [8],
            "memory_task_names": ["failed drawer"],
        },
        task_emb=torch.randn(2048),
        progress=0.42,
        progress_source="client",
        num_steps=10,
    )

    trace_path = tmp_path / "unknown_task_003_episode_004_call_0005.light.jsonl"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["positive_retrieval"]["task_names"] == ["open drawer", "pick up cup"]
    assert payload["negative_retrieval"]["indices"] == [8]
    assert payload["negative_retrieval"]["weights"] == [1.0]
    assert payload["steps"] == [step]
    serialized = trace_path.read_text(encoding="utf-8")
    for forbidden in ("x_trajectory", "memory_blocks", "task_embedding", "guidance_raw"):
        assert forbidden not in serialized
    assert not list(tmp_path.glob("*.npz"))
    manifest = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest["trace_level"] == "light"
    assert manifest["positive_selected"] == 2
    assert manifest["negative_selected"] == 1


def test_missing_trace_level_preserves_full_npz_format(tmp_path) -> None:
    model = _model(tmp_path, level="full")
    del model.__dict__["memory_guidance_trace_level"]
    tensor = torch.ones((1, 2, 2))
    step = {
        "v_base": tensor,
        "guidance_raw": tensor,
        "guidance": tensor,
        "guidance_positive": tensor,
        "guidance_negative": torch.zeros_like(tensor),
        "responsibilities": torch.ones((1, 1)),
        "dist2": torch.zeros((1, 1)),
        "cap_factor": torch.ones(1),
        "time": 1.0,
        "lambda": 0.2,
    }

    model._write_memory_guidance_trace(
        trace_x=[tensor, tensor],
        trace_steps=[step],
        blocks=tensor,
        weights=torch.ones(1),
        retrieval_debug={"memory_indices": [0], "scores": [0.9]},
        negative_weights=None,
        negative_debug={"reason": "disabled"},
        task_emb=torch.ones(4),
        progress=0.0,
        progress_source="client",
        num_steps=1,
    )

    trace_path = tmp_path / "unknown_task_003_episode_004_call_0005.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        assert int(trace["schema_version"]) == 1
        assert "x_trajectory" in trace
        assert trace["positive_memory_task_names"].tolist() == ["pick up cup"]
        assert trace["negative_memory_indices"].size == 0


def test_bank_trace_uses_async_local_writer_and_keeps_reusable_features(tmp_path) -> None:
    model = _model(tmp_path, level="bank")
    model.memory_guidance_trace_workers = 8
    model._write_memory_guidance_trace(
        trace_x=[],
        trace_steps=[],
        blocks=torch.ones((2, 10, 32)),
        weights=torch.tensor([0.75, 0.25]),
        retrieval_debug={"memory_indices": [0, 1], "scores": [0.9, 0.8], "window_starts": [2, 3]},
        negative_weights=torch.tensor([1.0]),
        negative_debug={"memory_indices": [4], "scores": [0.98]},
        task_emb=torch.ones(2048),
        progress=0.5,
        progress_source="client",
        num_steps=10,
        lower_retrieval_feature=torch.ones(6144),
        upper_retrieval_feature=torch.ones(4096),
        upper_vlm_age=0.1,
        upper_vlm_available=True,
        final_action=torch.ones((10, 32)),
    )
    waiter = threading.Event()
    manifest = tmp_path / "manifest.jsonl"
    for _ in range(100):
        if manifest.is_file():
            break
        waiter.wait(0.01)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    trace_path = tmp_path / row["trace"]
    assert row["trace_level"] == "bank"
    assert row["trace_size"] == trace_path.stat().st_size
    with np.load(trace_path, allow_pickle=False) as trace:
        assert trace["x_trajectory"].shape == (1, 1, 10, 32)
        assert trace["task_embedding"].shape == (2048,)
        assert trace["lower_retrieval_feature"].shape == (6144,)
        assert trace["upper_retrieval_feature"].shape == (4096,)


def test_fixed_nfe_batch_writes_one_light_trace_per_environment(monkeypatch, tmp_path) -> None:
    class FakeSession:
        s_global = 1.0

        def __init__(self, **_kwargs):
            pass

        def maybe_refresh(self, **_kwargs):
            return False

        def guidance_tensors(self, _progress, _device):
            return (
                torch.ones((1, 1, 1)),
                torch.ones(1),
                {"memory_indices": [0], "scores": [0.99], "weights": [1.0], "window_starts": [0]},
            )

    monkeypatch.setattr(pi0_module, "ActionMemorySession", FakeSession)
    model = _model(tmp_path)
    model.__dict__.update(
        config=SimpleNamespace(action_horizon=1, action_dim=1),
        task_head=torch.nn.Identity(),
        memory_top_k=1,
        memory_guidance_min_similarity=-1.0,
        memory_guidance_time_version="v0",
        use_negative_guidance=False,
        negative_memory_provider=None,
        _batched_memory_states={},
        _batch_runtime=[
            {
                "environment_id": "env-0",
                "reset": True,
                "progress": 0.1,
                "seed": 7,
                "context": {"task_id": 0, "episode_idx": 0, "policy_call_idx": 0},
            },
            {
                "environment_id": "env-1",
                "reset": True,
                "progress": 0.2,
                "seed": 8,
                "context": {"task_id": 1, "episode_idx": 2, "policy_call_idx": 3},
            },
        ],
    )
    model.denoise_step = lambda _state, _masks, _cache, x_t, _time: torch.ones_like(x_t)
    model._memory_flow_guidance = lambda _x, v_base, *_args, **_kwargs: (0.5 * v_base, {})

    result = model._sample_actions_batched_memory_guidance(
        device=torch.device("cpu"),
        state=torch.zeros((2, 1)),
        prefix_pad_masks=torch.ones((2, 1), dtype=torch.bool),
        past_key_values=None,
        outputs_embeds=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        noise=torch.zeros((2, 1, 1)),
        num_steps=10,
    )

    assert torch.isfinite(result).all()
    first = json.loads((tmp_path / "unknown_task_000_episode_000_call_0000.light.jsonl").read_text())
    second = json.loads((tmp_path / "unknown_task_001_episode_002_call_0003.light.jsonl").read_text())
    assert len(first["steps"]) == 10
    assert len(second["steps"]) == 10
    assert first["progress"] == 0.1
    assert second["progress"] == 0.2
    assert model._batched_memory_states["env-0"]["call_count"] == 1
    assert model._batched_memory_states["env-1"]["call_count"] == 1


def test_single_row_server_batch_writes_contextual_full_memory_trace(monkeypatch, tmp_path) -> None:
    class FakeSession:
        s_global = 1.0

        def __init__(self, **_kwargs):
            pass

        def maybe_refresh(self, **_kwargs):
            return False

        def guidance_tensors(self, _progress, _device):
            return (
                torch.ones((1, 1, 1)),
                torch.ones(1),
                {"memory_indices": [0], "scores": [0.99], "weights": [1.0], "window_starts": [4]},
            )

    monkeypatch.setattr(pi0_module, "ActionMemorySession", FakeSession)
    model = _model(tmp_path, level="full")
    model.__dict__.update(
        config=SimpleNamespace(action_horizon=1, action_dim=1),
        task_head=torch.nn.Identity(),
        memory_top_k=1,
        memory_guidance_min_similarity=-1.0,
        memory_guidance_time_version="v1",
        memory_guidance_total_norm_cap=0.2,
        use_negative_guidance=False,
        negative_memory_provider=None,
        _batched_memory_states={},
        _batch_runtime=[
            {
                "environment_id": "fusion-envslot0",
                "reset": True,
                "progress": 0.25,
                "seed": 7,
                "context": {
                    "task_suite": "robomemarena",
                    "task_id": 18,
                    "episode_idx": 2,
                    "policy_call_idx": 3,
                },
            }
        ],
    )
    model.denoise_step = lambda _state, _masks, _cache, x_t, _time: torch.ones_like(x_t)

    result = model._sample_actions_batched_memory_guidance(
        device=torch.device("cpu"),
        state=torch.zeros((1, 1)),
        prefix_pad_masks=torch.ones((1, 1), dtype=torch.bool),
        past_key_values=None,
        outputs_embeds=torch.tensor([[[1.0, 0.0]]]),
        noise=torch.zeros((1, 1, 1)),
        num_steps=10,
    )

    assert torch.isfinite(result).all()
    trace_path = tmp_path / "robomemarena_task_018_episode_002_call_0003.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        assert trace["x_trajectory"].shape == (11, 1, 1, 1)
        assert trace["v_base"].shape == (10, 1, 1, 1)
        assert trace["task_embedding"].shape == (2,)
        assert trace["lower_retrieval_feature"].shape == (2,)
        assert trace["upper_retrieval_feature"].shape == (0,)
        assert trace["memory_indices"].tolist() == [0]
        assert trace["window_starts"].tolist() == [4]
    manifest = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert manifest["task_id"] == 18
    assert manifest["episode_idx"] == 2
    assert manifest["policy_call_idx"] == 3


def test_upper_only_guidance_waits_for_first_upper_feature(monkeypatch, tmp_path) -> None:
    class ForbiddenSession:
        def __init__(self, **_kwargs):
            raise AssertionError("Upper-only retrieval must not start without an Upper feature")

    monkeypatch.setattr(pi0_module, "ActionMemorySession", ForbiddenSession)
    model = _model(tmp_path)
    head = DualTowerRetrievalHead(
        variant="upper", lower_dim=2, upper_dim=2, hidden=2, out_dim=2
    )
    model.__dict__.update(
        config=SimpleNamespace(action_horizon=1, action_dim=1),
        task_head=head,
        memory_provider=object(),
        memory_guidance_time_version="v1",
        use_negative_guidance=False,
        negative_memory_provider=None,
        _batched_memory_states={},
        _batch_runtime=[
            {
                "environment_id": "env-upper",
                "reset": True,
                "progress": 0.0,
                "seed": 7,
                "upper_vlm_available": False,
                "context": {"task_id": 1, "episode_idx": 0, "policy_call_idx": 0},
            }
        ],
    )
    model.denoise_step = lambda _state, _masks, _cache, x_t, _time: torch.ones_like(x_t)

    result = model._sample_actions_batched_memory_guidance(
        device=torch.device("cpu"),
        state=torch.zeros((1, 1)),
        prefix_pad_masks=torch.ones((1, 1), dtype=torch.bool),
        past_key_values=None,
        outputs_embeds=torch.tensor([[[1.0, 0.0]]]),
        noise=torch.zeros((1, 1, 1)),
        num_steps=10,
    )

    assert torch.isfinite(result).all()
    assert model._batched_memory_states["env-upper"]["session"] is None
    assert model._batched_memory_states["env-upper"]["call_count"] == 1
