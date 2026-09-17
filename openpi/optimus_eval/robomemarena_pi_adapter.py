from __future__ import annotations

import time
from typing import Any

import numpy as np
import websockets.sync.client

from openpi_client import msgpack_numpy
from policy_adapter import BasePolicyAdapter


class PiRoboMemArenaAdapter(BasePolicyAdapter):
    """Bridge benchmark observations to the training-aligned OpenPI websocket API."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8700,
        request_timeout: float = 1800.0,
        environment_id: str = "arena-env",
        replan_steps: int = 10,
        max_steps: int = 2500,
    ) -> None:
        self.uri = f"ws://{host}:{int(port)}"
        self.request_timeout = float(request_timeout)
        self.environment_id = str(environment_id)
        self.replan_steps = int(replan_steps)
        self.max_steps = int(max_steps)
        self.packer = msgpack_numpy.Packer()
        self.connection = None
        self.last_error: str | None = None
        self.task_id = 0
        self.episode_idx = 0
        self.episode_seed = 0
        self.inference_call = 0
        self.episode_reset_pending = True
        self.action_generation_timings: list[dict[str, float | int]] = []

    def set_episode_context(self, *, task_id: int, episode_idx: int, seed: int) -> None:
        self.task_id = int(task_id)
        self.episode_idx = int(episode_idx)
        self.episode_seed = int(seed)
        self.inference_call = 0
        self.episode_reset_pending = True
        self.action_generation_timings = []

    def connect(self) -> None:
        self.close()
        self.connection = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=300.0,
            close_timeout=10.0,
            ping_interval=None,
            ping_timeout=None,
        )
        self.connection.recv(timeout=30.0)  # Server metadata.

    def reset(self) -> None:
        self.last_error = None
        self.inference_call = 0
        self.episode_reset_pending = True
        self.action_generation_timings = []

    def timing_samples(self) -> list[dict[str, float | int]]:
        return [dict(item) for item in self.action_generation_timings]

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        del resize_size
        progress = min(
            float(self.inference_call * self.replan_steps) / float(max(1, self.max_steps)),
            1.0,
        )
        element = {
            "observation/image": np.asarray(obs["observation/image"], dtype=np.uint8),
            "observation/wrist_image": np.asarray(obs["observation/wrist_image"], dtype=np.uint8),
            "observation/state": np.asarray(obs["observation/state"], dtype=np.float32),
            "prompt": str(prompt),
            "episode_reset": self.episode_reset_pending,
            "progress": progress,
            "trace_context": {
                "environment_id": self.environment_id,
                "task_id": self.task_id,
                "episode_idx": self.episode_idx,
                "episode_seed": self.episode_seed,
                "inference_call": self.inference_call,
            },
        }
        try:
            if self.connection is None:
                self.connect()
            assert self.connection is not None
            roundtrip_start = time.perf_counter()
            self.connection.send(self.packer.pack(element))
            response = self.connection.recv(timeout=self.request_timeout)
            roundtrip_ms = (time.perf_counter() - roundtrip_start) * 1000.0
            if isinstance(response, str):
                raise RuntimeError(f"Policy server returned an error: {response}")
            payload = msgpack_numpy.unpackb(response)
            actions = np.asarray(payload["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise RuntimeError(f"Expected Pi actions [horizon, 7], got {actions.shape}")
            if not np.isfinite(actions).all():
                raise RuntimeError("Pi policy returned non-finite actions")
            policy_timing = payload.get("policy_timing", {})
            server_timing = payload.get("server_timing", {})
            policy_batch_ms = float(policy_timing.get("infer_ms", float("nan")))
            batch_size = max(1, int(policy_timing.get("batch_size", 1)))
            timing = {
                "inference_call": int(self.inference_call),
                "policy_batch_ms": policy_batch_ms,
                "policy_per_sample_ms": policy_batch_ms / batch_size,
                "server_infer_ms": float(server_timing.get("infer_ms", float("nan"))),
                "client_roundtrip_ms": roundtrip_ms,
                "batch_size": batch_size,
            }
            if all(
                np.isfinite(float(timing[key]))
                for key in (
                    "policy_batch_ms",
                    "policy_per_sample_ms",
                    "server_infer_ms",
                    "client_roundtrip_ms",
                )
            ):
                self.action_generation_timings.append(timing)
            self.episode_reset_pending = False
            self.inference_call += 1
            return actions
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.close()
            raise

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None


def build_adapter(**kwargs: Any) -> PiRoboMemArenaAdapter:
    return PiRoboMemArenaAdapter(**kwargs)
