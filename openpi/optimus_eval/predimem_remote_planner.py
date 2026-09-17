from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable

import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client


class RemotePrediMemPlanner:
    """Planner-compatible client for the shared batched upper VLM service."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        environment_id: str,
        task_info: Any,
        request_timeout: float = 1800.0,
        lower_feature_provider: Callable[[], np.ndarray | None] | None = None,
        lower_probe_provider: Callable[[], list[dict[str, Any]]] | None = None,
        lower_probe_candidate_consumer: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.uri = f"ws://{host}:{int(port)}"
        self.environment_id = environment_id
        self.task_info = task_info
        self.default_subtask_prompt = task_info.brief_description.strip()
        self._current_subtask = self.default_subtask_prompt
        self._latest_feature: np.ndarray | None = None
        self._latest_feature_step = -1
        self._episode_reset_pending = True
        self._completed_stage: int | None = None
        self.request_timeout = request_timeout
        self.connection = None
        self._connection_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self.packer = msgpack_numpy.Packer()
        self.logger = None
        self.last_error: str | None = None
        self._timing_samples: list[dict[str, float | int]] = []
        self._guidance_samples: list[dict[str, Any]] = []
        self.lower_feature_provider = lower_feature_provider
        self.lower_probe_provider = lower_probe_provider
        self.lower_probe_candidate_consumer = lower_probe_candidate_consumer

    def connect(self) -> None:
        with self._connection_lock:
            self.close()
            self.connection = websockets.sync.client.connect(
                self.uri,
                compression=None,
                max_size=None,
                open_timeout=300.0,
                close_timeout=10.0,
                ping_interval=None,
                ping_timeout=None,
                proxy=None,
            )
            msgpack_numpy.unpackb(self.connection.recv(timeout=30.0))

    def reset_episode(self, instruction: str | None = None, run_dir=None, logger=None) -> None:
        del instruction, run_dir
        self.logger = logger
        self._current_subtask = self.default_subtask_prompt
        self._latest_feature = None
        self._latest_feature_step = -1
        self._episode_reset_pending = True
        with self._state_lock:
            self._completed_stage = None
        self.last_error = None
        self._timing_samples = []
        self._guidance_samples = []

    def notify_stage_completed(self, stage: int, step: int) -> None:
        """Expose evaluator completion only for explicit diagnostic runs."""
        del step
        completed = int(stage)
        with self._state_lock:
            self._completed_stage = (
                completed if self._completed_stage is None else max(completed, self._completed_stage)
            )

    def infer_sync(
        self,
        step_idx: int,
        context_frames_np: list[tuple[np.ndarray, ...]],
    ) -> str:
        if not context_frames_np:
            return self._current_subtask
        three_view = len(context_frames_np[0]) == 3
        if any(len(frame_pack) != (3 if three_view else 2) for frame_pack in context_frames_np):
            raise ValueError("Upper context mixes dual-view and three-view frame packs")
        payload = {
            "environment_id": self.environment_id,
            "task_id": int(self.task_info.task_id),
            "episode_reset": self._episode_reset_pending,
            "step_idx": int(step_idx),
            "context_main": [np.asarray(frame_pack[0], dtype=np.uint8) for frame_pack in context_frames_np],
        }
        if three_view:
            payload["context_left_wrist"] = [
                np.asarray(frame_pack[1], dtype=np.uint8) for frame_pack in context_frames_np
            ]
            payload["context_right_wrist"] = [
                np.asarray(frame_pack[2], dtype=np.uint8) for frame_pack in context_frames_np
            ]
        else:
            payload["context_wrist"] = [
                np.asarray(frame_pack[1], dtype=np.uint8)
                if frame_pack[1] is not None
                else np.zeros_like(frame_pack[0], dtype=np.uint8)
                for frame_pack in context_frames_np
            ]
        if os.environ.get("WILL_GUIDANCE_ORACLE_COMPLETION", "0") == "1":
            with self._state_lock:
                payload["completed_stage"] = self._completed_stage
        if self.lower_feature_provider is not None:
            lower_feature = self.lower_feature_provider()
            if lower_feature is not None:
                payload["lower_retrieval_feature"] = np.asarray(lower_feature, dtype=np.float32)
        if self.lower_probe_provider is not None:
            probes = self.lower_probe_provider()
            if probes:
                payload["lower_subtask_probes"] = [
                    {
                        "candidate": str(item["candidate"]),
                        "feature": np.asarray(item["feature"], dtype=np.float32),
                    }
                    for item in probes
                ]
        with self._connection_lock:
            try:
                if self.connection is None:
                    self.connect()
                assert self.connection is not None
                roundtrip_started = time.perf_counter()
                self.connection.send(self.packer.pack(payload))
                raw = self.connection.recv(timeout=self.request_timeout)
                roundtrip_ms = (time.perf_counter() - roundtrip_started) * 1000.0
                response = msgpack_numpy.unpackb(raw)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.close()
                raise
        if "error" in response:
            raise RuntimeError(response["error"])
        feature = np.asarray(response["feature"], dtype=np.float32)
        if feature.ndim != 1 or not np.isfinite(feature).all():
            raise RuntimeError(f"Invalid remote upper feature: {feature.shape}")
        self._latest_feature = feature
        self._latest_feature_step = int(response["feature_step"])
        self._current_subtask = str(response["subtask"]) or self.default_subtask_prompt
        self._episode_reset_pending = False
        timing = dict(response.get("upper_timing", {}))
        timing["roundtrip_ms"] = float(roundtrip_ms)
        timing["step_idx"] = int(step_idx)
        self._timing_samples.append(timing)
        guidance = dict(response.get("upper_guidance", {}))
        if self.lower_probe_candidate_consumer is not None:
            self.lower_probe_candidate_consumer([str(value) for value in response.get("probe_candidates", [])])
        guidance["step_idx"] = int(step_idx)
        self._guidance_samples.append(guidance)
        if self.logger:
            self.logger.info(
                "batched VLM @t=%d subtask=%s keyframes=%s upper_guidance=%s",
                step_idx,
                self._current_subtask,
                response.get("keyframe_count"),
                guidance,
            )
        return self._current_subtask

    def timing_samples(self) -> list[dict[str, float | int]]:
        return [dict(item) for item in self._timing_samples]

    def guidance_samples(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._guidance_samples]

    def close(self) -> None:
        with self._connection_lock:
            if self.connection is not None:
                try:
                    self.connection.close()
                finally:
                    self.connection = None
