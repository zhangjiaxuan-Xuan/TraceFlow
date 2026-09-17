from __future__ import annotations

from typing import Any
import time

import numpy as np


class ContextualPiClient:
    """Inject per-environment lifecycle metadata into the official Pi client."""

    def __init__(
        self, client: Any, environment_id: str, *, cross_environment_probe_batch: bool = False
    ) -> None:
        self.client = client
        self.environment_id = environment_id
        self.cross_environment_probe_batch = bool(cross_environment_probe_batch)
        self.task_id = 0
        self.episode_idx = 0
        self.episode_seed = 0
        self.inference_call = 0
        self.episode_reset_pending = True
        self._timing_samples: list[dict[str, Any]] = []
        self._latest_lower_retrieval_feature: np.ndarray | None = None
        self._pending_probe_candidates: list[str] = []
        self._latest_probe_results: list[dict[str, Any]] = []

    def set_episode_context(self, *, task_id: int, episode_idx: int, seed: int) -> None:
        self.task_id = task_id
        self.episode_idx = episode_idx
        self.episode_seed = seed
        self.inference_call = 0
        self.episode_reset_pending = True
        self._timing_samples = []
        self._latest_lower_retrieval_feature = None
        self._pending_probe_candidates = []
        self._latest_probe_results = []

    def set_lower_probe_candidates(self, candidates: list[str]) -> None:
        self._pending_probe_candidates = [str(value).strip() for value in candidates if str(value).strip()]

    def latest_lower_probe_results(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._latest_probe_results]

    def _probe_subtasks(self, element: dict[str, Any]) -> None:
        candidates = self._pending_probe_candidates
        self._pending_probe_candidates = []
        if not candidates:
            return
        infer_batch = getattr(self.client, "infer_batch", None)
        if not callable(infer_batch):
            raise RuntimeError("Lower policy client does not support counterfactual probe batches")
        requests = []
        for candidate in candidates:
            request = dict(element)
            request["prompt"] = candidate
            request["_lower_probe_only"] = True
            request["episode_reset"] = False
            request["trace_context"] = {
                "environment_id": self.environment_id,
                "task_suite": "robomemarena",
                "task_id": self.task_id,
                "episode_idx": self.episode_idx,
                "episode_seed": self.episode_seed,
                "inference_call": self.inference_call,
                "policy_call_idx": self.inference_call,
            }
            requests.append(request)
        if self.cross_environment_probe_batch:
            # Each synchronous environment contributes one candidate at a time.
            # The server's dedicated probe queue combines matching candidate
            # positions across environments into large GPU batches.
            outputs = [self.client.infer(request) for request in requests]
        else:
            outputs = infer_batch(requests)
        self._latest_probe_results = []
        for candidate, output in zip(candidates, outputs, strict=True):
            feature = np.asarray(output["lower_retrieval_feature"], dtype=np.float32)
            if feature.ndim != 1 or not np.isfinite(feature).all():
                raise RuntimeError(f"Invalid counterfactual Lower feature: {feature.shape}")
            self._latest_probe_results.append(
                {
                    "candidate": candidate,
                    "feature": feature.copy(),
                    "timing": dict(output.get("policy_timing", {})),
                }
            )

    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        self._probe_subtasks(element)
        request = dict(element)
        request["episode_reset"] = self.episode_reset_pending
        request["trace_context"] = {
            "environment_id": self.environment_id,
            "task_suite": "robomemarena",
            "task_id": self.task_id,
            "episode_idx": self.episode_idx,
            "episode_seed": self.episode_seed,
            "inference_call": self.inference_call,
            "policy_call_idx": self.inference_call,
        }
        roundtrip_started = time.perf_counter()
        result = self.client.infer(request)
        lower_feature = result.pop("lower_retrieval_feature", None)
        if lower_feature is not None:
            value = np.asarray(lower_feature, dtype=np.float32)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid lower retrieval feature: {value.shape}")
            self._latest_lower_retrieval_feature = value.copy()
        roundtrip_ms = (time.perf_counter() - roundtrip_started) * 1000.0
        policy_timing = dict(result.get("policy_timing", {}))
        server_timing = dict(result.get("server_timing", {}))
        self._timing_samples.append(
            {
                "inference_call": int(self.inference_call),
                "roundtrip_ms": float(roundtrip_ms),
                "policy_batch_ms": float(policy_timing.get("infer_ms", float("nan"))),
                "batch_size": int(policy_timing.get("batch_size", 1)),
                "server_infer_ms": float(server_timing.get("infer_ms", float("nan"))),
                "components": dict(policy_timing.get("components", {})),
            }
        )
        self.episode_reset_pending = False
        self.inference_call += 1
        return result

    def latest_lower_retrieval_feature(self) -> np.ndarray | None:
        value = self._latest_lower_retrieval_feature
        return None if value is None else value.copy()

    def timing_samples(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._timing_samples]

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
