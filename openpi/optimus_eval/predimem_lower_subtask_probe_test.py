from __future__ import annotations

import numpy as np

from optimus_eval.predimem_contextual_pi_client import ContextualPiClient


class _Client:
    def __init__(self) -> None:
        self.probes = []
        self.actions = []

    def infer_batch(self, requests):
        self.probes.extend(requests)
        return [
            {
                "lower_retrieval_feature": np.full(6, index + 1, dtype=np.float32),
                "policy_timing": {"probe_ms": 3.0, "batch_size": len(requests)},
            }
            for index in range(len(requests))
        ]

    def infer(self, request):
        if request.get("_lower_probe_only", False):
            self.probes.append(request)
            return {
                "lower_retrieval_feature": np.full(6, len(self.probes), dtype=np.float32),
                "policy_timing": {"probe_ms": 3.0, "batch_size": 1},
            }
        self.actions.append(request)
        return {"actions": np.zeros((10, 7), dtype=np.float32)}


def test_counterfactual_probe_batch_does_not_replace_real_prompt() -> None:
    base = _Client()
    client = ContextualPiClient(base, "env-0")
    client.set_episode_context(task_id=18, episode_idx=2, seed=52)
    client.set_lower_probe_candidates(["pick apple", "place apple"])
    observation = {
        "prompt": "native subtask",
        "state": np.zeros(8, dtype=np.float32),
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    client.infer(observation)

    assert [request["prompt"] for request in base.probes] == ["pick apple", "place apple"]
    assert all(request["_lower_probe_only"] is True for request in base.probes)
    assert base.actions[0]["prompt"] == "native subtask"
    results = client.latest_lower_probe_results()
    assert [item["candidate"] for item in results] == ["pick apple", "place apple"]
    assert np.all(results[0]["feature"] == 1)


def test_probe_candidates_are_consumed_once() -> None:
    base = _Client()
    client = ContextualPiClient(base, "env-0")
    client.set_lower_probe_candidates(["pick apple"])
    observation = {"prompt": "native", "state": np.zeros(8, dtype=np.float32)}
    client.infer(observation)
    client.infer(observation)
    assert len(base.probes) == 1
    assert len(base.actions) == 2


def test_cross_environment_mode_streams_candidates_for_server_side_batching() -> None:
    base = _Client()
    client = ContextualPiClient(base, "env-7", cross_environment_probe_batch=True)
    client.set_episode_context(task_id=13, episode_idx=1, seed=51)
    client.set_lower_probe_candidates(["open drawer", "pick cookies", "place cookies"])
    client.infer({"prompt": "native", "state": np.zeros(8, dtype=np.float32)})

    assert [request["prompt"] for request in base.probes] == [
        "open drawer",
        "pick cookies",
        "place cookies",
    ]
    assert all(request["trace_context"]["environment_id"] == "env-7" for request in base.probes)
    assert base.actions[0]["prompt"] == "native"
    assert [item["candidate"] for item in client.latest_lower_probe_results()] == [
        "open drawer",
        "pick cookies",
        "place cookies",
    ]
