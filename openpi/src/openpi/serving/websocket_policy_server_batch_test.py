from __future__ import annotations

import asyncio
from collections import deque

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class _BatchPolicy:
    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def infer_batch(self, observations: list[int]) -> list[int]:
        self.batches.append(list(observations))
        return [value + 1 for value in observations]


def test_full_batch_does_not_wait_for_coalescing_deadline() -> None:
    async def run() -> None:
        policy = _BatchPolicy()
        server = WebsocketPolicyServer(policy, batch_size=8, batch_wait_ms=1000)
        server._batch_queue = asyncio.Queue()
        futures = []
        for value in range(8):
            future = asyncio.get_running_loop().create_future()
            futures.append(future)
            await server._batch_queue.put((value, future))

        task = asyncio.create_task(server._batch_loop())
        try:
            outputs = await asyncio.wait_for(asyncio.gather(*futures), timeout=0.2)
        finally:
            task.cancel()
        assert outputs == list(range(1, 9))
        assert policy.batches == [list(range(8))]

    asyncio.run(run())


def test_partial_batch_runs_at_deadline() -> None:
    async def run() -> None:
        policy = _BatchPolicy()
        server = WebsocketPolicyServer(policy, batch_size=8, batch_wait_ms=5)
        server._batch_queue = asyncio.Queue()
        futures = []
        for value in range(3):
            future = asyncio.get_running_loop().create_future()
            futures.append(future)
            await server._batch_queue.put((value, future))

        task = asyncio.create_task(server._batch_loop())
        try:
            outputs = await asyncio.wait_for(asyncio.gather(*futures), timeout=0.2)
        finally:
            task.cancel()
        assert outputs == [1, 2, 3]
        assert policy.batches == [[0, 1, 2]]

    asyncio.run(run())


def test_32_ready_requests_run_as_two_full_batch16_groups() -> None:
    async def run() -> None:
        policy = _BatchPolicy()
        server = WebsocketPolicyServer(policy, batch_size=16, batch_wait_ms=1000)
        server._batch_queue = asyncio.Queue()
        futures = []
        for value in range(32):
            future = asyncio.get_running_loop().create_future()
            futures.append(future)
            await server._batch_queue.put((value, future))

        task = asyncio.create_task(server._batch_loop())
        try:
            outputs = await asyncio.wait_for(asyncio.gather(*futures), timeout=1.0)
        finally:
            task.cancel()
        assert outputs == list(range(1, 33))
        assert [len(batch) for batch in policy.batches] == [16, 16]

    asyncio.run(run())


def test_grouped_batching_runs_two_fixed_eight_connection_lanes() -> None:
    async def run() -> None:
        policy = _BatchPolicy()
        server = WebsocketPolicyServer(policy, batch_size=8, batch_wait_ms=1000, batch_group_size=8)
        server._batch_queue = asyncio.Queue()
        server._batch_condition = asyncio.Condition()
        server._group_queues = {0: deque(), 1: deque()}
        server._group_active = {0: 8, 1: 8}
        futures = []
        for group_id in range(2):
            for value in range(group_id * 8, group_id * 8 + 8):
                future = asyncio.get_running_loop().create_future()
                futures.append(future)
                server._group_queues[group_id].append((value, future, group_id))

        task = asyncio.create_task(server._grouped_batch_loop())
        try:
            outputs = await asyncio.wait_for(asyncio.gather(*futures), timeout=1.0)
        finally:
            task.cancel()
        assert outputs == list(range(1, 17))
        assert [len(batch) for batch in policy.batches] == [8, 8]
        assert policy.batches[0] == list(range(8))
        assert policy.batches[1] == list(range(8, 16))

    asyncio.run(run())


def test_lower_probe_and_action_requests_use_separate_batches() -> None:
    class TaggedPolicy:
        def __init__(self) -> None:
            self.batches = []

        def infer_batch(self, observations):
            self.batches.append(list(observations))
            return [{"value": item["value"] + 1} for item in observations]

    async def run() -> None:
        policy = TaggedPolicy()
        server = WebsocketPolicyServer(
            policy,
            batch_size=2,
            batch_wait_ms=1000,
            separate_lower_probe_batches=True,
        )
        server._batch_queue = asyncio.Queue()
        server._probe_batch_queue = asyncio.Queue()
        loops = [
            asyncio.create_task(server._batch_loop(server._batch_queue, lane="action")),
            asyncio.create_task(server._batch_loop(server._probe_batch_queue, lane="lower-probe")),
        ]
        requests = [
            {"value": 0},
            {"value": 10, "_lower_probe_only": True},
            {"value": 1},
            {"value": 11, "_lower_probe_only": True},
        ]
        try:
            outputs = await asyncio.wait_for(
                asyncio.gather(*(server._infer(item) for item in requests)), timeout=1.0
            )
        finally:
            for loop in loops:
                loop.cancel()
        assert [item["value"] for item in outputs] == [1, 11, 2, 12]
        assert len(policy.batches) == 2
        assert all(
            all(bool(item.get("_lower_probe_only", False)) == probe for item in batch)
            for batch in policy.batches
            for probe in [bool(batch[0].get("_lower_probe_only", False))]
        )

    asyncio.run(run())


def test_64_environment_probe_wave_runs_as_two_batch32_calls() -> None:
    class TaggedPolicy:
        def __init__(self) -> None:
            self.batches = []

        def infer_batch(self, observations):
            self.batches.append(list(observations))
            return list(observations)

    async def run() -> None:
        policy = TaggedPolicy()
        server = WebsocketPolicyServer(
            policy,
            batch_size=32,
            batch_wait_ms=1000,
            separate_lower_probe_batches=True,
        )
        server._batch_queue = asyncio.Queue()
        server._probe_batch_queue = asyncio.Queue()
        futures = []
        for probe in (False, True):
            queue = server._probe_batch_queue if probe else server._batch_queue
            for slot in range(64):
                future = asyncio.get_running_loop().create_future()
                futures.append(future)
                await queue.put(({"slot": slot, "_lower_probe_only": probe}, future))
        loops = [
            asyncio.create_task(server._batch_loop(server._batch_queue, lane="action")),
            asyncio.create_task(server._batch_loop(server._probe_batch_queue, lane="lower-probe")),
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*futures), timeout=1.0)
        finally:
            for loop in loops:
                loop.cancel()
        assert sorted(len(batch) for batch in policy.batches) == [32, 32, 32, 32]
        assert all(
            len({bool(item["_lower_probe_only"]) for item in batch}) == 1
            for batch in policy.batches
        )

    asyncio.run(run())
