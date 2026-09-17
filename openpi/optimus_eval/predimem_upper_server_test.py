from __future__ import annotations

import queue
import sys
import threading
import time
import types

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoProcessor = object
transformers_stub.LogitsProcessor = object
transformers_stub.LogitsProcessorList = list
transformers_stub.Qwen3VLForConditionalGeneration = object
sys.modules["transformers"] = transformers_stub
from optimus_eval.predimem_upper_server import BatchedUpperPlanner


def test_same_environment_requests_are_serialized() -> None:
    planner = BatchedUpperPlanner.__new__(BatchedUpperPlanner)
    planner.state_lock = threading.Lock()
    planner.environment_locks = {}
    planner.requests = queue.Queue()
    active: set[str] = set()
    overlap = []

    def worker() -> None:
        for _ in range(2):
            request = planner.requests.get(timeout=2)
            environment_id = str(request.payload["environment_id"])
            if environment_id in active:
                overlap.append(environment_id)
            active.add(environment_id)
            time.sleep(0.02)
            request.response = {"environment_id": environment_id}
            active.remove(environment_id)
            request.done.set()

    batcher = threading.Thread(target=worker)
    batcher.start()
    outputs = []
    clients = [
        threading.Thread(target=lambda: outputs.append(planner.submit({"environment_id": "env0"})))
        for _ in range(2)
    ]
    for client in clients:
        client.start()
    for client in clients:
        client.join(timeout=2)
    batcher.join(timeout=2)
    assert not overlap
    assert outputs == [{"environment_id": "env0"}, {"environment_id": "env0"}]
