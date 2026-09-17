import asyncio
from collections import deque
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        batch_size: int = 1,
        batch_wait_ms: float = 5.0,
        batch_group_size: int = 0,
        separate_lower_probe_batches: bool = False,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._batch_size = max(1, int(batch_size))
        self._batch_wait_s = max(0.0, float(batch_wait_ms) / 1000.0)
        self._batch_group_size = max(0, int(batch_group_size))
        self._separate_lower_probe_batches = bool(separate_lower_probe_batches)
        if self._batch_group_size and self._batch_group_size != self._batch_size:
            raise ValueError("batch_group_size must equal batch_size when grouped batching is enabled")
        if self._batch_group_size and self._separate_lower_probe_batches:
            raise ValueError("grouped batching and separate Lower probe batching are mutually exclusive")
        self._batch_queue = None
        self._probe_batch_queue = None
        self._batch_index = 0
        self._connection_index = 0
        self._batch_condition: asyncio.Condition | None = None
        self._group_queues: dict[int, deque] = {}
        self._group_active: dict[int, int] = {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._batch_queue = asyncio.Queue()
        self._probe_batch_queue = asyncio.Queue() if self._separate_lower_probe_batches else None
        self._batch_condition = asyncio.Condition() if self._batch_group_size else None
        batch_tasks = []
        if self._batch_group_size:
            batch_tasks.append(asyncio.create_task(self._grouped_batch_loop()))
        elif self._batch_size > 1:
            batch_tasks.append(asyncio.create_task(self._batch_loop(self._batch_queue, lane="action")))
            if self._probe_batch_queue is not None:
                batch_tasks.append(
                    asyncio.create_task(self._batch_loop(self._probe_batch_queue, lane="lower-probe"))
                )
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            process_request=_health_check,
        ) as server:
            try:
                await server.serve_forever()
            finally:
                for batch_task in batch_tasks:
                    batch_task.cancel()

    def _register_connection(self) -> int | None:
        if not self._batch_group_size:
            return None
        connection_id = self._connection_index
        self._connection_index += 1
        group_id = connection_id // self._batch_group_size
        self._group_queues.setdefault(group_id, deque())
        self._group_active[group_id] = self._group_active.get(group_id, 0) + 1
        return group_id

    async def _unregister_connection(self, group_id: int | None) -> None:
        if group_id is None or self._batch_condition is None:
            return
        async with self._batch_condition:
            self._group_active[group_id] = max(0, self._group_active.get(group_id, 1) - 1)
            self._batch_condition.notify_all()

    def _full_group(self, start_group: int) -> int | None:
        groups = sorted(self._group_queues)
        if not groups:
            return None
        for offset in range(len(groups)):
            group_id = groups[(start_group + offset) % len(groups)]
            if len(self._group_queues[group_id]) >= self._batch_size:
                return group_id
        return None

    def _nonempty_group(self, start_group: int) -> int | None:
        groups = sorted(self._group_queues)
        if not groups:
            return None
        candidates = [
            groups[(start_group + offset) % len(groups)]
            for offset in range(len(groups))
            if self._group_queues[groups[(start_group + offset) % len(groups)]]
        ]
        return max(candidates, key=lambda group_id: len(self._group_queues[group_id])) if candidates else None

    async def _grouped_batch_loop(self) -> None:
        """Run fixed-size environment lanes before allowing partial batches.

        LIBERO clients are synchronous: after receiving a chunk, each client
        simulates locally before requesting again. A global queue therefore
        starts small batches whenever the clients are slightly out of phase.
        Grouping eight connections and waiting for that lane preserves the
        intended 8-env wave while retaining a bounded failure fallback.
        """
        assert self._batch_condition is not None
        next_group = 0
        while True:
            async with self._batch_condition:
                while self._full_group(next_group) is None and self._nonempty_group(next_group) is None:
                    await self._batch_condition.wait()
                group_id = self._full_group(next_group)
                if group_id is None:
                    group_id = self._nonempty_group(next_group)
                    deadline = asyncio.get_running_loop().time() + self._batch_wait_s
                    while group_id is not None and len(self._group_queues[group_id]) < self._batch_size:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            break
                        try:
                            await asyncio.wait_for(self._batch_condition.wait(), timeout=remaining)
                        except TimeoutError:
                            break
                        full_group = self._full_group(next_group)
                        if full_group is not None:
                            group_id = full_group
                            break
                        group_id = self._nonempty_group(next_group)
                if group_id is None:
                    continue
                queue = self._group_queues[group_id]
                items = [queue.popleft() for _ in range(min(self._batch_size, len(queue)))]
                next_group = (group_id + 1) % max(1, len(self._group_queues))
            await self._run_batch(items, group_id=group_id)

    async def _run_batch(self, items, *, group_id: int | None = None, lane: str = "action") -> None:
        try:
            started = time.monotonic()
            outputs = self._policy.infer_batch([item[0] for item in items])
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if len(outputs) != len(items):
                raise RuntimeError("Batched policy returned the wrong number of outputs")
            self._batch_index += 1
            if self._batch_index <= 10 or self._batch_index % 100 == 0:
                logger.info(
                    "Inference batch=%d lane=%s size=%d/%d group=%s infer_ms=%.1f queued=%d",
                    self._batch_index,
                    lane,
                    len(items),
                    self._batch_size,
                    "none" if group_id is None else group_id,
                    elapsed_ms,
                    self._batch_queue.qsize() if self._batch_queue is not None else 0,
                )
            for output, (_, item_future, *_) in zip(outputs, items):
                if not item_future.cancelled():
                    item_future.set_result(output)
        except Exception as exc:
            for _, item_future, *__ in items:
                if not item_future.cancelled():
                    item_future.set_exception(exc)

    async def _batch_loop(self, batch_queue=None, *, lane: str = "action"):
        batch_queue = self._batch_queue if batch_queue is None else batch_queue
        assert batch_queue is not None
        while True:
            obs, future = await batch_queue.get()
            items = [(obs, future)]
            while len(items) < self._batch_size:
                try:
                    items.append(batch_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if len(items) < self._batch_size and self._batch_wait_s:
                deadline = asyncio.get_running_loop().time() + self._batch_wait_s
                while len(items) < self._batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        items.append(
                            await asyncio.wait_for(batch_queue.get(), timeout=remaining)
                        )
                    except TimeoutError:
                        break
            await self._run_batch(items, lane=lane)

    async def _infer(self, obs, group_id: int | None = None):
        if self._batch_size <= 1 or isinstance(obs, list):
            return self._policy.infer_batch(obs) if isinstance(obs, list) else self._policy.infer(obs)
        if self._batch_group_size:
            assert self._batch_condition is not None and group_id is not None
            future = asyncio.get_running_loop().create_future()
            async with self._batch_condition:
                self._group_queues[group_id].append((obs, future, group_id))
                self._batch_condition.notify_all()
            return await future
        future = asyncio.get_running_loop().create_future()
        queue = self._batch_queue
        if (
            self._probe_batch_queue is not None
            and isinstance(obs, dict)
            and bool(obs.get("_lower_probe_only", False))
        ):
            queue = self._probe_batch_queue
        assert queue is not None
        await queue.put((obs, future))
        return await future

    async def _handler(self, websocket: _server.ServerConnection):
        group_id = self._register_connection()
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        try:
            await websocket.send(packer.pack(self._metadata))

            prev_total_time = None
            while True:
                try:
                    start_time = time.monotonic()
                    obs = msgpack_numpy.unpackb(await websocket.recv())

                    infer_time = time.monotonic()
                    action = await self._infer(obs, group_id)
                    infer_time = time.monotonic() - infer_time

                    timing = {"infer_ms": infer_time * 1000}
                    if isinstance(action, list):
                        for item in action:
                            item["server_timing"] = dict(timing)
                    else:
                        action["server_timing"] = timing
                    if prev_total_time is not None and not isinstance(action, list):
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            await self._unregister_connection(group_id)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
