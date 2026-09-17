from __future__ import annotations

import argparse
import os
import select
import time

import websockets.sync.client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if args.pid is not None:
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError(f"Policy server process {args.pid} exited before readiness") from exc
        try:
            connection = websockets.sync.client.connect(
                f"ws://{args.host}:{args.port}",
                compression=None,
                max_size=None,
                open_timeout=2.0,
                close_timeout=1.0,
                ping_interval=None,
                ping_timeout=None,
            )
            connection.recv(timeout=5.0)
            connection.close()
            print("ready", flush=True)
            return
        except Exception as exc:
            last_error = exc
            select.select([], [], [], 0.25)
    raise TimeoutError(f"Policy server did not become ready: {last_error}")


if __name__ == "__main__":
    main()
