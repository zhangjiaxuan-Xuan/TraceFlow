#!/usr/bin/env python3
"""Publish an already verified TraceFlow asset payload to ModelScope."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from traceflow.assets import REPO_ID, verify_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--message", default="Publish TraceFlow v1 reproduction assets")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--endpoint", default=os.environ.get("MODELSCOPE_ENDPOINT"))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    payload = args.payload.expanduser().resolve()
    if not args.skip_verify:
        print(f"Verifying upload payload: {payload}", flush=True)
        verify_snapshot(payload)
        print("Payload verification passed.", flush=True)
    token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if not token:
        if not os.isatty(0):
            raise RuntimeError(
                "run this command in an interactive terminal to enter a ModelScope token securely, "
                "or set MODELSCOPE_API_TOKEN"
            )
        token = getpass.getpass("ModelScope access token (input hidden): ").strip()
        if not token:
            raise RuntimeError("a ModelScope access token is required")
    from modelscope.hub.api import HubApi
    from modelscope_hub.errors import AlreadyExistsError

    api = HubApi(endpoint=args.endpoint)
    api.login(token)
    try:
        api.create_model(model_id=REPO_ID, visibility=5, license="Apache License 2.0")
    except AlreadyExistsError:
        pass
    except Exception as error:
        if "exist" not in str(error).lower():
            raise
    if hasattr(api, "upload_folder"):
        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=str(payload),
            commit_message=args.message,
            repo_type="model",
            revision=args.revision,
            max_workers=args.max_workers,
            use_cache=True,
        )
    else:
        api.push_model(model_id=REPO_ID, model_dir=str(payload), visibility=5, commit_message=args.message)


if __name__ == "__main__":
    main()
