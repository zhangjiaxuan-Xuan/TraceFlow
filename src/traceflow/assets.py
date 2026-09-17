"""Resolve and verify TraceFlow-owned artifacts from ModelScope."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

REPO_ID = "JoeyXuan/traceflow-models"
DEFAULT_ENDPOINT = "https://modelscope.ai"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = "manifest.lock.json"


class AssetError(RuntimeError):
    """An asset snapshot is missing, corrupt, or outside its release contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AssetError(f"snapshot has no {MANIFEST_NAME}: {root}") from error
    if value.get("schema_version") != 1:
        raise AssetError(f"unsupported asset manifest schema: {value.get('schema_version')!r}")
    if value.get("repository") != REPO_ID:
        raise AssetError(f"unexpected ModelScope repository: {value.get('repository')!r}")
    return value


def _metadata_contract(path: Path, spec: dict[str, Any]) -> None:
    expected = spec.get("records")
    expected_tasks = {int(value) for value in spec.get("task_ids", [])}
    if expected is None and not expected_tasks:
        return
    try:
        import torch
    except ImportError as error:
        raise AssetError("PyTorch is required for semantic metadata verification") from error
    try:
        rows = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        rows = torch.load(path, map_location="cpu")
    if not isinstance(rows, list):
        raise AssetError(f"metadata must contain a list: {path}")
    if expected is not None and len(rows) != int(expected):
        raise AssetError(f"metadata record mismatch for {path}: {len(rows)} != {expected}")
    if expected_tasks:
        actual = {int(row["task_id"]) for row in rows}
        if actual != expected_tasks:
            raise AssetError(f"metadata task scope mismatch for {path}: {sorted(actual)} != {sorted(expected_tasks)}")
    for index, row in enumerate(rows):
        if "action_id" not in row:
            raise AssetError(f"metadata row {index} has no action_id: {path}")


def _faiss_contract(path: Path, spec: dict[str, Any]) -> None:
    if "vectors" not in spec:
        return
    try:
        import faiss
    except ImportError as error:
        raise AssetError("FAISS is required for index verification") from error
    count = int(faiss.read_index(str(path)).ntotal)
    if count != int(spec["vectors"]):
        raise AssetError(f"FAISS vector mismatch for {path}: {count} != {spec['vectors']}")


def _actions_contract(path: Path, spec: dict[str, Any]) -> None:
    if "references" not in spec:
        return
    try:
        import numpy as np
    except ImportError as error:
        raise AssetError("NumPy is required for action-store verification") from error
    with np.load(path, allow_pickle=False) as packed:
        missing = {"actions", "offsets", "ids"}.difference(packed.files)
        if missing:
            raise AssetError(f"action store lacks {sorted(missing)}: {path}")
        references = len(packed["ids"])
        offsets = packed["offsets"]
        actions = packed["actions"]
        if references != int(spec["references"]):
            raise AssetError(f"action reference mismatch for {path}: {references} != {spec['references']}")
        if len(offsets) != references + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(actions):
            raise AssetError(f"invalid action offsets in {path}")


def verify_snapshot(root: Path, groups: Iterable[str] | None = None, *, semantic: bool = True) -> dict[str, Any]:
    """Verify selected groups and return their normalized manifest entries."""
    root = root.expanduser().resolve()
    manifest = _load_manifest(root)
    available = manifest.get("groups", {})
    selected = list(groups or available)
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise AssetError(f"unknown asset group(s): {', '.join(unknown)}")
    verified: dict[str, Any] = {}
    for group in selected:
        group_spec = available[group]
        files = group_spec.get("files", [])
        if not files:
            raise AssetError(f"asset group has no files: {group}")
        metadata_paths: list[Path] = []
        action_paths: list[Path] = []
        faiss_counts: list[int] = []
        metadata_counts: list[int] = []
        for spec in files:
            relative = Path(spec["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise AssetError(f"unsafe asset path in {group}: {relative}")
            path = root / relative
            if not path.is_file():
                raise AssetError(f"missing asset: {relative}")
            size = path.stat().st_size
            if size != int(spec["size"]):
                raise AssetError(f"size mismatch for {relative}: {size} != {spec['size']}")
            digest = _sha256(path)
            if digest != spec["sha256"]:
                raise AssetError(f"SHA-256 mismatch for {relative}: {digest} != {spec['sha256']}")
            if semantic:
                kind = spec.get("kind")
                if kind == "metadata":
                    _metadata_contract(path, spec)
                    metadata_paths.append(path)
                    metadata_counts.append(int(spec.get("records", 0)))
                elif kind == "faiss":
                    _faiss_contract(path, spec)
                    faiss_counts.append(int(spec.get("vectors", 0)))
                elif kind == "actions":
                    _actions_contract(path, spec)
                    action_paths.append(path)
        if semantic and metadata_paths and action_paths:
            import numpy as np
            import torch

            action_ids: set[str] = set()
            for path in action_paths:
                with np.load(path, allow_pickle=False) as packed:
                    action_ids.update(str(value) for value in packed["ids"].tolist())
            referenced: set[str] = set()
            for path in metadata_paths:
                rows = torch.load(path, map_location="cpu", weights_only=False)
                referenced.update(str(row["action_id"]) for row in rows)
            missing = referenced.difference(action_ids)
            if missing:
                raise AssetError(f"{group} metadata references {len(missing)} absent action IDs")
        if semantic and metadata_counts and faiss_counts:
            if sorted(metadata_counts) != sorted(faiss_counts):
                raise AssetError(f"{group} metadata/FAISS counts differ: {metadata_counts} != {faiss_counts}")
        verified[group] = group_spec
    return verified


def _verify_release_lock(root: Path, groups: Iterable[str]) -> None:
    remote = _load_manifest(root)
    lock_path = Path(__file__).with_name(LOCK_NAME)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if remote.get("revision") != lock.get("revision"):
        raise AssetError(f"asset revision is not release-locked: {remote.get('revision')!r} != {lock.get('revision')!r}")
    for group in groups:
        if remote["groups"].get(group) != lock["groups"].get(group):
            raise AssetError(f"asset group differs from bundled release lock: {group}")


def resolve_snapshot(groups: Iterable[str], *, revision: str | None = None, semantic: bool = True) -> Path:
    """Resolve an override or ModelScope snapshot and verify it before use."""
    override = os.environ.get("TRACEFLOW_ASSET_ROOT")
    if override:
        root = Path(override)
    else:
        try:
            from modelscope import snapshot_download
        except ImportError as error:
            raise AssetError("install TraceFlow's 'assets' extra to download ModelScope artifacts") from error
        kwargs: dict[str, Any] = {
            "model_id": REPO_ID,
            "endpoint": os.environ.get("MODELSCOPE_ENDPOINT", DEFAULT_ENDPOINT),
        }
        if revision:
            kwargs["revision"] = revision
        # ModelScope owns cache selection. The returned snapshot path is the only
        # path we trust; MODELSCOPE_CACHE and the user's default home stay intact.
        root = Path(snapshot_download(**kwargs))
    root = root.expanduser().resolve()
    selected = list(groups)
    _verify_release_lock(root, selected)
    verify_snapshot(root, selected, semantic=semantic)
    return root
