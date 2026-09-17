from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from traceflow import assets
from traceflow.assets import AssetError, verify_snapshot


def manifest(root: Path, digest: str) -> None:
    value = {
        "schema_version": 1,
        "repository": "JoeyXuan/traceflow-models",
        "revision": "test",
        "groups": {"fixture": {"description": "fixture", "producer_checkpoint": "fixture", "files": [
            {"path": "common/value.json", "size": 3, "sha256": digest, "kind": "json"}
        ]}},
    }
    (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


def test_verifies_hash_and_size(tmp_path: Path) -> None:
    path = tmp_path / "common/value.json"
    path.parent.mkdir()
    path.write_bytes(b"abc")
    manifest(tmp_path, hashlib.sha256(b"abc").hexdigest())
    assert "fixture" in verify_snapshot(tmp_path, ["fixture"])


def test_rejects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "common/value.json"
    path.parent.mkdir()
    path.write_bytes(b"abc")
    manifest(tmp_path, "0" * 64)
    with pytest.raises(AssetError, match="SHA-256 mismatch"):
        verify_snapshot(tmp_path, ["fixture"])


def test_rejects_unknown_group(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "repository": "JoeyXuan/traceflow-models", "groups": {}}))
    with pytest.raises(AssetError, match="unknown asset"):
        verify_snapshot(tmp_path, ["missing"])


def test_resolver_uses_override_without_modelscope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACEFLOW_ASSET_ROOT", str(tmp_path))
    monkeypatch.setattr(assets, "_verify_release_lock", lambda root, groups: None)
    monkeypatch.setattr(assets, "verify_snapshot", lambda root, groups, semantic=True: {})
    assert assets.resolve_snapshot(["fixture"]) == tmp_path.resolve()


def test_resolver_trusts_snapshot_download_return_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    returned = tmp_path / "sdk-owned-layout" / "snapshot"
    returned.mkdir(parents=True)
    calls: list[dict[str, str]] = []

    def snapshot_download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(returned)

    monkeypatch.delenv("TRACEFLOW_ASSET_ROOT", raising=False)
    monkeypatch.setitem(sys.modules, "modelscope", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(assets, "_verify_release_lock", lambda root, groups: None)
    monkeypatch.setattr(assets, "verify_snapshot", lambda root, groups, semantic=True: {})

    assert assets.resolve_snapshot(["fixture"], revision="release-v1") == returned.resolve()
    assert calls == [
        {
            "model_id": assets.REPO_ID,
            "revision": "release-v1",
            "endpoint": assets.DEFAULT_ENDPOINT,
        }
    ]
