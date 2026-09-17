"""Verify immutable assets for the frozen updated-RoboMemArena base protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bddl_digest(root: Path) -> tuple[str, int]:
    files = sorted(root.glob("*.bddl"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Frozen protocol verification failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--arena-root", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    benchmark = lock["benchmark"]
    for relative, expected in benchmark["files"].items():
        path = args.arena_root / relative
        _require(path.is_file(), f"missing benchmark file: {path}")
        _require(_sha256(path) == expected, f"benchmark hash changed: {path}")

    bddl_hash, bddl_count = _bddl_digest(args.arena_root / "bddl")
    _require(bddl_count == benchmark["bddl_count"], f"expected {benchmark['bddl_count']} BDDL files, got {bddl_count}")
    _require(bddl_hash == benchmark["bddl_content_sha256"], "BDDL content changed")

    model = lock["model"]
    vla = Path(model["vla_checkpoint"])
    vlm = Path(model["vlm_checkpoint"])
    vla_weights = vla / "model.safetensors"
    vlm_weights = vlm / "model.safetensors"
    _require(vla_weights.stat().st_size == model["vla_model_size"], "VLA checkpoint size changed")
    _require(vlm_weights.stat().st_size == model["vlm_model_size"], "VLM checkpoint size changed")
    _require(
        _sha256(vla / "assets/robomemarena/all26_pi05_reactive/norm_stats.json")
        == model["vla_norm_stats_sha256"],
        "VLA normalization statistics changed",
    )
    _require(_sha256(vlm / "config.json") == model["vlm_config_sha256"], "VLM config changed")
    print(f"Frozen protocol verified: {lock['protocol_id']}")


if __name__ == "__main__":
    main()
