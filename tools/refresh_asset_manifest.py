#!/usr/bin/env python3
"""Refresh byte-level fields after editing metadata-only release files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_release_assets import digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for group in manifest["groups"].values():
        for spec in group["files"]:
            artifact = root / spec["path"]
            spec["size"] = artifact.stat().st_size
            spec["sha256"] = digest(artifact)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
