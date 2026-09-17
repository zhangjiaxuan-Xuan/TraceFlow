#!/usr/bin/env python3
"""Build the ModelScope payload from audited private source artifacts.

The output contains only TraceFlow heads, sliced memory, release configuration,
and cryptographic provenance. Official Pi0.5 and PrediMem weights are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import faiss
import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_bank(source: Path, destination: Path, *, negative: bool = False) -> None:
    prefix = "gpm_negative_memory" if negative else "gpm_memory"
    for suffix in ("_meta.pt", ".index", "_actions.npz"):
        copy_file(source / f"{prefix}{suffix}", destination / f"{prefix}{suffix}")


def copy_arena_variant(source: Path, destination: Path, variant: str) -> None:
    copy_file(source / "heads" / variant / "best.pt", destination / "heads" / variant / "best.pt")
    copy_file(source / "memory" / variant / "gpm_memory.index", destination / "memory" / variant / "gpm_memory.index")
    meta = source / "memory" / variant / "gpm_memory_meta_dense_frame_v3.pt"
    copy_file(meta, destination / "memory" / variant / "gpm_memory_meta.pt")


def file_spec(root: Path, path: Path, task_ids: list[int] | None = None) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    spec: dict[str, Any] = {"path": relative, "size": path.stat().st_size, "sha256": digest(path)}
    if path.name.endswith("_meta.pt"):
        import torch

        rows = torch.load(path, map_location="cpu", weights_only=False)
        spec.update(kind="metadata", records=len(rows))
        if task_ids is not None:
            spec["task_ids"] = task_ids
    elif path.suffix == ".index":
        spec.update(kind="faiss", vectors=int(faiss.read_index(str(path)).ntotal))
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as packed:
            spec.update(kind="actions", references=len(packed["ids"]))
    elif path.suffix in {".pt", ".pth"}:
        spec["kind"] = "head"
    else:
        spec["kind"] = "json"
    return spec


def group(root: Path, paths: list[Path], description: str, producer: str, task_ids: list[int] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "description": description,
        "producer_checkpoint": producer,
        "files": [file_spec(root, path, task_ids if path.name.endswith("_meta.pt") else None) for path in paths],
    }
    if task_ids is not None:
        result["task_ids"] = task_ids
    return result


def files_below(path: Path) -> list[Path]:
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--libero-head", type=Path, required=True)
    parser.add_argument("--libero-b6500", type=Path, required=True)
    parser.add_argument("--libero-positive", type=Path, required=True)
    parser.add_argument("--libero-negative", type=Path, required=True)
    parser.add_argument("--arena-extra8", type=Path, required=True)
    parser.add_argument("--arena-full26", type=Path, required=True)
    parser.add_argument("--revision", default="v1.0.0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to mix into non-empty output: {root}")
    root.mkdir(parents=True, exist_ok=True)

    copy_file(args.libero_head, root / "common/libero/gpm_task_head.pt")
    copy_bank(args.libero_b6500, root / "libero/b6500_positive")
    copy_bank(args.libero_positive, root / "libero/b50_positive")
    copy_bank(args.libero_negative, root / "libero/ncpi_negative", negative=True)

    script_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(script_root / "openpi/scripts/memory/slice_libero10_memory_banks.py"),
         "--positive-source", str(args.libero_positive), "--negative-source", str(args.libero_negative),
         "--output-root", str(root / "libero/libero10")], check=True
    )
    for label, logical_source in (("positive", "libero.b50_positive"), ("negative", "libero.ncpi_negative")):
        summary_path = root / "libero/libero10" / label / "slice_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["source"] = logical_source
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for variant in ("upper", "fusion"):
        copy_arena_variant(args.arena_extra8, root / "arena/extra8", variant)
    copy_file(args.arena_extra8 / "memory/shared/gpm_memory_actions.npz", root / "arena/extra8/memory/shared/gpm_memory_actions.npz")

    slicer = script_root / "openpi/scripts/robomemarena/slice_predimem_bank.py"
    full_meta = args.arena_full26 / "memory/fusion/gpm_memory_meta_dense_frame_v3.pt"
    full_index = args.arena_full26 / "memory/fusion/gpm_memory.index"
    full_actions = args.arena_full26 / "memory/shared/gpm_memory_actions.npz"
    scopes = {"counting": [6, 7, 8, 9, 10, 15, 16], "occlusion": [4, 5, 11, 12, 13, 14, 17, 20, 21, 23, 24]}
    for name, task_ids in scopes.items():
        destination = root / "arena" / name
        subprocess.run(
            [sys.executable, str(slicer), "--meta", str(full_meta), "--index", str(full_index),
             "--actions", str(full_actions), "--output", str(destination / "memory/fusion"),
             "--task-ids", ",".join(map(str, task_ids)), "--label", "any"], check=True
        )
        copy_file(args.arena_full26 / "heads/fusion/best.pt", destination / "heads/fusion/best.pt")
        actions = destination / "memory/fusion/gpm_memory_actions.npz"
        shared_actions = destination / "memory/shared/gpm_memory_actions.npz"
        shared_actions.parent.mkdir(parents=True, exist_ok=True)
        actions.replace(shared_actions)

    # CL groups use the immutable release banks without duplicating multi-GB files.
    cl = root / "cl"
    cl.mkdir(exist_ok=True)
    (cl / "layout.json").write_text(json.dumps({
        "libero10": "libero/libero10",
        "transferring": "arena/extra8",
        "note": "CL base groups refer to immutable release banks; round outputs remain branch-local."
    }, indent=2) + "\n", encoding="utf-8")

    groups = {
        "common.libero_head": group(root, [root / "common/libero/gpm_task_head.pt"], "LIBERO retrieval head", "Pi0.5-LIBERO"),
        "libero.b6500_positive": group(root, files_below(root / "libero/b6500_positive"), "B6500 positive-only bank", "Pi0.5-LIBERO"),
        "libero.b50_positive": group(root, files_below(root / "libero/b50_positive"), "B50-per-task positive bank", "Pi0.5-LIBERO"),
        "libero.ncpi_negative": group(root, files_below(root / "libero/ncpi_negative"), "N497 plus C-pi116 failure bank", "Pi0.5-LIBERO"),
        "libero.libero10_positive": group(root, files_below(root / "libero/libero10/positive"), "LIBERO-10 positive slice", "Pi0.5-LIBERO", list(range(10))),
        "libero.libero10_negative": group(root, files_below(root / "libero/libero10/negative"), "LIBERO-10 negative slice", "Pi0.5-LIBERO", list(range(10))),
        "arena.extra8": group(root, files_below(root / "arena/extra8"), "Extra8 joint memory and Upper/Fusion heads", "PrediMem-public-e645741c"),
        "arena.counting": group(root, files_below(root / "arena/counting"), "Counting-only Fusion memory", "PrediMem-public-e645741c", scopes["counting"]),
        "arena.occlusion": group(root, files_below(root / "arena/occlusion"), "Occlusion-only Fusion memory", "PrediMem-public-e645741c", scopes["occlusion"]),
    }
    groups["cl.libero10"] = {
        **groups["libero.libero10_positive"],
        "description": "TraceBankStack LIBERO-10 release bank",
        "files": [
            *groups["libero.libero10_positive"]["files"],
            *groups["libero.libero10_negative"]["files"],
        ],
    }
    groups["cl.transferring"] = {**groups["arena.extra8"], "description": "TraceBankStack Transferring immutable Extra8 base"}
    manifest = {"schema_version": 1, "repository": "JoeyXuan/traceflow-models", "revision": args.revision, "groups": groups}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
