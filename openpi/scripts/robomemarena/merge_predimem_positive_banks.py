#!/usr/bin/env python3
"""Append a compatible online bank to an inherited PrediMem bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import faiss
import numpy as np
import torch


def _load_actions(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as packed:
        actions = np.asarray(packed["actions"], dtype=np.float32)
        offsets = np.asarray(packed["offsets"], dtype=np.int64)
        ids = [str(value) for value in np.asarray(packed["ids"]).tolist()]
    if offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(actions):
        raise ValueError(f"Invalid packed action store: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate action IDs inside {path}")
    return actions, offsets, ids


def _validate_metadata(metadata: list[dict], ids: set[str], path: Path) -> None:
    if not metadata:
        raise ValueError(f"Empty metadata: {path}")
    missing = sorted({str(row["action_id"]) for row in metadata} - ids)
    if missing:
        raise ValueError(f"Metadata references missing actions in {path}: {missing[:3]}")
    if not all("anchor_frame" in row and "anchor_progress" in row for row in metadata):
        raise ValueError(f"Hereditary banks require dense-frame metadata: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-meta", type=Path, required=True)
    parser.add_argument("--left-index", type=Path, required=True)
    parser.add_argument("--left-actions", type=Path, required=True)
    parser.add_argument("--right-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-name", default="inherited_parent")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    right_meta_path = args.right_bank / "gpm_memory_meta.pt"
    right_index_path = args.right_bank / "gpm_memory.index"
    right_actions_path = args.right_bank / "gpm_memory_actions.npz"
    required = (
        args.left_meta,
        args.left_index,
        args.left_actions,
        right_meta_path,
        right_index_path,
        right_actions_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")

    print("Loading metadata", flush=True)
    left_meta = torch.load(args.left_meta, map_location="cpu", weights_only=False)
    right_meta = torch.load(right_meta_path, map_location="cpu", weights_only=False)
    left_actions, left_offsets, left_ids = _load_actions(args.left_actions)
    right_actions, right_offsets, right_ids = _load_actions(right_actions_path)
    duplicate_ids = sorted(set(left_ids) & set(right_ids))
    if duplicate_ids:
        raise ValueError(f"Cross-bank duplicate action IDs: {duplicate_ids[:3]}")
    _validate_metadata(left_meta, set(left_ids), args.left_meta)
    _validate_metadata(right_meta, set(right_ids), right_meta_path)

    print("Loading FAISS indexes", flush=True)
    left_index = faiss.read_index(str(args.left_index))
    right_index = faiss.read_index(str(right_index_path))
    if left_index.d != right_index.d or left_index.d != 2048:
        raise ValueError(f"Incompatible FAISS dimensions: {left_index.d} vs {right_index.d}")
    if left_index.ntotal != len(left_meta) or right_index.ntotal != len(right_meta):
        raise ValueError("FAISS and metadata counts do not match")
    right_vectors = right_index.reconstruct_n(0, right_index.ntotal)
    left_index.add(np.ascontiguousarray(right_vectors, dtype=np.float32))

    combined_actions = np.concatenate((left_actions, right_actions), axis=0)
    combined_offsets = np.concatenate(
        (left_offsets, right_offsets[1:] + int(left_offsets[-1])), axis=0
    )
    combined_ids = left_ids + right_ids
    combined_meta = left_meta + right_meta

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        faiss.write_index(left_index, str(temporary / "gpm_memory.index"))
        torch.save(combined_meta, temporary / "gpm_memory_meta.pt")
        with (temporary / "gpm_memory_actions.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                actions=combined_actions,
                offsets=combined_offsets,
                ids=np.asarray(combined_ids),
            )
        (temporary / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol": "predimem_hereditary_bank_append_v1",
                    "left_name": args.left_name,
                    "left": {
                        "meta": str(args.left_meta),
                        "index": str(args.left_index),
                        "actions": str(args.left_actions),
                        "entries": len(left_meta),
                        "actions_count": len(left_ids),
                    },
                    "right": {
                        "bank": str(args.right_bank),
                        "entries": len(right_meta),
                        "actions_count": len(right_ids),
                    },
                    "output": {
                        "entries": len(combined_meta),
                        "actions_count": len(combined_ids),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.output.exists():
            shutil.rmtree(args.output)
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(args.output), "entries": len(combined_meta)}), flush=True)


if __name__ == "__main__":
    main()
