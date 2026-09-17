#!/usr/bin/env python3
"""Filter a PrediMem bank without changing its producer/head embedding space."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import torch


def load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--label", choices=("any", "success", "failure"), default="any")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to mix into non-empty output: {args.output}")
    wanted = {int(value) for value in args.task_ids.split(",") if value.strip()}
    if not wanted:
        raise ValueError("task-ids must not be empty")
    metadata = load(args.meta)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("metadata must be a non-empty list")
    index = faiss.read_index(str(args.index))
    if int(index.ntotal) != len(metadata):
        raise ValueError("metadata/index length mismatch")
    keys = np.asarray(index.reconstruct_n(0, int(index.ntotal)), dtype=np.float32)
    with np.load(args.actions, allow_pickle=False) as packed:
        actions = np.asarray(packed["actions"], dtype=np.float32)
        offsets = np.asarray(packed["offsets"], dtype=np.int64)
        ids = [str(value) for value in np.asarray(packed["ids"]).tolist()]
    if len(ids) != len(set(ids)) or offsets.shape != (len(ids) + 1,):
        raise ValueError("invalid or duplicate action-store identity table")
    action_positions = {value: index for index, value in enumerate(ids)}
    metadata_action_ids = {str(row.get("action_id", "")) for row in metadata}
    missing_actions = metadata_action_ids.difference(action_positions)
    if missing_actions:
        raise ValueError(f"metadata references {len(missing_actions)} missing actions")

    keep = []
    for i, row in enumerate(metadata):
        if int(row.get("task_id", -1)) not in wanted:
            continue
        if args.label != "any":
            success = bool(row.get("success", row.get("label") == "positive"))
            if (args.label == "success") != success:
                continue
        keep.append(i)
    if not keep:
        raise RuntimeError(f"No bank entries matched tasks={sorted(wanted)} label={args.label}")

    args.output.mkdir(parents=True, exist_ok=True)
    out_meta = [metadata[i] for i in keep]
    out_keys = np.ascontiguousarray(keys[keep], dtype=np.float32)
    out_index = faiss.IndexFlatIP(int(out_keys.shape[1]))
    out_index.add(out_keys)
    faiss.write_index(out_index, str(args.output / "gpm_memory.index"))
    selected_action_ids = sorted({str(metadata[i]["action_id"]) for i in keep}, key=action_positions.__getitem__)
    action_indices = [action_positions[value] for value in selected_action_ids]
    pieces = [actions[int(offsets[i]) : int(offsets[i + 1])] for i in action_indices]
    out_offsets = np.zeros(len(pieces) + 1, dtype=np.int64)
    for i, piece in enumerate(pieces):
        out_offsets[i + 1] = out_offsets[i] + len(piece)
    out_actions = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    with (args.output / "gpm_memory_actions.npz").open("wb") as handle:
        np.savez_compressed(handle, actions=out_actions, offsets=out_offsets, ids=np.asarray(selected_action_ids))
    torch.save(out_meta, args.output / "gpm_memory_meta.pt")
    print({"items": len(keep), "actions": len(selected_action_ids), "tasks": sorted(wanted), "label": args.label, "output": str(args.output)})


if __name__ == "__main__":
    main()
