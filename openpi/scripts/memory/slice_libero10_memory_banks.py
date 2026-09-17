#!/usr/bin/env python3
"""Create immutable LIBERO-10-only slices from existing GPM banks.

The slice keeps metadata, FAISS vectors, and packed actions aligned.  It never
recomputes features or modifies the source bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def suite_of(item: dict[str, Any]) -> str:
    provenance = item.get("provenance") or {}
    return str(item.get("suite") or provenance.get("suite") or provenance.get("task_suite") or "")


def slice_bank(source: Path, output: Path, prefix: str) -> dict[str, Any]:
    meta = list(load(source / f"{prefix}_meta.pt"))
    index = faiss.read_index(str(source / f"{prefix}.index"))
    with np.load(source / f"{prefix}_actions.npz", allow_pickle=False) as packed:
        actions = np.asarray(packed["actions"], dtype=np.float32)
        offsets = np.asarray(packed["offsets"], dtype=np.int64)
        ids = [str(value) for value in packed["ids"].tolist()]

    if len(meta) != index.ntotal or len(meta) != len(ids) or offsets.shape != (len(meta) + 1,):
        raise ValueError(f"Misaligned source bank: {source}")
    selected = [i for i, item in enumerate(meta) if suite_of(item) == "libero_10"]
    if not selected:
        suites = sorted({suite_of(item) for item in meta})
        raise ValueError(f"No libero_10 entries in {source}; observed suites={suites}")

    keys = np.asarray(index.reconstruct_n(0, len(meta)), dtype=np.float32)[selected]
    parts = [actions[int(offsets[i]) : int(offsets[i + 1])] for i in selected]
    new_offsets = np.zeros(len(selected) + 1, dtype=np.int64)
    for i, part in enumerate(parts):
        new_offsets[i + 1] = new_offsets[i] + len(part)
    packed_actions = np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    output.mkdir(parents=True, exist_ok=True)
    new_index = faiss.IndexFlatIP(int(keys.shape[1]))
    new_index.add(np.ascontiguousarray(keys))
    faiss.write_index(new_index, str(output / f"{prefix}.index"))
    torch.save([meta[i] for i in selected], output / f"{prefix}_meta.pt")
    np.savez_compressed(
        output / f"{prefix}_actions.npz",
        actions=packed_actions,
        offsets=new_offsets,
        ids=np.asarray([ids[i] for i in selected]),
    )
    summary = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "filter": {"suite": "libero_10"},
        "prefix": prefix,
        "items": len(selected),
        "embedding_dim": int(keys.shape[1]),
        "task_names": sorted({str(m.get("task_name", "")) for m in (meta[i] for i in selected)}),
    }
    (output / "slice_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-source", type=Path, required=True)
    parser.add_argument("--negative-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    positive = slice_bank(args.positive_source, args.output_root / "positive", "gpm_memory")
    negative = slice_bank(args.negative_source, args.output_root / "negative", "gpm_negative_memory")
    print(json.dumps({"positive": positive, "negative": negative}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
