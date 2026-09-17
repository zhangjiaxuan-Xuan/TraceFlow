#!/usr/bin/env python3
"""Materialize a PrediMem bank from outcome-indexed full inference traces."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile

import faiss
import numpy as np
import torch


def _load_trace(item: tuple[Path, dict]) -> tuple[np.ndarray, np.ndarray]:
    run_root, row = item
    trace_path = run_root / row["trace_path"]
    with np.load(trace_path, allow_pickle=False) as trace:
        key = np.asarray(trace["task_embedding"], dtype=np.float32).reshape(-1)
        chunk = np.asarray(trace["x_trajectory"][-1, 0], dtype=np.float32)
    if key.shape != (2048,) or chunk.shape != (10, 32):
        raise ValueError(f"Unexpected trace shapes in {trace_path}: key={key.shape}, action={chunk.shape}")
    if not np.isfinite(key).all() or not np.isfinite(chunk).all():
        raise ValueError(f"Non-finite trace data in {trace_path}")
    norm = float(np.linalg.norm(key))
    if norm <= 0:
        raise ValueError(f"Zero retrieval embedding in {trace_path}")
    return key / norm, chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--label", choices=("success", "failure"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--alignment", choices=("legacy_progress", "dense_frame_v3"), default="legacy_progress")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    indexed_rows: list[tuple[Path, dict]] = []
    index_paths = []
    for run_root in args.run_root:
        index_path = run_root / "memory_records" / f"{args.label}_index.jsonl"
        index_paths.append(index_path)
        indexed_rows.extend(
            (run_root, json.loads(line))
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if args.limit > 0:
        indexed_rows = indexed_rows[: args.limit]
    if not indexed_rows:
        raise RuntimeError(f"No {args.label} records in {index_paths}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        count = len(indexed_rows)
        keys = np.empty((count, 2048), dtype=np.float32)
        actions = np.empty((count * 10, 32), dtype=np.float32)
        offsets = np.arange(0, (count + 1) * 10, 10, dtype=np.int64)
        ids: list[str] = []
        metadata: list[dict] = []
        if args.workers < 1:
            raise ValueError("workers must be positive")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            loaded = executor.map(_load_trace, indexed_rows)
            for i, ((run_root, row), (key, chunk)) in enumerate(zip(indexed_rows, loaded, strict=True)):
                trace_path = run_root / row["trace_path"]
                keys[i] = key
                actions[i * 10 : (i + 1) * 10] = chunk
                source_tag = run_root.parent.name
                action_id = (
                    f"predimem_{args.label}_{source_tag}_task{int(row['task_id']):02d}_"
                    f"ep{int(row['episode_idx']):03d}_call{int(row['policy_call_idx']):04d}"
                )
                ids.append(action_id)
                entry = {
                        "action_id": action_id,
                        "task_emb": torch.from_numpy(keys[i].copy()),
                        "task_id": int(row["task_id"]),
                        "task_name": f"robomemarena_task_{int(row['task_id']):02d}",
                        "episode_idx": int(row["episode_idx"]),
                        "policy_call_idx": int(row["policy_call_idx"]),
                        "seed": int(row["seed"]),
                        "progress": float(row["progress"]),
                        "success": bool(row["success"]),
                        "failure_confidence": 1.0 if args.label == "failure" else 0.0,
                        "source": "predimem_full_inference_trace_v1",
                        "source_trace": str(trace_path),
                    }
                if args.alignment == "dense_frame_v3":
                    entry.update(
                        {
                            "anchor_frame": 0,
                            "anchor_progress": 0.0,
                            "length": 10,
                            "chunk_meta": {"T": 10, "chunk_len": 10, "stride": 1},
                        }
                    )
                metadata.append(entry)
                if (i + 1) % 2000 == 0 or i + 1 == count:
                    print(f"Loaded {i + 1}/{count}", flush=True)

        index = faiss.IndexFlatIP(2048)
        index.add(np.ascontiguousarray(keys))
        faiss.write_index(index, str(temporary / "gpm_memory.index"))
        torch.save(metadata, temporary / "gpm_memory_meta.pt")
        with (temporary / "gpm_memory_actions.npz").open("wb") as stream:
            np.savez_compressed(stream, actions=actions, offsets=offsets, ids=np.asarray(ids))
        (temporary / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "label": args.label,
                    "records": count,
                    "source_runs": [str(path) for path in args.run_root],
                    "source_indexes": [str(path) for path in index_paths],
                    "embedding_dim": 2048,
                    "action_shape": [10, 32],
                    "alignment": args.alignment,
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
    print(json.dumps({"output": str(args.output), "label": args.label, "records": len(indexed_rows)}))


if __name__ == "__main__":
    main()
