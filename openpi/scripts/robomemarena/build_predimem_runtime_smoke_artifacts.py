from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import h5py
import numpy as np
import torch

from openpi.task_head.dual_tower_head import DualTowerRetrievalHead, checkpoint_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lower-features", type=Path, required=True)
    parser.add_argument("--upper-features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as stream:
        row = json.loads(next(stream))
    lower = np.array(np.load(args.lower_features, mmap_mode="r")[0], dtype=np.float32, copy=True)
    upper = np.array(np.load(args.upper_features, mmap_mode="r")[0], dtype=np.float32, copy=True)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError("Runtime smoke requires completed real upper and lower feature row 0")

    torch.manual_seed(7)
    args.output_root.mkdir(parents=True, exist_ok=True)
    shared = args.output_root / "memory" / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    chunks = []
    for path in row["segment_paths"]:
        with h5py.File(path, "r") as data:
            chunks.append(np.asarray(data["data/demo_0/actions"], dtype=np.float32))
    raw = np.concatenate(chunks)
    actions = np.zeros((len(raw), 32), dtype=np.float32)
    actions[:, : raw.shape[1]] = raw
    with (shared / "gpm_memory_actions.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            actions=actions,
            offsets=np.asarray([0, len(actions)], dtype=np.int64),
            ids=np.asarray([row["action_id"]]),
        )

    for variant in ("lower", "upper", "fusion"):
        head = DualTowerRetrievalHead(
            variant=variant,
            lower_dim=lower.shape[0],
            upper_dim=upper.shape[0],
            hidden=1024,
            out_dim=256,
        ).eval()
        with torch.inference_mode():
            key = head(
                torch.from_numpy(lower[None]),
                torch.from_numpy(upper[None]),
                torch.zeros(1),
                torch.ones(1),
            )[0].numpy()
        head_dir = args.output_root / "heads" / variant
        bank_dir = args.output_root / "memory" / variant
        head_dir.mkdir(parents=True, exist_ok=True)
        bank_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(head, head.state_dict(), smoke_only=True),
            head_dir / "best.pt",
        )
        index = faiss.IndexFlatIP(256)
        index.add(np.ascontiguousarray(key[None], dtype=np.float32))
        faiss.write_index(index, str(bank_dir / "gpm_memory.index"))
        torch.save(
            [
                {
                    "task_name": row["prompt"],
                    "task_id": int(row["task_id"]),
                    "suite": row["suite"],
                    "stage_index": int(row["stage_index"]),
                    "anchor_progress": float(row["anchor_progress"]),
                    "task_emb": torch.from_numpy(key.copy()),
                    "action_id": row["action_id"],
                    "length": int(row["trajectory_length"]),
                    "chunk_meta": {"chunk_len": 10, "stride": 10, "T": int(row["trajectory_length"])},
                }
            ],
            bank_dir / "gpm_memory_meta.pt",
        )
    print(f"Runtime smoke artifacts: {args.output_root}")


if __name__ == "__main__":
    main()
