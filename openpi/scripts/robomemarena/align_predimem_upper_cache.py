from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _subtasks(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8") as stream:
        return {
            int(record["row_index"]): record
            for line in stream
            if line.strip()
            for record in (json.loads(line),)
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causally hold a slower PrediMem Upper cache on a dense Lower manifest."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--age-normalizer-steps", type=float, default=20.0)
    args = parser.parse_args()
    if args.age_normalizer_steps <= 0:
        raise ValueError("age-normalizer-steps must be positive")

    source_rows = _rows(args.source_manifest)
    target_rows = _rows(args.target_manifest)
    source_state_path = args.source_cache_dir / "upper_cache_state.json"
    source_features_path = args.source_cache_dir / "upper_features.npy"
    source_subtasks_path = args.source_cache_dir / "subtasks.jsonl"
    source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
    source_features = np.load(source_features_path, mmap_mode="r")
    source_subtasks = _subtasks(source_subtasks_path)
    if not bool(source_state.get("complete")):
        raise RuntimeError("Source Upper cache is incomplete")
    if len(source_rows) != len(source_features) or len(source_subtasks) != len(source_rows):
        raise RuntimeError("Source manifest, Upper features, and subtasks differ in length")

    output_state_path = args.output_dir / "upper_cache_state.json"
    output_features_path = args.output_dir / "upper_features.npy"
    output_subtasks_path = args.output_dir / "subtasks.jsonl"
    output_age_path = args.output_dir / "upper_age.npy"
    output_available_path = args.output_dir / "upper_available.npy"
    output_indices_path = args.output_dir / "upper_source_indices.npy"
    if output_state_path.is_file():
        output_state = json.loads(output_state_path.read_text(encoding="utf-8"))
        reusable = (
            bool(output_state.get("complete"))
            and int(output_state.get("rows", -1)) == len(target_rows)
            and output_state.get("manifest_sha256") == _digest(args.target_manifest)
            and output_state.get("source_manifest_sha256") == _digest(args.source_manifest)
            and output_state.get("source_cache") == str(args.source_cache_dir.resolve())
            and float(output_state.get("age_normalizer_steps", -1)) == args.age_normalizer_steps
            and all(
                path.is_file()
                for path in (
                    output_features_path,
                    output_subtasks_path,
                    output_age_path,
                    output_available_path,
                    output_indices_path,
                )
            )
        )
        if reusable:
            held_features = np.load(output_features_path, mmap_mode="r")
            held_age = np.load(output_age_path, mmap_mode="r")
            held_indices = np.load(output_indices_path, mmap_mode="r")
            held_available = np.load(output_available_path, mmap_mode="r")
            if (
                held_features.shape == (len(target_rows), int(source_features.shape[1]))
                and held_age.shape == (len(target_rows),)
                and held_indices.shape == (len(target_rows),)
                and held_available.shape == (len(target_rows),)
            ):
                print(f"Causal Upper alignment already complete: {len(target_rows)} rows")
                return

    by_action: dict[str, list[int]] = {}
    for index, row in enumerate(source_rows):
        by_action.setdefault(str(row["action_id"]), []).append(index)
    for indices in by_action.values():
        indices.sort(key=lambda index: int(source_rows[index]["global_frame_index"]))

    source_indices = np.empty(len(target_rows), dtype=np.int64)
    stale_steps = np.empty(len(target_rows), dtype=np.int32)
    available = np.ones(len(target_rows), dtype=np.bool_)
    for target_index, row in enumerate(target_rows):
        candidates = by_action.get(str(row["action_id"]))
        if not candidates:
            raise RuntimeError(f"No Upper trajectory for {row['action_id']}")
        frames = np.fromiter(
            (int(source_rows[index]["global_frame_index"]) for index in candidates),
            dtype=np.int64,
        )
        position = int(np.searchsorted(frames, int(row["global_frame_index"]), side="right") - 1)
        if position < 0:
            # No future leakage before the first causal Upper update.
            source_index = candidates[0]
            available[target_index] = False
        else:
            source_index = candidates[position]
        source_indices[target_index] = source_index
        stale_steps[target_index] = max(
            0,
            int(row["global_frame_index"])
            - int(source_rows[source_index]["global_frame_index"]),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_features_path,
        mode="w+",
        dtype=source_features.dtype,
        shape=(len(target_rows), int(source_features.shape[1])),
    )
    for start in range(0, len(target_rows), 8192):
        stop = min(start + 8192, len(target_rows))
        output[start:stop] = source_features[source_indices[start:stop]]
        output[start:stop][~available[start:stop]] = 0
    output.flush()
    del output

    age = np.clip(stale_steps.astype(np.float32) / args.age_normalizer_steps, 0.0, 1.0)
    age[~available] = 1.0
    reported_source_indices = source_indices.copy()
    reported_source_indices[~available] = -1
    np.save(args.output_dir / "upper_age.npy", age)
    np.save(args.output_dir / "upper_source_indices.npy", reported_source_indices)
    np.save(args.output_dir / "upper_available.npy", available)

    with output_subtasks_path.open("w", encoding="utf-8") as stream:
        for target_index, (row, source_index, stale) in enumerate(
            zip(target_rows, source_indices.tolist(), stale_steps.tolist(), strict=True)
        ):
            record = dict(source_subtasks[int(source_index)])
            if not bool(available[target_index]):
                record["subtask"] = str(row.get("prompt", ""))
                record["fallback_to_previous_subtask"] = True
            record.update(
                {
                    "row_index": target_index,
                    "global_frame_index": int(row["global_frame_index"]),
                    "held_source_row_index": (
                        int(source_index) if bool(available[target_index]) else -1
                    ),
                    "upper_stale_steps": int(stale),
                    "upper_available": bool(available[target_index]),
                }
            )
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    state = dict(source_state)
    state.update(
        {
            "rows": len(target_rows),
            "manifest": str(args.target_manifest.resolve()),
            "manifest_sha256": _digest(args.target_manifest),
            "subtask_records": str(output_subtasks_path.resolve()),
            "subtask_records_sha256": _digest(output_subtasks_path),
            "subtask_rows": len(target_rows),
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": _digest(args.source_manifest),
            "source_cache": str(args.source_cache_dir.resolve()),
            "causal_hold": True,
            "age_normalizer_steps": float(args.age_normalizer_steps),
            "max_stale_steps": int(stale_steps.max(initial=0)),
            "unavailable_rows": int((~available).sum()),
            "complete": True,
        }
    )
    output_state_path.write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "target_rows": len(target_rows),
                "source_rows": len(source_rows),
                "mean_stale_steps": float(stale_steps.mean()),
                "max_stale_steps": int(stale_steps.max(initial=0)),
                "output": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
