#!/usr/bin/env python3
"""Merge disjoint producer caches by manifest identity, preserving full order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row: dict) -> tuple[str, int]:
    return str(row["action_id"]), int(row["global_frame_index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("upper", "lower"), required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--extra-manifest", type=Path, required=True)
    parser.add_argument("--extra-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--prompt-overrides", type=Path, default=None)
    parser.add_argument("--config-name", default="")
    args = parser.parse_args()

    full = rows(args.full_manifest)
    base = rows(args.base_manifest)
    extra = rows(args.extra_manifest)
    completed_name = "upper_completed.npy" if args.kind == "upper" else "completed.npy"
    full_keys = [key(row) for row in full]
    if len(set(full_keys)) != len(full_keys):
        raise RuntimeError("Full manifest contains duplicate trajectory/frame identities")
    source = {}
    for label, manifest, cache in (("base", base, args.base_cache), ("extra", extra, args.extra_cache)):
        for row in manifest:
            identity = key(row)
            if identity in source:
                raise RuntimeError(f"Overlapping cache identity: {identity}")
            source[identity] = (label, row)
        if not (cache / completed_name).is_file():
            raise FileNotFoundError(cache / completed_name)
    if set(source) != set(full_keys):
        raise RuntimeError(
            f"Cache/manifests do not cover full manifest: missing={len(set(full_keys)-set(source))} "
            f"extra={len(set(source)-set(full_keys))}"
        )

    feature_name = "upper_features.npy" if args.kind == "upper" else "pooled_prefix.npy"
    base_features = np.load(args.base_cache / feature_name, mmap_mode="r")
    extra_features = np.load(args.extra_cache / feature_name, mmap_mode="r")
    base_done = np.load(args.base_cache / completed_name, mmap_mode="r")
    extra_done = np.load(args.extra_cache / completed_name, mmap_mode="r")
    if len(base) != len(base_features) or len(extra) != len(extra_features):
        raise RuntimeError(f"{args.kind} cache feature/manifest length mismatch")
    if not bool(np.all(base_done)) or not bool(np.all(extra_done)):
        raise RuntimeError(f"{args.kind} cache is incomplete")
    if base_features.ndim != 2 or extra_features.ndim != 2 or base_features.shape[1] != extra_features.shape[1]:
        raise RuntimeError(f"{args.kind} feature dimensions differ")

    base_index = {key(row): i for i, row in enumerate(base)}
    extra_index = {key(row): i for i, row in enumerate(extra)}
    args.output_cache.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output_cache / feature_name,
        mode="w+",
        dtype=np.result_type(base_features.dtype, extra_features.dtype),
        shape=(len(full), int(base_features.shape[1])),
    )
    for start in range(0, len(full), 8192):
        stop = min(start + 8192, len(full))
        for target, row in enumerate(full[start:stop], start):
            identity = key(row)
            label, _ = source[identity]
            index = (base_index if label == "base" else extra_index)[identity]
            values = base_features[index] if label == "base" else extra_features[index]
            output[target] = values
    output.flush()
    np.save(args.output_cache / completed_name, np.ones(len(full), dtype=np.bool_))

    if args.kind == "upper":
        records = {}
        for label, manifest, cache in (("base", base, args.base_cache), ("extra", extra, args.extra_cache)):
            source_records = cache / "subtasks.jsonl"
            if not source_records.is_file():
                raise FileNotFoundError(source_records)
            local = {int(row["row_index"]): row for row in rows(source_records)}
            for i, row in enumerate(manifest):
                record = dict(local[i])
                record["_identity"] = key(row)
                records[key(row)] = record
        with (args.output_cache / "subtasks.jsonl").open("w", encoding="utf-8") as stream:
            for index, row in enumerate(full):
                record = dict(records[key(row)])
                record.pop("_identity", None)
                record["row_index"] = index
                record["global_frame_index"] = int(row["global_frame_index"])
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        state_candidates = [json.loads((args.base_cache / "upper_cache_state.json").read_text()), json.loads((args.extra_cache / "upper_cache_state.json").read_text())]
        state = dict(state_candidates[0])
        state.update({
            "rows": len(full), "manifest": str(args.full_manifest.resolve()),
            "manifest_sha256": digest(args.full_manifest), "subtask_records": str((args.output_cache / "subtasks.jsonl").resolve()),
            "subtask_records_sha256": digest(args.output_cache / "subtasks.jsonl"), "subtask_rows": len(full), "complete": True,
        })
        (args.output_cache / "upper_cache_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    else:
        base_state = json.loads((args.base_cache / "cache_state.json").read_text())
        extra_state = json.loads((args.extra_cache / "cache_state.json").read_text())
        state = dict(base_state)
        state.update({"rows": len(full), "manifest": str(args.full_manifest.resolve()), "manifest_sha256": digest(args.full_manifest)})
        source_configs = sorted({str(base_state.get("config_name", "")), str(extra_state.get("config_name", ""))})
        state["source_config_names"] = source_configs
        if args.config_name:
            state["config_name"] = args.config_name
        if args.prompt_overrides is not None:
            state["prompt_overrides"] = str(args.prompt_overrides.resolve())
            state["prompt_overrides_sha256"] = digest(args.prompt_overrides)
        (args.output_cache / "cache_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kind": args.kind, "rows": len(full), "output": str(args.output_cache)}))


if __name__ == "__main__":
    main()
