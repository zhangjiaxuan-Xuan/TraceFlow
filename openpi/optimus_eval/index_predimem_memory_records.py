from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            tmp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def build_index(
    *,
    run_root: Path,
    task_ids: list[int],
    episodes_per_task: int,
    trace_level: str,
    verify_npz: bool = False,
) -> dict[str, int]:
    outcome_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for task_id in task_ids:
        for path in sorted((run_root / f"task{task_id}").glob("worker_*.jsonl")):
            for row in _read_jsonl(path):
                key = (int(row["task_id"]), int(row["episode"]))
                if key in outcome_by_key:
                    raise RuntimeError(f"Duplicate outcome record for task/episode {key}")
                outcome_by_key[key] = row
    expected = {(task_id, episode) for task_id in task_ids for episode in range(episodes_per_task)}
    if set(outcome_by_key) != expected:
        missing = sorted(expected.difference(outcome_by_key))
        extra = sorted(set(outcome_by_key).difference(expected))
        raise RuntimeError(f"Outcome coverage mismatch: missing={missing[:20]} extra={extra[:20]}")

    trace_root = run_root / "memory_records" / "traces"
    manifest = _read_jsonl(trace_root / "manifest.jsonl")
    if not manifest:
        raise RuntimeError(f"No memory trace manifest records found under {trace_root}")
    suffix = ".npz" if trace_level in ("full", "bank") else ".light.jsonl"
    required_full = {
        "x_trajectory",
        "v_base",
        "guidance_raw",
        "guidance_clipped",
        "memory_blocks",
        "retrieval_weights",
        "retrieval_scores",
        "memory_indices",
        "window_starts",
        "task_embedding",
        "lower_retrieval_feature",
        "upper_retrieval_feature",
        "upper_vlm_age",
        "upper_vlm_available",
    }
    required_bank = {
        "x_trajectory",
        "retrieval_weights",
        "retrieval_scores",
        "memory_indices",
        "window_starts",
        "task_embedding",
        "lower_retrieval_feature",
        "upper_retrieval_feature",
        "upper_vlm_age",
        "upper_vlm_available",
    }
    indexed = []
    seen_calls: set[tuple[int, int, int]] = set()
    total_traces = len(manifest)
    fast_records = 0
    verified_records = 0
    print(f"Indexing {total_traces} memory traces from {trace_root}", flush=True)
    for record_number, meta in enumerate(manifest, start=1):
        task_id = int(meta.get("task_id", -1))
        episode = int(meta.get("episode_idx", -1))
        policy_call = int(meta.get("policy_call_idx", -1))
        key = (task_id, episode)
        call_key = (task_id, episode, policy_call)
        if key not in outcome_by_key or policy_call < 0:
            raise RuntimeError(f"Unlinkable memory trace context: {call_key}")
        if call_key in seen_calls:
            raise RuntimeError(f"Duplicate memory trace context: {call_key}")
        seen_calls.add(call_key)
        trace_name = str(meta["trace"])
        if not trace_name.endswith(suffix):
            raise RuntimeError(f"Trace level/name mismatch: level={trace_level} trace={trace_name}")
        trace_path = trace_root / trace_name
        lower_feature_dim = 0
        upper_feature_dim = 0
        upper_vlm_available = False
        if trace_level in ("full", "bank"):
            manifest_complete = bool(
                meta.get("full_trace_schema_complete", False)
                if trace_level == "full"
                else meta.get("bank_trace_schema_complete", False)
            )
            if manifest_complete and not verify_npz:
                task_embedding_dim = int(meta.get("task_embedding_dim", 0))
                lower_feature_dim = int(meta.get("lower_feature_dim", 0))
                upper_feature_dim = int(meta.get("upper_feature_dim", 0))
                upper_vlm_available = bool(meta.get("upper_vlm_available", False))
                if min(task_embedding_dim, lower_feature_dim, upper_feature_dim) <= 0:
                    raise RuntimeError(f"Invalid full-trace dimensions in manifest for {trace_name}")
                fast_records += 1
            else:
                if not trace_path.is_file():
                    raise FileNotFoundError(trace_path)
                with np.load(trace_path, allow_pickle=False) as data:
                    required = required_full if trace_level == "full" else required_bank
                    missing = required.difference(data.files)
                    if missing:
                        raise RuntimeError(f"{trace_path} is missing memory arrays: {sorted(missing)}")
                    if data["task_embedding"].size == 0:
                        raise RuntimeError(f"{trace_path} has an empty retrieval query embedding")
                    if data["lower_retrieval_feature"].size == 0:
                        raise RuntimeError(f"{trace_path} has an empty Lower producer feature")
                    if data["upper_retrieval_feature"].size == 0:
                        raise RuntimeError(f"{trace_path} has an empty Upper producer feature")
                    lower_feature_dim = int(data["lower_retrieval_feature"].size)
                    upper_feature_dim = int(data["upper_retrieval_feature"].size)
                    upper_vlm_available = bool(data["upper_vlm_available"])
                verified_records += 1
        elif not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        outcome = outcome_by_key[key]
        success = float(outcome["TSR"]) >= 100.0
        indexed.append(
            {
                "schema_version": 1,
                "task_id": task_id,
                "episode_idx": episode,
                "policy_call_idx": policy_call,
                "seed": int(outcome["seed"]),
                "success": success,
                "TSR": float(outcome["TSR"]),
                "CSR": float(outcome["CSR"]),
                "failure_reason": str(outcome.get("failure_reason") or ""),
                "progress": float(meta.get("progress", np.nan)),
                "trace_level": trace_level,
                "lower_feature_dim": lower_feature_dim,
                "upper_feature_dim": upper_feature_dim,
                "upper_vlm_available": upper_vlm_available,
                "trace_path": str(trace_path.relative_to(run_root)),
            }
        )
        if record_number % 10_000 == 0 or record_number == total_traces:
            print(
                f"Indexed {record_number}/{total_traces} traces "
                f"(manifest_fast={fast_records}, npz_verified={verified_records})",
                flush=True,
            )
    indexed.sort(key=lambda row: (row["task_id"], row["episode_idx"], row["policy_call_idx"]))
    output_root = run_root / "memory_records"
    _atomic_jsonl(output_root / "index.jsonl", indexed)
    _atomic_jsonl(output_root / "failure_index.jsonl", [row for row in indexed if not row["success"]])
    _atomic_jsonl(output_root / "success_index.jsonl", [row for row in indexed if row["success"]])
    summary = {
        "episodes": len(outcome_by_key),
        "failed_episodes": sum(float(row["TSR"]) < 100.0 for row in outcome_by_key.values()),
        "successful_episodes": sum(float(row["TSR"]) >= 100.0 for row in outcome_by_key.values()),
        "memory_calls": len(indexed),
        "failed_memory_calls": sum(not row["success"] for row in indexed),
        "successful_memory_calls": sum(row["success"] for row in indexed),
        "manifest_fast_records": fast_records,
        "npz_verified_records": verified_records,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--trace-level", choices=("full", "light", "bank"), required=True)
    parser.add_argument("--verify-npz", action="store_true")
    args = parser.parse_args()
    task_ids = [int(value) for value in args.task_ids.split(",") if value]
    summary = build_index(
        run_root=args.run_root,
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        trace_level=args.trace_level,
        verify_npz=args.verify_npz,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
