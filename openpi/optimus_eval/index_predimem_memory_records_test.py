from __future__ import annotations

import json

import numpy as np

from optimus_eval.index_predimem_memory_records import build_index


def test_builds_outcome_linked_failure_index(tmp_path):
    task_root = tmp_path / "task1"
    task_root.mkdir()
    outcomes = [
        {"task_id": 1, "episode": 0, "seed": 7, "TSR": 0.0, "CSR": 50.0, "failure_reason": "incomplete"},
        {"task_id": 1, "episode": 1, "seed": 8, "TSR": 100.0, "CSR": 100.0, "failure_reason": None},
    ]
    (task_root / "worker_00.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in outcomes), encoding="utf-8"
    )
    trace_root = tmp_path / "memory_records" / "traces"
    trace_root.mkdir(parents=True)
    manifest = []
    for episode in range(2):
        name = f"robomemarena_task_001_episode_{episode:03d}_call_0000.npz"
        np.savez_compressed(
            trace_root / name,
            x_trajectory=np.zeros((11, 1, 10, 32), dtype=np.float32),
            v_base=np.zeros((10, 1, 10, 32), dtype=np.float32),
            guidance_raw=np.zeros((10, 1, 10, 32), dtype=np.float32),
            guidance_clipped=np.zeros((10, 1, 10, 32), dtype=np.float32),
            memory_blocks=np.zeros((8, 10, 32), dtype=np.float32),
            retrieval_weights=np.ones(8, dtype=np.float32) / 8,
            retrieval_scores=np.ones(8, dtype=np.float32),
            memory_indices=np.arange(8, dtype=np.int64),
            window_starts=np.arange(8, dtype=np.int64),
            task_embedding=np.ones(2048, dtype=np.float32),
            lower_retrieval_feature=np.ones(6144, dtype=np.float32),
            upper_retrieval_feature=np.ones(4096, dtype=np.float32),
            upper_vlm_age=np.asarray(0.0, dtype=np.float32),
            upper_vlm_available=np.asarray(True, dtype=np.bool_),
        )
        manifest.append(
            {
                "trace": name,
                "task_id": 1,
                "episode_idx": episode,
                "policy_call_idx": 0,
                "progress": 0.25,
            }
        )
    (trace_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
    )

    summary = build_index(run_root=tmp_path, task_ids=[1], episodes_per_task=2, trace_level="full")
    assert summary == {
        "episodes": 2,
        "failed_episodes": 1,
        "successful_episodes": 1,
        "memory_calls": 2,
        "failed_memory_calls": 1,
        "successful_memory_calls": 1,
        "manifest_fast_records": 0,
        "npz_verified_records": 2,
    }
    failure = json.loads((tmp_path / "memory_records/failure_index.jsonl").read_text())
    assert failure["episode_idx"] == 0
    assert not failure["success"]
    assert failure["lower_feature_dim"] == 6144
    assert failure["upper_feature_dim"] == 4096
    assert failure["upper_vlm_available"]


def test_uses_complete_manifest_without_opening_npz(tmp_path):
    task_root = tmp_path / "task1"
    task_root.mkdir()
    (task_root / "worker_00.jsonl").write_text(
        json.dumps({"task_id": 1, "episode": 0, "seed": 7, "TSR": 100.0, "CSR": 100.0}) + "\n"
    )
    trace_root = tmp_path / "memory_records" / "traces"
    trace_root.mkdir(parents=True)
    trace_name = "robomemarena_task_001_episode_000_call_0000.npz"
    # The manifest is appended only after the writer atomically publishes this file.
    (trace_root / trace_name).write_bytes(b"not-opened-by-fast-index")
    manifest = {
        "schema_version": 2,
        "trace": trace_name,
        "task_id": 1,
        "episode_idx": 0,
        "policy_call_idx": 0,
        "progress": 0.5,
        "full_trace_schema_complete": True,
        "task_embedding_dim": 2048,
        "lower_feature_dim": 6144,
        "upper_feature_dim": 4096,
        "upper_vlm_available": True,
    }
    (trace_root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n")

    summary = build_index(run_root=tmp_path, task_ids=[1], episodes_per_task=1, trace_level="full")

    assert summary["manifest_fast_records"] == 1
    assert summary["npz_verified_records"] == 0
