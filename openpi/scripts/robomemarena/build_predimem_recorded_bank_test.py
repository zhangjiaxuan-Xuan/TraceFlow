from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import faiss
import numpy as np
import torch


def test_builds_recorded_bank(tmp_path: Path) -> None:
    run = tmp_path / "run"
    traces = run / "memory_records" / "traces"
    traces.mkdir(parents=True)
    rows = []
    for call in range(2):
        name = f"trace_{call}.npz"
        np.savez(
            traces / name,
            task_embedding=np.full(2048, call + 1, dtype=np.float32),
            x_trajectory=np.full((11, 1, 10, 32), call + 3, dtype=np.float32),
        )
        rows.append(
            {
                "task_id": 18,
                "episode_idx": 2,
                "policy_call_idx": call,
                "seed": 9,
                "progress": call / 2,
                "success": False,
                "trace_path": f"memory_records/traces/{name}",
            }
        )
    (run / "memory_records/failure_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "bank"
    script = Path(__file__).with_name("build_predimem_recorded_bank.py")
    subprocess.run(
        [sys.executable, str(script), "--run-root", str(run), "--label", "failure", "--output", str(output)],
        check=True,
    )
    metadata = torch.load(output / "gpm_memory_meta.pt", map_location="cpu", weights_only=False)
    index = faiss.read_index(str(output / "gpm_memory.index"))
    with np.load(output / "gpm_memory_actions.npz", allow_pickle=False) as packed:
        assert packed["actions"].shape == (20, 32)
        assert packed["offsets"].tolist() == [0, 10, 20]
    assert len(metadata) == index.ntotal == 2
    assert np.isclose(np.linalg.norm(index.reconstruct(0)), 1.0)
    assert metadata[1]["policy_call_idx"] == 1
    assert len({row["action_id"] for row in metadata}) == 2


def test_merges_multiple_rounds_without_identity_collisions(tmp_path: Path) -> None:
    script = Path(__file__).with_name("build_predimem_recorded_bank.py")
    run_roots = []
    for round_index in (1, 2):
        run = tmp_path / f"round_{round_index:02d}" / "fusion"
        traces = run / "memory_records" / "traces"
        traces.mkdir(parents=True)
        np.savez(
            traces / "trace.npz",
            task_embedding=np.full(2048, round_index, dtype=np.float32),
            x_trajectory=np.full((11, 1, 10, 32), round_index, dtype=np.float32),
        )
        row = {
            "task_id": 18,
            "episode_idx": 0,
            "policy_call_idx": 0,
            "seed": 7,
            "progress": 0.0,
            "success": True,
            "trace_path": "memory_records/traces/trace.npz",
        }
        (run / "memory_records/success_index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        run_roots.append(run)
    output = tmp_path / "merged"
    command = [sys.executable, str(script)]
    for run in run_roots:
        command.extend(("--run-root", str(run)))
    command.extend(("--label", "success", "--output", str(output), "--workers", "2"))
    subprocess.run(command, check=True)
    metadata = torch.load(output / "gpm_memory_meta.pt", map_location="cpu", weights_only=False)
    assert len(metadata) == 2
    assert len({row["action_id"] for row in metadata}) == 2
