from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from optimus_eval.summarize_static_action_gate import summarize


def test_summarize_static_action_gate_joins_episode_outcomes(tmp_path: Path) -> None:
    batch = {
        "schema_version": 1,
        "mode": "gate",
        "rows": [
            {
                "task_id": 1,
                "episode_idx": 0,
                "mode": "gate",
                "gate": True,
                "gate_applied": True,
                "weighted_static_fraction": 0.75,
            },
            {
                "task_id": 1,
                "episode_idx": 1,
                "mode": "gate",
                "gate": False,
                "gate_applied": False,
                "weighted_static_fraction": 0.25,
            },
        ],
    }
    (tmp_path / "static_action_batches.jsonl").write_text(
        json.dumps(batch) + "\n", encoding="utf-8"
    )
    with (tmp_path / "episodes.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_id", "episode", "TSR", "CSR"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"task_id": 1, "episode": 0, "TSR": 1, "CSR": 100})
        writer.writerow({"task_id": 1, "episode": 1, "TSR": 0, "CSR": 50})

    payload = summarize(tmp_path)

    assert payload["calls"] == 2
    assert payload["gated_calls"] == 1
    assert payload["episodes_with_gate"] == 1
    task = payload["per_task"][0]
    assert task["gated_call_fraction"] == pytest.approx(0.5)
    assert task["TSR"] == pytest.approx(0.5)
    assert task["CSR_with_gate"] == pytest.approx(100.0)
