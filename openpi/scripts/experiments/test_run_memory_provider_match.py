from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).with_name("run_memory_provider_match.py")
SPEC = importlib.util.spec_from_file_location("run_memory_provider_match_under_test", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _job(tmp_path: Path, consumer: str = "pi"):
    return MODULE.Job(consumer, "pi-mem", tmp_path / consumer, {"identity": "fixed"}, {})


def _write_complete(job) -> None:
    path = job.run_root / ("eval/libero_10.jsonl" if job.consumer == "pi" else "episodes.jsonl")
    path.parent.mkdir(parents=True)
    rows = []
    for task in range(10):
        for episode in range(100):
            seed = 7 if job.consumer == "pi" else 7 + task * 10_000 + episode
            rows.append(
                {
                    "event": "episode_result",
                    "task_suite": "libero_10",
                    "task_id": task,
                    "episode_idx": episode,
                    "seed": seed,
                    "success": (task + episode) % 2 == 0,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_strict_partial_and_completed_resume(tmp_path: Path) -> None:
    job = _job(tmp_path)
    job.run_root.mkdir(parents=True)
    (job.run_root / "match.run_identity.json").write_text(json.dumps(job.identity), encoding="utf-8")
    assert MODULE._preflight(job, resume=True) == "resume"  # noqa: SLF001
    with pytest.raises(MODULE.MatchProtocolError, match="requires resume"):
        MODULE._preflight(job, resume=False)  # noqa: SLF001

    _write_complete(job)
    (job.run_root / "match.identity.json").write_text(json.dumps(job.identity), encoding="utf-8")
    assert MODULE._preflight(job, resume=True) == "skip"  # noqa: SLF001
    assert MODULE.validate_completion(job)["episodes"] == 1000


def test_completion_rejects_non_boolean_success_and_wrong_smol_seed(tmp_path: Path) -> None:
    job = _job(tmp_path)
    _write_complete(job)
    log = job.run_root / "eval/libero_10.jsonl"
    rows = log.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["success"] = "False"
    rows[0] = json.dumps(first)
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(MODULE.MatchProtocolError, match="JSON bool"):
        MODULE.validate_completion(job)

    smol = _job(tmp_path, "smol")
    _write_complete(smol)
    log = smol.run_root / "episodes.jsonl"
    rows = log.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["seed"] = 999
    rows[0] = json.dumps(first)
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(MODULE.MatchProtocolError, match="seed mismatch"):
        MODULE.validate_completion(smol)
