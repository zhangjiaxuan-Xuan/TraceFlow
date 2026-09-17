from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from openpi.experiments import cross_policy_long_eval as subject


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _condition(tmp_path: Path, *, suite: str = "libero_10") -> subject.Condition:
    tasks = subject.SUITE_TASK_COUNTS[suite]
    return subject.Condition(
        condition_id=f"pi_base_{suite}",
        consumer="pi",
        mode="base",
        suite=suite,
        expected_tasks=tasks,
        episodes_per_task=2,
        episode_start=4,
        seed=7,
        seed_mapping=subject.SeedMapping(
            scheme="base_plus_task_and_episode",
            task_stride=100,
            episode_stride=1,
            record_required=True,
        ),
        batch_size=8,
        model_sha256="1" * 64,
        config_sha256="2" * 64,
        command=("true",),
        cwd=tmp_path,
        output_root=tmp_path / "output",
        output_log=tmp_path / "output/eval.jsonl",
        existing=None,
    )


def _write_complete_log(path: Path, condition: subject.Condition) -> None:
    rows = [
        {
            "event": "episode_result",
            "task_suite": condition.suite,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "seed": 7 + task_id * 100 + episode_idx - 4,
            "success": (task_id + episode_idx) % 2 == 0,
        }
        for task_id in range(condition.expected_tasks)
        for episode_idx in range(4, 6)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize(("suite", "tasks"), [("libero_10", 10), ("libero_90", 90)])
def test_reconstructs_complete_task_set(tmp_path: Path, suite: str, tasks: int) -> None:
    condition = _condition(tmp_path, suite=suite)
    _write_complete_log(condition.output_log, condition)
    audit = subject.audit_log(condition, condition.output_log, source="output")
    assert audit.tasks == tasks
    assert audit.episodes == tasks * 2
    assert set(audit.task_results) == {str(index) for index in range(tasks)}


def test_incomplete_existing_becomes_gap(tmp_path: Path) -> None:
    condition = _condition(tmp_path)
    identity = tmp_path / "identity.json"
    log = tmp_path / "partial.jsonl"
    _write_json(identity, condition.expected_identity)
    log.write_text(
        json.dumps(
            {
                "event": "episode_result",
                "task_suite": "libero_10",
                "task_id": 0,
                "episode_idx": 4,
                "seed": 7,
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    condition = dataclasses.replace(
        condition,
        existing=subject.ExistingResult(
            log=subject.Artifact(log, _digest(log)),
            identity=subject.Artifact(identity, _digest(identity)),
        ),
    )
    assert subject.audit_existing(condition) is None


def test_adopt_requires_digests_identity_and_summary(tmp_path: Path) -> None:
    condition = _condition(tmp_path)
    log = tmp_path / "complete.jsonl"
    identity = tmp_path / "identity.json"
    summary = tmp_path / "summary.json"
    _write_complete_log(log, condition)
    _write_json(identity, condition.expected_identity)
    rebuilt = subject._summarize_rows(  # noqa: SLF001
        condition,
        subject._episode_rows(log, condition),  # noqa: SLF001
    )
    _write_json(summary, rebuilt)
    condition = dataclasses.replace(
        condition,
        existing=subject.ExistingResult(
            log=subject.Artifact(log, _digest(log)),
            identity=subject.Artifact(identity, _digest(identity)),
            summary=subject.Artifact(summary, _digest(summary)),
        ),
    )
    audit = subject.audit_existing(condition)
    assert audit is not None
    assert audit.status == "adopted"
    assert audit.task_results["0"]["episodes"] == 2

    condition.existing.log.path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(subject.AuditError, match="SHA-256"):
        subject.audit_existing(condition)


def test_manifest_forbids_smol_prior_and_wrong_long_size(tmp_path: Path) -> None:
    base = {
        "schema_version": 1,
        "run_root": "runs",
        "conditions": [
            {
                "id": "bad",
                "consumer": "smol",
                "mode": "fixed_prior",
                "suite": "libero_90",
                "expected_tasks": 10,
                "episodes_per_task": 1,
                "seed_mapping": {
                    "scheme": "base_plus_task_and_episode",
                    "task_stride": 0,
                    "episode_stride": 0,
                    "record_required": False,
                },
                "model_sha256": "1" * 64,
                "config_sha256": "2" * 64,
                "command": ["true"],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    _write_json(path, base)
    with pytest.raises(subject.AuditError, match="no action-prior"):
        subject.load_experiment(path)

    base["conditions"][0].update(consumer="pi", mode="base")
    _write_json(path, base)
    with pytest.raises(subject.AuditError, match="requires 90 tasks"):
        subject.load_experiment(path)


def test_manifest_requires_complete_pi_smol_base_matrix(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "run_root": "runs",
        "conditions": [
            {
                "id": "pi10",
                "consumer": "pi",
                "mode": "base",
                "suite": "libero_10",
                "expected_tasks": 10,
                "episodes_per_task": 1,
                "seed_mapping": {
                    "scheme": "base_plus_task_and_episode",
                    "task_stride": 0,
                    "episode_stride": 0,
                    "record_required": False,
                },
                "model_sha256": "1" * 64,
                "config_sha256": "2" * 64,
                "command": ["true"],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    _write_json(path, payload)
    with pytest.raises(subject.AuditError, match="missing required Pi/Smol base conditions"):
        subject.load_experiment(path)


def test_command_expansion_is_argv_safe(tmp_path: Path) -> None:
    condition = _condition(tmp_path)
    condition = dataclasses.replace(
        condition,
        command=("runner", "--suite", "{suite}", "--output", "{run_root}"),
    )
    expanded = subject.expand_command(condition)
    assert expanded[2] == "libero_10"
    assert expanded[4] == str(condition.output_root)


@pytest.mark.parametrize("bad_success", ["False", 0, 1, None])
def test_success_requires_json_boolean(tmp_path: Path, bad_success: object) -> None:
    condition = _condition(tmp_path)
    _write_complete_log(condition.output_log, condition)
    rows = [json.loads(line) for line in condition.output_log.read_text().splitlines()]
    rows[0]["success"] = bad_success
    condition.output_log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(subject.AuditError, match="success must be a JSON boolean"):
        subject.audit_log(condition, condition.output_log, source="output")


def test_seed_mapping_is_strict_when_seed_is_present(tmp_path: Path) -> None:
    condition = _condition(tmp_path)
    _write_complete_log(condition.output_log, condition)
    rows = [json.loads(line) for line in condition.output_log.read_text().splitlines()]
    rows[-1]["seed"] += 1
    condition.output_log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(subject.AuditError, match="seed mismatch"):
        subject.audit_log(condition, condition.output_log, source="output")


def test_optional_episode_seed_still_has_explicit_identity_mapping(tmp_path: Path) -> None:
    condition = dataclasses.replace(
        _condition(tmp_path),
        seed_mapping=subject.SeedMapping(
            scheme="base_plus_task_and_episode",
            task_stride=0,
            episode_stride=0,
            record_required=False,
        ),
    )
    _write_complete_log(condition.output_log, condition)
    rows = [json.loads(line) for line in condition.output_log.read_text().splitlines()]
    for row in rows:
        row.pop("seed")
    condition.output_log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit = subject.audit_log(condition, condition.output_log, source="output")
    assert audit.episodes == 20
    assert condition.expected_identity["seed_mapping"]["record_required"] is False
