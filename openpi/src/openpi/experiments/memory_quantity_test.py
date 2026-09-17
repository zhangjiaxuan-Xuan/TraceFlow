from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from openpi.experiments import memory_quantity as mq


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(task_id: int, index: int, *, success: bool) -> dict[str, object]:
    return {
        "action_id": f"task{task_id}-{'s' if success else 'f'}-{index}",
        "episode_idx": index,
        "prompt": f"task {task_id}",
        "producer": "natural-policy",
        "source_family": "natural_pi",
        "source_format": "eval_hdf5",
        "source_model": "pi0.5",
        "source_run": "frozen-natural-run",
        "success": success,
        "suite": "libero_10",
        "task_id": task_id,
        "trajectory_path": f"trajectories/{task_id}/{success}/{index}.hdf5",
    }


def _fixture(tmp_path: Path, *, per_outcome: int = 50, policy_family: str = "pi") -> mq.Experiment:
    root = tmp_path / "openpi"
    root.mkdir()
    natural = tmp_path / "natural.jsonl"
    rows = [
        _row(task_id, index, success=success)
        for task_id in mq.TASK_IDS
        for success in (True, False)
        for index in range(per_outcome)
    ]
    natural.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8")
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    model = policy_dir / "model.safetensors"
    model.write_bytes(b"fixed-policy")
    head = tmp_path / "head.pt"
    head.write_bytes(b"fixed-head")
    manifest = tmp_path / "experiment.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "natural_pi",
                "policy_family": policy_family,
                "openpi_root": str(root),
                "artifact_root": str(tmp_path / "artifacts"),
                "run_root": str(tmp_path / "runs"),
                "python": "python",
                "natural_manifest": {
                    "path": str(natural),
                    "sha256": _digest(natural),
                    "provenance": {
                        "producer": "natural-policy",
                        "source_family": "natural_pi",
                        "source_model": "pi0.5",
                        "source_run": "frozen-natural-run",
                    },
                },
                "policy": {
                    "directory": str(policy_dir),
                    "model_sha256": _digest(model),
                    "task_head_checkpoint": str(head),
                    "task_head_sha256": _digest(head),
                    "config_name": "pi05_libero",
                },
                "evaluation": {"seed": 7, "episodes_per_task": 100, "batch_size": 8},
            }
        ),
        encoding="utf-8",
    )
    return mq.load_experiment(manifest)


def test_seed7_subsets_are_exact_deterministic_and_nested(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    rows, _ = mq.preflight(experiment)
    small = mq.select_subset(rows, 1, 10)
    repeated = mq.select_subset(list(reversed(rows)), 1, 10)
    large = mq.select_subset(rows, 10, 50)
    assert [row["action_id"] for row in small] == [row["action_id"] for row in repeated]
    assert {row["action_id"] for row in small} <= {row["action_id"] for row in large}
    for task_id in mq.TASK_IDS:
        task_rows = [row for row in small if row["task_id"] == task_id]
        assert sum(row["success"] is True for row in task_rows) == 1
        assert sum(row["success"] is False for row in task_rows) == 10


def test_materialized_subset_has_hash_audit(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    audit = mq.materialize_subset(experiment, 10, 1)
    subset = Path(audit["subset_manifest"])
    assert audit["rows"] == 110
    assert audit["selection_seed"] == 7
    assert audit["subset_manifest_sha256"] == _digest(subset)
    assert json.loads(subset.with_suffix(".audit.json").read_text())["source_manifest_sha256"] == (
        experiment.natural.sha256
    )


def test_hard_fails_when_any_task_outcome_has_fewer_than_50(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path, per_outcome=49)
    with pytest.raises(mq.PreflightError, match="hard per-task/outcome quotas"):
        mq.preflight(experiment)


def test_pi_default_requires_only_ten_failures_per_task(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    rows = mq._read_jsonl(experiment.natural.manifest)  # noqa: SLF001
    rows = [row for row in rows if row["success"] or int(row["episode_idx"]) < 10]
    experiment.natural.manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    experiment = dataclasses.replace(
        experiment,
        natural=dataclasses.replace(experiment.natural, sha256=_digest(experiment.natural.manifest)),
    )
    _, audit = mq.preflight(experiment)
    assert min(audit["counts"]["failure"].values()) == 10


def test_hash_and_provenance_are_both_audited(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    rows = experiment.natural.manifest.read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[0])
    changed["source_run"] = "checkpoint-progress"
    rows[0] = json.dumps(changed)
    experiment.natural.manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(mq.PreflightError, match="digest mismatch"):
        mq.preflight(experiment)
    updated = dataclasses.replace(
        experiment,
        natural=dataclasses.replace(experiment.natural, sha256=_digest(experiment.natural.manifest)),
    )
    with pytest.raises(mq.PreflightError, match="provenance mismatch"):
        mq.preflight(updated)


def test_pi_excludes_failure_50_by_default_but_can_enable_it(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path, policy_family="pi")
    default_nodes = mq.build_dag(experiment, "eval")
    enabled_nodes = mq.build_dag(experiment, "eval", include_pi_failure_50=True)
    assert len(default_nodes) == 6
    assert len(enabled_nodes) == 9
    assert all(node.failure_count != 50 for node in default_nodes)
    assert {node.failure_count for node in enabled_nodes} == {1, 5, 50}


def test_non_pi_builds_full_v0_dag_with_fixed_controls(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path, policy_family="smol")
    nodes = mq.build_dag(experiment, "all")
    mq.validate_dag(nodes, phase="all")
    assert len(nodes) == 36
    eval_nodes = [node for node in nodes if node.kind == "eval"]
    assert len(eval_nodes) == 9
    env = dict(eval_nodes[0].env)
    assert env["BATCH_SIZE"] == "8"
    assert env["SEED"] == "7"
    assert eval_nodes[0].command == ("bash", "scripts/experiments/run_memory_quantity.sh")


def test_resume_rejects_identity_mismatch(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    node = mq.build_dag(experiment, "subsets")[0]
    node.identity_path.parent.mkdir(parents=True)
    node.identity_path.write_text(json.dumps({"artifact_identity_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(mq.PreflightError, match="completed identity mismatch"):
        mq.execute([node], experiment, dry_run=True, resume=True)
