from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from openpi.experiments import checkpoint_progress as cp


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, failures: int = 10) -> cp.Experiment:
    root = tmp_path / "openpi"
    root.mkdir()
    sources = {}
    for name, producer in (("B", "offline"), ("N", "pi0.5-base"), ("S", "smolvla-base")):
        manifest = tmp_path / f"{name}.jsonl"
        rows = []
        if name == "B":
            rows.extend({"suite": "libero_10", "task_id": index % 10, "success": True} for index in range(6500))
        else:
            for task_id in range(10):
                rows.extend(
                    {"suite": "libero_10", "task_id": task_id, "success": False, "producer": producer}
                    for _ in range(failures)
                )
                rows.append({"suite": "libero_10", "task_id": task_id, "success": True, "producer": producer})
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        sources[name] = {
            "manifest": str(manifest),
            "sha256": _digest(manifest),
            "producer": producer,
            "producer_field": "producer",
        }
    checkpoints = []
    for alias in cp.ALIASES:
        step = int(alias[:-1]) * 1000
        saved = step - 1
        jax_dir = tmp_path / "checkpoints" / str(saved)
        jax_dir.mkdir(parents=True)
        identity_file = jax_dir / "checkpoint_identity.json"
        identity_file.write_text(
            json.dumps(
                {
                    "alias": alias,
                    "train_state_step": step,
                    "saved_step": saved,
                    "jax_dir": str(jax_dir),
                    "checkpoint_digest": f"digest-{alias}",
                }
            ),
            encoding="utf-8",
        )
        checkpoints.append(
            {
                "alias": alias,
                "train_state_step": step,
                "saved_step": saved,
                "jax_dir": str(jax_dir),
                "pytorch_dir": str(tmp_path / "pytorch" / alias),
                "identity_file": str(identity_file),
            }
        )
    selection = tmp_path / "topk.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "consumer": "pi",
                "positive_top_k": 16,
                "negative_top_k": 32,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "experiment.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "openpi_root": str(root),
                "artifact_root": str(tmp_path / "artifacts"),
                "run_root": str(tmp_path / "runs"),
                "sources": sources,
                "failure_source": "N",
                "fixed_prior_top_k": 8,
                "topk_selection": {"path": str(selection), "sha256": _digest(selection)},
                "checkpoints": checkpoints,
                "evaluation": {"seed": 7, "episodes_per_task": 100, "batch_size": 8},
            }
        ),
        encoding="utf-8",
    )
    return cp.load_experiment(manifest)


def test_profiles_have_expected_eval_command_counts(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    anchors = cp.build_dag(experiment, cp.PROFILES["anchors"], "all")
    full = cp.build_dag(experiment, cp.PROFILES["full"], "all")
    assert sum(node.kind == "eval" for node in anchors) == 30
    assert sum(node.kind == "eval" for node in full) == 70


def test_dag_orders_checkpoint_aligned_artifacts(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    nodes = cp.build_dag(experiment, ("6k",), "all")
    cp.validate_dag(nodes, phase="all")
    by_id = {node.node_id: node for node in nodes}
    assert by_id["6k:head"].dependencies == ("6k:features:B",)
    assert set(by_id["6k:bank:N"].dependencies) == {"6k:features:N", "6k:head"}
    assert set(by_id["6k:eval:v0_success_failure:S"].dependencies) == {"6k:bank:S", "6k:bank:N"}
    assert "/6k/" in str(by_id["6k:features:S"].identity_path)
    prior_env = dict(by_id["6k:eval:prior:B"].env)
    positive_env = dict(by_id["6k:eval:v0_success:B"].env)
    both_env = dict(by_id["6k:eval:v0_success_failure:B"].env)
    assert prior_env["MEMORY_TOP_K"] == "8"
    assert positive_env["MEMORY_TOP_K"] == "16"
    assert both_env["MEMORY_TOP_K"] == "16"
    assert both_env["NEGATIVE_MEMORY_TOP_K"] == "32"
    assert by_id["6k:eval:v0_success_failure:B"].identity["top_k"] == {
        "positive": 16,
        "negative": 32,
        "prior": None,
    }


def test_failure_gate_rejects_fewer_than_ten_per_task(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path, failures=9)
    with pytest.raises(cp.PreflightError, match="at least 10"):
        cp.preflight(experiment, ("6k",), require_failures=True)
    cp.preflight(experiment, ("6k",), require_failures=False)


def test_off_by_one_checkpoint_mapping_is_rejected(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    bad = dataclasses.replace(experiment.checkpoints["6k"], saved_step=6000)
    with pytest.raises(cp.PreflightError, match="saved loop step 5999"):
        cp.audit_checkpoint(bad)
