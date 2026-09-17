from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from openpi.experiments import topk_sweep as sweep


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _touch(path: Path, content: bytes = b"fixed") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _fixture(tmp_path: Path) -> sweep.Experiment:
    root = tmp_path / "openpi"
    root.mkdir()
    natural = tmp_path / "natural.jsonl"
    rows = [
        {
            "action_id": f"{task_id}-{success}-{index}",
            "episode_idx": index,
            "prompt": f"task {task_id}",
            "producer": "pi0.5_base",
            "source_family": "new_pi",
            "source_format": "eval_hdf5",
            "source_model": "pi0.5",
            "source_run": "fixed",
            "success": success,
            "suite": "libero_10",
            "task_id": task_id,
            "trajectory_path": f"trajectory-{task_id}-{success}-{index}.hdf5",
        }
        for task_id in sweep.TASK_IDS
        for success, count in ((True, 50), (False, 5))
        for index in range(count)
    ]
    natural.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    policy = tmp_path / "policy"
    model = policy / "model.safetensors"
    head = tmp_path / "head.pt"
    _touch(model, b"model")
    _touch(head, b"head")
    quantity = tmp_path / "quantity.json"
    quantity.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "pi_quantity",
                "policy_family": "pi",
                "openpi_root": str(root),
                "artifact_root": str(tmp_path / "quantity_artifacts"),
                "run_root": str(tmp_path / "quantity_runs"),
                "python": "python",
                "pi_include_failure_50": False,
                "natural_manifest": {
                    "path": str(natural),
                    "sha256": _digest(natural),
                    "provenance": {
                        "producer": "pi0.5_base",
                        "source_family": "new_pi",
                        "source_model": "pi0.5",
                        "source_run": "fixed",
                    },
                },
                "policy": {
                    "directory": str(policy),
                    "model_sha256": _digest(model),
                    "task_head_checkpoint": str(head),
                    "task_head_sha256": _digest(head),
                },
                "evaluation": {"seed": 7, "episodes_per_task": 100, "batch_size": 8},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "sweep.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "memory_quantity_manifest": {"path": str(quantity), "sha256": _digest(quantity)},
                "run_root": str(tmp_path / "sweep_runs"),
                "python": "python",
            }
        ),
        encoding="utf-8",
    )
    experiment = sweep.load_experiment(manifest)
    for success, failure, outcome in (
        (1, 1, "positive"),
        (10, 1, "positive"),
        (50, 1, "positive"),
        (50, 1, "negative"),
        (50, 5, "negative"),
        (50, "max", "negative"),
    ):
        bank = sweep._quantity_bank(experiment, success, failure, outcome)  # noqa: SLF001
        for path in sweep._bank_files(bank, outcome):  # noqa: SLF001
            _touch(path, f"{success}-{failure}-{outcome}-{path.name}".encode())
    return experiment


def _write_eval(node: sweep.Node, successes: int) -> None:
    node.log_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    index = 0
    for task_id in sweep.TASK_IDS:
        for episode_idx in range(sweep.EPISODE_START, sweep.EPISODE_START + sweep.EPISODES_PER_TASK):
            records.append(
                json.dumps(
                    {
                        "event": "episode_result",
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "success": index < successes,
                    }
                )
            )
            index += 1
    node.log_path.write_text("\n".join(records) + "\n", encoding="utf-8")


def test_candidate_shapes_and_fixed_controls(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    positive = sweep.positive_nodes(experiment)
    assert len(positive) == 6
    assert {(node.success_count, node.positive_k) for node in positive} == set(sweep.POSITIVE_CANDIDATES)
    assert all(
        node.method == "guidance_negative" and (node.failure_count, node.negative_k) == (5, 4) for node in positive
    )

    negative = sweep.negative_nodes(experiment, 10, 8)
    assert len(negative) == 6
    assert {(node.failure_count, node.negative_k) for node in negative} == set(sweep.NEGATIVE_CANDIDATES)
    assert all(
        node.method == "guidance_negative" and (node.success_count, node.positive_k) == (10, 8) for node in negative
    )


def test_preflight_requires_only_five_failures_per_task(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    audit = sweep.preflight(experiment)
    assert min(audit["quantity"]["counts"]["failure"].values()) == 5
    assert set(audit["banks"]) == {
        "positive/1",
        "positive/10",
        "positive/50",
        "negative/1",
        "negative/5",
        "negative/max",
    }


def test_selector_prefers_smaller_count_then_k_within_half_point(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    successes = {
        (1, 8): 399,
        (10, 1): 401,
        (10, 8): 402,
        (50, 8): 401,
        (50, 16): 400,
        (50, 32): 399,
    }
    for node in sweep.positive_nodes(experiment):
        _write_eval(node, successes[(node.success_count, node.positive_k)])
    selection = sweep.write_positive_selection(experiment)
    assert (selection["positive_memory_per_task"], selection["positive_top_k"]) == (10, 1)
    assert len(selection["candidates"]) == 6
    assert len(selection["candidates_sha256"]) == 64
    assert all(row["method"] == "guidance_negative" for row in selection["candidates"])


def test_exact_500_episode_pairing_is_a_hard_gate(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    node = sweep.positive_nodes(experiment)[0]
    _write_eval(node, 400)
    lines = node.log_path.read_text(encoding="utf-8").splitlines()
    node.log_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(sweep.PreflightError, match="exactly 500 paired episodes"):
        sweep._load_paired_log(node.log_path)  # noqa: SLF001


def test_multisuite_eval_digest_hashes_files_not_directory(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    experiment = dataclasses.replace(
        experiment,
        quantity=dataclasses.replace(experiment.quantity, suites=("libero_spatial", "libero_10")),
    )
    eval_dir = tmp_path / "multi_eval"
    eval_dir.mkdir()
    (eval_dir / "libero_spatial.jsonl").write_text("spatial\n", encoding="utf-8")
    (eval_dir / "libero_10.jsonl").write_text("ten\n", encoding="utf-8")

    first = sweep._eval_log_sha256(eval_dir, experiment)  # noqa: SLF001
    (eval_dir / "libero_10.jsonl").write_text("changed\n", encoding="utf-8")
    second = sweep._eval_log_sha256(eval_dir, experiment)  # noqa: SLF001

    assert len(first) == 64
    assert first != second


def test_adoption_requires_exact_semantic_identity(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    node = sweep.positive_nodes(experiment)[0]
    adopted = tmp_path / "old" / "eval" / "libero_10.jsonl"
    adopted.parent.mkdir(parents=True)
    adopted.write_text("unused\n", encoding="utf-8")
    identity = adopted.parent.parent / "orchestrator.identity.json"
    identity.write_text(json.dumps(node.identity), encoding="utf-8")
    experiment = dataclasses.replace(experiment, adopt_logs={node.node_id: adopted})
    assert sweep.resolve_log(experiment, node) == adopted
    identity.write_text(json.dumps({"experiment_identity_sha256": "different"}), encoding="utf-8")
    with pytest.raises(sweep.PreflightError, match="cannot adopt"):
        sweep.resolve_log(experiment, node)


def test_final_selection_contains_counts_topk_and_candidate_hashes(tmp_path: Path) -> None:
    experiment = _fixture(tmp_path)
    positive_successes = (400, 401, 404, 402, 401, 400)
    for node, successes in zip(sweep.positive_nodes(experiment), positive_successes, strict=True):
        _write_eval(node, successes)
    positive = sweep.write_positive_selection(experiment)
    assert (positive["positive_memory_per_task"], positive["positive_top_k"]) == (10, 8)

    nodes = sweep.negative_nodes(experiment, 10, 8)
    for node, successes in zip(nodes, (400, 404, 403, 402, 401, 400), strict=True):
        _write_eval(node, successes)
    final = sweep.write_final_selection(experiment, positive)
    assert final["consumer"] == "pi"
    assert (final["positive_memory_per_task"], final["positive_top_k"]) == (10, 8)
    assert (final["negative_memory_per_task"], final["negative_top_k"]) == (1, 4)
    assert len(final["positive_candidates"]) == 6
    assert len(final["negative_candidates"]) == 6
    assert len(final["positive_candidates_sha256"]) == 64
    assert len(final["negative_candidates_sha256"]) == 64
