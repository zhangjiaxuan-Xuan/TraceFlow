#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = OPENPI_ROOT.parent
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from run_topk_campaign import Job  # noqa: E402
from run_topk_campaign import PoolScheduler  # noqa: E402

from openpi.experiments import cross_policy_long_eval as protocol  # noqa: E402


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _audit_output(condition: protocol.Condition) -> protocol.Audit:
    return protocol.audit_log(condition, condition.output_log, source="output")


def _job(condition: protocol.Condition) -> Job:
    env = {
        "RUN_ROOT": str(condition.output_root),
        "OUTPUT": str(condition.output_root),
        "OUTPUT_LOG": str(condition.output_log),
        "SUITE": condition.suite,
        "TASK_SUITE": condition.suite,
        "CONSUMER": condition.consumer,
        "EVAL_MODE": condition.mode,
        "SEED": str(condition.seed),
        "EPISODE_START": str(condition.episode_start),
        "EPISODES_PER_TASK": str(condition.episodes_per_task),
        "NUM_TRIALS_PER_TASK": str(condition.episodes_per_task),
        "BATCH_SIZE": "8",
        "INFERENCE_BATCH_SIZE": "8",
        "SAVE_VIDEOS": "0",
        "SAVE_EPISODE_DATA": "0",
    }
    return Job(
        job_id=condition.condition_id,
        stage="cross-policy-long",
        command=protocol.expand_command(condition),
        cwd=condition.cwd,
        env=tuple(sorted(env.items())),
        identity=condition.expected_identity,
        completion_path=condition.output_root / "condition.identity.json",
        completion_check=lambda condition=condition: _audit_output(condition),
    )


def _preflight_output(condition: protocol.Condition, *, resume: bool) -> None:
    if not condition.output_root.exists():
        return
    entries = list(condition.output_root.iterdir())
    if not entries:
        return
    completion = condition.output_root / "condition.identity.json"
    if completion.is_file():
        return
    if not resume:
        raise protocol.AuditError(f"refusing non-resume run into non-empty output_root: {condition.output_root}")


def plan(experiment: protocol.Experiment) -> tuple[list[protocol.Audit], list[protocol.Condition]]:
    adopted: list[protocol.Audit] = []
    gaps: list[protocol.Condition] = []
    for condition in experiment.conditions:
        audit = protocol.audit_existing(condition)
        if audit is None:
            _preflight_output(condition, resume=experiment.resume)
            gaps.append(condition)
        else:
            adopted.append(audit)
    return adopted, gaps


def run(experiment: protocol.Experiment, *, dry_run: bool) -> dict[str, Any]:
    adopted, gaps = plan(experiment)
    experiment.run_root.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema_version": 1,
        "protocol": "pi_smol_libero_10_90_v1",
        "status": "dry-run" if dry_run else "running",
        "gpu_pool": list(experiment.gpu_pool),
        "ports": list(experiment.ports),
        "adopted": [audit.condition_id for audit in adopted],
        "gaps": [
            {
                "condition_id": condition.condition_id,
                "consumer": condition.consumer,
                "mode": condition.mode,
                "suite": condition.suite,
                "command": list(protocol.expand_command(condition)),
                "output_root": str(condition.output_root),
            }
            for condition in gaps
        ],
    }
    _atomic_json(experiment.run_root / "plan.json", plan_payload)
    if dry_run:
        protocol.write_reports(experiment.run_root, adopted, gaps)
        return plan_payload

    scheduler = PoolScheduler(
        gpu_pool=experiment.gpu_pool,
        ports=experiment.ports,
        log_root=experiment.run_root / "stdout",
        fail_fast=experiment.fail_fast,
        resume=experiment.resume,
    )
    if gaps:
        scheduler.run_stage("cross-policy-long", [_job(condition) for condition in gaps])
    completed = [_audit_output(condition) for condition in gaps]
    audits = adopted + completed
    protocol.write_reports(experiment.run_root, audits, [])
    result = {
        **plan_payload,
        "status": "completed",
        "results": [dataclasses.asdict(audit) for audit in audits],
    }
    _atomic_json(experiment.run_root / "campaign_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and fill explicit Pi/Smol LIBERO-10/90 result gaps.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment = protocol.load_experiment(args.manifest)
    result = run(experiment, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
