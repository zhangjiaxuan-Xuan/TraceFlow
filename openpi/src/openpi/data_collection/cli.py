"""CLI for safe outcome-deficit collection planning and manifests."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

from openpi.data_collection.outcome_deficit import CollectionSpec
from openpi.data_collection.outcome_deficit import OutcomeQuota
from openpi.data_collection.outcome_deficit import atomic_write_jsonl
from openpi.data_collection.outcome_deficit import audit_records
from openpi.data_collection.outcome_deficit import build_plan
from openpi.data_collection.outcome_deficit import derive_quota_subset
from openpi.data_collection.outcome_deficit import expand_suites
from openpi.data_collection.outcome_deficit import merge_records
from openpi.data_collection.outcome_deficit import read_jsonl
from openpi.data_collection.outcome_deficit import select_collection_group_failures
from openpi.data_collection.outcome_deficit import select_collection_group_records
from openpi.data_collection.outcome_deficit import sha256_file


def _spec(args: argparse.Namespace) -> CollectionSpec:
    suites = expand_suites(args.suites)
    if "libero_90" in suites:
        raise ValueError("LIBERO-90 is excluded from the current experiment design")
    if args.producer == "pi0.5-libero":
        allowed_pi_suites = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        unsupported = set(suites).difference(allowed_pi_suites)
        if unsupported:
            raise ValueError(f"The Pi collector supports only the four 10-task suites, got: {sorted(unsupported)}")
    return CollectionSpec(
        producer=args.producer,
        checkpoint_digest=args.checkpoint_digest,
        suites=suites,
        quota=OutcomeQuota(args.success_quota, args.failure_quota),
        max_attempts_per_task=args.max_attempts_per_task,
        batch_attempts=args.batch_attempts,
        base_seed=args.seed,
        action_contract=args.action_contract,
        complete_suite_rounds=args.complete_suite_rounds,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--producer", choices=("pi0.5-libero", "smolvla"), required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--success-quota", type=int, required=True)
    parser.add_argument("--failure-quota", type=int, required=True)
    parser.add_argument("--max-attempts-per-task", type=int, required=True)
    parser.add_argument("--batch-attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-contract", default="libero-relative-ee-7d-v1")
    parser.add_argument("--collection-group", default="")
    parser.add_argument(
        "--complete-suite-rounds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Finish an equal attempt group for every task before stopping a suite.",
    )


def _manifest(root: Path) -> Path:
    return root / "natural_attempts.jsonl"


def _job_id(producer: str, suite: str, task_id: int, start: int, count: int) -> str:
    raw = f"{producer}:{suite}:{task_id}:{start}:{count}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _format_command(args: argparse.Namespace, spec: CollectionSpec, job: Any) -> str:
    job_id = _job_id(spec.producer, job.suite, job.task_id, job.attempt_start, job.attempt_count)
    run_root = (args.root / "runs" / job.suite / f"task_{job.task_id:03d}" / job_id).resolve()
    values = {
        "suite": job.suite,
        "task_id": job.task_id,
        "attempt_start": job.attempt_start,
        "attempt_count": job.attempt_count,
        "seed": job.seed,
        "run_root": str(run_root),
        "output_dir": str(run_root),
        "checkpoint": args.checkpoint,
        "checkpoint_digest": spec.checkpoint_digest,
        "collection_group": getattr(args, "collection_group", ""),
        "manifest": str(_manifest(args.root).resolve()),
        "gpu": args.gpu,
    }
    if spec.producer == "pi0.5-libero":
        suite_log = run_root / "eval" / f"{job.suite}.jsonl"
        collect = " ".join(
            [
                f"mkdir -p {shlex.quote(str(run_root / 'eval'))}",
                "&&",
                f"touch {shlex.quote(str(suite_log))}",
                "&&",
                f"GPU={shlex.quote(args.gpu)}",
                f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)}",
                f"SERVER_CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)}",
                f"CLIENT_CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)}",
                "OPENPI_DATA_HOME=/path/to/local/data/openpi",
                "MUJOCO_EGL_DEVICE_ID=0",
                f"LIBERO_CONFIG_PATH={shlex.quote(str(run_root / 'libero_config'))}",
                f"NUMBA_CACHE_DIR={shlex.quote(str(run_root / 'cache' / 'numba'))}",
                f"MPLCONFIGDIR={shlex.quote(str(run_root / 'cache' / 'matplotlib'))}",
                f"POLICY_DIR={shlex.quote(args.checkpoint)}",
                f"LOG_DIR={shlex.quote(str(run_root / 'eval'))}",
                f"RESULTS_TXT={shlex.quote(str(run_root / 'eval' / 'results.txt'))}",
                f"VIDEO_ROOT={shlex.quote(str(run_root / 'videos'))}",
                f"EPISODE_DATA_ROOT={shlex.quote(str(run_root / 'episode_data'))}",
                "EPISODE_DATA_MODE=all",
                f"TASK_IDS_CSV={job.task_id}",
                f"NUM_TRIALS_PER_TASK={job.attempt_count}",
                f"EPISODE_START={job.attempt_start}",
                f"SEED={job.seed}",
                f"INFERENCE_BATCH_SIZE={min(8, job.attempt_count)}",
                "INFERENCE_BATCH_WAIT_MS=20",
                f"LIBERO_CLIENTS_PER_SUITE={min(8, job.attempt_count)}",
                "SAVE_VIDEOS=0",
                "USE_MEMORY=0",
                "USE_LCM=0",
                "MEMORY_GUIDANCE_ONLY=0",
                "USE_NEGATIVE_GUIDANCE=0",
                "EXTERNAL_POLICY_SERVER=1" if getattr(args, "external_policy_server", False) else "",
                f"SERVER_LOG_PATH={shlex.quote(args.server_log_path)}"
                if getattr(args, "external_policy_server", False)
                else "",
                f"PORT={int(args.port)}" if getattr(args, "external_policy_server", False) else "",
                "RESUME=1",
                f"RUN_ID={job_id}",
                f"bash scripts/eval/run_libero_eval.sh {shlex.quote(job.suite)}",
            ]
        )
        ingest = " ".join(
            [
                shlex.quote(args.python),
                "scripts/data/outcome_deficit.py ingest",
                f"--root {shlex.quote(str(args.root))}",
                "--producer pi0.5-libero",
                f"--checkpoint {shlex.quote(args.checkpoint)}",
                f"--checkpoint-digest {shlex.quote(spec.checkpoint_digest)}",
                f"--source-manifest {shlex.quote(str(suite_log))}",
                f"--source-run {job_id}",
                f"--collection-group {shlex.quote(args.collection_group)}"
                if getattr(args, "collection_group", "")
                else "",
                f"--seed {job.seed}",
                f"--action-contract {shlex.quote(spec.action_contract)}",
            ]
        )
        return f"{collect} && {ingest}"
    if not args.runner_template:
        raise ValueError("Smol command generation requires --runner-template or SMOL_COLLECTOR_TEMPLATE")
    required = {"suite", "task_id", "attempt_count", "seed", "output_dir"}
    missing = sorted(name for name in required if "{" + name + "}" not in args.runner_template)
    if missing:
        raise ValueError(f"Smol runner template is missing placeholders: {missing}")
    return args.runner_template.format(**values)


def _gpu_pool(args: argparse.Namespace) -> tuple[str, ...]:
    raw = str(getattr(args, "gpus", "")).strip()
    values = tuple(value.strip() for value in raw.split(",") if value.strip()) if raw else (str(args.gpu),)
    if not values or len(values) != len(set(values)):
        raise ValueError("--gpus must contain unique comma-separated GPU identifiers")
    return values


def _commands_by_gpu(args: argparse.Namespace, spec: CollectionSpec, jobs: list[Any]) -> dict[str, list[str]]:
    queues = {gpu: [] for gpu in _gpu_pool(args)}
    gpus = tuple(queues)
    for index, job in enumerate(jobs):
        gpu = gpus[index % len(gpus)]
        job_args = argparse.Namespace(**vars(args))
        job_args.gpu = gpu
        queues[gpu].append(_format_command(job_args, spec, job))
    return queues


def _run_gpu_queue(commands: list[str]) -> None:
    for command in commands:
        subprocess.run(command, shell=True, executable="/bin/bash", check=True)


def _plan(args: argparse.Namespace) -> int:
    spec = _spec(args)
    plan_path = args.root / "plan.json"
    round_index = 0
    while True:
        records = read_jsonl(_manifest(args.root))
        if args.collection_group:
            records = [row for row in records if row.get("collection_group") == args.collection_group]
        jobs = build_plan(spec, records)
        payload = {
            "schema": "optimus-outcome-plan-v1",
            "round": round_index,
            "collection_group": args.collection_group,
            "spec": dataclasses.asdict(spec),
            "jobs": [dataclasses.asdict(job) for job in jobs],
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pending = [job for job in jobs if job.status == "pending"]
        incomplete = [job for job in jobs if job.status == "incomplete"]
        print(
            json.dumps(
                {
                    "plan": str(plan_path),
                    "round": round_index,
                    "records": len(records),
                    "pending": len(pending),
                    "incomplete": len(incomplete),
                },
                sort_keys=True,
            )
        )
        queues = _commands_by_gpu(args, spec, pending)
        commands = [command for gpu_commands in queues.values() for command in gpu_commands]
        if args.emit_commands:
            for command in commands:
                print(command)
        if incomplete:
            return 2
        if not pending:
            subset, summary = derive_quota_subset(spec, records)
            atomic_write_jsonl(args.root / "quota_subset.jsonl", subset)
            (args.root / "quota_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 0
        if not args.execute:
            return 0
        records_before = len(records)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queues)) as executor:
            futures = [executor.submit(_run_gpu_queue, gpu_commands) for gpu_commands in queues.values()]
            for future in futures:
                future.result()
        records_after = len(read_jsonl(_manifest(args.root)))
        if records_after <= records_before:
            raise RuntimeError(
                "Collection commands completed but natural_attempts.jsonl did not grow; "
                "check the collector's episode manifest and ingest adapter"
            )
        round_index += 1


def _canonicalize(args: argparse.Namespace, source: dict[str, Any]) -> dict[str, Any]:
    suite = str(source.get("suite", source.get("task_suite", "")))
    task_id = int(source["task_id"])
    episode_idx = int(source.get("episode_idx", source.get("episode_index", source.get("episode_ix"))))
    raw_success = source.get("success")
    if isinstance(raw_success, bool):
        success = raw_success
    elif raw_success in {0, 1}:
        success = bool(raw_success)
    else:
        raise ValueError(f"Source success must be boolean, got {raw_success!r}")
    path_value = source.get("trajectory_path", source.get("path"))
    if not path_value:
        raise ValueError("Source record has no trajectory_path/path")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory: {path}")
    seed = int(source.get("seed", args.seed + task_id * 1_000_000 + episode_idx))
    identity = [args.producer, args.checkpoint_digest, suite, task_id, seed, episode_idx]
    trajectory_id = "traj_" + hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:24]
    source_format = str(source.get("source_format", ""))
    prompt = str(source.get("prompt", ""))
    trajectory_length = int(source.get("trajectory_length", source.get("env_steps", 0)))
    if path.suffix.lower() in {".h5", ".hdf5"}:
        source_format = source_format or "eval_hdf5"
        import h5py  # noqa: PLC0415

        with h5py.File(path, "r") as handle:
            demo = handle["data/demo_0"]
            prompt = prompt or str(demo.attrs.get("language_instruction", ""))
            trajectory_length = int(demo["actions"].shape[0])
    elif path.suffix.lower() == ".npz":
        source_format = source_format or "smol_npz"
    if source_format not in {"eval_hdf5", "smol_npz"}:
        raise ValueError(f"Unsupported collected trajectory format for {path}: {source_format!r}")
    if not prompt:
        raise ValueError(f"Collected trajectory has no language prompt: {path}")
    if trajectory_length <= 0:
        raise ValueError(f"Collected trajectory is empty: {path}")
    result = {
        "schema": "optimus-outcome-attempt-v1",
        "trajectory_id": trajectory_id,
        "producer": args.producer,
        "checkpoint": args.checkpoint,
        "checkpoint_digest": args.checkpoint_digest,
        "suite": suite,
        "task_key": f"{suite}/task_{task_id:03d}",
        "task_id": task_id,
        "seed": seed,
        "episode_idx": episode_idx,
        "outcome": "success" if success else "failure",
        "success": success,
        "action_id": trajectory_id,
        "trajectory_path": str(path),
        "source_format": source_format,
        "frame_index": 0,
        "prompt": prompt,
        "trajectory_length": trajectory_length,
        "image_transform": "flip_height_width_then_policy_preprocess",
        "split": "all",
        "trajectory_sha256": sha256_file(path),
        "action_contract": args.action_contract,
        "source_run": args.source_run,
        "source_record": source,
    }
    if getattr(args, "collection_group", ""):
        result["collection_group"] = args.collection_group
    return result


def _ingest(args: argparse.Namespace) -> int:
    source_rows = []
    with args.source_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {args.source_manifest}:{line_number}") from exc
            if row.get("event") not in {None, "episode_result"}:
                continue
            if row.get("trajectory_path", row.get("path")):
                source_rows.append(row)
    if not source_rows:
        raise ValueError(f"No trajectory-bearing episode records in {args.source_manifest}")
    incoming = [_canonicalize(args, row) for row in source_rows]
    destination = _manifest(args.root)
    lock_path = args.root / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            merged = merge_records(read_jsonl(destination), incoming)
            atomic_write_jsonl(destination, merged)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    print(json.dumps({"manifest": str(destination), "ingested": len(incoming), "total": len(merged)}, sort_keys=True))
    return 0


def _audit(args: argparse.Namespace) -> int:
    spec = _spec(args)
    records = read_jsonl(_manifest(args.root))
    if args.collection_group:
        records = [row for row in records if row.get("collection_group") == args.collection_group]
    audit = audit_records(records, verify_files=args.verify_files)
    subset, quota = derive_quota_subset(spec, records)
    atomic_write_jsonl(args.root / "quota_subset.jsonl", subset)
    report = {"audit": audit, "quota": quota}
    (args.root / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if audit["status"] == "ok" and quota["status"] == "complete" else 2


def _select_group_failures(args: argparse.Namespace) -> int:
    source = args.source or _manifest(args.root)
    records = read_jsonl(source)
    selected, summary = select_collection_group_failures(records, args.collection_group)
    output = args.output or (args.root / f"{args.collection_group}_all_failures.jsonl")
    atomic_write_jsonl(output, selected)
    summary.update(
        {
            "source_manifest": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "output_manifest": str(output.resolve()),
        }
    )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _materialize_group(args: argparse.Namespace) -> int:
    source = args.source or _manifest(args.root)
    records = read_jsonl(source)
    output_dir = args.output_dir or (args.root / "capacity_manifests")
    source_digest = sha256_file(source)
    outputs: dict[str, Any] = {}
    # C successes remain in the immutable natural manifest for later studies.
    # The current top-k/capacity ablation uses B6500 for positive memory and
    # materializes only C failure capacities here.
    failure_capacities = (1, 5, None) if args.collection_group == "C-pi" else (1, 5, 50)
    for outcome, capacities in (("failure", failure_capacities),):
        for capacity in capacities:
            selected, summary = select_collection_group_records(
                records,
                args.collection_group,
                outcome,
                capacity,
                fallback_to_max=args.collection_group == "C-pi",
            )
            label = "max" if capacity is None else str(capacity)
            if summary["tasks"] != args.expected_tasks:
                raise ValueError(
                    f"{args.collection_group} {outcome}_{label} covers {summary['tasks']} tasks; "
                    f"expected {args.expected_tasks}"
                )
            if capacity is not None and args.collection_group != "C-pi":
                short = {task: count for task, count in summary["counts_by_task"].items() if count < capacity}
                if short:
                    raise ValueError(f"Insufficient records for {outcome}_{label}: {short}")
            output = output_dir / f"{outcome}_{label}.jsonl"
            atomic_write_jsonl(output, selected)
            summary.update(
                {
                    "source_manifest": str(source.resolve()),
                    "source_sha256": source_digest,
                    "output_manifest": str(output.resolve()),
                }
            )
            output.with_suffix(".summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            outputs[f"{outcome}_{label}"] = summary
    aggregate = {
        "schema": "optimus-isolated-group-capacities-v1",
        "collection_group": args.collection_group,
        "source_manifest": str(source.resolve()),
        "source_sha256": source_digest,
        "outputs": outputs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    _common(plan)
    plan.add_argument("--checkpoint", required=True)
    plan.add_argument("--python", default="/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python")
    plan.add_argument("--gpu", default="0")
    plan.add_argument("--gpus", default="", help="Comma-separated GPU pool; one serial queue is created per GPU")
    plan.add_argument("--emit-commands", action=argparse.BooleanOptionalAction, default=True)
    plan.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Execute planned jobs serially and re-plan until quotas are complete or attempt caps are reached.",
    )
    plan.add_argument("--runner-template", default="")
    plan.add_argument(
        "--external-policy-server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Connect Pi collection jobs to one server managed by the caller.",
    )
    plan.add_argument("--port", type=int, default=8300)
    plan.add_argument("--server-log-path", default="")
    plan.set_defaults(handler=_plan)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--root", type=Path, required=True)
    ingest.add_argument("--producer", choices=("pi0.5-libero", "smolvla"), required=True)
    ingest.add_argument("--checkpoint", required=True)
    ingest.add_argument("--checkpoint-digest", required=True)
    ingest.add_argument("--source-manifest", type=Path, required=True)
    ingest.add_argument("--source-run", required=True)
    ingest.add_argument("--collection-group", default="")
    ingest.add_argument("--seed", type=int, default=7)
    ingest.add_argument("--action-contract", default="libero-relative-ee-7d-v1")
    ingest.set_defaults(handler=_ingest)

    audit = subparsers.add_parser("audit")
    _common(audit)
    audit.add_argument("--verify-files", action=argparse.BooleanOptionalAction, default=True)
    audit.set_defaults(handler=_audit)

    select_group = subparsers.add_parser("select-group-failures")
    select_group.add_argument("--root", type=Path, required=True)
    select_group.add_argument("--collection-group", choices=("C-pi",), required=True)
    select_group.add_argument("--source", type=Path)
    select_group.add_argument("--output", type=Path)
    select_group.set_defaults(handler=_select_group_failures)

    materialize_group = subparsers.add_parser("materialize-group")
    materialize_group.add_argument("--root", type=Path, required=True)
    materialize_group.add_argument("--collection-group", choices=("C-pi", "C-smol"), required=True)
    materialize_group.add_argument("--source", type=Path)
    materialize_group.add_argument("--output-dir", type=Path)
    materialize_group.add_argument("--expected-tasks", type=int, default=40)
    materialize_group.set_defaults(handler=_materialize_group)
    args = parser.parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
