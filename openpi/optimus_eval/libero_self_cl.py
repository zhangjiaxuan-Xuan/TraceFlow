from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import h5py


BRANCH_ADMISSION = {
    "success_only": "success",
    "failure_only": "failure",
    "all": "both",
}
EXPECTED_TASKS = tuple(range(10))


def absolute_executable_path(path: Path) -> Path:
    """Keep a venv launcher path intact; resolving its symlink disables the venv."""
    return Path(os.path.abspath(path.expanduser()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def evaluation_complete(
    eval_log: Path,
    episodes_per_task: int,
    task_ids: tuple[int, ...] = EXPECTED_TASKS,
) -> bool:
    if not eval_log.is_file():
        return False
    records = [row for row in _read_jsonl(eval_log) if row.get("event") == "episode_result"]
    identities = {(int(row["task_id"]), int(row["episode_idx"])) for row in records}
    expected = {
        (task_id, episode_idx)
        for task_id in task_ids
        for episode_idx in range(episodes_per_task)
    }
    return (
        identities == expected
        and len(records) == len(expected)
        and all(row.get("task_suite") == "libero_10" for row in records)
        and not any(row.get("error") for row in records)
    )


def materialize_round_manifest(
    *,
    eval_log: Path,
    output: Path,
    branch: str,
    round_index: int,
    episodes_per_task: int,
    task_ids: tuple[int, ...] = EXPECTED_TASKS,
) -> list[dict[str, Any]]:
    if not evaluation_complete(eval_log, episodes_per_task, task_ids):
        raise RuntimeError(f"Evaluation is incomplete: {eval_log}")
    records = [row for row in _read_jsonl(eval_log) if row.get("event") == "episode_result"]
    rows = []
    for record in sorted(records, key=lambda row: (int(row["task_id"]), int(row["episode_idx"]))):
        trajectory = Path(str(record.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file():
            raise FileNotFoundError(f"Missing trajectory for completed episode: {trajectory}")
        with h5py.File(trajectory, "r") as handle:
            demo = handle["data/demo_0"]
            length = int(demo["actions"].shape[0])
            prompt = str(demo.attrs["language_instruction"])
            stored_success = bool(demo.attrs["success"])
        if length <= 0 or stored_success != bool(record["success"]):
            raise RuntimeError(f"Invalid trajectory identity/outcome: {trajectory}")
        task_id = int(record["task_id"])
        episode_idx = int(record["episode_idx"])
        rows.append(
            {
                "action_id": f"selfcl_{branch}_r{round_index:02d}_t{task_id:02d}_e{episode_idx:03d}",
                "trajectory_path": str(trajectory),
                "source_format": "eval_hdf5",
                "success": bool(record["success"]),
                "prompt": prompt,
                "suite": "libero_10",
                "task_id": task_id,
                "episode_idx": episode_idx,
                "frame_index": 0,
                "global_frame_index": 0,
                "split": "all",
                "trajectory_length": length,
                "source_campaign": "pi_v1_self_cl_libero10",
                "source_branch": branch,
                "source_round": round_index,
                "source_seed": int(record["seed"]),
            }
        )
    _write_jsonl_atomic(output, rows)
    return rows


def cumulative_rows(
    *, branch: str, round_manifests: list[Path]
) -> list[dict[str, Any]]:
    admission = BRANCH_ADMISSION[branch]
    rows: list[dict[str, Any]] = []
    for manifest in round_manifests:
        for row in _read_jsonl(manifest):
            admitted = (
                admission == "both"
                or (admission == "success" and bool(row["success"]))
                or (admission == "failure" and not bool(row["success"]))
            )
            if admitted:
                rows.append(row)
    ids = [str(row["action_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate cumulative action IDs in branch {branch}")
    return rows


def _bank_counts(bank_root: Path | None) -> tuple[int, int]:
    if bank_root is None:
        return 0, 0
    artifacts = (
        "build_summary.json",
        "positive/gpm_memory_meta.pt",
        "positive/gpm_memory.index",
        "positive/gpm_memory_actions.npz",
        "negative/gpm_negative_memory_meta.pt",
        "negative/gpm_negative_memory.index",
        "negative/gpm_negative_memory_actions.npz",
    )
    for relative in artifacts:
        if not (bank_root / relative).is_file():
            raise FileNotFoundError(f"Incomplete memory bank: {bank_root / relative}")
    summary = json.loads((bank_root / "build_summary.json").read_text(encoding="utf-8"))
    output = summary["output"]
    return int(output["positive"]["items"]), int(output["negative"]["items"])


def _runtime_mode(bank_root: Path | None) -> str:
    positive, negative = _bank_counts(bank_root)
    if positive and negative:
        return "both"
    if positive:
        return "success"
    if negative:
        return "failure"
    return "base"


def _run(
    command: list[str], *, env: dict[str, str] | None = None, log_path: Path | None = None
) -> None:
    print("+ " + " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True, env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _openpi_environment(args: argparse.Namespace) -> dict[str, str]:
    roots = (
        args.openpi_root / "src",
        args.openpi_root / "packages/openpi-client/src",
        args.openpi_root / "third_party/libero",
    )
    value = ":".join(str(path) for path in roots)
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        value = f"{value}:{existing}"
    return {**os.environ, "OPENPI_ROOT": str(args.openpi_root), "PYTHONPATH": value}


def _ensure_campaign_config(args: argparse.Namespace) -> None:
    expected = {
        "schema_version": 1,
        "protocol": "pi_v1_self_cl_libero10_10round_v1",
        "branches": list(BRANCH_ADMISSION),
        "rounds": args.rounds,
        "task_ids": list(args.task_ids),
        "tasks": len(args.task_ids),
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "positive_top_k": 16,
        "negative_top_k": 8,
        "memory_capacity": "unbounded",
        "round_1_memory": "disabled",
        "policy_dir": str(args.policy_dir),
        "task_head_checkpoint": str(args.task_head_checkpoint),
    }
    path = args.output_root / "campaign_config.json"
    with (args.output_root / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            actual = json.loads(path.read_text(encoding="utf-8"))
            if actual != expected:
                raise RuntimeError(f"Campaign configuration mismatch at {path}")
        else:
            _write_json_atomic(path, expected)


def _write_branch_summary(branch_root: Path, branch: str, rounds: int) -> None:
    rows = []
    for round_index in range(1, rounds + 1):
        path = branch_root / f"round_{round_index:02d}" / "round_summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "branch": branch,
                "round": round_index,
                "memory_mode": summary["evaluation_memory_mode"],
                "successes": int(summary["round_outcomes"]["success"]),
                "failures": int(summary["round_outcomes"]["failure"]),
                "success_rate": float(summary["round_success_rate"]),
                "cumulative_success_memory": int(summary["cumulative_admitted"]["success"]),
                "cumulative_failure_memory": int(summary["cumulative_admitted"]["failure"]),
                "cumulative_memory": int(summary["cumulative_admitted"]["total"]),
            }
        )
    _write_json_atomic(branch_root / "branch_summary.json", {"branch": branch, "rounds": rows})
    tsv = branch_root / "branch_summary.tsv"
    temporary = tsv.with_name(f".{tsv.name}.{os.getpid()}.tmp")
    columns = list(rows[0]) if rows else ["branch", "round"]
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in rows:
            stream.write("\t".join(str(row[column]) for column in columns) + "\n")
    os.replace(temporary, tsv)


def _write_campaign_summary(output_root: Path) -> None:
    with (output_root / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        available = {}
        for branch in BRANCH_ADMISSION:
            path = output_root / "branches" / branch / "branch_summary.json"
            if path.is_file():
                available[branch] = json.loads(path.read_text(encoding="utf-8"))["rounds"]
        _write_json_atomic(output_root / "campaign_summary.json", {"branches": available})


def run_branch(args: argparse.Namespace, branch: str) -> None:
    branch_root = args.output_root / "branches" / branch
    round_manifests: list[Path] = []
    previous_bank: Path | None = None
    previous_manifest: Path | None = None
    previous_features: Path | None = None

    for round_index in range(1, args.rounds + 1):
        round_root = branch_root / f"round_{round_index:02d}"
        eval_log = round_root / "eval" / "libero_10.jsonl"
        mode = _runtime_mode(previous_bank)
        input_config = {
            "schema_version": 1,
            "branch": branch,
            "round": round_index,
            "memory_mode": mode,
            "input_bank": str(previous_bank) if previous_bank else "disabled",
            "input_bank_summary_sha256": (
                _digest(previous_bank / "build_summary.json") if previous_bank else ""
            ),
            "positive_top_k": 16,
            "negative_top_k": 8,
            "guidance": "V1 guidance-only fixed-10-NFE",
            "task_ids": list(args.task_ids),
            "episodes_per_task": args.episodes_per_task,
            "seed": args.seed,
        }
        input_config_path = round_root / "round_input.json"
        if input_config_path.exists():
            actual = json.loads(input_config_path.read_text(encoding="utf-8"))
            if actual != input_config:
                raise RuntimeError(f"Round input identity mismatch: {input_config_path}")
        else:
            _write_json_atomic(input_config_path, input_config)
        if not evaluation_complete(eval_log, args.episodes_per_task, args.task_ids):
            env = {
                **os.environ,
                "MODE": mode,
                "RUN_ROOT": str(round_root),
                "BANK_ROOT": str(previous_bank) if previous_bank else "",
                "GPU": str(args.gpu),
                "PORT": str(args.port),
                "SEED": str(args.seed),
                "EPISODES_PER_TASK": str(args.episodes_per_task),
                "TASK_IDS_CSV": ",".join(str(task_id) for task_id in args.task_ids),
                "MAX_ENV_STEPS": str(args.max_env_steps),
                "ENV_WORKERS": str(args.env_workers),
                "SHARDS_PER_TASK": str(args.shards_per_task),
                "INFERENCE_BATCH_SIZE": str(args.inference_batch_size),
                "FEATURE_IMAGE_SIZE": str(args.trajectory_image_size),
                "SAVE_VIDEOS": "1" if args.save_videos else "0",
                "POLICY_DIR": str(args.policy_dir),
                "TASK_HEAD_CKPT": str(args.task_head_checkpoint),
                "OPENPI_PYTHON": str(args.python),
                "LIBERO_PYTHON": str(args.libero_python),
                "ENVIRONMENT_ID_PREFIX": f"selfcl-r{round_index:02d}",
            }
            _run(
                ["bash", str(args.round_script)],
                env=env,
                log_path=round_root / "orchestrator.log",
            )

        round_manifest = round_root / "manifests" / "all_outcomes.jsonl"
        rows = materialize_round_manifest(
            eval_log=eval_log,
            output=round_manifest,
            branch=branch,
            round_index=round_index,
            episodes_per_task=args.episodes_per_task,
            task_ids=args.task_ids,
        )
        round_manifests.append(round_manifest)
        cumulative = cumulative_rows(branch=branch, round_manifests=round_manifests)
        cumulative_manifest = round_root / "manifests" / "cumulative_admitted.jsonl"
        _write_jsonl_atomic(cumulative_manifest, cumulative)

        outcomes = Counter(bool(row["success"]) for row in rows)
        cumulative_outcomes = Counter(bool(row["success"]) for row in cumulative)
        summary = {
            "branch": branch,
            "round": round_index,
            "evaluation_memory_mode": mode,
            "round_outcomes": {"success": outcomes[True], "failure": outcomes[False]},
            "round_success_rate": outcomes[True] / len(rows),
            "cumulative_admitted": {
                "success": cumulative_outcomes[True],
                "failure": cumulative_outcomes[False],
                "total": len(cumulative),
            },
            "round_manifest_sha256": _digest(round_manifest),
            "cumulative_manifest_sha256": _digest(cumulative_manifest),
        }

        if cumulative:
            features = round_root / "features"
            cache_command = [
                str(args.python),
                str(args.openpi_root / "scripts/memory/cache_prior_head_features.py"),
                "--manifest", str(cumulative_manifest),
                "--output-dir", str(features),
                "--policy-dir", str(args.policy_dir),
                "--config-name", "pi05_libero",
                "--device", "cuda",
                "--batch-size", str(args.feature_batch_size),
                "--min-batch-size", "1",
                "--auto-batch",
            ]
            if previous_manifest is not None and previous_features is not None:
                cache_command.extend(
                    ["--seed-manifest", str(previous_manifest), "--seed-cache-dir", str(previous_features)]
                )
            _run(cache_command, env=_openpi_environment(args), log_path=round_root / "cache.log")

            bank = round_root / "bank_after_round"
            if not (bank / "build_summary.json").is_file():
                _run(
                    [
                        str(args.python),
                        str(args.openpi_root / "scripts/memory/build_cl_memory_bank.py"),
                        "--group", f"selfcl_{branch}_r{round_index:02d}",
                        "--manifest", str(cumulative_manifest),
                        "--feature-dir", str(features),
                        "--checkpoint", str(args.task_head_checkpoint),
                        "--output-dir", str(bank),
                        "--admission", "both",
                        "--device", "cuda",
                        "--overwrite",
                    ],
                    env=_openpi_environment(args),
                    log_path=round_root / "bank.log",
                )
            positive, negative = _bank_counts(bank)
            bank_summary = json.loads((bank / "build_summary.json").read_text(encoding="utf-8"))
            if bank_summary.get("manifest_sha256") != _digest(cumulative_manifest):
                raise RuntimeError(f"Bank manifest identity mismatch: {bank}")
            if (positive, negative) != (cumulative_outcomes[True], cumulative_outcomes[False]):
                raise RuntimeError(
                    f"Bank inventory mismatch in {bank}: bank={(positive, negative)} "
                    f"manifest={(cumulative_outcomes[True], cumulative_outcomes[False])}"
                )
            previous_bank = bank
            previous_manifest = cumulative_manifest
            previous_features = features
            summary["bank_after_round"] = str(bank)
        else:
            previous_bank = None
            previous_manifest = None
            previous_features = None
            summary["bank_after_round"] = "disabled_empty_admission"

        _write_json_atomic(round_root / "round_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        _write_branch_summary(branch_root, branch, args.rounds)
        _write_campaign_summary(args.output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--task-head-checkpoint", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--libero-python", type=Path, required=True)
    parser.add_argument("--round-script", type=Path, required=True)
    parser.add_argument("--branch", choices=(*BRANCH_ADMISSION, "all_branches"), default="all_branches")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--task-ids-csv", default=",".join(str(task_id) for task_id in EXPECTED_TASKS))
    parser.add_argument("--max-env-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--env-workers", type=int, default=32)
    parser.add_argument("--shards-per-task", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--trajectory-image-size", type=int, default=128)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    for name in ("openpi_root", "output_root", "policy_dir", "task_head_checkpoint", "round_script"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = absolute_executable_path(args.python)
    args.libero_python = absolute_executable_path(args.libero_python)
    try:
        args.task_ids = tuple(int(value) for value in args.task_ids_csv.split(","))
    except ValueError as error:
        raise ValueError("task-ids-csv must contain comma-separated integers") from error
    if (
        not args.task_ids
        or len(args.task_ids) != len(set(args.task_ids))
        or not set(args.task_ids).issubset(EXPECTED_TASKS)
    ):
        raise ValueError(f"task-ids-csv must contain unique ids from {EXPECTED_TASKS}")
    if not args.output_root.is_absolute():
        raise ValueError(f"Self-CL output root must be absolute: {args.output_root}")
    if min(args.rounds, args.episodes_per_task, args.inference_batch_size, args.env_workers, args.shards_per_task) <= 0:
        raise ValueError("Round, episode, batch, worker, and shard counts must be positive")
    if args.max_env_steps < 0:
        raise ValueError("max-env-steps must be non-negative")
    for path in (args.python, args.libero_python, args.round_script, args.task_head_checkpoint, args.policy_dir / "model.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _ensure_campaign_config(args)
    branches = BRANCH_ADMISSION if args.branch == "all_branches" else (args.branch,)
    for branch in branches:
        run_branch(args, branch)


if __name__ == "__main__":
    main()
