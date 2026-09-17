from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Literal

COUNTS = (1, 10, 50)
PI_FAILURE_COUNTS = (1, 5)
TASK_IDS = tuple(range(10))
SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
SELECTION_SEED = 7
Phase = Literal["subsets", "banks", "eval", "all"]


class PreflightError(RuntimeError):
    """Raised when an input or experiment identity is not auditable."""


@dataclasses.dataclass(frozen=True)
class NaturalSource:
    manifest: Path
    sha256: str
    provenance: dict[str, str]


@dataclasses.dataclass(frozen=True)
class Experiment:
    manifest_path: Path
    root: Path
    artifact_root: Path
    run_root: Path
    python: str
    name: str
    policy_family: str
    natural: NaturalSource
    policy_dir: Path
    policy_model_sha256: str
    task_head_checkpoint: Path
    task_head_sha256: str
    config_name: str
    seed: int
    episodes_per_task: int
    batch_size: int
    suites: tuple[str, ...]
    pi_include_failure_50: bool
    pi_sparse_failure_fallback: bool


@dataclasses.dataclass(frozen=True)
class Node:
    node_id: str
    kind: str
    success_count: int
    failure_count: int
    dependencies: tuple[str, ...]
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    identity_path: Path
    identity: dict[str, Any]

    def shell_command(self) -> str:
        environment = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env)
        command = shlex.join(self.command)
        return f"{environment} {command}" if environment else command


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_digest(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PreflightError(f"{label} must be a frozen lowercase sha256 digest")
    return digest


def load_experiment(path: str | Path) -> Experiment:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise PreflightError("memory-quantity manifest schema_version must be 1")
    base = manifest_path.parent
    natural = payload["natural_manifest"]
    policy = payload["policy"]
    evaluation = payload.get("evaluation", {})
    provenance = natural.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise PreflightError("natural_manifest.provenance must contain exact expected fields")
    return Experiment(
        manifest_path=manifest_path,
        root=_resolve(base, payload["openpi_root"]),
        artifact_root=_resolve(base, payload["artifact_root"]),
        run_root=_resolve(base, payload["run_root"]),
        python=str(payload.get("python", "/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python")),
        name=str(payload["name"]),
        policy_family=str(payload["policy_family"]).lower(),
        natural=NaturalSource(
            manifest=_resolve(base, natural["path"]),
            sha256=_require_digest(natural["sha256"], "natural_manifest.sha256"),
            provenance={str(key): str(value) for key, value in provenance.items()},
        ),
        policy_dir=_resolve(base, policy["directory"]),
        policy_model_sha256=_require_digest(policy["model_sha256"], "policy.model_sha256"),
        task_head_checkpoint=_resolve(base, policy["task_head_checkpoint"]),
        task_head_sha256=_require_digest(policy["task_head_sha256"], "policy.task_head_sha256"),
        config_name=str(policy.get("config_name", "pi05_libero")),
        seed=int(evaluation.get("seed", SELECTION_SEED)),
        episodes_per_task=int(evaluation.get("episodes_per_task", 100)),
        batch_size=int(evaluation.get("batch_size", 8)),
        suites=tuple(str(suite) for suite in evaluation.get("suites", ["libero_10"])),
        pi_include_failure_50=bool(payload.get("pi_include_failure_50", False)),
        pi_sparse_failure_fallback=bool(payload.get("pi_sparse_failure_fallback", False)),
    )


def task_keys(experiment: Experiment) -> tuple[tuple[str, int], ...]:
    unknown = sorted(set(experiment.suites).difference(SUITE_TASK_COUNTS))
    if unknown:
        raise PreflightError(f"unsupported evaluation suites: {unknown}")
    return tuple(
        (suite, task_id)
        for suite in experiment.suites
        for task_id in range(SUITE_TASK_COUNTS[suite])
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreflightError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise PreflightError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise PreflightError(f"empty natural manifest: {path}")
    return rows


def audit_natural_manifest(
    experiment: Experiment,
    *,
    required_failure_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, int]]]:
    source = experiment.natural
    if not source.manifest.is_file():
        raise PreflightError(f"missing natural manifest: {source.manifest}")
    actual = sha256_file(source.manifest)
    if actual != source.sha256:
        raise PreflightError(f"natural manifest digest mismatch: expected {source.sha256}, got {actual}")
    rows = _read_jsonl(source.manifest)
    required = {
        "action_id",
        "episode_idx",
        "prompt",
        "source_format",
        "success",
        "suite",
        "task_id",
        "trajectory_path",
    }
    seen_action_ids: set[str] = set()
    keys = task_keys(experiment)
    key_set = set(keys)
    counts_by_key = {"success": dict.fromkeys(keys, 0), "failure": dict.fromkeys(keys, 0)}
    counts = {
        "success": {f"{suite}/task_{task_id:03d}": 0 for suite, task_id in keys},
        "failure": {f"{suite}/task_{task_id:03d}": 0 for suite, task_id in keys},
    }
    selected_rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, 1):
        missing = sorted(required - row.keys())
        if missing:
            raise PreflightError(f"natural manifest row {line_number} is missing fields {missing}")
        if type(row["task_id"]) is not int:
            raise PreflightError(f"natural manifest row {line_number} has non-integer task_id")
        suite_task = (str(row["suite"]), int(row["task_id"]))
        if suite_task not in key_set:
            continue
        if type(row["success"]) is not bool:
            raise PreflightError(f"natural manifest row {line_number} success must be boolean")
        action_id = str(row["action_id"])
        if not action_id or action_id in seen_action_ids:
            raise PreflightError(f"natural manifest contains empty or duplicate action_id={action_id!r}")
        seen_action_ids.add(action_id)
        for key, expected in source.provenance.items():
            if str(row.get(key, "")) != expected:
                raise PreflightError(
                    f"natural manifest provenance mismatch at row {line_number}: "
                    f"{key}={row.get(key)!r}, expected {expected!r}"
                )
        outcome = "success" if row["success"] else "failure"
        counts_by_key[outcome][suite_task] += 1
        counts[outcome][f"{suite_task[0]}/task_{suite_task[1]:03d}"] += 1
        selected_rows.append(row)
    if required_failure_count is None:
        required_failure_count = 50 if experiment.policy_family != "pi" or experiment.pi_include_failure_50 else 10
    required = {"success": 50, "failure": int(required_failure_count)}
    insufficient = {
        outcome: {f"{suite}/task_{task_id:03d}": count for (suite, task_id), count in per_task.items() if count < required[outcome]}
        for outcome, per_task in counts_by_key.items()
    }
    insufficient = {outcome: values for outcome, values in insufficient.items() if values}
    if insufficient:
        raise PreflightError(f"natural manifest lacks hard per-task/outcome quotas {required}: {insufficient}")
    return selected_rows, counts


def preflight(
    experiment: Experiment,
    *,
    required_failure_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if experiment.seed != SELECTION_SEED:
        raise PreflightError("memory quantity selection/evaluation seed is fixed to 7")
    if experiment.batch_size != 8:
        raise PreflightError("memory quantity evaluation batch_size is fixed to 8")
    if experiment.episodes_per_task <= 0:
        raise PreflightError("episodes_per_task must be positive")
    if not experiment.name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in experiment.name
    ):
        raise PreflightError("name must contain only letters, digits, underscores, or hyphens")
    model = experiment.policy_dir / "model.safetensors"
    for path, expected, label in (
        (model, experiment.policy_model_sha256, "policy model"),
        (experiment.task_head_checkpoint, experiment.task_head_sha256, "task head"),
    ):
        if not path.is_file():
            raise PreflightError(f"missing {label}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PreflightError(f"{label} digest mismatch: expected {expected}, got {actual}")
    rows, counts = audit_natural_manifest(experiment, required_failure_count=required_failure_count)
    return rows, {
        "natural_manifest": str(experiment.natural.manifest),
        "natural_manifest_sha256": experiment.natural.sha256,
        "rows": len(rows),
        "counts": counts,
        "policy_model_sha256": experiment.policy_model_sha256,
        "task_head_sha256": experiment.task_head_sha256,
    }


def _selection_key(row: dict[str, Any]) -> tuple[str, str]:
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    material = f"memory-quantity-v1\0seed={SELECTION_SEED}\0{canonical}".encode()
    return hashlib.sha256(material).hexdigest(), canonical


def select_subset(rows: list[dict[str, Any]], success_count: int, failure_count: int) -> list[dict[str, Any]]:
    if success_count not in COUNTS or failure_count not in COUNTS:
        raise PreflightError(f"counts must be in {COUNTS}")
    selected = []
    keys = sorted({(str(row["suite"]), int(row["task_id"])) for row in rows})
    if not keys:
        raise PreflightError("cannot select subset from empty rows")
    for suite, task_id in keys:
        for success, quota in ((True, success_count), (False, failure_count)):
            candidates = [
                row
                for row in rows
                if str(row["suite"]) == suite and int(row["task_id"]) == task_id and row["success"] is success
            ]
            if len(candidates) < quota:
                outcome = "success" if success else "failure"
                raise PreflightError(
                    f"{suite}/task_{task_id:03d} {outcome} quota {quota} exceeds available {len(candidates)}"
                )
            selected.extend(sorted(candidates, key=_selection_key)[:quota])
    return selected


def _tag(success_count: int, failure_count: int) -> str:
    return f"s{success_count:02d}_f{failure_count:02d}"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def materialize_subset(experiment: Experiment, success_count: int, failure_count: int) -> dict[str, Any]:
    rows, _ = audit_natural_manifest(experiment, required_failure_count=failure_count)
    selected = select_subset(rows, success_count, failure_count)
    output = experiment.artifact_root / "subsets" / f"{_tag(success_count, failure_count)}.jsonl"
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in selected)
    _atomic_write(output, content)
    subset_digest = sha256_file(output)
    audit = {
        "schema_version": 1,
        "selection_protocol": "suite_task_per_outcome_sha256_rank_v2",
        "suites": list(experiment.suites),
        "task_count": len(task_keys(experiment)),
        "selection_seed": SELECTION_SEED,
        "source_manifest": str(experiment.natural.manifest),
        "source_manifest_sha256": experiment.natural.sha256,
        "subset_manifest": str(output),
        "subset_manifest_sha256": subset_digest,
        "success_per_task": success_count,
        "failure_per_task": failure_count,
        "rows": len(selected),
        "action_ids_sha256": hashlib.sha256("\n".join(str(row["action_id"]) for row in selected).encode()).hexdigest(),
    }
    _atomic_write(output.with_suffix(".audit.json"), json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def _node_identity(experiment: Experiment, kind: str, success_count: int, failure_count: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_manifest_sha256": sha256_file(experiment.manifest_path),
        "natural_manifest_sha256": experiment.natural.sha256,
        "policy_model_sha256": experiment.policy_model_sha256,
        "task_head_sha256": experiment.task_head_sha256,
        "selection_seed": SELECTION_SEED,
        "success_per_task": success_count,
        "failure_per_task": failure_count,
        "kind": kind,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["artifact_identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _combinations(experiment: Experiment, *, include_pi_failure_50: bool | None) -> tuple[tuple[int, int], ...]:
    include_failure_50 = experiment.pi_include_failure_50 if include_pi_failure_50 is None else include_pi_failure_50
    failure_counts = PI_FAILURE_COUNTS if experiment.policy_family == "pi" else COUNTS
    if experiment.policy_family == "pi" and include_failure_50:
        failure_counts = (*PI_FAILURE_COUNTS, 50)
    return tuple((success, failure) for success in COUNTS for failure in failure_counts)


def build_dag(
    experiment: Experiment,
    phase: Phase = "all",
    *,
    include_pi_failure_50: bool | None = None,
) -> list[Node]:
    nodes = []
    combinations = _combinations(experiment, include_pi_failure_50=include_pi_failure_50)
    for offset, (success_count, failure_count) in enumerate(combinations):
        tag = _tag(success_count, failure_count)
        subset = experiment.artifact_root / "subsets" / f"{tag}.jsonl"
        features = experiment.artifact_root / "features" / tag
        bank = experiment.artifact_root / "banks" / tag
        run = experiment.run_root / tag / "v0_success_failure"
        subset_id = f"{tag}:subset"
        feature_id = f"{tag}:features"
        bank_id = f"{tag}:bank"
        if phase in {"subsets", "all"}:
            nodes.append(
                Node(
                    subset_id,
                    "subset",
                    success_count,
                    failure_count,
                    (),
                    (
                        experiment.python,
                        "scripts/experiments/run_memory_quantity.py",
                        "--manifest",
                        str(experiment.manifest_path),
                        "--materialize-subset",
                        f"{success_count},{failure_count}",
                    ),
                    (),
                    subset.with_suffix(".identity.json"),
                    _node_identity(experiment, "subset", success_count, failure_count),
                )
            )
        if phase in {"banks", "all"}:
            nodes.append(
                Node(
                    feature_id,
                    "features",
                    success_count,
                    failure_count,
                    (subset_id,),
                    (
                        experiment.python,
                        "scripts/memory/cache_prior_head_features.py",
                        "--manifest",
                        str(subset),
                        "--output-dir",
                        str(features),
                        "--policy-dir",
                        str(experiment.policy_dir),
                        "--config-name",
                        experiment.config_name,
                        "--device",
                        "cuda",
                        "--batch-size",
                        "8",
                    ),
                    (),
                    features / "orchestrator.identity.json",
                    _node_identity(experiment, "features", success_count, failure_count),
                )
            )
            nodes.append(
                Node(
                    bank_id,
                    "bank",
                    success_count,
                    failure_count,
                    (feature_id,),
                    (
                        experiment.python,
                        "scripts/memory/build_cl_memory_bank.py",
                        "--group",
                        f"memory_quantity_{experiment.name}_{tag}",
                        "--manifest",
                        str(subset),
                        "--feature-dir",
                        str(features),
                        "--checkpoint",
                        str(experiment.task_head_checkpoint),
                        "--output-dir",
                        str(bank),
                        "--admission",
                        "both",
                        "--device",
                        "cuda",
                    ),
                    (),
                    bank / "orchestrator.identity.json",
                    _node_identity(experiment, "bank", success_count, failure_count),
                )
            )
        if phase in {"eval", "all"}:
            positive = bank / "positive"
            negative = bank / "negative"
            env = {
                "OPENPI_PYTHON": experiment.python,
                "POLICY_DIR": str(experiment.policy_dir),
                "TASK_HEAD_CKPT": str(experiment.task_head_checkpoint),
                "MEMORY_META_PATH": str(positive / "gpm_memory_meta.pt"),
                "FAISS_INDEX_PATH": str(positive / "gpm_memory.index"),
                "MEMORY_ACTIONS_PATH": str(positive / "gpm_memory_actions.npz"),
                "NEGATIVE_MEMORY_META_PATH": str(negative / "gpm_negative_memory_meta.pt"),
                "NEGATIVE_FAISS_INDEX_PATH": str(negative / "gpm_negative_memory.index"),
                "NEGATIVE_MEMORY_ACTIONS_PATH": str(negative / "gpm_negative_memory_actions.npz"),
                "RUN_ROOT": str(run),
                "GPU": "0",
                "PORT": str(8200 + offset),
                "SEED": str(SELECTION_SEED),
                "NUM_TRIALS_PER_TASK": str(experiment.episodes_per_task),
                "BATCH_SIZE": "8",
                "RESUME": "auto",
                "SAVE_VIDEOS": "0",
                "SAVE_EPISODE_DATA": "0",
                "MEMORY_GUIDANCE_TRACE_DIR": str(run / "traces"),
                "MEMORY_GUIDANCE_TRACE_LEVEL": "light",
            }
            nodes.append(
                Node(
                    f"{tag}:eval",
                    "eval",
                    success_count,
                    failure_count,
                    (bank_id,),
                    ("bash", "scripts/experiments/run_memory_quantity.sh"),
                    tuple(sorted(env.items())),
                    run / "orchestrator.identity.json",
                    _node_identity(experiment, "eval:v0_success_failure", success_count, failure_count),
                )
            )
    return nodes


def validate_dag(nodes: list[Node], *, phase: Phase) -> None:
    ids = [node.node_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise PreflightError("DAG contains duplicate node IDs")
    if phase == "all":
        available = set(ids)
        for node in nodes:
            missing = set(node.dependencies) - available
            if missing:
                raise PreflightError(f"{node.node_id} has missing dependencies: {sorted(missing)}")


def _read_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PreflightError(f"unreadable identity file: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"identity file is not an object: {path}")
    return value


def execute(nodes: list[Node], experiment: Experiment, *, dry_run: bool, resume: bool) -> None:
    completed: set[str] = set()
    node_ids = {node.node_id for node in nodes}
    for node in nodes:
        expected_path = experiment.artifact_root / ".orchestrator" / f"{node.node_id.replace(':', '__')}.expected.json"
        completed_identity = _read_identity(node.identity_path)
        expected_identity = _read_identity(expected_path)
        if completed_identity is not None and completed_identity != node.identity:
            raise PreflightError(f"completed identity mismatch for {node.node_id}: {node.identity_path}")
        if expected_identity is not None and expected_identity != node.identity:
            raise PreflightError(f"resume identity mismatch for {node.node_id}: {expected_path}")
        if resume and completed_identity == node.identity:
            print(f"SKIP {node.node_id}")
            completed.add(node.node_id)
            continue
        if (completed_identity is not None or expected_identity is not None) and not resume:
            raise PreflightError(f"existing artifact requires --resume: {node.node_id}")
        unresolved = (set(node.dependencies) & node_ids) - completed
        if unresolved:
            raise RuntimeError(f"cannot execute {node.node_id}; unresolved dependencies: {sorted(unresolved)}")
        print(f"RUN  {node.node_id}: {node.shell_command()}")
        if dry_run:
            completed.add(node.node_id)
            continue
        _atomic_write(expected_path, json.dumps(node.identity, indent=2, sort_keys=True) + "\n")
        subprocess.run(node.command, cwd=experiment.root, env={**os.environ, **dict(node.env)}, check=True)
        _atomic_write(node.identity_path, json.dumps(node.identity, indent=2, sort_keys=True) + "\n")
        expected_path.unlink()
        completed.add(node.node_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orchestrate deterministic LIBERO-10 memory-quantity experiments")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase", choices=("subsets", "banks", "eval", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-pi-failure-50", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--materialize-subset", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    experiment = load_experiment(args.manifest)
    require_failure_50 = args.include_pi_failure_50 is True or (
        args.include_pi_failure_50 is None and experiment.pi_include_failure_50
    )
    rows, audit = preflight(experiment, required_failure_count=50 if require_failure_50 else None)
    if args.materialize_subset:
        try:
            success_count, failure_count = (int(value) for value in args.materialize_subset.split(","))
        except (TypeError, ValueError) as exc:
            raise PreflightError("--materialize-subset must be SUCCESS,FAILURE") from exc
        result = materialize_subset(experiment, success_count, failure_count)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    del rows
    nodes = build_dag(experiment, args.phase, include_pi_failure_50=args.include_pi_failure_50)
    validate_dag(nodes, phase=args.phase)
    print(
        json.dumps(
            {
                "experiment": experiment.name,
                "phase": args.phase,
                "nodes": len(nodes),
                "subset_commands": sum(node.kind == "subset" for node in nodes),
                "bank_commands": sum(node.kind == "bank" for node in nodes),
                "eval_commands": sum(node.kind == "eval" for node in nodes),
                **audit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    execute(nodes, experiment, dry_run=args.dry_run, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
