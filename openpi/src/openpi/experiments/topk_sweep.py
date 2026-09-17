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

from openpi.experiments import memory_quantity

POSITIVE_CANDIDATES = ((1, 8), (10, 1), (10, 8), (50, 8), (50, 16), (50, 32))
NEGATIVE_CANDIDATES = ((1, 1), (1, 4), (5, 4), (5, 8), ("max", 4), ("max", 8))
POSITIVE_REFERENCE = (50, 8)
NEGATIVE_REFERENCE = (5, 4)
TASK_IDS = tuple(range(10))
EPISODE_START = 100
EPISODES_PER_TASK = 50
SEED = 7
BATCH_SIZE = 8
TOLERANCE = 0.005
PROTOCOL = "pi_libero10_memory_capacity_topk_v5_v1_guidance"
Phase = Literal["positive-select", "negative-select", "all"]


class PreflightError(RuntimeError):
    """Raised when a staged top-k experiment is incomplete or not reproducible."""


@dataclasses.dataclass(frozen=True)
class Experiment:
    manifest_path: Path
    root: Path
    run_root: Path
    python: str
    gpu: str
    port: int
    quantity: memory_quantity.Experiment
    quantity_manifest_sha256: str
    adopt_logs: dict[str, Path]


@dataclasses.dataclass(frozen=True)
class Node:
    node_id: str
    stage: str
    success_count: int
    failure_count: int | str
    positive_k: int
    negative_k: int
    run_root: Path
    log_path: Path
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    identity: dict[str, Any]

    @property
    def method(self) -> str:
        return "guidance_negative"

    @property
    def identity_path(self) -> Path:
        return self.run_root / "orchestrator.identity.json"

    def shell_command(self) -> str:
        environment = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env)
        return f"{environment} {shlex.join(self.command)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _require_digest(value: object, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PreflightError(f"{label} must be a lowercase sha256 digest")
    return digest


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_experiment(path: str | Path) -> Experiment:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise PreflightError("pi-topk-sweep manifest schema_version must be 1")
    base = manifest_path.parent
    quantity_spec = payload["memory_quantity_manifest"]
    quantity_path = _resolve(base, quantity_spec["path"])
    quantity_digest = _require_digest(quantity_spec["sha256"], "memory_quantity_manifest.sha256")
    if not quantity_path.is_file() or _sha256(quantity_path) != quantity_digest:
        raise PreflightError("memory quantity manifest is missing or its digest changed")
    quantity = memory_quantity.load_experiment(quantity_path)
    if quantity.policy_family != "pi":
        raise PreflightError("Pi top-k sweep requires a policy_family=pi quantity manifest")
    adopt = {str(node_id): _resolve(base, str(log_path)) for node_id, log_path in payload.get("adopt_logs", {}).items()}
    return Experiment(
        manifest_path=manifest_path,
        root=quantity.root,
        run_root=_resolve(base, payload["run_root"]),
        python=str(payload.get("python", quantity.python)),
        gpu=str(payload.get("gpu", "0")),
        port=int(payload.get("port", 8200)),
        quantity=quantity,
        quantity_manifest_sha256=quantity_digest,
        adopt_logs=adopt,
    )


def _bank_files(bank: Path, outcome: str) -> tuple[Path, Path, Path]:
    prefix = "gpm_memory" if outcome == "positive" else "gpm_negative_memory"
    return bank / f"{prefix}_meta.pt", bank / f"{prefix}.index", bank / f"{prefix}_actions.npz"


def _quantity_bank(experiment: Experiment, success: int, failure: int, outcome: str) -> Path:
    failure_tag = str(failure) if isinstance(failure, str) else f"{failure:02d}"
    tag = f"s{success:02d}_f{failure_tag}"
    return experiment.quantity.artifact_root / "banks" / tag / outcome


def _positive_bank(experiment: Experiment, count: int) -> Path:
    return _quantity_bank(experiment, count, 1, "positive")


def _negative_bank(experiment: Experiment, count: int | str) -> Path:
    # Negative artifacts are outcome-isolated, so the success-side count is fixed.
    count_tag = str(count) if isinstance(count, str) else f"{count:02d}"
    return experiment.quantity.artifact_root / "banks" / f"s50_f{count_tag}" / "negative"


def _task_keys(experiment: Experiment) -> tuple[tuple[str, int], ...]:
    return memory_quantity.task_keys(experiment.quantity)


def _suites_csv(experiment: Experiment) -> str:
    return ",".join(experiment.quantity.suites)


def _total_episodes(experiment: Experiment) -> int:
    return len(_task_keys(experiment)) * EPISODES_PER_TASK


def _artifact_digests(paths: tuple[Path, ...], label: str) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PreflightError(f"missing reused {label} artifacts: {missing}")
    return {path.name: _sha256(path) for path in paths}


def preflight(experiment: Experiment) -> dict[str, Any]:
    if experiment.quantity.seed != SEED or experiment.quantity.batch_size != BATCH_SIZE:
        raise PreflightError("source quantity experiment must use seed=7 and batch_size=8")
    required_failure_count = 0 if experiment.quantity.pi_sparse_failure_fallback else 5
    _, quantity_audit = memory_quantity.preflight(
        experiment.quantity,
        required_failure_count=required_failure_count,
    )
    banks: dict[str, dict[str, str]] = {}
    for count in sorted({count for count, _ in POSITIVE_CANDIDATES}):
        path = _positive_bank(experiment, count)
        banks[f"positive/{count}"] = _artifact_digests(_bank_files(path, "positive"), f"positive/{count}")
    for count in sorted({count for count, _ in NEGATIVE_CANDIDATES}, key=_capacity_sort_key):
        path = _negative_bank(experiment, count)
        banks[f"negative/{count}"] = _artifact_digests(_bank_files(path, "negative"), f"negative/{count}")
    return {"quantity": quantity_audit, "banks": banks}


def _protocol_identity(
    experiment: Experiment,
    *,
    stage: str,
    success_count: int,
    failure_count: int | str,
    positive_k: int,
    negative_k: int,
) -> dict[str, Any]:
    positive = _positive_bank(experiment, success_count)
    negative = _negative_bank(experiment, failure_count)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": stage,
        "method": "guidance_negative",
        "positive_guidance": True,
        "negative_guidance": True,
        "success_per_task": success_count,
        "failure_per_task": failure_count,
        "positive_top_k": positive_k,
        "negative_top_k": negative_k,
        "seed": SEED,
        "suites": list(experiment.quantity.suites),
        "episode_start": EPISODE_START,
        "episodes_per_task": EPISODES_PER_TASK,
        "task_keys": [{"suite": suite, "task_id": task_id} for suite, task_id in _task_keys(experiment)],
        "batch_size": BATCH_SIZE,
        "save_videos": False,
        "save_episode_data": False,
        "trace_level": "light",
        "guidance_version": "v1",
        "quantity_manifest_sha256": experiment.quantity_manifest_sha256,
        "policy_model_sha256": experiment.quantity.policy_model_sha256,
        "task_head_sha256": experiment.quantity.task_head_sha256,
        "positive_bank_sha256": _artifact_digests(_bank_files(positive, "positive"), "positive bank"),
        "negative_bank_sha256": _artifact_digests(_bank_files(negative, "negative"), "negative bank"),
    }
    payload["experiment_identity_sha256"] = _canonical_sha256(payload)
    return payload


def _make_node(
    experiment: Experiment,
    *,
    stage: str,
    success_count: int,
    failure_count: int | str,
    positive_k: int,
    negative_k: int,
    ordinal: int,
) -> Node:
    failure_tag = str(failure_count) if isinstance(failure_count, str) else f"{failure_count:02d}"
    count_tag = f"s{success_count:02d}_f{failure_tag}"
    k_tag = f"kp{positive_k:02d}_kn{negative_k:02d}"
    node_id = f"{stage}:guidance_negative:{count_tag}:{k_tag}"
    run_root = experiment.run_root / stage / f"{count_tag}_{k_tag}"
    positive = _positive_bank(experiment, success_count)
    negative = _negative_bank(experiment, failure_count)
    env = {
        "OPENPI_PYTHON": experiment.python,
        "POLICY_DIR": str(experiment.quantity.policy_dir),
        "TASK_HEAD_CKPT": str(experiment.quantity.task_head_checkpoint),
        "MEMORY_META_PATH": str(positive / "gpm_memory_meta.pt"),
        "FAISS_INDEX_PATH": str(positive / "gpm_memory.index"),
        "MEMORY_ACTIONS_PATH": str(positive / "gpm_memory_actions.npz"),
        "NEGATIVE_MEMORY_META_PATH": str(negative / "gpm_negative_memory_meta.pt"),
        "NEGATIVE_FAISS_INDEX_PATH": str(negative / "gpm_negative_memory.index"),
        "NEGATIVE_MEMORY_ACTIONS_PATH": str(negative / "gpm_negative_memory_actions.npz"),
        "METHOD": "guidance_negative",
        "SUITES": _suites_csv(experiment),
        "MEMORY_TOP_K": str(positive_k),
        "NEGATIVE_MEMORY_TOP_K": str(negative_k),
        "RUN_ROOT": str(run_root),
        "GPU": experiment.gpu,
        "PORT": str(experiment.port + ordinal),
    }
    identity = _protocol_identity(
        experiment,
        stage=stage,
        success_count=success_count,
        failure_count=failure_count,
        positive_k=positive_k,
        negative_k=negative_k,
    )
    log_path = run_root / "eval"
    if len(experiment.quantity.suites) == 1:
        log_path = log_path / f"{experiment.quantity.suites[0]}.jsonl"
    return Node(
        node_id=node_id,
        stage=stage,
        success_count=success_count,
        failure_count=failure_count,
        positive_k=positive_k,
        negative_k=negative_k,
        run_root=run_root,
        log_path=log_path,
        command=("bash", "scripts/experiments/run_pi_topk_eval.sh"),
        env=tuple(sorted(env.items())),
        identity=identity,
    )


def positive_nodes(experiment: Experiment) -> list[Node]:
    failure_count, negative_k = NEGATIVE_REFERENCE
    return [
        _make_node(
            experiment,
            stage="positive",
            success_count=success_count,
            failure_count=failure_count,
            positive_k=positive_k,
            negative_k=negative_k,
            ordinal=ordinal,
        )
        for ordinal, (success_count, positive_k) in enumerate(POSITIVE_CANDIDATES)
    ]


def negative_nodes(experiment: Experiment, positive_count: int, positive_k: int) -> list[Node]:
    return [
        _make_node(
            experiment,
            stage="negative",
            success_count=positive_count,
            failure_count=failure_count,
            positive_k=positive_k,
            negative_k=negative_k,
            ordinal=ordinal,
        )
        for ordinal, (failure_count, negative_k) in enumerate(NEGATIVE_CANDIDATES)
    ]


def _read_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PreflightError(f"unreadable identity: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"identity is not an object: {path}")
    return value


def _adopted_log(experiment: Experiment, node: Node) -> Path | None:
    path = experiment.adopt_logs.get(node.node_id)
    if path is None:
        return None
    identity_path = path.parent.parent / "orchestrator.identity.json"
    actual = _read_identity(identity_path)
    if actual != node.identity:
        raise PreflightError(
            f"cannot adopt {path}: {identity_path} is not the exact protocol identity for {node.node_id}"
        )
    if not path.is_file():
        raise PreflightError(f"adopt log does not exist: {path}")
    return path


def resolve_log(experiment: Experiment, node: Node) -> Path:
    return _adopted_log(experiment, node) or node.log_path


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def execute(nodes: list[Node], experiment: Experiment, *, dry_run: bool, resume: bool) -> None:
    for node in nodes:
        adopted = _adopted_log(experiment, node)
        if adopted is not None:
            print(f"ADOPT {node.node_id}: {adopted}")
            continue
        completed = _read_identity(node.identity_path)
        if completed is not None and completed != node.identity:
            raise PreflightError(f"completed identity mismatch for {node.node_id}: {node.identity_path}")
        if resume and completed == node.identity:
            print(f"SKIP  {node.node_id}")
            continue
        if completed is not None and not resume:
            raise PreflightError(f"existing run requires --resume: {node.node_id}")
        if node.run_root.exists() and any(node.run_root.iterdir()) and completed is None:
            raise PreflightError(f"non-empty run root has no valid identity: {node.run_root}")
        print(f"RUN   {node.node_id}: {node.shell_command()}")
        if dry_run:
            continue
        subprocess.run(node.command, cwd=experiment.root, env={**os.environ, **dict(node.env)}, check=True)
        _atomic_write(node.identity_path, json.dumps(node.identity, indent=2, sort_keys=True) + "\n")


def _load_paired_log(path: Path, experiment: Experiment | None = None) -> dict[tuple[str, int, int], bool]:
    if path.is_dir():
        if experiment is None:
            raise PreflightError(f"experiment is required when loading eval directory: {path}")
        log_files = [path / f"{suite}.jsonl" for suite in experiment.quantity.suites]
    else:
        log_files = [path]
    missing_files = [str(log_file) for log_file in log_files if not log_file.is_file()]
    if missing_files:
        raise PreflightError(f"missing eval logs: {missing_files}")
    episodes: dict[tuple[str, int, int], bool] = {}
    for log_file in log_files:
        with log_file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreflightError(f"invalid JSON at {log_file}:{line_number}") from exc
                if row.get("event") != "episode_result":
                    continue
                suite = str(row.get("task_suite", row.get("suite", log_file.stem)))
                key = (suite, int(row["task_id"]), int(row["episode_idx"]))
                if row.get("error"):
                    raise PreflightError(f"errored episode {key} in {log_file}: {row['error']}")
                if key in episodes:
                    raise PreflightError(f"duplicate paired episode {key} in {log_file}")
                if type(row.get("success")) is not bool:
                    raise PreflightError(f"invalid success label for {key} in {log_file}")
                episodes[key] = row["success"]
    if experiment is None:
        expected_keys = sorted({(suite, task_id) for suite, task_id, _ in episodes})
    else:
        expected_keys = list(_task_keys(experiment))
    expected = {
        (suite, task_id, episode_idx)
        for suite, task_id in expected_keys
        for episode_idx in range(EPISODE_START, EPISODE_START + EPISODES_PER_TASK)
    }
    if set(episodes) != expected:
        missing = len(expected - set(episodes))
        extra = len(set(episodes) - expected)
        raise PreflightError(
            f"{path} must contain exactly {len(expected)} paired episodes; "
            f"found={len(episodes)} missing={missing} extra={extra}"
        )
    return episodes


def _eval_log_sha256(path: Path, experiment: Experiment) -> str:
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise PreflightError(f"missing eval log path: {path}")
    entries = []
    for suite in experiment.quantity.suites:
        log_file = path / f"{suite}.jsonl"
        if not log_file.is_file():
            raise PreflightError(f"missing eval log: {log_file}")
        entries.append({"suite": suite, "sha256": _sha256(log_file)})
    return _canonical_sha256(entries)


def _capacity_sort_key(value: int | str) -> tuple[int, str]:
    if value == "max":
        return (10**9, "max")
    return (int(value), str(value))


def _select_capacity_topk(rows: list[dict[str, Any]], count_key: str, top_k_key: str) -> tuple[int | str, int]:
    best = max(float(row["success_rate"]) for row in rows)
    eligible = [row for row in rows if float(row["success_rate"]) >= best - TOLERANCE]
    selected = min(eligible, key=lambda row: (_capacity_sort_key(row[count_key]), int(row[top_k_key])))
    count = selected[count_key]
    return (count if count == "max" else int(count), int(selected[top_k_key]))


def _evaluate_nodes(experiment: Experiment, nodes: list[Node]) -> list[dict[str, Any]]:
    rows = []
    reference_keys: set[tuple[int, int]] | None = None
    for node in nodes:
        path = resolve_log(experiment, node)
        episodes = _load_paired_log(path, experiment)
        if reference_keys is None:
            reference_keys = set(episodes)
        elif set(episodes) != reference_keys:
            raise PreflightError("candidate logs are not episode-paired")
        successes = sum(episodes.values())
        rows.append(
            {
                "node_id": node.node_id,
                "method": node.method,
                "success_per_task": node.success_count,
                "failure_per_task": node.failure_count,
                "positive_top_k": node.positive_k,
                "negative_top_k": node.negative_k,
                "episodes": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes),
                "eval_log": str(path),
                "eval_log_sha256": _eval_log_sha256(path, experiment),
                "experiment_identity_sha256": node.identity["experiment_identity_sha256"],
                "adopted": path != node.log_path,
            }
        )
    return rows


def _build_positive_selection(experiment: Experiment) -> dict[str, Any]:
    rows = _evaluate_nodes(experiment, positive_nodes(experiment))
    count, top_k = _select_capacity_topk(rows, "success_per_task", "positive_top_k")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "selection_rule": "highest_global_sr_within_0.5pp_then_smallest_count_then_smallest_k",
        "paired_episodes": _total_episodes(experiment),
        "input_manifest": str(experiment.manifest_path),
        "input_manifest_sha256": _sha256(experiment.manifest_path),
        "quantity_manifest_sha256": experiment.quantity_manifest_sha256,
        "positive_memory_per_task": count,
        "positive_top_k": top_k,
        "negative_reference_memory_per_task": NEGATIVE_REFERENCE[0],
        "negative_reference_top_k": NEGATIVE_REFERENCE[1],
        "candidates_sha256": _canonical_sha256(rows),
        "candidates": rows,
    }


def write_positive_selection(experiment: Experiment) -> dict[str, Any]:
    result = _build_positive_selection(experiment)
    _atomic_write(experiment.run_root / "positive_selection.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def load_positive_selection(experiment: Experiment) -> dict[str, Any]:
    path = experiment.run_root / "positive_selection.json"
    if not path.is_file():
        raise PreflightError(f"negative phase requires {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("input_manifest_sha256") != _sha256(experiment.manifest_path):
        raise PreflightError("positive selection belongs to a different input manifest")
    if result != _build_positive_selection(experiment):
        raise PreflightError("positive selection no longer matches its six frozen eval logs")
    return result


def _build_final_selection(experiment: Experiment, positive: dict[str, Any]) -> dict[str, Any]:
    positive_count = int(positive["positive_memory_per_task"])
    positive_k = int(positive["positive_top_k"])
    rows = _evaluate_nodes(experiment, negative_nodes(experiment, positive_count, positive_k))
    negative_count, negative_k = _select_capacity_topk(rows, "failure_per_task", "negative_top_k")
    return {
        "schema_version": 1,
        "consumer": "pi",
        "protocol": PROTOCOL,
        "selection_rule": "highest_global_sr_within_0.5pp_then_smallest_count_then_smallest_k",
        "paired_episodes": _total_episodes(experiment),
        "input_manifest": str(experiment.manifest_path),
        "input_manifest_sha256": _sha256(experiment.manifest_path),
        "quantity_manifest_sha256": experiment.quantity_manifest_sha256,
        "positive_selection_sha256": _sha256(experiment.run_root / "positive_selection.json"),
        "positive_memory_per_task": positive_count,
        "positive_top_k": positive_k,
        "negative_memory_per_task": negative_count,
        "negative_top_k": negative_k,
        "positive_candidates_sha256": positive["candidates_sha256"],
        "negative_candidates_sha256": _canonical_sha256(rows),
        "positive_candidates": positive["candidates"],
        "negative_candidates": rows,
    }


def write_final_selection(experiment: Experiment, positive: dict[str, Any]) -> dict[str, Any]:
    result = _build_final_selection(experiment, positive)
    _atomic_write(experiment.run_root / "final_selection.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _run_positive(experiment: Experiment, *, dry_run: bool, resume: bool) -> dict[str, Any] | None:
    execute(positive_nodes(experiment), experiment, dry_run=dry_run, resume=resume)
    return None if dry_run else write_positive_selection(experiment)


def _run_negative(experiment: Experiment, *, dry_run: bool, resume: bool) -> dict[str, Any] | None:
    positive = load_positive_selection(experiment)
    nodes = negative_nodes(
        experiment,
        int(positive["positive_memory_per_task"]),
        int(positive["positive_top_k"]),
    )
    execute(nodes, experiment, dry_run=dry_run, resume=resume)
    return None if dry_run else write_final_selection(experiment, positive)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pi LIBERO-10 +/- guidance memory capacity and top-k ablation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase", choices=("positive-select", "negative-select", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    experiment = load_experiment(args.manifest)
    preflight(experiment)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "positive_selection_runs": len(POSITIVE_CANDIDATES),
                "negative_selection_runs": len(NEGATIVE_CANDIDATES),
                "paired_episodes_per_run": _total_episodes(experiment),
                "guidance": "positive+negative",
            },
            indent=2,
        )
    )
    if args.phase == "positive-select":
        _run_positive(experiment, dry_run=args.dry_run, resume=args.resume)
    elif args.phase == "negative-select":
        _run_negative(experiment, dry_run=args.dry_run, resume=args.resume)
    elif args.dry_run:
        _run_positive(experiment, dry_run=True, resume=args.resume)
        print("DEFER negative-select until positive_selection.json exists")
    else:
        _run_positive(experiment, dry_run=False, resume=args.resume)
        _run_negative(experiment, dry_run=False, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
