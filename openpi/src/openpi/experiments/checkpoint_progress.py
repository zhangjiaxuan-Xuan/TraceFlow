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

from openpi.experiments.topk_contract import TopKSelection
from openpi.experiments.topk_contract import load_topk_selection

ALIASES = ("6k", "10k", "14k", "18k", "22k", "26k", "30k")
PROFILES = {"anchors": ("6k", "18k", "30k"), "full": ALIASES}
SOURCES = ("B", "N", "S")
MODES = ("policy", "prior", "v0_success", "v0_success_failure")
Phase = Literal["prepare", "eval", "all"]


class PreflightError(RuntimeError):
    """Raised when an experiment identity or prerequisite is unsafe."""


@dataclasses.dataclass(frozen=True)
class Source:
    name: str
    manifest: Path
    sha256: str
    producer: str
    producer_field: str


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    alias: str
    train_state_step: int
    saved_step: int
    jax_dir: Path
    pytorch_dir: Path
    identity_file: Path


@dataclasses.dataclass(frozen=True)
class Node:
    node_id: str
    kind: str
    checkpoint: str
    source: str | None
    mode: str | None
    dependencies: tuple[str, ...]
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    identity_path: Path
    identity: dict[str, Any]

    def shell_command(self) -> str:
        environment = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env)
        command = shlex.join(self.command)
        return f"{environment} {command}" if environment else command


@dataclasses.dataclass(frozen=True)
class Experiment:
    manifest_path: Path
    root: Path
    artifact_root: Path
    run_root: Path
    python: str
    config_name: str
    sources: dict[str, Source]
    checkpoints: dict[str, Checkpoint]
    failure_source: str
    seed: int
    episodes_per_task: int
    batch_size: int
    topk_selection: TopKSelection
    fixed_prior_top_k: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_experiment(path: str | Path) -> Experiment:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise PreflightError("checkpoint-progress manifest schema_version must be 1")
    base = manifest_path.parent
    root = _resolve(base, payload["openpi_root"])
    sources = {
        name: Source(
            name=name,
            manifest=_resolve(base, spec["manifest"]),
            sha256=str(spec["sha256"]),
            producer=str(spec["producer"]),
            producer_field=str(spec.get("producer_field", "producer")),
        )
        for name, spec in payload["sources"].items()
    }
    if tuple(sorted(sources)) != SOURCES:
        raise PreflightError(f"sources must be exactly {SOURCES}")
    checkpoints = {}
    for item in payload["checkpoints"]:
        checkpoint = Checkpoint(
            alias=str(item["alias"]),
            train_state_step=int(item["train_state_step"]),
            saved_step=int(item["saved_step"]),
            jax_dir=_resolve(base, item["jax_dir"]),
            pytorch_dir=_resolve(base, item["pytorch_dir"]),
            identity_file=_resolve(base, item["identity_file"]),
        )
        checkpoints[checkpoint.alias] = checkpoint
    if tuple(checkpoints) != ALIASES:
        raise PreflightError(f"checkpoint aliases must be ordered as {ALIASES}")
    evaluation = payload.get("evaluation", {})
    selection_spec = payload.get("topk_selection")
    if not isinstance(selection_spec, dict):
        raise PreflightError("checkpoint-progress manifest requires topk_selection path and sha256")
    try:
        topk_selection = load_topk_selection(
            _resolve(base, selection_spec["path"]),
            str(selection_spec["sha256"]),
            consumer="pi",
        )
    except (KeyError, ValueError) as error:
        raise PreflightError(f"invalid Pi top-k selection: {error}") from error
    fixed_prior_top_k = int(payload.get("fixed_prior_top_k", 0))
    if fixed_prior_top_k < 1:
        raise PreflightError("checkpoint-progress manifest requires positive fixed_prior_top_k")
    return Experiment(
        manifest_path=manifest_path,
        root=root,
        artifact_root=_resolve(base, payload["artifact_root"]),
        run_root=_resolve(base, payload["run_root"]),
        python=str(payload.get("python", "/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python")),
        config_name=str(payload.get("config_name", "pi05_libero")),
        sources=sources,
        checkpoints=checkpoints,
        failure_source=str(payload.get("failure_source", "N")),
        seed=int(evaluation.get("seed", 7)),
        episodes_per_task=int(evaluation.get("episodes_per_task", 100)),
        batch_size=int(evaluation.get("batch_size", 8)),
        topk_selection=topk_selection,
        fixed_prior_top_k=fixed_prior_top_k,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise PreflightError(f"{path}:{line_number} is not an object")
                rows.append(row)
    if not rows:
        raise PreflightError(f"empty source manifest: {path}")
    return rows


def _audit_source(source: Source) -> list[dict[str, Any]]:
    if not source.manifest.is_file():
        raise PreflightError(f"missing {source.name} manifest: {source.manifest}")
    actual = sha256_file(source.manifest)
    if source.sha256 in {"", "REQUIRED"}:
        raise PreflightError(f"source {source.name} requires a frozen sha256 digest")
    if actual != source.sha256:
        raise PreflightError(f"source {source.name} digest mismatch: expected {source.sha256}, got {actual}")
    rows = _read_jsonl(source.manifest)
    if source.name == "B":
        if len(rows) != 6500:
            raise PreflightError(f"B must be the frozen 6500-trajectory demonstration set; found {len(rows)}")
        if any(row.get("success") is False for row in rows):
            raise PreflightError("B must be the fixed success-only demonstration set")
        return rows
    producers = {str(row.get(source.producer_field, "")) for row in rows}
    if producers != {source.producer}:
        raise PreflightError(
            f"source {source.name} must contain only producer={source.producer!r}; found {sorted(producers)}"
        )
    forbidden = ("stage1_26", "26_fail", "checkpoint_rollout")
    for row in rows:
        if str(row.get("suite", "")) != "libero_10" or int(row.get("task_id", -1)) not in range(10):
            raise PreflightError(f"source {source.name} must contain only LIBERO-10 tasks 0..9")
        provenance = json.dumps(row, sort_keys=True).lower()
        if any(marker in provenance for marker in forbidden):
            raise PreflightError(f"source {source.name} contains forbidden provenance")
    return rows


def audit_checkpoint(checkpoint: Checkpoint) -> dict[str, Any]:
    expected_step = int(checkpoint.alias.removesuffix("k")) * 1000
    if checkpoint.train_state_step != expected_step:
        raise PreflightError(f"{checkpoint.alias} must map to train_state.step={expected_step}")
    if checkpoint.saved_step != checkpoint.train_state_step - 1:
        raise PreflightError(
            f"{checkpoint.alias} must use saved loop step {checkpoint.train_state_step - 1}; "
            f"got {checkpoint.saved_step}"
        )
    if checkpoint.jax_dir.name != str(checkpoint.saved_step):
        raise PreflightError(
            f"{checkpoint.alias} JAX directory must end in saved step {checkpoint.saved_step}: {checkpoint.jax_dir}"
        )
    if not checkpoint.identity_file.is_file():
        raise PreflightError(f"missing checkpoint identity: {checkpoint.identity_file}")
    identity = json.loads(checkpoint.identity_file.read_text(encoding="utf-8"))
    expected = {
        "alias": checkpoint.alias,
        "train_state_step": checkpoint.train_state_step,
        "saved_step": checkpoint.saved_step,
        "jax_dir": str(checkpoint.jax_dir),
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise PreflightError(f"{checkpoint.alias} identity mismatch for {key}: {identity.get(key)!r} != {value!r}")
    if not str(identity.get("checkpoint_digest", "")):
        raise PreflightError(f"{checkpoint.alias} identity is missing checkpoint_digest")
    return identity


def _failure_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    counts = dict.fromkeys(range(10), 0)
    for row in rows:
        suite = str(row.get("suite", "libero_10"))
        is_failure = row.get("success") is False or row.get("outcome") == "failure"
        if suite == "libero_10" and is_failure:
            task_id = int(row["task_id"])
            if task_id in counts:
                counts[task_id] += 1
    return counts


def preflight(experiment: Experiment, aliases: tuple[str, ...], *, require_failures: bool) -> dict[str, Any]:
    if experiment.failure_source not in SOURCES:
        raise PreflightError(f"failure_source must be one of {SOURCES}")
    if experiment.batch_size != 8:
        raise PreflightError("checkpoint progress evaluation is fixed to batch_size=8")
    source_rows = {name: _audit_source(source) for name, source in experiment.sources.items()}
    identities = {alias: audit_checkpoint(experiment.checkpoints[alias]) for alias in aliases}
    counts = _failure_counts(source_rows[experiment.failure_source])
    if require_failures and min(counts.values()) < 10:
        missing = {task: count for task, count in counts.items() if count < 10}
        raise PreflightError(f"V0 +/- requires at least 10 LIBERO-10 failures per task; insufficient: {missing}")
    return {
        "sources": {
            name: {"rows": len(rows), "sha256": experiment.sources[name].sha256} for name, rows in source_rows.items()
        },
        "failure_counts": counts,
        "checkpoints": identities,
    }


def _node_identity(experiment: Experiment, checkpoint: Checkpoint, kind: str, source: str | None) -> dict[str, Any]:
    identity = json.loads(checkpoint.identity_file.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_manifest_sha256": sha256_file(experiment.manifest_path),
        "checkpoint_alias": checkpoint.alias,
        "checkpoint_digest": identity["checkpoint_digest"],
        "train_state_step": checkpoint.train_state_step,
        "kind": kind,
        "topk_selection_sha256": experiment.topk_selection.sha256,
    }
    if source is not None:
        payload.update(source=source, source_manifest_sha256=experiment.sources[source].sha256)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["artifact_identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _paths(experiment: Experiment, alias: str) -> dict[str, Path]:
    root = experiment.artifact_root / alias
    return {
        "root": root,
        "features": root / "features",
        "head": root / "head",
        "banks": root / "banks",
    }


def build_dag(experiment: Experiment, aliases: tuple[str, ...], phase: Phase = "all") -> list[Node]:
    nodes: list[Node] = []
    python = experiment.python
    for alias in aliases:
        checkpoint = experiment.checkpoints[alias]
        paths = _paths(experiment, alias)
        convert_id = f"{alias}:convert"
        if phase in {"prepare", "all"}:
            nodes.append(
                Node(
                    convert_id,
                    "convert",
                    alias,
                    None,
                    None,
                    (),
                    (
                        python,
                        "examples/convert_jax_model_to_pytorch.py",
                        "--checkpoint-dir",
                        str(checkpoint.jax_dir),
                        "--config-name",
                        experiment.config_name,
                        "--output-path",
                        str(checkpoint.pytorch_dir),
                    ),
                    (),
                    paths["root"] / "convert.identity.json",
                    _node_identity(experiment, checkpoint, "convert", None),
                )
            )
        feature_ids = []
        for source_name in SOURCES:
            source = experiment.sources[source_name]
            feature_id = f"{alias}:features:{source_name}"
            feature_ids.append(feature_id)
            if phase in {"prepare", "all"}:
                output = paths["features"] / source_name
                nodes.append(
                    Node(
                        feature_id,
                        "features",
                        alias,
                        source_name,
                        None,
                        (convert_id,),
                        (
                            python,
                            "scripts/memory/cache_prior_head_features.py",
                            "--manifest",
                            str(source.manifest),
                            "--output-dir",
                            str(output),
                            "--policy-dir",
                            str(checkpoint.pytorch_dir),
                            "--config-name",
                            experiment.config_name,
                            "--device",
                            "cuda",
                            "--batch-size",
                            "8",
                        ),
                        (),
                        output / "orchestrator.identity.json",
                        _node_identity(experiment, checkpoint, "features", source_name),
                    )
                )
        head_id = f"{alias}:head"
        if phase in {"prepare", "all"}:
            nodes.append(
                Node(
                    head_id,
                    "head",
                    alias,
                    "B",
                    None,
                    (f"{alias}:features:B",),
                    (
                        python,
                        "scripts/memory/train_prior_head.py",
                        "--manifest",
                        str(experiment.sources["B"].manifest),
                        "--feature-dir",
                        str(paths["features"] / "B"),
                        "--output-dir",
                        str(paths["head"]),
                        "--seed",
                        "7",
                        "--device",
                        "cuda",
                    ),
                    (),
                    paths["head"] / "orchestrator.identity.json",
                    _node_identity(experiment, checkpoint, "head", "B"),
                )
            )
        bank_ids = {}
        for source_name in SOURCES:
            source = experiment.sources[source_name]
            bank_id = f"{alias}:bank:{source_name}"
            bank_ids[source_name] = bank_id
            if phase in {"prepare", "all"}:
                output = paths["banks"] / source_name
                if source_name == "B":
                    command = (
                        python,
                        "scripts/memory/build_prior_head_memory.py",
                        "--manifest",
                        str(source.manifest),
                        "--feature-dir",
                        str(paths["features"] / source_name),
                        "--checkpoint",
                        str(paths["head"] / "best.pt"),
                        "--output-dir",
                        str(output / "positive"),
                        "--device",
                        "cuda",
                    )
                else:
                    command = (
                        python,
                        "scripts/memory/build_cl_memory_bank.py",
                        "--group",
                        source_name,
                        "--manifest",
                        str(source.manifest),
                        "--feature-dir",
                        str(paths["features"] / source_name),
                        "--checkpoint",
                        str(paths["head"] / "best.pt"),
                        "--output-dir",
                        str(output),
                        "--admission",
                        "both",
                        "--device",
                        "cuda",
                    )
                nodes.append(
                    Node(
                        bank_id,
                        "bank",
                        alias,
                        source_name,
                        None,
                        (f"{alias}:features:{source_name}", head_id),
                        command,
                        (),
                        output / "orchestrator.identity.json",
                        _node_identity(experiment, checkpoint, "bank", source_name),
                    )
                )
        if phase not in {"eval", "all"}:
            continue
        modes = (("policy", None), *((mode, source) for source in SOURCES for mode in MODES[1:]))
        for offset, (mode, source_name) in enumerate(modes):
            node_id = f"{alias}:eval:{mode}" + (f":{source_name}" if source_name else "")
            run_dir = experiment.run_root / alias / (source_name or "none") / mode
            dependencies = () if mode == "policy" else (bank_ids[str(source_name)],)
            env = {
                "MODE": mode,
                "POLICY_DIR": str(checkpoint.pytorch_dir),
                "TASK_HEAD_CKPT": str(paths["head"] / "best.pt"),
                "RUN_ROOT": str(run_dir),
                "GPU": "0",
                "PORT": str(8200 + offset),
                "SEED": str(experiment.seed),
                "NUM_TRIALS_PER_TASK": str(experiment.episodes_per_task),
                "BATCH_SIZE": str(experiment.batch_size),
                "RESUME": "auto",
                "SAVE_VIDEOS": "0",
                "SAVE_EPISODE_DATA": "0",
                "TOPK_SELECTION": str(experiment.topk_selection.path),
                "TOPK_SELECTION_SHA256": experiment.topk_selection.sha256,
            }
            if source_name:
                positive = paths["banks"] / source_name / "positive"
                negative = paths["banks"] / experiment.failure_source / "negative"
                env.update(
                    MEMORY_META_PATH=str(positive / "gpm_memory_meta.pt"),
                    FAISS_INDEX_PATH=str(positive / "gpm_memory.index"),
                    MEMORY_ACTIONS_PATH=str(positive / "gpm_memory_actions.npz"),
                    NEGATIVE_MEMORY_META_PATH=str(negative / "gpm_negative_memory_meta.pt"),
                    NEGATIVE_FAISS_INDEX_PATH=str(negative / "gpm_negative_memory.index"),
                    NEGATIVE_MEMORY_ACTIONS_PATH=str(negative / "gpm_negative_memory_actions.npz"),
                )
                if mode == "v0_success_failure":
                    dependencies += (bank_ids[experiment.failure_source],)
                if mode == "prior":
                    env["FIXED_PRIOR_TOP_K"] = str(experiment.fixed_prior_top_k)
                    env["MEMORY_TOP_K"] = str(experiment.fixed_prior_top_k)
                elif mode in {"v0_success", "v0_success_failure"}:
                    env["MEMORY_TOP_K"] = str(experiment.topk_selection.positive_top_k)
                if mode == "v0_success_failure":
                    env["NEGATIVE_MEMORY_TOP_K"] = str(experiment.topk_selection.negative_top_k)
                if mode in {"v0_success", "v0_success_failure"}:
                    env.update(
                        MEMORY_GUIDANCE_TRACE_DIR=str(run_dir / "traces"),
                        MEMORY_GUIDANCE_TRACE_LEVEL="light",
                    )
            identity = _node_identity(experiment, checkpoint, f"eval:{mode}", source_name)
            identity["failure_source"] = experiment.failure_source if mode == "v0_success_failure" else None
            identity["failure_source_manifest_sha256"] = (
                experiment.sources[experiment.failure_source].sha256 if mode == "v0_success_failure" else None
            )
            identity["top_k"] = {
                "positive": experiment.topk_selection.positive_top_k if mode.startswith("v0_") else None,
                "negative": (experiment.topk_selection.negative_top_k if mode == "v0_success_failure" else None),
                "prior": experiment.fixed_prior_top_k if mode == "prior" else None,
            }
            nodes.append(
                Node(
                    node_id,
                    "eval",
                    alias,
                    source_name,
                    mode,
                    dependencies,
                    ("bash", "scripts/experiments/run_checkpoint_progress_eval.sh"),
                    tuple(sorted(env.items())),
                    run_dir / "orchestrator.identity.json",
                    identity,
                )
            )
    return nodes


def validate_dag(nodes: list[Node], *, phase: Phase) -> None:
    ids = [node.node_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise PreflightError("DAG contains duplicate node IDs")
    available = set(ids)
    for node in nodes:
        if phase == "all" and not set(node.dependencies).issubset(available):
            raise PreflightError(f"{node.node_id} has missing dependency: {set(node.dependencies) - available}")


def _identity_matches(node: Node) -> bool:
    if not node.identity_path.is_file():
        return False
    try:
        return json.loads(node.identity_path.read_text(encoding="utf-8")) == node.identity
    except (json.JSONDecodeError, OSError):
        return False


def execute(nodes: list[Node], experiment: Experiment, *, dry_run: bool, resume: bool) -> None:
    completed: set[str] = set()
    for node in nodes:
        if resume and _identity_matches(node):
            print(f"SKIP {node.node_id}")
            completed.add(node.node_id)
            continue
        unresolved = set(node.dependencies) - completed
        if unresolved and any(dep in {item.node_id for item in nodes} for dep in unresolved):
            raise RuntimeError(f"cannot execute {node.node_id}; unresolved dependencies: {sorted(unresolved)}")
        print(f"RUN  {node.node_id}: {node.shell_command()}")
        if dry_run:
            completed.add(node.node_id)
            continue
        subprocess.run(node.command, cwd=experiment.root, env={**os.environ, **dict(node.env)}, check=True)
        node.identity_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = node.identity_path.with_suffix(node.identity_path.suffix + ".tmp")
        temporary.write_text(json.dumps(node.identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(node.identity_path)
        completed.add(node.node_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orchestrate checkpoint-progress LIBERO-10 experiments")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="anchors")
    parser.add_argument("--phase", choices=("prepare", "eval", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    experiment = load_experiment(args.manifest)
    aliases = PROFILES[args.profile]
    audit = preflight(experiment, aliases, require_failures=args.phase in {"eval", "all"})
    nodes = build_dag(experiment, aliases, args.phase)
    validate_dag(nodes, phase=args.phase)
    eval_count = sum(node.kind == "eval" for node in nodes)
    print(
        json.dumps(
            {"profile": args.profile, "phase": args.phase, "nodes": len(nodes), "eval_commands": eval_count, **audit},
            indent=2,
        )
    )
    execute(nodes, experiment, dry_run=args.dry_run, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
