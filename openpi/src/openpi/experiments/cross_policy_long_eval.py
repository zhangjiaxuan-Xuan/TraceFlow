from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

SUITE_TASK_COUNTS = {"libero_10": 10, "libero_90": 90}
Consumer = Literal["pi", "smol"]
Mode = Literal["base", "fixed_prior"]


class AuditError(RuntimeError):
    """Raised when a result cannot be safely identified or adopted."""


@dataclasses.dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str


@dataclasses.dataclass(frozen=True)
class ExistingResult:
    log: Artifact
    identity: Artifact
    summary: Artifact | None = None


@dataclasses.dataclass(frozen=True)
class SeedMapping:
    scheme: str
    task_stride: int
    episode_stride: int
    record_required: bool

    def expected(self, *, base_seed: int, task_id: int, episode_idx: int, episode_start: int) -> int:
        return base_seed + task_id * self.task_stride + (episode_idx - episode_start) * self.episode_stride


@dataclasses.dataclass(frozen=True)
class Condition:
    condition_id: str
    consumer: Consumer
    mode: Mode
    suite: str
    expected_tasks: int
    episodes_per_task: int
    episode_start: int
    seed: int
    seed_mapping: SeedMapping
    batch_size: int
    model_sha256: str
    config_sha256: str
    command: tuple[str, ...]
    cwd: Path
    output_root: Path
    output_log: Path
    existing: ExistingResult | None

    @property
    def expected_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": "pi_smol_libero_10_90_v1",
            "condition_id": self.condition_id,
            "consumer": self.consumer,
            "mode": self.mode,
            "suite": self.suite,
            "task_ids": list(range(self.expected_tasks)),
            "episodes_per_task": self.episodes_per_task,
            "episode_start": self.episode_start,
            "seed": self.seed,
            "seed_mapping": dataclasses.asdict(self.seed_mapping),
            "batch_size": self.batch_size,
            "model_sha256": self.model_sha256,
            "config_sha256": self.config_sha256,
        }


@dataclasses.dataclass(frozen=True)
class Experiment:
    manifest_path: Path
    run_root: Path
    gpu_pool: tuple[str, ...]
    ports: tuple[int, ...]
    resume: bool
    fail_fast: bool
    conditions: tuple[Condition, ...]


@dataclasses.dataclass(frozen=True)
class Audit:
    condition_id: str
    status: str
    source: str
    log_path: str
    suite: str
    consumer: str
    mode: str
    tasks: int
    episodes: int
    successes: int
    success_rate: float
    task_results: dict[str, dict[str, float | int]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: object, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise AuditError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _artifact(base: Path, payload: dict[str, Any], label: str) -> Artifact:
    return Artifact(
        path=_resolve(base, payload["path"]),
        sha256=_require_digest(payload["sha256"], f"{label}.sha256"),
    )


def _load_existing(base: Path, payload: dict[str, Any] | None) -> ExistingResult | None:
    if payload is None:
        return None
    summary = payload.get("summary")
    return ExistingResult(
        log=_artifact(base, payload["log"], "existing.log"),
        identity=_artifact(base, payload["identity"], "existing.identity"),
        summary=_artifact(base, summary, "existing.summary") if summary else None,
    )


def load_experiment(path: str | Path) -> Experiment:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise AuditError("cross-policy manifest schema_version must be 1")
    base = manifest_path.parent
    run_root = _resolve(base, payload["run_root"])
    gpu_pool = tuple(str(item) for item in payload.get("gpu_pool", [0, 1, 2, 3]))
    ports = tuple(int(item) for item in payload.get("ports", [8600, 8700, 8800, 8900]))
    if not gpu_pool or len(gpu_pool) != len(set(gpu_pool)):
        raise AuditError("gpu_pool must contain unique GPU identifiers")
    if len(ports) != len(gpu_pool) or len(ports) != len(set(ports)):
        raise AuditError("ports must be unique and match gpu_pool length")

    conditions: list[Condition] = []
    condition_ids: set[str] = set()
    condition_cells: set[tuple[str, str, str]] = set()
    output_roots: set[Path] = set()
    for item in payload.get("conditions", []):
        condition_id = str(item["id"])
        if condition_id in condition_ids:
            raise AuditError(f"duplicate condition id: {condition_id}")
        consumer = str(item["consumer"])
        mode = str(item["mode"])
        suite = str(item["suite"])
        if consumer not in {"pi", "smol"}:
            raise AuditError(f"unsupported consumer for {condition_id}: {consumer}")
        if mode not in {"base", "fixed_prior"}:
            raise AuditError(f"unsupported mode for {condition_id}: {mode}")
        if consumer == "smol" and mode == "fixed_prior":
            raise AuditError("SmolVLA has no action-prior runtime; fixed_prior is forbidden")
        if suite not in SUITE_TASK_COUNTS:
            raise AuditError(f"{condition_id} must use libero_10 or libero_90")
        cell = (consumer, mode, suite)
        if cell in condition_cells:
            raise AuditError(f"duplicate cross-policy condition cell: {cell}")
        expected_tasks = int(item["expected_tasks"])
        if expected_tasks != SUITE_TASK_COUNTS[suite]:
            raise AuditError(f"{condition_id}: {suite} requires {SUITE_TASK_COUNTS[suite]} tasks, not {expected_tasks}")
        episodes_per_task = int(item["episodes_per_task"])
        episode_start = int(item.get("episode_start", 0))
        batch_size = int(item.get("batch_size", 8))
        if episodes_per_task <= 0 or episode_start < 0:
            raise AuditError(f"{condition_id}: invalid episode range")
        if batch_size != 8:
            raise AuditError(f"{condition_id}: cross-policy evaluation is fixed to batch_size=8")
        command = tuple(str(part) for part in item.get("command", []))
        if not command:
            raise AuditError(f"{condition_id}: command must be a non-empty argv list")
        output_root = _resolve(base, item.get("output_root", run_root / condition_id))
        try:
            output_root.relative_to(run_root)
        except ValueError as exc:
            raise AuditError(f"{condition_id}: output_root must be under run_root") from exc
        if output_root in output_roots:
            raise AuditError(f"duplicate output_root: {output_root}")
        output_log = _resolve(base, item.get("output_log", output_root / "eval.jsonl"))
        try:
            output_log.relative_to(output_root)
        except ValueError as exc:
            raise AuditError(f"{condition_id}: output_log must be under output_root") from exc
        conditions.append(
            Condition(
                condition_id=condition_id,
                consumer=consumer,  # type: ignore[arg-type]
                mode=mode,  # type: ignore[arg-type]
                suite=suite,
                expected_tasks=expected_tasks,
                episodes_per_task=episodes_per_task,
                episode_start=episode_start,
                seed=int(item.get("seed", 7)),
                seed_mapping=_load_seed_mapping(item.get("seed_mapping"), condition_id),
                batch_size=batch_size,
                model_sha256=_require_digest(item["model_sha256"], f"{condition_id}.model_sha256"),
                config_sha256=_require_digest(item["config_sha256"], f"{condition_id}.config_sha256"),
                command=command,
                cwd=_resolve(base, item.get("cwd", base)),
                output_root=output_root,
                output_log=output_log,
                existing=_load_existing(base, item.get("existing")),
            )
        )
        condition_ids.add(condition_id)
        condition_cells.add(cell)
        output_roots.add(output_root)
    if not conditions:
        raise AuditError("manifest must define at least one condition")
    required = {(consumer, "base", suite) for consumer in ("pi", "smol") for suite in SUITE_TASK_COUNTS}
    missing = required - condition_cells
    if missing:
        raise AuditError(f"manifest is missing required Pi/Smol base conditions: {sorted(missing)}")
    prior_suites = {suite for consumer, mode, suite in condition_cells if consumer == "pi" and mode == "fixed_prior"}
    if prior_suites and prior_suites != set(SUITE_TASK_COUNTS):
        raise AuditError("Pi fixed_prior must include both libero_10 and libero_90 or be omitted entirely")
    return Experiment(
        manifest_path=manifest_path,
        run_root=run_root,
        gpu_pool=gpu_pool,
        ports=ports,
        resume=bool(payload.get("resume", True)),
        fail_fast=bool(payload.get("fail_fast", True)),
        conditions=tuple(conditions),
    )


def _load_seed_mapping(payload: object, condition_id: str) -> SeedMapping:
    if not isinstance(payload, dict):
        raise AuditError(f"{condition_id}: seed_mapping must be explicitly declared")
    scheme = str(payload.get("scheme", ""))
    if scheme != "base_plus_task_and_episode":
        raise AuditError(f"{condition_id}: unsupported seed_mapping scheme {scheme!r}")
    record_required = payload.get("record_required")
    if type(record_required) is not bool:
        raise AuditError(f"{condition_id}: seed_mapping.record_required must be boolean")
    return SeedMapping(
        scheme=scheme,
        task_stride=int(payload.get("task_stride", 0)),
        episode_stride=int(payload.get("episode_stride", 0)),
        record_required=record_required,
    )


def _verify_artifact(artifact: Artifact, label: str) -> None:
    if not artifact.path.is_file():
        raise FileNotFoundError(artifact.path)
    actual = sha256_file(artifact.path)
    if actual != artifact.sha256:
        raise AuditError(f"{label} SHA-256 mismatch: {artifact.path}")


def _episode_rows(log_path: Path, condition: Condition) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"invalid JSON at {log_path}:{line_number}") from exc
            if row.get("event", "episode_result") != "episode_result":
                continue
            suite = row.get("task_suite", row.get("suite"))
            if suite != condition.suite:
                raise AuditError(f"{condition.condition_id}: unexpected suite {suite!r} in episode row")
            if not all(key in row for key in ("task_id", "success")):
                raise AuditError(f"{condition.condition_id}: malformed episode row")
            if type(row["success"]) is not bool:
                raise AuditError(f"{condition.condition_id}: success must be a JSON boolean at line {line_number}")
            episode_idx = row.get("episode_idx", row.get("episode_ix"))
            if episode_idx is None:
                raise AuditError(f"{condition.condition_id}: episode row lacks episode index")
            task_id = int(row["task_id"])
            episode_idx = int(episode_idx)
            expected_seed = condition.seed_mapping.expected(
                base_seed=condition.seed,
                task_id=task_id,
                episode_idx=episode_idx,
                episode_start=condition.episode_start,
            )
            if "seed" not in row:
                if condition.seed_mapping.record_required:
                    raise AuditError(f"{condition.condition_id}: episode seed is required at line {line_number}")
            elif type(row["seed"]) is not int or row["seed"] != expected_seed:
                raise AuditError(
                    f"{condition.condition_id}: seed mismatch at task={task_id}, "
                    f"episode={episode_idx}; expected {expected_seed}, got {row['seed']!r}"
                )
            rows.append(
                {
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "success": row["success"],
                }
            )
    return rows


def _summarize_rows(condition: Condition, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_task_ids = set(range(condition.expected_tasks))
    expected_episode_ids = set(range(condition.episode_start, condition.episode_start + condition.episodes_per_task))
    observed: dict[int, dict[int, bool]] = {task_id: {} for task_id in expected_task_ids}
    for row in rows:
        task_id = row["task_id"]
        episode_idx = row["episode_idx"]
        if task_id not in expected_task_ids:
            raise AuditError(f"{condition.condition_id}: unexpected task_id={task_id}")
        if episode_idx not in expected_episode_ids:
            raise AuditError(f"{condition.condition_id}: unexpected episode_idx={episode_idx}")
        if episode_idx in observed[task_id]:
            raise AuditError(f"{condition.condition_id}: duplicate task={task_id}, episode={episode_idx}")
        observed[task_id][episode_idx] = row["success"]
    task_results: dict[str, dict[str, float | int]] = {}
    for task_id in sorted(expected_task_ids):
        episode_ids = set(observed[task_id])
        if episode_ids != expected_episode_ids:
            missing = sorted(expected_episode_ids - episode_ids)
            raise AuditError(f"{condition.condition_id}: task {task_id} is incomplete; missing {len(missing)} episodes")
        successes = sum(observed[task_id].values())
        task_results[str(task_id)] = {
            "episodes": condition.episodes_per_task,
            "successes": successes,
            "success_rate": successes / condition.episodes_per_task,
        }
    episodes = condition.expected_tasks * condition.episodes_per_task
    successes = sum(int(item["successes"]) for item in task_results.values())
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "task_results": task_results,
    }


def _verify_identity(condition: Condition, artifact: Artifact) -> None:
    _verify_artifact(artifact, f"{condition.condition_id} identity")
    identity = json.loads(artifact.path.read_text(encoding="utf-8"))
    if identity != condition.expected_identity:
        raise AuditError(f"{condition.condition_id}: existing identity does not match manifest")


def _verify_summary(condition: Condition, artifact: Artifact, reconstructed: dict[str, Any]) -> None:
    _verify_artifact(artifact, f"{condition.condition_id} summary")
    summary = json.loads(artifact.path.read_text(encoding="utf-8"))
    for field in ("episodes", "successes", "task_results"):
        if summary.get(field) != reconstructed[field]:
            raise AuditError(f"{condition.condition_id}: summary field {field} disagrees with JSONL")
    if abs(float(summary.get("success_rate", -1.0)) - reconstructed["success_rate"]) > 1e-12:
        raise AuditError(f"{condition.condition_id}: summary success_rate disagrees with JSONL")


def audit_log(
    condition: Condition,
    log_path: Path,
    *,
    source: str,
    expected_sha256: str | None = None,
    summary: Artifact | None = None,
) -> Audit:
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    if expected_sha256 is not None and sha256_file(log_path) != expected_sha256:
        raise AuditError(f"{condition.condition_id} log SHA-256 mismatch: {log_path}")
    reconstructed = _summarize_rows(condition, _episode_rows(log_path, condition))
    if summary is not None:
        _verify_summary(condition, summary, reconstructed)
    return Audit(
        condition_id=condition.condition_id,
        status="adopted" if source == "existing" else "completed",
        source=source,
        log_path=str(log_path),
        suite=condition.suite,
        consumer=condition.consumer,
        mode=condition.mode,
        tasks=condition.expected_tasks,
        episodes=reconstructed["episodes"],
        successes=reconstructed["successes"],
        success_rate=reconstructed["success_rate"],
        task_results=reconstructed["task_results"],
    )


def audit_existing(condition: Condition) -> Audit | None:
    existing = condition.existing
    if existing is None:
        return None
    present = [
        existing.log.path.exists(),
        existing.identity.path.exists(),
        existing.summary.path.exists() if existing.summary else True,
    ]
    if not any(present):
        return None
    if not all(present):
        return None
    _verify_identity(condition, existing.identity)
    try:
        return audit_log(
            condition,
            existing.log.path,
            source="existing",
            expected_sha256=existing.log.sha256,
            summary=existing.summary,
        )
    except AuditError as exc:
        if "incomplete" in str(exc):
            return None
        raise


def expand_command(condition: Condition) -> tuple[str, ...]:
    replacements = {
        "condition_id": condition.condition_id,
        "consumer": condition.consumer,
        "mode": condition.mode,
        "suite": condition.suite,
        "run_root": str(condition.output_root),
        "output_log": str(condition.output_log),
        "episodes_per_task": str(condition.episodes_per_task),
        "episode_start": str(condition.episode_start),
        "seed": str(condition.seed),
        "batch_size": str(condition.batch_size),
    }
    try:
        return tuple(part.format_map(replacements) for part in condition.command)
    except KeyError as exc:
        raise AuditError(f"{condition.condition_id}: unsupported command placeholder {exc}") from exc


def write_reports(path: Path, audits: list[Audit], gaps: list[Condition]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": "pi_smol_libero_10_90_v1",
        "results": [dataclasses.asdict(audit) for audit in audits],
        "gaps": [condition.condition_id for condition in gaps],
    }
    (path / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["condition\tconsumer\tmode\tsuite\tstatus\ttasks\tepisodes\tsuccesses\tsuccess_rate\tlog"]
    lines.extend(
        "\t".join(
            [
                audit.condition_id,
                audit.consumer,
                audit.mode,
                audit.suite,
                audit.status,
                str(audit.tasks),
                str(audit.episodes),
                str(audit.successes),
                f"{audit.success_rate:.6f}",
                audit.log_path,
            ]
        )
        for audit in sorted(audits, key=lambda item: item.condition_id)
    )
    (path / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
