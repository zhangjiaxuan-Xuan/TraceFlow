"""Pure scheduling and manifest logic for outcome-deficit collection.

The natural attempt manifest is append-only. Balanced quota subsets are derived
artifacts, so collecting more data never changes or truncates the raw record.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
SUITE_ALIASES = {
    "spatial": ("libero_spatial",),
    "object": ("libero_object",),
    "goal": ("libero_goal",),
    # In this experiment registry, "long" means the reported 10-task
    # OptimusVLA condition. LIBERO-90 is intentionally excluded.
    "long": ("libero_10",),
    # Pi CL uses the four standard 10-task suites. LIBERO-90 is a separate
    # future evaluation branch and is not part of this collection.
    "pi": ("libero_spatial", "libero_object", "libero_goal", "libero_10"),
    "smol": ("libero_10",),
}
REQUIRED_RECORD_FIELDS = {
    "schema",
    "trajectory_id",
    "producer",
    "checkpoint_digest",
    "suite",
    "task_id",
    "seed",
    "episode_idx",
    "outcome",
    "trajectory_path",
    "trajectory_sha256",
    "action_contract",
}


@dataclasses.dataclass(frozen=True)
class OutcomeQuota:
    success: int
    failure: int

    def __post_init__(self) -> None:
        if self.success < 0 or self.failure < 0:
            raise ValueError("Outcome quotas must be non-negative")


@dataclasses.dataclass(frozen=True)
class CollectionSpec:
    producer: str
    checkpoint_digest: str
    suites: tuple[str, ...]
    quota: OutcomeQuota
    max_attempts_per_task: int
    batch_attempts: int
    base_seed: int = 7
    action_contract: str = "libero-relative-ee-7d-v1"
    complete_suite_rounds: bool = False

    def __post_init__(self) -> None:
        if not self.producer or not self.checkpoint_digest:
            raise ValueError("producer and checkpoint_digest are required")
        if self.max_attempts_per_task <= 0 or self.batch_attempts <= 0:
            raise ValueError("Attempt limits must be positive")
        unknown = set(self.suites).difference(SUITE_TASK_COUNTS)
        if unknown:
            raise ValueError(f"Unsupported suites: {sorted(unknown)}")


@dataclasses.dataclass(frozen=True)
class PlanJob:
    suite: str
    task_id: int
    attempt_start: int
    attempt_count: int
    seed: int
    successes: int
    failures: int
    success_deficit: int
    failure_deficit: int
    status: str


def expand_suites(values: Iterable[str]) -> tuple[str, ...]:
    expanded: list[str] = []
    for raw in values:
        for value in raw.split(","):
            name = value.strip().lower()
            if not name:
                continue
            suites = SUITE_ALIASES.get(name, (name,))
            for suite in suites:
                if suite not in SUITE_TASK_COUNTS:
                    raise ValueError(f"Unsupported suite or category: {name}")
                if suite not in expanded:
                    expanded.append(suite)
    if not expanded:
        raise ValueError("At least one suite is required")
    return tuple(expanded)


def record_identity(row: dict[str, Any]) -> tuple[str, str, str, int, int, int]:
    return (
        str(row["producer"]),
        str(row["checkpoint_digest"]),
        str(row["suite"]),
        int(row["task_id"]),
        int(row["seed"]),
        int(row["episode_idx"]),
    )


def validate_record(row: dict[str, Any]) -> None:
    missing = REQUIRED_RECORD_FIELDS.difference(row)
    if missing:
        raise ValueError(f"Attempt record is missing fields: {sorted(missing)}")
    if row["schema"] != "optimus-outcome-attempt-v1":
        raise ValueError(f"Unsupported attempt schema: {row['schema']!r}")
    if row["suite"] not in SUITE_TASK_COUNTS:
        raise ValueError(f"Unsupported suite: {row['suite']!r}")
    task_id = int(row["task_id"])
    if not 0 <= task_id < SUITE_TASK_COUNTS[str(row["suite"])]:
        raise ValueError(f"Task {task_id} is outside suite {row['suite']}")
    if row["outcome"] not in {"success", "failure"}:
        raise ValueError(f"Invalid outcome: {row['outcome']!r}")
    if not row["trajectory_id"] or not row["trajectory_sha256"] or not row["action_contract"]:
        raise ValueError("trajectory_id, trajectory_sha256, and action_contract must be non-empty")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            validate_record(row)
            rows.append(row)
    return rows


def merge_records(existing: Iterable[dict[str, Any]], incoming: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, int, int, int], dict[str, Any]] = {}
    for row in [*existing, *incoming]:
        validate_record(row)
        identity = record_identity(row)
        previous = merged.get(identity)
        if previous is not None and previous != row:
            raise ValueError(f"Conflicting resumed attempt identity: {identity}")
        merged[identity] = row
    return sorted(merged.values(), key=record_identity)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def select_collection_group_records(
    records: Iterable[dict[str, Any]],
    collection_group: str,
    outcome: str,
    capacity_per_task: int | None,
    *,
    fallback_to_max: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one outcome/capacity view from an isolated collection group."""
    expected_producers = {"C-pi": "pi0.5-libero", "C-smol": "smolvla"}
    if collection_group not in expected_producers:
        raise ValueError(f"Unsupported isolated collection group: {collection_group!r}")
    failure_capacities = {"C-pi": {1, 5, None}, "C-smol": {1, 5, 50}}
    allowed_capacities = {"success": {1, 10, 50, None}, "failure": failure_capacities[collection_group]}
    if outcome not in allowed_capacities:
        raise ValueError(f"Unsupported outcome: {outcome!r}")
    if capacity_per_task not in allowed_capacities[outcome]:
        label = "max" if capacity_per_task is None else capacity_per_task
        raise ValueError(f"Unsupported {outcome} capacity: {label!r}")
    expected_producer = expected_producers[collection_group]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        validate_record(row)
        if row.get("collection_group") != collection_group:
            continue
        if row["producer"] != expected_producer:
            raise ValueError(
                f"Collection group {collection_group} contains producer {row['producer']!r}; "
                f"expected {expected_producer!r}"
            )
        if row["outcome"] != outcome:
            continue
        task_key = str(row.get("task_key", f"{row['suite']}/task_{int(row['task_id']):03d}"))
        grouped[task_key].append(row)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    fallback_tasks: list[str] = []
    for task_key, rows in sorted(grouped.items()):
        rows.sort(key=record_identity)
        use_max = (
            fallback_to_max
            and collection_group == "C-pi"
            and outcome == "failure"
            and capacity_per_task is not None
            and len(rows) < capacity_per_task
        )
        chosen = rows if capacity_per_task is None or use_max else rows[:capacity_per_task]
        if use_max:
            fallback_tasks.append(task_key)
        selected.extend(chosen)
        counts[task_key] = len(chosen)
    if not selected:
        raise ValueError(f"No {outcome} records found for collection group {collection_group}")
    selected.sort(key=record_identity)
    capacity_label = "max" if capacity_per_task is None else str(capacity_per_task)
    summary = {
        "schema": "optimus-isolated-group-selection-v1",
        "collection_group": collection_group,
        "producer": expected_producer,
        "outcome": outcome,
        "capacity_per_task": capacity_label,
        "selection": "exact_group_only",
        "records": len(selected),
        "tasks": len(counts),
        "counts_by_task": dict(sorted(counts.items())),
        "fallback_to_max": bool(fallback_to_max),
        "fallback_to_max_tasks": sorted(fallback_tasks),
    }
    return selected, summary


def select_collection_group_failures(
    records: Iterable[dict[str, Any]], collection_group: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible helper for the uncapped failure view."""
    return select_collection_group_records(records, collection_group, "failure", None)


def build_plan(spec: CollectionSpec, records: Iterable[dict[str, Any]]) -> list[PlanJob]:
    counts: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for row in records:
        validate_record(row)
        if row["producer"] != spec.producer or row["checkpoint_digest"] != spec.checkpoint_digest:
            continue
        key = (str(row["suite"]), int(row["task_id"]))
        if key[0] in spec.suites:
            counts[key][str(row["outcome"])] += 1

    jobs: list[PlanJob] = []
    for suite in spec.suites:
        suite_complete = all(
            counts[(suite, task_id)]["success"] >= spec.quota.success
            and counts[(suite, task_id)]["failure"] >= spec.quota.failure
            for task_id in range(SUITE_TASK_COUNTS[suite])
        )
        for task_id in range(SUITE_TASK_COUNTS[suite]):
            outcome_counts = counts[(suite, task_id)]
            successes = outcome_counts["success"]
            failures = outcome_counts["failure"]
            attempted = successes + failures
            success_deficit = max(0, spec.quota.success - successes)
            failure_deficit = max(0, spec.quota.failure - failures)
            if success_deficit == 0 and failure_deficit == 0 and (not spec.complete_suite_rounds or suite_complete):
                status = "complete"
                attempt_count = 0
            elif attempted >= spec.max_attempts_per_task:
                status = "incomplete"
                attempt_count = 0
            else:
                status = "pending"
                attempt_count = min(spec.batch_attempts, spec.max_attempts_per_task - attempted)
            seed = spec.base_seed + task_id * 1_000_000 + attempted
            jobs.append(
                PlanJob(
                    suite=suite,
                    task_id=task_id,
                    attempt_start=attempted,
                    attempt_count=attempt_count,
                    seed=seed,
                    successes=successes,
                    failures=failures,
                    success_deficit=success_deficit,
                    failure_deficit=failure_deficit,
                    status=status,
                )
            )
    return jobs


def derive_quota_subset(
    spec: CollectionSpec, records: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        validate_record(row)
        if row["producer"] == spec.producer and row["checkpoint_digest"] == spec.checkpoint_digest:
            grouped[(str(row["suite"]), int(row["task_id"]), str(row["outcome"]))].append(row)
    subset: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for suite in spec.suites:
        for task_id in range(SUITE_TASK_COUNTS[suite]):
            for outcome, target in (("success", spec.quota.success), ("failure", spec.quota.failure)):
                candidates = sorted(grouped[(suite, task_id, outcome)], key=record_identity)
                selected = candidates[:target]
                subset.extend(selected)
                if len(selected) < target:
                    incomplete.append(
                        {
                            "suite": suite,
                            "task_id": task_id,
                            "outcome": outcome,
                            "found": len(selected),
                            "target": target,
                        }
                    )
    summary = {
        "schema": "optimus-outcome-quota-summary-v1",
        "status": "complete" if not incomplete else "incomplete",
        "producer": spec.producer,
        "checkpoint_digest": spec.checkpoint_digest,
        "quota": dataclasses.asdict(spec.quota),
        "selected": len(subset),
        "deficits": incomplete,
    }
    return subset, summary


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_records(records: Iterable[dict[str, Any]], *, verify_files: bool) -> dict[str, Any]:
    rows = list(records)
    identities: set[tuple[str, str, str, int, int, int]] = set()
    paths: set[str] = set()
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for row in rows:
        try:
            validate_record(row)
            identity = record_identity(row)
            if identity in identities:
                errors.append(f"duplicate identity: {identity}")
            identities.add(identity)
            path_string = str(row["trajectory_path"])
            if path_string in paths:
                errors.append(f"duplicate trajectory path: {path_string}")
            paths.add(path_string)
            counts[f"{row['suite']}/task_{int(row['task_id']):03d}"][str(row["outcome"])] += 1
            if verify_files:
                path = Path(path_string)
                if not path.is_file():
                    errors.append(f"missing trajectory: {path}")
                elif sha256_file(path) != row["trajectory_sha256"]:
                    errors.append(f"hash mismatch: {path}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    return {
        "schema": "optimus-outcome-audit-v1",
        "status": "ok" if not errors else "failed",
        "records": len(rows),
        "counts": dict(sorted(counts.items())),
        "errors": errors,
    }
