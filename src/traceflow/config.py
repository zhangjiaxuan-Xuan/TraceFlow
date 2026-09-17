"""Release configuration loading and invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load(name: str) -> dict[str, Any]:
    path = repository_root() / "configs" / "release" / f"{name}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    validate(value, name=name)
    return value


def validate(value: dict[str, Any], *, name: str = "config") -> None:
    required = {"schema_version", "benchmark", "protocol", "tasks", "evaluation", "memory"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{name} lacks required fields: {sorted(missing)}")
    if value["schema_version"] != 1:
        raise ValueError(f"{name} has unsupported schema_version")
    evaluation = value["evaluation"]
    for field in ("seed", "episodes_per_task", "batch_size", "env_workers"):
        if not isinstance(evaluation.get(field), int) or evaluation[field] <= 0:
            raise ValueError(f"{name}.evaluation.{field} must be a positive integer")
    memory = value["memory"]
    if memory.get("positive_top_k", 0) < 0 or memory.get("negative_top_k", 0) < 0:
        raise ValueError(f"{name} contains a negative top-k")
    if len(value["tasks"]) != len(set(value["tasks"])):
        raise ValueError(f"{name} contains duplicate tasks")

