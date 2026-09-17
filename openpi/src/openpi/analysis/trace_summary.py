from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else math.nan


def _normalize_task(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _retrieval_metrics(retrieval: dict[str, Any], query_task: str) -> dict[str, float]:
    names = [_normalize_task(value) for value in retrieval.get("task_names", [])]
    scores = np.asarray(retrieval.get("scores", []), dtype=np.float64)
    weights = np.asarray(retrieval.get("weights", []), dtype=np.float64)
    query = _normalize_task(query_task)
    matches = np.asarray([bool(query) and name == query for name in names], dtype=np.bool_)
    purity = float(matches.mean()) if matches.size else math.nan
    margin = math.nan
    if scores.size == matches.size and np.any(matches) and np.any(~matches):
        margin = float(scores[matches].max() - scores[~matches].max())
    if weights.size:
        total = float(weights.sum())
        normalized = weights / total if total > 0 else weights
        ess = float(1.0 / np.square(normalized).sum()) if np.square(normalized).sum() > 0 else math.nan
    else:
        ess = math.nan
    return {"purity": purity, "margin": margin, "ess": ess, "selected": float(len(names))}


def _scalar_metric(step: dict[str, Any], name: str, metric: str) -> float:
    values = step.get(name, {}).get(metric, [])
    return float(values[0]) if values else math.nan


def summarize_light_trace(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("schema_version") != 2 or record.get("trace_level") != "light":
        raise ValueError("Expected a schema-v2 light memory-guidance trace")
    task_name = str(record.get("task_name", ""))
    positive = _retrieval_metrics(record.get("positive_retrieval", {}), task_name)
    negative = _retrieval_metrics(record.get("negative_retrieval", {}), task_name)
    steps = list(record.get("steps", []))
    base_norm = _mean(_scalar_metric(step, "v_base", "norm") for step in steps)
    guidance_norm = _mean(_scalar_metric(step, "guidance_final", "norm") for step in steps)
    ratio = guidance_norm / base_norm if np.isfinite(base_norm) and base_norm > 0 else math.nan
    return {
        "task_suite": str(record.get("task_suite", "")),
        "task_id": int(record["task_id"]),
        "episode_idx": int(record["episode_idx"]),
        "policy_call_idx": int(record["policy_call_idx"]),
        "task_name": task_name,
        "progress": float(record.get("progress", math.nan)),
        "purity": positive["purity"],
        "margin": positive["margin"],
        "retrieval_ess": positive["ess"],
        "retrieval_selected": positive["selected"],
        "negative_purity": negative["purity"],
        "negative_margin": negative["margin"],
        "negative_retrieval_ess": negative["ess"],
        "negative_retrieval_selected": negative["selected"],
        "base_velocity_norm": base_norm,
        "guidance_norm": guidance_norm,
        "guidance_norm_ratio": ratio,
        "guidance_cosine": _mean(
            _scalar_metric(step, "guidance_final", "cosine_to_v_base") for step in steps
        ),
        "positive_guidance_norm": _mean(
            _scalar_metric(step, "guidance_positive", "norm") for step in steps
        ),
        "negative_guidance_norm": _mean(
            _scalar_metric(step, "guidance_negative", "norm") for step in steps
        ),
    }


def load_episode_results(path: str | Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "episode_result":
                continue
            key = (str(row.get("task_suite", "")), int(row["task_id"]), int(row["episode_idx"]))
            if key in results and results[key] != row:
                raise ValueError(f"Conflicting episode result in {path}:{line_number}: {key}")
            results[key] = row
    if not results:
        raise ValueError(f"No episode_result rows in {path}")
    return results


def summarize_trace_directory(
    trace_dir: str | Path,
    eval_log: str | Path,
    *,
    base_eval_log: str | Path | None = None,
    condition: str = "",
    memory_group: str = "",
) -> list[dict[str, Any]]:
    calls: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path(trace_dir).glob("*.light.jsonl")):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"Light trace must contain exactly one JSON record: {path}")
        call = summarize_light_trace(json.loads(lines[0]))
        key = (call["task_suite"], call["task_id"], call["episode_idx"])
        calls[key].append(call)
    if not calls:
        raise ValueError(f"No *.light.jsonl traces in {trace_dir}")
    outcomes = load_episode_results(eval_log)
    base = load_episode_results(base_eval_log) if base_eval_log else {}
    rows: list[dict[str, Any]] = []
    numeric = (
        "purity",
        "margin",
        "retrieval_ess",
        "retrieval_selected",
        "negative_purity",
        "negative_margin",
        "negative_retrieval_ess",
        "negative_retrieval_selected",
        "base_velocity_norm",
        "guidance_norm",
        "guidance_norm_ratio",
        "guidance_cosine",
        "positive_guidance_norm",
        "negative_guidance_norm",
    )
    for key, episode_calls in sorted(calls.items()):
        if key not in outcomes:
            raise ValueError(f"Trace has no matching episode_result: {key}")
        outcome = outcomes[key]
        row: dict[str, Any] = {
            "task_suite": key[0],
            "task_id": key[1],
            "episode_idx": key[2],
            "episode_cluster": f"{key[0]}:{key[1]}:{key[2]}",
            "condition": condition,
            "memory_group": memory_group,
            "policy_calls": len(episode_calls),
            "success": int(bool(outcome["success"])),
        }
        if key in base:
            row["base_success"] = int(bool(base[key]["success"]))
            row["success_flip"] = row["success"] - row["base_success"]
        for name in numeric:
            row[name] = _mean(call[name] for call in episode_calls)
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(destination)
