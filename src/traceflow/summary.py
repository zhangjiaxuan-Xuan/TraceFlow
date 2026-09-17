"""Aggregate suite result files without depending on simulator packages."""

from __future__ import annotations

import json
from pathlib import Path
import re


def summarize_libero(run_root: Path, output: Path) -> dict[str, object]:
    suites: dict[str, dict[str, float | int]] = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        directory = run_root / suite
        rows: list[bool] = []
        episodes_by_key: dict[tuple[object, ...], bool] = {}
        combined = directory / "eval" / f"{suite}.jsonl"
        candidates = [combined] if combined.is_file() else list(directory.rglob("*.jsonl")) if directory.exists() else []
        for candidate in candidates:
            for line in candidate.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = row.get("event")
                if "success" not in row or event not in (None, "episode_result"):
                    continue
                if event == "episode_result" and "task_id" in row and "episode_idx" in row:
                    key = (row.get("task_suite", suite), row["task_id"], row["episode_idx"], row.get("seed"))
                    episodes_by_key[key] = bool(row["success"])
                else:
                    rows.append(bool(row["success"]))
        rows.extend(episodes_by_key.values())
        if not rows:
            text_files = list(directory.rglob("results.txt")) if directory.exists() else []
            for candidate in text_files:
                match = re.search(r"(\d+)\s*/\s*(\d+)", candidate.read_text(encoding="utf-8"))
                if match:
                    successes, episodes = map(int, match.groups())
                    rows = [True] * successes + [False] * (episodes - successes)
                    break
        if not rows:
            raise FileNotFoundError(f"no completed result rows for {suite} below {directory}")
        suites[suite] = {"successes": sum(rows), "episodes": len(rows), "success_rate": sum(rows) / len(rows)}
    successes = sum(int(row["successes"]) for row in suites.values())
    episodes = sum(int(row["episodes"]) for row in suites.values())
    result = {
        "metric": "best-of-suite envelope reproduction",
        "warning": "The published 98.3% is an envelope of independent suite configurations, not one 1966/2000 run.",
        "suites": suites,
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
