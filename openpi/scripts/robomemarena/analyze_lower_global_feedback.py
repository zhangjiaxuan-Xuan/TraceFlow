from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def _mean(values: list[float]) -> float | None:
    return float(fmean(values)) if values else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [call for row in rows for call in row.get("upper_guidance", [])]
    available = [call for call in calls if call.get("lower_global_available", False)]
    accepted = [call for call in available if call.get("lower_global_gate_passed", False)]
    applied = [call for call in available if call.get("lower_global_applied", False)]
    scores = [float(call["lower_global_best_score"]) for call in available if call.get("lower_global_best_score") is not None]
    posteriors = [
        float(call["lower_global_best_posterior"])
        for call in available
        if call.get("lower_global_best_posterior") is not None
    ]
    margins = [float(call["lower_global_margin"]) for call in available if call.get("lower_global_margin") is not None]
    trajectory_concentration = []
    progress_std = []
    for call in available:
        ranked = call.get("lower_global_scores", [])
        if not ranked:
            continue
        trajectory_concentration.append(float(ranked[0].get("trajectory_concentration", 0.0)))
        progress_std.append(float(ranked[0].get("progress_std", 0.0)))
    return {
        "episodes": len(rows),
        "tsr": _mean([float(row.get("TSR", 0.0)) for row in rows]),
        "csr": _mean([float(row.get("CSR", 0.0)) for row in rows]),
        "calls": len(calls),
        "available_calls": len(available),
        "available_rate": len(available) / len(calls) if calls else 0.0,
        "gate_accept_calls": len(accepted),
        "gate_accept_rate": len(accepted) / len(available) if available else 0.0,
        "applied_calls": len(applied),
        "output_override_calls": sum(bool(call.get("lower_global_output_overridden", False)) for call in calls),
        "mean_best_score": _mean(scores),
        "mean_best_posterior": _mean(posteriors),
        "mean_candidate_margin": _mean(margins),
        "mean_trajectory_concentration": _mean(trajectory_concentration),
        "mean_progress_std": _mean(progress_std),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Lower-global feedback diagnostics from PrediMem eval logs")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    paths = sorted(args.run_root.glob("**/worker_*.jsonl"))
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                by_task[int(row["task_id"])].append(row)
    if not by_task:
        raise FileNotFoundError(f"No worker JSONL records below {args.run_root}")

    all_rows = [row for rows in by_task.values() for row in rows]
    report = {
        "run_root": str(args.run_root),
        "overall": _summarize(all_rows),
        "tasks": {str(task_id): _summarize(rows) for task_id, rows in sorted(by_task.items())},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
