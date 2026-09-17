#!/usr/bin/env python3
"""Compile the offline two-metric gate to the existing V1 serve schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.fit_config.read_text())
    if payload.get("name") != "predimem_two_metric_gate_v1":
        raise ValueError(f"Unexpected fit config: {payload.get('name')}")
    rows = payload.get("task_gate")
    if not isinstance(rows, dict) or set(rows) != {str(i) for i in range(1, 27)}:
        raise ValueError("Fit config must contain task_gate entries for tasks 1..26")

    profiles = {}
    for task_id, row in rows.items():
        cap = float(row["v1_equivalent_cap"])
        cutoff = float(row["guidance_cutoff"])
        if not 0.0 <= cap <= 1.0:
            raise ValueError(f"Invalid cap for task {task_id}: {cap}")
        if not 0.0 <= cutoff < 1.0:
            raise ValueError(f"Invalid cutoff for task {task_id}: {cutoff}")
        profiles[task_id] = {
            "source": "predimem_two_metric_gate_v1",
            "separability": float(row["separability"]),
            "alignment": float(row["full_length_alignment"]),
            "eligibility": float(row["eligibility"]),
            "upper_v0_weight": float(row["upper_v0_weight"]),
            "fusion_v1_weight": float(row["fusion_v1_weight"]),
            "v0_lambda_max": 0.20,
            "v1_norm_cap": cap,
            "t_cut": cutoff,
        }

    result = {
        "schema": "arena_suite_guidance_gate_v1",
        "description": (
            "Compatibility profile compiled from the offline PrediMem two-metric fit. "
            "Runtime mode is V1 Fusion; this file does not implement Upper/Fusion mixing."
        ),
        "source_fit_config": str(args.fit_config),
        "task_profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tasks": len(profiles)}, indent=2))


if __name__ == "__main__":
    main()
