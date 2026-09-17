#!/usr/bin/env python3
"""Simulate a suite-aware V1 guidance gate from retrieval separability."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


FAVORABLE_SUITES = {"Sequence", "Transferring"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-separability", type=float, default=0.90)
    parser.add_argument("--full-guidance-separability", type=float, default=1.45)
    parser.add_argument("--min-cap", type=float, default=0.05)
    parser.add_argument("--full-cap", type=float, default=0.20)
    parser.add_argument("--full-cutoff-steps", type=int, default=3)
    parser.add_argument("--max-cutoff-steps", type=int, default=6)
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.task_metrics.open()))
    suites = sorted({row["suite"] for row in rows})
    device = torch.device(args.device)

    suite_separability = torch.tensor(
        [
            sum(float(row["suite_separation_ratio"]) for row in rows if row["suite"] == suite)
            / sum(row["suite"] == suite for row in rows)
            for suite in suites
        ],
        dtype=torch.float32,
        device=device,
    )
    denominator = args.full_guidance_separability - args.low_separability
    if denominator <= 0:
        raise ValueError("full-guidance-separability must exceed low-separability")

    confidence = ((suite_separability - args.low_separability) / denominator).clamp(0.0, 1.0)
    caps = args.min_cap + confidence * (args.full_cap - args.min_cap)
    cutoff_float = args.max_cutoff_steps - confidence * (
        args.max_cutoff_steps - args.full_cutoff_steps
    )
    cutoff_steps = torch.round(cutoff_float).to(torch.int64)

    records = []
    for index, suite in enumerate(suites):
        record = {
            "suite": suite,
            "tasks": sum(row["suite"] == suite for row in rows),
            "retrieval_separability": float(suite_separability[index]),
            "guidance_confidence": float(confidence[index]),
            "guidance_norm_cap": float(caps[index]),
            "guidance_cutoff_steps": int(cutoff_steps[index]),
            "memory_guidance_t_cut": int(cutoff_steps[index]) / args.num_denoise_steps,
            "expected_full_guidance": suite in FAVORABLE_SUITES,
        }
        record["meets_expected_bucket"] = (
            record["guidance_norm_cap"] >= args.full_cap - 1e-6
            and record["guidance_cutoff_steps"] == args.full_cutoff_steps
            if record["expected_full_guidance"]
            else record["guidance_norm_cap"] < args.full_cap
            and record["guidance_cutoff_steps"] > args.full_cutoff_steps
        )
        records.append(record)

    payload = {
        "protocol": "arena_suite_retrieval_separability_v1_gate_simulation",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "mapping": {
            "confidence": "clip((suite_separability - low) / (full - low), 0, 1)",
            "low_separability": args.low_separability,
            "full_guidance_separability": args.full_guidance_separability,
            "min_cap": args.min_cap,
            "full_cap": args.full_cap,
            "full_cutoff_steps": args.full_cutoff_steps,
            "max_cutoff_steps": args.max_cutoff_steps,
            "num_denoise_steps": args.num_denoise_steps,
        },
        "suites": records,
        "all_expected_buckets_met": all(record["meets_expected_bucket"] for record in records),
        "warning": "This tests parameter separability only; it does not establish causal SR improvement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
