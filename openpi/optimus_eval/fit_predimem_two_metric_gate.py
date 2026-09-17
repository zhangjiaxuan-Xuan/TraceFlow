#!/usr/bin/env python3
"""Fit the two-metric PrediMem gate from task metrics and observed outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


POSITIVE_SUITES = {"Sequence", "Transferring"}
UPPER_V0_SUITES = {"Sequence"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-metrics", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--alignment-column", default="full_length_alignment")
    parser.add_argument("--v0-aggregate", type=Path, required=True)
    parser.add_argument("--v1-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--v1-cap", type=float, default=0.2)
    parser.add_argument("--v0-cap", type=float, default=1.0)
    parser.add_argument("--base-cutoff", type=float, default=0.3)
    parser.add_argument("--max-cutoff", type=float, default=0.6)
    parser.add_argument("--l2", type=float, default=0.005)
    return parser.parse_args()


def _aggregate(path: Path) -> dict[int, dict]:
    return {int(row["task_id"]): row for row in json.loads(path.read_text())["tasks"]}


def _weighted_logistic_fit(
    feature: np.ndarray,
    target: np.ndarray,
    group: list[str],
    *,
    l2: float,
    device: torch.device,
) -> dict:
    x = torch.tensor(feature, dtype=torch.float64, device=device)
    y = torch.tensor(target, dtype=torch.float64, device=device)
    counts = Counter(group)
    weights = torch.tensor(
        [1.0 / counts[value] for value in group], dtype=torch.float64, device=device
    )
    weights = weights / weights.mean()
    mean = x.mean()
    scale = x.std().clamp_min(1e-8)
    z = (x - mean) / scale
    parameter = torch.zeros(2, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameter], lr=0.5, max_iter=200, tolerance_grad=1e-12, tolerance_change=1e-12
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = parameter[0] + parameter[1] * z
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            * weights
        ).mean() + l2 * parameter[1].square()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        intercept_z, slope_z = parameter
        slope = slope_z / scale
        intercept = intercept_z - slope_z * mean / scale
        probability = torch.sigmoid(intercept + slope * x)
        threshold = -intercept / slope
        temperature = 1.0 / slope.abs()
        logits = parameter[0] + parameter[1] * z
        balanced_bce = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            * weights
        ).mean() + l2 * parameter[1].square()
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "threshold": float(threshold),
        "temperature": float(temperature),
        "probability": probability.cpu().numpy(),
        "balanced_bce": float(balanced_bce),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --device cuda")
    task_metrics = {
        int(row["task_id"]): row for row in csv.DictReader(args.task_metrics.open())
    }
    alignment = {
        int(row["task_id"]): row for row in csv.DictReader(args.alignment.open())
    }
    v0 = _aggregate(args.v0_aggregate)
    v1 = _aggregate(args.v1_aggregate)
    task_ids = sorted(set(task_metrics) & set(alignment) & set(v0) & set(v1))
    if task_ids != list(range(1, 27)):
        raise RuntimeError(f"Expected task IDs 1..26, found {task_ids}")

    suites = [task_metrics[task_id]["suite"] for task_id in task_ids]
    separability = np.asarray(
        [float(task_metrics[task_id]["suite_separation_ratio"]) for task_id in task_ids]
    )
    align = np.asarray(
        [float(alignment[task_id][args.alignment_column]) for task_id in task_ids]
    )
    eligible_target = np.asarray([suite in POSITIVE_SUITES for suite in suites], dtype=np.float64)
    eligible_fit = _weighted_logistic_fit(
        separability, eligible_target, suites, l2=args.l2, device=device
    )

    positive = np.asarray([suite in POSITIVE_SUITES for suite in suites])
    upper_target = np.asarray(
        [suite in UPPER_V0_SUITES for suite in np.asarray(suites)[positive]], dtype=np.float64
    )
    upper_fit = _weighted_logistic_fit(
        align[positive],
        upper_target,
        list(np.asarray(suites)[positive]),
        l2=args.l2,
        device=device,
    )
    upper_probability = np.zeros(len(task_ids), dtype=np.float64)
    upper_probability[positive] = upper_fit.pop("probability")
    eligibility = eligible_fit.pop("probability")

    equivalent_cap = eligibility * (
        args.v1_cap + (args.v0_cap - args.v1_cap) * upper_probability
    )
    cutoff = args.base_cutoff + (args.max_cutoff - args.base_cutoff) * (1.0 - eligibility)
    upper_branch_weight = eligibility * upper_probability
    fusion_branch_weight = eligibility * (1.0 - upper_probability)

    rows = []
    for index, task_id in enumerate(task_ids):
        rows.append(
            {
                "task_id": task_id,
                "suite": suites[index],
                "separability": separability[index],
                "full_length_alignment": align[index],
                "eligibility": eligibility[index],
                "upper_v0_weight": upper_branch_weight[index],
                "fusion_v1_weight": fusion_branch_weight[index],
                "v1_equivalent_cap": equivalent_cap[index],
                "guidance_cutoff": cutoff[index],
                "v0_tsr": float(v0[task_id]["TSR"]),
                "v1_tsr": float(v1[task_id]["TSR"]),
                "v1_minus_v0_tsr": float(v1[task_id]["TSR"] - v0[task_id]["TSR"]),
            }
        )

    suite_rows = []
    for suite in dict.fromkeys(suites):
        selected = [row for row in rows if row["suite"] == suite]
        suite_rows.append(
            {
                "suite": suite,
                "tasks": len(selected),
                **{
                    key: float(np.mean([row[key] for row in selected]))
                    for key in (
                        "separability",
                        "full_length_alignment",
                        "eligibility",
                        "upper_v0_weight",
                        "fusion_v1_weight",
                        "v1_equivalent_cap",
                        "guidance_cutoff",
                        "v0_tsr",
                        "v1_tsr",
                    )
                },
            }
        )

    config = {
        "schema_version": 1,
        "name": "predimem_two_metric_gate_v1",
        "metric_definitions": {
            "separability": "Full-trajectory, own-length-normalized demonstration suite separation ratio.",
            "alignment": f"Column {args.alignment_column} from {args.alignment}; expected to be full-trajectory and own-length normalized.",
        },
        "fit": {
            "eligibility": eligible_fit,
            "upper_direction": upper_fit,
            "balanced_by_suite": True,
            "l2": args.l2,
        },
        "mapping": {
            "eligibility": "sigmoid((S_sep - eligibility.threshold) / eligibility.temperature)",
            "upper_probability": "sigmoid((A_align - upper_direction.threshold) / upper_direction.temperature)",
            "upper_v0_weight": "eligibility * upper_probability",
            "fusion_v1_weight": "eligibility * (1 - upper_probability)",
            "actual_velocity": "v_base + upper_v0_weight * g_upper_raw + fusion_v1_weight * cap_relative(g_fusion_raw, v_base, 0.2)",
            "v1_equivalent_cap": f"eligibility * ({args.v1_cap} + ({args.v0_cap} - {args.v1_cap}) * upper_probability); diagnostic only",
            "guidance_cutoff": f"{args.base_cutoff} + ({args.max_cutoff} - {args.base_cutoff}) * (1 - eligibility)",
        },
        "task_gate": {str(row["task_id"]): row for row in rows},
        "suite_summary": suite_rows,
        "outcome_provenance": {
            "v0": str(args.v0_aggregate),
            "v1": str(args.v1_aggregate),
            "label_policy": {
                "Sequence": "Upper-V0 target, selected by completed PrediMem Extra-8 head/mode comparison.",
                "Transferring": "Fusion-V1 target, selected by completed PrediMem Extra-8 head/mode comparison.",
                "Counting": "minimal-intervention target; Full26 guidance remains below the paper-level aggregate and has no registered suite baseline.",
                "Occlusion": "minimal-intervention target; Full26 guidance remains below the paper-level aggregate and has no registered suite baseline.",
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name, data in (("task_gate.csv", rows), ("suite_gate.csv", suite_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({"fit": config["fit"], "suites": suite_rows}, indent=2))


if __name__ == "__main__":
    main()
