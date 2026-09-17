#!/usr/bin/env python3
"""Grouped validation of Arena cap-gate signals against V0/V1 outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _results(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["task"] != "ALL":
                rows[int(row["task"])] = {"TSR": float(row["TSR"]), "CSR": float(row["CSR"])}
    return rows


def _ridge_group_cv(x: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, alpha: float = 1.0) -> dict:
    prediction = torch.empty_like(y)
    for group in groups.unique():
        test = groups == group
        train = ~test
        mean = x[train].mean(0)
        scale = x[train].std(0).clamp_min(1e-6)
        z = (x[train] - mean) / scale
        y_mean = y[train].mean()
        eye = torch.eye(z.shape[1], device=x.device)
        weight = torch.linalg.solve(z.T @ z + alpha * eye, z.T @ (y[train] - y_mean))
        prediction[test] = ((x[test] - mean) / scale) @ weight + y_mean
    null = torch.empty_like(y)
    for group in groups.unique():
        test = groups == group
        null[test] = y[~test].mean()
    return {
        "grouped_mae": float((prediction - y).abs().mean()),
        "null_mae": float((null - y).abs().mean()),
        "mae_gain": float((null - y).abs().mean() - (prediction - y).abs().mean()),
        "sign_accuracy": float(((prediction > 0) == (y > 0)).float().mean()),
        "pearson": float(torch.corrcoef(torch.stack([prediction, y]))[0, 1]),
        "predictions": prediction.tolist(),
    }


def _logistic_group_cv(x: torch.Tensor, y: torch.Tensor, groups: torch.Tensor) -> dict:
    probabilities = torch.empty_like(y)
    null_probabilities = torch.empty_like(y)
    for group in groups.unique():
        test = groups == group
        train = ~test
        mean = x[train].mean(0)
        scale = x[train].std(0).clamp_min(1e-6)
        z = (x[train] - mean) / scale
        weight = torch.zeros(z.shape[1], device=x.device, requires_grad=True)
        bias = torch.zeros((), device=x.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(z @ weight + bias, y[train])
            loss = loss + 0.2 * weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        probabilities[test] = torch.sigmoid(((x[test] - mean) / scale) @ weight + bias)
        null_probabilities[test] = (y[train].sum() + 0.5) / (train.sum() + 1.0)
    return {
        "grouped_cross_entropy": float(F.binary_cross_entropy(probabilities, y)),
        "null_cross_entropy": float(F.binary_cross_entropy(null_probabilities, y)),
        "cross_entropy_gain": float(
            F.binary_cross_entropy(null_probabilities, y) - F.binary_cross_entropy(probabilities, y)
        ),
        "accuracy": float(((probabilities >= 0.5) == y.bool()).float().mean()),
        "probabilities": probabilities.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-metrics", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--v0-root", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    task_metrics = {int(row["task_id"]): row for row in csv.DictReader(args.task_metrics.open())}
    records = []
    for head_id, head in enumerate(("upper", "fusion")):
        trace = json.loads((args.trace_root / f"{head}.json").read_text())
        v0 = _results(args.v0_root / head / "results.txt")
        v1 = _results(args.v1_root / head / "results.txt")
        for suite in trace["suites"].values():
            for task_string, values in suite["per_task"].items():
                task_id = int(task_string)
                step0 = values["by_denoise_step"][0]
                record = {
                    "task_id": task_id,
                    "suite": task_metrics[task_id]["suite"],
                    "head": head,
                    "head_id": head_id,
                    "separability": float(task_metrics[task_id]["suite_separation_ratio"]),
                    "alignment": float(values["guidance_base_cosine_mean"]),
                    "step0_alignment": float(step0["guidance_base_cosine_mean"]),
                    "effective_cap": float(values["effective_cap_mean"]),
                    "parallel_cap": float(values["parallel_effective_cap_mean"]),
                    "orthogonal_cap": float(values["orthogonal_effective_cap_mean"]),
                    "retrieval_entropy": float(values["retrieval_weight_entropy_mean"]),
                    "retrieval_margin": float(values["retrieval_top1_top2_margin_mean"]),
                    "memory_coherence": float(values["memory_action_directional_coherence_mean"]),
                    "responsibility_entropy": float(values["responsibility_entropy_mean"]),
                    "v0_minus_v1_tsr": v0[task_id]["TSR"] - v1[task_id]["TSR"],
                    "v0_minus_v1_csr": v0[task_id]["CSR"] - v1[task_id]["CSR"],
                }
                record["cap_gate_product"] = record["separability"] * record["step0_alignment"]
                records.append(record)

    feature_sets = {
        "separability_only": ["separability"],
        "alignment_only": ["step0_alignment"],
        "two_signal_gate": ["separability", "step0_alignment", "cap_gate_product"],
        "trace_geometry": [
            "separability",
            "step0_alignment",
            "parallel_cap",
            "orthogonal_cap",
            "retrieval_entropy",
            "retrieval_margin",
            "memory_coherence",
            "responsibility_entropy",
            "head_id",
        ],
    }
    groups = torch.tensor([row["task_id"] for row in records], device=device)
    tests = {}
    for target_name in ("v0_minus_v1_tsr", "v0_minus_v1_csr"):
        target = torch.tensor([row[target_name] for row in records], dtype=torch.float32, device=device)
        target_tests = {}
        for name, columns in feature_sets.items():
            feature = torch.tensor(
                [[row[column] for column in columns] for row in records], dtype=torch.float32, device=device
            )
            target_tests[name] = {
                "columns": columns,
                "regression": _ridge_group_cv(feature, target, groups),
                "positive_effect_classification": _logistic_group_cv(
                    feature, (target > 0).float(), groups
                ),
            }
        tests[target_name] = target_tests

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "task_head_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "protocol": "arena_cap_gate_grouped_task_cv_v1",
        "device": torch.cuda.get_device_name(device),
        "grouping": "leave both Upper/Fusion rows of one task out together",
        "rows": len(records),
        "unique_tasks": len(set(row["task_id"] for row in records)),
        "tests": tests,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
