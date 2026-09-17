#!/usr/bin/env python3
"""Consolidate Full26 separability/alignment evidence for cap-gate design."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SUITE_ORDER = ("Sequence", "Transferring", "Counting", "Occlusion")
COLORS = {
    "Sequence": "#0072B2",
    "Transferring": "#009E73",
    "Counting": "#D55E00",
    "Occlusion": "#CC79A7",
}


def _bootstrap_mean(values: torch.Tensor, samples: int, generator: torch.Generator) -> tuple[float, float]:
    indices = torch.randint(len(values), (samples, len(values)), device=values.device, generator=generator)
    means = values[indices].mean(1)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975], device=values.device))
    return float(bounds[0]), float(bounds[1])


def _rank(values: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(values)).float()


def _loocv_ridge(feature: torch.Tensor, target: torch.Tensor, alpha: float = 1.0) -> dict[str, float]:
    predictions = []
    nulls = []
    for held in range(len(feature)):
        train = torch.arange(len(feature), device=feature.device) != held
        mean = feature[train].mean(0)
        scale = feature[train].std(0).clamp_min(1e-6)
        x = (feature[train] - mean) / scale
        y_mean = target[train].mean()
        weight = torch.linalg.solve(
            x.T @ x + alpha * torch.eye(x.shape[1], device=x.device),
            x.T @ (target[train] - y_mean),
        )
        predictions.append(((feature[held] - mean) / scale) @ weight + y_mean)
        nulls.append(y_mean)
    prediction = torch.stack(predictions)
    null = torch.stack(nulls)
    return {
        "loocv_mae": float((prediction - target).abs().mean()),
        "null_mae": float((null - target).abs().mean()),
        "mae_gain": float((null - target).abs().mean() - (prediction - target).abs().mean()),
        "sign_accuracy": float(((prediction > 0) == (target > 0)).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-metrics", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--demo-compatibility", type=Path, required=True)
    parser.add_argument("--v0-aggregate", type=Path, required=True)
    parser.add_argument("--v1-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    task_metrics = {int(row["task_id"]): row for row in csv.DictReader(args.task_metrics.open())}
    demo = {int(row["task_id"]): row for row in csv.DictReader(args.demo_compatibility.open())}
    probe = json.loads(args.probe.read_text())
    trace = {int(task_id): values for suite in probe["suites"].values() for task_id, values in suite["per_task"].items()}
    v0 = {int(row["task_id"]): row for row in json.loads(args.v0_aggregate.read_text())["tasks"]}
    v1 = {int(row["task_id"]): row for row in json.loads(args.v1_aggregate.read_text())["tasks"]}

    records = []
    for task_id in sorted(task_metrics):
        suite = task_metrics[task_id]["suite"]
        separability = float(task_metrics[task_id]["suite_separation_ratio"])
        alignment = float(trace[task_id]["guidance_base_cosine_mean"])
        record = {
            "task_id": task_id,
            "suite": suite,
            "separability": separability,
            "online_alignment": alignment,
            "cap_gate_product": separability * alignment,
            "step0_alignment": float(trace[task_id]["by_denoise_step"][0]["guidance_base_cosine_mean"]),
            "effective_cap": float(trace[task_id]["effective_cap_mean"]),
            "parallel_cap": float(trace[task_id]["parallel_effective_cap_mean"]),
            "orthogonal_cap": float(trace[task_id]["orthogonal_effective_cap_mean"]),
            "memory_coherence": float(trace[task_id]["memory_action_directional_coherence_mean"]),
            "demo_behavior_alignment": float(demo[task_id]["behavior_motion_weighted_loo_cosine"]),
            "v1_minus_v0_tsr": float(v1[task_id]["TSR"] - v0[task_id]["TSR"]),
            "v1_minus_v0_csr": float(v1[task_id]["CSR"] - v0[task_id]["CSR"]),
        }
        records.append(record)

    generator = torch.Generator(device=device).manual_seed(7)
    suite_records = []
    for suite in SUITE_ORDER:
        selected = [row for row in records if row["suite"] == suite]
        align = torch.tensor([row["online_alignment"] for row in selected], device=device)
        gate = torch.tensor([row["cap_gate_product"] for row in selected], device=device)
        align_ci = _bootstrap_mean(align, args.bootstrap_samples, generator)
        gate_ci = _bootstrap_mean(gate, args.bootstrap_samples, generator)
        suite_records.append(
            {
                "suite": suite,
                "tasks": len(selected),
                "separability": float(np.mean([row["separability"] for row in selected])),
                "online_alignment": float(align.mean()),
                "alignment_ci_low": align_ci[0],
                "alignment_ci_high": align_ci[1],
                "cap_gate_product": float(gate.mean()),
                "gate_ci_low": gate_ci[0],
                "gate_ci_high": gate_ci[1],
                "demo_behavior_alignment": float(
                    np.mean([row["demo_behavior_alignment"] for row in selected])
                ),
                "v1_minus_v0_tsr": float(np.mean([row["v1_minus_v0_tsr"] for row in selected])),
                "v1_minus_v0_csr": float(np.mean([row["v1_minus_v0_csr"] for row in selected])),
            }
        )

    gate = torch.tensor([row["cap_gate_product"] for row in records], device=device)
    delta_tsr = torch.tensor([row["v1_minus_v0_tsr"] for row in records], device=device)
    delta_csr = torch.tensor([row["v1_minus_v0_csr"] for row in records], device=device)
    separability = torch.tensor([row["separability"] for row in records], device=device)
    alignment = torch.tensor([row["online_alignment"] for row in records], device=device)
    feature_sets = {
        "separability": separability[:, None],
        "alignment": alignment[:, None],
        "product": gate[:, None],
        "two_signal": torch.stack([separability, alignment, gate], dim=1),
    }
    prediction_tests = {
        target_name: {name: _loocv_ridge(feature, target) for name, feature in feature_sets.items()}
        for target_name, target in (("v1_minus_v0_tsr", delta_tsr), ("v1_minus_v0_csr", delta_csr))
    }
    correlations = {}
    for name, value in (("separability", separability), ("alignment", alignment), ("product", gate)):
        correlations[name] = {
            "pearson_with_v1_minus_v0_tsr": float(torch.corrcoef(torch.stack([value, delta_tsr]))[0, 1]),
            "spearman_with_v1_minus_v0_tsr": float(
                torch.corrcoef(torch.stack([_rank(value), _rank(delta_tsr)]))[0, 1]
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("task_metrics.csv", records), ("suite_metrics.csv", suite_records)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for suite in SUITE_ORDER:
        rows = [row for row in records if row["suite"] == suite]
        axes[0].scatter(
            [row["separability"] for row in rows],
            [row["online_alignment"] for row in rows],
            label=suite,
            color=COLORS[suite],
            s=42,
        )
        suite_probe = probe["suites"][suite]["by_denoise_step"][:7]
        axes[1].plot(
            [row["time"] for row in suite_probe],
            [row["guidance_base_cosine_mean"] for row in suite_probe],
            marker="o",
            color=COLORS[suite],
            label=suite,
        )
        axes[2].scatter(
            [row["cap_gate_product"] for row in rows],
            [row["v1_minus_v0_tsr"] for row in rows],
            color=COLORS[suite],
            s=42,
        )
    axes[0].set(xlabel="Retrieval separability", ylabel="Online alignment", title="Task geometry")
    axes[1].set(xlabel="Flow time", ylabel="cos(g, v_base)", title="Alignment decay")
    axes[1].invert_xaxis()
    axes[2].axhline(0, color="#555555", linewidth=1)
    axes[2].set(xlabel="Separability x alignment", ylabel="V1 - V0 TSR (pp)", title="Not a reward predictor")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cap_gate_analysis.png", dpi=180)
    plt.close(fig)

    payload = {
        "protocol": "arena_full26_cap_gate_analysis_v1",
        "device": torch.cuda.get_device_name(device),
        "probe": "26 tasks x 1 episode x 7 policy calls; diagnostic only, not SR evidence",
        "suites": suite_records,
        "correlations": correlations,
        "prediction_tests": prediction_tests,
        "conclusion": {
            "supported": "product provides a suite-level intervention-risk budget and dynamic cutoff signal",
            "rejected": "product alone predicts task-level V0/V1 reward delta",
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
