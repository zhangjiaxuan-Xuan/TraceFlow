from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from optimus_eval.trajectory_quality_audit import Trajectory, _behavior_progress, load_arena


def _resample_gpu(values: torch.Tensor, positions: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    indices = torch.searchsorted(positions, grid).clamp(1, len(positions) - 1)
    left = indices - 1
    right = indices
    denominator = (positions[right] - positions[left]).clamp_min(1e-8)
    weight = ((grid - positions[left]) / denominator).unsqueeze(1)
    return values[left] * (1.0 - weight) + values[right] * weight


def _standardize_trajectories(items: list[Trajectory], device: torch.device, grid_size: int):
    arm_dim = min(item.actions.shape[1] - 1 for item in items)
    pooled = np.concatenate([item.actions[:, :arm_dim] for item in items], axis=0)
    mean = torch.as_tensor(pooled.mean(0), dtype=torch.float32, device=device)
    scale = torch.as_tensor(pooled.std(0), dtype=torch.float32, device=device).clamp_min(1e-6)
    grid = torch.linspace(0.0, 1.0, grid_size, device=device)
    rows = []
    samples = []
    for item in items:
        action = torch.as_tensor(item.actions[:, :arm_dim], dtype=torch.float32, device=device)
        action = (action - mean) / scale
        time = torch.linspace(0.0, 1.0, len(action), device=device)
        behavior = torch.as_tensor(_behavior_progress(item.actions), dtype=torch.float32, device=device)
        # searchsorted requires a nondecreasing coordinate; repeated behavior
        # positions are separated by the smallest representable forward step.
        behavior = torch.maximum(behavior, torch.arange(len(behavior), device=device) * 1e-8)
        behavior = behavior / behavior[-1].clamp_min(1e-8)
        time_row = _resample_gpu(action, time, grid)
        behavior_row = _resample_gpu(action, behavior, grid)
        rows.append((item.task_id, time_row, behavior_row))
        take = torch.linspace(0, len(action) - 1, min(256, len(action)), device=device).long()
        samples.append((item.task_id, action[take]))
    return rows, samples


def _gpu_kmeans(values: torch.Tensor, clusters: int, iterations: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=values.device).manual_seed(seed)
    centers = values[torch.randperm(len(values), generator=generator, device=values.device)[:clusters]].clone()
    for _ in range(iterations):
        totals = torch.zeros_like(centers)
        counts = torch.zeros(clusters, dtype=torch.float32, device=values.device)
        for batch in values.split(32768):
            labels = torch.cdist(batch, centers).argmin(1)
            totals.index_add_(0, labels, batch)
            counts.index_add_(0, labels, torch.ones(len(batch), device=values.device))
        nonempty = counts > 0
        updated = centers.clone()
        updated[nonempty] = totals[nonempty] / counts[nonempty, None]
        if torch.max(torch.abs(updated - centers)) < 1e-4:
            centers = updated
            break
        centers = updated
    return centers


def _entropy(labels: torch.Tensor, clusters: int) -> tuple[float, float]:
    probability = torch.bincount(labels, minlength=clusters).float()
    probability = probability / probability.sum().clamp_min(1.0)
    probability = probability[probability > 0]
    entropy = -(probability * probability.log()).sum()
    return float(entropy / math.log(clusters)), float(entropy.exp())


def _fit_logistic(x: torch.Tensor, y: torch.Tensor, steps: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    bias = torch.zeros((), device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y) + 0.05 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach(), bias.detach()


def _loocv_cross_entropy(features: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    model_losses = []
    null_losses = []
    probabilities = []
    for held in range(len(features)):
        train = torch.arange(len(features), device=features.device) != held
        mean = features[train].mean(0)
        scale = features[train].std(0).clamp_min(1e-6)
        weight, bias = _fit_logistic((features[train] - mean) / scale, target[train])
        probability = torch.sigmoid(((features[held] - mean) / scale) @ weight + bias).clamp(1e-6, 1 - 1e-6)
        null_probability = ((target[train].sum() + 0.5) / (train.sum() + 1.0)).clamp(1e-6, 1 - 1e-6)
        model_losses.append(F.binary_cross_entropy(probability, target[held]))
        null_losses.append(F.binary_cross_entropy(null_probability, target[held]))
        probabilities.append(probability)
    prediction = torch.stack(probabilities)
    return {
        "loocv_cross_entropy": float(torch.stack(model_losses).mean()),
        "null_cross_entropy": float(torch.stack(null_losses).mean()),
        "cross_entropy_gain": float(torch.stack(null_losses).mean() - torch.stack(model_losses).mean()),
        "accuracy_at_0.5": float(((prediction >= 0.5) == target.bool()).float().mean()),
    }


def _permutation_test(values: torch.Tensor, group: torch.Tensor, permutations: int, seed: int) -> dict[str, float]:
    observed = values[group].mean() - values[~group].mean()
    generator = torch.Generator(device=values.device).manual_seed(seed)
    null = []
    for _ in range(permutations):
        shuffled = group[torch.randperm(len(group), generator=generator, device=values.device)]
        null.append(values[shuffled].mean() - values[~shuffled].mean())
    null_tensor = torch.stack(null)
    return {
        "st_minus_co": float(observed),
        "permutation_p_two_sided": float((null_tensor.abs() >= observed.abs()).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--v0-aggregate", type=Path, required=True)
    parser.add_argument("--v1-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=29)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This formal analysis requires CUDA")
    device = torch.device("cuda")
    task_config = json.loads(args.task_config.read_text())
    suites = {int(row["task_id"]): str(row["suite"]) for row in task_config["tasks"]}
    task_ids = set(suites)
    trajectories = load_arena(args.manifest, task_ids, args.episodes_per_task, args.workers)
    rows, sampled = _standardize_trajectories(trajectories, device, args.grid_size)
    all_samples = torch.cat([value for _, value in sampled], dim=0)
    centers = _gpu_kmeans(all_samples, args.clusters, args.kmeans_iterations, args.seed)

    by_task_rows: defaultdict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    by_task_samples: defaultdict[int, list[torch.Tensor]] = defaultdict(list)
    for task_id, time_row, behavior_row in rows:
        by_task_rows[task_id].append((time_row, behavior_row))
    for task_id, value in sampled:
        by_task_samples[task_id].append(value)

    task_records = []
    trajectory_centroids = {}
    for task_id in sorted(task_ids):
        time_stack = torch.stack([row[0] for row in by_task_rows[task_id]])
        behavior_stack = torch.stack([row[1] for row in by_task_rows[task_id]])
        action_values = torch.cat(by_task_samples[task_id])
        labels = []
        for batch in action_values.split(32768):
            labels.append(torch.cdist(batch, centers).argmin(1))
        entropy, effective_modes = _entropy(torch.cat(labels), args.clusters)
        time_variance = (time_stack - time_stack.mean(0, keepdim=True)).square().mean()
        behavior_variance = (behavior_stack - behavior_stack.mean(0, keepdim=True)).square().mean()
        trajectory_feature = time_stack.flatten(1)
        centroid = trajectory_feature.mean(0)
        trajectory_centroids[task_id] = centroid
        centroid_dispersion = (trajectory_feature - centroid).square().mean()
        task_records.append(
            {
                "task_id": task_id,
                "suite": suites[task_id],
                "episodes": len(time_stack),
                "action_entropy_normalized": entropy,
                "effective_action_modes": effective_modes,
                "time_aligned_action_variance": float(time_variance),
                "behavior_aligned_action_variance": float(behavior_variance),
                "trajectory_centroid_dispersion": float(centroid_dispersion),
            }
        )

    centroid_stack = torch.stack([trajectory_centroids[row["task_id"]] for row in task_records])
    centroid_distance = torch.cdist(centroid_stack, centroid_stack)
    centroid_distance.fill_diagonal_(float("inf"))
    for index, record in enumerate(task_records):
        same_suite = torch.tensor(
            [other["suite"] == record["suite"] for other in task_records], device=device
        )
        same_suite[index] = False
        other_suite = ~same_suite
        other_suite[index] = False
        record["nearest_task_distance"] = float(centroid_distance[index].min())
        record["same_suite_centroid_distance"] = float(centroid_distance[index][same_suite].mean())
        record["different_suite_centroid_distance"] = float(centroid_distance[index][other_suite].mean())
        record["suite_separation_ratio"] = float(
            centroid_distance[index][other_suite].mean()
            / centroid_distance[index][same_suite].mean().clamp_min(1e-8)
        )

    v0 = {int(row["task_id"]): row for row in json.loads(args.v0_aggregate.read_text())["tasks"]}
    v1 = {int(row["task_id"]): row for row in json.loads(args.v1_aggregate.read_text())["tasks"]}
    for row in task_records:
        task_id = row["task_id"]
        row["v1_minus_v0_tsr"] = float(v1[task_id]["TSR"] - v0[task_id]["TSR"])
        row["v1_minus_v0_csr"] = float(v1[task_id]["CSR"] - v0[task_id]["CSR"])

    metric_names = [
        "action_entropy_normalized",
        "effective_action_modes",
        "time_aligned_action_variance",
        "behavior_aligned_action_variance",
        "trajectory_centroid_dispersion",
        "suite_separation_ratio",
    ]
    feature = torch.tensor([[row[name] for name in metric_names] for row in task_records], device=device)
    target = torch.tensor(
        [row["suite"] in {"Sequence", "Transferring"} for row in task_records],
        dtype=torch.float32,
        device=device,
    )
    cross_entropy = _loocv_cross_entropy(feature, target)
    guidance_delta = torch.tensor(
        [row["v1_minus_v0_tsr"] for row in task_records], dtype=torch.float32, device=device
    )
    guidance_median = guidance_delta.median()
    guidance_cross_entropy = _loocv_cross_entropy(feature, (guidance_delta > guidance_median).float())
    guidance_cross_entropy.update(
        {
            "target": "V1-V0 TSR above task median",
            "median_tsr_delta": float(guidance_median),
        }
    )
    tests = {
        name: _permutation_test(feature[:, index], target.bool(), args.permutations, args.seed + index)
        for index, name in enumerate(metric_names)
    }

    suite_records = []
    for suite in ("Sequence", "Transferring", "Counting", "Occlusion"):
        selected = [row for row in task_records if row["suite"] == suite]
        suite_records.append(
            {
                "suite": suite,
                "tasks": len(selected),
                **{name: float(np.mean([row[name] for row in selected])) for name in metric_names},
                "v1_minus_v0_tsr": float(np.mean([row["v1_minus_v0_tsr"] for row in selected])),
                "v1_minus_v0_csr": float(np.mean([row["v1_minus_v0_csr"] for row in selected])),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "arena_suite_discreteness_gpu_v1",
        "device": torch.cuda.get_device_name(0),
        "episodes_per_task": args.episodes_per_task,
        "tasks": task_records,
        "suites": suite_records,
        "st_vs_counting_occlusion_tests": tests,
        "st_membership_cross_entropy": cross_entropy,
        "high_guidance_gain_cross_entropy": guidance_cross_entropy,
        "metric_names": metric_names,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, records in (("task_metrics.csv", task_records), ("suite_metrics.csv", suite_records)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({"suites": suite_records, "cross_entropy": cross_entropy, "tests": tests}, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
