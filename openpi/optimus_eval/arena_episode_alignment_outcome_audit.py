#!/usr/bin/env python3
"""Audit V0 flow geometry against outcome with episode/task clustering."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--head", choices=("upper", "fusion"), required=True)
    parser.add_argument("--task-ids", default="1,2,3,18,19,22,25,26")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _trace_geometry(path: Path) -> tuple[float, float, float, int]:
    with np.load(path) as trace:
        base = np.asarray(trace["v_base"], dtype=np.float32).reshape(10, -1)
        guidance = np.asarray(trace["guidance_clipped"], dtype=np.float32).reshape(10, -1)
        time = np.asarray(trace["time"], dtype=np.float32).reshape(10)
    base_norm = np.linalg.norm(base, axis=1).clip(1e-12)
    guidance_norm = np.linalg.norm(guidance, axis=1)
    selected = (time > 0.300001) & (guidance_norm > 1e-12)
    if not selected.any():
        return 0.0, 0.0, 0.0, 0
    cosine = np.sum(base * guidance, axis=1) / (base_norm * guidance_norm.clip(1e-12))
    mixed = base + guidance
    mixed_norm = np.linalg.norm(mixed, axis=1).clip(1e-12)
    mixed_cosine = np.sum(base * mixed, axis=1) / (base_norm * mixed_norm)
    angle = np.rad2deg(np.arccos(np.clip(mixed_cosine, -1.0, 1.0)))
    return (
        float(cosine[selected].sum()),
        float((guidance_norm[selected] / base_norm[selected]).sum()),
        float(angle[selected].sum()),
        int(selected.sum()),
    )


def _stratified_test(
    rows: list[dict], metric: str, samples: int, seed: int, device: torch.device
) -> dict:
    grouped: dict[int, dict[bool, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["task_id"])][bool(row["success"])].append(float(row[metric]))
    eligible = {
        task_id: outcomes
        for task_id, outcomes in grouped.items()
        if outcomes[True] and outcomes[False]
    }
    generator = torch.Generator(device=device).manual_seed(seed)
    observed = []
    bootstrap = []
    permutation = []
    for task_id, outcomes in sorted(eligible.items()):
        success = torch.tensor(outcomes[True], device=device)
        failure = torch.tensor(outcomes[False], device=device)
        observed.append(success.mean() - failure.mean())
        success_idx = torch.randint(
            len(success), (samples, len(success)), generator=generator, device=device
        )
        failure_idx = torch.randint(
            len(failure), (samples, len(failure)), generator=generator, device=device
        )
        bootstrap.append(success[success_idx].mean(1) - failure[failure_idx].mean(1))
        combined = torch.cat((success, failure))
        random = torch.rand((samples, len(combined)), generator=generator, device=device)
        order = random.argsort(1)
        shuffled = combined[order]
        permutation.append(
            shuffled[:, : len(success)].mean(1) - shuffled[:, len(success) :].mean(1)
        )
    observed_tensor = torch.stack(observed)
    observed_macro = observed_tensor.mean()
    bootstrap_macro = torch.stack(bootstrap).mean(0)
    permutation_macro = torch.stack(permutation).mean(0)
    bounds = torch.quantile(
        bootstrap_macro, torch.tensor([0.025, 0.975], device=device)
    )
    p_value = (1.0 + (permutation_macro.abs() >= observed_macro.abs()).sum()) / (
        samples + 1.0
    )
    return {
        "eligible_tasks": sorted(eligible),
        "task_macro_success_minus_failure": float(observed_macro),
        "bootstrap_95ci": [float(bounds[0]), float(bounds[1])],
        "stratified_permutation_p": float(p_value),
        "per_task_success_minus_failure": {
            str(task_id): float(value)
            for task_id, value in zip(sorted(eligible), observed_tensor, strict=True)
        },
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    task_ids = {int(value) for value in args.task_ids.split(",") if value}
    record_root = args.run_root / args.head / "memory_records"
    index_rows = [
        json.loads(line)
        for line in (record_root / "index.jsonl").read_text().splitlines()
        if line
    ]
    selected = [row for row in index_rows if int(row["task_id"]) in task_ids]
    paths = [args.run_root / args.head / row["trace_path"] for row in selected]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        geometry = list(pool.map(_trace_geometry, paths))

    episodes: dict[tuple[int, int], dict] = {}
    for row, (cosine, cap, angle, count) in zip(selected, geometry, strict=True):
        key = (int(row["task_id"]), int(row["episode_idx"]))
        episode = episodes.setdefault(
            key,
            {
                "task_id": key[0],
                "episode_idx": key[1],
                "success": bool(row["success"]),
                "cosine_sum": 0.0,
                "cap_sum": 0.0,
                "angle_sum": 0.0,
                "active_steps": 0,
            },
        )
        if bool(row["success"]) != episode["success"]:
            raise ValueError(f"Inconsistent outcome for episode {key}")
        episode["cosine_sum"] += cosine
        episode["cap_sum"] += cap
        episode["angle_sum"] += angle
        episode["active_steps"] += count

    episode_rows = []
    for episode in episodes.values():
        count = max(int(episode.pop("active_steps")), 1)
        episode_rows.append(
            {
                "task_id": episode["task_id"],
                "episode_idx": episode["episode_idx"],
                "success": episode["success"],
                "alignment": episode.pop("cosine_sum") / count,
                "effective_cap": episode.pop("cap_sum") / count,
                "rotation_deg": episode.pop("angle_sum") / count,
            }
        )

    output = {
        "protocol": "arena_episode_clustered_v0_geometry_v1",
        "run_root": str(args.run_root),
        "head": args.head,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "episodes": len(episode_rows),
        "tests": {
            metric: _stratified_test(
                episode_rows, metric, args.samples, args.seed + index, device
            )
            for index, metric in enumerate(("alignment", "effective_cap", "rotation_deg"))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
