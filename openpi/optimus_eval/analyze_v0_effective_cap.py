#!/usr/bin/env python3
"""Measure the V1-equivalent relative cap used by unbounded V0 traces."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_SUITE_TASKS = {
    "Sequence": {1, 2, 3, 22},
    "Transferring": {18, 19, 25, 26},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--head", choices=("lower", "upper", "fusion"), default="upper")
    parser.add_argument("--suites", default="Sequence,Transferring")
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_trace(item: tuple[Path, bool]) -> tuple[np.ndarray, ...]:
    path, success = item
    with np.load(path) as trace:
        base = np.asarray(trace["v_base"], dtype=np.float32).reshape(10, -1)
        guidance = np.asarray(trace["guidance_clipped"], dtype=np.float32).reshape(10, -1)
        time = np.asarray(trace["time"], dtype=np.float32).reshape(10)
        weights = np.asarray(trace["retrieval_weights"], dtype=np.float32).reshape(-1)
        scores = np.sort(np.asarray(trace["retrieval_scores"], dtype=np.float32).reshape(-1))[::-1]
        responsibilities = np.asarray(trace["responsibilities"], dtype=np.float32).reshape(10, -1)
        blocks_raw = np.asarray(trace["memory_blocks"], dtype=np.float32)
    if len(weights) == 0:
        call_features = np.zeros(4, dtype=np.float32)
        responsibility_entropy = np.zeros(10, dtype=np.float32)
        return (
            base,
            guidance,
            time,
            np.full(10, success, dtype=np.bool_),
            np.repeat(call_features[None], 10, axis=0),
            responsibility_entropy,
        )
    blocks = blocks_raw.reshape(len(weights), -1)
    weights = weights / max(float(weights.sum()), 1e-12)
    weight_entropy = (
        -float(np.sum(weights * np.log(np.maximum(weights, 1e-12)))) / np.log(len(weights))
        if len(weights) > 1
        else 0.0
    )
    block_norm = np.linalg.norm(blocks, axis=1, keepdims=True).clip(1e-12)
    memory_coherence = float(np.linalg.norm((weights[:, None] * blocks / block_norm).sum(0)))
    responsibility_entropy = (
        -np.sum(responsibilities * np.log(np.maximum(responsibilities, 1e-12)), axis=1)
        / np.log(responsibilities.shape[1])
        if responsibilities.shape[1] > 1
        else np.zeros(responsibilities.shape[0], dtype=np.float32)
    )
    call_features = np.asarray(
        [
            weight_entropy,
            float(scores[0] - scores[1]) if len(scores) > 1 else 0.0,
            float(scores[0] - scores.mean()),
            memory_coherence,
        ],
        dtype=np.float32,
    )
    return (
        base,
        guidance,
        time,
        np.full(10, success, dtype=np.bool_),
        np.repeat(call_features[None], 10, axis=0),
        responsibility_entropy.astype(np.float32),
    )


def _summary(
    base: torch.Tensor,
    guidance: torch.Tensor,
    active: torch.Tensor,
    call_features: torch.Tensor | None = None,
    responsibility_entropy: torch.Tensor | None = None,
) -> dict:
    base_norm = base.norm(dim=1).clamp_min(1e-12)
    guidance_norm = guidance.norm(dim=1)
    ratio = guidance_norm / base_norm
    cosine = (base * guidance).sum(dim=1) / (base_norm * guidance_norm.clamp_min(1e-12))
    mixed_speed_ratio = (base + guidance).norm(dim=1) / base_norm
    angle = torch.rad2deg(
        torch.acos(
            ((base * (base + guidance)).sum(dim=1) / (base_norm * (base + guidance).norm(dim=1).clamp_min(1e-12)))
            .clamp(-1.0, 1.0)
        )
    )
    selected = active & (guidance_norm > 1e-12)
    ratio_selected = ratio[selected]
    cosine_selected = cosine[selected]
    speed_selected = mixed_speed_ratio[selected]
    angle_selected = angle[selected]
    parallel_ratio = ratio_selected * cosine_selected
    orthogonal_ratio = ratio_selected * torch.sqrt((1.0 - cosine_selected.square()).clamp_min(0.0))
    if ratio_selected.numel() == 0:
        return {"active_guidance_steps": 0}
    quantiles = torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99], device=base.device)
    ratio_q = torch.quantile(ratio_selected, quantiles)
    result = {
        "active_guidance_steps": int(selected.sum()),
        "effective_cap_mean": float(ratio_selected.mean()),
        "effective_cap_quantiles": {
            name: float(value)
            for name, value in zip(("p50", "p75", "p90", "p95", "p99"), ratio_q, strict=True)
        },
        "fraction_exceeding_v1_cap_0_2": float((ratio_selected > 0.2).float().mean()),
        "guidance_base_cosine_mean": float(cosine_selected.mean()),
        "guidance_base_cosine_positive_fraction": float((cosine_selected > 0).float().mean()),
        "parallel_effective_cap_mean": float(parallel_ratio.mean()),
        "orthogonal_effective_cap_mean": float(orthogonal_ratio.mean()),
        "parallel_to_orthogonal_ratio": float(
            parallel_ratio.clamp_min(0.0).mean() / orthogonal_ratio.mean().clamp_min(1e-12)
        ),
        "mixed_speed_ratio_mean": float(speed_selected.mean()),
        "mixed_speed_ratio_p95": float(torch.quantile(speed_selected, 0.95)),
        "base_to_mixed_angle_deg_mean": float(angle_selected.mean()),
        "base_to_mixed_angle_deg_p95": float(torch.quantile(angle_selected, 0.95)),
    }
    if call_features is not None:
        selected_features = call_features[selected]
        result.update(
            {
                "retrieval_weight_entropy_mean": float(selected_features[:, 0].mean()),
                "retrieval_top1_top2_margin_mean": float(selected_features[:, 1].mean()),
                "retrieval_top1_mean_margin_mean": float(selected_features[:, 2].mean()),
                "memory_action_directional_coherence_mean": float(selected_features[:, 3].mean()),
            }
        )
    if responsibility_entropy is not None:
        result["responsibility_entropy_mean"] = float(responsibility_entropy[selected].mean())
    return result


def _step_summary(base: torch.Tensor, guidance: torch.Tensor, time: torch.Tensor) -> list[dict]:
    rows = []
    for step in range(10):
        selected = torch.arange(base.shape[0], device=base.device) % 10 == step
        base_step = base[selected]
        guidance_step = guidance[selected]
        base_norm = base_step.norm(dim=1).clamp_min(1e-12)
        guidance_norm = guidance_step.norm(dim=1)
        ratio = guidance_norm / base_norm
        cosine = (base_step * guidance_step).sum(dim=1) / (
            base_norm * guidance_norm.clamp_min(1e-12)
        )
        rows.append(
            {
                "step": step,
                "time": float(time[selected][0]),
                "effective_cap_mean": float(ratio.mean()),
                "effective_cap_p50": float(torch.quantile(ratio, 0.5)),
                "effective_cap_p95": float(torch.quantile(ratio, 0.95)),
                "fraction_exceeding_0_2": float((ratio > 0.2).float().mean()),
                "guidance_base_cosine_mean": float(cosine.mean()) if bool((guidance_norm > 0).any()) else 0.0,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.task_config:
        task_config = json.loads(args.task_config.read_text())
        suite_tasks: dict[str, set[int]] = {}
        for row in task_config["tasks"]:
            suite_tasks.setdefault(str(row["suite"]), set()).add(int(row["task_id"]))
    else:
        suite_tasks = DEFAULT_SUITE_TASKS
    suites = [value.strip() for value in args.suites.split(",") if value.strip()]
    unknown = set(suites) - set(suite_tasks)
    if unknown:
        raise ValueError(f"Unknown suites: {sorted(unknown)}")
    record_root = args.run_root / args.head / "memory_records"
    manifest_path = record_root / "index.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    device = torch.device(args.device)
    output: dict[str, object] = {
        "protocol": "v0_unbounded_v1_equivalent_cap_v1",
        "run_root": str(args.run_root),
        "head": args.head,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "definition": "effective_cap = ||g_v0||_2 / ||v_base||_2 per active denoising step",
        "suites": {},
    }
    for suite in suites:
        selected_rows = [row for row in rows if int(row["task_id"]) in suite_tasks[suite]]
        items = [(args.run_root / args.head / row["trace_path"], bool(row["success"])) for row in selected_rows]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            loaded = list(pool.map(_load_trace, items))
        base = torch.from_numpy(np.concatenate([item[0] for item in loaded])).to(device)
        guidance = torch.from_numpy(np.concatenate([item[1] for item in loaded])).to(device)
        time = torch.from_numpy(np.concatenate([item[2] for item in loaded])).to(device)
        success = torch.from_numpy(np.concatenate([item[3] for item in loaded])).to(device)
        call_features = torch.from_numpy(np.concatenate([item[4] for item in loaded])).to(device)
        responsibility_entropy = torch.from_numpy(np.concatenate([item[5] for item in loaded])).to(device)
        active = time > 0.3 + 1e-6
        task_results = {}
        row_task_ids = np.concatenate(
            [np.full(10, int(row["task_id"]), dtype=np.int64) for row in selected_rows]
        )
        task_ids = torch.from_numpy(row_task_ids).to(device)
        for task_id in sorted(suite_tasks[suite]):
            task_mask = task_ids == task_id
            task_results[str(task_id)] = _summary(
                base[task_mask],
                guidance[task_mask],
                active[task_mask],
                call_features[task_mask],
                responsibility_entropy[task_mask],
            )
            task_results[str(task_id)]["by_denoise_step"] = _step_summary(
                base[task_mask], guidance[task_mask], time[task_mask]
            )
        suite_result = {
            "tasks": sorted(suite_tasks[suite]),
            "policy_calls": len(items),
            "all_episodes": _summary(base, guidance, active, call_features, responsibility_entropy),
            "successful_episodes": _summary(
                base[success], guidance[success], active[success], call_features[success], responsibility_entropy[success]
            ),
            "failed_episodes": _summary(
                base[~success], guidance[~success], active[~success], call_features[~success], responsibility_entropy[~success]
            ),
            "by_denoise_step": _step_summary(base, guidance, time),
            "per_task": task_results,
        }
        output["suites"][suite] = suite_result
        del base, guidance, time, success
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
