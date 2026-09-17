#!/usr/bin/env python3
"""Measure producer-matched PrediMem retrieval separability on held-out trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors-per-trajectory", type=int, default=4)
    parser.add_argument("--positive-progress-radius", type=float, default=0.025)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _manifest_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels(manifest: Path, cache: Path) -> dict[str, np.ndarray]:
    digest = _manifest_digest(manifest)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["manifest_sha256"].item()) == digest:
                return {name: np.asarray(data[name]) for name in data.files if name != "manifest_sha256"}

    try:
        import orjson

        loads = orjson.loads
    except ImportError:
        loads = json.loads
    task: list[int] = []
    stage: list[int] = []
    progress: list[float] = []
    seed: list[int] = []
    trajectory: list[int] = []
    trajectory_ids: dict[str, int] = {}
    with manifest.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = loads(line)
            action_id = str(row["action_id"])
            trajectory_id = trajectory_ids.setdefault(action_id, len(trajectory_ids))
            task.append(int(row["task_id"]))
            stage.append(int(row["stage_index"]))
            progress.append(float(row["stage_progress"]))
            seed.append(int(row["seed"]))
            trajectory.append(trajectory_id)
            if line_number % 100000 == 0:
                print(f"Parsed manifest rows={line_number}", flush=True)
    arrays = {
        "task": np.asarray(task, dtype=np.int16),
        "stage": np.asarray(stage, dtype=np.int16),
        "progress": np.asarray(progress, dtype=np.float32),
        "seed": np.asarray(seed, dtype=np.int32),
        "trajectory": np.asarray(trajectory, dtype=np.int32),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, manifest_sha256=np.asarray(digest), **arrays)
    return arrays


def _sample_queries(labels: dict[str, np.ndarray], anchors_per_trajectory: int) -> np.ndarray:
    if anchors_per_trajectory < 1:
        raise ValueError("anchors-per-trajectory must be positive")
    validation = labels["seed"] % 5 == 0
    selected: list[int] = []
    for trajectory_id in np.unique(labels["trajectory"][validation]):
        indices = np.flatnonzero(validation & (labels["trajectory"] == trajectory_id))
        order = np.lexsort((labels["progress"][indices], labels["stage"][indices]))
        indices = indices[order]
        positions = np.linspace(0, len(indices) - 1, anchors_per_trajectory).round().astype(np.int64)
        selected.extend(indices[np.unique(positions)].tolist())
    return np.asarray(selected, dtype=np.int64)


def _bootstrap_ci(
    values: torch.Tensor,
    trajectory: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    unique = trajectory.unique()
    trajectory_means = torch.stack([values[trajectory == item].mean() for item in unique])
    indices = torch.randint(
        len(trajectory_means),
        (samples, len(trajectory_means)),
        device=values.device,
        generator=generator,
    )
    means = trajectory_means[indices].mean(1)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975], device=values.device))
    return float(bounds[0]), float(bounds[1])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact Full26 retrieval analysis")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_labels(args.manifest, args.output_dir / "manifest_labels.npz")
    index = faiss.read_index(str(args.index))
    rows = len(labels["task"])
    if index.ntotal != rows:
        raise RuntimeError(f"Index/manifest mismatch: {index.ntotal} != {rows}")
    queries = _sample_queries(labels, args.anchors_per_trajectory)
    candidates = np.flatnonzero(labels["seed"] % 5 != 0)
    print(
        f"Reconstructing keys rows={rows} candidates={len(candidates)} queries={len(queries)}",
        flush=True,
    )
    all_keys = np.asarray(index.reconstruct_n(0, rows), dtype=np.float32)
    query_keys = torch.from_numpy(np.ascontiguousarray(all_keys[queries])).to(device)
    candidate_keys = torch.from_numpy(np.ascontiguousarray(all_keys[candidates])).to(device)
    del all_keys, index
    candidate_keys_t = candidate_keys.T.contiguous()
    candidate_task = torch.from_numpy(labels["task"][candidates].astype(np.int64)).to(device)
    candidate_stage = torch.from_numpy(labels["stage"][candidates].astype(np.int64)).to(device)
    candidate_progress = torch.from_numpy(labels["progress"][candidates]).to(device)
    query_task = torch.from_numpy(labels["task"][queries].astype(np.int64)).to(device)
    query_stage = torch.from_numpy(labels["stage"][queries].astype(np.int64)).to(device)
    query_progress = torch.from_numpy(labels["progress"][queries]).to(device)
    query_trajectory = torch.from_numpy(labels["trajectory"][queries].astype(np.int64)).to(device)

    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            for start in range(0, len(queries), args.batch_size):
                stop = min(start + args.batch_size, len(queries))
                query = query_keys[start:stop]
                similarity = query @ candidate_keys_t
                batch = stop - start
                task_max = torch.full((batch, 27), -torch.inf, device=device)
                task_max.scatter_reduce_(
                    1,
                    candidate_task.unsqueeze(0).expand(batch, -1),
                    similarity,
                    reduce="amax",
                    include_self=True,
                )
                task_max = task_max[:, 1:]
                true_task = query_task[start:stop]
                true_task_score = task_max.gather(1, (true_task - 1).unsqueeze(1)).squeeze(1)
                other_scores = task_max.clone()
                other_scores.scatter_(1, (true_task - 1).unsqueeze(1), -torch.inf)
                other_task_score = other_scores.max(1).values
                task_probability = torch.softmax(float(args.temperature) * task_max, dim=1).gather(
                    1, (true_task - 1).unsqueeze(1)
                ).squeeze(1)

                same_task = candidate_task.unsqueeze(0) == true_task.unsqueeze(1)
                positive = (
                    same_task
                    & (candidate_stage.unsqueeze(0) == query_stage[start:stop].unsqueeze(1))
                    & (
                        candidate_progress.unsqueeze(0) - query_progress[start:stop].unsqueeze(1)
                    ).abs().le(float(args.positive_progress_radius))
                )
                positive_score = similarity.masked_fill(~positive, -torch.inf).max(1).values
                incompatible_score = similarity.masked_fill(~(same_task & ~positive), -torch.inf).max(1).values
                if not bool(torch.isfinite(positive_score).all()):
                    raise RuntimeError("At least one query has no held-out-compatible training positive")
                phase_margin = positive_score - incompatible_score
                phase_probability = torch.sigmoid(float(args.temperature) * phase_margin)
                top_index = similarity.argmax(1)
                top8 = similarity.topk(k=8, dim=1).indices
                outputs["task_margin"].append(true_task_score - other_task_score)
                outputs["task_probability"].append(task_probability)
                outputs["task_recall1"].append((candidate_task[top_index] == true_task).float())
                outputs["phase_margin"].append(phase_margin)
                outputs["phase_probability"].append(phase_probability)
                outputs["phase_recall1"].append(positive.gather(1, top_index.unsqueeze(1)).squeeze(1).float())
                outputs["phase_recall8"].append(positive.gather(1, top8).any(1).float())
                outputs["c_mem"].append(task_probability * phase_probability)
                del similarity, task_max, other_scores, positive, same_task
                if stop % 256 == 0 or stop == len(queries):
                    print(f"Scored queries={stop}/{len(queries)}", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    metrics = {name: torch.cat(parts) for name, parts in outputs.items()}

    config = json.loads(args.task_config.read_text())
    suite_by_task = {int(row["task_id"]): str(row["suite"]) for row in config["tasks"]}
    query_suite = np.asarray([suite_by_task[int(task)] for task in labels["task"][queries]])
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def summarize(mask: torch.Tensor) -> dict[str, float | list[float] | int]:
        result: dict[str, float | list[float] | int] = {"queries": int(mask.sum())}
        for name, values in metrics.items():
            selected_values = values[mask]
            selected_trajectory = query_trajectory[mask]
            result[name] = float(selected_values.mean())
            result[f"{name}_ci95"] = list(
                _bootstrap_ci(
                    selected_values,
                    selected_trajectory,
                    args.bootstrap_samples,
                    generator,
                )
            )
        return result

    task_rows = []
    for task_id in range(1, 27):
        mask = query_task == task_id
        task_rows.append({"task_id": task_id, "suite": suite_by_task[task_id], **summarize(mask)})
    suite_rows = []
    for suite in ("Sequence", "Transferring", "Counting", "Occlusion"):
        mask = torch.from_numpy(query_suite == suite).to(device)
        suite_rows.append({"suite": suite, **summarize(mask)})

    payload = {
        "protocol": "predimem_fusion_heldout_retrieval_separability_v1",
        "definition": {
            "split": "validation query seed%5==0; training candidate seed%5!=0",
            "sampling": f"{args.anchors_per_trajectory} progress-uniform anchors/validation trajectory",
            "task_margin": "max cosine correct task - max cosine best other task",
            "phase_margin": "best same-task stage/progress positive - best same-task incompatible",
            "c_mem": "softmax_temperature(task max scores)[correct] * sigmoid(temperature*phase_margin)",
            "temperature": args.temperature,
            "positive_progress_radius": args.positive_progress_radius,
        },
        "device": torch.cuda.get_device_name(device),
        "manifest": str(args.manifest),
        "index": str(args.index),
        "queries": len(queries),
        "candidates": len(candidates),
        "suites": suite_rows,
        "tasks": task_rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, rows_out in (("suite_metrics.csv", suite_rows), ("task_metrics.csv", task_rows)):
        with (args.output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
    print(json.dumps({"suites": suite_rows}, indent=2))


if __name__ == "__main__":
    main()
