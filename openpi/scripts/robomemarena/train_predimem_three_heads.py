from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import threading
import time

import faiss
import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from openpi.task_head.dual_tower_head import DualTowerRetrievalHead, checkpoint_payload


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _inputs(
    variant: str, lower: np.ndarray, upper: np.ndarray, indices: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_batch = torch.from_numpy(np.asarray(lower[indices], dtype=np.float32).copy()).to(device)
    upper_batch = torch.from_numpy(np.asarray(upper[indices], dtype=np.float32).copy()).to(device)
    return lower_batch, upper_batch


def _contrastive_loss(
    emb: torch.Tensor,
    task: torch.Tensor,
    stage: torch.Tensor,
    progress: torch.Tensor,
    trajectory: torch.Tensor,
    temperature: float,
    positive_progress_radius: float = 0.15,
    require_cross_trajectory_positive: bool = False,
    temporal_regression_weight: float = 0.0,
    temporal_regression_scale: float = 0.05,
) -> torch.Tensor:
    similarity = emb @ emb.T
    logits = similarity / temperature
    eye, positive = _positive_pair_mask(
        task,
        stage,
        progress,
        trajectory,
        positive_progress_radius=positive_progress_radius,
        require_cross_trajectory_positive=require_cross_trajectory_positive,
    )
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        raise RuntimeError("Batch contains no task/progress positive pairs")
    logits = logits.masked_fill(eye, -torch.inf)
    log_denom = torch.logsumexp(logits, dim=1)
    log_positive = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    loss = -(log_positive[valid] - log_denom[valid]).mean()
    if temporal_regression_weight > 0:
        temporal_pairs = (
            (task[:, None] == task[None, :])
            & (stage[:, None] == stage[None, :])
            & (trajectory[:, None] != trajectory[None, :])
            & ~eye
        )
        if bool(temporal_pairs.any()):
            distance = (progress[:, None] - progress[None, :]).abs()
            target = torch.exp(-distance / temporal_regression_scale)
            predicted = (similarity + 1.0) * 0.5
            temporal_loss = F.smooth_l1_loss(
                predicted[temporal_pairs],
                target[temporal_pairs],
            )
            loss = loss + temporal_regression_weight * temporal_loss
    return loss


def _positive_pair_mask(
    task: torch.Tensor,
    stage: torch.Tensor,
    progress: torch.Tensor,
    trajectory: torch.Tensor,
    *,
    positive_progress_radius: float,
    require_cross_trajectory_positive: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    eye = torch.eye(len(task), dtype=torch.bool, device=task.device)
    positive = (
        (task[:, None] == task[None, :])
        & (stage[:, None] == stage[None, :])
        & ((progress[:, None] - progress[None, :]).abs() <= positive_progress_radius)
        & ~eye
    )
    if require_cross_trajectory_positive:
        positive &= trajectory[:, None] != trajectory[None, :]
    return eye, positive


class BalancedProgressPairSampler:
    """Build task-balanced batches with a cross-trajectory positive for every row."""

    def __init__(
        self,
        indices: np.ndarray,
        task: np.ndarray,
        stage: np.ndarray,
        progress: np.ndarray,
        trajectory: np.ndarray,
        *,
        batch_size: int,
        positive_progress_radius: float,
    ) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Balanced pair sampling requires an even batch size >= 2")
        if positive_progress_radius <= 0:
            raise ValueError("positive_progress_radius must be positive")
        self.batch_size = int(batch_size)
        self.positive_progress_radius = float(positive_progress_radius)

        raw: dict[int, dict[int, dict[int, dict[int, list[int]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        for index in np.asarray(indices, dtype=np.int64).tolist():
            progress_bin = int(np.floor(float(progress[index]) / self.positive_progress_radius))
            raw[int(task[index])][int(stage[index])][progress_bin][int(trajectory[index])].append(index)

        self.groups: dict[int, dict[int, list[dict[int, np.ndarray]]]] = {}
        dropped_stages: dict[int, list[int]] = defaultdict(list)
        for task_id, stages in raw.items():
            usable_stages: dict[int, list[dict[int, np.ndarray]]] = {}
            for stage_id, bins in stages.items():
                usable_bins = []
                for trajectories in bins.values():
                    if len(trajectories) >= 2:
                        usable_bins.append(
                            {
                                trajectory_id: np.asarray(values, dtype=np.int64)
                                for trajectory_id, values in trajectories.items()
                            }
                        )
                if usable_bins:
                    usable_stages[stage_id] = usable_bins
                else:
                    dropped_stages[task_id].append(stage_id)
            if usable_stages:
                self.groups[task_id] = usable_stages

        expected_tasks = sorted({int(task[index]) for index in np.asarray(indices, dtype=np.int64)})
        missing_tasks = sorted(set(expected_tasks) - set(self.groups))
        if missing_tasks:
            raise RuntimeError(f"Tasks lack cross-trajectory progress positives: {missing_tasks}")
        self.task_ids = np.asarray(sorted(self.groups), dtype=np.int64)
        self.summary = {
            "strategy": "task_stage_progress_pairs",
            "tasks": len(self.task_ids),
            "stages": sum(len(stages) for stages in self.groups.values()),
            "progress_bins": sum(
                len(bins) for stages in self.groups.values() for bins in stages.values()
            ),
            "dropped_stages": {str(key): value for key, value in sorted(dropped_stages.items())},
            "batch_size": self.batch_size,
            "positive_progress_radius": self.positive_progress_radius,
        }

    def sample(self, generator: np.random.Generator) -> np.ndarray:
        pair_count = self.batch_size // 2
        task_order: list[int] = []
        while len(task_order) < pair_count:
            task_order.extend(generator.permutation(self.task_ids).tolist())

        batch: list[int] = []
        used: set[int] = set()
        for task_id in task_order[:pair_count]:
            stages = self.groups[int(task_id)]
            stage_id = int(generator.choice(np.asarray(sorted(stages), dtype=np.int64)))
            bins = stages[stage_id]
            trajectories = bins[int(generator.integers(len(bins)))]
            trajectory_ids = np.asarray(sorted(trajectories), dtype=np.int64)
            chosen_trajectories = generator.choice(trajectory_ids, size=2, replace=False)
            pair = []
            for trajectory_id in chosen_trajectories.tolist():
                candidates = trajectories[int(trajectory_id)]
                available = candidates[~np.isin(candidates, np.fromiter(used, dtype=np.int64))]
                source = available if len(available) else candidates
                index = int(generator.choice(source))
                pair.append(index)
                used.add(index)
            batch.extend(pair)
        generator.shuffle(batch)
        return np.asarray(batch, dtype=np.int64)


@torch.inference_mode()
def _project(
    head: DualTowerRetrievalHead,
    lower: np.ndarray,
    upper: np.ndarray,
    device: str,
    upper_age: np.ndarray | None = None,
    upper_available: np.ndarray | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    output = []
    for start in range(0, len(lower), batch_size):
        indices = np.arange(start, min(start + batch_size, len(lower)))
        lo, up = _inputs(head.variant, lower, upper, indices, device)
        age = (
            torch.zeros(len(indices), device=device)
            if upper_age is None
            else torch.from_numpy(np.asarray(upper_age[indices], dtype=np.float32).copy()).to(device)
        )
        available = (
            torch.ones(len(indices), device=device)
            if upper_available is None
            else torch.from_numpy(
                np.asarray(upper_available[indices], dtype=np.float32).copy()
            ).to(device)
        )
        output.append(head(lo, up, age, available).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(output), dtype=np.float32)


def _exact_search(
    keys: np.ndarray,
    query_indices: np.ndarray,
    key_indices: np.ndarray,
    k: int,
    *,
    device: str = "cpu",
    query_batch_size: int = 4096,
) -> np.ndarray:
    k = min(k, len(key_indices))
    if k < 1:
        raise ValueError("Exact retrieval requires at least one key")
    started = time.monotonic()
    if device == "cpu":
        index = faiss.IndexFlatIP(keys.shape[1])
        index.add(np.ascontiguousarray(keys[key_indices], dtype=np.float32))
        _, found = index.search(np.ascontiguousarray(keys[query_indices], dtype=np.float32), k)
    else:
        if query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        retrieval_device = torch.device(device)
        if retrieval_device.type != "cuda":
            raise ValueError(f"Unsupported retrieval device: {device}")
        key_tensor = torch.from_numpy(
            np.ascontiguousarray(keys[key_indices], dtype=np.float32)
        ).to(retrieval_device)
        key_tensor_t = key_tensor.T.contiguous()
        found_batches = []
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.inference_mode():
                for start in range(0, len(query_indices), query_batch_size):
                    batch_indices = query_indices[start : start + query_batch_size]
                    query_tensor = torch.from_numpy(
                        np.ascontiguousarray(keys[batch_indices], dtype=np.float32)
                    ).to(retrieval_device)
                    similarity = query_tensor @ key_tensor_t
                    found_batches.append(
                        torch.topk(similarity, k=k, dim=1, largest=True, sorted=True)
                        .indices.cpu().numpy()
                    )
                    del query_tensor, similarity
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        found = np.concatenate(found_batches, axis=0)
        del key_tensor, key_tensor_t
    print(
        f"Exact retrieval: device={device} queries={len(query_indices)} "
        f"keys={len(key_indices)} dim={keys.shape[1]} k={k} "
        f"seconds={time.monotonic() - started:.2f}",
        flush=True,
    )
    return found


def _retrieval_metrics(
    keys: np.ndarray,
    rows: list[dict],
    query_indices: np.ndarray,
    key_indices: np.ndarray,
    k: int = 8,
    oracle_neighbors: np.ndarray | None = None,
    progress_field: str = "anchor_progress",
    progress_tolerance: float = 0.20,
    retrieval_device: str = "cpu",
    retrieval_query_batch_size: int = 4096,
) -> dict[str, float]:
    found = _exact_search(
        keys,
        query_indices,
        key_indices,
        k,
        device=retrieval_device,
        query_batch_size=retrieval_query_batch_size,
    )
    task = np.asarray([row["task_id"] for row in rows])
    progress = np.asarray([row[progress_field] for row in rows], dtype=np.float32)
    stage = np.asarray([row["stage_index"] for row in rows], dtype=np.int64)
    actual = key_indices[found]
    task_match = task[actual] == task[query_indices, None]
    progress_match = (
        np.abs(progress[actual] - progress[query_indices, None]) <= progress_tolerance
    )
    stage_match = stage[actual] == stage[query_indices, None]
    both = task_match & progress_match
    stage_both = both & stage_match
    metrics = {
        "task_recall@1": float(task_match[:, :1].any(axis=1).mean()),
        "task_progress_recall@1": float(both[:, :1].any(axis=1).mean()),
        "task_progress_recall@4": float(both[:, :4].any(axis=1).mean()),
        "task_progress_recall@8": float(both.any(axis=1).mean()),
        "task_stage_progress_recall@1": float(stage_both[:, :1].any(axis=1).mean()),
        "task_stage_progress_recall@4": float(stage_both[:, :4].any(axis=1).mean()),
        "task_stage_progress_recall@8": float(stage_both.any(axis=1).mean()),
    }
    if all("stage_progress" in row and "stage_length" in row for row in rows):
        query_stage_length = np.asarray(
            [int(rows[index]["stage_length"]) for index in query_indices],
            dtype=np.float32,
        )
        query_stage_frame = np.asarray(
            [int(rows[index]["stage_frame_index"]) for index in query_indices],
            dtype=np.float32,
        )
        predicted_frames = (
            np.asarray(
                [[float(rows[index]["stage_progress"]) for index in row] for row in actual],
                dtype=np.float32,
            )
            * np.maximum(query_stage_length[:, None] - 1.0, 1.0)
        )
        frame_error = np.abs(predicted_frames - query_stage_frame[:, None])
        stage_valid = task_match & stage_match
        first_valid = stage_valid[:, 0]
        first_error = frame_error[:, 0]
        conditional = first_error[first_valid]
        metrics.update(
            {
                "task_stage_recall@1": float(first_valid.mean()),
                "task_stage_frame_recall@1_6": float(
                    (first_valid & (first_error <= 6.0)).mean()
                ),
                "task_stage_frame_recall@8_6": float(
                    (stage_valid & (frame_error <= 6.0)).any(axis=1).mean()
                ),
                "stage_frame_mae@1": (
                    float(conditional.mean()) if len(conditional) else float("inf")
                ),
                "stage_frame_p95@1": (
                    float(np.percentile(conditional, 95)) if len(conditional) else float("inf")
                ),
            }
        )
        per_task = {}
        query_tasks = task[query_indices]
        for task_id in sorted({int(value) for value in query_tasks.tolist()}):
            selected = query_tasks == task_id
            selected_valid = stage_valid[selected, 0]
            selected_error = frame_error[selected, 0]
            conditional_error = selected_error[selected_valid]
            per_task[str(task_id)] = {
                "queries": int(selected.sum()),
                "suite": str(rows[int(query_indices[np.flatnonzero(selected)[0]])].get("suite", "")),
                "task_recall@1": float(task_match[selected, 0].mean()),
                "task_recall@8": float(task_match[selected].any(axis=1).mean()),
                "task_stage_recall@1": float(selected_valid.mean()),
                "task_stage_frame_recall@1_6": float(
                    (selected_valid & (selected_error <= 6.0)).mean()
                ),
                "task_stage_frame_recall@8_6": float(
                    (stage_valid[selected] & (frame_error[selected] <= 6.0)).any(axis=1).mean()
                ),
                "stage_frame_mae@1": (
                    float(conditional_error.mean()) if len(conditional_error) else float("inf")
                ),
            }
        metrics["per_task"] = per_task
        for name in (
            "task_recall@1",
            "task_recall@8",
            "task_stage_recall@1",
            "task_stage_frame_recall@1_6",
            "task_stage_frame_recall@8_6",
        ):
            metrics[f"macro_{name}"] = float(
                np.mean([float(values[name]) for values in per_task.values()])
            )
    if oracle_neighbors is not None:
        overlaps = [
            len(set(candidate.tolist()) & set(oracle.tolist())) / max(len(oracle), 1)
            for candidate, oracle in zip(actual, oracle_neighbors, strict=True)
        ]
        metrics["candidate_oracle_topk_overlap"] = float(np.mean(overlaps))
    return metrics


def _neighbors(
    keys: np.ndarray,
    query_indices: np.ndarray,
    key_indices: np.ndarray,
    k: int = 8,
    *,
    retrieval_device: str = "cpu",
    retrieval_query_batch_size: int = 4096,
) -> np.ndarray:
    found = _exact_search(
        keys,
        query_indices,
        key_indices,
        k,
        device=retrieval_device,
        query_batch_size=retrieval_query_batch_size,
    )
    return key_indices[found]


def _quantize(keys: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "fp32":
        result = keys.copy()
    elif dtype == "fp16":
        result = keys.astype(np.float16).astype(np.float32)
    elif dtype == "int8":
        scale = np.maximum(np.abs(keys).max(axis=1, keepdims=True) / 127.0, 1e-8)
        result = np.rint(keys / scale).clip(-127, 127).astype(np.int8).astype(np.float32) * scale
    else:
        raise ValueError(dtype)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-8)
    return np.ascontiguousarray(result, dtype=np.float32)


def _anchor_subset(rows: list[dict], budget: int, allowed: np.ndarray | None = None) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    source = range(len(rows)) if allowed is None else allowed.tolist()
    for index in source:
        row = rows[index]
        groups[str(row["action_id"])].append(index)
    selected = []
    for values in groups.values():
        values.sort(key=lambda index: float(rows[index]["anchor_progress"]))
        positions = np.unique(np.rint(np.linspace(0, len(values) - 1, min(budget, len(values)))).astype(int))
        selected.extend(values[position] for position in positions)
    return np.asarray(sorted(selected), dtype=np.int64)


def _read_action_trajectory(item: tuple[str, dict], action_dim: int) -> tuple[str, np.ndarray]:
    action_id, row = item
    chunks = []
    source_format = str(row.get("source_format", "official_hdf5"))
    for source in row["segment_paths"]:
        if source_format == "lerobot_traceflow":
            chunks.append(
                np.asarray(
                    pq.read_table(source, columns=["action"])["action"].to_pylist(),
                    dtype=np.float32,
                )
            )
        else:
            with h5py.File(source, "r") as data:
                chunks.append(np.asarray(data["data/demo_0/actions"], dtype=np.float32))
    raw = np.concatenate(chunks)
    if raw.shape[1] > action_dim:
        raise ValueError(f"Action width {raw.shape[1]} exceeds packed width {action_dim}: {action_id}")
    padded = np.zeros((len(raw), action_dim), dtype=np.float32)
    padded[:, : raw.shape[1]] = raw
    return action_id, padded


def _pack_actions(
    rows: list[dict], path: Path, action_dim: int = 32, *, workers: int = 0
) -> None:
    if workers < 0:
        raise ValueError("action pack workers cannot be negative")
    representatives = {}
    for row in rows:
        representatives.setdefault(str(row["action_id"]), row)
    items = sorted(representatives.items())
    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(items)), thread_name_prefix="action-pack"
        ) as executor:
            packed = list(executor.map(lambda item: _read_action_trajectory(item, action_dim), items))
    else:
        packed = [_read_action_trajectory(item, action_dim) for item in items]
    actions, offsets, ids = [], [0], []
    for action_id, padded in packed:
        actions.append(padded)
        offsets.append(offsets[-1] + len(padded))
        ids.append(action_id)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            actions=np.concatenate(actions),
            offsets=np.asarray(offsets, dtype=np.int64),
            ids=np.asarray(ids),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lower-features", type=Path, required=True)
    parser.add_argument(
        "--lower-feature-tail-dim",
        type=int,
        default=0,
        help="Use only the final N lower-feature dimensions; 0 preserves the complete cache.",
    )
    parser.add_argument("--upper-features", type=Path)
    parser.add_argument("--upper-age", type=Path)
    parser.add_argument("--upper-available", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--retrieval-device",
        default="auto",
        help="Exact validation search device: auto, cpu, or a CUDA device.",
    )
    parser.add_argument("--retrieval-query-batch-size", type=int, default=4096)
    parser.add_argument("--projection-batch-size", type=int, default=8192)
    parser.add_argument("--action-pack-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--sampling-strategy",
        choices=("random", "task_stage_progress_pairs"),
        default="random",
    )
    parser.add_argument("--variants", default="lower,upper,fusion")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--bank-budget", type=int, default=0)
    parser.add_argument("--bank-dtype", choices=("auto", "int8", "fp16", "fp32"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-joint-conditioning", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--progress-coordinate", choices=("global", "stage"), default="global")
    parser.add_argument("--positive-progress-radius", type=float, default=0.15)
    parser.add_argument("--progress-recall-tolerance", type=float, default=0.20)
    parser.add_argument("--require-cross-trajectory-positive", action="store_true")
    parser.add_argument("--temporal-regression-weight", type=float, default=0.0)
    parser.add_argument("--temporal-regression-scale", type=float, default=0.05)
    parser.add_argument("--deploy-all-anchors", action="store_true")
    parser.add_argument("--skip-compression-study", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if not variants or any(value not in ("lower", "upper", "fusion") for value in variants):
        raise ValueError(f"Invalid head variants: {variants}")
    if args.hidden_dim <= 0 or args.out_dim <= 0 or args.lower_feature_tail_dim < 0:
        raise ValueError("hidden_dim and out_dim must be positive")
    retrieval_device = args.device if args.retrieval_device == "auto" else args.retrieval_device
    if retrieval_device != "cpu" and not retrieval_device.startswith("cuda"):
        raise ValueError(f"Invalid retrieval device: {retrieval_device}")
    if min(args.retrieval_query_batch_size, args.projection_batch_size) < 1:
        raise ValueError("retrieval and projection batch sizes must be positive")
    if args.action_pack_workers < 0:
        raise ValueError("action-pack-workers cannot be negative")
    if (
        args.positive_progress_radius <= 0
        or args.progress_recall_tolerance <= 0
        or args.temporal_regression_weight < 0
        or args.temporal_regression_scale <= 0
    ):
        raise ValueError("Invalid progress or temporal-loss parameters")
    if args.bank_budget not in (0, 1, 2, 4, 8, 16, 32, 64):
        raise ValueError("bank_budget must be 0 (auto), 1, 2, 4, 8, 16, 32, or 64")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("Head training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = _rows(args.manifest)
    lower = np.load(args.lower_features, mmap_mode="r")
    source_lower_dim = int(lower.shape[1])
    if args.lower_feature_tail_dim:
        if args.lower_feature_tail_dim > source_lower_dim:
            raise ValueError(
                f"lower-feature-tail-dim={args.lower_feature_tail_dim} exceeds "
                f"cached dimension {source_lower_dim}"
            )
        lower = lower[:, -int(args.lower_feature_tail_dim) :]
    needs_upper = any(variant in ("upper", "fusion") for variant in variants)
    if needs_upper and args.upper_features is None:
        raise ValueError("upper or fusion variants require --upper-features")
    upper = (
        np.load(args.upper_features, mmap_mode="r")
        if args.upper_features is not None
        else np.zeros((len(rows), 1), dtype=np.float32)
    )
    upper_age = (
        np.load(args.upper_age, mmap_mode="r")
        if args.upper_age is not None
        else np.zeros(len(rows), dtype=np.float32)
    )
    upper_available = (
        np.load(args.upper_available, mmap_mode="r")
        if args.upper_available is not None
        else np.ones(len(rows), dtype=np.bool_)
    )
    if (
        len(rows) != len(lower)
        or len(rows) != len(upper)
        or len(rows) != len(upper_age)
        or len(rows) != len(upper_available)
    ):
        raise RuntimeError("Manifest and feature cache lengths differ")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.isfinite(upper_age).all():
        raise RuntimeError("Feature caches are incomplete or non-finite")
    if np.any(upper_age < 0.0) or np.any(upper_age > 1.0):
        raise RuntimeError("Upper ages must be normalized to [0, 1]")
    if not np.isin(upper_available, (False, True)).all():
        raise RuntimeError("Upper availability must be boolean")
    args.output_root.mkdir(parents=True, exist_ok=True)
    task = np.asarray([row["task_id"] for row in rows], dtype=np.int64)
    progress = np.asarray([row["anchor_progress"] for row in rows], dtype=np.float32)
    if args.progress_coordinate == "stage":
        if not all("stage_progress" in row for row in rows):
            raise RuntimeError("stage progress training requires stage_progress in every manifest row")
        progress = np.asarray([row["stage_progress"] for row in rows], dtype=np.float32)
    stage = np.asarray([row["stage_index"] for row in rows], dtype=np.int64)
    trajectory_names = {name: index for index, name in enumerate(sorted({row["action_id"] for row in rows}))}
    trajectory = np.asarray(
        [trajectory_names[row["action_id"]] for row in rows],
        dtype=np.int64,
    )
    seed_values = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    train_indices = np.flatnonzero(seed_values % 5 != 0)
    validation_indices = np.flatnonzero(seed_values % 5 == 0)
    pair_sampler = (
        BalancedProgressPairSampler(
            train_indices,
            task,
            stage,
            progress,
            trajectory,
            batch_size=args.batch_size,
            positive_progress_radius=args.positive_progress_radius,
        )
        if args.sampling_strategy == "task_stage_progress_pairs"
        else None
    )
    split_by_trajectory: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        split = "validation" if seed_values[index] % 5 == 0 else "train"
        split_by_trajectory.setdefault(str(row["action_id"]), set()).add(split)
    leaked = [trajectory for trajectory, splits in split_by_trajectory.items() if len(splits) != 1]
    if leaked:
        raise RuntimeError(f"Trajectory-level train/validation leakage: {leaked[:8]}")

    lower_state_path = args.lower_features.parent / "cache_state.json"
    upper_state_path = args.upper_features.parent / "upper_cache_state.json" if args.upper_features else None
    lower_state = json.loads(lower_state_path.read_text()) if lower_state_path.is_file() else {}
    upper_state = (
        json.loads(upper_state_path.read_text())
        if upper_state_path is not None and upper_state_path.is_file()
        else {}
    )
    if args.require_joint_conditioning:
        expected_upper_protocols = {
            "predimem_trajectory_ordered_runtime_generate_subtask_hidden_v4",
            "traceflow_three_view_runtime_generate_subtask_hidden_v1",
        }
        if upper_state.get("protocol") not in expected_upper_protocols:
            raise RuntimeError(
                "Joint training requires a causal runtime-generated Upper protocol, "
                f"got {upper_state.get('protocol')}"
            )
        if not bool(upper_state.get("complete")):
            raise RuntimeError("Joint training requires a finalized upper cache")
        if lower_state.get("conditioning_protocol") != "upper_generated_subtask_v1":
            raise RuntimeError("Joint training requires lower features conditioned on upper-generated subtasks")
        upper_subtask_digest = str(upper_state.get("subtask_records_sha256", ""))
        lower_subtask_digest = str(lower_state.get("prompt_overrides_sha256", ""))
        if not upper_subtask_digest or upper_subtask_digest != lower_subtask_digest:
            raise RuntimeError(
                "Upper subtask records and lower prompt overrides do not have the same digest"
            )
        if int(upper_state.get("subtask_rows", -1)) != len(rows):
            raise RuntimeError("Joint upper cache does not contain one subtask per manifest row")
        subtask_records: dict[int, str] = {}
        with Path(lower_state["prompt_overrides"]).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    subtask_records[int(record["row_index"])] = str(record["subtask"])
        if len(subtask_records) != len(rows):
            raise RuntimeError("Joint subtask records do not cover every manifest row")
    else:
        subtask_records = {}
    provenance = {
        "manifest": str(args.manifest.resolve()),
        "lower_feature_state": lower_state,
        "upper_feature_state": upper_state,
        "head_spec": {
            "variants": list(variants),
            "hidden_dim": int(args.hidden_dim),
            "out_dim": int(args.out_dim),
            "bank_budget": int(args.bank_budget),
            "bank_dtype": str(args.bank_dtype),
            "joint_conditioning_required": bool(args.require_joint_conditioning),
            "progress_coordinate": str(args.progress_coordinate),
            "positive_progress_radius": float(args.positive_progress_radius),
            "temporal_regression_weight": float(args.temporal_regression_weight),
            "temporal_regression_scale": float(args.temporal_regression_scale),
            "deploy_all_anchors": bool(args.deploy_all_anchors),
            "source_lower_dim": source_lower_dim,
            "lower_feature_tail_dim": int(args.lower_feature_tail_dim),
            "upper_age": str(args.upper_age.resolve()) if args.upper_age is not None else "",
            "upper_available": (
                str(args.upper_available.resolve()) if args.upper_available is not None else ""
            ),
            "retrieval_device": retrieval_device,
            "retrieval_query_batch_size": int(args.retrieval_query_batch_size),
            "projection_batch_size": int(args.projection_batch_size),
            "sampling_strategy": str(args.sampling_strategy),
            "sampler": pair_sampler.summary if pair_sampler is not None else None,
        },
    }
    (args.output_root / "retrieval_training_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    all_keys = {}
    best_metrics = {}
    for variant in variants:
        head = DualTowerRetrievalHead(
            variant=variant,
            lower_dim=lower.shape[1],
            upper_dim=upper.shape[1],
            hidden=args.hidden_dim,
            out_dim=args.out_dim,
        ).to(args.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        best_score = -1.0
        variant_dir = args.output_root / "heads" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        start_epoch = 1
        last_path = variant_dir / "last.pt"
        if args.resume and last_path.is_file():
            last = torch.load(last_path, map_location=args.device, weights_only=False)
            expected = (variant, int(lower.shape[1]), int(upper.shape[1]), args.hidden_dim, args.out_dim)
            actual = (
                str(last["variant"]),
                int(last["lower_dim"]),
                int(last["upper_dim"]),
                int(last["hidden"]),
                int(last["out_dim"]),
            )
            if actual != expected:
                raise RuntimeError(f"Resume head dimensions differ: actual={actual} expected={expected}")
            head.load_state_dict(last["state_dict"])
            optimizer_state = last.get("optimizer_state")
            if optimizer_state is None:
                raise RuntimeError(f"Resume checkpoint lacks optimizer state: {last_path}")
            optimizer.load_state_dict(optimizer_state)
            best_score = float(last.get("best_score", -1.0))
            start_epoch = int(last["epoch"]) + 1
            print(f"Resuming {variant} head from epoch {start_epoch}", flush=True)
        for epoch in range(start_epoch, args.epochs + 1):
            generator = np.random.default_rng(args.seed * 1_000_003 + epoch)
            head.train()
            losses = []
            positive_coverages = []
            for _ in range(args.steps_per_epoch):
                indices = (
                    pair_sampler.sample(generator)
                    if pair_sampler is not None
                    else generator.choice(
                        train_indices,
                        size=min(args.batch_size, len(train_indices)),
                        replace=False,
                    )
                )
                lo, up = _inputs(variant, lower, upper, indices, args.device)
                if variant == "fusion":
                    age = torch.from_numpy(
                        np.asarray(upper_age[indices], dtype=np.float32).copy()
                    ).to(args.device)
                    source_available = torch.from_numpy(
                        np.asarray(upper_available[indices], dtype=np.float32).copy()
                    ).to(args.device)
                    available = source_available * (
                        torch.rand(len(indices), device=args.device) >= 0.10
                    ).float()
                else:
                    age = torch.zeros(len(indices), device=args.device)
                    available = torch.ones(len(indices), device=args.device)
                embedding = head(lo, up, age, available)
                batch_task = torch.from_numpy(task[indices]).to(args.device)
                batch_stage = torch.from_numpy(stage[indices]).to(args.device)
                batch_progress = torch.from_numpy(progress[indices]).to(args.device)
                batch_trajectory = torch.from_numpy(trajectory[indices]).to(args.device)
                _, positive_mask = _positive_pair_mask(
                    batch_task,
                    batch_stage,
                    batch_progress,
                    batch_trajectory,
                    positive_progress_radius=args.positive_progress_radius,
                    require_cross_trajectory_positive=args.require_cross_trajectory_positive,
                )
                positive_coverage = float(positive_mask.any(dim=1).float().mean().item())
                if pair_sampler is not None and positive_coverage != 1.0:
                    raise RuntimeError(
                        f"Balanced sampler produced incomplete positive coverage: {positive_coverage:.6f}"
                    )
                loss = _contrastive_loss(
                    embedding,
                    batch_task,
                    batch_stage,
                    batch_progress,
                    batch_trajectory,
                    args.temperature,
                    positive_progress_radius=args.positive_progress_radius,
                    require_cross_trajectory_positive=args.require_cross_trajectory_positive,
                    temporal_regression_weight=args.temporal_regression_weight,
                    temporal_regression_scale=args.temporal_regression_scale,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                positive_coverages.append(positive_coverage)
            head.eval()
            keys = _project(
                head,
                lower,
                upper,
                args.device,
                upper_age=upper_age,
                upper_available=upper_available,
                batch_size=args.projection_batch_size,
            )
            metrics = _retrieval_metrics(
                keys,
                rows,
                validation_indices,
                train_indices,
                progress_field=(
                    "stage_progress" if args.progress_coordinate == "stage" else "anchor_progress"
                ),
                progress_tolerance=args.progress_recall_tolerance,
                retrieval_device=retrieval_device,
                retrieval_query_batch_size=args.retrieval_query_batch_size,
            )
            score = (
                metrics["macro_task_stage_frame_recall@1_6"]
                if args.progress_coordinate == "stage"
                else metrics["task_stage_progress_recall@1"]
            )
            record = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "positive_anchor_fraction": float(np.mean(positive_coverages)),
                **metrics,
            }
            with (variant_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            improved = score > best_score
            best_score = max(best_score, score)
            payload = checkpoint_payload(
                head,
                head.state_dict(),
                epoch=epoch,
                metrics=metrics,
                optimizer_state=optimizer.state_dict(),
                best_score=best_score,
                lower_producer_policy_dir=lower_state.get("policy_dir"),
                lower_producer_config_name=lower_state.get("config_name"),
                upper_producer_checkpoint=upper_state.get("checkpoint"),
                feature_manifest_sha256=lower_state.get("manifest_sha256"),
                joint_conditioned=bool(args.require_joint_conditioning),
                lower_conditioning_protocol=lower_state.get("conditioning_protocol"),
                lower_feature_protocol=(
                    "cl_prefix_current_frame_view_v1"
                    if args.lower_feature_tail_dim
                    else lower_state.get("extraction_protocol")
                ),
                temporal_window=(
                    1
                    if args.lower_feature_tail_dim
                    else int(lower_state.get("temporal_window", 1))
                ),
                temporal_offsets=(
                    [0]
                    if args.lower_feature_tail_dim
                    else list(lower_state.get("temporal_offsets", [0]))
                ),
                upper_feature_protocol=upper_state.get("protocol"),
                subtask_records_sha256=lower_state.get("prompt_overrides_sha256"),
            )
            torch.save(payload, variant_dir / "last.pt")
            if improved:
                torch.save(payload, variant_dir / "best.pt")
            print(json.dumps({"variant": variant, **record}), flush=True)
        best = torch.load(variant_dir / "best.pt", map_location=args.device, weights_only=False)
        head.load_state_dict(best["state_dict"])
        all_keys[variant] = _project(
            head.eval(),
            lower,
            upper,
            args.device,
            upper_age=upper_age,
            upper_available=upper_available,
            batch_size=args.projection_batch_size,
        )
        best_metrics[variant] = dict(best["metrics"])

    compression = {}
    selected_configs = {}
    for variant, source_keys in all_keys.items():
        if args.skip_compression_study:
            selected = {
                "budget": "all" if args.deploy_all_anchors else int(args.bank_budget),
                "dtype": "fp32" if args.bank_dtype == "auto" else str(args.bank_dtype),
                "keys": int(len(source_keys) if args.deploy_all_anchors else 0),
                **best_metrics[variant],
            }
            compression[variant] = {"candidates": [], "selected": selected}
            selected_configs[variant] = selected
            continue
        oracle_budget = max(16, int(args.bank_budget))
        oracle_indices = _anchor_subset(rows, oracle_budget, train_indices)
        oracle = _retrieval_metrics(
            source_keys,
            rows,
            validation_indices,
            oracle_indices,
            retrieval_device=retrieval_device,
            retrieval_query_batch_size=args.retrieval_query_batch_size,
        )
        oracle_neighbors = _neighbors(
            source_keys,
            validation_indices,
            oracle_indices,
            retrieval_device=retrieval_device,
            retrieval_query_batch_size=args.retrieval_query_batch_size,
        )
        candidates = []
        for budget in (1, 2, 4, 8, 16, 32, 64):
            indices = _anchor_subset(rows, budget, train_indices)
            for dtype in ("int8", "fp16", "fp32"):
                keys = _quantize(source_keys, dtype)
                metrics = _retrieval_metrics(
                    keys,
                    rows,
                    validation_indices,
                    indices,
                    oracle_neighbors=oracle_neighbors,
                    retrieval_device=retrieval_device,
                    retrieval_query_batch_size=args.retrieval_query_batch_size,
                )
                bytes_per_value = {"int8": 1, "fp16": 2, "fp32": 4}[dtype]
                candidates.append(
                    {
                        "budget": budget,
                        "dtype": dtype,
                        "keys": len(indices),
                        "key_bytes": len(indices) * int(args.out_dim) * bytes_per_value,
                        **metrics,
                    }
                )
        if args.bank_budget and args.bank_dtype != "auto":
            selected = next(
                item
                for item in candidates
                if item["budget"] == args.bank_budget and item["dtype"] == args.bank_dtype
            )
        else:
            eligible = [
                item for item in candidates
                if item["task_stage_progress_recall@1"] >= oracle["task_stage_progress_recall@1"] - 0.02
                and item["task_stage_progress_recall@8"] >= oracle["task_stage_progress_recall@8"] - 0.01
                and item["candidate_oracle_topk_overlap"] >= 0.80
            ]
            if eligible:
                selected = min(
                    eligible, key=lambda item: (item["key_bytes"], -item["task_stage_progress_recall@1"])
                )
            else:
                selected = max(
                    candidates,
                    key=lambda item: (
                        item["task_stage_progress_recall@1"],
                        item["task_stage_progress_recall@8"],
                        -item["key_bytes"],
                    ),
                )
        compression[variant] = {"candidate_oracle": oracle, "candidates": candidates, "selected": selected}
        selected_configs[variant] = selected

    shared = args.output_root / "memory" / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    _pack_actions(
        rows,
        shared / "gpm_memory_actions.npz",
        workers=args.action_pack_workers,
    )
    for variant, source_keys in all_keys.items():
        selected = selected_configs[variant]
        indices = (
            np.arange(len(rows), dtype=np.int64)
            if selected["budget"] == "all"
            else _anchor_subset(rows, int(selected["budget"]))
        )
        keys = _quantize(source_keys, str(selected["dtype"]))[indices]
        bank = args.output_root / "memory" / variant
        bank.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(keys.shape[1])
        index.add(keys)
        faiss.write_index(index, str(bank / "gpm_memory.index"))
        metadata = []
        for source_index, key in zip(indices.tolist(), keys, strict=True):
            row = rows[source_index]
            metadata.append(
                {
                    "task_name": row["prompt"],
                    "task_id": int(row["task_id"]),
                    "suite": row["suite"],
                    "stage_index": int(row["stage_index"]),
                    "stage_start_frame": int(row.get("stage_start_frame", 0)),
                    "stage_length": int(row.get("stage_length", row["trajectory_length"])),
                    "stage_frame_index": int(row.get("stage_frame_index", row["global_frame_index"])),
                    "stage_progress": float(row.get("stage_progress", row["anchor_progress"])),
                    "anchor_progress": float(row["anchor_progress"]),
                    "conditioning_subtask": subtask_records.get(source_index, ""),
                    "task_emb": torch.from_numpy(key.copy()),
                    "action_id": row["action_id"],
                    "length": int(row["trajectory_length"]),
                    "chunk_meta": {"chunk_len": 10, "stride": 10, "T": int(row["trajectory_length"])},
                }
            )
        torch.save(metadata, bank / "gpm_memory_meta.pt")
    (args.output_root / "compression_report.json").write_text(
        json.dumps(compression, indent=2) + "\n", encoding="utf-8"
    )
    if args.progress_coordinate == "stage":
        best_variant = max(
            variants,
            key=lambda variant: (
                best_metrics[variant]["macro_task_stage_frame_recall@1_6"],
                -best_metrics[variant]["stage_frame_mae@1"],
                best_metrics[variant]["macro_task_stage_recall@1"],
            ),
        )
    else:
        best_variant = max(
            variants,
            key=lambda variant: (
                best_metrics[variant]["task_stage_progress_recall@1"],
                best_metrics[variant]["task_stage_progress_recall@8"],
                best_metrics[variant]["task_progress_recall@1"],
            ),
        )
    best_payload = {
        "variant": best_variant,
        "metrics": best_metrics[best_variant],
        "head": str((args.output_root / "heads" / best_variant / "best.pt").resolve()),
        "memory": str((args.output_root / "memory" / best_variant).resolve()),
    }
    (args.output_root / "best_variant.json").write_text(
        json.dumps(best_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "selected": selected_configs,
                "best_variant": best_payload,
            },
            indent=2,
        )
    )


def _run_with_heartbeat() -> None:
    started = time.monotonic()
    finished = threading.Event()

    def report_progress() -> None:
        while not finished.wait(timeout=30.0):
            print(f"PrediMem head pipeline active: elapsed_seconds={time.monotonic() - started:.0f}", flush=True)

    reporter = threading.Thread(target=report_progress, name="head-pipeline-reporter", daemon=True)
    reporter.start()
    try:
        main()
    finally:
        finished.set()
        reporter.join(timeout=1.0)


if __name__ == "__main__":
    _run_with_heartbeat()
