from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


LIBERO_SUITES = ("libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def manifest_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_snapshot_root(dataset_root: str | Path) -> tuple[Path, str]:
    root = Path(dataset_root).expanduser().resolve()
    refs_main = root / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = root / "snapshots" / revision
    else:
        snapshot = root
        revision = root.name if len(root.name) == 40 else "unknown"
    missing = [suite for suite in LIBERO_SUITES if not (snapshot / suite).is_dir()]
    if missing:
        raise FileNotFoundError(f"Dataset snapshot is missing suites {missing}: {snapshot}")
    return snapshot, revision


def numeric_demo_key(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected LIBERO demonstration name: {name}") from exc


def label_values(rows: Sequence[dict[str, Any]], mode: str) -> tuple[np.ndarray, dict[str, int]]:
    if mode == "task_file":
        values = [str(row["task_key"]) for row in rows]
    elif mode == "language":
        values = [str(row["prompt"]) for row in rows]
    else:
        raise ValueError(f"Unknown label mode: {mode}")
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.asarray([mapping[value] for value in values], dtype=np.int64), mapping


def select_indices(
    rows: Sequence[dict[str, Any]],
    completed: np.ndarray,
    split: str,
) -> np.ndarray:
    if split == "all":
        selected = np.arange(len(rows), dtype=np.int64)
    else:
        selected = np.asarray([i for i, row in enumerate(rows) if row["split"] == split], dtype=np.int64)
    selected = selected[np.asarray(completed[selected], dtype=np.bool_)]
    if selected.size == 0:
        raise ValueError(f"No completed feature rows found for split={split!r}")
    return selected


class TaskPairBatchSampler(Sampler[list[int]]):
    """Sample a fixed number of different labels and episodes per label."""

    def __init__(
        self,
        indices: Sequence[int] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        *,
        tasks_per_batch: int,
        episodes_per_task: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        self.tasks_per_batch = int(tasks_per_batch)
        self.episodes_per_task = int(episodes_per_task)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        if self.tasks_per_batch <= 0 or self.episodes_per_task < 2 or self.batches_per_epoch <= 0:
            raise ValueError("Task-pair batches require positive sizes and at least two episodes per task")

        self.by_label: dict[int, np.ndarray] = {}
        for index, label in zip(np.asarray(indices, dtype=np.int64), np.asarray(labels, dtype=np.int64), strict=True):
            self.by_label.setdefault(int(label), []).append(int(index))  # type: ignore[union-attr]
        self.by_label = {
            label: np.asarray(values, dtype=np.int64)
            for label, values in self.by_label.items()
            if len(values) >= self.episodes_per_task
        }
        if len(self.by_label) < self.tasks_per_batch:
            raise ValueError(
                f"Need {self.tasks_per_batch} labels with at least {self.episodes_per_task} samples; "
                f"found {len(self.by_label)}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterable[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        label_ids = np.asarray(sorted(self.by_label), dtype=np.int64)
        for _ in range(self.batches_per_epoch):
            chosen_labels = rng.choice(label_ids, size=self.tasks_per_batch, replace=False)
            batch: list[int] = []
            for label in chosen_labels.tolist():
                examples = self.by_label[int(label)]
                batch.extend(rng.choice(examples, size=self.episodes_per_task, replace=False).tolist())
            rng.shuffle(batch)
            yield batch


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Expected embeddings [B,D] and labels [B]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = embeddings @ embeddings.T / float(temperature)
    self_mask = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    if not bool(torch.all(positives.any(dim=1))):
        raise ValueError("Every contrastive anchor must have at least one positive")
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # masked_fill avoids the undefined -inf * 0 operation.
    positive_log_prob = log_prob.masked_fill(~positives, 0.0)
    return -(positive_log_prob.sum(dim=1) / positives.sum(dim=1)).mean()


@torch.inference_mode()
def retrieval_metrics(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    keys: torch.Tensor,
    key_labels: torch.Tensor,
    *,
    ks: Sequence[int] = (1, 4, 8),
    block_size: int = 256,
) -> dict[str, float]:
    query = F.normalize(query.float(), dim=-1)
    keys = F.normalize(keys.float(), dim=-1)
    maximum_k = min(max(int(k) for k in ks), keys.shape[0])
    hits = {int(k): 0 for k in ks}
    total = 0
    positive_scores: list[torch.Tensor] = []
    negative_scores: list[torch.Tensor] = []
    for start in range(0, query.shape[0], int(block_size)):
        stop = min(start + int(block_size), query.shape[0])
        scores = query[start:stop] @ keys.T
        values, neighbors = torch.topk(scores, k=maximum_k, dim=1)
        neighbor_labels = key_labels[neighbors]
        expected = query_labels[start:stop, None]
        for k in hits:
            hits[k] += int((neighbor_labels[:, : min(k, maximum_k)] == expected).any(dim=1).sum().item())
        match = neighbor_labels[:, 0].eq(expected[:, 0])
        positive_scores.append(values[:, 0][match].cpu())
        negative_scores.append(values[:, 0][~match].cpu())
        total += stop - start
    result = {f"recall@{k}": hits[k] / max(total, 1) for k in sorted(hits)}
    positives = torch.cat([x for x in positive_scores if x.numel()]) if any(x.numel() for x in positive_scores) else None
    negatives = torch.cat([x for x in negative_scores if x.numel()]) if any(x.numel() for x in negative_scores) else None
    result["top1_positive_score"] = float(positives.mean().item()) if positives is not None else float("nan")
    result["top1_wrong_score"] = float(negatives.mean().item()) if negatives is not None else float("nan")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
