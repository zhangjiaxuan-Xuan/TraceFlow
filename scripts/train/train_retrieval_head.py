#!/usr/bin/env python3
"""Train a runtime-compatible TraceFlow retrieval head from cached features.

The feature-cache boundary is intentionally generic: users can connect any
lower policy or upper VLM while reusing the head, sampler, loss, metrics, and
checkpoint format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OPENPI_SRC = ROOT / "openpi" / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

from openpi.task_head.dual_tower_head import DualTowerRetrievalHead, checkpoint_payload  # noqa: E402
from openpi.task_head.reproduction import (  # noqa: E402
    TaskPairBatchSampler,
    retrieval_metrics,
    supervised_contrastive_loss,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path.name}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"metadata row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("metadata is empty")
    return rows


def _labels(rows: list[dict[str, Any]], field: str) -> tuple[np.ndarray, dict[str, int]]:
    missing = [index for index, row in enumerate(rows) if field not in row]
    if missing:
        raise ValueError(f"metadata lacks label field {field!r} at rows {missing[:8]}")
    names = [str(row[field]) for row in rows]
    mapping = {name: index for index, name in enumerate(sorted(set(names)))}
    return np.asarray([mapping[name] for name in names], dtype=np.int64), mapping


def _stable_validation_group(value: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    uniform = int.from_bytes(digest[:8], "big") / float(2**64)
    return uniform < fraction


def _split_indices(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    *,
    split_field: str,
    group_field: str,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    explicit = bool(split_field) and all(split_field in row for row in rows)
    if explicit:
        values = [str(row[split_field]).strip().lower() for row in rows]
        train = np.asarray(
            [i for i, value in enumerate(values) if value in {"train", "training"}], dtype=np.int64
        )
        validation = np.asarray(
            [i for i, value in enumerate(values) if value in {"val", "valid", "validation", "test"}],
            dtype=np.int64,
        )
        known = {"train", "training", "val", "valid", "validation", "test"}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"unsupported values in {split_field!r}: {unknown}")
    else:
        missing = [i for i, row in enumerate(rows) if group_field not in row]
        if missing:
            raise ValueError(
                f"metadata needs either a complete {split_field!r} field or group field {group_field!r}"
            )
        validation_mask = np.asarray(
            [_stable_validation_group(str(row[group_field]), seed, validation_fraction) for row in rows],
            dtype=np.bool_,
        )
        validation = np.flatnonzero(validation_mask)
        train = np.flatnonzero(~validation_mask)
    if not len(train) or not len(validation):
        raise ValueError("training and validation splits must both be non-empty")
    absent = sorted(set(labels[validation].tolist()).difference(labels[train].tolist()))
    if absent:
        raise ValueError(f"validation contains labels absent from training: {absent}")
    if group_field and all(group_field in row for row in rows):
        train_groups = {str(rows[i][group_field]) for i in train}
        validation_groups = {str(rows[i][group_field]) for i in validation}
        overlap = sorted(train_groups.intersection(validation_groups))
        if overlap:
            raise ValueError(f"group leakage between splits for {group_field!r}: {overlap[:8]}")
    return train, validation


def _load_array(path: Path | None, name: str, rows: int) -> np.ndarray | None:
    if path is None:
        return None
    value = np.load(path, mmap_mode="r")
    if value.ndim != 2 or value.shape[0] != rows:
        raise ValueError(f"{name} must have shape [N,D] with N={rows}; got {value.shape}")
    if value.shape[1] < 1:
        raise ValueError(f"{name} has no feature dimensions")
    return value


def _device(value: str) -> torch.device:
    selected = "cuda:0" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _batch_inputs(
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    def convert(value: np.ndarray | None) -> torch.Tensor | None:
        if value is None:
            return None
        array = np.asarray(value[indices], dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("feature batch contains NaN or infinity")
        return torch.from_numpy(array.copy()).to(device)

    return convert(lower), convert(upper)


@torch.inference_mode()
def _project(
    head: DualTowerRetrievalHead,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    age: np.ndarray,
    available: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    head.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        lo, up = _batch_inputs(lower, upper, selected, device)
        batch_age = torch.from_numpy(np.asarray(age[selected], dtype=np.float32).copy()).to(device)
        batch_available = torch.from_numpy(np.asarray(available[selected], dtype=np.float32).copy()).to(device)
        chunks.append(head(lo, up, batch_age, batch_available).cpu())
    return torch.cat(chunks)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="JSONL with one object per feature row")
    parser.add_argument("--lower-features", type=Path, help="[N,D] .npy lower-policy features")
    parser.add_argument("--upper-features", type=Path, help="[N,D] .npy upper-model features")
    parser.add_argument("--upper-age", type=Path, help="optional [N] normalized age .npy for fusion")
    parser.add_argument("--upper-available", type=Path, help="optional [N] availability .npy for fusion")
    parser.add_argument("--variant", choices=("lower", "upper", "fusion"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-field", default="task_id")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--group-field", default="action_id")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--labels-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-label", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection-batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    return parser


def train(args: argparse.Namespace) -> Path:
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1")
    positive_names = (
        "hidden_dim", "out_dim", "epochs", "steps_per_epoch", "labels_per_batch",
        "samples_per_label", "projection_batch_size",
    )
    for name in positive_names:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.samples_per_label < 2:
        raise ValueError("samples-per-label must be at least 2")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.temperature <= 0:
        raise ValueError("optimizer values and temperature are invalid")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    rows = _read_jsonl(args.metadata)
    labels, label_mapping = _labels(rows, args.label_field)
    lower = _load_array(args.lower_features, "lower-features", len(rows))
    upper = _load_array(args.upper_features, "upper-features", len(rows))
    if args.variant in {"lower", "fusion"} and lower is None:
        raise ValueError(f"{args.variant} training requires --lower-features")
    if args.variant in {"upper", "fusion"} and upper is None:
        raise ValueError(f"{args.variant} training requires --upper-features")

    age = np.load(args.upper_age, mmap_mode="r") if args.upper_age else np.zeros(len(rows), dtype=np.float32)
    available = np.load(args.upper_available, mmap_mode="r") if args.upper_available else np.ones(len(rows), dtype=np.float32)
    if age.shape != (len(rows),) or available.shape != (len(rows),):
        raise ValueError("upper age and availability arrays must have shape [N]")
    if not np.isfinite(age).all() or np.any(age < 0) or np.any(age > 1):
        raise ValueError("upper age must be finite and normalized to [0,1]")
    if not np.isin(available, (0, 1, False, True)).all():
        raise ValueError("upper availability must contain only 0/1 values")

    train_indices, validation_indices = _split_indices(
        rows,
        labels,
        split_field=args.split_field,
        group_field=args.group_field,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    sampler = TaskPairBatchSampler(
        train_indices,
        labels[train_indices],
        tasks_per_batch=args.labels_per_batch,
        episodes_per_task=args.samples_per_label,
        batches_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )
    lower_dim = int(lower.shape[1]) if lower is not None else 1
    upper_dim = int(upper.shape[1]) if upper is not None else 1
    head = DualTowerRetrievalHead(
        variant=args.variant,
        lower_dim=lower_dim,
        upper_dim=upper_dim,
        hidden=args.hidden_dim,
        out_dim=args.out_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    metrics_path = args.output_dir / "metrics.jsonl"
    start_epoch, best_score = 1, -1.0
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        expected = (args.variant, lower_dim, upper_dim, args.hidden_dim, args.out_dim)
        actual = tuple(checkpoint[name] for name in ("variant", "lower_dim", "upper_dim", "hidden", "out_dim"))
        if actual != expected:
            raise ValueError(f"resume architecture mismatch: {actual} != {expected}")
        head.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -1.0))

    provenance = {
        "schema": "traceflow_generic_retrieval_training_v1",
        "variant": args.variant,
        "metadata": {"name": args.metadata.name, "sha256": _sha256(args.metadata), "rows": len(rows)},
        "lower_features": None if args.lower_features is None else {
            "name": args.lower_features.name, "sha256": _sha256(args.lower_features), "shape": list(lower.shape)
        },
        "upper_features": None if args.upper_features is None else {
            "name": args.upper_features.name, "sha256": _sha256(args.upper_features), "shape": list(upper.shape)
        },
        "label_field": args.label_field,
        "label_mapping": label_mapping,
        "split_field": args.split_field,
        "group_field": args.group_field,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "seed": int(args.seed),
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez(args.output_dir / "split_indices.npz", train=train_indices, validation=validation_indices)

    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        head.train()
        losses: list[float] = []
        for selected_list in sampler:
            selected = np.asarray(selected_list, dtype=np.int64)
            lo, up = _batch_inputs(lower, upper, selected, device)
            batch_age = torch.from_numpy(np.asarray(age[selected], dtype=np.float32).copy()).to(device)
            batch_available = torch.from_numpy(np.asarray(available[selected], dtype=np.float32).copy()).to(device)
            embedding = head(lo, up, batch_age, batch_available)
            batch_labels = torch.from_numpy(labels[selected]).to(device)
            loss = supervised_contrastive_loss(embedding, batch_labels, temperature=args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_keys = _project(
            head, lower, upper, age, available, train_indices, device, args.projection_batch_size
        )
        validation_query = _project(
            head, lower, upper, age, available, validation_indices, device, args.projection_batch_size
        )
        metrics = retrieval_metrics(
            validation_query,
            torch.from_numpy(labels[validation_indices]),
            train_keys,
            torch.from_numpy(labels[train_indices]),
        )
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        score = float(metrics["recall@1"])
        improved = score > best_score
        best_score = max(best_score, score)
        payload = checkpoint_payload(
            head,
            head.state_dict(),
            epoch=epoch,
            metrics=metrics,
            best_score=best_score,
            optimizer_state=optimizer.state_dict(),
            training_provenance=provenance,
            label_mapping=label_mapping,
            temporal_window=1,
            temporal_offsets=[0],
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        print(json.dumps(record, sort_keys=True), flush=True)
    return best_path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(train(args))


if __name__ == "__main__":
    main()
