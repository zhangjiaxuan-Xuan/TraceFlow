from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
from typing import Any, Literal

import faiss
import h5py
import numpy as np
import torch
import tyro

from openpi.task_head.reproduction import manifest_digest
from openpi.task_head.reproduction import read_jsonl
from openpi.task_head.reproduction import write_json_atomic
from openpi.task_head.task_head_mlp import TaskHeadMLP

Admission = Literal["success", "failure", "both"]
_REQUIRED_MANIFEST_FIELDS = {
    "action_id",
    "trajectory_path",
    "source_format",
    "success",
    "prompt",
    "task_id",
    "episode_idx",
}


@dataclasses.dataclass
class Args:
    group: str
    manifest: str
    feature_dir: str
    checkpoint: str
    output_dir: str
    baseline_positive_meta: str = ""
    baseline_positive_index: str = ""
    baseline_positive_actions: str = ""
    baseline_negative_meta: str = ""
    baseline_negative_index: str = ""
    baseline_negative_actions: str = ""
    admission: Admission = "both"
    device: str = "cuda"
    overwrite: bool = False


@dataclasses.dataclass
class _Bank:
    metadata: list[dict[str, Any]]
    keys: np.ndarray
    action_parts: list[np.ndarray]
    ids: list[str]


def _empty_bank(embedding_dim: int) -> _Bank:
    return _Bank(
        metadata=[],
        keys=np.empty((0, int(embedding_dim)), dtype=np.float32),
        action_parts=[],
        ids=[],
    )


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _task_embedding(entry: dict[str, Any], *, source: Path, row: int) -> np.ndarray:
    if "task_emb" not in entry:
        raise ValueError(f"Missing task_emb in {source} metadata row {row}")
    value = entry["task_emb"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(embedding).all():
        raise ValueError(f"Non-finite task_emb in {source} metadata row {row}")
    return embedding


def _index_keys(index: faiss.Index, count: int, *, source: Path) -> np.ndarray:
    try:
        keys = np.asarray(index.reconstruct_n(0, count), dtype=np.float32)
    except RuntimeError as exc:
        raise ValueError(f"Baseline index does not support key reconstruction: {source}") from exc
    if keys.shape != (count, int(index.d)):
        raise ValueError(f"Unexpected reconstructed index shape {keys.shape} in {source}")
    if not np.isfinite(keys).all():
        raise ValueError(f"Non-finite key in baseline index: {source}")
    return np.ascontiguousarray(keys)


def _load_bank(meta_path: str, index_path: str, actions_path: str, *, name: str) -> _Bank:
    meta_source = Path(meta_path).resolve()
    index_source = Path(index_path).resolve()
    actions_source = Path(actions_path).resolve()
    metadata = _torch_load(meta_source)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"Baseline {name} metadata must be a non-empty list: {meta_source}")
    if not all(isinstance(entry, dict) for entry in metadata):
        raise ValueError(f"Baseline {name} metadata contains a non-dict entry: {meta_source}")

    index = faiss.read_index(str(index_source))
    with np.load(actions_source, allow_pickle=False) as packed:
        required = {"actions", "offsets", "ids"}
        if not required.issubset(packed.files):
            raise ValueError(f"Baseline {name} actions must contain {sorted(required)}: {actions_source}")
        actions = np.asarray(packed["actions"], dtype=np.float32)
        offsets = np.asarray(packed["offsets"], dtype=np.int64)
        ids = [str(value) for value in np.asarray(packed["ids"]).tolist()]

    count = len(metadata)
    if int(index.ntotal) != count or len(ids) != count or offsets.shape != (count + 1,):
        raise ValueError(
            f"Baseline {name} length mismatch: meta={count}, index={int(index.ntotal)}, "
            f"actions={len(ids)}, offsets={offsets.shape}"
        )
    if actions.ndim != 2 or actions.shape[1] != 32 or not np.isfinite(actions).all():
        raise ValueError(f"Baseline {name} actions must be finite [steps,32], got {actions.shape}")
    if offsets[0] != 0 or offsets[-1] != actions.shape[0] or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"Invalid baseline {name} packed action offsets: {actions_source}")
    if len(set(ids)) != count:
        raise ValueError(f"Baseline {name} action IDs are not unique: {actions_source}")

    meta_ids = []
    meta_embeddings = []
    for row, entry in enumerate(metadata):
        if "action_id" not in entry:
            raise ValueError(f"Missing action_id in baseline {name} metadata row {row}")
        meta_ids.append(str(entry["action_id"]))
        meta_embeddings.append(_task_embedding(entry, source=meta_source, row=row))
    if meta_ids != ids:
        raise ValueError(f"Baseline {name} metadata/action IDs differ or are out of order")
    if len(set(meta_ids)) != count:
        raise ValueError(f"Baseline {name} metadata action IDs are not unique")

    embedding_dims = {embedding.shape[0] for embedding in meta_embeddings}
    if embedding_dims != {int(index.d)}:
        raise ValueError(
            f"Baseline {name} embedding dimensions {sorted(embedding_dims)} do not match index dim {index.d}"
        )
    keys = _index_keys(index, count, source=index_source)
    expected = np.stack(meta_embeddings)
    if not np.allclose(keys, expected, rtol=1e-4, atol=1e-5):
        max_error = float(np.max(np.abs(keys - expected)))
        raise ValueError(f"Baseline {name} index keys differ from metadata task_emb (max error {max_error:.3g})")

    action_parts = [actions[int(offsets[i]) : int(offsets[i + 1])].copy() for i in range(count)]
    return _Bank(metadata=list(metadata), keys=keys, action_parts=action_parts, ids=ids)


def _validate_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    for row_number, row in enumerate(rows, 1):
        missing = sorted(_REQUIRED_MANIFEST_FIELDS - row.keys())
        if missing:
            raise ValueError(f"Manifest row {row_number} is missing fields {missing}")
        if not any(str(key).startswith("source_") and key != "source_format" for key in row):
            raise ValueError(f"Manifest row {row_number} must contain provenance source_* fields")
        if row["source_format"] not in {"eval_hdf5", "smol_npz"}:
            raise ValueError(f"Unsupported source_format={row['source_format']!r} on manifest row {row_number}")
        if not isinstance(row["success"], bool):
            raise ValueError(f"Manifest success must be boolean on row {row_number}")
        int(row["task_id"])
        int(row["episode_idx"])


def _load_raw_actions(row: dict[str, Any], manifest_path: Path) -> np.ndarray:
    trajectory_path = Path(str(row["trajectory_path"])).expanduser()
    if not trajectory_path.is_absolute():
        trajectory_path = manifest_path.parent / trajectory_path
    if row["source_format"] == "eval_hdf5":
        with h5py.File(trajectory_path, "r") as handle:
            raw = np.asarray(handle["data/demo_0/actions"], dtype=np.float32)
    else:
        with np.load(trajectory_path, allow_pickle=False) as packed:
            if "actions" not in packed.files:
                raise KeyError(f"Missing actions in {trajectory_path}")
            raw = np.asarray(packed["actions"], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"Expected raw actions [T,7], got {raw.shape} from {trajectory_path}")
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite raw actions in {trajectory_path}")
    # Match the command that LIBERO's OSC controller actually executes. This is
    # especially important for cross-policy SmolVLA trajectories, whose inverse
    # normalization can produce small excursions beyond the controller range.
    raw = np.clip(raw, -1.0, 1.0)
    raw[:, 6] = np.where(raw[:, 6] < 0.0, -1.0, 1.0)
    padded = np.zeros((raw.shape[0], 32), dtype=np.float32)
    padded[:, :7] = raw
    return padded


def _unique_action_id(row: dict[str, Any], label: str, used_ids: set[str]) -> str:
    source_id = str(row["action_id"]).strip()
    if not source_id:
        raise ValueError("Manifest action_id must not be empty")
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"cl_{label}_{source_id}")
    if candidate in used_ids:
        raise ValueError(f"Duplicate action_id after normalization: {candidate}")
    used_ids.add(candidate)
    return candidate


@torch.inference_mode()
def _project_features(checkpoint_path: str, features: np.ndarray, device: str) -> np.ndarray:
    checkpoint = _torch_load(checkpoint_path)
    required = {"state_dict", "in_dim", "hidden", "out_dim"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Invalid TaskHeadMLP checkpoint: {checkpoint_path}")
    if features.ndim != 2 or features.shape[1] != int(checkpoint["in_dim"]):
        raise ValueError(f"Feature shape {features.shape} does not match checkpoint input dim {checkpoint['in_dim']}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    head = TaskHeadMLP(
        in_dim=int(checkpoint["in_dim"]),
        hidden=int(checkpoint["hidden"]),
        out_dim=int(checkpoint["out_dim"]),
    )
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head = head.eval().float().to(device)
    projected = []
    for start in range(0, features.shape[0], 512):
        batch = torch.from_numpy(np.asarray(features[start : start + 512], dtype=np.float32).copy()).to(device)
        projected.append(head(batch).cpu().numpy())
    keys = np.ascontiguousarray(np.concatenate(projected), dtype=np.float32)
    if not np.isfinite(keys).all():
        raise ValueError("TaskHeadMLP produced non-finite keys")
    faiss.normalize_L2(keys)
    return keys


def _append(bank: _Bank, row: dict[str, Any], key: np.ndarray, actions: np.ndarray, action_id: str) -> None:
    label = "positive" if bool(row["success"]) else "negative"
    provenance = dict(row)
    metadata: dict[str, Any] = {
        "label": label,
        "task_name": str(row["prompt"]),
        "task_id": int(row["task_id"]),
        "episode_idx": int(row["episode_idx"]),
        "success": bool(row["success"]),
        "task_emb": torch.from_numpy(key.copy()),
        "action_id": action_id,
        "length": int(actions.shape[0]),
        "provenance": provenance,
        "chunk_meta": {
            "chunk_len": 10,
            "stride": 10,
            "num_chunks": max(0, 1 + (int(actions.shape[0]) - 10) // 10),
            "T": int(actions.shape[0]),
            "A_raw": 7,
            "A_model": 32,
        },
        "action_protocol": "libero_osc_clip_and_binary_gripper_v1",
    }
    metadata.update({key: value for key, value in row.items() if str(key).startswith("source_")})
    metadata["trajectory_path"] = str(row["trajectory_path"])
    if label == "negative":
        metadata["failure_confidence"] = 1.0
    bank.metadata.append(metadata)
    bank.keys = np.concatenate([bank.keys, key.reshape(1, -1)], axis=0)
    bank.action_parts.append(actions)
    bank.ids.append(action_id)


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _write_bank(bank: _Bank, directory: Path, *, negative: bool) -> dict[str, int]:
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "gpm_negative_memory" if negative else "gpm_memory"
    meta_path = directory / f"{prefix}_meta.pt"
    index_path = directory / f"{prefix}.index"
    actions_path = directory / f"{prefix}_actions.npz"
    temporary_meta = _temporary(meta_path)
    temporary_index = _temporary(index_path)
    temporary_actions = _temporary(actions_path)
    for path in (temporary_meta, temporary_index, temporary_actions):
        path.unlink(missing_ok=True)

    keys = np.ascontiguousarray(bank.keys, dtype=np.float32)
    index = faiss.IndexFlatIP(keys.shape[1])
    index.add(keys)
    faiss.write_index(index, str(temporary_index))
    torch.save(bank.metadata, temporary_meta)
    offsets = np.zeros(len(bank.action_parts) + 1, dtype=np.int64)
    for i, actions in enumerate(bank.action_parts):
        offsets[i + 1] = offsets[i] + actions.shape[0]
    packed_actions = (
        np.concatenate(bank.action_parts, axis=0).astype(np.float32, copy=False)
        if bank.action_parts
        else np.empty((0, 32), dtype=np.float32)
    )
    with temporary_actions.open("wb") as stream:
        np.savez_compressed(stream, actions=packed_actions, offsets=offsets, ids=np.asarray(bank.ids, dtype=np.str_))

    temporary_index.replace(index_path)
    temporary_actions.replace(actions_path)
    temporary_meta.replace(meta_path)
    return {"items": len(bank.metadata), "action_steps": int(packed_actions.shape[0])}


def main(args: Args) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.group):
        raise ValueError("group must contain only letters, digits, underscores, or hyphens")
    if args.admission not in {"success", "failure", "both"}:
        raise ValueError(f"Invalid admission mode: {args.admission!r}")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to write to non-empty output directory: {output}")

    manifest_path = Path(args.manifest).resolve()
    rows = read_jsonl(manifest_path)
    _validate_manifest(rows, manifest_path)
    feature_dir = Path(args.feature_dir).resolve()
    cache_state_path = feature_dir / "cache_state.json"
    if not cache_state_path.is_file():
        raise FileNotFoundError(f"Missing feature cache state: {cache_state_path}")
    cache_state = json.loads(cache_state_path.read_text(encoding="utf-8"))
    expected_digest = manifest_digest(manifest_path)
    if cache_state.get("manifest_sha256") != expected_digest or int(cache_state.get("rows", -1)) != len(rows):
        raise ValueError("Feature cache state does not match the CL manifest")
    features = np.load(feature_dir / "pooled_prefix.npy", mmap_mode="r")
    completed = np.load(feature_dir / "completed.npy", mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != len(rows) or completed.shape != (len(rows),):
        raise ValueError("Feature cache and manifest lengths differ")
    if not bool(np.all(completed)):
        raise ValueError("CL bank construction requires a complete feature cache")
    if not np.isfinite(features).all():
        raise ValueError("Feature cache contains non-finite values")
    new_keys = _project_features(args.checkpoint, features, args.device)

    positive_paths = (
        args.baseline_positive_meta,
        args.baseline_positive_index,
        args.baseline_positive_actions,
    )
    if any(positive_paths) and not all(positive_paths):
        raise ValueError("Either provide all three baseline positive paths or leave all three empty")
    positive = (
        _load_bank(*positive_paths, name="positive")
        if all(positive_paths)
        else _empty_bank(new_keys.shape[1])
    )
    negative_paths = (args.baseline_negative_meta, args.baseline_negative_index, args.baseline_negative_actions)
    if any(negative_paths) and not all(negative_paths):
        raise ValueError("Either provide all three baseline negative paths or leave all three empty")
    negative = (
        _load_bank(*negative_paths, name="negative")
        if all(negative_paths)
        else _empty_bank(new_keys.shape[1])
    )
    if positive.keys.shape[1] != negative.keys.shape[1] or positive.keys.shape[1] != new_keys.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch among positive baseline, negative baseline, and TaskHeadMLP checkpoint"
        )
    baseline_counts = {"positive": len(positive.metadata), "negative": len(negative.metadata)}
    used_ids = set(positive.ids) | set(negative.ids)
    admitted = {"positive": 0, "negative": 0}
    for row, key in zip(rows, new_keys, strict=True):
        selected = args.admission == "both" or (args.admission == "success") == bool(row["success"])
        if not selected:
            continue
        label = "positive" if bool(row["success"]) else "negative"
        actions = _load_raw_actions(row, manifest_path)
        action_id = _unique_action_id(row, label, used_ids)
        _append(positive if label == "positive" else negative, row, key, actions, action_id)
        admitted[label] += 1

    output.mkdir(parents=True, exist_ok=True)
    positive_stats = _write_bank(positive, output / "positive", negative=False)
    negative_stats = _write_bank(negative, output / "negative", negative=True)
    snapshot = output / "manifest.jsonl"
    temporary_snapshot = _temporary(snapshot)
    temporary_snapshot.write_bytes(manifest_path.read_bytes())
    temporary_snapshot.replace(snapshot)
    summary = {
        "schema_version": 1,
        "group": args.group,
        "admission": args.admission,
        "manifest": str(manifest_path),
        "manifest_sha256": expected_digest,
        "manifest_rows": len(rows),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "embedding_dim": int(new_keys.shape[1]),
        "action_dim": 32,
        "chunk_len": 10,
        "stride": 10,
        "baseline_items": baseline_counts,
        "admitted_items": admitted,
        "output": {"positive": positive_stats, "negative": negative_stats},
    }
    write_json_atomic(output / "build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
