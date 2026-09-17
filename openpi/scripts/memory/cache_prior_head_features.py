from __future__ import annotations

import dataclasses
import errno
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
from typing import Any
from concurrent.futures import Future, ThreadPoolExecutor

import jax
import numpy as np
import torch
import tyro

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.task_head.cl_features import load_cl_observations
from openpi.task_head.reproduction import manifest_digest, read_jsonl, write_json_atomic
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    manifest: str = "artifacts/prior_head_reproduction/manifest.jsonl"
    output_dir: str = "artifacts/prior_head_reproduction/features"
    policy_dir: str = "/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
    config_name: str = "pi05_libero"
    device: str = "cuda"
    batch_size: int = 8
    inference_batch_size: int = 0
    feature_dim: int = 2048
    max_items: int = 0
    prompt_overrides: str = ""
    require_cuda: bool = True
    overwrite: bool = False
    auto_batch: bool = False
    min_batch_size: int = 1
    seed_manifest: str = ""
    seed_cache_dir: str = ""
    seed_only: bool = False
    progress_every_batches: int = 5
    loader_workers: int = 0
    prefetch_batches: int = 0
    trajectory_batches: bool = False


def _safe_memmap_flush(array: np.memmap) -> None:
    try:
        array.flush()
    except OSError as exc:
        if exc.errno != errno.EINVAL:
            raise


def _is_cuda_capacity_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cudacachingallocator",
            "nvml_success == r internal assert failed",
        )
    )


def _pwrite_all(fd: int, payload: memoryview, offset: int) -> None:
    written = 0
    while written < len(payload):
        count = os.pwrite(fd, payload[written:], offset + written)
        if count <= 0:
            raise OSError("pwrite returned no progress")
        written += count


def _commit_npy_rows(
    feature_fd: int,
    completed_fd: int,
    features: np.memmap,
    completed: np.memmap,
    indices: np.ndarray,
    batch: np.ndarray,
) -> None:
    batch = np.ascontiguousarray(batch, dtype=features.dtype)
    row_bytes = int(np.prod(features.shape[1:])) * features.dtype.itemsize
    for index, row in zip(indices, batch, strict=True):
        _pwrite_all(
            feature_fd,
            memoryview(np.ascontiguousarray(row)).cast("B"),
            int(features.offset) + int(index) * row_bytes,
        )
    os.fsync(feature_fd)
    marker = memoryview(np.asarray(True, dtype=completed.dtype)).cast("B")
    for index in indices:
        _pwrite_all(
            completed_fd,
            marker,
            int(completed.offset) + int(index) * completed.dtype.itemsize,
        )
    os.fsync(completed_fd)
    if not bool(np.asarray(completed[indices]).all()):
        raise RuntimeError("Committed feature rows are not visible through the cache mapping")


@torch.inference_mode()
def _extract_loaded_batch(policy, observations: list[dict[str, Any]], device: str) -> np.ndarray:
    transformed = [
        policy._input_transform(observation)  # noqa: SLF001
        for observation in observations
    ]
    inputs = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *transformed)
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device), inputs)
    observation = _model.Observation.from_dict(inputs)
    model = policy._model  # noqa: SLF001
    images, image_masks, language_tokens, language_masks, _ = model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, image_masks, language_tokens, language_masks
    )
    if hasattr(model, "make_att_2d_masks"):
        prefix_att_2d_masks = model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    else:
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    attention_mask = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    outputs_embeds, _ = model.paligemma_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        # Match the released OptimusVLA inference path exactly. The returned KV cache is discarded.
        use_cache=True,
    )
    tokens = outputs_embeds[0] if isinstance(outputs_embeds, (list, tuple)) else outputs_embeds
    if tokens.ndim != 3:
        raise RuntimeError(f"Expected prefix output [B,L,D], got {tuple(tokens.shape)}")
    return tokens.mean(dim=1).float().cpu().numpy()


def _extract_observations(
    policy,
    observations: list[dict[str, Any]],
    device: str,
    *,
    inference_batch_size: int = 0,
    auto_batch: bool = False,
    min_batch_size: int = 1,
) -> np.ndarray:
    active = min(inference_batch_size or len(observations), len(observations))
    while True:
        try:
            chunks = [
                _extract_loaded_batch(policy, observations[start : start + active], device)
                for start in range(0, len(observations), active)
            ]
            return np.concatenate(chunks, axis=0)
        except Exception as exc:
            if not auto_batch or not _is_cuda_capacity_error(exc) or active <= min_batch_size:
                raise
            next_batch_size = max(min_batch_size, active // 2)
            print(
                f"CUDA OOM at inference_batch={active}; retrying inference_batch={next_batch_size}",
                flush=True,
            )
            active = next_batch_size
            gc.collect()
            torch.cuda.empty_cache()


def _extract_batch(
    policy,
    rows: list[dict[str, Any]],
    device: str,
    *,
    loader_workers: int = 0,
    inference_batch_size: int = 0,
    auto_batch: bool = False,
    min_batch_size: int = 1,
) -> np.ndarray:
    observations = load_cl_observations(rows, loader_workers=loader_workers)
    return _extract_observations(
        policy,
        observations,
        device,
        inference_batch_size=inference_batch_size,
        auto_batch=auto_batch,
        min_batch_size=min_batch_size,
    )


@dataclasses.dataclass(frozen=True)
class _PreparedFeatureBatch:
    indices: np.ndarray
    source_rows: list[dict[str, Any]]
    inverse: np.ndarray
    observations: list[dict[str, Any]]
    requested_frames: int


def _pending_chunks(
    pending: np.ndarray,
    rows: list[dict[str, Any]],
    max_batch_size: int,
    trajectory_batches: bool,
) -> list[np.ndarray]:
    if not trajectory_batches:
        return [pending[start : start + max_batch_size] for start in range(0, len(pending), max_batch_size)]
    chunks: list[np.ndarray] = []
    start = 0
    while start < len(pending):
        action_id = str(rows[int(pending[start])].get("action_id", ""))
        stop = start + 1
        while (
            stop < len(pending)
            and stop - start < max_batch_size
            and str(rows[int(pending[stop])].get("action_id", "")) == action_id
        ):
            stop += 1
        chunks.append(pending[start:stop])
        start = stop
    return chunks


def _prepare_feature_batch(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    temporal_window: int,
    loader_workers: int,
) -> _PreparedFeatureBatch:
    source_rows = [rows[int(index)] for index in indices]
    flattened_rows = [
        _temporal_observation_row(row, position)
        for position in range(temporal_window)
        for row in source_rows
    ]
    unique_rows, inverse = _deduplicate_observation_rows(flattened_rows)
    observations = load_cl_observations(unique_rows, loader_workers=loader_workers)
    return _PreparedFeatureBatch(
        indices=indices,
        source_rows=source_rows,
        inverse=inverse,
        observations=observations,
        requested_frames=len(flattened_rows),
    )


def _prefetched_batches(
    rows: list[dict[str, Any]],
    chunks: list[np.ndarray],
    temporal_window: int,
    loader_workers: int,
    prefetch_batches: int,
):
    if prefetch_batches <= 1:
        for indices in chunks:
            yield _prepare_feature_batch(rows, indices, temporal_window, loader_workers)
        return
    with ThreadPoolExecutor(max_workers=prefetch_batches, thread_name_prefix="feature-prefetch") as executor:
        futures: dict[int, Future[_PreparedFeatureBatch]] = {}
        next_submit = 0
        while next_submit < min(prefetch_batches, len(chunks)):
            futures[next_submit] = executor.submit(
                _prepare_feature_batch,
                rows,
                chunks[next_submit],
                temporal_window,
                loader_workers,
            )
            next_submit += 1
        for position in range(len(chunks)):
            prepared = futures.pop(position).result()
            if next_submit < len(chunks):
                futures[next_submit] = executor.submit(
                    _prepare_feature_batch,
                    rows,
                    chunks[next_submit],
                    temporal_window,
                    loader_workers,
                )
                next_submit += 1
            yield prepared


def _temporal_layout(rows: list[dict[str, Any]]) -> tuple[tuple[int, ...], int]:
    first = rows[0].get("temporal_context")
    if not first:
        return (0,), 1
    offsets = tuple(int(item["offset"]) for item in first)
    if not offsets or offsets[-1] != 0:
        raise ValueError("temporal_context offsets must end at 0")
    for index, row in enumerate(rows):
        context = row.get("temporal_context")
        row_offsets = tuple(int(item["offset"]) for item in context or ())
        if row_offsets != offsets:
            raise RuntimeError(
                f"Manifest temporal context differs at row {index}: {row_offsets} != {offsets}"
            )
    return offsets, len(offsets)


def _temporal_observation_row(row: dict[str, Any], position: int) -> dict[str, Any]:
    context = row.get("temporal_context")
    if not context:
        if position != 0:
            raise IndexError(position)
        return row
    item = context[position]
    result = {
        **row,
        "frame_index": int(item["frame_index"]),
        "trajectory_path": str(item["trajectory_path"]),
    }
    if "dataset_index" in item:
        result["dataset_index"] = int(item["dataset_index"])
    return result


def _deduplicate_observation_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    unique: list[dict[str, Any]] = []
    index_by_identity: dict[tuple[str, str, str, int, str], int] = {}
    inverse = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        source_format = str(row.get("source_format", "official_hdf5"))
        path = str(row.get("trajectory_path", row.get("task_file", "")))
        identity = (
            source_format,
            path,
            str(row.get("demo", "demo_0")),
            int(row["frame_index"]),
            str(row["prompt"]),
        )
        unique_index = index_by_identity.get(identity)
        if unique_index is None:
            unique_index = len(unique)
            index_by_identity[identity] = unique_index
            unique.append(row)
        inverse[index] = unique_index
    return unique, inverse


def _seed_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    context = tuple(
        (
            int(item["offset"]),
            int(item["global_frame_index"]),
            int(item["stage_index"]),
            int(item["frame_index"]),
            str(Path(item["trajectory_path"]).resolve()),
        )
        for item in row.get("temporal_context", ())
    )
    return (
        str(row["action_id"]),
        int(row["global_frame_index"]),
        int(row["task_id"]),
        str(row["prompt"]),
        context,
    )


def _seed_existing_features(
    *,
    args: Args,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    completed: np.ndarray,
    state_path: Path,
    output_feature_dim: int,
    temporal_offsets: tuple[int, ...],
) -> int:
    if not args.seed_manifest and not args.seed_cache_dir:
        return 0
    if not args.seed_manifest or not args.seed_cache_dir:
        raise ValueError("seed_manifest and seed_cache_dir must be provided together")

    seed_manifest = Path(args.seed_manifest).resolve()
    seed_cache_dir = Path(args.seed_cache_dir).resolve()
    seed_rows = read_jsonl(seed_manifest)
    seed_state_path = seed_cache_dir / "cache_state.json"
    seed_feature_path = seed_cache_dir / "pooled_prefix.npy"
    seed_completed_path = seed_cache_dir / "completed.npy"
    for path in (seed_manifest, seed_state_path, seed_feature_path, seed_completed_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed cache input: {path}")

    seed_state = json.loads(seed_state_path.read_text(encoding="utf-8"))
    expected = {
        "manifest_sha256": manifest_digest(seed_manifest),
        "rows": len(seed_rows),
        "feature_dim": output_feature_dim,
        "base_feature_dim": int(args.feature_dim),
        "temporal_offsets": list(temporal_offsets),
        "policy_dir": str(Path(args.policy_dir).resolve()),
        "config_name": args.config_name,
    }
    for key, value in expected.items():
        if seed_state.get(key) != value:
            raise RuntimeError(
                f"Seed cache {key} mismatch: {seed_state.get(key)!r} != {value!r}"
            )

    seed_features = np.load(seed_feature_path, mmap_mode="r")
    seed_completed = np.load(seed_completed_path, mmap_mode="r")
    if seed_features.shape != (len(seed_rows), output_feature_dim):
        raise RuntimeError(f"Unexpected seed feature shape: {seed_features.shape}")
    if len(seed_completed) != len(seed_rows) or not bool(seed_completed.all()):
        raise RuntimeError("Seed feature cache is incomplete")

    target_by_key: dict[tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        key = (str(row["action_id"]), int(row["global_frame_index"]))
        if key in target_by_key:
            raise RuntimeError(f"Duplicate target seed key: {key}")
        target_by_key[key] = index

    source_indices: list[int] = []
    target_indices: list[int] = []
    compatible_rows = 0
    for source_index, row in enumerate(seed_rows):
        key = (str(row["action_id"]), int(row["global_frame_index"]))
        target_index = target_by_key.get(key)
        if target_index is None:
            continue
        if _seed_signature(row) != _seed_signature(rows[target_index]):
            raise RuntimeError(f"Seed observation context differs for {key}")
        compatible_rows += 1
        if bool(completed[target_index]):
            continue
        source_indices.append(source_index)
        target_indices.append(target_index)

    for start in range(0, len(source_indices), 2048):
        source_chunk = np.asarray(source_indices[start : start + 2048], dtype=np.int64)
        target_chunk = np.asarray(target_indices[start : start + 2048], dtype=np.int64)
        batch = np.asarray(seed_features[source_chunk])
        if not np.isfinite(batch).all():
            raise FloatingPointError("Seed cache contains non-finite features")
        features[target_chunk] = batch
        completed[target_chunk] = True
    _safe_memmap_flush(features)
    _safe_memmap_flush(completed)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["seed_cache"] = {
        "manifest": str(seed_manifest),
        "manifest_sha256": expected["manifest_sha256"],
        "cache_dir": str(seed_cache_dir),
        "compatible_rows": compatible_rows,
        "newly_reused_rows": len(source_indices),
    }
    write_json_atomic(state_path, state)
    print(
        f"Seeded compatible feature rows: compatible={compatible_rows} "
        f"newly_reused={len(source_indices)} "
        f"completed={int(completed.sum())}/{len(rows)}",
        flush=True,
    )
    return len(source_indices)


def main(args: Args) -> None:
    if args.batch_size <= 0 or args.feature_dim <= 0 or args.min_batch_size <= 0:
        raise ValueError("batch_size, feature_dim, and min_batch_size must be positive")
    if args.inference_batch_size < 0:
        raise ValueError("inference_batch_size cannot be negative")
    if args.loader_workers < 0:
        raise ValueError("loader_workers cannot be negative")
    if args.prefetch_batches < 0:
        raise ValueError("prefetch_batches cannot be negative")
    if args.min_batch_size > args.batch_size:
        raise ValueError("min_batch_size cannot exceed batch_size")
    if args.progress_every_batches <= 0:
        raise ValueError("progress_every_batches must be positive")
    if args.require_cuda and (not torch.cuda.is_available() or not args.device.startswith("cuda")):
        raise RuntimeError("CUDA is required; pass --no-require-cuda only for a deliberate CPU run")

    manifest_path = Path(args.manifest).resolve()
    rows = read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    digest = manifest_digest(manifest_path)
    temporal_offsets, temporal_window = _temporal_layout(rows)
    output_feature_dim = int(args.feature_dim) * temporal_window
    extraction_protocol = (
        "cl_prefix_temporal_cat_v1" if temporal_window > 1 else "cl_prefix_v1"
    )
    prompt_override_path = Path(args.prompt_overrides).resolve() if args.prompt_overrides else None
    prompt_override_digest = ""
    if prompt_override_path is not None:
        overrides: dict[int, str] = {}
        with prompt_override_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                index = int(record["row_index"])
                if index in overrides:
                    raise RuntimeError(
                        f"Duplicate prompt override row {index} in {prompt_override_path}:{line_number}"
                    )
                value = str(record["subtask"]).strip()
                if not value:
                    raise RuntimeError(f"Empty prompt override row {index}")
                overrides[index] = value
        missing = sorted(set(range(len(rows))) - overrides.keys())
        extra = sorted(overrides.keys() - set(range(len(rows))))
        if missing or extra:
            raise RuntimeError(
                f"Prompt overrides do not match manifest rows: missing={missing[:8]} extra={extra[:8]}"
            )
        rows = [{**row, "prompt": overrides[index]} for index, row in enumerate(rows)]
        prompt_override_digest = hashlib.sha256(prompt_override_path.read_bytes()).hexdigest()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "pooled_prefix.npy"
    completed_path = output / "completed.npy"
    state_path = output / "cache_state.json"

    if args.overwrite:
        feature_path.unlink(missing_ok=True)
        completed_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["manifest_sha256"] != digest or int(state["rows"]) != len(rows):
            raise RuntimeError("Existing feature cache does not match this manifest")
        if int(state["feature_dim"]) != output_feature_dim:
            raise RuntimeError("Existing feature cache has a different feature_dim")
        if int(state.get("base_feature_dim", args.feature_dim)) != args.feature_dim:
            raise RuntimeError("Existing feature cache has a different base_feature_dim")
        if tuple(state.get("temporal_offsets", [0])) != temporal_offsets:
            raise RuntimeError("Existing feature cache uses different temporal offsets")
        expected_policy_dir = str(Path(args.policy_dir).resolve())
        if state.get("policy_dir") != expected_policy_dir or state.get("config_name") != args.config_name:
            raise RuntimeError("Existing feature cache uses a different Pi0.5 checkpoint or config")
        if state.get("prompt_overrides_sha256", "") != prompt_override_digest:
            raise RuntimeError("Existing feature cache uses different prompt overrides")
        if state.get("extraction_protocol") not in (None, extraction_protocol):
            raise RuntimeError("Existing feature cache uses a different extraction protocol")
    else:
        print(
            f"Initializing resumable feature cache: rows={len(rows)} "
            f"feature_dim={output_feature_dim} path={feature_path}",
            flush=True,
        )
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=(len(rows), output_feature_dim)
        )
        completed = np.lib.format.open_memmap(completed_path, mode="w+", dtype=np.bool_, shape=(len(rows),))
        completed[:] = False
        _safe_memmap_flush(completed)
        write_json_atomic(
            state_path,
            {
                "manifest": str(manifest_path),
                "manifest_sha256": digest,
                "rows": len(rows),
                "feature_dim": output_feature_dim,
                "base_feature_dim": int(args.feature_dim),
                "temporal_window": temporal_window,
                "temporal_offsets": list(temporal_offsets),
                "policy_dir": str(Path(args.policy_dir).resolve()),
                "config_name": args.config_name,
                "pooling": "unmasked_mean_post_transformer_prefix",
                "extraction_protocol": extraction_protocol,
                "conditioning_protocol": (
                    "upper_generated_subtask_v1" if prompt_override_path is not None else "manifest_prompt_v1"
                ),
                "prompt_overrides": str(prompt_override_path) if prompt_override_path is not None else "",
                "prompt_overrides_sha256": prompt_override_digest,
            },
        )
        print("Resumable feature cache initialized; completion bitmap and state are durable", flush=True)

    features = np.load(feature_path, mmap_mode="r+")
    completed = np.load(completed_path, mmap_mode="r+")
    _seed_existing_features(
        args=args,
        rows=rows,
        features=features,
        completed=completed,
        state_path=state_path,
        output_feature_dim=output_feature_dim,
        temporal_offsets=temporal_offsets,
    )
    pending = np.flatnonzero(~completed)
    if args.seed_only:
        print(
            f"Seed-only feature cache pass complete: "
            f"{int(completed.sum())}/{len(rows)} rows available",
            flush=True,
        )
        return
    if args.max_items > 0:
        pending = pending[: int(args.max_items)]
    if pending.size == 0:
        print(f"Feature cache already complete: {int(completed.sum())}/{len(rows)}")
        return

    feature_fd = os.open(feature_path, os.O_RDWR)
    completed_fd = os.open(completed_path, os.O_RDWR)

    # Prefix extraction never calls sample_actions; compiling it only adds a large one-time delay.
    os.environ.setdefault("OPENPI_TORCH_COMPILE", "0")
    print(
        f"Loading policy for feature cache: completed={int(completed.sum())}/{len(rows)} "
        f"pending={len(pending)} device={args.device}",
        flush=True,
    )
    load_started = time.monotonic()
    load_finished = threading.Event()

    def report_load_progress() -> None:
        while not load_finished.wait(timeout=30.0):
            print(
                f"Policy load still active: elapsed_seconds={time.monotonic() - load_started:.0f}",
                flush=True,
            )

    reporter = threading.Thread(target=report_load_progress, name="policy-load-reporter", daemon=True)
    reporter.start()
    try:
        policy = policy_config.create_trained_policy(
            training_config.get_config(args.config_name),
            args.policy_dir,
            pytorch_device=args.device,
        )
    finally:
        load_finished.set()
        reporter.join(timeout=1.0)
    print("Policy loaded; starting resumable feature extraction", flush=True)
    processed = 0
    batches_processed = 0
    run_started = time.perf_counter()
    requested_frames = 0
    unique_frames = 0
    chunks = _pending_chunks(
        pending,
        rows,
        int(args.batch_size),
        args.trajectory_batches,
    )
    print(
        f"Feature pipeline: chunks={len(chunks)} trajectory_batches={args.trajectory_batches} "
        f"prefetch_batches={args.prefetch_batches} inference_batch={args.inference_batch_size}",
        flush=True,
    )
    for prepared in _prefetched_batches(
        rows,
        chunks,
        temporal_window,
        args.loader_workers,
        args.prefetch_batches,
    ):
        indices = prepared.indices
        try:
            unique_features = _extract_observations(
                policy,
                prepared.observations,
                args.device,
                inference_batch_size=args.inference_batch_size,
                auto_batch=args.auto_batch,
                min_batch_size=args.min_batch_size,
            )
            flattened = unique_features[prepared.inverse]
            temporal_parts = np.split(flattened, temporal_window, axis=0)
        except Exception as exc:
            action_ids = sorted(
                {str(row.get("action_id", "")) for row in prepared.source_rows}
            )
            raise RuntimeError(
                f"Feature extraction failed for rows {int(indices[0])}..{int(indices[-1])}; "
                f"action_ids={action_ids}"
            ) from exc
        batch = np.concatenate(temporal_parts, axis=1)
        if batch.shape != (len(indices), output_feature_dim):
            raise RuntimeError(f"Unexpected pooled feature shape: {batch.shape}")
        if not np.isfinite(batch).all():
            raise FloatingPointError("Non-finite pooled prefix feature")
        _commit_npy_rows(
            feature_fd,
            completed_fd,
            features,
            completed,
            indices,
            batch,
        )
        processed += len(indices)
        requested_frames += prepared.requested_frames
        unique_frames += len(prepared.observations)
        batches_processed += 1
        if batches_processed % args.progress_every_batches == 0 or processed == len(pending):
            elapsed = max(time.perf_counter() - run_started, 1e-6)
            rows_per_second = processed / elapsed
            eta_seconds = (len(pending) - processed) / max(rows_per_second, 1e-6)
            max_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            if args.device.startswith("cuda"):
                gpu_allocated_gib = torch.cuda.memory_allocated() / 2**30
                gpu_reserved_gib = torch.cuda.memory_reserved() / 2**30
                gpu_free_gib = torch.cuda.mem_get_info()[0] / 2**30
            else:
                gpu_allocated_gib = gpu_reserved_gib = gpu_free_gib = 0.0
            print(
                f"cached={int(completed.sum())}/{len(rows)} "
                f"current_run={processed}/{len(pending)} "
                f"anchor_batch={len(indices)} "
                f"inference_batch={args.inference_batch_size or len(indices)} "
                f"loader_workers={args.loader_workers} "
                f"prefetch_batches={args.prefetch_batches} "
                f"requested_frame_batch={len(indices) * temporal_window} "
                f"unique_frame_batch={len(prepared.observations)} "
                f"frame_reuse={requested_frames / max(unique_frames, 1):.2f}x "
                f"rows_per_second={rows_per_second:.2f} "
                f"eta_minutes={eta_seconds / 60:.1f} "
                f"host_max_rss_gib={max_rss_gib:.1f} "
                f"gpu_allocated_gib={gpu_allocated_gib:.1f} "
                f"gpu_reserved_gib={gpu_reserved_gib:.1f} "
                f"gpu_free_gib={gpu_free_gib:.1f}",
                flush=True,
            )
    os.close(feature_fd)
    os.close(completed_fd)
    print(f"Feature extraction finished: {int(completed.sum())}/{len(rows)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
