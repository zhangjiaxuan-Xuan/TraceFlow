from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import errno
from io import BytesIO
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


PROTOCOL = "predimem_keyframes_recent5_dual_camera_causal_v2"
JOINT_PROTOCOL = "predimem_trajectory_ordered_runtime_generate_subtask_hidden_v4"
SYSTEM_PROMPT = """You are an embodied-memory robot VLM planner.

You will observe two kinds of visual evidence from the same long-horizon execution:
1. Historical keyframes: moments before the current step, used to remember important past states.
2. A recent 5-frame dual-camera window ending at the current frame, used to infer the current primitive.
Temporal order: historical keyframes are ordered from earliest to latest; the recent 5-frame window is also ordered from earliest to latest, and the last timestep in that window is the current frame."""


def _safe_memmap_flush(array: np.memmap) -> None:
    try:
        array.flush()
    except OSError as exc:
        # Some AOSS/FUSE mounts accept ordinary writes and fsync but reject
        # mmap(MS_SYNC). Runtime cache commits use pwrite below, so EINVAL is
        # safe to ignore here during initial file creation only.
        if exc.errno != errno.EINVAL:
            raise


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
    feature_offset = int(features.offset)
    for index, row in zip(indices, batch, strict=True):
        _pwrite_all(
            feature_fd,
            memoryview(np.ascontiguousarray(row)).cast("B"),
            feature_offset + int(index) * row_bytes,
        )
    # A completed marker is only durable after all feature rows are durable.
    os.fsync(feature_fd)
    marker = memoryview(np.asarray(True, dtype=completed.dtype)).cast("B")
    completed_offset = int(completed.offset)
    for index in indices:
        _pwrite_all(
            completed_fd,
            marker,
            completed_offset + int(index) * completed.dtype.itemsize,
        )
    os.fsync(completed_fd)
    if not bool(np.asarray(completed[indices]).all()):
        raise RuntimeError("Committed upper rows are not visible through the cache mapping")


def _load_context(
    row: dict,
    n_recent: int,
    historical_frames: list[int] | None = None,
) -> tuple[list[bytes], list[bytes], list[bytes], list[bytes]]:
    global_frame = int(row["global_frame_index"])
    lengths = [int(value) for value in row["segment_lengths"]]
    paths = list(row["segment_paths"])
    offsets = np.cumsum([0, *lengths])
    start = max(0, global_frame - n_recent + 1)
    memory_main: list[bytes] = []
    memory_wrist: list[bytes] = []
    context_main: list[bytes] = []
    context_wrist: list[bytes] = []
    opened: dict[int, h5py.File] = {}
    try:
        # The legacy cache uses completed stage endpoints. Joint conditioning
        # supplies planner-selected frames replayed in trajectory order.
        historical = (
            [int(offsets[index + 1] - 1) for index in range(int(row["stage_index"]))]
            if historical_frames is None
            else [int(value) for value in historical_frames]
        )
        for frame, targets in [
            *((frame, (memory_main, memory_wrist)) for frame in historical),
            *((frame, (context_main, context_wrist)) for frame in range(start, global_frame + 1)),
        ]:
            stage = int(np.searchsorted(offsets[1:], frame, side="right"))
            local = int(frame - offsets[stage])
            if stage not in opened:
                opened[stage] = h5py.File(paths[stage], "r")
            obs = opened[stage]["data/demo_0/obs"]
            for source, target in (
                (obs["agentview_rgb"][local], targets[0]),
                (obs["eye_in_hand_rgb"][local], targets[1]),
            ):
                image = Image.fromarray(np.asarray(source, dtype=np.uint8))
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                target.append(buffer.getvalue())
    finally:
        for data in opened.values():
            data.close()
    return memory_main, memory_wrist, context_main, context_wrist


def _messages(
    row: dict,
    memory_main: list[Image.Image],
    memory_wrist: list[Image.Image],
    context_main: list[Image.Image],
    context_wrist: list[Image.Image],
) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Global objective: infer the robot's current primitive action from historical keyframes before the current step and recent visual history within the same execution.\n\n"
                f"Task objective:\n{row['task_block']}\n\n"
                f"Scene description:\n{row['scene_description'] or row['prompt']}\n\n"
                "At every timestep images are ordered as main camera, then wrist camera.\n"
                "Current observation:"
            ),
        }
    ]
    if memory_main:
        content.append(
            {
                "type": "text",
                "text": (
                    "Historical keyframes from moments before the current step in the same execution "
                    f"({len(memory_main)} timesteps, {2 * len(memory_main)} images):"
                ),
            }
        )
        for main_image, wrist_image in zip(memory_main, memory_wrist, strict=True):
            content.extend(({"type": "image", "image": main_image}, {"type": "image", "image": wrist_image}))
    content.append(
        {
            "type": "text",
            "text": (
                f"Recent visual context: {len(context_main)} consecutive frames ending at the current frame "
                f"({2 * len(context_main)} images):"
            ),
        }
    )
    for main_image, wrist_image in zip(context_main, context_wrist, strict=True):
        content.extend(({"type": "image", "image": main_image}, {"type": "image", "image": wrist_image}))
    content.append(
        {
            "type": "text",
            "text": (
                "Output strict JSON with exactly two fields: current_primitive and keyframe_positions. "
                "keyframe_positions are 1-indexed keyframe positions inside the recent visual window."
            ),
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def _parse_output(text: str, max_pos: int) -> tuple[str, list[int]]:
    value = text.strip()
    if "</think>" in value:
        value = value[value.rfind("</think>") + len("</think>") :].strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    primitive = ""
    positions: list[int] = []
    try:
        parsed = json.loads(value)
        primitive = str(parsed.get("current_primitive", parsed.get("current_subtask", ""))).strip()
        for item in parsed.get("keyframe_positions", []):
            try:
                position = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= position <= max_pos:
                positions.append(position)
    except Exception:
        primitive = value
    return primitive, positions


def _build_visual_memory(
    history: list[list[int]],
    step: int,
    recent_count: int,
    merge_distance: int,
) -> list[int]:
    candidates = sorted(index for group in history for index in group)
    if not candidates:
        return []
    clusters: list[list[int]] = [[candidates[0]]]
    for index in candidates[1:]:
        if index - clusters[-1][-1] <= merge_distance:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    selected = [cluster[len(cluster) // 2] for cluster in clusters]
    cutoff = step - recent_count + 1
    return [index for index in selected if index <= cutoff]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_subtask_records(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["row_index"])
            # A crash can occur after fsyncing a record but before marking the
            # feature row complete. Keep the latest record and compact on finish.
            records[index] = record
    return records


def _write_subtask_records(path: Path, records: dict[int, dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for index in sorted(records):
            stream.write(json.dumps(records[index], ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--n-recent", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--merge-distance", type=int, default=6)
    parser.add_argument("--keyframe-max", type=int, default=0)
    parser.add_argument("--generate-subtasks", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("Upper feature extraction requires CUDA")

    rows = _read_rows(args.manifest)
    if args.generate_subtasks and args.task_config is None:
        raise ValueError("--task-config is required with --generate-subtasks")
    protocol = JOINT_PROTOCOL if args.generate_subtasks else PROTOCOL
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "upper_features.npy"
    completed_path = args.output_dir / "upper_completed.npy"
    state_path = args.output_dir / "upper_cache_state.json"
    subtasks_path = args.output_dir / "subtasks.jsonl"
    if args.overwrite:
        feature_path.unlink(missing_ok=True)
        completed_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        subtasks_path.unlink(missing_ok=True)
    if not feature_path.exists():
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float16, shape=(len(rows), args.feature_dim)
        )
        features[:] = np.nan
        _safe_memmap_flush(features)
        completed = np.lib.format.open_memmap(completed_path, mode="w+", dtype=np.bool_, shape=(len(rows),))
        completed[:] = False
        _safe_memmap_flush(completed)
        state_path.write_text(
            json.dumps(
                {
                    "rows": len(rows),
                    "feature_dim": args.feature_dim,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": _sha256(args.manifest),
                    "task_config": str(args.task_config.resolve()) if args.task_config is not None else "",
                    "task_config_sha256": _sha256(args.task_config) if args.task_config is not None else "",
                    "protocol": protocol,
                    "generates_subtasks": bool(args.generate_subtasks),
                    "n_recent": int(args.n_recent),
                    "merge_distance": int(args.merge_distance),
                    "keyframe_max": int(args.keyframe_max),
                    "ordering": "strict_within_trajectory_frontier_batch_across_trajectories",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        int(state["rows"]) != len(rows)
        or int(state["feature_dim"]) != args.feature_dim
        or state.get("protocol") != protocol
        or state.get("checkpoint") != str(args.checkpoint.resolve())
        or state.get("manifest_sha256") != _sha256(args.manifest)
        or (
            args.generate_subtasks
            and (
                state.get("task_config") != str(args.task_config.resolve())
                or state.get("task_config_sha256") != _sha256(args.task_config)
            )
        )
        or int(state.get("n_recent", args.n_recent)) != args.n_recent
        or (
            args.generate_subtasks
            and (
                int(state.get("merge_distance", -1)) != args.merge_distance
                or int(state.get("keyframe_max", -1)) != args.keyframe_max
            )
        )
    ):
        raise RuntimeError("Existing upper feature cache is incompatible with the manifest")
    features = np.load(feature_path, mmap_mode="r+")
    completed = np.load(completed_path, mmap_mode="r+")
    subtask_records = _read_subtask_records(subtasks_path) if args.generate_subtasks else {}
    if args.generate_subtasks:
        completed_without_subtask = [int(index) for index in np.flatnonzero(completed) if int(index) not in subtask_records]
        if completed_without_subtask:
            raise RuntimeError(
                f"Upper cache marks rows complete without subtask records: {completed_without_subtask[:8]}"
            )
        stale_records = {index for index in subtask_records if not bool(completed[index])}
        if stale_records:
            subtask_records = {
                index: record for index, record in subtask_records.items() if index not in stale_records
            }
            _write_subtask_records(subtasks_path, subtask_records)

    def finalize_joint_state(records: dict[int, dict]) -> None:
        if len(records) != len(rows) or not bool(completed.all()):
            raise RuntimeError(
                f"Joint upper cache is incomplete: features={int(completed.sum())} subtasks={len(records)}"
            )
        _write_subtask_records(subtasks_path, records)
        state["subtask_records"] = str(subtasks_path.resolve())
        state["subtask_records_sha256"] = _sha256(subtasks_path)
        state["subtask_rows"] = len(records)
        state["fallback_rows"] = sum(
            bool(record["fallback_to_previous_subtask"]) for record in records.values()
        )
        state["complete"] = True
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    pending = np.flatnonzero(~completed)
    if args.max_items and not args.generate_subtasks:
        pending = pending[: args.max_items]
    if not len(pending):
        if args.generate_subtasks:
            finalize_joint_state(subtask_records)
        print(f"Upper feature cache complete: {int(completed.sum())}/{len(rows)}")
        return

    feature_fd = os.open(feature_path, os.O_RDWR)
    completed_fd = os.open(completed_path, os.O_RDWR)

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    benchmark_started = time.perf_counter()
    benchmark_start_count = int(completed.sum())
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if args.generate_subtasks:
        from optimus_eval.predimem_upper_server import (
            BatchedUpperPlanner,
            _build_visual_memory as runtime_build_visual_memory,
            _load_tasks,
            _parse_output as runtime_parse_output,
        )

        runtime_tasks = _load_tasks(args.task_config)

        def build_messages(
            row: dict,
            memory_main: list[Image.Image],
            memory_wrist: list[Image.Image],
            context_main: list[Image.Image],
            context_wrist: list[Image.Image],
        ) -> list[dict]:
            return BatchedUpperPlanner._messages(
                runtime_tasks[int(row["task_id"])],
                memory_main,
                memory_wrist,
                context_main,
                context_wrist,
            )

        parse_output = runtime_parse_output
        build_visual_memory = runtime_build_visual_memory
    else:
        build_messages = _messages
        parse_output = _parse_output
        build_visual_memory = _build_visual_memory
    completed_at_start = int(completed.sum())
    next_progress_report = (
        (completed_at_start // args.progress_every) + 1
    ) * args.progress_every

    def run_batch(
        indices: np.ndarray,
        contexts: list[tuple[list[bytes], list[bytes], list[bytes], list[bytes]]],
        trajectory_states: list[dict] | None = None,
    ) -> None:
        messages = []
        flat_images = []
        for index, (memory_main_raw, memory_wrist_raw, context_main_raw, context_wrist_raw) in zip(
            indices, contexts, strict=True
        ):
            memory_main = [Image.open(BytesIO(value)).convert("RGB") for value in memory_main_raw]
            memory_wrist = [Image.open(BytesIO(value)).convert("RGB") for value in memory_wrist_raw]
            context_main = [Image.open(BytesIO(value)).convert("RGB") for value in context_main_raw]
            context_wrist = [Image.open(BytesIO(value)).convert("RGB") for value in context_wrist_raw]
            sample_messages = build_messages(
                rows[int(index)],
                memory_main,
                memory_wrist,
                context_main,
                context_wrist,
            )
            messages.append(processor.apply_chat_template(sample_messages, tokenize=False, add_generation_prompt=True))
            for main, wrist in [
                *zip(memory_main, memory_wrist, strict=True),
                *zip(context_main, context_wrist, strict=True),
            ]:
                flat_images.extend((main, wrist))
        inputs = processor(text=messages, images=flat_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            if args.generate_subtasks:
                generation = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                if not generation.hidden_states:
                    raise RuntimeError("Upper generation did not return hidden states")
                first_step = generation.hidden_states[0]
                last_layer = first_step[-1] if isinstance(first_step, (tuple, list)) else first_step
                batch = last_layer[:, -1].detach().float().cpu().numpy()
                prompt_width = inputs["input_ids"].shape[1]
                generated = generation.sequences[:, prompt_width:]
                outputs = processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            else:
                outputs_model = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                hidden = outputs_model.hidden_states[-1]
                token_index = inputs["attention_mask"].sum(dim=1) - 1
                batch = hidden[torch.arange(hidden.shape[0], device=hidden.device), token_index]
                batch = batch.float().cpu().numpy()
                outputs = []
        if batch.shape != (len(indices), args.feature_dim) or not np.isfinite(batch).all():
            raise RuntimeError(f"Invalid upper feature batch: {batch.shape}")

        records = []
        if args.generate_subtasks:
            assert trajectory_states is not None
            for index, output, context, trajectory_state in zip(
                indices, outputs, contexts, trajectory_states, strict=True
            ):
                context_count = len(context[2])
                primitive, positions = parse_output(output, context_count)
                fallback = not bool(primitive)
                subtask = primitive or str(trajectory_state["current_subtask"]).strip()
                if not subtask:
                    subtask = str(rows[int(index)]["prompt"]).strip()
                if not subtask:
                    raise RuntimeError(f"Upper generated an empty subtask for row {int(index)}")
                global_frame = int(rows[int(index)]["global_frame_index"])
                recent_start = global_frame - context_count + 1
                absolute_positions = [recent_start + position - 1 for position in positions]
                history = [*trajectory_state["history"], absolute_positions]
                keyframes = build_visual_memory(
                    history,
                    global_frame + 1,
                    context_count,
                    args.merge_distance,
                )
                if args.keyframe_max > 0:
                    keyframes = keyframes[-args.keyframe_max :]
                records.append(
                    {
                        "row_index": int(index),
                        "trajectory_id": str(trajectory_state["trajectory_id"]),
                        "trajectory_sequence_index": int(trajectory_state["cursor"]),
                        "global_frame_index": global_frame,
                        "subtask": subtask,
                        "fallback_to_previous_subtask": fallback,
                        "keyframe_positions": positions,
                        "keyframe_count": len(keyframes),
                        "raw_output": output,
                        "_next_history": history,
                        "_next_keyframes": keyframes,
                    }
                )
            with subtasks_path.open("a", encoding="utf-8") as stream:
                for record in records:
                    persistent = {key: value for key, value in record.items() if not key.startswith("_")}
                    stream.write(json.dumps(persistent, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        _commit_npy_rows(
            feature_fd,
            completed_fd,
            features,
            completed,
            indices,
            batch,
        )
        if args.generate_subtasks:
            assert trajectory_states is not None
            for record, trajectory_state in zip(records, trajectory_states, strict=True):
                trajectory_state["history"] = record["_next_history"]
                trajectory_state["keyframes"] = record["_next_keyframes"]
                trajectory_state["current_subtask"] = record["subtask"]
                trajectory_state["cursor"] += 1
        nonlocal next_progress_report
        completed_count = int(completed.sum())
        if completed_count >= next_progress_report or completed_count == len(rows):
            elapsed = max(time.perf_counter() - benchmark_started, 1e-6)
            processed = completed_count - benchmark_start_count
            if args.device.startswith("cuda"):
                peak_allocated = torch.cuda.max_memory_allocated() / 2**30
                peak_reserved = torch.cuda.max_memory_reserved() / 2**30
                free_gib = torch.cuda.mem_get_info()[0] / 2**30
            else:
                peak_allocated = peak_reserved = free_gib = 0.0
            print(
                f"upper={completed_count}/{len(rows)} "
                f"rows_per_second={processed / elapsed:.3f} "
                f"gpu_peak_allocated_gib={peak_allocated:.2f} "
                f"gpu_peak_reserved_gib={peak_reserved:.2f} "
                f"gpu_free_gib={free_gib:.2f}",
                flush=True,
            )
            while next_progress_report <= completed_count:
                next_progress_report += args.progress_every

    # The model has already initialized CUDA above. The default Linux "fork"
    # context would copy that CUDA process state into every HDF5 loader and can
    # make a long cache run fail after otherwise valid batches. Spawn clean,
    # CPU-only loader processes instead.
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        if not args.generate_subtasks:
            for start in range(0, len(pending), args.batch_size):
                indices = pending[start : start + args.batch_size]
                contexts = list(
                    pool.map(
                        _load_context,
                        (rows[int(index)] for index in indices),
                        [args.n_recent] * len(indices),
                    )
                )
                run_batch(indices, contexts)
        else:
            grouped: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                trajectory_id = str(row["action_id"])
                grouped.setdefault(trajectory_id, []).append(index)
            states = []
            for trajectory_id, indices_list in grouped.items():
                indices_list.sort(key=lambda index: int(rows[index]["global_frame_index"]))
                seen_pending = False
                cursor = 0
                history: list[list[int]] = []
                keyframes: list[int] = []
                current_subtask = runtime_tasks[int(rows[indices_list[0]]["task_id"])].brief_description.strip()
                for sequence_index, index in enumerate(indices_list):
                    if not bool(completed[index]):
                        seen_pending = True
                        continue
                    if seen_pending:
                        raise RuntimeError(
                            f"Completed joint cache row follows a pending row in trajectory {trajectory_id}"
                        )
                    record = subtask_records[index]
                    global_frame = int(rows[index]["global_frame_index"])
                    context_count = min(args.n_recent, global_frame + 1)
                    recent_start = global_frame - context_count + 1
                    absolute_positions = [
                        recent_start + int(position) - 1 for position in record["keyframe_positions"]
                    ]
                    history.append(absolute_positions)
                    keyframes = build_visual_memory(
                        history,
                        global_frame + 1,
                        context_count,
                        args.merge_distance,
                    )
                    if args.keyframe_max > 0:
                        keyframes = keyframes[-args.keyframe_max :]
                    current_subtask = str(record["subtask"])
                    cursor = sequence_index + 1
                if cursor < len(indices_list):
                    states.append(
                        {
                            "trajectory_id": trajectory_id,
                            "indices": indices_list,
                            "cursor": cursor,
                            "history": history,
                            "keyframes": keyframes,
                            "current_subtask": current_subtask,
                        }
                    )

            queue = deque(states)
            processed_this_run = 0
            while queue and (args.max_items <= 0 or processed_this_run < args.max_items):
                count = min(args.batch_size, len(queue))
                active_states = [queue.popleft() for _ in range(count)]
                if args.max_items > 0:
                    active_states = active_states[: args.max_items - processed_this_run]
                indices = np.asarray(
                    [state["indices"][state["cursor"]] for state in active_states],
                    dtype=np.int64,
                )
                contexts = list(
                    pool.map(
                        _load_context,
                        (rows[int(index)] for index in indices),
                        [args.n_recent] * len(indices),
                        (list(state["keyframes"]) for state in active_states),
                    )
                )
                run_batch(indices, contexts, active_states)
                processed_this_run += len(indices)
                for trajectory_state in active_states:
                    if trajectory_state["cursor"] < len(trajectory_state["indices"]):
                        queue.append(trajectory_state)

    os.close(feature_fd)
    os.close(completed_fd)

    if args.generate_subtasks and bool(completed.all()):
        records = _read_subtask_records(subtasks_path)
        finalize_joint_state(records)
    elif args.generate_subtasks:
        print(
            f"Joint upper cache paused: features={int(completed.sum())}/{len(rows)} "
            f"(resume with the same command)",
            flush=True,
        )


if __name__ == "__main__":
    main()
