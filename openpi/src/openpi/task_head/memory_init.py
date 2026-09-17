from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import faiss  # type: ignore
import numpy as np
import torch

try:
    from torch._dynamo import disable as dynamo_disable
except Exception:

    def dynamo_disable(fn):
        return fn


def load_action_normalizer(
    norm_stats_path: Optional[str],
    *,
    use_quantiles: bool = True,
) -> Optional[dict[str, np.ndarray]]:
    if not norm_stats_path:
        return None
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats_root = data.get("norm_stats", data)
    if "actions" not in stats_root:
        raise KeyError(f"Cannot find actions norm stats in {norm_stats_path}")

    stats = stats_root["actions"]
    out: dict[str, np.ndarray] = {"use_quantiles": np.asarray(bool(use_quantiles))}
    for key in ("mean", "std", "q01", "q99"):
        value = stats.get(key)
        if value is not None:
            out[key] = np.asarray(value, dtype=np.float32)

    if use_quantiles and ("q01" not in out or "q99" not in out):
        raise KeyError(f"actions q01/q99 are required for quantile normalization: {norm_stats_path}")
    if not use_quantiles and ("mean" not in out or "std" not in out):
        raise KeyError(f"actions mean/std are required for z-score normalization: {norm_stats_path}")
    return out


def normalize_actions_array(
    actions: np.ndarray,
    normalizer: Optional[dict[str, np.ndarray]],
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if normalizer is None:
        return actions
    out = actions.copy()
    if bool(normalizer["use_quantiles"]):
        q01 = normalizer["q01"]
        q99 = normalizer["q99"]
        dim = min(out.shape[-1], q01.shape[-1])
        out[..., :dim] = (out[..., :dim] - q01[:dim]) / (q99[:dim] - q01[:dim] + 1e-6) * 2.0 - 1.0
    else:
        mean = normalizer["mean"]
        std = normalizer["std"]
        dim = min(out.shape[-1], mean.shape[-1])
        out[..., :dim] = (out[..., :dim] - mean[:dim]) / (std[:dim] + 1e-6)
    return out.astype(np.float32, copy=False)


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dynamo_disable
def _load_actions_npz(path: str) -> np.ndarray:
    with np.load(path) as npz:
        return np.asarray(npz["actions"], dtype=np.float32)


def _enumerate_chunks(actions: np.ndarray, chunk_len: int, stride: int) -> np.ndarray:
    total_steps, action_dim = actions.shape
    horizon = int(chunk_len)
    if total_steps < horizon:
        return np.empty((0, horizon, action_dim), dtype=np.float32)
    count = 1 + (total_steps - horizon) // int(stride)
    out = np.empty((count, horizon, action_dim), dtype=np.float32)
    start = 0
    for i in range(count):
        out[i] = actions[start : start + horizon]
        start += int(stride)
    return out


def _slice_actions_from_anchor(actions: np.ndarray, anchor_frame: int, horizon: int) -> np.ndarray:
    """Take the forward action chunk attached to one retrieved visual anchor."""
    total_steps, action_dim = actions.shape
    if total_steps <= 0:
        raise ValueError("Cannot slice an anchor chunk from an empty action trajectory")
    start = int(np.clip(int(anchor_frame), 0, total_steps - 1))
    stop = min(total_steps, start + int(horizon))
    block = np.asarray(actions[start:stop], dtype=np.float32)
    if len(block) < int(horizon):
        padding = np.repeat(block[-1:, :], int(horizon) - len(block), axis=0)
        block = np.concatenate((block, padding), axis=0)
    if block.shape != (int(horizon), action_dim):
        raise RuntimeError(f"Invalid anchor chunk shape: {block.shape}")
    return block


def _interpolate_local_peak(frames: np.ndarray, scores: np.ndarray) -> float:
    """Estimate a continuous frame from a discrete local similarity peak."""
    frames = np.asarray(frames, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if frames.ndim != 1 or scores.shape != frames.shape or len(frames) == 0:
        raise ValueError("Anchor frames and scores must be non-empty 1D arrays with equal shape")
    peak = int(np.argmax(scores))
    if peak == 0 or peak == len(frames) - 1:
        return float(frames[peak])

    local_frames = frames[peak - 1 : peak + 2]
    local_scores = scores[peak - 1 : peak + 2]
    center = float(local_frames[1])
    scale = float(max(local_frames[2] - local_frames[0], 1.0))
    x = (local_frames - center) / scale
    quadratic, linear, _ = np.polyfit(x, local_scores, deg=2)
    if not np.isfinite(quadratic) or not np.isfinite(linear) or quadratic >= -1e-12:
        return center
    vertex = center + float(-linear / (2.0 * quadratic)) * scale
    return float(np.clip(vertex, local_frames[0], local_frames[2]))


def _time_resample_to_horizon(actions: np.ndarray, horizon: int) -> np.ndarray:
    total_steps, action_dim = actions.shape
    if total_steps == horizon:
        return actions.copy()
    xs = np.linspace(0, total_steps - 1, num=total_steps, dtype=np.float32)
    qs = np.linspace(0, total_steps - 1, num=int(horizon), dtype=np.float32)
    out = np.empty((int(horizon), action_dim), dtype=np.float32)
    for i in range(action_dim):
        out[:, i] = np.interp(qs, xs, actions[:, i])
    return out


class PackedActionStore:
    """Read all memory trajectories from a single compressed NPZ file."""

    def __init__(self, path: str):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)
        self.actions = np.asarray(data["actions"], dtype=np.float32)
        self.offsets = np.asarray(data["offsets"], dtype=np.int64)
        if self.actions.ndim != 2 or not np.isfinite(self.actions).all():
            raise ValueError(f"Packed actions must be a finite [T,A] array in {self.path}")
        if self.offsets.ndim != 1 or self.offsets.shape[0] < 2:
            raise ValueError(f"Invalid offsets array in {self.path}")
        if self.offsets[0] != 0 or np.any(np.diff(self.offsets) < 0):
            raise ValueError(f"Packed action offsets must start at zero and be nondecreasing in {self.path}")
        if self.offsets[-1] != self.actions.shape[0]:
            raise ValueError(f"Packed actions and offsets do not match in {self.path}")
        self.ids: list[str] | None = None
        self.id_to_index: dict[str, int] = {}
        if "ids" in data:
            self.ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(data["ids"]).tolist()]
            if len(self.ids) != len(self):
                raise ValueError(f"Packed action IDs and offsets have different lengths in {self.path}")
            if len(set(self.ids)) != len(self.ids):
                raise ValueError(f"Packed action IDs must be unique in {self.path}")
            self.id_to_index = {value: i for i, value in enumerate(self.ids)}

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def get(self, key: int | str) -> np.ndarray:
        if isinstance(key, str):
            if key not in self.id_to_index:
                raise KeyError(f"Unknown packed action id: {key}")
            index = self.id_to_index[key]
        else:
            index = int(key)
        if index < 0 or index >= len(self):
            raise IndexError(f"Packed action index out of range: {index}")
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return np.asarray(self.actions[start:end], dtype=np.float32)


class MemoryInitProvider:
    """Retrieve task-conditioned action priors from GPM memory."""

    def __init__(
        self,
        memory_meta_path: str,
        faiss_index_path: str,
        *,
        memory_actions_path: str | None = None,
        align_mode: str = "hybrid",
        mixture_mode: str = "gaussian",
        temperature: float = 10.0,
        sigma_min: float = 0.05,
        noise_scale_range: tuple[float, float] = (0.2, 1.0),
        nfe_range: tuple[int, int] = (1, 10),
        nfe_floor: int | None = None,
        device: str = "cuda",
        action_norm_stats_path: str | None = None,
        action_use_quantile_norm: bool = True,
        progress_window: float = 0.20,
        action_alignment: str = "auto",
        retrieval_backend: str = "auto",
        allowed_task_ids: set[int] | None = None,
    ):
        dev = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.align_mode = str(align_mode)
        self.mixture_mode = str(mixture_mode)
        self.temperature = float(temperature)
        self.sigma_min = float(sigma_min)
        self.noise_low, self.noise_high = map(float, noise_scale_range)
        self.nfe_min, self.nfe_max = map(int, nfe_range)
        self.nfe_floor = self.nfe_min if nfe_floor is None else int(nfe_floor)
        if not self.nfe_min <= self.nfe_floor <= self.nfe_max:
            raise ValueError(
                f"nfe_floor must be in [{self.nfe_min}, {self.nfe_max}], got {self.nfe_floor}"
            )
        self.progress_window = float(progress_window)
        self.action_normalizer = load_action_normalizer(
            action_norm_stats_path,
            use_quantiles=bool(action_use_quantile_norm),
        )

        self.memory: list[dict] = _torch_load_cpu(memory_meta_path)
        if not self.memory:
            raise RuntimeError(f"Empty memory metadata: {memory_meta_path}")
        self.num_items = len(self.memory)
        self.allowed_task_ids = None if allowed_task_ids is None else frozenset(map(int, allowed_task_ids))
        if self.allowed_task_ids is not None:
            if not self.allowed_task_ids:
                raise ValueError("allowed_task_ids cannot be empty")
            self._retrieval_indices = np.asarray(
                [
                    index
                    for index, entry in enumerate(self.memory)
                    if int(entry.get("task_id", -1)) in self.allowed_task_ids
                ],
                dtype=np.int64,
            )
            if not len(self._retrieval_indices):
                raise ValueError(
                    f"No memory entries match allowed_task_ids={sorted(self.allowed_task_ids)}"
                )
        else:
            self._retrieval_indices = np.arange(self.num_items, dtype=np.int64)
        anchor_flags = ["anchor_frame" in entry for entry in self.memory]
        if any(anchor_flags) and not all(anchor_flags):
            raise ValueError("Memory metadata cannot mix anchor-aligned and legacy entries")
        self.anchor_aligned = bool(anchor_flags[0])
        if self.anchor_aligned:
            for index, entry in enumerate(self.memory):
                anchor_frame = int(entry["anchor_frame"])
                length = int(entry.get("length", entry.get("chunk_meta", {}).get("T", 0)))
                if anchor_frame < 0 or (length > 0 and anchor_frame >= length):
                    raise ValueError(
                        f"Invalid anchor_frame for memory entry {index}: frame={anchor_frame} length={length}"
                    )
        if action_alignment == "auto":
            action_alignment = "anchor_forward_v1" if self.anchor_aligned else "legacy_progress"
        if action_alignment not in (
            "legacy_progress",
            "anchor_forward_v1",
            "continuous_frame_v2",
            "dense_frame_v3",
        ):
            raise ValueError(f"Unknown memory action alignment: {action_alignment}")
        if action_alignment != "legacy_progress" and not self.anchor_aligned:
            raise ValueError(f"{action_alignment} requires metadata with anchor_frame")
        if action_alignment == "legacy_progress" and self.anchor_aligned:
            raise ValueError("legacy_progress cannot consume anchor-aligned metadata")
        self.action_alignment = str(action_alignment)

        self._anchors_by_action: dict[int | str, dict[str, np.ndarray]] = {}
        if self.action_alignment in ("continuous_frame_v2", "dense_frame_v3"):
            grouped: dict[int | str, list[tuple[int, float, int, np.ndarray]]] = {}
            for index, entry in enumerate(self.memory):
                key = self._entry_action_key(index)
                embedding = torch.as_tensor(entry["task_emb"], dtype=torch.float32).numpy()
                norm = float(np.linalg.norm(embedding))
                if norm <= 0.0 or not np.isfinite(norm):
                    raise ValueError(f"Invalid anchor embedding at memory entry {index}")
                grouped.setdefault(key, []).append(
                    (
                        int(entry["anchor_frame"]),
                        float(entry["anchor_progress"]),
                        int(index),
                        np.asarray(embedding / norm, dtype=np.float32),
                    )
                )
            for key, rows in grouped.items():
                rows.sort(key=lambda row: row[0])
                frames = np.asarray([row[0] for row in rows], dtype=np.int64)
                if len(np.unique(frames)) != len(frames):
                    raise ValueError(f"Duplicate anchor frames for action trajectory {key}")
                self._anchors_by_action[key] = {
                    "frames": frames,
                    "progress": np.asarray([row[1] for row in rows], dtype=np.float32),
                    "indices": np.asarray([row[2] for row in rows], dtype=np.int64),
                    "embeddings": np.stack([row[3] for row in rows], axis=0),
                }
        self.max_anchors_per_action = max(
            (len(group["frames"]) for group in self._anchors_by_action.values()),
            default=1,
        )
        self.index = faiss.read_index(faiss_index_path)
        if int(getattr(self.index, "ntotal", self.num_items)) != self.num_items:
            raise RuntimeError(
                f"FAISS index size and metadata length differ: "
                f"index={getattr(self.index, 'ntotal', 'unknown')} metadata={self.num_items}"
            )
        first_embedding = torch.as_tensor(self.memory[0].get("task_emb"))
        if first_embedding.ndim != 1 or int(self.index.d) != int(first_embedding.numel()):
            raise RuntimeError(
                f"FAISS index dimension and metadata embedding differ: index={self.index.d} "
                f"metadata={tuple(first_embedding.shape)}"
            )
        if retrieval_backend not in ("auto", "cpu", "gpu"):
            raise ValueError(f"Unknown retrieval backend: {retrieval_backend}")
        use_gpu = retrieval_backend == "gpu" or (
            retrieval_backend == "auto" and self.device.type == "cuda"
        )
        if use_gpu and self.device.type != "cuda":
            raise RuntimeError("GPU retrieval requested without a CUDA memory-provider device")
        self.retrieval_backend = "gpu" if use_gpu else "cpu"
        self._retrieval_keys = None
        self._filtered_cpu_index = None
        self._retrieval_index_map = self._retrieval_indices
        task_positions: dict[int, list[int]] = {}
        for position, memory_index in enumerate(self._retrieval_indices.tolist()):
            task_id = int(self.memory[memory_index].get("task_id", -1))
            task_positions.setdefault(task_id, []).append(position)
        self._retrieval_positions_by_task = {
            task_id: np.asarray(positions, dtype=np.int64)
            for task_id, positions in task_positions.items()
        }
        if use_gpu or self.allowed_task_ids is not None:
            keys = self.index.reconstruct_n(0, self.num_items)[self._retrieval_indices]
        if use_gpu:
            self._retrieval_keys = torch.from_numpy(
                np.ascontiguousarray(keys, dtype=np.float32)
            ).to(self.device)
        elif self.allowed_task_ids is not None:
            self._filtered_cpu_index = faiss.IndexFlatIP(int(self.index.d))
            self._filtered_cpu_index.add(np.ascontiguousarray(keys, dtype=np.float32))
        self.action_store = PackedActionStore(memory_actions_path) if memory_actions_path else None
        if self.action_store is not None:
            unresolved = []
            for index in range(self.num_items):
                key = self._entry_action_key(index)
                if isinstance(key, str) and key not in self.action_store.id_to_index:
                    unresolved.append(key)
                elif isinstance(key, int) and not 0 <= key < len(self.action_store):
                    unresolved.append(str(key))
            if unresolved:
                raise RuntimeError(
                    "Memory metadata contains action references missing from the packed action store: "
                    f"{unresolved[:5]}"
                )

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return normalize_actions_array(actions, self.action_normalizer)

    def _entry_action_key(self, memory_index: int) -> int | str:
        entry = self.memory[int(memory_index)]
        for key in ("action_id", "trajectory_id", "packed_id", "id"):
            if key in entry:
                return str(entry[key])
        return int(memory_index)

    def _load_actions_for_index(self, memory_index: int) -> np.ndarray:
        return self.normalize_actions(self._load_raw_actions_for_index(memory_index))

    def _load_raw_actions_for_index(self, memory_index: int) -> np.ndarray:
        entry = self.memory[int(memory_index)]
        if self.action_store is not None:
            actions = self.action_store.get(self._entry_action_key(memory_index))
        elif "actions_path" in entry:
            actions = _load_actions_npz(str(entry["actions_path"]))
        else:
            raise KeyError(
                "Memory metadata must contain action_id for packed actions or actions_path for legacy loading."
            )
        return np.asarray(actions, dtype=np.float32)

    def _weights_from_scores(self, scores: np.ndarray) -> np.ndarray:
        x = (scores - scores.max()) * self.temperature
        weights = np.exp(x)
        weights = weights / (weights.sum() + 1e-12)
        return weights.astype(np.float32)

    def _lambda_from_similarity(self, similarity: float) -> float:
        if not hasattr(self, "nfe_min") or not hasattr(self, "nfe_max"):
            s01 = (max(-1.0, min(1.0, float(similarity))) + 1.0) / 2.0
            return float(self.noise_high - s01 * (self.noise_high - self.noise_low))
        if self.nfe_max == self.nfe_min:
            return float(self.noise_low)
        effective_nfe = self._nfe_continuous_from_similarity(similarity)
        distance = (effective_nfe - self.nfe_min) / (self.nfe_max - self.nfe_min)
        return float(self.noise_low + distance * (self.noise_high - self.noise_low))

    def _raw_nfe_continuous_from_similarity(self, similarity: float) -> float:
        s01 = (max(-1.0, min(1.0, float(similarity))) + 1.0) / 2.0
        return float(self.nfe_min + (1.0 - s01) * (self.nfe_max - self.nfe_min))

    def _nfe_continuous_from_similarity(self, similarity: float) -> float:
        floor = int(getattr(self, "nfe_floor", self.nfe_min))
        return max(float(floor), self._raw_nfe_continuous_from_similarity(similarity))

    def _nfe_from_similarity(self, similarity: float) -> int:
        return int(round(self._nfe_continuous_from_similarity(similarity)))

    def _block_for_entry(
        self,
        memory_index: int,
        horizon: int,
        progress: float,
        *,
        frame_override: int | None = None,
    ) -> np.ndarray | None:
        entry = self.memory[int(memory_index)]
        chunk_meta = entry.get("chunk_meta", {})
        chunk_len = int(chunk_meta.get("chunk_len", horizon))
        stride = int(chunk_meta.get("stride", 1))
        actions = self._load_actions_for_index(memory_index)
        block = None

        if self.anchor_aligned:
            frame = int(entry["anchor_frame"]) if frame_override is None else int(frame_override)
            block = _slice_actions_from_anchor(actions, frame, int(horizon))
        elif self.align_mode in ("chunk", "hybrid"):
            chunks = _enumerate_chunks(actions, chunk_len, stride)
            if chunks.shape[0] > 0:
                chunk_index = int(np.clip(np.floor(progress * (chunks.shape[0] - 1)), 0, chunks.shape[0] - 1))
                block = chunks[chunk_index]

        if block is None and self.align_mode in ("resample", "hybrid"):
            block = _time_resample_to_horizon(actions, chunk_len)

        if block is not None and block.shape[0] != int(horizon):
            block = _time_resample_to_horizon(block, int(horizon))
        return block.astype(np.float32) if block is not None else None

    def _estimate_continuous_frame(
        self,
        query: np.ndarray,
        memory_index: int,
        horizon: int,
        tracker_progress: float | None,
        tracker_advance_steps: int,
    ) -> dict[str, float | int]:
        key = self._entry_action_key(memory_index)
        anchors = self._anchors_by_action[key]
        frames = anchors["frames"]
        scores = anchors["embeddings"] @ query

        length = int(
            self.memory[memory_index].get(
                "length",
                self.memory[memory_index].get("chunk_meta", {}).get("T", int(frames[-1]) + 1),
            )
        )
        max_frame = max(length - 1, 0)
        predicted_frame: float
        local_search = self.action_alignment == "dense_frame_v3" and tracker_progress is not None
        if local_search and max_frame > 0:
            previous_frame = float(tracker_progress) * max_frame
            predicted_frame = min(
                float(max_frame),
                previous_frame + max(0, int(tracker_advance_steps)),
            )
            correction_radius = float(max(1, 2 * horizon))
            local_mask = (
                (frames >= max(previous_frame, predicted_frame - correction_radius))
                & (frames <= min(float(max_frame), predicted_frame + correction_radius))
            )
            local_indices = np.flatnonzero(local_mask)
            if len(local_indices) == 0:
                peak = int(np.argmin(np.abs(frames.astype(np.float32) - predicted_frame)))
            else:
                peak = int(local_indices[int(np.argmax(scores[local_indices]))])
        else:
            peak = int(np.argmax(scores))
            predicted_frame = float(frames[peak])

        if self.action_alignment == "dense_frame_v3":
            estimated_frame = float(frames[peak])
        elif float(scores[peak]) >= 1.0 - 1e-5:
            estimated_frame = float(frames[peak])
        else:
            local_start = max(0, peak - 1)
            local_stop = min(len(frames), peak + 2)
            estimated_frame = _interpolate_local_peak(
                frames[local_start:local_stop],
                scores[local_start:local_stop],
            )
        if self.action_alignment == "continuous_frame_v2" and tracker_progress is not None and max_frame > 0:
            previous_frame = float(tracker_progress) * max_frame
            predicted_frame = min(
                float(max_frame),
                previous_frame + max(0, int(tracker_advance_steps)),
            )
            correction_radius = float(max(1, horizon))
            estimated_frame = float(
                np.clip(
                    estimated_frame,
                    max(previous_frame, predicted_frame - correction_radius),
                    min(float(max_frame), predicted_frame + correction_radius),
                )
            )
        else:
            predicted_frame = estimated_frame
        estimated_frame = float(np.clip(estimated_frame, 0.0, max_frame))
        rounded_frame = int(np.rint(estimated_frame))
        return {
            "estimated_frame_float": estimated_frame,
            "estimated_frame": rounded_frame,
            "estimated_progress": estimated_frame / max(max_frame, 1),
            "peak_anchor_frame": int(frames[peak]),
            "peak_anchor_index": int(anchors["indices"][peak]),
            "peak_anchor_score": float(scores[peak]),
            "predicted_frame": float(predicted_frame),
            "local_search": bool(local_search),
        }

    @dynamo_disable
    def search_batch(
        self,
        query_embs: torch.Tensor,
        k: int,
        required_task_ids: list[int] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if query_embs.ndim != 2:
            raise ValueError(f"query_embs must be [B,D], got {tuple(query_embs.shape)}")
        if required_task_ids is not None and len(required_task_ids) != int(query_embs.shape[0]):
            raise ValueError("required_task_ids must match the query batch size")
        anchors_per_action = (
            self.max_anchors_per_action
            if self.action_alignment in ("continuous_frame_v2", "dense_frame_v3")
            else 16
        )
        search_k = min(len(self._retrieval_indices), max(int(k) * anchors_per_action, 64))
        if required_task_ids is not None:
            queries = torch.nn.functional.normalize(
                query_embs.detach().to(self.device, dtype=torch.float32), dim=1
            )
            results: list[tuple[np.ndarray, np.ndarray]] = []
            for query, task_id in zip(queries, required_task_ids, strict=True):
                positions = self._retrieval_positions_by_task.get(int(task_id))
                if positions is None or not len(positions):
                    raise ValueError(f"No memory entries are available for required task_id={task_id}")
                global_indices = self._retrieval_indices[positions]
                local_k = min(len(positions), search_k)
                if self._retrieval_keys is not None:
                    position_tensor = torch.as_tensor(positions, device=self.device, dtype=torch.long)
                    task_keys = self._retrieval_keys.index_select(0, position_tensor)
                    scores, local_indices = torch.topk(
                        task_keys @ query, k=local_k, largest=True, sorted=True
                    )
                    scores_np = scores.cpu().numpy()
                    indices_np = global_indices[local_indices.cpu().numpy()]
                else:
                    keys = np.ascontiguousarray(
                        self.index.reconstruct_batch(global_indices), dtype=np.float32
                    )
                    query_np = query.detach().cpu().numpy()
                    scores_all = keys @ query_np
                    local_indices = np.argpartition(scores_all, -local_k)[-local_k:]
                    local_indices = local_indices[np.argsort(-scores_all[local_indices])]
                    scores_np = scores_all[local_indices]
                    indices_np = global_indices[local_indices]
                results.append((scores_np, indices_np))
            return results
        if self._retrieval_keys is not None:
            queries = torch.nn.functional.normalize(
                query_embs.detach().to(self.device, dtype=torch.float32), dim=1
            )
            with torch.inference_mode():
                scores, indices = torch.topk(
                    queries @ self._retrieval_keys.T,
                    k=search_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
            scores_np = scores.cpu().numpy()
            indices_np = self._retrieval_index_map[indices.cpu().numpy()]
        else:
            queries = np.ascontiguousarray(
                query_embs.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            faiss.normalize_L2(queries)
            search_index = self._filtered_cpu_index if self._filtered_cpu_index is not None else self.index
            scores_np, local_indices = search_index.search(queries, search_k)
            indices_np = (
                self._retrieval_index_map[local_indices]
                if self._filtered_cpu_index is not None
                else local_indices
            )
        return [(scores_np[i], indices_np[i]) for i in range(len(scores_np))]

    @dynamo_disable
    def query(
        self,
        query_emb: torch.Tensor,
        k: int,
        action_horizon: int,
        *,
        progress: float = 0.0,
        tracker_progress: float | None = None,
        tracker_advance_steps: int = 0,
        search_result: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        if query_emb.ndim != 1:
            raise ValueError(f"query_emb must be [D], got {tuple(query_emb.shape)}")

        query = query_emb.detach().cpu().numpy().astype("float32")[None, :]
        faiss.normalize_L2(query)
        if search_result is None:
            scores_raw, idxs_raw = self.search_batch(query_emb.unsqueeze(0), k)[0]
        else:
            scores_raw, idxs_raw = search_result

        selected_scores: list[float] = []
        selected_idxs: list[int] = []
        selected_actions: set[int | str] = set()
        candidates = [
            (float(score), int(idx))
            for score, idx in zip(scores_raw.tolist(), idxs_raw.tolist())
            if idx >= 0
        ]
        retrieval_progress = float(progress)
        progress_source = "client"
        frame_estimates: dict[int | str, dict[str, float | int]] = {}
        if self.action_alignment in ("continuous_frame_v2", "dense_frame_v3") and candidates:
            top_task = self.memory[candidates[0][1]].get("task_id")
            same_task = [item for item in candidates if self.memory[item[1]].get("task_id") == top_task]
            ranked = same_task or candidates
            candidates = []
            candidate_actions: set[int | str] = set()
            for item in ranked:
                action_key = self._entry_action_key(item[1])
                if action_key in candidate_actions:
                    continue
                candidate_actions.add(action_key)
                candidates.append(item)
                if len(candidates) >= int(k):
                    break
            for _, idx in candidates:
                key = self._entry_action_key(idx)
                frame_estimates[key] = self._estimate_continuous_frame(
                    query[0],
                    idx,
                    int(action_horizon),
                    tracker_progress,
                    tracker_advance_steps,
                )
            progress_source = (
                "retrieval_continuous_frame"
                if self.action_alignment == "continuous_frame_v2"
                else "retrieval_dense_frame_v3"
            )
        elif self.anchor_aligned and candidates:
            # Infer the window center from visual retrieval itself. This avoids
            # coupling memory alignment to a benchmark-specific episode clock.
            top_task = self.memory[candidates[0][1]].get("task_id")
            same_task = [
                item for item in candidates if self.memory[item[1]].get("task_id") == top_task
            ]
            frontier = same_task[: max(8, min(len(same_task), int(k) * 2))]
            frontier_scores = np.asarray([item[0] for item in frontier], dtype=np.float32)
            frontier_progress = np.asarray(
                [float(self.memory[item[1]]["anchor_progress"]) for item in frontier],
                dtype=np.float32,
            )
            frontier_weights = self._weights_from_scores(frontier_scores)
            order = np.argsort(frontier_progress)
            cumulative = np.cumsum(frontier_weights[order])
            median_index = min(int(np.searchsorted(cumulative, 0.5)), len(order) - 1)
            retrieval_progress = float(frontier_progress[order[median_index]])
            progress_source = "retrieval_anchor"
            in_window = [
                item
                for item in same_task
                if abs(float(self.memory[item[1]]["anchor_progress"]) - retrieval_progress)
                <= self.progress_window
            ]
            candidates = in_window or same_task
        elif any("anchor_progress" in self.memory[idx] for _, idx in candidates):
            in_window = [
                item
                for item in candidates
                if abs(float(self.memory[item[1]].get("anchor_progress", progress)) - float(progress))
                <= self.progress_window
            ]
            if in_window:
                candidates = in_window
            else:
                candidates.sort(
                    key=lambda item: (
                        abs(float(self.memory[item[1]].get("anchor_progress", progress)) - float(progress)),
                        -item[0],
                    )
                )
        for score, idx in candidates:
            action_key = self._entry_action_key(idx)
            if action_key in selected_actions:
                continue
            selected_actions.add(action_key)
            selected_scores.append(float(score))
            selected_idxs.append(int(idx))
            if len(selected_idxs) >= int(k):
                break
        if not selected_idxs:
            raise RuntimeError("FAISS returned no valid memory entries.")

        scores = np.asarray(selected_scores, dtype=np.float32)
        idxs = np.asarray(selected_idxs, dtype=np.int64)
        weights = self._weights_from_scores(scores)
        similarity = float((weights * scores).sum())
        if self.action_alignment in ("continuous_frame_v2", "dense_frame_v3"):
            selected_progress = np.asarray(
                [float(frame_estimates[self._entry_action_key(idx)]["estimated_progress"]) for idx in idxs],
                dtype=np.float32,
            )
            retrieval_progress = float((weights * selected_progress).sum())
            if tracker_progress is not None:
                retrieval_progress = max(float(tracker_progress), retrieval_progress)

        blocks = []
        block_weights = []
        sources = []
        horizon = int(action_horizon)
        for rank, memory_index in enumerate(idxs.tolist()):
            frame_info = frame_estimates.get(self._entry_action_key(memory_index))
            frame_override = int(frame_info["estimated_frame"]) if frame_info is not None else None
            block = self._block_for_entry(
                memory_index,
                horizon,
                float(progress),
                frame_override=frame_override,
            )
            if block is None:
                continue
            blocks.append(block)
            block_weights.append(float(weights[rank]))
            source = {
                "idx": int(memory_index),
                "action_id": self._entry_action_key(memory_index),
                "score": float(scores[rank]),
            }
            if frame_info is not None:
                source.update(frame_info)
            sources.append(source)
        if not blocks:
            raise RuntimeError("No valid action blocks were found in memory.")

        stacked = np.stack(blocks, axis=0)
        weights = np.asarray(block_weights, dtype=np.float32).reshape(-1, 1, 1)
        weights = weights / (weights.sum() + 1e-12)
        sample, lam = self._sample_from_blocks(stacked, weights, similarity)
        nfe_continuous_raw = self._raw_nfe_continuous_from_similarity(similarity)
        nfe_continuous = max(float(self.nfe_floor), nfe_continuous_raw)
        nfe = int(round(nfe_continuous))
        x_init = torch.from_numpy(sample).to(torch.float32).to(self.device)
        info = {
            "k": int(stacked.shape[0]),
            "scores": scores[: stacked.shape[0]].tolist(),
            "weights": weights[:, 0, 0].tolist(),
            "similarity_global": similarity,
            "lambda_noise": float(lam),
            "nfe": int(nfe),
            "nfe_continuous": float(nfe_continuous),
            "nfe_continuous_raw": float(nfe_continuous_raw),
            "nfe_floor": int(self.nfe_floor),
            "align_mode": self.align_mode,
            "mixture_mode": self.mixture_mode,
            "action_alignment": self.action_alignment,
            "retrieval_progress": float(retrieval_progress),
            "progress_source": progress_source,
            "sources": sources,
            "prior_mean": float(x_init.mean().item()),
            "prior_std": float(x_init.std(unbiased=False).item()),
            "win_start": 0,
            "win_end": horizon,
        }
        return x_init, nfe, info

    def _sample_from_blocks(
        self,
        blocks: np.ndarray,
        weights: np.ndarray,
        similarity: float,
        *,
        mixture_mode: str | None = None,
    ) -> tuple[np.ndarray, float]:
        mode = mixture_mode or self.mixture_mode
        lam = self._lambda_from_similarity(similarity)
        if mode == "gaussian":
            mu = (weights * blocks).sum(axis=0)
            var = (weights * (blocks - mu[None]) ** 2).sum(axis=0)
            var = np.maximum(var, self.sigma_min**2)
            eps = np.random.randn(*mu.shape).astype(np.float32)
            sample = mu + lam * eps * np.sqrt(var).astype(np.float32)
            return sample.astype(np.float32), lam
        if mode == "mog":
            probs = weights[:, 0, 0] / (weights[:, 0, 0].sum() + 1e-12)
            choice = np.random.choice(blocks.shape[0], size=(blocks.shape[1],), p=probs)
            sample = np.empty((blocks.shape[1], blocks.shape[2]), dtype=np.float32)
            base_std = max(self.sigma_min, 0.03)
            for step in range(blocks.shape[1]):
                action = blocks[choice[step], step]
                sample[step] = action + lam * base_std * np.random.randn(*action.shape).astype(np.float32)
            return sample.astype(np.float32), lam
        raise ValueError(f"Unknown mixture_mode: {mode}")

    def _effective_prior_noise_rms(
        self,
        blocks: np.ndarray,
        weights: np.ndarray,
        similarity: float,
        *,
        mixture_mode: str | None = None,
    ) -> float:
        """Estimate the normalized action-space uncertainty of the V1 prior."""
        mode = mixture_mode or self.mixture_mode
        lam = self._lambda_from_similarity(similarity)
        mu = (weights * blocks).sum(axis=0)
        component_var = (weights * (blocks - mu[None]) ** 2).sum(axis=0)

        if mode == "gaussian":
            noise_var = (lam**2) * np.maximum(component_var, self.sigma_min**2)
        elif mode == "mog":
            base_std = max(self.sigma_min, 0.03)
            noise_var = component_var + (lam * base_std) ** 2
        else:
            raise ValueError(f"Unknown mixture_mode: {mode}")
        return float(np.sqrt(np.mean(noise_var, dtype=np.float64)))


class NegativeMemoryProvider:
    """Negative trajectory retrieval with the same retrieve-then-align semantics as GPM."""

    def __init__(
        self,
        *,
        memory_meta_path: str,
        faiss_index_path: str,
        memory_actions_path: str,
        action_norm_stats_path: str | None,
        action_use_quantile_norm: bool = True,
        temperature: float = 10.0,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.memory: list[dict] = _torch_load_cpu(memory_meta_path)
        self.action_store = PackedActionStore(memory_actions_path)
        self.action_normalizer = load_action_normalizer(
            action_norm_stats_path, use_quantiles=bool(action_use_quantile_norm)
        )
        self.temperature = float(temperature)
        self.index = faiss.read_index(faiss_index_path)
        if int(self.index.ntotal) != len(self.memory) or len(self.action_store) != len(self.memory):
            raise RuntimeError("Negative memory metadata, action store, and FAISS index sizes do not match")

    @dynamo_disable
    def query_blocks(
        self,
        query_emb: torch.Tensor,
        *,
        k: int,
        horizon: int,
        progress: float,
        min_similarity: float,
        min_confidence: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict]:
        if int(self.index.ntotal) == 0:
            return None, None, {
                "reason": "empty_negative_bank",
                "available": 0,
                "selected": 0,
                "progress": float(progress),
            }
        query = query_emb.detach().cpu().numpy().astype(np.float32, copy=False).reshape(1, -1)
        query = np.ascontiguousarray(query)
        faiss.normalize_L2(query)
        search_k = min(int(self.index.ntotal), max(int(k), 1))
        scores_raw, idxs_raw = self.index.search(query, search_k)
        candidates = []
        all_scores = [float(score) for score in scores_raw[0] if np.isfinite(score)]
        for score, global_idx in zip(scores_raw[0].tolist(), idxs_raw[0].tolist()):
            if global_idx < 0:
                continue
            if score < float(min_similarity):
                continue
            entry = self.memory[int(global_idx)]
            confidence = float(entry.get("failure_confidence", 0.0))
            if confidence < float(min_confidence):
                continue
            candidates.append((int(global_idx), float(score), confidence))
        selected = candidates
        if not selected:
            return None, None, {
                "reason": "below_trajectory_similarity",
                "available": int(self.index.ntotal),
                "selected": 0,
                "best_similarity": max(all_scores, default=float("nan")),
                "progress": float(progress),
            }

        blocks = []
        logits = []
        accepted = []
        for global_idx, score, confidence in selected:
            entry = self.memory[global_idx]
            actions = self.action_store.get(str(entry["action_id"]))
            actions = normalize_actions_array(actions, self.action_normalizer)
            chunk_meta = entry.get("chunk_meta", {})
            chunk_len = int(chunk_meta.get("chunk_len", horizon))
            stride = int(chunk_meta.get("stride", 1))
            chunks = _enumerate_chunks(actions, chunk_len, stride)
            if len(chunks) == 0:
                continue
            chunk_idx = int(np.clip(np.floor(float(progress) * (len(chunks) - 1)), 0, len(chunks) - 1))
            if chunk_len != int(horizon):
                chunk = _time_resample_to_horizon(chunks[chunk_idx], int(horizon))
            else:
                chunk = chunks[chunk_idx]
            blocks.append(chunk)
            logits.append(self.temperature * score + math.log(max(confidence, 1e-12)))
            accepted.append((global_idx, score, confidence, chunk_idx))
        if not blocks:
            return None, None, {
                "reason": "no_valid_action_blocks",
                "available": int(self.index.ntotal),
                "selected": 0,
                "progress": float(progress),
            }
        logits_np = np.asarray(logits, dtype=np.float32)
        weights = np.exp(logits_np - logits_np.max())
        weights /= weights.sum() + 1e-12
        block_t = torch.from_numpy(np.stack(blocks).astype(np.float32)).to(self.device)
        weight_t = torch.from_numpy(weights.astype(np.float32)).to(self.device)
        return block_t, weight_t, {
            "reason": "ok",
            "available": int(self.index.ntotal),
            "selected": len(blocks),
            "scores": [item[1] for item in accepted],
            "confidences": [item[2] for item in accepted],
            "memory_indices": [item[0] for item in accepted],
            "memory_task_names": [str(self.memory[item[0]].get("task_name", "")) for item in accepted],
            "embedding_indices": [item[3] for item in accepted],
            "progress": float(progress),
        }


class ActionMemorySession:
    """Cache a top-k memory query for one episode and sample action priors by progress."""

    def __init__(
        self,
        provider: MemoryInitProvider,
        init_task_emb: torch.Tensor,
        k: int,
        H: int,
        *,
        progress: float = 0.0,
        tracker_progress: float | None = None,
        tracker_advance_steps: int = 0,
        search_result: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.provider = provider
        self.H = int(H)
        self.init_emb_cpu = init_task_emb.detach().to("cpu", copy=True).contiguous()
        _, _, info = provider.query(
            init_task_emb,
            k=k,
            action_horizon=H,
            progress=progress,
            tracker_progress=tracker_progress,
            tracker_advance_steps=tracker_advance_steps,
            search_result=search_result,
        )
        self.init_info = info
        self.tracker_progress = (
            float(info["retrieval_progress"])
            if provider.action_alignment in ("continuous_frame_v2", "dense_frame_v3")
            else None
        )
        self.nfe_adapt = int(info["nfe"])
        self.nfe_continuous = float(info.get("nfe_continuous", self.nfe_adapt))
        self.nfe_continuous_raw = float(info.get("nfe_continuous_raw", self.nfe_continuous))
        self.nfe_floor = int(info.get("nfe_floor", provider.nfe_min))
        self.lambda_noise = float(info.get("lambda_noise", 0.0))
        self.s_global = float(info["similarity_global"])
        self.blocks_all: list[np.ndarray] = []
        self.raw_blocks_all: list[np.ndarray] = []
        self.source_indices: list[int] = []
        self.source_scores: list[float] = []
        raw_scores: list[float] = []

        for source in info["sources"]:
            memory_index = int(source["idx"])
            entry = provider.memory[memory_index]
            chunk_meta = entry.get("chunk_meta", {})
            chunk_len = int(chunk_meta.get("chunk_len", self.H))
            stride = int(chunk_meta.get("stride", 1))
            raw_actions = provider._load_raw_actions_for_index(memory_index)
            actions = provider.normalize_actions(raw_actions)

            if provider.anchor_aligned:
                frame_override = source.get("estimated_frame")
                block = provider._block_for_entry(
                    memory_index,
                    self.H,
                    progress,
                    frame_override=int(frame_override) if frame_override is not None else None,
                )
                if block is not None:
                    raw_block = _slice_actions_from_anchor(
                        raw_actions,
                        int(frame_override) if frame_override is not None else int(entry["anchor_frame"]),
                        self.H,
                    )
                    self.blocks_all.append(block[None, ...].astype(np.float32))
                    self.raw_blocks_all.append(raw_block[None, ...].astype(np.float32))
                    self.source_indices.append(memory_index)
                    self.source_scores.append(float(source["score"]))
                    raw_scores.append(float(source["score"]))
                    continue

            if provider.align_mode in ("chunk", "hybrid"):
                chunks = _enumerate_chunks(actions, chunk_len, stride)
                raw_chunks = _enumerate_chunks(raw_actions, chunk_len, stride)
                if chunks.shape[0] > 0:
                    if chunk_len != self.H:
                        chunks = np.stack(
                            [_time_resample_to_horizon(chunk, self.H) for chunk in chunks],
                            axis=0,
                        )
                        raw_chunks = np.stack(
                            [_time_resample_to_horizon(chunk, self.H) for chunk in raw_chunks],
                            axis=0,
                        )
                    self.blocks_all.append(chunks.astype(np.float32))
                    self.raw_blocks_all.append(raw_chunks.astype(np.float32))
                    self.source_indices.append(memory_index)
                    self.source_scores.append(float(source["score"]))
                    raw_scores.append(float(source["score"]))
                    continue

            if provider.align_mode in ("resample", "hybrid"):
                block = _time_resample_to_horizon(actions, self.H)[None, ...]
                raw_block = _time_resample_to_horizon(raw_actions, self.H)[None, ...]
                self.blocks_all.append(block.astype(np.float32))
                self.raw_blocks_all.append(raw_block.astype(np.float32))
                self.source_indices.append(memory_index)
                self.source_scores.append(float(source["score"]))
                raw_scores.append(float(source["score"]))

        if raw_scores:
            scores = np.asarray(raw_scores, dtype=np.float32)
            self.weights = provider._weights_from_scores(scores)
        else:
            self.weights = np.asarray([1.0], dtype=np.float32)

    def _chosen_blocks(self, progress: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
        chosen = []
        win_starts = []
        for source_offset, blocks in enumerate(self.blocks_all):
            count = blocks.shape[0]
            if count <= 0:
                continue
            chunk_index = int(np.clip(np.floor(float(progress) * (count - 1)), 0, count - 1))
            chosen.append(blocks[chunk_index])
            if self.provider.action_alignment in ("continuous_frame_v2", "dense_frame_v3"):
                source = self.init_info["sources"][source_offset]
                win_starts.append(int(source["estimated_frame"]))
            else:
                win_starts.append(chunk_index)
        if not chosen:
            raise RuntimeError("No cached action blocks in memory session.")
        blocks = np.stack(chosen, axis=0)
        weights = self.weights[: blocks.shape[0]].reshape(-1, 1, 1)
        return blocks, weights, win_starts

    def _chosen_raw_blocks(self, progress: float) -> tuple[np.ndarray, np.ndarray]:
        chosen = []
        for blocks in self.raw_blocks_all:
            count = blocks.shape[0]
            if count <= 0:
                continue
            chunk_index = int(np.clip(np.floor(float(progress) * (count - 1)), 0, count - 1))
            chosen.append(blocks[chunk_index])
        if not chosen:
            raise RuntimeError("No cached raw action blocks in memory session.")
        blocks = np.stack(chosen, axis=0)
        weights = self.weights[: blocks.shape[0]].astype(np.float32)
        weights = weights / (weights.sum() + 1e-12)
        return blocks, weights

    @dynamo_disable
    def static_action_diagnostics(
        self,
        progress: float,
        *,
        arm_dim: int = 6,
        action_threshold: float = 1e-8,
        gate_fraction: float = 0.5,
    ) -> dict:
        """Measure exact-static exposure in raw, unnormalized retrieved actions."""
        blocks, weights = self._chosen_raw_blocks(progress)
        selected_arm_dim = min(max(1, int(arm_dim)), int(blocks.shape[-1]))
        arm_norm = np.linalg.norm(blocks[..., :selected_arm_dim], axis=-1)
        static_steps = arm_norm <= float(action_threshold)
        block_fractions = static_steps.mean(axis=1, dtype=np.float64)
        weighted_fraction = float(np.dot(weights.astype(np.float64), block_fractions))
        return {
            "raw_arm_dim": selected_arm_dim,
            "action_threshold": float(action_threshold),
            "gate_fraction": float(gate_fraction),
            "weighted_static_fraction": weighted_fraction,
            "block_static_fractions": [float(value) for value in block_fractions],
            "static_steps": int(static_steps.sum()),
            "total_steps": int(static_steps.size),
            "gate": bool(weighted_fraction >= float(gate_fraction)),
        }

    @dynamo_disable
    def prior_mean(self, progress: float) -> torch.Tensor:
        blocks, weights, _ = self._chosen_blocks(progress)
        mu = (weights * blocks).sum(axis=0).astype(np.float32)
        return torch.from_numpy(mu).to(torch.float32).to(self.provider.device)

    @dynamo_disable
    def estimate_progress(self, step_idx: int, replan_steps: int) -> float:
        lengths = []
        for source in self.init_info["sources"]:
            entry = self.provider.memory[int(source["idx"])]
            length = int(entry.get("length", 0))
            if length <= 0:
                length = int(entry.get("chunk_meta", {}).get("T", 0))
            lengths.append(length)
        if not lengths:
            return 0.0
        weights = self.weights[: len(lengths)]
        weights = weights / (weights.sum() + 1e-12)
        expected_length = float((weights * np.asarray(lengths, dtype=np.float32)).sum())
        executed_steps = float(int(step_idx) * max(1, int(replan_steps)))
        return float(np.clip(executed_steps / max(expected_length, 1.0), 0.0, 1.0))

    @dynamo_disable
    def sample_chunk(
        self,
        progress: float,
        *,
        mixture_mode: str | None = None,
        return_debug: bool = False,
    ):
        blocks, weights, win_starts = self._chosen_blocks(progress)
        sample, lam = self.provider._sample_from_blocks(
            blocks,
            weights,
            self.s_global,
            mixture_mode=mixture_mode,
        )
        x_t = torch.from_numpy(sample).to(torch.float32).to(self.provider.device)
        if not return_debug:
            return x_t

        top = np.argsort(-self.weights)[:3]
        debug = {
            "k": int(blocks.shape[0]),
            "win_start": int(win_starts[0]) if win_starts else -1,
            "win_end": int(win_starts[0] + self.H) if win_starts else -1,
            "top3_sims": [float(self.init_info["scores"][i]) for i in top[:3]],
            "top3_weights": [float(self.weights[i]) for i in top[:3]],
            "prior_mean": float(x_t.mean().item()),
            "prior_std": float(x_t.std(unbiased=False).item()),
            "noise_sigma": float(lam),
            "effective_noise_rms": self.provider._effective_prior_noise_rms(
                blocks,
                weights,
                self.s_global,
                mixture_mode=mixture_mode,
            ),
            "action_alignment": self.init_info.get("action_alignment", "legacy_progress"),
            "retrieval_progress": float(self.init_info.get("retrieval_progress", progress)),
        }
        return x_t, debug

    @dynamo_disable
    def sample_clean_chunk(self, progress: float, *, mixture_mode: str | None = None) -> torch.Tensor:
        """Sample an action-space endpoint before adding flow-path Gaussian noise."""
        blocks, weights, _ = self._chosen_blocks(progress)
        mode = mixture_mode or self.provider.mixture_mode
        if mode == "gaussian":
            sample = (weights * blocks).sum(axis=0)
        elif mode == "mog":
            probs = weights[:, 0, 0] / (weights[:, 0, 0].sum() + 1e-12)
            choices = np.random.choice(blocks.shape[0], size=(blocks.shape[1],), p=probs)
            sample = np.stack([blocks[choice, step] for step, choice in enumerate(choices)], axis=0)
        else:
            raise ValueError(f"Unknown mixture_mode: {mode}")
        return torch.from_numpy(sample.astype(np.float32)).to(torch.float32).to(self.provider.device)

    @dynamo_disable
    def guidance_tensors(self, progress: float, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return the current top-k action chunks and retrieval weights for flow guidance."""
        blocks, weights, win_starts = self._chosen_blocks(progress)
        weights_1d = weights.reshape(-1).astype(np.float32)
        weights_1d = weights_1d / (weights_1d.sum() + 1e-12)
        target_device = device if device is not None else self.provider.device
        block_t = torch.from_numpy(blocks.astype(np.float32)).to(device=target_device, dtype=torch.float32)
        weight_t = torch.from_numpy(weights_1d).to(device=target_device, dtype=torch.float32)
        top = np.argsort(-weights_1d)[:3]
        debug = {
            "k": int(blocks.shape[0]),
            "win_start": int(win_starts[0]) if win_starts else -1,
            "win_end": int(win_starts[0] + self.H) if win_starts else -1,
            "top3_sims": [float(self.init_info["scores"][i]) for i in top[:3]],
            "top3_weights": [float(weights_1d[i]) for i in top[:3]],
            "similarity_global": float(self.s_global),
            "scores": [float(x) for x in self.source_scores[: blocks.shape[0]]],
            "weights": [float(x) for x in weights_1d],
            "memory_indices": [int(x) for x in self.source_indices[: blocks.shape[0]]],
            "window_starts": [int(x) for x in win_starts],
            "action_alignment": self.init_info.get("action_alignment", "legacy_progress"),
            "retrieval_progress": float(self.init_info.get("retrieval_progress", progress)),
            "progress_source": self.init_info.get("progress_source", "client"),
            "estimated_frames": [
                int(source["estimated_frame"])
                for source in self.init_info["sources"][: blocks.shape[0]]
                if "estimated_frame" in source
            ],
            "estimated_frames_float": [
                float(source["estimated_frame_float"])
                for source in self.init_info["sources"][: blocks.shape[0]]
                if "estimated_frame_float" in source
            ],
            "peak_anchor_frames": [
                int(source["peak_anchor_frame"])
                for source in self.init_info["sources"][: blocks.shape[0]]
                if "peak_anchor_frame" in source
            ],
            "predicted_frames": [
                float(source["predicted_frame"])
                for source in self.init_info["sources"][: blocks.shape[0]]
                if "predicted_frame" in source
            ],
            "tracker_progress": (
                float(self.tracker_progress) if self.tracker_progress is not None else None
            ),
        }
        return block_t, weight_t, debug

    @dynamo_disable
    def maybe_refresh(
        self,
        new_task_emb: Optional[torch.Tensor],
        *,
        step_idx: int,
        refresh_every: int = 0,
        sim_threshold: float = 0.0,
        k: Optional[int] = None,
        advance_steps: int = 0,
        search_result: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> bool:
        should_refresh = False
        if refresh_every > 0 and step_idx > 0 and step_idx % refresh_every == 0:
            should_refresh = True
        if not should_refresh and sim_threshold > 0.0 and new_task_emb is not None:
            new_emb = new_task_emb.detach().to("cpu").contiguous().numpy().astype("float32")
            ref_emb = self.init_emb_cpu.detach().contiguous().numpy().astype("float32")
            cosine = float((new_emb * ref_emb).sum()) / float(
                np.linalg.norm(new_emb) * np.linalg.norm(ref_emb) + 1e-12
            )
            should_refresh = cosine < float(sim_threshold)

        if not should_refresh:
            return False

        query_emb = new_task_emb if new_task_emb is not None else self.init_emb_cpu.to(self.provider.device)
        self.__init__(
            provider=self.provider,
            init_task_emb=query_emb,
            k=int(k) if k is not None else max(1, len(self.blocks_all)),
            H=self.H,
            progress=0.0,
            tracker_progress=self.tracker_progress,
            tracker_advance_steps=int(advance_steps),
            search_result=search_result,
        )
        return True
