from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import os
import re
from pathlib import Path
import queue
import random
import threading
import time
from typing import Any

import numpy as np
from openpi.task_head.dual_tower_head import DualTowerRetrievalHead
from openpi_client import msgpack_numpy
from PIL import Image
import torch
from transformers import AutoProcessor, LogitsProcessorList, Qwen3VLForConditionalGeneration
from websockets.exceptions import ConnectionClosed
import websockets.sync.server

from optimus_eval.upper_knn_language_guidance import aggregate_subtask_candidates
from optimus_eval.upper_knn_language_guidance import KnnSubtaskLogitsProcessor
from optimus_eval.upper_knn_language_guidance import NumpyInnerProductIndex


@dataclass(frozen=True)
class TaskInfo:
    task_id: int
    brief_description: str
    task_block: str
    scene_description: str


@dataclass
class PlannerState:
    task: TaskInfo
    current_subtask: str
    j_hist: list[list[int]] = field(default_factory=list)
    frame_store_main: dict[int, Image.Image] = field(default_factory=dict)
    frame_store_wrist: dict[int, Image.Image] = field(default_factory=dict)
    frame_store_left_wrist: dict[int, Image.Image] = field(default_factory=dict)
    k_indices: list[int] = field(default_factory=list)
    guidance_locked_stage: int | None = None
    guidance_progress: float | None = None
    guidance_recent_native_stages: list[int] = field(default_factory=list)
    guidance_last_native_step: int | None = None
    guidance_pending_stage: int | None = None
    guidance_pending_count: int = 0
    guidance_temporal_stage: int = 0
    guidance_temporal_progress: float = 0.0
    guidance_temporal_evidence: float = 0.0
    guidance_temporal_same_used: int = 0
    guidance_temporal_previous_candidate: int | None = None
    guidance_feedback_target: str | None = None
    guidance_feedback_stage: int | None = None
    guidance_feedback_progress: float = 0.0
    guidance_feedback_baseline_similarity: float = 0.0
    guidance_feedback_baseline_purity: float = 0.0
    guidance_feedback_baseline_confidence: float = 0.0
    guidance_feedback_age: int = 0
    guidance_feedback_positive_count: int = 0
    guidance_feedback_negative_count: int = 0
    guidance_feedback_reinforcements: int = 0
    guidance_feedback_confirmed: bool = False
    guidance_feedback_rescue_count: int = 0
    guidance_feedback_rescue_confirmed: bool = False


@dataclass
class PendingRequest:
    payload: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: BaseException | None = None


def _normalize_subtask_text(text: str) -> str:
    return " ".join(str(text).replace("_", " ").replace("-", " ").lower().split())


def _infer_subtask_from_row(row: dict[str, Any]) -> str:
    task_block = str(row.get("task_block", "")).strip()
    try:
        stage_index = int(row.get("stage_index", -1))
        segment_paths = row.get("segment_paths")
        if isinstance(segment_paths, list) and 0 <= stage_index < len(segment_paths):
            stem = Path(str(segment_paths[stage_index])).stem
            stem = re.sub(r"_\d+_seed\d+_task\d+$", "", stem)
            stem = _normalize_subtask_text(stem)
            if stem:
                return stem
    except (TypeError, ValueError):
        pass
    prompt = str(row.get("prompt", task_block))
    return _normalize_subtask_text(prompt)


def _load_upper_guidance(
    *,
    head_path: Path,
    manifest_path: Path,
    upper_features: Path,
    lower_features: Path | None,
    upper_age: Path | None,
    top_k: int,
    temperature: float,
    min_task_purity: float,
    min_subtask_confidence: float,
    interpolation: float,
    bank_per_primitive: int,
    bank_seed: int,
    device: str,
    allowed_task_ids: set[int] | None = None,
) -> tuple["UpperLanguageGuidance",]:
    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    variant = str(payload.get("variant", ""))
    if variant not in {"upper", "fusion"}:
        raise ValueError(f"Language guidance requires variant='upper' or 'fusion', got {variant!r}")

    if bank_per_primitive < 1:
        raise ValueError("Upper guidance bank_per_primitive must be positive")
    rng = random.Random(bank_seed)
    reservoirs: dict[tuple[int, str], list[dict[str, Any]]] = {}
    seen: dict[tuple[int, str], int] = {}
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                task_id = int(row.get("task_id", -1))
                if allowed_task_ids is not None and task_id not in allowed_task_ids:
                    continue
                subtask = _infer_subtask_from_row(row)
                if task_id < 0 or not subtask:
                    continue
                key = (task_id, subtask)
                seen[key] = seen.get(key, 0) + 1
                bucket = reservoirs.setdefault(key, [])
                if len(bucket) < bank_per_primitive:
                    bucket.append(row)
                else:
                    replacement = rng.randrange(seen[key])
                    if replacement < bank_per_primitive:
                        bucket[replacement] = row
    rows = [row for key in sorted(reservoirs) for row in reservoirs[key]]
    if not rows:
        raise ValueError(f"Upper guidance manifest empty: {manifest_path}")
    rows.sort(key=lambda row: int(row["row_index"]))
    row_indices = np.asarray([int(row["row_index"]) for row in rows], dtype=np.int64)
    task_ids = np.asarray([int(row.get("task_id", -1)) for row in rows], dtype=np.int64)
    subtasks = np.asarray([_infer_subtask_from_row(row) for row in rows], dtype=object)
    stage_indices = np.asarray([int(row.get("stage_index", -1)) for row in rows], dtype=np.int64)
    anchor_progress = np.asarray([float(row.get("anchor_progress", 0.0)) for row in rows], dtype=np.float32)
    stage_progress = np.asarray([float(row.get("stage_progress", 0.0)) for row in rows], dtype=np.float32)
    action_ids = np.asarray([str(row.get("action_id", "")) for row in rows], dtype=object)

    upper = np.load(upper_features, mmap_mode="r")
    if upper.shape[0] <= int(row_indices.max()):
        raise ValueError(
            f"Upper feature bank length {upper.shape[0]} incompatible with max row_index {int(row_indices.max())}"
        )

    head = DualTowerRetrievalHead(
        variant=variant,
        lower_dim=int(payload["lower_dim"]),
        upper_dim=int(payload["upper_dim"]),
        hidden=int(payload["hidden"]),
        out_dim=int(payload["out_dim"]),
    )
    head.load_state_dict(payload["state_dict"], strict=True)
    if upper.shape[1] != head.upper_dim:
        raise ValueError(
            f"Upper feature dimension {upper.shape[1]} does not match upper head input {head.upper_dim}"
        )

    lower = None
    ages = None
    if variant == "fusion":
        if lower_features is None or upper_age is None:
            raise ValueError("Fusion language guidance requires lower_features and upper_age")
        lower = np.load(lower_features, mmap_mode="r")
        ages = np.load(upper_age, mmap_mode="r")
        if lower.shape != (upper.shape[0], head.lower_dim):
            raise ValueError(f"Fusion lower feature mismatch: lower={lower.shape} upper_rows={upper.shape[0]}")
        if ages.shape != (upper.shape[0],):
            raise ValueError(f"Fusion upper-age mismatch: ages={ages.shape} upper_rows={upper.shape[0]}")

    head = head.to(device).eval()
    projected_chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), 2048):
            stop = min(len(rows), start + 2048)
            indices = row_indices[start:stop]
            upper_batch = torch.from_numpy(np.asarray(upper[indices], dtype=np.float32)).to(device)
            if variant == "fusion":
                assert lower is not None and ages is not None
                lower_batch = torch.from_numpy(np.asarray(lower[indices], dtype=np.float32)).to(device)
                age_batch = torch.from_numpy(np.asarray(ages[indices], dtype=np.float32)).to(device)
                available = torch.ones(len(indices), dtype=torch.float32, device=device)
                projected = head(lower_batch, upper_batch, age_batch, available).float().cpu().numpy()
            else:
                projected = head(None, upper_batch).float().cpu().numpy()
            projected_chunks.append(projected)

    index = NumpyInnerProductIndex(np.concatenate(projected_chunks, axis=0))

    if index.ntotal != len(rows):
        raise RuntimeError(f"Upper guidance index size mismatch: expected {len(rows)} got {index.ntotal}")
    return (
        UpperLanguageGuidance(
            head=head,
            index=index,
            task_ids=task_ids,
            subtasks=subtasks,
            stage_indices=stage_indices,
            anchor_progress=anchor_progress,
            stage_progress=stage_progress,
            action_ids=action_ids,
            top_k=int(top_k),
            temperature=float(temperature),
            min_task_purity=float(min_task_purity),
            min_subtask_confidence=float(min_subtask_confidence),
            interpolation=float(interpolation),
            compute_device=device,
        ),
    )


SYSTEM_PROMPT = """You are an embodied-memory robot VLM planner.

You will observe two kinds of visual evidence from the same long-horizon execution:
1. Historical keyframes: moments before the current step, used to remember important past states.
2. A recent 5-frame dual-camera window ending at the current frame, used to infer the current primitive.
Temporal order: historical keyframes are ordered from earliest to latest; the recent 5-frame window is also ordered from earliest to latest, and the last timestep in that window is the current frame.

Your goal is not to narrate the full execution. Your goal is to infer the primitive the robot is currently executing, or should execute now, from these images.

Important rules:
- Historical keyframes are always earlier than the recent visual window.
- If there is no keyframe in the recent window, keyframe_positions must be an empty list.
- keyframe_positions are 1-indexed positions within the recent 5-frame window.
- Output strict JSON only, with no extra text.
- The JSON must contain exactly two fields: current_primitive and keyframe_positions."""


def _load_tasks(path: Path) -> dict[int, TaskInfo]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(task["task_id"]): TaskInfo(
            task_id=int(task["task_id"]),
            brief_description=str(task["brief_description"]),
            task_block=str(task["task_block"]),
            scene_description=str(task.get("scene_description", "")),
        )
        for task in raw["tasks"]
    }


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


def _build_visual_memory(j_hist: list[list[int]], step: int, recent_count: int, merge_distance: int) -> list[int]:
    candidates = sorted(index for group in j_hist for index in group)
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


@dataclass
class UpperLanguageGuidance:
    head: DualTowerRetrievalHead
    index: Any
    task_ids: np.ndarray
    subtasks: np.ndarray
    stage_indices: np.ndarray
    anchor_progress: np.ndarray
    stage_progress: np.ndarray
    action_ids: np.ndarray
    top_k: int
    temperature: float
    min_task_purity: float
    min_subtask_confidence: float
    interpolation: float
    compute_device: str = "cpu"

    def candidate_location(
        self,
        task_id: int,
        candidate: str,
        indices: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[int, float] | None:
        normalized = _normalize_subtask_text(candidate)
        selected = np.asarray(
            [
                index
                for index in range(len(indices))
                if self.task_ids[int(indices[index])] == task_id
                and _normalize_subtask_text(self.subtasks[int(indices[index])]) == normalized
            ],
            dtype=np.int64,
        )
        if not len(selected):
            return None
        bank_indices = indices[selected]
        candidate_scores = np.asarray(scores[selected], dtype=np.float64)
        weights = np.exp((candidate_scores - candidate_scores.max()) / max(self.temperature, 1e-6))
        weights /= weights.sum()
        stages = self.stage_indices[bank_indices]
        stage = int(np.bincount(stages[stages >= 0]).argmax()) if np.any(stages >= 0) else -1
        progress = float(np.sum(weights * self.anchor_progress[bank_indices]))
        return stage, progress

    def lookup_stage(self, task_id: int, subtask: str) -> int | None:
        normalized = _normalize_subtask_text(subtask)
        mask = np.asarray(
            [
                int(task) == int(task_id) and _normalize_subtask_text(value) == normalized
                for task, value in zip(self.task_ids, self.subtasks, strict=True)
            ],
            dtype=bool,
        )
        stages = self.stage_indices[mask]
        stages = stages[stages >= 0]
        if not len(stages):
            return None
        return int(np.bincount(stages).argmax())

    def build_probe_candidates(
        self,
        *,
        task_id: int,
        main_subtask: str,
        retrieved_candidates: list[str],
        center_stage: int | None,
        max_count: int,
    ) -> list[str]:
        """Preserve retrieval proposals, then fill collapsed Top-M from the task stage catalog."""
        output: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            text = str(value).strip()
            normalized = _normalize_subtask_text(text)
            if text and normalized not in seen and len(output) < max_count:
                output.append(text)
                seen.add(normalized)

        add(main_subtask)
        for value in retrieved_candidates:
            add(value)

        task_rows = np.flatnonzero(self.task_ids == int(task_id))
        catalog: dict[int, list[str]] = {}
        for index in task_rows:
            stage = int(self.stage_indices[int(index)])
            subtask = str(self.subtasks[int(index)])
            if stage < 0:
                continue
            values = catalog.setdefault(stage, [])
            if _normalize_subtask_text(subtask) not in {_normalize_subtask_text(value) for value in values}:
                values.append(subtask)
        origin = int(center_stage) if center_stage is not None and center_stage >= 0 else 0
        for stage in sorted(catalog, key=lambda value: (abs(value - origin), value < origin, value)):
            for value in catalog[stage]:
                add(value)
        return output

    def score_lower_subtask_probes_label_masked(
        self,
        raw_upper_feature: np.ndarray,
        task_id: int,
        probes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.head.variant != "fusion" or not probes:
            return {"available": False}
        lower = np.stack([np.asarray(item["feature"], dtype=np.float32) for item in probes], axis=0)
        upper = np.repeat(np.asarray(raw_upper_feature, dtype=np.float32)[None, :], len(probes), axis=0)
        device = self.compute_device
        with torch.inference_mode():
            projected = self.head(
                torch.from_numpy(lower).to(device),
                torch.from_numpy(upper).to(device),
                torch.zeros(len(probes), dtype=torch.float32, device=device),
                torch.ones(len(probes), dtype=torch.float32, device=device),
            ).float().cpu().numpy()
        scores = []
        for row, item in zip(projected, probes, strict=True):
            candidate = _normalize_subtask_text(item.get("candidate", ""))
            mask = np.asarray(
                [
                    int(memory_task) == int(task_id) and _normalize_subtask_text(memory_subtask) == candidate
                    for memory_task, memory_subtask in zip(self.task_ids, self.subtasks, strict=True)
                ],
                dtype=bool,
            )
            score = float(np.max(self.index.keys[mask] @ row)) if mask.any() else -1.0e9
            scores.append({"candidate": str(item.get("candidate", "")), "score": score})
        ranked = sorted(scores, key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        margin = best["score"] - ranked[1]["score"] if len(ranked) > 1 else 0.0
        return {
            "available": True,
            "best_candidate": best["candidate"],
            "best_score": best["score"],
            "margin": margin,
            "scores": ranked,
        }

    def score_lower_subtask_probes_global(
        self,
        raw_upper_feature: np.ndarray,
        task_id: int,
        probes: list[dict[str, Any]],
        *,
        top_k: int,
        current_stage: int | None,
        current_progress: float | None,
        progress_scale: float,
    ) -> dict[str, Any]:
        """Score candidate prompts against one shared task bank without label pre-filtering."""
        if self.head.variant != "fusion" or not probes:
            return {"available": False, "reason": "no-fusion-probes"}

        task_bank = np.flatnonzero(self.task_ids == int(task_id))
        if not len(task_bank):
            return {"available": False, "reason": "no-task-bank"}
        lower = np.stack([np.asarray(item["feature"], dtype=np.float32) for item in probes], axis=0)
        upper = np.repeat(np.asarray(raw_upper_feature, dtype=np.float32)[None, :], len(probes), axis=0)
        device = self.compute_device
        with torch.inference_mode():
            projected = self.head(
                torch.from_numpy(lower).to(device),
                torch.from_numpy(upper).to(device),
                torch.zeros(len(probes), dtype=torch.float32, device=device),
                torch.ones(len(probes), dtype=torch.float32, device=device),
            ).float().cpu().numpy()

        # Every candidate searches exactly the same complete current-task bank.
        task_scores = projected @ self.index.keys[task_bank].T
        k = min(max(int(top_k), 1), len(task_bank))
        if k == len(task_bank):
            local_top = np.argsort(-task_scores, axis=1)[:, :k]
        else:
            partition = np.argpartition(-task_scores, kth=k - 1, axis=1)[:, :k]
            partition_scores = np.take_along_axis(task_scores, partition, axis=1)
            order = np.argsort(-partition_scores, axis=1)
            local_top = np.take_along_axis(partition, order, axis=1)

        candidate_rows: list[dict[str, Any]] = []
        scale = max(float(progress_scale), 1e-6)
        for probe_idx, (probe, local_indices) in enumerate(zip(probes, local_top, strict=True)):
            bank_indices = task_bank[local_indices]
            similarities = task_scores[probe_idx, local_indices].astype(np.float64)
            weights = np.exp((similarities - similarities.max()) / max(self.temperature, 1e-6))
            weights /= max(float(weights.sum()), np.finfo(np.float64).tiny)

            subtask_mass: dict[str, float] = {}
            trajectory_mass: dict[str, float] = {}
            stage_mass: dict[int, float] = {}
            for bank_index, weight in zip(bank_indices, weights, strict=True):
                subtask = _normalize_subtask_text(self.subtasks[int(bank_index)])
                action_id = str(self.action_ids[int(bank_index)])
                stage = int(self.stage_indices[int(bank_index)])
                subtask_mass[subtask] = subtask_mass.get(subtask, 0.0) + float(weight)
                trajectory_mass[action_id] = trajectory_mass.get(action_id, 0.0) + float(weight)
                if stage >= 0:
                    stage_mass[stage] = stage_mass.get(stage, 0.0) + float(weight)

            candidate = str(probe.get("candidate", "")).strip()
            normalized_candidate = _normalize_subtask_text(candidate)
            inferred_subtask, inferred_posterior = max(subtask_mass.items(), key=lambda item: item[1])
            inferred_stage, stage_concentration = (
                max(stage_mass.items(), key=lambda item: item[1]) if stage_mass else (-1, 0.0)
            )
            trajectory_concentration = max(trajectory_mass.values(), default=0.0)
            progress_values = self.anchor_progress[bank_indices].astype(np.float64)
            inferred_progress = float(np.sum(weights * progress_values))
            progress_std = float(np.sqrt(np.sum(weights * (progress_values - inferred_progress) ** 2)))
            progress_coherence = float(np.exp(-progress_std / scale))
            candidate_posterior = float(subtask_mass.get(normalized_candidate, 0.0))
            similarity_quality = float(np.clip((similarities[0] + 1.0) * 0.5, 0.0, 1.0))

            temporal_consistency = 1.0
            if current_stage is not None and current_stage >= 0 and inferred_stage >= 0:
                if inferred_stage < current_stage or inferred_stage > current_stage + 1:
                    temporal_consistency *= 0.25
            if current_progress is not None:
                rollback = max(float(current_progress) - inferred_progress - 0.10, 0.0)
                jump = max(inferred_progress - float(current_progress) - 0.35, 0.0)
                temporal_consistency *= float(np.exp(-(rollback + jump) / scale))

            base_score = (
                0.45 * candidate_posterior
                + 0.20 * trajectory_concentration
                + 0.15 * float(stage_concentration)
                + 0.10 * progress_coherence
                + 0.10 * similarity_quality
            )
            score = float(base_score * temporal_consistency)
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "candidate_posterior": candidate_posterior,
                    "inferred_subtask": inferred_subtask,
                    "inferred_subtask_posterior": float(inferred_posterior),
                    "inferred_stage": int(inferred_stage),
                    "inferred_progress": inferred_progress,
                    "top_similarity": float(similarities[0]),
                    "similarity_margin": float(similarities[0] - similarities[1]) if len(similarities) > 1 else 0.0,
                    "trajectory_concentration": float(trajectory_concentration),
                    "stage_concentration": float(stage_concentration),
                    "progress_std": progress_std,
                    "progress_coherence": progress_coherence,
                    "temporal_consistency": temporal_consistency,
                    "candidate_agrees_with_posterior": inferred_subtask == normalized_candidate,
                    "top_neighbors": [
                        {
                            "subtask": str(self.subtasks[int(index)]),
                            "stage": int(self.stage_indices[int(index)]),
                            "progress": float(self.anchor_progress[int(index)]),
                            "action_id": str(self.action_ids[int(index)]),
                            "similarity": float(similarity),
                            "weight": float(weight),
                        }
                        for index, similarity, weight in zip(bank_indices, similarities, weights, strict=True)
                    ],
                }
            )

        ranked = sorted(candidate_rows, key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        margin = float(best["score"] - ranked[1]["score"]) if len(ranked) > 1 else 0.0
        return {
            "available": True,
            "bank_scope": "current-task-global",
            "bank_size": int(len(task_bank)),
            "top_k": int(k),
            "best_candidate": best["candidate"],
            "best_score": float(best["score"]),
            "best_candidate_posterior": float(best["candidate_posterior"]),
            "best_inferred_subtask": best["inferred_subtask"],
            "best_inferred_stage": int(best["inferred_stage"]),
            "best_inferred_progress": float(best["inferred_progress"]),
            "best_agrees_with_posterior": bool(best["candidate_agrees_with_posterior"]),
            "margin": margin,
            "scores": ranked,
        }

    def retrieve(
        self,
        raw_features: np.ndarray,
        task_ids: np.ndarray,
        lower_features: list[np.ndarray | None] | None = None,
        upper_ages: np.ndarray | None = None,
    ) -> list[tuple[list[tuple[str, float]], float, dict[str, Any]]]:
        if raw_features.size == 0:
            return []
        projected_np, available_rows = self.project_queries(
            raw_features,
            lower_features=lower_features,
            upper_ages=upper_ages,
        )
        with torch.inference_mode():
            scores, indices = self.index.search(projected_np, self.top_k)

        outputs = []
        for sample_idx in range(len(task_ids)):
            if not available_rows[sample_idx]:
                outputs.append(([], 0.0, {"enabled": True, "applied": False, "reason": "no-lower-feature"}))
                continue
            task_id = int(task_ids[sample_idx])
            row_scores = scores[sample_idx]
            row_neighbors = indices[sample_idx]
            valid = row_neighbors >= 0
            if not valid.any():
                outputs.append(([], 0.0, {"enabled": True, "applied": False, "reason": "no-neighbor"}))
                continue
            row_scores = row_scores[valid]
            row_neighbors = row_neighbors[valid]
            same_task = self.task_ids[row_neighbors] == task_id
            if not same_task.any():
                outputs.append(([], 0.0, {"enabled": True, "applied": False, "reason": "no-same-task"}))
                continue
            same_indices = row_neighbors[same_task]
            same_scores = row_scores[same_task]
            same_purity = float(same_task.mean())
            top_similarity = float(row_scores.max())
            same_task_similarity = float(same_scores.max())
            nonempty = np.asarray([bool(str(self.subtasks[int(idx)]).strip()) for idx in same_indices])
            same_indices = same_indices[nonempty]
            same_scores = same_scores[nonempty]
            same_subtasks = [str(self.subtasks[int(idx)]) for idx in same_indices]
            if not same_subtasks:
                outputs.append(
                    (
                        [],
                        0.0,
                        {
                            "enabled": True,
                            "applied": False,
                            "reason": "same-task-empty-candidates",
                            "same_task_purity": same_purity,
                            "top_similarity": top_similarity,
                            "same_task_similarity": same_task_similarity,
                        },
                    )
                )
                continue
            candidates = aggregate_subtask_candidates(
                same_subtasks,
                same_scores,
                temperature=self.temperature,
            )
            if not candidates:
                outputs.append(
                    (
                        [],
                        0.0,
                        {
                            "enabled": True,
                            "applied": False,
                            "reason": "candidate-collapse",
                            "same_task_purity": same_purity,
                            "top_similarity": top_similarity,
                            "same_task_similarity": same_task_similarity,
                        },
                    )
                )
                continue
            top_subtask, top_confidence = candidates[0]
            location = self.candidate_location(task_id, top_subtask, row_neighbors, row_scores)
            location_meta = (
                {"candidate_stage": location[0], "candidate_progress": location[1]}
                if location is not None
                else {}
            )
            if same_purity < self.min_task_purity or top_confidence < self.min_subtask_confidence:
                outputs.append(
                    (
                        candidates,
                        0.0,
                        {
                            "enabled": True,
                            "applied": False,
                            "reason": "gate-fail",
                            "same_task_purity": same_purity,
                            "same_task_confidence": top_confidence,
                            "top_similarity": top_similarity,
                            "same_task_similarity": same_task_similarity,
                            "candidate": top_subtask,
                            **location_meta,
                        },
                    )
                )
                continue

            outputs.append(
                (
                    candidates,
                    self.interpolation,
                    {
                        "enabled": True,
                        "applied": True,
                        "same_task_purity": same_purity,
                        "same_task_confidence": top_confidence,
                        "top_similarity": top_similarity,
                        "same_task_similarity": same_task_similarity,
                        "candidate": top_subtask,
                        "interpolation": self.interpolation,
                        **location_meta,
                    },
                )
            )
        return outputs

    def project_queries(
        self,
        raw_features: np.ndarray,
        *,
        lower_features: list[np.ndarray | None] | None = None,
        upper_ages: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[bool]]:
        """Project online states while retaining Lower-feature availability."""
        with torch.inference_mode():
            upper = torch.from_numpy(raw_features.astype(np.float32)).to(self.compute_device)
            if self.head.variant != "fusion":
                projected = self.head(None, upper)
                return projected.float().cpu().numpy(), [True] * len(raw_features)

            if lower_features is None:
                lower_features = [None] * len(raw_features)
            available_rows = [value is not None for value in lower_features]
            lower = torch.zeros(
                (len(raw_features), self.head.lower_dim), dtype=torch.float32, device=self.compute_device
            )
            for index, value in enumerate(lower_features):
                if value is None:
                    continue
                row = torch.from_numpy(np.asarray(value, dtype=np.float32).copy()).to(self.compute_device).flatten()
                if row.numel() != self.head.lower_dim:
                    raise ValueError(
                        f"Fusion online lower feature mismatch: expected {self.head.lower_dim}, got {row.numel()}"
                    )
                lower[index] = row
            age = torch.as_tensor(
                np.zeros(len(raw_features), dtype=np.float32)
                if upper_ages is None
                else np.asarray(upper_ages, dtype=np.float32),
                dtype=torch.float32,
                device=self.compute_device,
            ).flatten()
            if age.numel() != len(raw_features):
                raise ValueError(f"Fusion online upper age mismatch: expected {len(raw_features)}, got {age.numel()}")
            available = torch.tensor(available_rows, dtype=torch.float32, device=self.compute_device)
            projected = self.head(lower, upper, age, available)
            return projected.float().cpu().numpy(), available_rows


class BatchedUpperPlanner:
    def __init__(
        self,
        *,
        checkpoint: Path,
        processor_dir: Path,
        task_config: Path,
        device: str,
        batch_size: int,
        batch_wait_ms: float,
        max_new_tokens: int,
        merge_distance: int,
        keyframe_max: int,
        upper_guidance_enabled: bool = False,
        upper_guidance_head: Path | None = None,
        upper_guidance_manifest: Path | None = None,
        upper_guidance_features: Path | None = None,
        upper_guidance_lower_features: Path | None = None,
        upper_guidance_upper_age: Path | None = None,
        upper_guidance_top_k: int = 16,
        upper_guidance_temperature: float = 0.07,
        upper_guidance_min_task_purity: float = 0.0,
        upper_guidance_min_subtask_confidence: float = 0.0,
        upper_guidance_interpolation: float = 0.8,
        upper_guidance_bank_per_primitive: int = 16,
        upper_guidance_bank_seed: int = 17,
        upper_guidance_device: str = "cpu",
        upper_guidance_allowed_task_ids: str = "",
        upper_guidance_history_enabled: bool = False,
        upper_guidance_history_lock_observations: int = 2,
        upper_guidance_history_advance_confirmations: int = 2,
        upper_guidance_history_max_rollback: float = 0.10,
        upper_guidance_history_max_advance: float = 0.35,
        upper_guidance_temporal_enabled: bool = False,
        upper_guidance_temporal_posterior: float = 0.75,
        upper_guidance_temporal_purity: float = 0.35,
        upper_guidance_temporal_evidence_decay: float = 0.95,
        upper_guidance_temporal_advance_evidence: float = 0.45,
        upper_guidance_temporal_same_stage_budget: int = 2,
        upper_guidance_temporal_max_rollback: float = 0.15,
        upper_guidance_temporal_max_advance: float = 0.40,
        upper_guidance_feedback_mode: str = "off",
        upper_guidance_feedback_horizon: int = 4,
        upper_guidance_feedback_confirmations: int = 2,
        upper_guidance_feedback_similarity_tolerance: float = 0.02,
        upper_guidance_feedback_purity_tolerance: float = 0.10,
        upper_guidance_feedback_max_reinforcements: int = 2,
        upper_guidance_feedback_lower_rescue_enabled: bool = False,
        upper_guidance_feedback_lower_rescue_confirmations: int = 2,
        upper_guidance_feedback_lower_rescue_min_margin: float = 0.05,
        upper_guidance_lower_global_mode: str = "off",
        upper_guidance_lower_global_top_k: int = 16,
        upper_guidance_lower_global_probe_candidates: int = 4,
        upper_guidance_lower_global_min_posterior: float = 0.35,
        upper_guidance_lower_global_min_score: float = 0.45,
        upper_guidance_lower_global_min_margin: float = 0.03,
        upper_guidance_lower_global_progress_scale: float = 0.15,
        upper_stage_guidance_config: Path | None = None,
        upper_guidance_shadow_native: bool = False,
    ) -> None:
        self.tasks = _load_tasks(task_config)
        self.batch_size = batch_size
        self.batch_wait_seconds = batch_wait_ms / 1000.0
        self.max_new_tokens = max_new_tokens
        self.merge_distance = merge_distance
        self.keyframe_max = keyframe_max
        self.record_component_timing = os.environ.get("PREDIMEM_COMPONENT_TIMING", "0") == "1"
        self.upper_guidance_shadow_native = bool(upper_guidance_shadow_native)
        self.states: dict[str, PlannerState] = {}
        self.state_lock = threading.Lock()
        self.environment_locks: dict[str, threading.Lock] = {}
        self.requests: queue.Queue[PendingRequest | None] = queue.Queue()
        self.batch_index = 0

        self.processor = AutoProcessor.from_pretrained(
            processor_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        self.device = device
        self.worker = threading.Thread(target=self._batch_loop, name="predimem-upper-batcher", daemon=True)

        self.upper_guidance: UpperLanguageGuidance | None = None
        self.upper_stage_guidance = None
        self.upper_guidance_history_enabled = bool(upper_guidance_history_enabled)
        self.upper_guidance_history_lock_observations = max(int(upper_guidance_history_lock_observations), 1)
        self.upper_guidance_history_advance_confirmations = max(
            int(upper_guidance_history_advance_confirmations), 1
        )
        self.upper_guidance_history_max_rollback = max(float(upper_guidance_history_max_rollback), 0.0)
        self.upper_guidance_history_max_advance = max(float(upper_guidance_history_max_advance), 0.0)
        self.upper_guidance_temporal_enabled = bool(upper_guidance_temporal_enabled)
        if self.upper_guidance_history_enabled and self.upper_guidance_temporal_enabled:
            raise ValueError("Upper hard-history and temporal-v2 gates are mutually exclusive")
        self.upper_guidance_temporal_posterior = float(upper_guidance_temporal_posterior)
        self.upper_guidance_temporal_purity = float(upper_guidance_temporal_purity)
        self.upper_guidance_temporal_evidence_decay = float(upper_guidance_temporal_evidence_decay)
        self.upper_guidance_temporal_advance_evidence = float(upper_guidance_temporal_advance_evidence)
        self.upper_guidance_temporal_same_stage_budget = max(int(upper_guidance_temporal_same_stage_budget), 0)
        self.upper_guidance_temporal_max_rollback = max(float(upper_guidance_temporal_max_rollback), 0.0)
        self.upper_guidance_temporal_max_advance = max(float(upper_guidance_temporal_max_advance), 0.0)
        if upper_guidance_feedback_mode not in {"off", "shadow", "control"}:
            raise ValueError("Upper guidance feedback mode must be off, shadow, or control")
        self.upper_guidance_feedback_mode = upper_guidance_feedback_mode
        self.upper_guidance_feedback_horizon = max(int(upper_guidance_feedback_horizon), 1)
        self.upper_guidance_feedback_confirmations = max(int(upper_guidance_feedback_confirmations), 1)
        self.upper_guidance_feedback_similarity_tolerance = max(
            float(upper_guidance_feedback_similarity_tolerance), 0.0
        )
        self.upper_guidance_feedback_purity_tolerance = max(float(upper_guidance_feedback_purity_tolerance), 0.0)
        self.upper_guidance_feedback_max_reinforcements = max(
            int(upper_guidance_feedback_max_reinforcements), 0
        )
        self.upper_guidance_feedback_lower_rescue_enabled = bool(
            upper_guidance_feedback_lower_rescue_enabled
        )
        self.upper_guidance_feedback_lower_rescue_confirmations = max(
            int(upper_guidance_feedback_lower_rescue_confirmations), 1
        )
        self.upper_guidance_feedback_lower_rescue_min_margin = max(
            float(upper_guidance_feedback_lower_rescue_min_margin), 0.0
        )
        if upper_guidance_lower_global_mode not in {"off", "shadow", "control"}:
            raise ValueError("Lower global feedback mode must be off, shadow, or control")
        self.upper_guidance_lower_global_mode = upper_guidance_lower_global_mode
        self.upper_guidance_lower_global_top_k = max(int(upper_guidance_lower_global_top_k), 1)
        self.upper_guidance_lower_global_probe_candidates = max(
            int(upper_guidance_lower_global_probe_candidates), 1
        )
        self.upper_guidance_lower_global_min_posterior = float(
            np.clip(upper_guidance_lower_global_min_posterior, 0.0, 1.0)
        )
        self.upper_guidance_lower_global_min_score = float(
            np.clip(upper_guidance_lower_global_min_score, 0.0, 1.0)
        )
        self.upper_guidance_lower_global_min_margin = max(
            float(upper_guidance_lower_global_min_margin), 0.0
        )
        self.upper_guidance_lower_global_progress_scale = max(
            float(upper_guidance_lower_global_progress_scale), 1e-6
        )
        if upper_guidance_enabled:
            if upper_guidance_head is None or upper_guidance_manifest is None or upper_guidance_features is None:
                raise ValueError("Upper guidance enabled but head/manifest/features path missing")
            allowed_task_ids = {
                int(value.strip())
                for value in upper_guidance_allowed_task_ids.split(",")
                if value.strip()
            } or None
            (self.upper_guidance,) = _load_upper_guidance(
                head_path=upper_guidance_head,
                manifest_path=upper_guidance_manifest,
                upper_features=upper_guidance_features,
                lower_features=upper_guidance_lower_features,
                upper_age=upper_guidance_upper_age,
                top_k=upper_guidance_top_k,
                temperature=upper_guidance_temperature,
                min_task_purity=upper_guidance_min_task_purity,
                min_subtask_confidence=upper_guidance_min_subtask_confidence,
                interpolation=upper_guidance_interpolation,
                bank_per_primitive=upper_guidance_bank_per_primitive,
                bank_seed=upper_guidance_bank_seed,
                device=upper_guidance_device,
                allowed_task_ids=allowed_task_ids,
            )
            if self.upper_guidance_lower_global_mode != "off" and self.upper_guidance.head.variant != "fusion":
                raise ValueError("Lower global feedback requires a fusion retrieval head")

        if upper_stage_guidance_config is not None:
            if self.upper_guidance is None:
                raise ValueError("Stage-conditioned guidance requires --upper-guidance-enabled")
            if self.upper_guidance.head.variant != "fusion":
                raise ValueError(
                    "Stage-conditioned guidance with Lower feedback requires a Fusion retrieval head"
                )
            if (
                self.upper_guidance_history_enabled
                or self.upper_guidance_temporal_enabled
                or self.upper_guidance_feedback_mode != "off"
            ):
                raise ValueError(
                    "Stage-conditioned guidance replaces legacy history/temporal state machines; "
                    "Lower global counterfactual feedback remains supported"
                )
            if self.upper_guidance_lower_global_mode == "off":
                raise ValueError(
                    "Stage-conditioned guidance requires Lower global feedback; use shadow or control"
                )
            from will_guidance.integrations.predimem_runtime import PrediMemStageRuntime

            self.upper_stage_guidance = PrediMemStageRuntime.from_config(
                keys=self.upper_guidance.index.keys,
                task_ids=self.upper_guidance.task_ids,
                stage_ids=self.upper_guidance.stage_indices,
                labels=self.upper_guidance.subtasks,
                stage_progress=self.upper_guidance.stage_progress,
                anchor_progress=self.upper_guidance.anchor_progress,
                config_path=upper_stage_guidance_config,
            )

        self.worker.start()

    def _observe_guidance_feedback(self, state: PlannerState, meta: dict[str, Any]) -> None:
        if self.upper_guidance_feedback_mode == "off" or state.guidance_feedback_target is None:
            return
        state.guidance_feedback_age += 1
        candidate = _normalize_subtask_text(meta.get("candidate", ""))
        target = _normalize_subtask_text(state.guidance_feedback_target)
        similarity = float(meta.get("same_task_similarity", -1.0))
        purity = float(meta.get("same_task_purity", 0.0))
        confidence = float(meta.get("same_task_confidence", 0.0))
        similarity_delta = similarity - state.guidance_feedback_baseline_similarity
        purity_delta = purity - state.guidance_feedback_baseline_purity
        confidence_delta = confidence - state.guidance_feedback_baseline_confidence
        candidate_stage = int(meta.get("candidate_stage", -1))
        raw_retrieval_supported = (
            bool(candidate)
            and candidate == target
            and candidate_stage == state.guidance_feedback_stage
            and similarity
            >= state.guidance_feedback_baseline_similarity - self.upper_guidance_feedback_similarity_tolerance
            and purity >= state.guidance_feedback_baseline_purity - self.upper_guidance_feedback_purity_tolerance
        )
        probe_available = bool(meta.get("lower_probe_available", False))
        probe_supported = (
            not probe_available
            or _normalize_subtask_text(meta.get("lower_probe_best_candidate", "")) == target
        )
        probe_margin = float(meta.get("lower_probe_margin", 0.0) or 0.0)
        retrieval_supported = raw_retrieval_supported and probe_supported
        lower_rescue_evidence = (
            self.upper_guidance_feedback_lower_rescue_enabled
            and probe_available
            and probe_supported
            and not raw_retrieval_supported
            and probe_margin >= self.upper_guidance_feedback_lower_rescue_min_margin
        )
        if lower_rescue_evidence:
            state.guidance_feedback_rescue_count += 1
        else:
            state.guidance_feedback_rescue_count = 0
        state.guidance_feedback_rescue_confirmed = (
            state.guidance_feedback_rescue_count
            >= self.upper_guidance_feedback_lower_rescue_confirmations
        )
        if retrieval_supported:
            state.guidance_feedback_positive_count += 1
            state.guidance_feedback_negative_count = 0
        elif lower_rescue_evidence:
            state.guidance_feedback_negative_count = 0
        else:
            state.guidance_feedback_negative_count += 1
        state.guidance_feedback_confirmed = (
            state.guidance_feedback_positive_count >= self.upper_guidance_feedback_confirmations
            or state.guidance_feedback_rescue_confirmed
        )
        meta.update(
            feedback_mode=self.upper_guidance_feedback_mode,
            feedback_active=True,
            feedback_target=state.guidance_feedback_target,
            feedback_target_stage=state.guidance_feedback_stage,
            feedback_age=state.guidance_feedback_age,
            feedback_similarity_delta=similarity_delta,
            feedback_purity_delta=purity_delta,
            feedback_confidence_delta=confidence_delta,
            feedback_retrieval_supported=raw_retrieval_supported,
            feedback_joint_supported=retrieval_supported,
            feedback_lower_probe_supported=probe_supported,
            feedback_lower_rescue_evidence=lower_rescue_evidence,
            feedback_lower_rescue_count=state.guidance_feedback_rescue_count,
            feedback_lower_rescue_confirmed=state.guidance_feedback_rescue_confirmed,
            feedback_positive_count=state.guidance_feedback_positive_count,
            feedback_negative_count=state.guidance_feedback_negative_count,
            feedback_confirmed=state.guidance_feedback_confirmed,
            feedback_reinforcements=state.guidance_feedback_reinforcements,
        )
        if (
            state.guidance_feedback_age >= self.upper_guidance_feedback_horizon
            or state.guidance_feedback_negative_count >= self.upper_guidance_feedback_confirmations
        ):
            meta["feedback_closed"] = True
            state.guidance_feedback_target = None
            state.guidance_feedback_stage = None
            state.guidance_feedback_confirmed = False
            state.guidance_feedback_rescue_count = 0
            state.guidance_feedback_rescue_confirmed = False

    def _start_guidance_feedback(self, state: PlannerState, meta: dict[str, Any]) -> None:
        if self.upper_guidance_feedback_mode == "off":
            return
        target = str(meta.get("candidate", "")).strip()
        stage = int(meta.get("candidate_stage", -1))
        if not target or stage < 0:
            return
        if (
            state.guidance_feedback_target is not None
            and _normalize_subtask_text(state.guidance_feedback_target) == _normalize_subtask_text(target)
            and state.guidance_feedback_stage == stage
        ):
            meta["feedback_continued"] = True
            return
        state.guidance_feedback_target = target
        state.guidance_feedback_stage = stage
        state.guidance_feedback_progress = float(meta.get("candidate_progress", 0.0))
        state.guidance_feedback_baseline_similarity = float(meta.get("same_task_similarity", 0.0))
        state.guidance_feedback_baseline_purity = float(meta.get("same_task_purity", 0.0))
        state.guidance_feedback_baseline_confidence = float(meta.get("same_task_confidence", 0.0))
        state.guidance_feedback_age = 0
        state.guidance_feedback_positive_count = 0
        state.guidance_feedback_negative_count = 0
        state.guidance_feedback_reinforcements = 0
        state.guidance_feedback_confirmed = False
        state.guidance_feedback_rescue_count = 0
        state.guidance_feedback_rescue_confirmed = False
        meta["feedback_started"] = True

    def _apply_lower_global_feedback(
        self,
        guidance: tuple[list[tuple[str, float]], float, dict[str, Any]],
        diagnostic: dict[str, Any],
    ) -> tuple[list[tuple[str, float]], float, dict[str, Any]]:
        candidates, interpolation, source_meta = guidance
        meta = dict(source_meta)
        meta.update(
            lower_global_mode=self.upper_guidance_lower_global_mode,
            lower_global_source_candidates=[str(value[0]) for value in candidates],
            lower_global_available=bool(diagnostic.get("available", False)),
            lower_global_bank_scope=diagnostic.get("bank_scope", ""),
            lower_global_bank_size=diagnostic.get("bank_size", 0),
            lower_global_top_k=diagnostic.get("top_k", 0),
            lower_global_best_candidate=diagnostic.get("best_candidate", ""),
            lower_global_best_score=diagnostic.get("best_score"),
            lower_global_best_posterior=diagnostic.get("best_candidate_posterior"),
            lower_global_inferred_subtask=diagnostic.get("best_inferred_subtask", ""),
            lower_global_inferred_stage=diagnostic.get("best_inferred_stage", -1),
            lower_global_inferred_progress=diagnostic.get("best_inferred_progress", -1.0),
            lower_global_margin=diagnostic.get("margin"),
            lower_global_scores=diagnostic.get("scores", []),
        )
        if self.upper_guidance_lower_global_mode == "off":
            meta["lower_global_decision"] = "off"
            return candidates, interpolation, meta
        if not diagnostic.get("available", False):
            meta["lower_global_decision"] = diagnostic.get("reason", "unavailable")
            return candidates, interpolation, meta

        accepted = (
            bool(diagnostic.get("best_agrees_with_posterior", False))
            and float(diagnostic.get("best_candidate_posterior", 0.0))
            >= self.upper_guidance_lower_global_min_posterior
            and float(diagnostic.get("best_score", 0.0)) >= self.upper_guidance_lower_global_min_score
            and float(diagnostic.get("margin", 0.0)) >= self.upper_guidance_lower_global_min_margin
        )
        meta["lower_global_gate_passed"] = accepted
        if not accepted:
            meta["lower_global_decision"] = "gate-reject"
            return candidates, interpolation, meta

        corrected = str(diagnostic["best_candidate"]).strip()
        stage = int(diagnostic.get("best_inferred_stage", -1))
        progress = float(diagnostic.get("best_inferred_progress", -1.0))
        meta.update(
            lower_global_decision=(
                "shadow-accept" if self.upper_guidance_lower_global_mode == "shadow" else "control-accept"
            ),
            lower_global_corrected_subtask=corrected,
            lower_global_original_candidate=meta.get("candidate", ""),
        )
        if self.upper_guidance_lower_global_mode == "shadow":
            return candidates, interpolation, meta

        meta.update(
            candidate=corrected,
            candidate_stage=stage,
            candidate_progress=progress,
            same_task_confidence=float(diagnostic.get("best_candidate_posterior", 0.0)),
            lower_global_applied=True,
        )
        return [(corrected, 1.0)], interpolation, meta

    def _history_gate(
        self,
        state: PlannerState,
        guidance: tuple[list[tuple[str, float]], float, dict[str, Any]],
    ) -> tuple[list[tuple[str, float]], float, dict[str, Any]]:
        candidates, interpolation, source_meta = guidance
        meta = dict(source_meta)
        meta["history_enabled"] = True
        meta["history_locked_stage"] = state.guidance_locked_stage
        meta["history_progress"] = state.guidance_progress
        if state.guidance_locked_stage is None or not candidates:
            meta["history_decision"] = "unlocked-base-gate"
            return candidates, interpolation, meta

        confidence = float(meta.get("same_task_confidence", -1.0))
        stage = int(meta.get("candidate_stage", -1))
        progress = float(meta.get("candidate_progress", -1.0))
        if confidence < self.upper_guidance.min_subtask_confidence or stage < 0 or progress < 0:
            meta.update(applied=False, reason="history-invalid-candidate", history_decision="reject")
            return candidates, 0.0, meta

        locked_stage = int(state.guidance_locked_stage)
        locked_progress = float(state.guidance_progress if state.guidance_progress is not None else 0.0)
        if stage < locked_stage:
            meta.update(applied=False, reason="history-stage-regression", history_decision="reject")
            return candidates, 0.0, meta
        if stage > locked_stage + 1:
            meta.update(applied=False, reason="history-stage-jump", history_decision="reject")
            return candidates, 0.0, meta
        if progress < locked_progress - self.upper_guidance_history_max_rollback:
            meta.update(applied=False, reason="history-progress-regression", history_decision="reject")
            return candidates, 0.0, meta
        if progress > locked_progress + self.upper_guidance_history_max_advance:
            meta.update(applied=False, reason="history-progress-jump", history_decision="reject")
            return candidates, 0.0, meta

        if stage == locked_stage + 1:
            if state.guidance_pending_stage == stage:
                state.guidance_pending_count += 1
            else:
                state.guidance_pending_stage = stage
                state.guidance_pending_count = 1
            if state.guidance_pending_count < self.upper_guidance_history_advance_confirmations:
                meta.update(
                    applied=False,
                    reason="history-next-stage-pending",
                    history_decision="pending",
                    history_pending_count=state.guidance_pending_count,
                )
                return candidates, 0.0, meta
            state.guidance_locked_stage = stage
            state.guidance_pending_stage = None
            state.guidance_pending_count = 0
            meta["history_stage_advanced"] = True
        else:
            state.guidance_pending_stage = None
            state.guidance_pending_count = 0

        state.guidance_progress = max(locked_progress, progress)
        recovered = not bool(meta.get("applied", False))
        meta.update(
            applied=True,
            reason="history-recovered-purity" if recovered else "history-base-admit",
            history_decision="admit",
            history_recovered=recovered,
            history_locked_stage=state.guidance_locked_stage,
            history_progress=state.guidance_progress,
        )
        return candidates, self.upper_guidance.interpolation, meta

    def _temporal_gate(
        self,
        state: PlannerState,
        guidance: tuple[list[tuple[str, float]], float, dict[str, Any]],
    ) -> tuple[list[tuple[str, float]], float, dict[str, Any]]:
        candidates, _, source_meta = guidance
        meta = dict(source_meta)
        self._observe_guidance_feedback(state, meta)
        if (
            self.upper_guidance_feedback_mode == "control"
            and state.guidance_feedback_rescue_confirmed
            and state.guidance_feedback_target is not None
            and state.guidance_feedback_stage is not None
        ):
            meta.update(
                feedback_lower_rescue_applied=True,
                feedback_lower_rescue_original_candidate=meta.get("candidate", ""),
                candidate=state.guidance_feedback_target,
                candidate_stage=state.guidance_feedback_stage,
                candidate_progress=state.guidance_feedback_progress,
                same_task_similarity=state.guidance_feedback_baseline_similarity,
                same_task_purity=state.guidance_feedback_baseline_purity,
                same_task_confidence=state.guidance_feedback_baseline_confidence,
            )
            candidates = [(state.guidance_feedback_target, 1.0)]
        state.guidance_temporal_evidence *= self.upper_guidance_temporal_evidence_decay
        confidence = float(meta.get("same_task_confidence", -1.0))
        purity = float(meta.get("same_task_purity", 0.0))
        stage = int(meta.get("candidate_stage", -1))
        progress = float(meta.get("candidate_progress", -1.0))
        meta.update(
            temporal_enabled=True,
            temporal_stage=state.guidance_temporal_stage,
            temporal_progress=state.guidance_temporal_progress,
            temporal_evidence=state.guidance_temporal_evidence,
            temporal_same_used=state.guidance_temporal_same_used,
        )

        def reject(reason: str) -> tuple[list[tuple[str, float]], float, dict[str, Any]]:
            meta.update(applied=False, reason=reason, temporal_decision="reject")
            return candidates, 0.0, meta

        if (
            bool(meta.get("feedback_lower_rescue_evidence", False))
            and not bool(meta.get("feedback_lower_rescue_confirmed", False))
        ):
            return reject("feedback-lower-rescue-pending")

        if not candidates or stage < 0 or progress < 0 or confidence < self.upper_guidance_temporal_posterior:
            return reject("temporal-invalid-candidate")

        current_stage = state.guidance_temporal_stage
        current_progress = state.guidance_temporal_progress
        if (
            self.upper_guidance_feedback_mode == "control"
            and stage == current_stage + 1
            and state.guidance_feedback_target is not None
            and state.guidance_feedback_stage == current_stage
        ):
            return reject("feedback-current-stage-hold")
        if stage == current_stage + 1:
            state.guidance_temporal_evidence += confidence * max(
                purity, self.upper_guidance_temporal_purity
            )
        if stage < current_stage:
            return reject("temporal-stage-regression")
        if stage > current_stage + 1:
            return reject("temporal-stage-jump")
        delta = progress - current_progress
        if delta < -self.upper_guidance_temporal_max_rollback:
            return reject("temporal-progress-regression")
        if delta > self.upper_guidance_temporal_max_advance:
            return reject("temporal-progress-jump")

        use = False
        if stage == current_stage:
            if state.guidance_temporal_previous_candidate != stage:
                state.guidance_temporal_same_used = 0
            use = (
                purity >= self.upper_guidance_temporal_purity
                and state.guidance_temporal_same_used < self.upper_guidance_temporal_same_stage_budget
            )
            if use:
                state.guidance_temporal_same_used += 1
            elif (
                self.upper_guidance_feedback_mode == "control"
                and state.guidance_feedback_confirmed
                and state.guidance_feedback_target is not None
                and _normalize_subtask_text(meta.get("candidate", ""))
                == _normalize_subtask_text(state.guidance_feedback_target)
                and state.guidance_feedback_reinforcements
                < self.upper_guidance_feedback_max_reinforcements
            ):
                use = True
                state.guidance_feedback_reinforcements += 1
                meta["feedback_reinforced"] = True
        elif state.guidance_temporal_evidence >= self.upper_guidance_temporal_advance_evidence:
            state.guidance_temporal_stage = stage
            state.guidance_temporal_evidence = 0.0
            state.guidance_temporal_same_used = 0
            use = True
            meta["temporal_stage_advanced"] = True

        state.guidance_temporal_previous_candidate = stage
        if not use:
            return reject("temporal-budget-or-evidence")
        state.guidance_temporal_progress = max(current_progress, progress)
        meta.update(
            applied=True,
            reason="temporal-admit",
            temporal_decision="admit",
            temporal_stage=state.guidance_temporal_stage,
            temporal_progress=state.guidance_temporal_progress,
            temporal_evidence=state.guidance_temporal_evidence,
            temporal_same_used=state.guidance_temporal_same_used,
        )
        self._start_guidance_feedback(state, meta)
        return candidates, self.upper_guidance.interpolation, meta

    def _observe_native_subtask(
        self,
        state: PlannerState,
        subtask: str,
        guidance_meta: dict[str, Any],
        step_idx: int,
    ) -> None:
        if (
            not self.upper_guidance_history_enabled
            or self.upper_guidance is None
            or bool(guidance_meta.get("applied", False))
            or state.guidance_locked_stage is not None
        ):
            return
        if state.guidance_last_native_step is not None and step_idx <= state.guidance_last_native_step:
            guidance_meta["history_native_observation"] = "duplicate-step"
            return
        stage = self.upper_guidance.lookup_stage(state.task.task_id, subtask)
        if stage is None:
            return
        state.guidance_last_native_step = int(step_idx)
        state.guidance_recent_native_stages.append(stage)
        keep = self.upper_guidance_history_lock_observations
        state.guidance_recent_native_stages = state.guidance_recent_native_stages[-keep:]
        if len(state.guidance_recent_native_stages) == keep and len(set(state.guidance_recent_native_stages)) == 1:
            state.guidance_locked_stage = stage
            stage_progress = self.upper_guidance.anchor_progress[
                (self.upper_guidance.task_ids == state.task.task_id)
                & (self.upper_guidance.stage_indices == stage)
            ]
            state.guidance_progress = float(stage_progress.min()) if len(stage_progress) else 0.0
            guidance_meta["history_lock_acquired"] = True
            guidance_meta["history_locked_stage"] = stage
            guidance_meta["history_progress"] = state.guidance_progress

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        environment_id = str(payload["environment_id"])
        with self.state_lock:
            environment_lock = self.environment_locks.setdefault(environment_id, threading.Lock())
        with environment_lock:
            request = PendingRequest(payload=payload)
            self.requests.put(request)
            request.done.wait()
            if request.error is not None:
                raise request.error
            assert request.response is not None
            return request.response

    def _new_state(self, task_id: int) -> PlannerState:
        task = self.tasks[task_id]
        return PlannerState(task=task, current_subtask=task.brief_description.strip())

    def _prepare(
        self, payload: dict[str, Any]
    ) -> tuple[PlannerState, str, list[Image.Image], list[Image.Image], int, list[int]]:
        environment_id = str(payload["environment_id"])
        task_id = int(payload["task_id"])
        with self.state_lock:
            state = self.states.get(environment_id)
            if bool(payload.get("episode_reset")) or state is None or state.task.task_id != task_id:
                state = self._new_state(task_id)
                self.states[environment_id] = state

        step_idx = int(payload["step_idx"])
        main_raw = [np.asarray(frame, dtype=np.uint8) for frame in payload["context_main"]]
        three_view = "context_left_wrist" in payload or "context_right_wrist" in payload
        if three_view:
            left_raw = [np.asarray(frame, dtype=np.uint8) for frame in payload["context_left_wrist"]]
            wrist_raw = [np.asarray(frame, dtype=np.uint8) for frame in payload["context_right_wrist"]]
            if not main_raw or not (len(main_raw) == len(left_raw) == len(wrist_raw)):
                raise ValueError("Three-view Upper request requires equal main/left/right contexts")
        else:
            left_raw = []
            wrist_raw = [np.asarray(frame, dtype=np.uint8) for frame in payload["context_wrist"]]
            if not main_raw or len(main_raw) != len(wrist_raw):
                raise ValueError("Upper request requires equally sized non-empty main and wrist contexts")

        recent_start = step_idx - len(main_raw) + 1
        context_main = [Image.fromarray(frame) for frame in main_raw]
        context_wrist = [Image.fromarray(frame) for frame in wrist_raw]
        context_left = [Image.fromarray(frame) for frame in left_raw]
        for offset, (main, wrist) in enumerate(zip(context_main, context_wrist, strict=True)):
            index = recent_start + offset
            state.frame_store_main[index] = main
            state.frame_store_wrist[index] = wrist
            if three_view:
                state.frame_store_left_wrist[index] = context_left[offset]

        memory_indices = [index for index in state.k_indices if index in state.frame_store_main]
        memory_main = [state.frame_store_main[index] for index in memory_indices]
        memory_wrist = [state.frame_store_wrist[index] for index in memory_indices]
        memory_left = [state.frame_store_left_wrist[index] for index in memory_indices] if three_view else []
        messages = self._messages(
            state.task,
            memory_main,
            memory_wrist,
            context_main,
            context_wrist,
            memory_left=memory_left if three_view else None,
            context_left=context_left if three_view else None,
        )
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(text, list):
            text = text[0]
        images: list[Image.Image] = []
        for index, (main, wrist) in enumerate(
            [*zip(memory_main, memory_wrist, strict=True), *zip(context_main, context_wrist, strict=True)]
        ):
            if three_view:
                left = [*memory_left, *context_left][index]
                images.extend((main, left, wrist))
            else:
                images.extend((main, wrist))
        return state, text, images, recent_start, memory_indices

    @staticmethod
    def _messages(
        task: TaskInfo,
        memory_main: list[Image.Image],
        memory_wrist: list[Image.Image],
        context_main: list[Image.Image],
        context_wrist: list[Image.Image],
        *,
        memory_left: list[Image.Image] | None = None,
        context_left: list[Image.Image] | None = None,
    ) -> list[dict[str, Any]]:
        three_view = memory_left is not None and context_left is not None
        memory_left = memory_left or []
        context_left = context_left or []
        camera_text = (
            "Camera order for every timestep: third_person_camera, left_arm_camera, right_arm_camera. "
            "The first is the external view and the latter two are the left and right wrist cameras."
            if three_view
            else "Camera order for every timestep: agentview_rgb, eye_in_hand_rgb. "
            "agentview_rgb is the external main-view camera, and eye_in_hand_rgb is the wrist/end-effector camera."
        )
        views_per_timestep = 3 if three_view else 2
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Global objective: infer the robot's current primitive action from historical keyframes "
                    "and recent visual history within the same execution.\n\n"
                    f"Task objective:\n{task.task_block}\n\n"
                    f"Scene description:\n{task.scene_description or task.brief_description}\n\n"
                    f"{camera_text}\nCurrent observation:"
                ),
            }
        ]
        if memory_main:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Historical keyframes from moments before the current step in the same execution "
                        f"({len(memory_main)} timesteps, {views_per_timestep * len(memory_main)} images):"
                    ),
                }
            )
            for index, (main, wrist) in enumerate(zip(memory_main, memory_wrist, strict=True)):
                content.append({"type": "image", "image": main})
                if three_view:
                    content.append({"type": "image", "image": memory_left[index]})
                content.append({"type": "image", "image": wrist})
        content.append(
            {
                "type": "text",
                "text": (
                    "Recent visual context: "
                    f"{len(context_main)} consecutive frames ending at the current frame "
                    f"({views_per_timestep * len(context_main)} images):"
                ),
            }
        )
        for index, (main, wrist) in enumerate(zip(context_main, context_wrist, strict=True)):
            content.append({"type": "image", "image": main})
            if three_view:
                content.append({"type": "image", "image": context_left[index]})
            content.append({"type": "image", "image": wrist})
        content.append(
            {
                "type": "text",
                "text": (
                    "Output strict JSON with exactly two fields: current_primitive and keyframe_positions. "
                    "keyframe_positions are 1-indexed keyframe positions inside the recent visual window."
                ),
            }
        )
        system_prompt = SYSTEM_PROMPT.replace("dual-camera", "three-camera") if three_view else SYSTEM_PROMPT
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def _batch_loop(self) -> None:
        while True:
            first = self.requests.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.batch_wait_seconds
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.requests.put(None)
                    break
                batch.append(item)
            try:
                self._run_batch(batch)
            except BaseException as exc:
                logging.exception("Upper VLM batch failed")
                for request in batch:
                    request.error = exc
                    request.done.set()

    def _run_batch(self, batch: list[PendingRequest]) -> None:
        batch_started = time.perf_counter() if self.record_component_timing else None
        environment_ids = [str(request.payload["environment_id"]) for request in batch]
        if len(set(environment_ids)) != len(environment_ids):
            raise RuntimeError(f"One upper batch contains duplicate environment IDs: {environment_ids}")
        self.batch_index += 1
        logging.info(
            "upper_batch=%d size=%d environments=%s",
            self.batch_index,
            len(batch),
            ",".join(environment_ids),
        )
        heartbeat_fd = int(os.environ.get("PREDIMEM_HEARTBEAT_FD", "-1"))
        heartbeat_every = max(int(os.environ.get("PREDIMEM_HEARTBEAT_EVERY_BATCHES", "1")), 1)
        if heartbeat_fd >= 0 and self.batch_index % heartbeat_every == 0:
            try:
                os.write(
                    heartbeat_fd,
                    (
                        f"[predimem-heartbeat] upper_batch={self.batch_index} "
                        f"size={len(batch)} time={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                    ).encode("ascii"),
                )
            except OSError:
                pass

        prepared = [self._prepare(request.payload) for request in batch]
        texts = [item[1] for item in prepared]
        flat_images = [image for item in prepared for image in item[2]]
        inputs = self.processor(text=texts, images=flat_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        prompt_width = inputs["input_ids"].shape[1]
        shadow_native_primitives = [""] * len(batch)
        if self.upper_guidance_shadow_native and self.upper_guidance is not None:
            with torch.inference_mode():
                shadow_generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
            shadow_tokens = shadow_generation.sequences[:, prompt_width:]
            shadow_outputs = self.processor.batch_decode(
                shadow_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(shadow_outputs) != len(batch):
                raise RuntimeError(
                    f"Upper native-shadow batch mismatch: text={len(shadow_outputs)} batch={len(batch)}"
                )
            shadow_native_primitives = [
                _parse_output(output, len(request.payload["context_main"]))[0]
                for request, output in zip(batch, shadow_outputs, strict=True)
            ]
        logits_processor = None
        if self.upper_guidance is not None:
            with torch.inference_mode():
                context = self.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            features = context.hidden_states[-1][:, -1].detach().float().cpu().numpy()
            del context
            if self.upper_stage_guidance is not None:
                lower_features = [request.payload.get("lower_retrieval_feature") for request in batch]
                projected, lower_available = self.upper_guidance.project_queries(
                    features,
                    lower_features=lower_features,
                )
                stage_contexts = [
                    self.upper_stage_guidance.current_context(
                        str(request.payload["environment_id"]),
                        int(request.payload["task_id"]),
                    )
                    for request in batch
                ]
                global_probe_diagnostics = [
                    self.upper_guidance.score_lower_subtask_probes_global(
                        feature,
                        int(request.payload["task_id"]),
                        list(request.payload.get("lower_subtask_probes", [])),
                        top_k=self.upper_guidance_lower_global_top_k,
                        current_stage=stage_context[0],
                        current_progress=stage_context[1],
                        progress_scale=self.upper_guidance_lower_global_progress_scale,
                    )
                    for feature, request, stage_context in zip(features, batch, stage_contexts, strict=True)
                ]
                from will_guidance.types import LowerFeedbackEvidence

                lower_feedbacks = []
                for diagnostic in global_probe_diagnostics:
                    accepted = bool(
                        diagnostic.get("available", False)
                        and diagnostic.get("best_agrees_with_posterior", False)
                        and float(diagnostic.get("best_candidate_posterior", 0.0))
                        >= self.upper_guidance_lower_global_min_posterior
                        and float(diagnostic.get("best_score", 0.0))
                        >= self.upper_guidance_lower_global_min_score
                        and float(diagnostic.get("margin", 0.0))
                        >= self.upper_guidance_lower_global_min_margin
                    )
                    lower_feedbacks.append(
                        LowerFeedbackEvidence(
                            available=bool(diagnostic.get("available", False)),
                            accepted=accepted,
                            candidate=str(diagnostic.get("best_candidate", "")),
                            inferred_stage=int(diagnostic.get("best_inferred_stage", -1)),
                            inferred_progress=float(diagnostic.get("best_inferred_progress", -1.0)),
                            score=float(diagnostic.get("best_score", 0.0) or 0.0),
                            posterior=float(diagnostic.get("best_candidate_posterior", 0.0) or 0.0),
                            margin=float(diagnostic.get("margin", 0.0) or 0.0),
                        )
                    )
                guidance_candidates = self.upper_stage_guidance.observe_batch(
                    projected_queries=projected,
                    task_ids=[int(request.payload["task_id"]) for request in batch],
                    environment_ids=[str(request.payload["environment_id"]) for request in batch],
                    steps=[int(request.payload["step_idx"]) for request in batch],
                    # The current decode does not exist until after logits are
                    # guided. The previous decode is history, not a valid
                    # counterfactual native prediction for this observation.
                    native_stages=[None] * len(batch),
                    completed_stages=[request.payload.get("completed_stage") for request in batch],
                    lower_feedbacks=lower_feedbacks,
                )
                guidance_candidates = [
                    (
                        guidance[0],
                        guidance[1],
                        {
                            **guidance[2],
                            "lower_feature_available": bool(available),
                            "lower_global_mode": self.upper_guidance_lower_global_mode,
                            "lower_global_available": bool(diagnostic.get("available", False)),
                            "lower_global_bank_scope": diagnostic.get("bank_scope", ""),
                            "lower_global_bank_size": diagnostic.get("bank_size", 0),
                            "lower_global_top_k": diagnostic.get("top_k", 0),
                            "lower_global_scores": diagnostic.get("scores", []),
                            "lower_global_decision": (
                                "stage-evidence-accept"
                                if feedback.accepted
                                else diagnostic.get("reason", "stage-evidence-reject")
                            ),
                            # The stage controller consumes this evidence. It
                            # never authorizes a second legacy output override.
                            "lower_global_output_authorized": False,
                        },
                    )
                    for guidance, available, diagnostic, feedback in zip(
                        guidance_candidates,
                        lower_available,
                        global_probe_diagnostics,
                        lower_feedbacks,
                        strict=True,
                    )
                ]
            else:
                guidance_candidates = self.upper_guidance.retrieve(
                    raw_features=features,
                    task_ids=np.asarray([int(request.payload["task_id"]) for request in batch], dtype=np.int64),
                    lower_features=[request.payload.get("lower_retrieval_feature") for request in batch],
                )
            probe_diagnostics = [
                self.upper_guidance.score_lower_subtask_probes_label_masked(
                    feature,
                    int(request.payload["task_id"]),
                    list(request.payload.get("lower_subtask_probes", [])),
                )
                if self.upper_guidance_feedback_lower_rescue_enabled
                else {"available": False}
                for feature, request in zip(features, batch, strict=True)
            ]
            global_probe_diagnostics = []
            if self.upper_stage_guidance is None:
                for feature, request, prepared_item in zip(features, batch, prepared, strict=True):
                    state = prepared_item[0]
                    current_stage = (
                        state.guidance_temporal_stage
                        if self.upper_guidance_temporal_enabled
                        else state.guidance_locked_stage
                    )
                    current_progress = (
                        state.guidance_temporal_progress
                        if self.upper_guidance_temporal_enabled
                        else state.guidance_progress
                    )
                    global_probe_diagnostics.append(
                        self.upper_guidance.score_lower_subtask_probes_global(
                            feature,
                            int(request.payload["task_id"]),
                            list(request.payload.get("lower_subtask_probes", [])),
                            top_k=self.upper_guidance_lower_global_top_k,
                            current_stage=current_stage,
                            current_progress=current_progress,
                            progress_scale=self.upper_guidance_lower_global_progress_scale,
                        )
                        if self.upper_guidance_lower_global_mode != "off"
                        else {"available": False, "reason": "off"}
                    )
            guidance_candidates = [
                (
                    guidance[0],
                    guidance[1],
                    {
                        **guidance[2],
                        "lower_probe_available": bool(probe.get("available", False)),
                        "lower_probe_best_candidate": probe.get("best_candidate", ""),
                        "lower_probe_best_score": probe.get("best_score"),
                        "lower_probe_margin": probe.get("margin"),
                        "lower_probe_scores": probe.get("scores", []),
                    },
                )
                for guidance, probe in zip(guidance_candidates, probe_diagnostics, strict=True)
            ]
            if self.upper_stage_guidance is None:
                guidance_candidates = [
                    self._apply_lower_global_feedback(guidance, diagnostic)
                    for guidance, diagnostic in zip(guidance_candidates, global_probe_diagnostics, strict=True)
                ]
            if self.upper_guidance_history_enabled:
                guidance_candidates = [
                    self._history_gate(item[0], guidance)
                    for item, guidance in zip(prepared, guidance_candidates, strict=True)
                ]
            elif self.upper_guidance_temporal_enabled:
                guidance_candidates = [
                    self._temporal_gate(item[0], guidance)
                    for item, guidance in zip(prepared, guidance_candidates, strict=True)
                ]
            guidance_candidates = [
                (
                    candidates,
                    interpolation,
                    {
                        **meta,
                        "lower_global_output_authorized": bool(
                            meta.get("lower_global_applied", False)
                            and interpolation > 0.0
                            and meta.get("applied", True)
                        ),
                    },
                )
                for candidates, interpolation, meta in guidance_candidates
            ]
            logits_processor = LogitsProcessorList(
                [
                    KnnSubtaskLogitsProcessor(
                        self.processor.tokenizer,
                        [item[0] for item in guidance_candidates],
                        prompt_width=prompt_width,
                        interpolation=[item[1] for item in guidance_candidates],
                    )
                ]
            )
        else:
            guidance_candidates = [([], 0.0, {"enabled": False}) for _ in batch]

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_hidden_states": self.upper_guidance is None,
        }
        if logits_processor is not None:
            generation_kwargs["logits_processor"] = logits_processor
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        if self.upper_guidance is None:
            hidden_steps = generation.hidden_states
            if not hidden_steps:
                raise RuntimeError("Qwen generation did not return hidden states")
            first_step = hidden_steps[0]
            last_layer = first_step[-1] if isinstance(first_step, (tuple, list)) else first_step
            features = last_layer[:, -1].detach().float().cpu().numpy()

        generated = generation.sequences[:, prompt_width:]
        outputs = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if len(outputs) != len(batch) or features.shape[0] != len(batch):
            raise RuntimeError(f"Upper batch shape mismatch: text={len(outputs)} features={features.shape}")

        for request, item, output, feature, guidance, shadow_native in zip(
            batch,
            prepared,
            outputs,
            features,
            guidance_candidates,
            shadow_native_primitives,
            strict=True,
        ):
            state, _, _, recent_start, _ = item
            context_count = len(request.payload["context_main"])
            primitive, relative_positions = _parse_output(output, context_count)
            absolute_positions = [recent_start + position - 1 for position in relative_positions]
            state.j_hist.append(absolute_positions)
            state.k_indices = _build_visual_memory(
                state.j_hist,
                int(request.payload["step_idx"]) + 1,
                context_count,
                self.merge_distance,
            )
            if self.keyframe_max > 0:
                state.k_indices = state.k_indices[-self.keyframe_max :]
            keep = set(state.k_indices)
            keep.update(index for group in state.j_hist for index in group)
            state.frame_store_main = {
                index: image for index, image in state.frame_store_main.items() if index in keep
            }
            state.frame_store_wrist = {
                index: image for index, image in state.frame_store_wrist.items() if index in keep
            }
            state.frame_store_left_wrist = {
                index: image for index, image in state.frame_store_left_wrist.items() if index in keep
            }

            _, _, guidance_meta = guidance
            guidance_meta["generated_subtask"] = primitive
            if shadow_native:
                guidance_meta["shadow_native_subtask"] = shadow_native
                guidance_meta["shadow_native_stage"] = self.upper_guidance.lookup_stage(
                    int(request.payload["task_id"]), shadow_native
                )
            if (
                self.upper_guidance_lower_global_mode == "control"
                and bool(guidance_meta.get("lower_global_output_authorized", False))
            ):
                guidance_meta["lower_global_native_generated_subtask"] = primitive
                primitive = str(guidance_meta["lower_global_corrected_subtask"])
                guidance_meta["lower_global_output_overridden"] = True
            self._observe_native_subtask(
                state,
                primitive,
                guidance_meta,
                int(request.payload["step_idx"]),
            )
            state.current_subtask = primitive
            if feature.ndim != 1 or not np.isfinite(feature).all():
                raise RuntimeError(f"Invalid upper feature: {feature.shape}")
            response: dict[str, Any] = {
                "subtask": state.current_subtask,
                "feature": feature.astype(np.float32),
                "feature_step": int(request.payload["step_idx"]),
                "keyframe_positions": relative_positions,
                "keyframe_count": len(state.k_indices),
                "upper_timing": (
                    {
                        "batch_ms": float((time.perf_counter() - batch_started) * 1000.0 if batch_started is not None else float("nan")),
                        "batch_size": int(len(batch)),
                        "per_sample_ms": float((time.perf_counter() - batch_started) * 1000.0 / len(batch))
                        if batch_started is not None
                        else float("nan"),
                    }
                    if self.record_component_timing
                    else {}
                ),
                "upper_guidance": guidance_meta,
                "probe_candidates": (
                    (
                        self.upper_stage_guidance.probe_candidates(
                            int(request.payload["task_id"]),
                            int(guidance_meta.get("candidate_stage", 0)),
                            self.upper_guidance_lower_global_probe_candidates,
                        )
                        if self.upper_stage_guidance is not None
                        else self.upper_guidance.build_probe_candidates(
                            task_id=int(request.payload["task_id"]),
                            main_subtask=state.current_subtask,
                            retrieved_candidates=list(
                                guidance_meta.get("lower_global_source_candidates", [])
                            ),
                            center_stage=int(guidance_meta.get("candidate_stage", -1)),
                            max_count=self.upper_guidance_lower_global_probe_candidates,
                        )
                    )
                    if (
                        self.upper_guidance_feedback_mode != "off"
                        or self.upper_guidance_lower_global_mode != "off"
                    )
                    and self.upper_guidance is not None
                    and self.upper_guidance.head.variant == "fusion"
                    else []
                ),
            }
            request.response = response
            request.done.set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8330)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=50.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--merge-distance", type=int, default=6)
    parser.add_argument("--keyframe-max", type=int, default=0)
    parser.add_argument("--upper-guidance-enabled", action="store_true")
    parser.add_argument("--upper-guidance-head", type=Path)
    parser.add_argument("--upper-guidance-manifest", type=Path)
    parser.add_argument("--upper-guidance-features", type=Path)
    parser.add_argument("--upper-guidance-lower-features", type=Path)
    parser.add_argument("--upper-guidance-upper-age", type=Path)
    parser.add_argument("--upper-guidance-top-k", type=int, default=16)
    parser.add_argument("--upper-guidance-temperature", type=float, default=0.07)
    parser.add_argument("--upper-guidance-min-task-purity", type=float, default=0.0)
    parser.add_argument("--upper-guidance-min-subtask-confidence", type=float, default=0.0)
    parser.add_argument("--upper-guidance-interpolation", type=float, default=0.8)
    parser.add_argument("--upper-guidance-bank-per-primitive", type=int, default=16)
    parser.add_argument("--upper-guidance-bank-seed", type=int, default=17)
    parser.add_argument("--upper-guidance-device", type=str, default="cpu")
    parser.add_argument("--upper-guidance-allowed-task-ids", default="")
    parser.add_argument("--upper-guidance-history-enabled", action="store_true")
    parser.add_argument("--upper-guidance-history-lock-observations", type=int, default=2)
    parser.add_argument("--upper-guidance-history-advance-confirmations", type=int, default=2)
    parser.add_argument("--upper-guidance-history-max-rollback", type=float, default=0.10)
    parser.add_argument("--upper-guidance-history-max-advance", type=float, default=0.35)
    parser.add_argument("--upper-guidance-temporal-enabled", action="store_true")
    parser.add_argument("--upper-guidance-temporal-posterior", type=float, default=0.75)
    parser.add_argument("--upper-guidance-temporal-purity", type=float, default=0.35)
    parser.add_argument("--upper-guidance-temporal-evidence-decay", type=float, default=0.95)
    parser.add_argument("--upper-guidance-temporal-advance-evidence", type=float, default=0.45)
    parser.add_argument("--upper-guidance-temporal-same-stage-budget", type=int, default=2)
    parser.add_argument("--upper-guidance-temporal-max-rollback", type=float, default=0.15)
    parser.add_argument("--upper-guidance-temporal-max-advance", type=float, default=0.40)
    parser.add_argument("--upper-guidance-feedback-mode", choices=("off", "shadow", "control"), default="off")
    parser.add_argument("--upper-guidance-feedback-horizon", type=int, default=4)
    parser.add_argument("--upper-guidance-feedback-confirmations", type=int, default=2)
    parser.add_argument("--upper-guidance-feedback-similarity-tolerance", type=float, default=0.02)
    parser.add_argument("--upper-guidance-feedback-purity-tolerance", type=float, default=0.10)
    parser.add_argument("--upper-guidance-feedback-max-reinforcements", type=int, default=2)
    parser.add_argument("--upper-guidance-feedback-lower-rescue-enabled", action="store_true")
    parser.add_argument("--upper-guidance-feedback-lower-rescue-confirmations", type=int, default=2)
    parser.add_argument("--upper-guidance-feedback-lower-rescue-min-margin", type=float, default=0.05)
    parser.add_argument("--upper-guidance-lower-global-mode", choices=("off", "shadow", "control"), default="off")
    parser.add_argument("--upper-guidance-lower-global-top-k", type=int, default=16)
    parser.add_argument("--upper-guidance-lower-global-probe-candidates", type=int, default=4)
    parser.add_argument("--upper-guidance-lower-global-min-posterior", type=float, default=0.35)
    parser.add_argument("--upper-guidance-lower-global-min-score", type=float, default=0.45)
    parser.add_argument("--upper-guidance-lower-global-min-margin", type=float, default=0.03)
    parser.add_argument("--upper-guidance-lower-global-progress-scale", type=float, default=0.15)
    parser.add_argument("--upper-stage-guidance-config", type=Path)
    parser.add_argument("--upper-guidance-shadow-native", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    planner = BatchedUpperPlanner(
        checkpoint=args.checkpoint,
        processor_dir=args.processor_dir or args.checkpoint,
        task_config=args.task_config,
        device=args.device,
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
        max_new_tokens=args.max_new_tokens,
        merge_distance=args.merge_distance,
        keyframe_max=args.keyframe_max,
        upper_guidance_enabled=args.upper_guidance_enabled,
        upper_guidance_head=args.upper_guidance_head,
        upper_guidance_manifest=args.upper_guidance_manifest,
        upper_guidance_features=args.upper_guidance_features,
        upper_guidance_lower_features=args.upper_guidance_lower_features,
        upper_guidance_upper_age=args.upper_guidance_upper_age,
        upper_guidance_top_k=args.upper_guidance_top_k,
        upper_guidance_temperature=args.upper_guidance_temperature,
        upper_guidance_min_task_purity=args.upper_guidance_min_task_purity,
        upper_guidance_min_subtask_confidence=args.upper_guidance_min_subtask_confidence,
        upper_guidance_interpolation=args.upper_guidance_interpolation,
        upper_guidance_bank_per_primitive=args.upper_guidance_bank_per_primitive,
        upper_guidance_bank_seed=args.upper_guidance_bank_seed,
        upper_guidance_device=args.upper_guidance_device,
        upper_guidance_allowed_task_ids=args.upper_guidance_allowed_task_ids,
        upper_guidance_history_enabled=args.upper_guidance_history_enabled,
        upper_guidance_history_lock_observations=args.upper_guidance_history_lock_observations,
        upper_guidance_history_advance_confirmations=args.upper_guidance_history_advance_confirmations,
        upper_guidance_history_max_rollback=args.upper_guidance_history_max_rollback,
        upper_guidance_history_max_advance=args.upper_guidance_history_max_advance,
        upper_guidance_temporal_enabled=args.upper_guidance_temporal_enabled,
        upper_guidance_temporal_posterior=args.upper_guidance_temporal_posterior,
        upper_guidance_temporal_purity=args.upper_guidance_temporal_purity,
        upper_guidance_temporal_evidence_decay=args.upper_guidance_temporal_evidence_decay,
        upper_guidance_temporal_advance_evidence=args.upper_guidance_temporal_advance_evidence,
        upper_guidance_temporal_same_stage_budget=args.upper_guidance_temporal_same_stage_budget,
        upper_guidance_temporal_max_rollback=args.upper_guidance_temporal_max_rollback,
        upper_guidance_temporal_max_advance=args.upper_guidance_temporal_max_advance,
        upper_guidance_feedback_mode=args.upper_guidance_feedback_mode,
        upper_guidance_feedback_horizon=args.upper_guidance_feedback_horizon,
        upper_guidance_feedback_confirmations=args.upper_guidance_feedback_confirmations,
        upper_guidance_feedback_similarity_tolerance=args.upper_guidance_feedback_similarity_tolerance,
        upper_guidance_feedback_purity_tolerance=args.upper_guidance_feedback_purity_tolerance,
        upper_guidance_feedback_max_reinforcements=args.upper_guidance_feedback_max_reinforcements,
        upper_guidance_feedback_lower_rescue_enabled=args.upper_guidance_feedback_lower_rescue_enabled,
        upper_guidance_feedback_lower_rescue_confirmations=args.upper_guidance_feedback_lower_rescue_confirmations,
        upper_guidance_feedback_lower_rescue_min_margin=args.upper_guidance_feedback_lower_rescue_min_margin,
        upper_guidance_lower_global_mode=args.upper_guidance_lower_global_mode,
        upper_guidance_lower_global_top_k=args.upper_guidance_lower_global_top_k,
        upper_guidance_lower_global_probe_candidates=args.upper_guidance_lower_global_probe_candidates,
        upper_guidance_lower_global_min_posterior=args.upper_guidance_lower_global_min_posterior,
        upper_guidance_lower_global_min_score=args.upper_guidance_lower_global_min_score,
        upper_guidance_lower_global_min_margin=args.upper_guidance_lower_global_min_margin,
        upper_guidance_lower_global_progress_scale=args.upper_guidance_lower_global_progress_scale,
        upper_stage_guidance_config=args.upper_stage_guidance_config,
        upper_guidance_shadow_native=args.upper_guidance_shadow_native,
    )
    packer = msgpack_numpy.Packer()

    def handler(connection) -> None:
        connection.send(packer.pack({"service": "predimem-upper", "batch_size": args.batch_size}))
        for message in connection:
            try:
                payload = msgpack_numpy.unpackb(message)
                connection.send(packer.pack(planner.submit(payload)))
            except ConnectionClosed:
                logging.info("Upper client disconnected after request completion")
                return
            except Exception as exc:
                logging.exception("Upper request failed")
                try:
                    connection.send(packer.pack({"error": f"{type(exc).__name__}: {exc}"}))
                except ConnectionClosed:
                    return

    logging.info("PrediMem upper server listening on %s:%d batch=%d", args.host, args.port, args.batch_size)
    with websockets.sync.server.serve(
        handler,
        args.host,
        args.port,
        compression=None,
        max_size=None,
        ping_interval=None,
        ping_timeout=None,
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
