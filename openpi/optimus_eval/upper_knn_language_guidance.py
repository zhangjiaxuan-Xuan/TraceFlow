from __future__ import annotations

from collections import defaultdict
import json
from typing import Sequence

import numpy as np
import torch
from transformers import LogitsProcessor


class NumpyInnerProductIndex:
    """Small exact inner-product index used by the compact language bank."""

    def __init__(self, keys: np.ndarray) -> None:
        values = np.asarray(keys, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"Expected non-empty rank-2 keys, got {values.shape}")
        self.keys = np.ascontiguousarray(values)
        self.ntotal = int(values.shape[0])

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(queries, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.keys.shape[1]:
            raise ValueError(f"Query/key mismatch: query={values.shape} keys={self.keys.shape}")
        count = min(max(int(top_k), 0), self.ntotal)
        if count == 0:
            shape = (values.shape[0], 0)
            return np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.int64)
        similarities = values @ self.keys.T
        if count == self.ntotal:
            indices = np.argsort(-similarities, axis=1)[:, :count]
        else:
            indices = np.argpartition(-similarities, count - 1, axis=1)[:, :count]
            selected = np.take_along_axis(similarities, indices, axis=1)
            order = np.argsort(-selected, axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
        scores = np.take_along_axis(similarities, indices, axis=1)
        return scores.astype(np.float32, copy=False), indices.astype(np.int64, copy=False)


def subtask_prefix(subtask: str) -> str:
    encoded = json.dumps({"current_primitive": str(subtask)}, ensure_ascii=False)
    return encoded[:-1] + ', "keyframe_positions": ['


def aggregate_subtask_candidates(
    subtasks: Sequence[str],
    scores: np.ndarray,
    *,
    temperature: float,
) -> list[tuple[str, float]]:
    if len(subtasks) != len(scores):
        raise ValueError("Subtasks and scores must have equal length")
    if not subtasks:
        return []
    values = np.asarray(scores, dtype=np.float64)
    weights = np.exp((values - values.max()) / max(float(temperature), 1e-6))
    totals: dict[str, float] = defaultdict(float)
    display: dict[str, str] = {}
    for subtask, weight in zip(subtasks, weights, strict=True):
        normalized = " ".join(str(subtask).lower().strip().split())
        if not normalized:
            continue
        totals[normalized] += float(weight)
        display.setdefault(normalized, " ".join(str(subtask).strip().split()))
    denominator = sum(totals.values())
    if denominator <= 0:
        return []
    return [
        (display[key], value / denominator)
        for key, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


class KnnSubtaskLogitsProcessor(LogitsProcessor):
    """Interpolate Qwen logits with per-sample retrieved subtask token prefixes."""

    def __init__(
        self,
        tokenizer,
        candidates: Sequence[Sequence[tuple[str, float]]],
        *,
        prompt_width: int,
        interpolation: float | Sequence[float],
    ) -> None:
        self.prompt_width = int(prompt_width)
        if isinstance(interpolation, (float, int)):
            self.interpolations = [float(interpolation)] * len(candidates)
        else:
            self.interpolations = [float(value) for value in interpolation]
        if len(self.interpolations) != len(candidates):
            raise ValueError("Per-sample interpolation length must match candidates")
        if any(not 0.0 <= value <= 1.0 for value in self.interpolations):
            raise ValueError("interpolation must be in [0, 1]")
        self.prefixes: list[list[tuple[list[int], float]]] = []
        for sample in candidates:
            encoded = []
            for subtask, weight in sample:
                tokens = tokenizer.encode(subtask_prefix(subtask), add_special_tokens=False)
                if tokens and weight > 0:
                    encoded.append(([int(token) for token in tokens], float(weight)))
            total = sum(weight for _, weight in encoded)
            self.prefixes.append(
                [(tokens, weight / total) for tokens, weight in encoded] if total > 0 else []
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != len(self.prefixes):
            raise ValueError(f"Batch mismatch: logits={scores.shape[0]} candidates={len(self.prefixes)}")
        if not any(self.interpolations):
            return scores
        output = scores.clone()
        for row, (candidates, interpolation) in enumerate(zip(self.prefixes, self.interpolations, strict=True)):
            if not candidates or interpolation <= 0:
                continue
            generated = input_ids[row, self.prompt_width :].tolist()
            alive = [
                (tokens, weight)
                for tokens, weight in candidates
                if len(generated) < len(tokens) and tokens[: len(generated)] == generated
            ]
            if not alive:
                continue
            memory = defaultdict(float)
            alive_total = sum(weight for _, weight in alive)
            for tokens, weight in alive:
                memory[tokens[len(generated)]] += weight / alive_total
            base = torch.softmax(scores[row].float(), dim=-1) * (1.0 - interpolation)
            for token, probability in memory.items():
                base[token] += interpolation * probability
            output[row] = torch.log(base.clamp_min(torch.finfo(base.dtype).tiny)).to(output.dtype)
        return output
