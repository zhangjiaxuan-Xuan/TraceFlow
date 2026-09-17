#!/usr/bin/env python3
"""Fit a local temporal-gate surrogate from paired online kNN-LM runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TASKS = (6, 7, 8, 9, 10, 15, 16)
FEATURE_NAMES = (
    "admission_rate",
    "early_pressure",
    "stall_pressure",
    "same_stage_pressure",
    "repeat_pressure",
    "advance_rate",
)
MODEL_FEATURE_NAMES = (
    "admission_rate",
    "early_pressure",
    "same_stage_pressure",
    "repeat_pressure",
    "advance_rate",
)
MODEL_FEATURE_INDICES = tuple(FEATURE_NAMES.index(name) for name in MODEL_FEATURE_NAMES)


@dataclass(frozen=True)
class GateParams:
    posterior: float
    purity: float
    evidence_decay: float
    advance_evidence: float
    same_stage_budget: int
    max_rollback: float
    max_advance: float


@dataclass
class Episode:
    task_id: int
    seed: int
    success: float
    calls: list[dict[str, Any]]


def normalize(text: str) -> str:
    return " ".join(str(text).replace("_", " ").replace("-", " ").lower().split())


def primitive(row: dict[str, Any]) -> str:
    stage = int(row.get("stage_index", -1))
    paths = row.get("segment_paths", [])
    if isinstance(paths, list) and 0 <= stage < len(paths):
        stem = Path(str(paths[stage])).stem
        stem = re.sub(r"_\d+_seed\d+_task\d+$", "", stem)
        if normalized := normalize(stem):
            return normalized
    return normalize(row.get("prompt", ""))


def load_stage_map(manifest: Path) -> tuple[dict[tuple[int, str], int], dict[tuple[int, str], float]]:
    stages: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    progresses: dict[tuple[int, str], list[float]] = defaultdict(list)
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = int(row.get("task_id", -1))
            label = primitive(row)
            stage = int(row.get("stage_index", -1))
            if task_id not in TASKS or not label or stage < 0:
                continue
            key = (task_id, label)
            stages[key][stage] += 1
            progresses[key].append(float(row.get("anchor_progress", 0.0)))
    stage_map = {key: counts.most_common(1)[0][0] for key, counts in stages.items()}
    progress_map = {key: float(np.median(values)) for key, values in progresses.items()}
    return stage_map, progress_map


def load_episodes(run_root: Path) -> dict[tuple[int, int], Episode]:
    episodes: dict[tuple[int, int], Episode] = {}
    for path in sorted((run_root / "base").glob("task*/worker_*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "seed" not in row or "task_id" not in row:
                    continue
                task_id = int(row["task_id"])
                seed = int(row["seed"])
                if task_id not in TASKS:
                    continue
                episodes[(task_id, seed)] = Episode(
                    task_id=task_id,
                    seed=seed,
                    success=float(row.get("TSR", 0.0)) / 100.0,
                    calls=sorted(row.get("upper_guidance", []), key=lambda item: int(item.get("step_idx", 0))),
                )
    return episodes


def load_base(path: Path) -> dict[tuple[int, int], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (int(row["task_id"]), int(row["seed"])): float(row["TSR"]) / 100.0
            for row in csv.DictReader(stream, delimiter="\t")
            if int(row["task_id"]) in TASKS
        }


def call_stage(
    task_id: int,
    call: dict[str, Any],
    stage_map: dict[tuple[int, str], int],
    field: str,
) -> int | None:
    direct = call.get("candidate_stage") if field == "candidate" else None
    if direct is not None and int(direct) >= 0:
        return int(direct)
    label = normalize(call.get(field, ""))
    return stage_map.get((task_id, label))


def call_progress(
    task_id: int,
    call: dict[str, Any],
    progress_map: dict[tuple[int, str], float],
) -> float | None:
    if call.get("candidate_progress") is not None:
        return float(call["candidate_progress"])
    return progress_map.get((task_id, normalize(call.get("candidate", ""))))


def reference_stages(
    episode: Episode,
    stage_map: dict[tuple[int, str], int],
) -> list[int]:
    """Estimate visual phase from distinct-step native decodes, ignoring guided outputs."""
    current = 0
    pending: int | None = None
    pending_count = 0
    last_native_step: int | None = None
    result = []
    for call in episode.calls:
        step = int(call.get("step_idx", 0))
        if not bool(call.get("applied", False)) and step != last_native_step:
            native = call_stage(episode.task_id, call, stage_map, "generated_subtask")
            last_native_step = step
            if native is not None and native == current + 1:
                if pending == native:
                    pending_count += 1
                else:
                    pending, pending_count = native, 1
                if pending_count >= 2:
                    current = native
                    pending = None
                    pending_count = 0
            elif native == current:
                pending_count = max(pending_count - 1, 0)
            elif native is not None and native > current + 1:
                pending_count = max(pending_count - 1, 0)
        result.append(current)
    return result


def summarize_decisions(
    episode: Episode,
    applied: list[bool],
    stages: list[int | None],
    references: list[int],
    confidences: list[float],
) -> np.ndarray:
    calls = max(len(episode.calls), 1)
    admitted = 0
    early = 0.0
    stall = 0.0
    same = 0.0
    repeat = 0.0
    advances = 0
    previous_stage: int | None = None
    repeat_count = 0
    for use, stage, reference, confidence in zip(applied, stages, references, confidences, strict=True):
        if not use or stage is None:
            continue
        admitted += 1
        early += max(stage - reference, 0) * confidence
        stall += max(reference - stage, 0) * confidence
        same += float(stage == reference) * confidence
        if previous_stage == stage:
            repeat_count += 1
            repeat += repeat_count * confidence
        else:
            if previous_stage is not None and stage > previous_stage:
                advances += 1
            repeat_count = 0
        previous_stage = stage
    return np.asarray(
        [admitted / calls, early / calls, stall / calls, same / calls, repeat / calls, advances / calls],
        dtype=np.float64,
    )


def observed_features(
    episode: Episode,
    stage_map: dict[tuple[int, str], int],
) -> np.ndarray:
    references = reference_stages(episode, stage_map)
    stages = [call_stage(episode.task_id, call, stage_map, "candidate") for call in episode.calls]
    applied = [bool(call.get("applied", False)) for call in episode.calls]
    confidence = [float(call.get("same_task_confidence", 0.0)) for call in episode.calls]
    return summarize_decisions(episode, applied, stages, references, confidence)


def replay_features(
    episode: Episode,
    params: GateParams,
    stage_map: dict[tuple[int, str], int],
    progress_map: dict[tuple[int, str], float],
) -> np.ndarray:
    references = reference_stages(episode, stage_map)
    state_stage = 0
    state_progress = 0.0
    next_evidence = 0.0
    same_used = 0
    previous_candidate: int | None = None
    decisions: list[bool] = []
    stages: list[int | None] = []
    confidences: list[float] = []
    for call in episode.calls:
        stage = call_stage(episode.task_id, call, stage_map, "candidate")
        progress = call_progress(episode.task_id, call, progress_map)
        confidence = float(call.get("same_task_confidence", 0.0))
        purity = float(call.get("same_task_purity", 0.0))
        stages.append(stage)
        confidences.append(confidence)
        next_evidence *= params.evidence_decay
        use = False
        if stage is None or progress is None or confidence < params.posterior:
            decisions.append(False)
            continue
        support = confidence * max(purity, params.purity)
        if stage == state_stage + 1:
            next_evidence += support
        if stage < state_stage or stage > state_stage + 1:
            decisions.append(False)
            continue
        delta = progress - state_progress
        if delta < -params.max_rollback or delta > params.max_advance:
            decisions.append(False)
            continue
        if stage == state_stage:
            if previous_candidate != stage:
                same_used = 0
            use = same_used < params.same_stage_budget and purity >= params.purity
            if use:
                same_used += 1
        elif next_evidence >= params.advance_evidence:
            state_stage = stage
            state_progress = max(state_progress, progress)
            next_evidence = 0.0
            same_used = 0
            use = True
        if use:
            state_progress = max(state_progress, progress)
        previous_candidate = stage
        decisions.append(use)
    return summarize_decisions(episode, decisions, stages, references, confidences)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weight = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return weight, mean, scale


def predict(feature: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weight, mean, scale = model
    z = (feature - mean) / scale
    return weight[0] + z @ weight[1:]


def loocv_diagnostics(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, float]:
    predictions = np.empty_like(y)
    null_predictions = np.empty_like(y)
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        predictions[held_out] = predict(x[held_out], fit_ridge(x[train], y[train], alpha=alpha))
        null_predictions[held_out] = y[train].mean()
    return {
        "loocv_mae": float(np.mean(np.abs(predictions - y))),
        "null_loocv_mae": float(np.mean(np.abs(null_predictions - y))),
        "prediction_correlation": float(np.corrcoef(predictions, y)[0, 1]),
        "sign_accuracy": float(np.mean(np.sign(predictions) == np.sign(y))),
    }


def candidate_grid(limit: int, seed: int) -> list[GateParams]:
    values = list(
        itertools.product(
            (0.75, 0.80, 0.85, 0.90),
            (0.20, 0.35, 0.50),
            (0.50, 0.70, 0.85, 0.95),
            (0.45, 0.70, 1.00, 1.35),
            (1, 2, 4, 8),
            (0.05, 0.10, 0.15),
            (0.20, 0.30, 0.40),
        )
    )
    rng = np.random.default_rng(seed)
    if limit and len(values) > limit:
        selected = np.sort(rng.choice(len(values), size=limit, replace=False))
        values = [values[int(index)] for index in selected]
    return [GateParams(*item) for item in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-episodes", type=Path, required=True)
    parser.add_argument("--stateless-run", type=Path, required=True)
    parser.add_argument("--history-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=768)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    stage_map, progress_map = load_stage_map(args.manifest)
    base = load_base(args.base_episodes)
    sources = {
        "stateless": load_episodes(args.stateless_run),
        "history": load_episodes(args.history_run),
    }
    expected = {(task, seed) for task in TASKS for seed in range(50, 101)}
    if set(base) != expected or any(set(rows) != expected for rows in sources.values()):
        raise RuntimeError("Base/stateless/history must all contain the same 357 task-seed pairs")

    observed_rows = []
    for source, episodes in sources.items():
        for key in sorted(expected):
            episode = episodes[key]
            features = observed_features(episode, stage_map)
            target = episode.success - base[key]
            observed_rows.append(
                {
                    "source": source,
                    "task_id": key[0],
                    "seed": key[1],
                    "base_success": base[key],
                    "guided_success": episode.success,
                    "paired_delta": target,
                    **dict(zip(FEATURE_NAMES, features, strict=True)),
                }
            )
    task_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in observed_rows:
        task_groups[(row["source"], row["task_id"])].append(row)
    observed_task_rows = []
    for (source, task_id), rows in sorted(task_groups.items()):
        observed_task_rows.append(
            {
                "source": source,
                "task_id": task_id,
                "episodes": len(rows),
                "base_success": float(np.mean([row["base_success"] for row in rows])),
                "guided_success": float(np.mean([row["guided_success"] for row in rows])),
                "paired_delta": float(np.mean([row["paired_delta"] for row in rows])),
                **{
                    name: float(np.mean([row[name] for row in rows]))
                    for name in FEATURE_NAMES
                },
            }
        )
    x_all = np.asarray([[row[name] for name in FEATURE_NAMES] for row in observed_task_rows])
    x = x_all[:, MODEL_FEATURE_INDICES]
    y = np.asarray([row["paired_delta"] for row in observed_task_rows])
    ridge_alpha = 1.0
    model = fit_ridge(x, y, alpha=ridge_alpha)
    fitted = predict(x, model)
    fit_summary = {
        "level": "task_endpoint",
        "samples": len(y),
        "ridge_alpha": ridge_alpha,
        "mae": float(np.mean(np.abs(fitted - y))),
        "null_mae": float(np.mean(np.abs(y - y.mean()))),
        **loocv_diagnostics(x, y, alpha=ridge_alpha),
        "correlations": {
            name: float(np.corrcoef(x[:, index], y)[0, 1])
            for index, name in enumerate(MODEL_FEATURE_NAMES)
        },
        "coefficients_standardized": {
            name: float(model[0][index + 1]) for index, name in enumerate(MODEL_FEATURE_NAMES)
        },
    }

    candidate_rows = []
    for params in candidate_grid(args.candidates, args.seed):
        source_predictions = {}
        source_features = {}
        task_predictions: dict[int, list[float]] = defaultdict(list)
        for source, episodes in sources.items():
            episode_features = {
                key: replay_features(episodes[key], params, stage_map, progress_map)
                for key in sorted(expected)
            }
            task_features = {
                task: np.mean(
                    [feature for key, feature in episode_features.items() if key[0] == task],
                    axis=0,
                )
                for task in TASKS
            }
            prediction = {
                task: float(predict(np.take(feature, MODEL_FEATURE_INDICES), model))
                for task, feature in task_features.items()
            }
            source_predictions[source] = float(np.mean(list(prediction.values())))
            source_features[source] = np.mean(list(task_features.values()), axis=0)
            for task, value in prediction.items():
                task_predictions[task].append(value)
        mean_prediction = float(np.mean(list(source_predictions.values())))
        source_uncertainty = float(abs(source_predictions["stateless"] - source_predictions["history"]))
        task_means = {task: float(np.mean(values)) for task, values in task_predictions.items()}
        worst_task = min(task_means.values())
        score = mean_prediction - 0.5 * source_uncertainty + 0.15 * worst_task
        mean_features = np.mean(np.stack(list(source_features.values())), axis=0)
        candidate_rows.append(
            {
                **params.__dict__,
                "score": score,
                "predicted_delta": mean_prediction,
                "source_uncertainty": source_uncertainty,
                "worst_task_delta": worst_task,
                "stateless_trace_prediction": source_predictions["stateless"],
                "history_trace_prediction": source_predictions["history"],
                **{f"mean_{name}": float(value) for name, value in zip(FEATURE_NAMES, mean_features, strict=True)},
                **{f"task{task}_predicted_delta": task_means[task] for task in TASKS},
            }
        )
    candidate_rows.sort(key=lambda row: row["score"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("observed_episode_metrics.csv", observed_rows),
        ("observed_task_metrics.csv", observed_task_rows),
        ("candidate_ranking.csv", candidate_rows),
    ):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    top = candidate_rows[:20]
    payload = {
        "protocol": "upper_knnlm_temporal_local_surrogate_v1",
        "evidence": {
            "paired_episodes": len(observed_rows),
            "tasks": list(TASKS),
            "sources": list(sources),
            "candidate_count": len(candidate_rows),
            "note": (
                "Task-level local online-trace surrogate. Predictions only rank nearby short-test "
                "candidates; they are not SR claims and must be confirmed online."
            ),
        },
        "fit": fit_summary,
        "top_candidates": top,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = np.where(
        np.asarray([row["source"] for row in observed_task_rows]) == "stateless",
        "#0072B2",
        "#D55E00",
    )
    axes[0].scatter(x[:, 1], y, c=colors, alpha=0.55, s=20)
    axes[0].set(xlabel="Early pressure", ylabel="Paired success delta", title="Observed endpoint pressure")
    axes[0].axhline(0, color="#555555", linewidth=1)
    axes[1].scatter(x[:, 3], y, c=colors, alpha=0.55, s=20)
    axes[1].set(xlabel="Repeat pressure", ylabel="Paired success delta", title="Observed persistence pressure")
    axes[1].axhline(0, color="#555555", linewidth=1)
    fig.tight_layout()
    fig.savefig(args.output_dir / "temporal_pressure_endpoints.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "fit": fit_summary, "top_candidates": top[:5]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
