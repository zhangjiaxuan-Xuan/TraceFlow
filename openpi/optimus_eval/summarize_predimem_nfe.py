from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def _completed_episodes(run_root: Path, task_ids: list[int]) -> set[tuple[int, int]]:
    completed: set[tuple[int, int]] = set()
    for task_id in task_ids:
        for path in sorted((run_root / f"task{task_id}").glob("worker_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    completed.add((task_id, int(record["episode"])))
    return completed


def _load_batches(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--task-ids", default="1,2,3,18,19,22,25,26")
    parser.add_argument("--stats-file", type=Path)
    args = parser.parse_args()
    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    stats_path = args.stats_file or (args.run_root / "nfe_stats.jsonl")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing NFE statistics: {stats_path}")

    completed = _completed_episodes(args.run_root, task_ids)
    batches = _load_batches(stats_path)
    # Resume can repeat an unfinished call. Keep the latest observation for each
    # completed task/episode/call key, independent of which env slot resumed it.
    latest: dict[tuple[int, int, int], tuple[int, int, dict[str, Any]]] = {}
    for batch_index, batch in enumerate(batches):
        contexts = batch.get("contexts", [])
        logical = batch.get("logical_nfe", [])
        if len(contexts) != len(logical):
            raise RuntimeError(f"Malformed NFE batch {batch_index}: contexts={len(contexts)} nfe={len(logical)}")
        for row_index, (context, nfe) in enumerate(zip(contexts, logical, strict=True)):
            key = (
                int(context.get("task_id", -1)),
                int(context.get("episode_idx", -1)),
                int(context.get("inference_call", -1)),
            )
            if key[:2] in completed:
                latest[key] = (batch_index, row_index, {"context": context, "nfe": int(nfe)})

    selected_by_batch: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for batch_index, row_index, row in latest.values():
        selected_by_batch.setdefault(batch_index, []).append((row_index, row))
    logical_values = [row["nfe"] for rows in selected_by_batch.values() for _, row in rows]
    if not logical_values:
        raise RuntimeError("No completed-episode NFE records were found")

    baseline = max(int(batch.get("fixed_nfe_baseline", 10)) for batch in batches)
    compact = all(str(batch.get("sampler", "")).startswith("v3_re_") for batch in batches)
    logical_mean = sum(logical_values) / len(logical_values)
    actual_slots = 0
    infer_ms = 0.0
    observed_batches = 0
    batch_max_hist: Counter[int] = Counter()
    mask_completion_ms: list[float] = []
    amortized_compute_ms: list[float] = []
    for batch_index, indexed_rows in selected_by_batch.items():
        rows = [row for _, row in indexed_rows]
        batch_max = max(row["nfe"] for row in rows)
        actual_slots += sum(row["nfe"] for row in rows) if compact else len(rows) * batch_max
        batch_max_hist[batch_max] += 1
        timing = float(batches[batch_index].get("infer_ms", float("nan")))
        if math.isfinite(timing):
            infer_ms += timing
        step_ms = batches[batch_index].get("denoise_step_ms")
        active_sizes = batches[batch_index].get("active_batch_sizes")
        if (
            compact
            and math.isfinite(timing)
            and isinstance(step_ms, list)
            and isinstance(active_sizes, list)
            and len(step_ms) == len(active_sizes)
            and all(float(value) >= 0.0 for value in step_ms)
            and all(int(value) > 0 for value in active_sizes)
        ):
            step_values = [float(value) for value in step_ms]
            active_values = [int(value) for value in active_sizes]
            batch_size = len(batches[batch_index].get("contexts", []))
            overhead_ms = max(0.0, timing - sum(step_values))
            for _, row in indexed_rows:
                row_nfe = int(row["nfe"])
                mask_completion_ms.append(overhead_ms + sum(step_values[:row_nfe]))
                amortized_compute_ms.append(
                    overhead_ms / max(1, batch_size)
                    + sum(
                        step_values[index] / active_values[index]
                        for index in range(min(row_nfe, len(step_values)))
                    )
                )
        observed_batches += 1
    compute_equivalent_mean = actual_slots / len(logical_values)

    def optional_values(key: str) -> list[float]:
        values = []
        for batch_index, indexed_rows in selected_by_batch.items():
            source = batches[batch_index].get(key)
            if not isinstance(source, list) or len(source) != len(batches[batch_index].get("contexts", [])):
                return []
            values.extend(float(source[row_index]) for row_index, _ in indexed_rows)
        return values

    raw_nfe = optional_values("raw_nfe_continuous")
    effective_nfe = optional_values("effective_nfe_continuous")
    prior_noise = optional_values("prior_noise_scale")
    summary = {
        "sampler": str(batches[-1].get("sampler", "unknown")),
        "completed_episodes": len(completed),
        "inference_rows": len(logical_values),
        "batches": observed_batches,
        "fixed_nfe_baseline": baseline,
        "logical_mean_nfe": logical_mean,
        "batch_compute_equivalent_mean_nfe": compute_equivalent_mean,
        "logical_nfe_saving_pct": 100.0 * (1.0 - logical_mean / baseline),
        "realized_denoise_compute_saving_pct": 100.0 * (1.0 - compute_equivalent_mean / baseline),
        "observed_model_infer_seconds": infer_ms / 1000.0,
        "mean_batch_infer_ms": infer_ms / max(1, observed_batches),
        "logical_nfe_histogram": dict(sorted(Counter(logical_values).items())),
        "batch_max_nfe_histogram": dict(sorted(batch_max_hist.items())),
    }
    if raw_nfe and effective_nfe and prior_noise:
        summary.update(
            {
                "raw_continuous_mean_nfe": sum(raw_nfe) / len(raw_nfe),
                "effective_continuous_mean_nfe": sum(effective_nfe) / len(effective_nfe),
                "nfe_floor_hit_rate": sum(
                    raw < effective - 1e-12 for raw, effective in zip(raw_nfe, effective_nfe, strict=True)
                )
                / len(raw_nfe),
                "prior_noise_scale_mean": sum(prior_noise) / len(prior_noise),
                "prior_noise_scale_min": min(prior_noise),
                "prior_noise_scale_max": max(prior_noise),
            }
        )
    if mask_completion_ms and amortized_compute_ms:
        summary["mask_aware_timing"] = {
            "definition": {
                "logical_completion_ms": (
                    "shared pre-denoise overhead plus step wall time until this row exits the active mask"
                ),
                "amortized_compute_ms": (
                    "shared overhead divided by batch size plus each active step divided by active rows"
                ),
            },
            "rows": len(mask_completion_ms),
            "logical_completion_ms": _distribution(mask_completion_ms),
            "amortized_compute_ms": _distribution(amortized_compute_ms),
        }
    (args.run_root / "nfe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"NFE rows: {summary['inference_rows']} across {summary['batches']} batches")
    print(f"Logical mean NFE: {logical_mean:.4f} / {baseline}")
    print(f"Batch compute-equivalent mean NFE: {compute_equivalent_mean:.4f} / {baseline}")
    print(f"Logical NFE saving: {summary['logical_nfe_saving_pct']:.2f}%")
    print(f"Realized denoise compute saving: {summary['realized_denoise_compute_saving_pct']:.2f}%")
    print(f"Observed model inference time: {summary['observed_model_infer_seconds']:.2f}s")
    print(f"Mean model time per batch: {summary['mean_batch_infer_ms']:.2f}ms")
    if "mask_aware_timing" in summary:
        mask_timing = summary["mask_aware_timing"]
        print(
            "Mask-aware row time: "
            f"completion={mask_timing['logical_completion_ms']['mean']:.2f}ms "
            f"amortized_compute={mask_timing['amortized_compute_ms']['mean']:.2f}ms"
        )


if __name__ == "__main__":
    main()
