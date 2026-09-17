from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_records(task_dir: Path) -> list[dict[str, Any]]:
    by_episode: dict[int, dict[str, Any]] = {}
    for path in sorted(task_dir.glob("worker_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            episode = int(record["episode"])
            if episode in by_episode and by_episode[episode] != record:
                raise RuntimeError(f"Conflicting records: {task_dir}, episode={episode}")
            by_episode[episode] = record
    return [by_episode[index] for index in sorted(by_episode)]


def _timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        sample
        for record in records
        for sample in record.get("action_generation_timing", [])
    ]
    fields = (
        "policy_batch_ms",
        "policy_per_sample_ms",
        "server_infer_ms",
        "client_roundtrip_ms",
        "batch_size",
    )
    output: dict[str, Any] = {"samples": len(samples)}
    for field in fields:
        values = np.asarray([float(sample[field]) for sample in samples], dtype=np.float64)
        if len(values) == 0:
            continue
        output[field] = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def _stats(values: list[float]) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _component_timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    upper = [sample for record in records for sample in record.get("upper_timing", [])]
    lower = [sample for record in records for sample in record.get("lower_timing", [])]
    output: dict[str, Any] = {"upper_samples": len(upper), "lower_samples": len(lower)}
    for field in ("batch_ms", "per_sample_ms", "roundtrip_ms"):
        value = _stats([float(sample.get(field, float("nan"))) for sample in upper])
        if value is not None:
            output[f"upper_{field}"] = value
    for field in ("policy_batch_ms", "server_infer_ms", "roundtrip_ms"):
        value = _stats([float(sample.get(field, float("nan"))) for sample in lower])
        if value is not None:
            output[f"lower_{field}"] = value
    lower_per_sample = [
        float(sample.get("policy_batch_ms", float("nan"))) / max(1, int(sample.get("batch_size", 1)))
        for sample in lower
    ]
    value = _stats(lower_per_sample)
    if value is not None:
        output["lower_policy_per_sample_ms"] = value
    upper_compute = [float(sample.get("per_sample_ms", float("nan"))) for sample in upper]
    finite_upper = [value for value in upper_compute if np.isfinite(value)]
    finite_lower = [value for value in lower_per_sample if np.isfinite(value)]
    if finite_lower:
        output["upper_calls_per_lower_call"] = float(len(finite_upper) / len(finite_lower))
        output["amortized_full_stack_per_lower_call_ms"] = float(
            np.mean(finite_lower) + sum(finite_upper) / len(finite_lower)
        )
    for field in ("prefix_ms", "base_denoise_ms", "base_vla_ms", "retrieval_ms", "guidance_ms"):
        batch_values = [
            float(sample.get("components", {}).get(field, float("nan"))) for sample in lower
        ]
        value = _stats(batch_values)
        if value is not None:
            output[f"lower_{field}"] = value
        per_sample_values = [
            raw / max(1, int(sample.get("batch_size", 1)))
            for raw, sample in zip(batch_values, lower, strict=True)
        ]
        value = _stats(per_sample_values)
        if value is not None:
            output[f"lower_{field.removesuffix('_ms')}_per_sample_ms"] = value
    base = output.get("lower_base_vla_per_sample_ms", {}).get("mean")
    retrieval = output.get("lower_retrieval_per_sample_ms", {}).get("mean")
    guidance = output.get("lower_guidance_per_sample_ms", {}).get("mean")
    if base is not None and retrieval is not None and guidance is not None:
        output["memory_overhead_per_sample_ms"] = float(retrieval + guidance)
        output["memory_overhead_vs_base_percent"] = float(100.0 * (retrieval + guidance) / max(base, 1e-12))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--task-ids", default="1,2,3,18,19,22,25,26")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    args = parser.parse_args()

    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    all_records = []
    summaries = []
    incomplete = []
    for task_id in task_ids:
        records = _load_records(args.run_root / f"task{task_id}")
        if len(records) != args.episodes_per_task:
            incomplete.append(f"task{task_id}={len(records)}/{args.episodes_per_task}")
        summaries.append(
            {
                "task_id": task_id,
                "episodes": len(records),
                "TSR": sum(float(item["TSR"]) for item in records) / max(1, len(records)),
                "CSR": sum(float(item["CSR"]) for item in records) / max(1, len(records)),
            }
        )
        all_records.extend(records)
    if incomplete:
        raise RuntimeError("Incomplete evaluation: " + ", ".join(incomplete))

    config_path = args.run_root / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    timing = _timing_summary(all_records)
    component_timing = _component_timing_summary(all_records)
    aggregate = {
        "protocol": config.get("protocol", "RoboMemArena-unspecified"),
        "checkpoint": config.get("checkpoint"),
        "task_ids": task_ids,
        "num_tasks": len(task_ids),
        "num_episodes": len(all_records),
        "TSR": sum(float(item["TSR"]) for item in all_records) / max(1, len(all_records)),
        "CSR": sum(float(item["CSR"]) for item in all_records) / max(1, len(all_records)),
        "tasks": summaries,
        "action_generation_timing": timing,
        "component_timing": component_timing,
    }
    (args.run_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (args.run_root / "action_generation_timing.json").write_text(
        json.dumps(timing, indent=2) + "\n"
    )
    (args.run_root / "component_timing.json").write_text(
        json.dumps(component_timing, indent=2) + "\n"
    )

    with (args.run_root / "task_summary.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task_id", "episodes", "TSR", "CSR"), delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)
    with (args.run_root / "episodes.tsv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("task_id", "episode", "seed", "TSR", "CSR", "failure_reason", "stage_done")
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for record in all_records:
            row = dict(record)
            row["stage_done"] = json.dumps(row.get("stage_done", {}), sort_keys=True)
            writer.writerow(row)

    print("task\tepisodes\tTSR\tCSR")
    for item in summaries:
        print(f"{item['task_id']}\t{item['episodes']}\t{item['TSR']:.2f}\t{item['CSR']:.2f}")
    print(f"ALL\t{len(all_records)}\t{aggregate['TSR']:.2f}\t{aggregate['CSR']:.2f}")
    if timing["samples"]:
        policy = timing["policy_per_sample_ms"]
        print(
            f"ACTION_TIME\tsamples={timing['samples']}\t"
            f"policy_per_sample_ms={policy['mean']:.2f}\tP95={policy['p95']:.2f}"
        )


if __name__ == "__main__":
    main()
