from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "openpi-client" / "src"))

from optimus_eval.predimem_upper_server import _load_upper_guidance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Upper and Fusion language retrieval on online traces.")
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--upper-head", type=Path, required=True)
    parser.add_argument("--fusion-head", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--upper-features", type=Path, required=True)
    parser.add_argument("--lower-features", type=Path, required=True)
    parser.add_argument("--upper-age", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--bank-per-primitive", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_samples(path: Path, max_samples: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trace_by_call = {
        (str(row.get("environment_id", "")), int(row.get("episode_idx", -1)), int(row.get("inference_call", -1))): row
        for row in rows
    }
    if len(rows) > max_samples:
        indices = np.linspace(0, len(rows) - 1, max_samples, dtype=np.int64)
        rows = [rows[int(index)] for index in indices]
    samples = []
    for row in rows:
        trace = path.parent / str(row["trace"])
        with np.load(trace, allow_pickle=False) as payload:
            upper = np.asarray(payload["upper_retrieval_feature"], dtype=np.float32)
            lower = np.asarray(payload["lower_retrieval_feature"], dtype=np.float32)
            upper_age = float(np.asarray(payload["upper_vlm_age"]).item())
        previous_row = trace_by_call.get(
            (
                str(row.get("environment_id", "")),
                int(row.get("episode_idx", -1)),
                int(row.get("inference_call", -1)) - 1,
            )
        )
        previous_lower = None
        if previous_row is not None:
            previous_trace = path.parent / str(previous_row["trace"])
            with np.load(previous_trace, allow_pickle=False) as previous_payload:
                previous_lower = np.asarray(previous_payload["lower_retrieval_feature"], dtype=np.float32)
        if upper.ndim != 1 or lower.ndim != 1 or not np.isfinite(upper).all() or not np.isfinite(lower).all():
            continue
        samples.append(
            {
                "task_id": int(row["task_id"]),
                "progress": float(row.get("progress", 0.0)),
                "upper": upper,
                "lower": lower,
                "previous_lower": previous_lower,
                "upper_age": upper_age,
            }
        )
    return samples


def summarize(items: list[dict]) -> dict:
    if not items:
        return {"queries": 0, "candidate_coverage": 0.0, "recommended_gate": None}
    valid = [item for item in items if item.get("candidate_progress") is not None]
    progress_errors = [abs(float(item["candidate_progress"]) - float(item["progress"])) for item in valid]
    summary = {
        "queries": len(items),
        "candidate_coverage": len(valid) / max(len(items), 1),
        "same_task_purity_mean": float(np.mean([item["purity"] for item in items])),
        "same_task_purity_p10": float(np.quantile([item["purity"] for item in items], 0.10)),
        "candidate_confidence_mean": float(np.mean([item["confidence"] for item in items])),
        "progress_mae": float(np.mean(progress_errors)) if progress_errors else None,
        "progress_within_010": float(np.mean(np.asarray(progress_errors) <= 0.10)) if progress_errors else None,
        "progress_within_020": float(np.mean(np.asarray(progress_errors) <= 0.20)) if progress_errors else None,
    }
    gates = []
    for purity in (0.25, 0.35, 0.50, 0.65, 0.75):
        for confidence in (0.50, 0.60, 0.70, 0.75, 0.80):
            accepted = [
                item
                for item in valid
                if item["purity"] >= purity and item["confidence"] >= confidence
            ]
            if not accepted:
                continue
            good = [abs(item["candidate_progress"] - item["progress"]) <= 0.20 for item in accepted]
            gates.append(
                {
                    "min_purity": purity,
                    "min_confidence": confidence,
                    "coverage": len(accepted) / max(len(items), 1),
                    "progress_precision_020": float(np.mean(good)),
                    "score": float(np.mean(good)) * (len(accepted) / max(len(items), 1)) ** 0.5,
                }
            )
    summary["recommended_gate"] = max(gates, key=lambda item: item["score"]) if gates else None
    return summary


def evaluate(guidance, samples: list[dict], batch_size: int, fusion_mode: str) -> list[dict]:
    output = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        retrieved = guidance.retrieve(
            raw_features=np.stack([item["upper"] for item in batch]),
            task_ids=np.asarray([item["task_id"] for item in batch], dtype=np.int64),
            lower_features=(
                [item["lower"] for item in batch]
                if fusion_mode == "recorded"
                else [item["previous_lower"] for item in batch]
                if fusion_mode == "causal"
                else None
            ),
            upper_ages=(
                np.asarray([item["upper_age"] for item in batch], dtype=np.float32)
                if fusion_mode == "recorded"
                else None
            ),
        )
        for sample, (_, _, meta) in zip(batch, retrieved, strict=True):
            output.append(
                {
                    "task_id": sample["task_id"],
                    "progress": sample["progress"],
                    "purity": float(meta.get("same_task_purity", 0.0)),
                    "confidence": float(meta.get("same_task_confidence", 0.0)),
                    "candidate_progress": meta.get("candidate_progress"),
                    "candidate_stage": meta.get("candidate_stage"),
                    "reason": meta.get("reason", "accepted"),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    samples = load_samples(args.trace_manifest, args.max_samples)
    if not samples:
        raise RuntimeError("No valid online trace samples found")
    common = dict(
        manifest_path=args.bank_manifest,
        upper_features=args.upper_features,
        top_k=args.top_k,
        temperature=args.temperature,
        min_task_purity=0.0,
        min_subtask_confidence=0.0,
        interpolation=0.8,
        bank_per_primitive=args.bank_per_primitive,
        bank_seed=17,
        device=args.device,
    )
    (upper,) = _load_upper_guidance(
        head_path=args.upper_head,
        lower_features=None,
        upper_age=None,
        **common,
    )
    (fusion,) = _load_upper_guidance(
        head_path=args.fusion_head,
        lower_features=args.lower_features,
        upper_age=args.upper_age,
        **common,
    )
    upper_rows = evaluate(upper, samples, args.batch_size, fusion_mode="upper")
    fusion_rows = evaluate(fusion, samples, args.batch_size, fusion_mode="recorded")
    causal_rows = evaluate(fusion, samples, args.batch_size, fusion_mode="causal")
    fresh_samples = [sample for sample in samples if sample["upper_age"] <= 0.0 and sample["previous_lower"] is not None]
    causal_fresh_rows = evaluate(fusion, fresh_samples, args.batch_size, fusion_mode="causal")
    report = {
        "trace_manifest": str(args.trace_manifest),
        "sample_count": len(samples),
        "upper": summarize(upper_rows),
        "fusion": summarize(fusion_rows),
        "fusion_causal_previous_lower": summarize(causal_rows),
        "fusion_causal_fresh_upper": summarize(causal_fresh_rows),
    }
    report["fusion_minus_upper"] = {
        key: report["fusion"][key] - report["upper"][key]
        for key in ("candidate_coverage", "same_task_purity_mean", "candidate_confidence_mean", "progress_within_020")
        if report["fusion"][key] is not None and report["upper"][key] is not None
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
