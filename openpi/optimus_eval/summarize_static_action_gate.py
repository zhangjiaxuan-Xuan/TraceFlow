from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(run_root: Path) -> dict:
    stats_path = run_root / "static_action_batches.jsonl"
    episodes_path = run_root / "episodes.tsv"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing static-action statistics: {stats_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing episode summary: {episodes_path}")

    rows = []
    with stats_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            batch = json.loads(line)
            if int(batch.get("schema_version", -1)) != 1:
                raise ValueError(f"Unsupported static-action schema at line {line_number}")
            rows.extend(batch.get("rows", []))

    episodes = {}
    with episodes_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            key = (int(row["task_id"]), int(row["episode"]))
            episodes[key] = {"TSR": float(row["TSR"]), "CSR": float(row["CSR"])}

    grouped: defaultdict[int, list[dict]] = defaultdict(list)
    gated_episode_keys = set()
    for row in rows:
        task_id = int(row["task_id"])
        grouped[task_id].append(row)
        if bool(row.get("gate_applied", False)):
            gated_episode_keys.add((task_id, int(row["episode_idx"])))

    per_task = []
    for task_id in sorted(grouped):
        task_rows = grouped[task_id]
        task_episode_keys = sorted(key for key in episodes if key[0] == task_id)
        gated_keys = [key for key in task_episode_keys if key in gated_episode_keys]
        ungated_keys = [key for key in task_episode_keys if key not in gated_episode_keys]
        per_task.append(
            {
                "task_id": task_id,
                "calls": len(task_rows),
                "static_candidate_calls": sum(bool(row.get("gate", False)) for row in task_rows),
                "gated_calls": sum(bool(row.get("gate_applied", False)) for row in task_rows),
                "gated_call_fraction": _mean(
                    [float(bool(row.get("gate_applied", False))) for row in task_rows]
                ),
                "mean_weighted_static_fraction": _mean(
                    [float(row["weighted_static_fraction"]) for row in task_rows]
                ),
                "episodes": len(task_episode_keys),
                "episodes_with_gate": len(gated_keys),
                "TSR": _mean([episodes[key]["TSR"] for key in task_episode_keys]),
                "CSR": _mean([episodes[key]["CSR"] for key in task_episode_keys]),
                "TSR_with_gate": _mean([episodes[key]["TSR"] for key in gated_keys]),
                "CSR_with_gate": _mean([episodes[key]["CSR"] for key in gated_keys]),
                "TSR_without_gate": _mean([episodes[key]["TSR"] for key in ungated_keys]),
                "CSR_without_gate": _mean([episodes[key]["CSR"] for key in ungated_keys]),
            }
        )

    return {
        "schema_version": 1,
        "run_root": str(run_root.resolve()),
        "mode": rows[0].get("mode", "unknown") if rows else "unknown",
        "calls": len(rows),
        "static_candidate_calls": sum(bool(row.get("gate", False)) for row in rows),
        "gated_calls": sum(bool(row.get("gate_applied", False)) for row in rows),
        "episodes": len(episodes),
        "episodes_with_gate": len(gated_episode_keys & set(episodes)),
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    payload = summarize(args.run_root)
    output_path = args.run_root / "static_action_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("task\tcalls\tstatic_candidates\tgated\tgate_rate\tepisodes_with_gate\tTSR\tCSR")
    for row in payload["per_task"]:
        print(
            f"{row['task_id']}\t{row['calls']}\t{row['static_candidate_calls']}\t"
            f"{row['gated_calls']}\t{100.0 * row['gated_call_fraction']:.2f}\t"
            f"{row['episodes_with_gate']}\t{row['TSR']:.2f}\t{row['CSR']:.2f}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
