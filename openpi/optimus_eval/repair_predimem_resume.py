from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimus_eval.predimem_dual_worker import _atomic_rewrite_jsonl
from optimus_eval.predimem_dual_worker import _prune_pending_memory_traces


DERIVED_FILES = (
    "action_generation_timing.json",
    "aggregate.json",
    "episodes.tsv",
    "results.txt",
    "task_summary.tsv",
    "memory_records/index.jsonl",
    "memory_records/failure_index.jsonl",
    "memory_records/success_index.jsonl",
    "memory_records/summary.json",
)


def _parse_episodes(value: str) -> set[tuple[int, int]]:
    selected = set()
    for group in value.split(";"):
        if not group.strip():
            continue
        task_text, episodes_text = group.split(":", maxsplit=1)
        task_id = int(task_text)
        selected.update((task_id, int(item)) for item in episodes_text.split(",") if item)
    if not selected:
        raise ValueError("No task episodes selected")
    return selected


def repair(run_root: Path, selected: set[tuple[int, int]], *, apply: bool) -> dict[str, int]:
    outcome_rows = 0
    affected_files: list[tuple[Path, list[dict], int]] = []
    for task_id in sorted({task_id for task_id, _ in selected}):
        for path in sorted((run_root / f"task{task_id}").glob("worker_*.jsonl")):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept = [row for row in rows if (int(row["task_id"]), int(row["episode"])) not in selected]
            removed = len(rows) - len(kept)
            if removed:
                outcome_rows += removed
                affected_files.append((path, kept, removed))
    if apply:
        for path, kept, _ in affected_files:
            _atomic_rewrite_jsonl(path, kept)
        trace_files = _prune_pending_memory_traces(run_root, sorted(selected))
        derived_removed = 0
        for relative in DERIVED_FILES:
            path = run_root / relative
            if path.is_file():
                path.unlink()
                derived_removed += 1
    else:
        trace_root = run_root / "memory_records" / "traces"
        trace_files = sum(
            1
            for task_id, episode in selected
            for _ in trace_root.glob(f"robomemarena_task_{task_id:03d}_episode_{episode:03d}_call_*")
        )
        derived_removed = sum((run_root / relative).is_file() for relative in DERIVED_FILES)
    return {
        "selected_episodes": len(selected),
        "outcome_rows": outcome_rows,
        "trace_files": trace_files,
        "derived_files": derived_removed,
        "applied": int(apply),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes", required=True, help='For example: "26:4,9;25:44"')
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    summary = repair(args.run_root, _parse_episodes(args.episodes), apply=args.apply)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
