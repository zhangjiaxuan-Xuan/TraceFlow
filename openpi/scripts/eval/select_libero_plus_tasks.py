#!/usr/bin/env python3
"""Deterministically select a stratified LIBERO-plus task subset.

The Plus classification ids are one-based and are not assumed to match the
benchmark task order. This utility resolves names against the actual task map
and emits zero-based ids understood by examples/libero/main.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path


def load_task_map(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = tree.body[0]
    if not isinstance(assignment, ast.Assign):
        raise ValueError(f"Expected a task-map assignment in {path}")
    value = ast.literal_eval(assignment.value)
    if not isinstance(value, dict):
        raise ValueError(f"Task map is not a dictionary: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plus-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--task-ids-csv", help="Explicit zero-based benchmark task ids for smoke/debug runs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task_ids_csv and args.all_tasks:
        raise SystemExit("--task-ids-csv and --all-tasks are mutually exclusive")
    if not args.all_tasks and not args.task_ids_csv and args.per_category <= 0:
        raise SystemExit("--per-category must be positive")

    root = args.plus_root.resolve()
    classification_path = root / "libero/libero/benchmark/task_classification.json"
    task_map_path = root / "libero/libero/benchmark/libero_suite_task_map.py"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    task_map = load_task_map(task_map_path)
    entries = classification[args.suite]
    names = {name: index for index, name in enumerate(task_map[args.suite])}
    if len(names) != len(task_map[args.suite]):
        raise SystemExit("Task map contains duplicate names")

    groups: dict[str, list[dict]] = {}
    for entry in entries:
        name = str(entry["name"])
        if name not in names:
            raise SystemExit(f"Classification task is absent from task map: {name}")
        groups.setdefault(str(entry["category"]), []).append(
            {**entry, "benchmark_task_id": names[name]}
        )

    selected: list[dict] = []
    if args.task_ids_csv:
        try:
            requested_ids = [int(value) for value in args.task_ids_csv.split(",")]
        except ValueError as error:
            raise SystemExit("--task-ids-csv must contain comma-separated integers") from error
        if not requested_ids or len(requested_ids) != len(set(requested_ids)) or min(requested_ids) < 0:
            raise SystemExit("--task-ids-csv must contain unique non-negative ids")
        by_id = {
            int(entry["benchmark_task_id"]): entry
            for category in groups.values()
            for entry in category
        }
        unknown = sorted(set(requested_ids).difference(by_id))
        if unknown:
            raise SystemExit(f"Unknown benchmark task ids: {unknown}")
        selected = [by_id[task_id] for task_id in requested_ids]
    elif args.all_tasks:
        selected = [entry for category in sorted(groups) for entry in groups[category]]
    else:
        rng = random.Random(args.seed)
        for category in sorted(groups):
            candidates = groups[category]
            if len(candidates) < args.per_category:
                raise SystemExit(
                    f"Category {category!r} has only {len(candidates)} tasks, "
                    f"cannot select {args.per_category}"
                )
            selected.extend(
                sorted(rng.sample(candidates, args.per_category), key=lambda x: x["benchmark_task_id"])
            )

    selected.sort(key=lambda x: int(x["benchmark_task_id"]))
    payload = {
        "schema": (
            "libero_plus_explicit_task_selection_v1"
            if args.task_ids_csv
            else "libero_plus_full_task_selection_v1"
            if args.all_tasks
            else "libero_plus_stratified_task_selection_v1"
        ),
        "plus_root": str(root),
        "suite": args.suite,
        "seed": args.seed,
        "per_category": None if args.all_tasks else args.per_category,
        "categories": sorted({str(entry["category"]) for entry in selected}),
        "task_count": len(selected),
        "task_ids": [int(x["benchmark_task_id"]) for x in selected],
        "tasks": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(",".join(str(x["benchmark_task_id"]) for x in selected))


if __name__ == "__main__":
    main()
