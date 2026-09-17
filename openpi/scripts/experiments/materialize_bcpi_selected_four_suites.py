from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from scripts.experiments.prepare_pi_topk_banks import task_key
    from scripts.experiments.prepare_pi_topk_banks import write_slice
    from scripts.experiments.run_bcpi_positive_topk_ablation import CANDIDATES
except ModuleNotFoundError:
    from prepare_pi_topk_banks import task_key
    from prepare_pi_topk_banks import write_slice
    from run_bcpi_positive_topk_ablation import CANDIDATES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-positive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capacity-per-task", default="")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema") != "bcpi_positive_topk_selection_v1":
        raise RuntimeError("Unexpected B+C-pi positive selection schema")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(CANDIDATES):
        raise RuntimeError("Positive selection does not contain all nine audited candidates")
    count = selection["selected_positive_capacity_per_task"]
    top_k = int(selection["selected_positive_top_k"])
    override_requested = bool(args.capacity_per_task) or args.top_k != 0
    if override_requested:
        if not args.capacity_per_task or args.top_k <= 0:
            raise ValueError("capacity-per-task and top-k must be provided together")
        count = args.capacity_per_task
        top_k = args.top_k
    normalized_count = count if count == "max" else int(count)
    if (normalized_count, top_k) not in CANDIDATES:
        raise RuntimeError(f"Selected pair is outside the registered candidate grid: {(count, top_k)}")

    source = args.source_positive.resolve()
    source_meta = torch.load(source / "gpm_memory_meta.pt", map_location="cpu", weights_only=False)
    four_suite_items = [
        item
        for item in source_meta
        if task_key(item)[0] in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    ]
    if len(four_suite_items) != 7334:
        raise RuntimeError(f"Expected 7,334 B+C-pi positive items across four suites, found {len(four_suite_items)}")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "capacity_per_task": normalized_count,
                    "top_k": top_k,
                    "source_items": len(four_suite_items),
                    "manual_audit_override": override_requested,
                    "status": "preflight_passed",
                },
                sort_keys=True,
            )
        )
        return

    count_tag = str(normalized_count)
    output = args.output_root.resolve() / f"capacity_{count_tag}" / "positive"
    write_slice(
        source,
        output,
        "gpm_memory",
        normalized_count,
        allowed_suites={"libero_spatial", "libero_object", "libero_goal", "libero_10"},
    )
    build_summary = json.loads((output / "build_summary.json").read_text(encoding="utf-8"))
    runtime = {
        "schema": "bcpi_selected_four_suites_runtime_v1",
        "selection": str(args.selection.resolve()),
        "positive_capacity_per_task": normalized_count,
        "positive_top_k": top_k,
        "positive_bank": str(output),
        "positive_items": int(build_summary["items"]),
        "manual_audit_override": override_requested,
    }
    selected_meta = torch.load(output / "gpm_memory_meta.pt", map_location="cpu", weights_only=False)
    source_composition = {"B": 0, "C_pi": 0}
    for item in selected_meta:
        if item.get("provenance", {}).get("collection_group") == "C-pi":
            source_composition["C_pi"] += 1
        else:
            source_composition["B"] += 1
    runtime["positive_source_composition"] = source_composition
    runtime_path = args.output_root.resolve() / "selected_runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
