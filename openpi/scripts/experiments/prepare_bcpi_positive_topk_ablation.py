from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from scripts.experiments.prepare_pi_topk_banks import task_key
    from scripts.experiments.prepare_pi_topk_banks import write_slice
except ModuleNotFoundError:
    from prepare_pi_topk_banks import task_key
    from prepare_pi_topk_banks import write_slice


def _load(path: Path) -> list[dict]:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_bank.resolve()
    output = args.output_root.resolve()
    positive_source = source / "positive"
    negative_source = source / "negative"
    positive = _load(positive_source / "gpm_memory_meta.pt")
    negative = _load(negative_source / "gpm_negative_memory_meta.pt")

    libero10_positive = [item for item in positive if task_key(item)[0] == "libero_10"]
    b_items = [item for item in libero10_positive if not item.get("provenance")]
    c_items = [
        item
        for item in libero10_positive
        if item.get("provenance", {}).get("collection_group") == "C-pi"
    ]
    if len(libero10_positive) != 1448 or len(b_items) != 500 or len(c_items) != 948:
        raise RuntimeError(
            "Unexpected B+C-pi LIBERO-10 positive inventory: "
            f"all={len(libero10_positive)} B={len(b_items)} C-pi={len(c_items)}"
        )
    libero10_failures = [
        item
        for item in negative
        if task_key(item)[0] == "libero_10"
    ]
    c_failures = [
        item
        for item in libero10_failures
        if item.get("provenance", {}).get("collection_group") == "C-pi"
    ]
    n_failures = [
        item
        for item in libero10_failures
        if item.get("provenance", {}).get("source_family") == "new_pi"
    ]
    if len(libero10_failures) != 549 or len(c_failures) != 52 or len(n_failures) != 497:
        raise RuntimeError(
            "Unexpected LIBERO-10 failure inventory: "
            f"all={len(libero10_failures)} C-pi={len(c_failures)} N={len(n_failures)}"
        )

    for count in (1, 10, 50, "max"):
        count_tag = str(count) if count == "max" else f"{count:02d}"
        write_slice(
            positive_source,
            output / "banks" / f"s{count_tag}_fmax" / "positive",
            "gpm_memory",
            count,
            allowed_suites={"libero_10"},
        )
    write_slice(
        negative_source,
        output / "banks" / "fixed_cpi_n_fmax" / "negative",
        "gpm_negative_memory",
        "max",
        allowed_suites={"libero_10"},
    )
    summary = {
        "schema": "bcpi_positive_topk_ablation_banks_v1",
        "positive_source": "B500 LIBERO-10 + C-pi 948 successes",
        "negative_reference": "all C-pi + N LIBERO-10 failures",
        "positive_candidates": [
            {"capacity_per_task": count, "top_k": top_k}
            for count, top_k in (
                (1, 8),
                (10, 1),
                (10, 8),
                (50, 8),
                (50, 16),
                (50, 32),
                ("max", 8),
                ("max", 16),
                ("max", 32),
            )
        ],
        "negative_top_k": 8,
        "libero10_positive_items": len(libero10_positive),
        "libero10_cpi_failure_items": len(c_failures),
        "libero10_n_failure_items": len(n_failures),
        "libero10_failure_items": len(libero10_failures),
    }
    (output / "bank_preparation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
