from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import torch


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _file_record(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    return {"path": str(path.resolve()), "bytes": size, "mib": size / 2**20}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    aggregate = _load_json(args.run_root / "aggregate.json")
    run_config = _load_json(args.run_root / "run_config.json")
    memory_config = _load_json(args.run_root / "memory_config.json")
    if not aggregate or not memory_config:
        raise RuntimeError("Ablation summary requires aggregate.json and memory_config.json")

    paths = {
        "retrieval_head": Path(memory_config["task_head_ckpt"]),
        "metadata": Path(memory_config["memory_meta_path"]),
        "faiss_index": Path(memory_config["faiss_index_path"]),
        "packed_actions": Path(memory_config["memory_actions_path"]),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    storage = {label: _file_record(path) for label, path in paths.items()}
    runtime_bytes = sum(item["bytes"] for item in storage.values())

    checkpoint = torch.load(paths["retrieval_head"], map_location="cpu", weights_only=False)
    index = faiss.read_index(str(paths["faiss_index"]))
    memory_items = int(index.ntotal)
    temporal_window = int(checkpoint.get("temporal_window", 1))
    alignment = str(memory_config.get("memory_action_alignment", "legacy_progress"))
    artifact_paths = " ".join(str(path) for path in paths.values())
    if alignment == "dense_frame_v3" and "dense5" in artifact_paths:
        bank_layout = "dense5"
    elif alignment == "dense_frame_v3":
        bank_layout = "dense10"
    elif memory_items == 12_800:
        bank_layout = "anchors16"
    elif memory_items == 51_200:
        bank_layout = "anchors64"
    else:
        bank_layout = f"anchors_or_dense_{memory_items}"

    record = {
        "protocol": "pi05_arena_memory_compute_storage_success_ablation_v1",
        "run_root": str(args.run_root.resolve()),
        "checkpoint": aggregate.get("checkpoint"),
        "memory_mode": memory_config["memory_mode"],
        "bank_layout": bank_layout,
        "memory_items": memory_items,
        "query_frames": temporal_window,
        "temporal_offsets": list(checkpoint.get("temporal_offsets", [0])),
        "action_alignment": alignment,
        "episodes": int(aggregate["num_episodes"]),
        "TSR": float(aggregate["TSR"]),
        "CSR": float(aggregate["CSR"]),
        "SR": float(aggregate["CSR"]),
        "success_metric_definition": "SR aliases Arena complete success rate (CSR); TSR is reported separately",
        "batch_size": int(run_config.get("batch_size", 1)),
        "env_workers": int(run_config.get("env_workers", 1)),
        "action_generation_timing": aggregate.get("action_generation_timing", {}),
        "denoise_compute": (
            _load_json(args.run_root / "nfe_summary.json")
            if memory_config["memory_mode"]
            in {"v3", "v3_prior_only", "v3_prior_decay", "v3_1", "v3_re"}
            else {
                "sampler": (
                    "v1_fixed_flow_guidance_capped"
                    if memory_config["memory_mode"] == "v1"
                    else "v0_fixed_flow_guidance"
                ),
                "fixed_nfe": 10,
            }
        ),
        "runtime_memory_storage": {
            "bytes": runtime_bytes,
            "mib": runtime_bytes / 2**20,
            "gib": runtime_bytes / 2**30,
            "files": storage,
        },
    }
    output = args.run_root / "memory_ablation_record.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        "ABLATION\t"
        f"mode={record['memory_mode']}\tbank={bank_layout}\tframes={temporal_window}\t"
        f"items={memory_items}\tstorage_GiB={runtime_bytes / 2**30:.3f}\t"
        f"TSR={record['TSR']:.2f}\tCSR={record['CSR']:.2f}"
    )


if __name__ == "__main__":
    main()
