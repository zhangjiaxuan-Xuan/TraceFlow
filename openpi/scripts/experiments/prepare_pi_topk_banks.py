#!/usr/bin/env python3
"""Prepare independent B6500-positive and C-pi-negative banks for Pi top-k sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import re
from typing import Any

import faiss
import numpy as np
import torch

EVALUATION_SUITES = ("libero_10",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def task_key(item: dict[str, Any]) -> tuple[str, int]:
    provenance = item.get("provenance", {})
    suite = item.get("suite", provenance.get("suite", provenance.get("task_suite", "")))
    task_id = item.get("task_id", provenance.get("task_id"))
    return str(suite), int(task_id)


def capacity_task_key(item: dict[str, Any]) -> tuple[str, str]:
    """Canonical capacity bucket independent of dataset-specific task numbering."""
    provenance = item.get("provenance", {})
    suite = item.get("suite") or provenance.get("suite") or provenance.get("task_suite", "")
    name = item.get("task_name") or provenance.get("prompt") or provenance.get("task_key", "")
    normalized_name = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if not suite or not normalized_name:
        raise ValueError(f"Memory item lacks a canonical suite/task name: {item.get('action_id', '<unknown>')}")
    return str(suite), normalized_name


def load_bank(
    root: Path, prefix: str,
) -> tuple[list[dict[str, Any]], faiss.Index, np.ndarray, np.ndarray, list[str], np.ndarray]:
    meta = torch.load(root / f"{prefix}_meta.pt", map_location="cpu", weights_only=False)
    index = faiss.read_index(str(root / f"{prefix}.index"))
    with np.load(root / f"{prefix}_actions.npz", allow_pickle=False) as packed:
        actions = np.asarray(packed["actions"], dtype=np.float32)
        offsets = np.asarray(packed["offsets"], dtype=np.int64)
        ids = [str(value) for value in packed["ids"].tolist()]
    if len(meta) != index.ntotal or len(meta) != len(ids) or offsets.shape != (len(meta) + 1,):
        raise RuntimeError(f"Inconsistent bank files under {root}")
    keys = np.asarray(index.reconstruct_n(0, len(meta)), dtype=np.float32)
    return list(meta), index, actions, offsets, ids, keys


def write_slice(
    source: Path,
    output: Path,
    prefix: str,
    capacity: int | str,
    *,
    allowed_suites: set[str] | None = None,
    required_collection_group: str | None = None,
) -> None:
    source_meta = source / f"{prefix}_meta.pt"
    contract = {
        "schema_version": 3,
        "source_bank": str(source.resolve()),
        "source_bank_sha256": sha256(source_meta),
        "capacity_per_task": capacity,
        "prefix": prefix,
        "allowed_suites": sorted(allowed_suites) if allowed_suites is not None else None,
        "capacity_grouping": "canonical_suite_task_name_v1",
        "selection_order": "source_index_v1",
    }
    if required_collection_group is not None:
        contract["required_collection_group"] = required_collection_group
    if output.exists() and any(output.iterdir()):
        summary = output / "build_summary.json"
        if summary.is_file():
            existing = json.loads(summary.read_text(encoding="utf-8"))
            mismatched = {key: (existing.get(key), value) for key, value in contract.items() if existing.get(key) != value}
            if mismatched:
                raise RuntimeError(f"Existing bank contract mismatch at {output}: {mismatched}")
            print(f"Validated existing bank: {output}")
            return
        raise RuntimeError(f"Non-empty incomplete bank output: {output}")
    meta, _index, actions, offsets, ids, keys = load_bank(source, prefix)
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, item in enumerate(meta):
        key = capacity_task_key(item)
        provenance = item.get("provenance", {})
        if allowed_suites is not None and key[0] not in allowed_suites:
            continue
        if required_collection_group is not None and provenance.get("collection_group") != required_collection_group:
            continue
        grouped.setdefault(key, []).append(index)
    if allowed_suites is not None and prefix == "gpm_memory":
        observed_suites = {suite for suite, _ in grouped}
        expected_tasks = 10 * len(allowed_suites)
        if observed_suites != allowed_suites or len(grouped) != expected_tasks:
            raise RuntimeError(
                "Positive source bank does not match the requested 10-task suites: "
                f"suites={sorted(observed_suites)} tasks={len(grouped)} expected_tasks={expected_tasks}"
            )
    selected: list[int] = []
    for key in sorted(grouped):
        candidates = sorted(
            grouped[key],
            key=lambda index: (int(meta[index].get("episode_idx", 0)), str(meta[index].get("action_id", ids[index]))),
        )
        selected.extend(candidates if capacity == "max" else candidates[: int(capacity)])
    selected.sort()
    if not selected:
        raise RuntimeError(f"No entries selected from {source}")
    selected_meta = [meta[index] for index in selected]
    selected_keys = np.ascontiguousarray(keys[selected], dtype=np.float32)
    selected_actions: list[np.ndarray] = []
    selected_ids: list[str] = []
    new_offsets = np.zeros(len(selected) + 1, dtype=np.int64)
    for output_index, source_index in enumerate(selected):
        selected_actions.append(actions[int(offsets[source_index]) : int(offsets[source_index + 1])])
        selected_ids.append(ids[source_index])
        new_offsets[output_index + 1] = new_offsets[output_index] + selected_actions[-1].shape[0]
    packed_actions = np.concatenate(selected_actions, axis=0).astype(np.float32, copy=False)
    output.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(selected_keys.shape[1])
    index.add(selected_keys)
    faiss.write_index(index, str(output / f"{prefix}.index"))
    torch.save(selected_meta, output / f"{prefix}_meta.pt")
    np.savez_compressed(output / f"{prefix}_actions.npz", actions=packed_actions, offsets=new_offsets, ids=np.asarray(selected_ids))
    (output / "build_summary.json").write_text(
        json.dumps(
            {
                **contract,
                "items": len(selected_meta),
                "tasks": len(grouped),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Built {output}: items={len(selected_meta)} capacity={capacity}")


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def feature_cache_complete(feature_dir: Path, manifest: Path) -> bool:
    state_path = feature_dir / "cache_state.json"
    completed_path = feature_dir / "completed.npy"
    if not state_path.is_file() or not completed_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed = np.load(completed_path, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    rows = len(read_jsonl(manifest))
    return (
        state.get("manifest_sha256") == sha256(manifest)
        and int(state.get("rows", -1)) == rows
        and completed.shape == (rows,)
        and bool(np.all(completed))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--b-bank", type=Path, required=True)
    parser.add_argument("--c-pi-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--gpu-pool", default="0,1")
    parser.add_argument("--base-port", type=int, default=8200)
    args = parser.parse_args()
    gpu_pool = [item.strip() for item in args.gpu_pool.split(",") if item.strip()]
    if not gpu_pool or len(gpu_pool) != len(set(gpu_pool)):
        raise ValueError("--gpu-pool must contain unique comma-separated GPU indices")
    if any(not item.isdigit() for item in gpu_pool):
        raise ValueError("--gpu-pool currently supports numeric CUDA device indices only")
    ports = [args.base_port + 100 * index for index in range(len(gpu_pool))]
    root = args.openpi_root.resolve()
    artifact = args.artifact_root.resolve()
    b_bank = args.b_bank.resolve()
    c_manifest = args.c_pi_manifest.resolve()
    for path in (args.policy_dir / "model.safetensors", args.head, c_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for prefix in ("gpm_memory",):
        for suffix in ("_meta.pt", ".index", "_actions.npz"):
            if not (b_bank / f"{prefix}{suffix}").is_file():
                raise FileNotFoundError(b_bank / f"{prefix}{suffix}")
    b_meta, *_ = load_bank(b_bank, "gpm_memory")
    if len(b_meta) != 6500 or not all(bool(item.get("success", True)) for item in b_meta):
        raise RuntimeError(f"B6500 positive bank must contain 6500 positive entries, got {len(b_meta)}")

    banks = artifact / "banks"
    evaluation_suites = set(EVALUATION_SUITES)
    for count in (1, 10, 50):
        write_slice(
            b_bank,
            banks / f"s{count:02d}_f01" / "positive",
            "gpm_memory",
            count,
            allowed_suites=evaluation_suites,
        )

    c_rows = [row for row in read_jsonl(c_manifest) if row.get("outcome") == "failure" or row.get("success") is False]
    if not c_rows:
        raise RuntimeError("C-pi manifest has no failures")
    failure_counts: dict[tuple[str, int], int] = {}
    for row in c_rows:
        key = (str(row["suite"]), int(row["task_id"]))
        failure_counts[key] = failure_counts.get(key, 0) + 1
    missing = {key: count for key, count in failure_counts.items() if count < 1}
    if missing:
        raise RuntimeError(f"C-pi has no failure for tasks: {missing}")
    c_failure_manifest = artifact / "manifests" / "c_pi_failure.jsonl"
    write_jsonl(c_failure_manifest, c_rows)
    c_features = artifact / "features" / "c_pi_failure"
    if feature_cache_complete(c_features, c_failure_manifest):
        print(f"Validated complete feature cache: {c_features}")
    else:
        run([
            args.python,
            str(root / "scripts/memory/cache_prior_head_features.py"),
            "--manifest", str(c_failure_manifest),
            "--output-dir", str(c_features),
            "--policy-dir", str(args.policy_dir),
            "--config-name", "pi05_libero",
            "--device", args.device,
            "--batch-size", str(args.feature_batch_size),
        ])
    full_build = artifact / "c_pi_failure_full"
    if not (full_build / "negative/gpm_negative_memory_meta.pt").is_file():
        if full_build.exists() and any(full_build.iterdir()):
            raise RuntimeError(f"Incomplete C-pi full bank output: {full_build}")
        run([
            args.python,
            str(root / "scripts/memory/build_cl_memory_bank.py"),
            "--group", "pi_topk_C_pi_failure_max",
            "--manifest", str(c_failure_manifest),
            "--feature-dir", str(c_features),
            "--checkpoint", str(args.head),
            "--output-dir", str(full_build),
            "--admission", "failure",
            "--device", args.device,
        ])
    negative_source = full_build / "negative"
    for count in (1, 5, "max"):
        count_tag = str(count) if isinstance(count, str) else f"{count:02d}"
        write_slice(
            negative_source,
            banks / f"s50_f{count_tag}" / "negative",
            "gpm_negative_memory",
            count,
            allowed_suites=evaluation_suites,
        )

    quantity_manifest = artifact / "pi_topk_quantity.json"
    # source_run and action_contract can legitimately vary between collection
    # jobs.  Keep only the stable producer identity in the natural-manifest
    # audit; the exact C-pi manifest digest is recorded separately.
    provenance = {"producer": str(c_rows[0]["producer"])} if "producer" in c_rows[0] else {}
    quantity = {
        "schema_version": 1,
        "name": "pi_v1_topk_C_pi",
        "policy_family": "pi",
        "openpi_root": str(root),
        "artifact_root": str(artifact),
        "run_root": str(artifact / "runs"),
        "python": args.python,
        "pi_include_failure_50": False,
        "pi_sparse_failure_fallback": True,
        "natural_manifest": {"path": str(c_manifest), "sha256": sha256(c_manifest), "provenance": provenance},
        "policy": {
            "directory": str(args.policy_dir),
            "model_sha256": sha256(args.policy_dir / "model.safetensors"),
            "task_head_checkpoint": str(args.head),
            "task_head_sha256": sha256(args.head),
            "config_name": "pi05_libero",
        },
        "evaluation": {"seed": 7, "suites": list(EVALUATION_SUITES), "episodes_per_task": 50, "batch_size": 8},
        "positive_source": "B6500",
        "negative_source": "C-pi",
    }
    quantity_manifest.write_text(json.dumps(quantity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sweep_manifest = artifact / "pi_topk_sweep_manifest.json"
    sweep_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "memory_quantity_manifest": {"path": str(quantity_manifest), "sha256": sha256(quantity_manifest)},
                "run_root": str(artifact / "sweep_runs"),
                "python": args.python,
                "gpu_pool": gpu_pool,
                "ports": ports,
                "source_contract": {"positive": "B6500", "negative": "C-pi"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    campaign_manifest = artifact / "pi_topk_campaign_manifest.json"
    campaign_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pi_manifest": str(sweep_manifest),
                "run_root": str(artifact / "campaign_runs"),
                "gpu_pool": gpu_pool,
                "ports": ports,
                "fail_fast": True,
                "resume": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "quantity_manifest": str(quantity_manifest),
                "sweep_manifest": str(sweep_manifest),
                "campaign_manifest": str(campaign_manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
