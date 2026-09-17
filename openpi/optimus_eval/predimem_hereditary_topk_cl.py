from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


CANDIDATES = (
    ("p16_n0", 16, 0),
    ("p12_n4", 12, 4),
    ("p8_n8", 8, 8),
    ("p4_n12", 4, 12),
    ("p0_n16", 0, 16),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _require(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def _admission(positive_k: int, negative_k: int) -> str:
    if positive_k > 0 and negative_k > 0:
        return "both"
    if positive_k > 0:
        return "success"
    if negative_k > 0:
        return "failure"
    raise ValueError("At least one retrieval budget must be positive")


def _result(run_root: Path) -> tuple[float, float]:
    aggregate = run_root / "fusion" / "aggregate.json"
    if not aggregate.is_file():
        raise FileNotFoundError(f"Incomplete evaluation: {aggregate}")
    value = json.loads(aggregate.read_text(encoding="utf-8"))
    return float(value["TSR"]), float(value["CSR"])


def _eval(
    args: argparse.Namespace,
    *,
    run_root: Path,
    seed: int,
    positive_k: int,
    negative_k: int,
    positive_bank: tuple[Path, Path, Path],
    negative_bank: tuple[Path, Path, Path] | None,
) -> None:
    if (run_root / "fusion" / "results.txt").is_file():
        print(f"[skip] completed evaluation: {run_root}", flush=True)
        return
    admission = _admission(positive_k, negative_k)
    if negative_k > 0 and negative_bank is None:
        raise RuntimeError("Negative retrieval requested before a failure bank exists")
    resume = "1" if (run_root / "run_config.json").is_file() else "0"
    env = {
        **os.environ,
        "AOSS_ROOT": str(args.extra8_root),
        "VLA_CKPT": str(args.vla_checkpoint),
        "VLA_CONFIG": "pi05_robomemarena_extra8_reactive",
        "ACTION_STATS": str(args.action_stats),
        "OPENPI_DATA_HOME": str(args.openpi_data_home),
        "HEAD_VARIANTS": "fusion",
        "TASK_IDS": "18,19,25,26",
        "MEMORY_ALLOWED_TASK_IDS": "18,19,25,26",
        "MEMORY_ADMISSION": admission,
        "MEMORY_TOP_K": str(max(positive_k, 1)),
        "NEGATIVE_MEMORY_TOP_K": str(max(negative_k, 1)),
        "NEGATIVE_MEMORY_MIN_SIMILARITY": "0.975",
        "NEGATIVE_MEMORY_MIN_CONFIDENCE": "0.0",
        "POSITIVE_MEMORY_META_PATH": str(positive_bank[0]),
        "POSITIVE_FAISS_INDEX_PATH": str(positive_bank[1]),
        "POSITIVE_MEMORY_ACTIONS_PATH": str(positive_bank[2]),
        "MEMORY_META_BASENAME": "gpm_memory_meta.pt",
        "MEMORY_ALIGNMENT_TAG": "dense-frame-v3-hereditary-cl-v1",
        "MEMORY_GUIDANCE_NORM_CAP": "0.20",
        "MEMORY_GUIDANCE_TOTAL_NORM_CAP": "0.20",
        "EPISODES_PER_TASK": str(args.episodes_per_task),
        "SEED": str(seed),
        "UPPER_GPU": str(args.upper_gpu),
        "LOWER_GPU": str(args.lower_gpu),
        "UPPER_BATCH_SIZE": str(args.upper_batch_size),
        "LOWER_BATCH_SIZE": str(args.lower_batch_size),
        "ENV_WORKERS": str(args.env_workers),
        "SAVE_VIDEO": "1" if args.save_video else "0",
        "RECORD_MEMORY_DATA": "1",
        "MEMORY_TRACE_LEVEL": "bank",
        "TRACE_LOCAL_WORKERS": "8",
        "TRACE_TRANSFER_WORKERS": "8",
        "TRACE_TRANSFER_BATCH_SIZE": "1024",
        "RUN_ROOT": str(run_root),
        "RESUME": resume,
    }
    if negative_bank is not None:
        env.update(
            {
                "NEGATIVE_MEMORY_META_PATH": str(negative_bank[0]),
                "NEGATIVE_FAISS_INDEX_PATH": str(negative_bank[1]),
                "NEGATIVE_MEMORY_ACTIONS_PATH": str(negative_bank[2]),
            }
        )
    _run(["bash", str(args.runner), "v1"], env=env)


def _recorded_bank(
    args: argparse.Namespace, *, run_root: Path, label: str, output: Path
) -> tuple[Path, Path, Path]:
    if not (output / "provenance.json").is_file():
        command = [
            str(args.python),
            str(args.bank_builder),
            "--run-root",
            str(run_root / "fusion"),
            "--label",
            label,
            "--output",
            str(output),
            "--workers",
            str(args.bank_workers),
            "--alignment",
            "dense_frame_v3",
        ]
        if output.exists():
            command.append("--overwrite")
        _run(command)
    return (
        output / "gpm_memory_meta.pt",
        output / "gpm_memory.index",
        output / "gpm_memory_actions.npz",
    )


def _append_positive(
    args: argparse.Namespace,
    *,
    left: tuple[Path, Path, Path],
    right_dir: Path,
    output: Path,
    left_name: str,
) -> tuple[Path, Path, Path]:
    if not (output / "provenance.json").is_file():
        command = [
            str(args.python),
            str(args.bank_merger),
            "--left-meta",
            str(left[0]),
            "--left-index",
            str(left[1]),
            "--left-actions",
            str(left[2]),
            "--right-bank",
            str(right_dir),
            "--output",
            str(output),
            "--left-name",
            left_name,
        ]
        if output.exists():
            command.append("--overwrite")
        _run(command)
    return (
        output / "gpm_memory_meta.pt",
        output / "gpm_memory.index",
        output / "gpm_memory_actions.npz",
    )


def _campaign_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "predimem_transferring_hereditary_topk_cl_v2_async_bank",
        "tasks": [18, 19, 25, 26],
        "round0": {"positive_top_k": 8, "negative_top_k": 0, "seed": 50},
        "search_rounds": 3,
        "round_seeds": [101, 152, 203],
        "episodes_per_task": args.episodes_per_task,
        "candidates": [
            {"name": name, "positive_top_k": positive, "negative_top_k": negative}
            for name, positive, negative in CANDIDATES
        ],
        "winner_rule": "max_TSR_then_CSR_then_candidate_order",
        "inheritance": "immutable_extra8_plus_all_winner_lineage_outcomes",
        "negative_gate": {"min_similarity": 0.975, "min_confidence": 0.0},
        "extra8_root": str(args.extra8_root),
        "upper_batch_size": args.upper_batch_size,
        "lower_batch_size": args.lower_batch_size,
        "env_workers": args.env_workers,
        "save_video": args.save_video,
        "trace_pipeline": "bank-local8-transfer8-resumable",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/path/to/storage/datasets/robotics/RoboMemArena/derived/"
            "transferring_hereditary_topk_cl_v2_async_bank_seedblocks"
        ),
    )
    parser.add_argument(
        "--extra8-root",
        type=Path,
        default=Path(
            "/path/to/storage/datasets/robotics/RoboMemArena/derived/"
            "retrieval_joint_v6_tdense5/predimem_2048_fp32"
        ),
    )
    parser.add_argument(
        "--vla-checkpoint",
        type=Path,
        default=Path(
            "/path/to/storage/datasets/robotics/RoboMemArena/derived/"
            "predimem_dual_tower/checkpoints/vla_alltask_pytorch"
        ),
    )
    parser.add_argument(
        "--openpi-data-home",
        type=Path,
        default=Path(
            "/path/to/storage/datasets/robotics/RoboMemArena/derived/"
            "predimem_dual_tower/runtime/openpi_data"
        ),
    )
    parser.add_argument("--python", type=Path, default=Path("/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python"))
    parser.add_argument("--runner", type=Path, default=root / "scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh")
    parser.add_argument("--bank-builder", type=Path, default=root / "scripts/robomemarena/build_predimem_recorded_bank.py")
    parser.add_argument("--bank-merger", type=Path, default=root / "scripts/robomemarena/merge_predimem_positive_banks.py")
    parser.add_argument("--episodes-per-task", type=int, default=51)
    parser.add_argument("--upper-gpu", type=int, default=0)
    parser.add_argument("--lower-gpu", type=int, default=1)
    parser.add_argument("--upper-batch-size", type=int, default=32)
    parser.add_argument("--lower-batch-size", type=int, default=32)
    parser.add_argument("--env-workers", type=int, default=64)
    parser.add_argument("--bank-workers", type=int, default=32)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.action_stats = args.vla_checkpoint / "assets/robomemarena/extra8_pi05_reactive/norm_stats.json"

    fixed_positive = (
        args.extra8_root / "memory/fusion/gpm_memory_meta_dense_frame_v3.pt",
        args.extra8_root / "memory/fusion/gpm_memory.index",
        args.extra8_root / "memory/shared/gpm_memory_actions.npz",
    )
    _require(
        [
            *fixed_positive,
            args.extra8_root / "heads/fusion/best.pt",
            args.vla_checkpoint / "model.safetensors",
            args.action_stats,
            args.python,
            args.runner,
            args.bank_builder,
            args.bank_merger,
        ]
    )
    config = _campaign_config(args)
    config_path = args.output_root / "campaign_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Campaign configuration mismatch: {config_path}")
    elif not args.preflight_only:
        _write_json(config_path, config)
    if args.preflight_only:
        print(json.dumps(config, indent=2), flush=True)
        print("Hereditary CL preflight passed", flush=True)
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    round0_root = args.output_root / "runs/round00_collector_k8"
    _eval(
        args,
        run_root=round0_root,
        seed=50,
        positive_k=8,
        negative_k=0,
        positive_bank=fixed_positive,
        negative_bank=None,
    )
    round0_online = args.output_root / "online/round00_collector_k8"
    round0_success = _recorded_bank(
        args, run_root=round0_root, label="success", output=round0_online / "success"
    )
    round0_failure = _recorded_bank(
        args, run_root=round0_root, label="failure", output=round0_online / "failure"
    )
    parent_positive = _append_positive(
        args,
        left=fixed_positive,
        right_dir=round0_online / "success",
        output=args.output_root / "banks/after_round00/positive",
        left_name="immutable_extra8",
    )
    parent_negative = round0_failure
    round0_tsr, round0_csr = _result(round0_root)
    _write_json(
        args.output_root / "selections/round00.json",
        {"name": "collector_k8", "TSR": round0_tsr, "CSR": round0_csr, "seed": 50},
    )

    for round_index, seed in enumerate((101, 152, 203), 1):
        scores = []
        for order, (name, positive_k, negative_k) in enumerate(CANDIDATES):
            run_name = f"round{round_index:02d}_{name}"
            run_root = args.output_root / "runs" / run_name
            _eval(
                args,
                run_root=run_root,
                seed=seed,
                positive_k=positive_k,
                negative_k=negative_k,
                positive_bank=parent_positive,
                negative_bank=parent_negative,
            )
            tsr, csr = _result(run_root)
            scores.append(
                {
                    "name": name,
                    "positive_top_k": positive_k,
                    "negative_top_k": negative_k,
                    "TSR": tsr,
                    "CSR": csr,
                    "seed": seed,
                    "run_root": str(run_root),
                    "candidate_order": order,
                }
            )
        winner = max(scores, key=lambda row: (row["TSR"], row["CSR"], -row["candidate_order"]))
        selection_path = args.output_root / f"selections/round{round_index:02d}.json"
        selection = {"round": round_index, "seed": seed, "candidates": scores, "winner": winner}
        if selection_path.is_file():
            existing = json.loads(selection_path.read_text(encoding="utf-8"))
            if existing != selection:
                raise RuntimeError(f"Winner changed for completed round: {selection_path}")
        else:
            _write_json(selection_path, selection)

        winner_root = Path(winner["run_root"])
        online_root = args.output_root / "online" / f"round{round_index:02d}_{winner['name']}"
        _recorded_bank(args, run_root=winner_root, label="success", output=online_root / "success")
        _recorded_bank(args, run_root=winner_root, label="failure", output=online_root / "failure")
        next_bank_root = args.output_root / f"banks/after_round{round_index:02d}"
        parent_positive = _append_positive(
            args,
            left=parent_positive,
            right_dir=online_root / "success",
            output=next_bank_root / "positive",
            left_name=f"parent_after_round{round_index - 1:02d}",
        )
        parent_negative = _append_positive(
            args,
            left=parent_negative,
            right_dir=online_root / "failure",
            output=next_bank_root / "negative",
            left_name=f"negative_parent_after_round{round_index - 1:02d}",
        )

    print(f"CAMPAIGN_ROOT={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
