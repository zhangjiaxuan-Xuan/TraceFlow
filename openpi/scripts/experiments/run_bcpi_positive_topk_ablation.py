from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
from typing import Any

try:
    from scripts.experiments.run_topk_campaign import Job
    from scripts.experiments.run_topk_campaign import PoolScheduler
    from scripts.experiments.run_topk_campaign import _validate_gpu_pool
except ModuleNotFoundError:
    from run_topk_campaign import Job
    from run_topk_campaign import PoolScheduler
    from run_topk_campaign import _validate_gpu_pool


CANDIDATES: tuple[tuple[int | str, int], ...] = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _bank_files(root: Path, negative: bool) -> tuple[Path, Path, Path]:
    prefix = "gpm_negative_memory" if negative else "gpm_memory"
    return (
        root / f"{prefix}_meta.pt",
        root / f"{prefix}.index",
        root / f"{prefix}_actions.npz",
    )


def _capacity_tag(count: int | str) -> str:
    return "max" if count == "max" else f"{int(count):02d}"


def _read_eval(path: Path) -> dict[tuple[int, int], bool]:
    episodes: dict[tuple[int, int], bool] = {}
    if not path.is_file():
        raise RuntimeError(f"Missing evaluation log: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "episode_result":
                continue
            if row.get("task_suite") != "libero_10":
                raise RuntimeError(f"Unexpected suite at {path}:{line_number}")
            key = (int(row["task_id"]), int(row["episode_idx"]))
            if key in episodes:
                raise RuntimeError(f"Duplicate episode {key} in {path}")
            if row.get("error"):
                raise RuntimeError(f"Episode error {key} in {path}: {row['error']}")
            episodes[key] = bool(row["success"])
    expected = {(task_id, episode) for task_id in range(10) for episode in range(100, 150)}
    if set(episodes) != expected:
        raise RuntimeError(
            f"Incomplete paired log {path}: found={len(episodes)} "
            f"missing={len(expected - set(episodes))} extra={len(set(episodes) - expected)}"
        )
    return episodes


def _capacity_sort(count: int | str) -> int:
    return 10**9 if count == "max" else int(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--task-head", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu-pool", default="0,1")
    parser.add_argument("--base-port", type=int, default=8200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.openpi_root.resolve()
    artifact = args.artifact_root.resolve()
    gpu_pool = tuple(item.strip() for item in args.gpu_pool.split(",") if item.strip())
    if not gpu_pool or len(gpu_pool) != len(set(gpu_pool)) or any(not item.isdigit() for item in gpu_pool):
        raise ValueError("--gpu-pool must contain unique numeric GPU indices")
    ports = tuple(args.base_port + 100 * index for index in range(len(gpu_pool)))
    for path in (args.policy_dir / "model.safetensors", args.task_head, artifact / "bank_preparation.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    negative = artifact / "banks/fixed_cpi_n_fmax/negative"
    for path in _bank_files(negative, True):
        if not path.is_file():
            raise FileNotFoundError(path)
    negative_digests = {path.name: _sha256(path) for path in _bank_files(negative, True)}
    policy_digest = _sha256(args.policy_dir / "model.safetensors")
    task_head_digest = _sha256(args.task_head)
    positive_digest_cache: dict[Path, dict[str, str]] = {}

    jobs = []
    specs: list[tuple[int | str, int, Path, Path, dict[str, Any]]] = []
    for count, top_k in CANDIDATES:
        tag = f"s{_capacity_tag(count)}_kp{top_k:02d}"
        positive = artifact / "banks" / f"s{_capacity_tag(count)}_fmax" / "positive"
        for path in _bank_files(positive, False):
            if not path.is_file():
                raise FileNotFoundError(path)
        if positive not in positive_digest_cache:
            positive_digest_cache[positive] = {
                path.name: _sha256(path) for path in _bank_files(positive, False)
            }
        run_root = artifact / "runs" / tag
        log_path = run_root / "eval/libero_10.jsonl"
        identity = {
            "schema": "bcpi_positive_topk_ablation_node_v1",
            "protocol": "pi_v1_bcpi_positive_capacity_topk_libero10_v1",
            "positive_capacity_per_task": count,
            "positive_top_k": top_k,
            "negative_reference": "all C-pi + N LIBERO-10 failures",
            "negative_top_k": 8,
            "seed": 7,
            "episode_start": 100,
            "episodes_per_task": 50,
            "policy_batch_size": 16,
            "policy_batch_wait_ms": 100,
            "environment_workers": 32,
            "shards_per_task": 4,
            "max_shard_retries": 2,
            "dynamic_assignment": True,
            "torch_compile": False,
            "websocket_keepalive_disabled": True,
            "positive_bank_sha256": positive_digest_cache[positive],
            "negative_bank_sha256": negative_digests,
            "policy_sha256": policy_digest,
            "task_head_sha256": task_head_digest,
        }
        env = {
            "OPENPI_PYTHON": args.python,
            "POLICY_DIR": str(args.policy_dir.resolve()),
            "TASK_HEAD_CKPT": str(args.task_head.resolve()),
            "MEMORY_META_PATH": str(positive / "gpm_memory_meta.pt"),
            "FAISS_INDEX_PATH": str(positive / "gpm_memory.index"),
            "MEMORY_ACTIONS_PATH": str(positive / "gpm_memory_actions.npz"),
            "NEGATIVE_MEMORY_META_PATH": str(negative / "gpm_negative_memory_meta.pt"),
            "NEGATIVE_FAISS_INDEX_PATH": str(negative / "gpm_negative_memory.index"),
            "NEGATIVE_MEMORY_ACTIONS_PATH": str(negative / "gpm_negative_memory_actions.npz"),
            "METHOD": "guidance_negative",
            "SUITES": "libero_10",
            "MEMORY_TOP_K": str(top_k),
            "NEGATIVE_MEMORY_TOP_K": "8",
            "RUN_ROOT": str(run_root),
            "RESUME": "auto",
        }
        jobs.append(
            Job(
                job_id=tag,
                stage="positive",
                command=("bash", "scripts/experiments/run_pi_topk_eval_dynamic_batch16_env32.sh"),
                cwd=root,
                env=tuple(sorted(env.items())),
                identity=identity,
                completion_path=run_root / "orchestrator.identity.json",
                completion_check=lambda path=log_path: _read_eval(path),
            )
        )
        specs.append((count, top_k, run_root, log_path, identity))

    if not args.dry_run:
        campaign = type(
            "GpuContract",
            (),
            {"gpu_pool": gpu_pool},
        )()
        _validate_gpu_pool(campaign, args.python)
    scheduler = PoolScheduler(
        gpu_pool=gpu_pool,
        ports=ports,
        log_root=artifact / "stdout",
        fail_fast=True,
        resume=True,
        dry_run=args.dry_run,
    )
    previous_handlers = {}

    def stop(_signum: int, _frame: Any) -> None:
        scheduler.terminate()
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, stop)
    try:
        scheduler.run_stage("positive", jobs)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if args.dry_run:
        print(f"B+C-pi positive ablation dry-run passed: candidates={len(jobs)} slots={len(gpu_pool)}")
        return

    rows = []
    reference_keys = None
    for count, top_k, _run_root, log_path, identity in specs:
        episodes = _read_eval(log_path)
        if reference_keys is None:
            reference_keys = set(episodes)
        elif set(episodes) != reference_keys:
            raise RuntimeError("Ablation candidates are not episode-paired")
        successes = sum(episodes.values())
        rows.append(
            {
                "positive_capacity_per_task": count,
                "positive_top_k": top_k,
                "episodes": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes),
                "eval_log": str(log_path),
                "eval_log_sha256": _sha256(log_path),
                "identity_sha256": hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    best = max(row["success_rate"] for row in rows)
    eligible = [row for row in rows if row["success_rate"] >= best - 0.005]
    selected = min(
        eligible,
        key=lambda row: (
            _capacity_sort(row["positive_capacity_per_task"]),
            int(row["positive_top_k"]),
        ),
    )
    selection = {
        "schema": "bcpi_positive_topk_selection_v1",
        "status": "awaiting_manual_audit",
        "selection_rule": "within_0.5pp_of_best_then_smallest_capacity_then_smallest_topk",
        "selected_positive_capacity_per_task": selected["positive_capacity_per_task"],
        "selected_positive_top_k": selected["positive_top_k"],
        "negative_reference_capacity_per_task": "max",
        "negative_reference_top_k": 8,
        "candidates": rows,
    }
    _atomic_json(artifact / "selection/positive_selection.json", selection)
    _atomic_json(
        artifact / "campaign_summary.json",
        {
            "schema": "bcpi_positive_topk_campaign_v1",
            "status": "completed",
            "gpu_pool": list(gpu_pool),
            "results": [result.__dict__ for result in scheduler.history],
        },
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
