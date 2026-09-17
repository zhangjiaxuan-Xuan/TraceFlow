#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENPI_ROOT = REPO_ROOT / "openpi"
SMOL_ROOT = REPO_ROOT / "smolvla"


class CampaignError(RuntimeError):
    pass


def _visible_cuda_device_count(python: str) -> int:
    probe = subprocess.run(
        [python, "-c", "import torch; print(torch.cuda.device_count())"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return int(probe.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise CampaignError(f"could not parse CUDA device count from: {probe.stdout!r}") from exc


def _validate_gpu_pool(campaign: "Campaign", python: str) -> None:
    non_numeric = [gpu for gpu in campaign.gpu_pool if not gpu.isdigit()]
    if non_numeric:
        raise CampaignError(f"top-k campaign requires numeric GPU indices, got {non_numeric}")
    visible_count = _visible_cuda_device_count(python)
    unavailable = [gpu for gpu in campaign.gpu_pool if int(gpu) >= visible_count]
    if unavailable:
        raise CampaignError(
            "campaign GPU pool is not available in this workload: "
            f"requested={list(campaign.gpu_pool)}, torch.cuda.device_count()={visible_count}, "
            f"unavailable={unavailable}. Request the required GPUs or set GPU_POOL_CSV."
        )


@dataclasses.dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    command: tuple[str, ...]
    cwd: Path
    env: tuple[tuple[str, str], ...] = ()
    identity: dict[str, Any] = dataclasses.field(default_factory=dict)
    completion_path: Path | None = None
    completion_check: Callable[[], None] | None = dataclasses.field(default=None, compare=False, repr=False)
    adopted: bool = False


@dataclasses.dataclass(frozen=True)
class Campaign:
    manifest_path: Path
    pi_manifest: Path
    smol_manifest: Path | None
    run_root: Path
    gpu_pool: tuple[str, ...]
    ports: tuple[int, ...]
    fail_fast: bool
    resume: bool


@dataclasses.dataclass
class Result:
    job_id: str
    stage: str
    status: str
    gpu: str | None
    port: int | None
    returncode: int
    started_at: float
    finished_at: float
    stdout_log: str
    command: list[str]
    error: str | None = None


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity_digest(identity: dict[str, Any]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(canonical)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_campaign(path: str | Path) -> Campaign:
    manifest = Path(path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise CampaignError("campaign schema_version must be 1")
    base = manifest.parent
    gpu_pool = tuple(str(item) for item in payload.get("gpu_pool", (0, 1, 2, 3)))
    if not gpu_pool or len(set(gpu_pool)) != len(gpu_pool):
        raise CampaignError("gpu_pool must contain unique GPU identifiers")
    ports = tuple(int(item) for item in payload.get("ports", (8200, 8300, 8400, 8500)))
    if len(ports) != len(gpu_pool) or len(set(ports)) != len(ports):
        raise CampaignError("ports must be unique and match gpu_pool length")
    return Campaign(
        manifest_path=manifest,
        pi_manifest=_resolve(base, payload["pi_manifest"]),
        smol_manifest=_resolve(base, payload["smol_manifest"]) if payload.get("smol_manifest") else None,
        run_root=_resolve(base, payload["run_root"]),
        gpu_pool=gpu_pool,
        ports=ports,
        fail_fast=bool(payload.get("fail_fast", True)),
        resume=bool(payload.get("resume", True)),
    )


def _read_completion(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid completion identity: {path}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"completion identity is not an object: {path}")
    return value


class PoolScheduler:
    def __init__(
        self,
        *,
        gpu_pool: tuple[str, ...],
        ports: tuple[int, ...],
        log_root: Path,
        fail_fast: bool,
        resume: bool,
        dry_run: bool = False,
    ) -> None:
        self.slots = tuple(zip(gpu_pool, ports, strict=True))
        self.log_root = log_root
        self.fail_fast = fail_fast
        self.resume = resume
        self.dry_run = dry_run
        self._stop = threading.Event()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self.history: list[Result] = []

    def terminate(self) -> None:
        self._stop.set()
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)

    def _run_one(self, job: Job, gpu: str, port: int) -> Result:
        started = time.time()
        log_path = self.log_root / job.stage / f"{job.job_id.replace(':', '__')}.log"
        expected = job.identity
        if job.completion_path is not None:
            existing = _read_completion(job.completion_path)
            if existing is not None:
                if existing != expected:
                    raise CampaignError(f"resume identity mismatch for {job.job_id}: {job.completion_path}")
                if not self.resume:
                    raise CampaignError(f"completed job requires resume: {job.job_id}")
                if job.completion_check is not None:
                    job.completion_check()
                return Result(
                    job.job_id,
                    job.stage,
                    "skipped",
                    gpu,
                    port,
                    0,
                    started,
                    time.time(),
                    str(log_path),
                    list(job.command),
                )
        if self.dry_run:
            return Result(
                job.job_id,
                job.stage,
                "dry-run",
                gpu,
                port,
                0,
                started,
                time.time(),
                str(log_path),
                list(job.command),
            )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **dict(job.env)}
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "SERVER_CUDA_VISIBLE_DEVICES": gpu,
                "CLIENT_CUDA_VISIBLE_DEVICES": gpu,
                "MUJOCO_EGL_DEVICE_ID": "0",
                "GPU": gpu,
                "PORT": str(port),
                "CAMPAIGN_GPU_SLOT": gpu,
                "CAMPAIGN_PORT": str(port),
            }
        )
        returncode = 1
        error: str | None = None
        with log_path.open("w", encoding="utf-8") as stream:
            stream.write(f"JOB={job.job_id}\nGPU={gpu}\nPORT={port}\n")
            stream.write(f"COMMAND={shlex.join(job.command)}\n\n")
            stream.flush()
            process = subprocess.Popen(
                job.command,
                cwd=job.cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with self._lock:
                self._processes[job.job_id] = process
            try:
                returncode = process.wait()
            finally:
                with self._lock:
                    self._processes.pop(job.job_id, None)
        if returncode == 0:
            try:
                if job.completion_check is not None:
                    job.completion_check()
                if job.completion_path is not None:
                    _atomic_json(job.completion_path, expected)
            except Exception as exc:  # validation failure is a failed node
                returncode = 1
                error = str(exc)
        status = "completed" if returncode == 0 else "failed"
        return Result(
            job.job_id,
            job.stage,
            status,
            gpu,
            port,
            returncode,
            started,
            time.time(),
            str(log_path),
            list(job.command),
            error,
        )

    def run_stage(self, stage: str, jobs: Iterable[Job]) -> list[Result]:
        pending: queue.Queue[Job] = queue.Queue()
        materialized = list(jobs)
        results: list[Result] = []
        for job in materialized:
            if job.stage != stage:
                raise CampaignError(f"job {job.job_id} belongs to {job.stage}, not {stage}")
            if job.adopted:
                if job.completion_check is not None:
                    job.completion_check()
                now = time.time()
                results.append(
                    Result(
                        job.job_id,
                        stage,
                        "adopted",
                        None,
                        None,
                        0,
                        now,
                        now,
                        "",
                        list(job.command),
                    )
                )
            else:
                pending.put(job)
        result_lock = threading.Lock()

        def worker(gpu: str, port: int) -> None:
            while not self._stop.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    result = self._run_one(job, gpu, port)
                except Exception as exc:
                    now = time.time()
                    result = Result(
                        job.job_id,
                        stage,
                        "failed",
                        gpu,
                        port,
                        1,
                        now,
                        now,
                        str(self.log_root / stage / f"{job.job_id.replace(':', '__')}.log"),
                        list(job.command),
                        str(exc),
                    )
                with result_lock:
                    results.append(result)
                if result.returncode and self.fail_fast:
                    self._stop.set()
                    self.terminate()
                pending.task_done()

        threads = [
            threading.Thread(target=worker, args=slot, daemon=True)
            for slot in self.slots[: max(1, min(len(self.slots), len(materialized)))]
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        results.sort(key=lambda item: item.job_id)
        self.history.extend(results)
        if any(result.returncode for result in results):
            failed = {
                result.job_id: result.error or f"exit code {result.returncode}"
                for result in results
                if result.returncode
            }
            raise CampaignError(f"stage {stage} failed: {failed}")
        if len(results) != len(materialized):
            raise CampaignError(f"stage {stage} stopped before all jobs ran")
        return results


def _import_modules() -> Any:
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    return importlib.import_module("openpi.experiments.topk_sweep")


def _pi_job(module: Any, experiment: Any, node: Any) -> Job:
    adopted_log = module._adopted_log(experiment, node)  # noqa: SLF001
    return Job(
        job_id=f"pi:{node.node_id}",
        stage=node.stage,
        command=tuple(node.command),
        cwd=OPENPI_ROOT,
        env=tuple(node.env),
        identity=dict(node.identity),
        completion_path=None if adopted_log is not None else node.identity_path,
        completion_check=lambda: module._load_paired_log(  # noqa: SLF001
            adopted_log or node.log_path, experiment
        ),
        adopted=adopted_log is not None,
    )


def _smol_job(module: Any, experiment: Any, spec: Any) -> Job:
    command = tuple(module.eval_command(experiment, spec))
    summary = experiment.run_root / spec.phase / spec.tag / "summary.json"
    identity = {
        "schema_version": 1,
        "consumer": "smol",
        "manifest_sha256": module._sha256(experiment.manifest_path),  # noqa: SLF001
        "identity_sha256": _identity_digest(
            {
                "consumer": "smol",
                "phase": spec.phase,
                "tag": spec.tag,
                "positive_top_k": spec.positive_top_k,
                "negative_top_k": spec.negative_top_k,
            }
        ),
        "phase": spec.phase,
        "tag": spec.tag,
        "success_count": spec.success_count,
        "failure_count": spec.failure_count,
        "positive_top_k": spec.positive_top_k,
        "negative_top_k": spec.negative_top_k,
    }
    return Job(
        job_id=f"smol:{spec.phase}:{spec.tag}",
        stage={
            "positive-selection": "positive",
            "negative-selection": "negative",
        }[spec.phase],
        command=command,
        cwd=REPO_ROOT,
        identity=identity,
        completion_path=summary.parent / "campaign.identity.json",
        completion_check=lambda: module._validated_summary(experiment, spec),  # noqa: SLF001
    )


def _prepare_job(campaign: Campaign, module: Any, experiment: Any) -> Job:
    command = (
        experiment.python,
        str(SMOL_ROOT / "scripts/experiments/run_memory_topk.py"),
        "--manifest",
        str(campaign.smol_manifest),
        "--phase",
        "prepare",
    )
    identity = {
        "schema_version": 1,
        "consumer": "smol",
        "phase": "prepare",
        "manifest_sha256": module._sha256(experiment.manifest_path),  # noqa: SLF001
    }
    return Job(
        job_id="smol:prepare",
        stage="prepare",
        command=command,
        cwd=REPO_ROOT,
        identity=identity,
        completion_path=campaign.run_root / "identities/smol_prepare.json",
        completion_check=lambda: module.audit_inputs(experiment),
    )


def build_stages(campaign: Campaign) -> tuple[dict[str, list[Job]], dict[str, Callable[[], None]]]:
    pi = _import_modules()
    pi_experiment = pi.load_experiment(campaign.pi_manifest)
    pi.preflight(pi_experiment)

    stages: dict[str, list[Job]] = {
        "positive": [_pi_job(pi, pi_experiment, node) for node in pi.positive_nodes(pi_experiment)],
    }

    def select_positive() -> None:
        pi.write_positive_selection(pi_experiment)

    def populate_negative() -> None:
        pi_positive = pi.load_positive_selection(pi_experiment)
        pi_count = int(pi_positive["positive_memory_per_task"])
        pi_k = int(pi_positive["positive_top_k"])
        stages["negative"] = [_pi_job(pi, pi_experiment, node) for node in pi.negative_nodes(pi_experiment, pi_count, pi_k)]

    def select_final() -> None:
        pi_positive = pi.load_positive_selection(pi_experiment)
        pi.write_final_selection(pi_experiment, pi_positive)

    barriers = {
        "after_positive": lambda: (select_positive(), populate_negative()),
        "after_negative": select_final,
    }
    return stages, barriers


def _placeholder_plan(campaign: Campaign) -> dict[str, list[str]]:
    return {
        "positive": [
            *(f"pi:positive:s{count}-k{k}" for count, k in ((1, 8), (10, 1), (10, 8), (50, 8), (50, 16), (50, 32))),
        ],
        "negative": [
            *(f"pi:negative:f{count}-k{k}" for count, k in ((1, 1), (1, 4), (5, 4), (5, 8), ("max", 4), ("max", 8))),
        ],
    }


def run(campaign: Campaign, *, dry_run: bool) -> dict[str, Any]:
    campaign.run_root.mkdir(parents=True, exist_ok=True)
    if dry_run:
        summary = {
            "schema_version": 1,
            "status": "dry-run",
            "gpu_pool": list(campaign.gpu_pool),
            "ports": list(campaign.ports),
            "stages": _placeholder_plan(campaign),
            "note": "Dry-run intentionally does not read model or memory artifacts.",
        }
        _atomic_json(campaign.run_root / "campaign_dry_run.json", summary)
        return summary

    pi = _import_modules()
    pi_experiment = pi.load_experiment(campaign.pi_manifest)
    _validate_gpu_pool(campaign, pi_experiment.python)
    stages, barriers = build_stages(campaign)
    scheduler = PoolScheduler(
        gpu_pool=campaign.gpu_pool,
        ports=campaign.ports,
        log_root=campaign.run_root / "stdout",
        fail_fast=campaign.fail_fast,
        resume=campaign.resume,
    )
    previous_handlers: dict[int, Any] = {}

    def stop(signum: int, _frame: Any) -> None:
        scheduler.terminate()
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, stop)
    results: list[Result] = []
    status = "completed"
    try:
        results.extend(scheduler.run_stage("positive", stages["positive"]))
        barriers["after_positive"]()
        results.extend(scheduler.run_stage("negative", stages["negative"]))
        barriers["after_negative"]()
    except BaseException:
        status = "failed"
        scheduler.terminate()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        summary = {
            "schema_version": 1,
            "status": status,
            "campaign_manifest": str(campaign.manifest_path),
            "gpu_pool": list(campaign.gpu_pool),
            "ports": list(campaign.ports),
            "results": [dataclasses.asdict(result) for result in scheduler.history],
        }
        _atomic_json(campaign.run_root / "campaign_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the staged Pi-only top-k campaign on one GPU pool")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_campaign(args.manifest)
    summary = run(campaign, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
