from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import run_topk_campaign as campaign


def _job(tmp_path: Path, stage: str, index: int, *, fail: bool = False) -> campaign.Job:
    output = tmp_path / "outputs" / f"{stage}-{index}.json"
    code = (
        "import json,os,pathlib,time,sys;"
        "time.sleep(0.05);"
        f"p=pathlib.Path({str(output)!r});p.parent.mkdir(parents=True,exist_ok=True);"
        "p.write_text(json.dumps({'gpu':os.environ['GPU'],'port':os.environ['PORT']}));"
        f"sys.exit({1 if fail else 0})"
    )
    return campaign.Job(
        job_id=f"{stage}:{index}",
        stage=stage,
        command=(sys.executable, "-c", code),
        cwd=tmp_path,
        env=(("GPU", "99"), ("PORT", "1")),
        identity={"stage": stage, "index": index},
        completion_path=tmp_path / "identities" / f"{stage}-{index}.json",
        completion_check=lambda: json.loads(output.read_text()),
    )


def test_pool_assigns_one_slot_per_job_and_resumes(tmp_path: Path) -> None:
    scheduler = campaign.PoolScheduler(
        gpu_pool=("2", "3"),
        ports=(9200, 9300),
        log_root=tmp_path / "logs",
        fail_fast=True,
        resume=True,
    )
    jobs = [_job(tmp_path, "positive", index) for index in range(4)]
    results = scheduler.run_stage("positive", jobs)
    assert {result.status for result in results} == {"completed"}
    payloads = [json.loads(path.read_text()) for path in sorted((tmp_path / "outputs").iterdir())]
    assert {(item["gpu"], int(item["port"])) for item in payloads} <= {
        ("2", 9200),
        ("3", 9300),
    }
    resumed = scheduler.run_stage("positive", jobs)
    assert {result.status for result in resumed} == {"skipped"}


def test_stage_barrier_is_explicit(tmp_path: Path) -> None:
    scheduler = campaign.PoolScheduler(
        gpu_pool=("0",),
        ports=(8200,),
        log_root=tmp_path / "logs",
        fail_fast=True,
        resume=True,
    )
    scheduler.run_stage("positive", [_job(tmp_path, "positive", 0)])
    assert (tmp_path / "outputs/positive-0.json").is_file()
    scheduler.run_stage("negative", [_job(tmp_path, "negative", 0)])
    assert (tmp_path / "outputs/negative-0.json").is_file()


def test_identity_mismatch_refuses_resume(tmp_path: Path) -> None:
    scheduler = campaign.PoolScheduler(
        gpu_pool=("0",),
        ports=(8200,),
        log_root=tmp_path / "logs",
        fail_fast=True,
        resume=True,
    )
    job = _job(tmp_path, "positive", 0)
    assert job.completion_path is not None
    job.completion_path.parent.mkdir(parents=True)
    job.completion_path.write_text("{}")
    with pytest.raises(campaign.CampaignError, match="identity mismatch"):
        scheduler.run_stage("positive", [job])


def test_dry_run_needs_no_child_files(tmp_path: Path) -> None:
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pi_manifest": "missing-pi.json",
                "smol_manifest": "missing-smol.json",
                "run_root": "run",
            }
        )
    )
    loaded = campaign.load_campaign(manifest)
    summary = campaign.run(loaded, dry_run=True)
    assert summary["status"] == "dry-run"
    assert len(summary["stages"]["positive"]) == 6
    assert len(summary["stages"]["negative"]) == 6
    assert "interaction" not in summary["stages"]


def test_fail_fast_reports_failed_node(tmp_path: Path) -> None:
    scheduler = campaign.PoolScheduler(
        gpu_pool=("0",),
        ports=(8200,),
        log_root=tmp_path / "logs",
        fail_fast=True,
        resume=True,
    )
    with pytest.raises(campaign.CampaignError, match="stage positive failed"):
        scheduler.run_stage("positive", [_job(tmp_path, "positive", 0, fail=True)])


def test_adopted_job_does_not_execute_or_lease_gpu(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    job = campaign.Job(
        job_id="pi:adopted",
        stage="positive",
        command=(sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"),
        cwd=tmp_path,
        identity={"exact": True},
        completion_check=lambda: None,
        adopted=True,
    )
    scheduler = campaign.PoolScheduler(
        gpu_pool=("0",),
        ports=(8200,),
        log_root=tmp_path / "logs",
        fail_fast=True,
        resume=True,
    )
    results = scheduler.run_stage("positive", [job])
    assert results[0].status == "adopted"
    assert results[0].gpu is None
    assert results[0].port is None
    assert not marker.exists()


def test_pi_adopted_log_becomes_non_executing_job(tmp_path: Path) -> None:
    adopted_log = tmp_path / "adopted.jsonl"
    adopted_log.write_text('{"event":"episode_result"}\n')
    marker = tmp_path / "pi-command-must-not-run"

    class FakePiModule:
        @staticmethod
        def _adopted_log(_experiment: object, _node: object) -> Path:
            return adopted_log

        @staticmethod
        def _load_paired_log(path: Path, _experiment: object) -> dict[tuple[int, int], bool]:
            assert path == adopted_log
            return {}

    node = SimpleNamespace(
        node_id="positive:guidance:s50:k8",
        stage="positive",
        command=(sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"),
        env=(),
        identity={"exact_identity": True},
        identity_path=tmp_path / "new-run/orchestrator.identity.json",
        log_path=tmp_path / "new-run/eval/libero_10.jsonl",
    )
    job = campaign._pi_job(FakePiModule, object(), node)  # noqa: SLF001
    assert job.adopted
    scheduler = campaign.PoolScheduler(
        gpu_pool=("0",),
        ports=(8200,),
        log_root=tmp_path / "logs",
        fail_fast=True,
        resume=True,
    )
    result = scheduler.run_stage("positive", [job])[0]
    assert result.status == "adopted"
    assert not marker.exists()
    assert not node.identity_path.exists()
