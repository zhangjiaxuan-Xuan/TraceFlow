from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPO_ROOT / "scripts/experiments/run_cl_v0_eval.sh"
BASE_SCRIPT_NAME = "run_libero10_guidance_only_v0_batch8.sh"


def _make_bank(root: Path, *, group: str = "BNS", admission: str = "both") -> Path:
    bank = root / "bank"
    artifacts = (
        "positive/gpm_memory_meta.pt",
        "positive/gpm_memory.index",
        "positive/gpm_memory_actions.npz",
        "negative/gpm_negative_memory_meta.pt",
        "negative/gpm_negative_memory.index",
        "negative/gpm_negative_memory_actions.npz",
    )
    for relative_path in artifacts:
        path = bank / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (bank / "summary.json").write_text(json.dumps({"group": group, "admission": admission}), encoding="utf-8")
    return bank


def _make_fake_entrypoint(tmp_path: Path) -> tuple[Path, Path]:
    scripts = tmp_path / "scripts"
    experiments = scripts / "experiments"
    eval_scripts = scripts / "eval"
    experiments.mkdir(parents=True)
    eval_scripts.mkdir()
    entrypoint = experiments / ENTRYPOINT.name
    shutil.copyfile(ENTRYPOINT, entrypoint)
    capture = tmp_path / "calls.jsonl"
    fake_base = eval_scripts / BASE_SCRIPT_NAME
    fake_base.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python3 - "$CAPTURE_PATH" <<'PY'
import json
import os
import sys

names = (
    "PREFLIGHT_ONLY", "MEMORY_VARIANT", "MEMORY_GUIDANCE_VERSION", "BATCH_SIZE",
    "MEMORY_META_PATH", "FAISS_INDEX_PATH", "MEMORY_ACTIONS_PATH",
    "NEGATIVE_MEMORY_META_PATH", "NEGATIVE_FAISS_INDEX_PATH",
    "NEGATIVE_MEMORY_ACTIONS_PATH", "NEGATIVE_MEMORY_TOP_K",
    "NEGATIVE_MEMORY_MIN_SIMILARITY", "NEGATIVE_MEMORY_MIN_CONFIDENCE",
    "GPU", "PORT", "SEED", "RESUME", "RUN_ROOT", "NUM_TRIALS_PER_TASK",
    "SAVE_VIDEOS", "SAVE_EPISODE_DATA", "MEMORY_SOURCE_TAG",
)
with open(sys.argv[1], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({name: os.environ.get(name) for name in names}) + "\\n")
PY
""",
        encoding="utf-8",
    )
    return entrypoint, capture


def _run(
    tmp_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
    group: str = "BNS",
    admission: str = "both",
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    entrypoint, capture = _make_fake_entrypoint(tmp_path)
    bank = _make_bank(tmp_path, group=group, admission=admission)
    env = {
        **os.environ,
        "CL_BANK_DIR": str(bank),
        "CL_GROUP_TAG": group,
        "CL_ADMISSION": admission,
        "CAPTURE_PATH": str(capture),
        "OPENPI_ROOT": str(tmp_path / "openpi"),
    }
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", str(entrypoint)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return result, capture, bank


def _calls(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_entrypoint_statically_targets_existing_v0_batch8_script() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert BASE_SCRIPT_NAME in source
    assert "require_fixed_value MEMORY_VARIANT success" in source
    assert "require_fixed_value MEMORY_VARIANT success_fail" in source
    assert 'require_fixed_value MEMORY_GUIDANCE_VERSION "${CL_GUIDANCE_VERSION}"' in source
    assert "NEGATIVE_MEMORY_TOP_K 8" in source
    assert "NEGATIVE_MEMORY_MIN_SIMILARITY -1.0" in source
    assert "NEGATIVE_MEMORY_MIN_CONFIDENCE 0.0" in source


def test_preflight_passes_fixed_values_and_all_six_artifacts(tmp_path: Path) -> None:
    result, capture, bank = _run(
        tmp_path,
        extra_env={
            "PREFLIGHT_ONLY": "1",
            "GPU": "3",
            "PORT": "9012",
            "SEED": "37",
            "RESUME": "0",
            "RUN_ROOT": str(tmp_path / "custom-run"),
        },
    )

    assert result.returncode == 0, result.stderr
    calls = _calls(capture)
    assert len(calls) == 1
    call = calls[0]
    assert call["PREFLIGHT_ONLY"] == "1"
    assert call["MEMORY_VARIANT"] == "success_fail"
    assert call["MEMORY_GUIDANCE_VERSION"] == "v0"
    assert call["BATCH_SIZE"] == "8"
    assert call["NEGATIVE_MEMORY_TOP_K"] == "8"
    assert call["NEGATIVE_MEMORY_MIN_SIMILARITY"] == "-1.0"
    assert call["NEGATIVE_MEMORY_MIN_CONFIDENCE"] == "0.0"
    assert call["MEMORY_META_PATH"] == str(bank / "positive/gpm_memory_meta.pt")
    assert call["FAISS_INDEX_PATH"] == str(bank / "positive/gpm_memory.index")
    assert call["MEMORY_ACTIONS_PATH"] == str(bank / "positive/gpm_memory_actions.npz")
    assert call["NEGATIVE_MEMORY_META_PATH"] == str(bank / "negative/gpm_negative_memory_meta.pt")
    assert call["NEGATIVE_FAISS_INDEX_PATH"] == str(bank / "negative/gpm_negative_memory.index")
    assert call["NEGATIVE_MEMORY_ACTIONS_PATH"] == str(bank / "negative/gpm_negative_memory_actions.npz")
    assert (call["GPU"], call["PORT"], call["SEED"], call["RESUME"]) == (
        "3",
        "9012",
        "37",
        "0",
    )
    assert call["NUM_TRIALS_PER_TASK"] == "100"
    assert call["SAVE_VIDEOS"] == "1"
    assert call["SAVE_EPISODE_DATA"] == "0"


def test_real_run_first_preflights_and_uses_cl_run_root(tmp_path: Path) -> None:
    result, capture, _ = _run(tmp_path, extra_env={"SEED": "47"}, group="BN", admission="success")

    assert result.returncode == 0, result.stderr
    calls = _calls(capture)
    assert [call["PREFLIGHT_ONLY"] for call in calls] == ["1", "0"]
    assert "/logs/cl_v0/BN/success/seed_47/" in calls[1]["RUN_ROOT"]
    assert "guidance_only_v0_success_fail_batch8" in calls[1]["RUN_ROOT"]
    assert calls[1]["MEMORY_SOURCE_TAG"] == "cl_v0_BN_success"
    assert calls[1]["MEMORY_VARIANT"] == "success"


@pytest.mark.parametrize(
    ("extra_env", "group", "admission", "message"),
    [
        ({"MEMORY_VARIANT": "success"}, "BNS", "both", "MEMORY_VARIANT is fixed"),
        ({"MEMORY_GUIDANCE_VERSION": "v1"}, "BNS", "both", "MEMORY_GUIDANCE_VERSION is fixed"),
        ({"BATCH_SIZE": "4"}, "BNS", "both", "BATCH_SIZE is fixed"),
        ({"NEGATIVE_MEMORY_TOP_K": "4"}, "BNS", "both", "NEGATIVE_MEMORY_TOP_K is fixed"),
        (
            {"NEGATIVE_MEMORY_MIN_SIMILARITY": "0.9"},
            "BNS",
            "both",
            "NEGATIVE_MEMORY_MIN_SIMILARITY is fixed",
        ),
        (
            {"NEGATIVE_MEMORY_MIN_CONFIDENCE": "0.75"},
            "BNS",
            "both",
            "NEGATIVE_MEMORY_MIN_CONFIDENCE is fixed",
        ),
        ({}, "B+N", "both", "CL_GROUP_TAG"),
        ({}, "BNS", "all", "CL_ADMISSION"),
    ],
)
def test_rejects_illegal_values(
    tmp_path: Path,
    extra_env: dict[str, str],
    group: str,
    admission: str,
    message: str,
) -> None:
    result, capture, _ = _run(tmp_path, extra_env=extra_env, group=group, admission=admission)

    assert result.returncode == 2
    assert message in result.stderr
    assert not capture.exists()


def test_rejects_summary_identity_mismatch(tmp_path: Path) -> None:
    entrypoint, capture = _make_fake_entrypoint(tmp_path)
    bank = _make_bank(tmp_path, group="BS", admission="failure")
    env = {
        **os.environ,
        "CL_BANK_DIR": str(bank),
        "CL_GROUP_TAG": "BN",
        "CL_ADMISSION": "failure",
        "CAPTURE_PATH": str(capture),
    }

    result = subprocess.run(["bash", str(entrypoint)], text=True, capture_output=True, env=env, check=False)

    assert result.returncode != 0
    assert "summary group mismatch" in result.stderr
    assert not capture.exists()
