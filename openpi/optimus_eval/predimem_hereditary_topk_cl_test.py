from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import faiss
import numpy as np
import torch

from optimus_eval import predimem_hereditary_topk_cl as cl


def _bank(path: Path, prefix: str, count: int) -> tuple[Path, Path, Path]:
    path.mkdir(parents=True)
    vectors = np.eye(count, 2048, dtype=np.float32)
    index = faiss.IndexFlatIP(2048)
    index.add(vectors)
    faiss.write_index(index, str(path / "gpm_memory.index"))
    ids = [f"{prefix}_{index}" for index in range(count)]
    metadata = [
        {
            "action_id": action_id,
            "task_emb": torch.from_numpy(vectors[index]),
            "anchor_frame": 0,
            "anchor_progress": 0.0,
        }
        for index, action_id in enumerate(ids)
    ]
    torch.save(metadata, path / "gpm_memory_meta.pt")
    np.savez_compressed(
        path / "gpm_memory_actions.npz",
        actions=np.zeros((count * 10, 32), dtype=np.float32),
        offsets=np.arange(0, (count + 1) * 10, 10, dtype=np.int64),
        ids=np.asarray(ids),
    )
    return (
        path / "gpm_memory_meta.pt",
        path / "gpm_memory.index",
        path / "gpm_memory_actions.npz",
    )


def test_admission() -> None:
    assert cl._admission(8, 0) == "success"
    assert cl._admission(0, 8) == "failure"
    assert cl._admission(8, 8) == "both"


def test_eval_passes_active_negative_gate(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        captured.update(env or {})

    monkeypatch.setattr(cl, "_run", fake_run)
    bank = _bank(tmp_path / "bank", "memory", 1)
    args = argparse.Namespace(
        extra8_root=tmp_path,
        vla_checkpoint=tmp_path,
        action_stats=tmp_path / "stats.json",
        openpi_data_home=tmp_path,
        episodes_per_task=51,
        upper_gpu=0,
        lower_gpu=1,
        upper_batch_size=32,
        lower_batch_size=32,
        env_workers=64,
        save_video=True,
        runner=tmp_path / "runner.sh",
    )
    cl._eval(
        args,
        run_root=tmp_path / "run",
        seed=101,
        positive_k=8,
        negative_k=8,
        positive_bank=bank,
        negative_bank=bank,
    )
    assert captured["MEMORY_ADMISSION"] == "both"
    assert captured["NEGATIVE_MEMORY_MIN_SIMILARITY"] == "0.975"
    assert captured["NEGATIVE_MEMORY_MIN_CONFIDENCE"] == "0.0"
    assert captured["RECORD_MEMORY_DATA"] == "1"
    assert captured["MEMORY_TRACE_LEVEL"] == "bank"
    assert captured["TRACE_LOCAL_WORKERS"] == "8"
    assert captured["TRACE_TRANSFER_WORKERS"] == "8"


def test_result_reads_fusion_aggregate(tmp_path: Path) -> None:
    aggregate = tmp_path / "fusion" / "aggregate.json"
    aggregate.parent.mkdir()
    aggregate.write_text(json.dumps({"TSR": 0.5, "CSR": 0.25}), encoding="utf-8")
    assert cl._result(tmp_path) == (0.5, 0.25)


def test_complete_hereditary_campaign_with_fake_runner(tmp_path: Path) -> None:
    extra8 = tmp_path / "extra8"
    fixed = _bank(extra8 / "memory/fusion", "fixed", 2)
    fixed[0].replace(extra8 / "memory/fusion/gpm_memory_meta_dense_frame_v3.pt")
    shared = extra8 / "memory/shared"
    shared.mkdir(parents=True)
    fixed[2].replace(shared / "gpm_memory_actions.npz")
    (extra8 / "heads/fusion").mkdir(parents=True)
    (extra8 / "heads/fusion/best.pt").touch()

    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "assets/robomemarena/extra8_pi05_reactive").mkdir(parents=True)
    (checkpoint / "model.safetensors").touch()
    (checkpoint / "assets/robomemarena/extra8_pi05_reactive/norm_stats.json").write_text("{}")
    fake_runner = tmp_path / "fake_runner.sh"
    fake_runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "root=\"${RUN_ROOT}/fusion\"\n"
        "mkdir -p \"${root}/memory_records/traces\"\n"
        f'"{sys.executable}" - "${{root}}" "${{SEED}}" <<\'PY\'\n'
        "import json, pathlib, sys, numpy as np\n"
        "root=pathlib.Path(sys.argv[1]); seed=int(sys.argv[2])\n"
        "(root/'aggregate.json').write_text(json.dumps({'TSR': 0.5, 'CSR': 0.25}))\n"
        "(root/'results.txt').write_text('complete\\n')\n"
        "for label, success in [('success', True), ('failure', False)]:\n"
        " p=root/'memory_records/traces'/f'{label}.npz'\n"
        " np.savez(p, task_embedding=np.ones(2048,dtype=np.float32), x_trajectory=np.zeros((1,1,10,32),dtype=np.float32))\n"
        " row={'trace_path':str(p.relative_to(root)),'task_id':18,'episode_idx':0,'policy_call_idx':0,'seed':seed,'progress':0.5,'success':success}\n"
        " (root/'memory_records'/f'{label}_index.jsonl').write_text(json.dumps(row)+'\\n')\n"
        "PY\n",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    output = tmp_path / "campaign"
    repository = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "optimus_eval.predimem_hereditary_topk_cl",
        "--output-root",
        str(output),
        "--extra8-root",
        str(extra8),
        "--vla-checkpoint",
        str(checkpoint),
        "--openpi-data-home",
        str(tmp_path),
        "--python",
        sys.executable,
        "--runner",
        str(fake_runner),
        "--bank-builder",
        str(repository / "scripts/robomemarena/build_predimem_recorded_bank.py"),
        "--bank-merger",
        str(repository / "scripts/robomemarena/merge_predimem_positive_banks.py"),
        "--no-save-video",
    ]
    subprocess.run(command, cwd=repository, env=os.environ, check=True)

    assert len(list((output / "runs").iterdir())) == 16
    assert json.loads((output / "selections/round03.json").read_text())["winner"]["name"] == "p16_n0"
    with np.load(output / "banks/after_round03/positive/gpm_memory_actions.npz") as packed:
        assert len(packed["ids"]) == 6
    with np.load(output / "banks/after_round03/negative/gpm_memory_actions.npz") as packed:
        assert len(packed["ids"]) == 4
