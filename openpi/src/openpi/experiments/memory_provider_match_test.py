from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openpi.experiments.memory_provider_match import MatchProtocolError
from openpi.experiments.memory_provider_match import job_identity
from openpi.experiments.memory_provider_match import load_experiment


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, value: bytes = b"fixture") -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"path": str(path), "sha256": _sha(path)}


def _manifest(tmp_path: Path) -> Path:
    policies = {}
    for consumer in ("pi", "smol"):
        checkpoint = tmp_path / "checkpoints" / consumer
        checkpoint.mkdir(parents=True)
        marker = _artifact(checkpoint / "model.safetensors", consumer.encode())
        selection_path = tmp_path / f"{consumer}_selection.json"
        selection = (
            {
                "schema_version": 1,
                "consumer": "pi",
                "positive_memory_per_task": 50,
                "positive_top_k": 16,
                "negative_memory_per_task": 10,
                "negative_top_k": 8,
            }
            if consumer == "pi"
            else {
                "schema_version": 2,
                "selection": "smol_final_selection",
                "positive": {"count_per_task": 50, "top_k": 16},
                "negative": {"count_per_task": 10, "top_k": 8},
            }
        )
        selection_path.write_text(
            json.dumps(selection),
            encoding="utf-8",
        )
        policies[consumer] = {
            "checkpoint_dir": str(checkpoint),
            "checkpoint_identity": marker,
            "head": _artifact(tmp_path / f"{consumer}_head.pt", consumer.encode()),
            "selection": {"path": str(selection_path), "sha256": _sha(selection_path)},
            "python": "/fixture/python",
        }
    providers = {}
    for provider in ("pi-mem", "smol-mem"):
        encodings = {}
        for consumer in ("pi", "smol"):
            root = tmp_path / "banks" / provider / consumer
            positive_names = {
                "meta": "gpm_memory_meta.pt",
                "index": "gpm_memory.index",
                "actions": "gpm_memory_actions.npz",
            }
            negative_names = {
                "meta": "gpm_negative_memory_meta.pt",
                "index": "gpm_negative_memory.index",
                "actions": "gpm_negative_memory_actions.npz",
            }
            encodings[consumer] = {
                "root": str(root),
                "summary": _artifact(root / "build_summary.json", b"{}"),
                "positive_count_per_task": 50,
                "negative_count_per_task": 10,
                "positive": {
                    name: _artifact(
                        root / "positive" / positive_names[name], f"{provider}-{consumer}-p-{name}".encode()
                    )
                    for name in ("meta", "index", "actions")
                },
                "negative": {
                    name: _artifact(
                        root / "negative" / negative_names[name], f"{provider}-{consumer}-n-{name}".encode()
                    )
                    for name in ("meta", "index", "actions")
                },
            }
        providers[provider] = {"encodings": encodings}
    path = tmp_path / "match.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_root": str(tmp_path / "runs"),
                "gpu_pool": [0, 1, 2, 3],
                "ports": [8200, 8300, 8400, 8500],
                "resume": True,
                "policies": policies,
                "providers": providers,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loads_exact_four_match_cells_with_independent_sign_banks(tmp_path: Path) -> None:
    experiment = load_experiment(_manifest(tmp_path))
    identities = {
        (consumer, provider): job_identity(experiment, consumer, provider)
        for consumer in ("pi", "smol")
        for provider in ("pi-mem", "smol-mem")
    }
    assert len(identities) == 4
    assert all(identity["guidance"] == "v0-positive-negative" for identity in identities.values())
    assert identities[("pi", "pi-mem")]["positive_count"] == 50
    assert identities[("smol", "smol-mem")]["negative_top_k"] == 8
    assert identities[("pi", "pi-mem")]["positive_artifacts"] != identities[("pi", "smol-mem")]["positive_artifacts"]


def test_rejects_missing_negative_artifact_and_wrong_selection_digest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["providers"]["pi-mem"]["encodings"]["pi"]["negative"]["index"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MatchProtocolError):
        load_experiment(manifest)

    manifest = _manifest(tmp_path / "second")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["policies"]["smol"]["selection"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MatchProtocolError, match="SHA-256 mismatch"):
        load_experiment(manifest)


def test_rejects_obsolete_smol_positive_negative_selection(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    selection_path = Path(payload["policies"]["smol"]["selection"]["path"])
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "consumer": "smol",
                "selection": "smol_final_selection",
                "positive": {"success_count": 50, "positive_top_k": 16},
                "positive_negative": {"failure_count": 10, "negative_top_k": 8},
            }
        ),
        encoding="utf-8",
    )
    payload["policies"]["smol"]["selection"]["sha256"] = _sha(selection_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MatchProtocolError, match="schema_version must be 2"):
        load_experiment(manifest)
