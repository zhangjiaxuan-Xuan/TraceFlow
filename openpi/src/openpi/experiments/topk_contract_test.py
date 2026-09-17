from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openpi.experiments.topk_contract import TopKContractError
from openpi.experiments.topk_contract import load_topk_selection


def _write(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pi_selection_accepts_legacy_fixed_prior_metadata(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    digest = _write(
        path,
        {
            "schema_version": 1,
            "consumer": "pi",
            "positive_top_k": 8,
            "negative_top_k": 16,
            "prior_top_k": 8,
        },
    )
    selection = load_topk_selection(path, digest, consumer="pi")
    assert (selection.positive_top_k, selection.negative_top_k, selection.prior_top_k) == (8, 16, 8)


def test_smol_selection_may_omit_prior(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    digest = _write(
        path,
        {"schema_version": 1, "consumer": "smol", "positive_top_k": 8, "negative_top_k": 4},
    )
    assert load_topk_selection(path, digest, consumer="smol").prior_top_k is None


def test_pi_guidance_selection_may_omit_fixed_prior_baseline(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    digest = _write(
        path,
        {"schema_version": 1, "consumer": "pi", "positive_top_k": 8, "negative_top_k": 16},
    )
    assert load_topk_selection(path, digest, consumer="pi").prior_top_k is None


def test_native_smol_final_selection_is_normalized(tmp_path: Path) -> None:
    path = tmp_path / "smol_final_selection.json"
    digest = _write(
        path,
        {
            "schema_version": 2,
            "selection": "smol_final_selection",
            "positive": {"count_per_task": 50, "top_k": 16},
            "negative": {"count_per_task": 10, "top_k": 32},
        },
    )
    selection = load_topk_selection(path, digest, consumer="smol")
    assert (selection.positive_top_k, selection.negative_top_k, selection.prior_top_k) == (16, 32, None)
    assert (selection.positive_memory_per_task, selection.negative_memory_per_task) == (50, 10)
    with pytest.raises(TopKContractError, match="consumer mismatch"):
        load_topk_selection(path, digest, consumer="pi")


@pytest.mark.parametrize("fault", ["digest", "consumer"])
def test_selection_rejects_identity_and_schema_faults(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "selection.json"
    payload = {
        "schema_version": 1,
        "consumer": "pi" if fault != "consumer" else "smol",
        "positive_top_k": 8,
        "negative_top_k": 8,
    }
    payload["prior_top_k"] = 8
    digest = _write(path, payload)
    if fault == "digest":
        digest = "0" * 64
    with pytest.raises(TopKContractError):
        load_topk_selection(path, digest, consumer="pi")
