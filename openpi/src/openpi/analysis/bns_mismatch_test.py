from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from openpi.analysis.bns_mismatch import AnalysisConfig
from openpi.analysis.bns_mismatch import DomainData
from openpi.analysis.bns_mismatch import analyze_bns_mismatch
from openpi.analysis.bns_mismatch import analyze_mismatch_study
from openpi.analysis.bns_mismatch import benjamini_hochberg
from openpi.analysis.bns_mismatch import linear_cka
from openpi.analysis.bns_mismatch import load_study_manifest


def _domain(seed: int, *, shared_anchors: bool) -> DomainData:
    rng = np.random.default_rng(seed)
    task_basis = np.eye(4, 6)
    raw: list[np.ndarray] = []
    manifest = []
    for task in range(4):
        for trajectory in range(5):
            raw.append(task_basis[task] + rng.normal(scale=0.03, size=6))
            anchor = f"shared-{task}-{trajectory}" if shared_anchors else f"domain-{seed}-{task}-{trajectory}"
            manifest.append({"task_id": task, "action_id": anchor})
    raw_array = np.asarray(raw)
    head = raw_array @ rng.normal(size=(6, 5))
    return DomainData(tuple(manifest), raw_array, head)


def test_linear_cka_is_symmetric_and_identical_on_diagonal() -> None:
    rng = np.random.default_rng(3)
    left = rng.normal(size=(30, 7))
    right = rng.normal(size=(30, 5))
    assert np.isclose(linear_cka(left, left), 1.0)
    assert np.isclose(linear_cka(left, right), linear_cka(right, left))


def test_bns_matrices_are_symmetric_except_directional_retrieval() -> None:
    domains = {name: _domain(index + 1, shared_anchors=True) for index, name in enumerate(("B", "N", "S"))}
    report = analyze_bns_mismatch(domains, AnalysisConfig(top_k=3, permutations=19, bootstraps=19, seed=11))
    for representation in ("raw", "head"):
        for metric in ("centroid_cosine", "rsa_spearman", "linear_cka"):
            matrix = np.asarray(report["representations"][representation][metric])
            assert np.allclose(matrix, matrix.T, equal_nan=True)
            assert np.allclose(np.diag(matrix), 1.0)
    retrieval = np.asarray(report["retrieval_compatibility"]["purity_at_k"])
    assert retrieval.shape == (3, 3)
    assert np.all((retrieval >= 0) & (retrieval <= 1))


def test_resampling_is_reproducible_and_missing_shared_anchors_are_explicit() -> None:
    domains = {name: _domain(index + 5, shared_anchors=False) for index, name in enumerate(("B", "N", "S"))}
    config = AnalysisConfig(top_k=3, permutations=29, bootstraps=29, seed=19)
    first = analyze_bns_mismatch(domains, config)
    second = analyze_bns_mismatch(domains, config)
    assert first["pairwise"] == second["pairwise"]
    cka = np.asarray(first["representations"]["raw"]["linear_cka"])
    assert np.all(np.isnan(cka[np.triu_indices(3, k=1)]))
    assert all(row["shared_anchors"] == 0 for row in first["pairwise"])


def test_bh_fdr_matches_known_step_up_values_and_preserves_nan() -> None:
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_twelve_unit_study_keeps_signs_independent_and_aligns_consumers(tmp_path) -> None:
    units = []
    for consumer_index, consumer in enumerate(("pi", "smol")):
        for domain_index, domain in enumerate(("B", "N", "S")):
            for sign_index, sign in enumerate(("positive", "negative")):
                stem = f"{consumer}_{domain}_{sign}"
                manifest = tmp_path / f"{stem}.jsonl"
                rows = [
                    {"task_id": task, "action_id": f"{domain}-{sign}-{task}-{item}"}
                    for task in range(4)
                    for item in range(4)
                ]
                manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                rng = np.random.default_rng(100 * consumer_index + 10 * domain_index + sign_index)
                values = np.stack([np.eye(4, 6)[row["task_id"]] + rng.normal(0, 0.02, 6) for row in rows])
                raw = tmp_path / f"{stem}_raw.npy"
                head = tmp_path / f"{stem}_head.npy"
                np.save(raw, values)
                np.save(head, values @ rng.normal(size=(6, 5)))
                units.append(
                    {
                        "consumer": consumer,
                        "distribution": domain,
                        "sign": sign,
                        "manifest": {"path": str(manifest), "sha256": _sha(manifest)},
                        "raw": {"path": str(raw), "sha256": _sha(raw)},
                        "head": {"path": str(head), "sha256": _sha(head)},
                    }
                )
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps({"schema_version": 1, "units": units}), encoding="utf-8")
    study = load_study_manifest(study_path)
    report = analyze_mismatch_study(study, AnalysisConfig(top_k=2, permutations=3, bootstraps=3))
    assert set(report["signs"]) == {"positive", "negative"}
    for sign in ("positive", "negative"):
        assert set(report["signs"][sign]["consumers"]) == {"pi", "smol"}
        assert report["signs"][sign]["cross_consumer_exact_anchors"]["B"]["raw"]["exact_anchors"] == 16
        clustering = report["signs"][sign]["consumers"]["pi"]["domains"]["B"]["clustering"]["raw"]
        assert clustering["mean_intra_task_cosine"] > clustering["mean_inter_task_cosine"]


def test_study_manifest_rejects_a_missing_sign(tmp_path) -> None:
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"schema_version": 1, "units": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 12"):
        load_study_manifest(path)
