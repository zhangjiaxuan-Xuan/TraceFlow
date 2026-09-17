from __future__ import annotations

import numpy as np
import pytest

from openpi.analysis.cross_consumer import ConsumerDomain
from openpi.analysis.cross_consumer import CrossConsumerConfig
from openpi.analysis.cross_consumer import analyze_cross_consumer


def _paired_domain(seed: int, *, smol: bool, inconsistent_task: bool = False) -> ConsumerDomain:
    rng = np.random.default_rng(seed)
    latent = []
    manifest = []
    for task in range(4):
        for episode in range(6):
            vector = np.zeros(5)
            vector[task] = 1.0
            latent.append(vector + rng.normal(scale=0.02, size=5))
            label = task + 1 if inconsistent_task and task == 0 and episode == 0 else task
            manifest.append({"task_id": label, "action_id": f"anchor-{task}-{episode}"})
    latent_array = np.asarray(latent)
    if smol:
        raw = latent_array @ rng.normal(size=(5, 9))
        head = latent_array @ rng.normal(size=(5, 4))
    else:
        raw = latent_array @ rng.normal(size=(5, 7))
        head = latent_array @ rng.normal(size=(5, 6))
    return ConsumerDomain(tuple(manifest), raw, head)


def test_cross_consumer_supports_different_dimensions_and_is_reproducible() -> None:
    pi = {name: _paired_domain(index + 1, smol=False) for index, name in enumerate(("B", "N", "S"))}
    smol = {name: _paired_domain(index + 1, smol=True) for index, name in enumerate(("B", "N", "S"))}
    config = CrossConsumerConfig(permutations=29, bootstraps=29, seed=17)
    first = analyze_cross_consumer(pi, smol, config)
    second = analyze_cross_consumer(pi, smol, config)
    assert first == second
    assert len(first["comparisons"]) == 6
    assert {row["domain"] for row in first["comparisons"]} == {"B", "N", "S"}
    assert all(row["shared_anchors"] == 24 and row["shared_tasks"] == 4 for row in first["comparisons"])
    assert all(0.0 <= row["linear_cka"] <= 1.0 for row in first["comparisons"])
    assert all(0.0 <= row["rsa_q_value_bh"] <= 1.0 for row in first["comparisons"])


def test_alignment_uses_only_exact_anchor_intersection() -> None:
    pi_domain = _paired_domain(3, smol=False)
    smol_domain = _paired_domain(3, smol=True)
    shortened = ConsumerDomain(smol_domain.manifest[:-3], smol_domain.raw[:-3], smol_domain.head[:-3])
    report = analyze_cross_consumer(
        dict.fromkeys(("B", "N", "S"), pi_domain),
        dict.fromkeys(("B", "N", "S"), shortened),
        CrossConsumerConfig(permutations=9, bootstraps=9),
    )
    assert all(row["shared_anchors"] == 21 for row in report["comparisons"])


def test_inconsistent_task_labels_for_shared_anchor_are_rejected() -> None:
    pi_domain = _paired_domain(5, smol=False)
    smol_domain = _paired_domain(5, smol=True, inconsistent_task=True)
    with pytest.raises(ValueError, match="inconsistent task labels"):
        analyze_cross_consumer(
            dict.fromkeys(("B", "N", "S"), pi_domain),
            dict.fromkeys(("B", "N", "S"), smol_domain),
            CrossConsumerConfig(permutations=1, bootstraps=1),
        )
