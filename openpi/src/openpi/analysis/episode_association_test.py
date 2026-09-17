from __future__ import annotations

from openpi.analysis.episode_association import cluster_bootstrap_associations


def test_cluster_bootstrap_is_reproducible() -> None:
    rows = []
    for task in range(4):
        for episode in range(10):
            purity = (task * 10 + episode) / 40
            rows.append(
                {
                    "task_id": task,
                    "success_flip": float(purity > 0.5),
                    "purity": purity,
                    "margin": purity - 0.2,
                    "guidance_norm": 1.0 - purity,
                    "guidance_cosine": purity,
                }
            )
    first = cluster_bootstrap_associations(rows, outcome="success_flip", bootstraps=39, seed=13)
    second = cluster_bootstrap_associations(rows, outcome="success_flip", bootstraps=39, seed=13)
    assert first == second
    assert first[0]["spearman"] > 0
