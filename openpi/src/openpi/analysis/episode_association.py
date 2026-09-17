from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from openpi.analysis.bns_mismatch import benjamini_hochberg
from openpi.analysis.bns_mismatch import spearman_correlation

DEFAULT_PREDICTORS = ("purity", "margin", "guidance_norm", "guidance_cosine")


def cluster_bootstrap_associations(
    rows: list[Mapping[str, Any]],
    *,
    outcome: str = "success",
    cluster: str = "task_id",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
    bootstraps: int = 1999,
    confidence: float = 0.95,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Estimate rank associations while resampling whole task/trajectory clusters."""
    required = {outcome, cluster, *predictors}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise KeyError(f"Episode trace is missing fields: {sorted(missing)}")
    clusters = np.asarray([str(row[cluster]) for row in rows])
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2:
        raise ValueError("Cluster bootstrap requires at least two task/trajectory clusters")
    y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    alpha = 1.0 - confidence
    for predictor in predictors:
        x = np.asarray([float(row[predictor]) for row in rows], dtype=np.float64)
        observed = spearman_correlation(x, y)
        samples: list[float] = []
        for _ in range(bootstraps):
            selected = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            indices = np.concatenate([np.flatnonzero(clusters == value) for value in selected])
            value = spearman_correlation(x[indices], y[indices])
            if np.isfinite(value):
                samples.append(value)
        if samples:
            low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
            sign_probability = 2 * min(np.mean(np.asarray(samples) <= 0), np.mean(np.asarray(samples) >= 0))
        else:
            low = high = sign_probability = math.nan
        results.append(
            {
                "predictor": predictor,
                "spearman": observed,
                "ci_low": float(low),
                "ci_high": float(high),
                # This is descriptive bootstrap stability, not a null-hypothesis p-value.
                "bootstrap_sign_probability": float(min(sign_probability, 1.0)),
                "clusters": len(unique_clusters),
                "episodes": len(rows),
                "valid_bootstraps": len(samples),
            }
        )
    return results


def gee_associations(
    rows: list[Mapping[str, Any]],
    *,
    outcome: str = "success",
    cluster: str = "task_id",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
) -> list[dict[str, Any]]:
    """Optional logistic GEE; the outcome must be binary."""
    try:
        import statsmodels.api as sm  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("GEE analysis requires the optional 'statsmodels' package") from exc
    y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError(f"Logistic GEE requires binary {outcome} encoded as 0/1")
    x = np.asarray([[float(row[name]) for name in predictors] for row in rows], dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    if np.any(scales == 0):
        raise ValueError("GEE predictors must vary")
    design = np.column_stack([np.ones(len(x)), (x - means) / scales])
    groups = np.asarray([str(row[cluster]) for row in rows])
    result = sm.GEE(y, design, groups=groups, family=sm.families.Binomial()).fit()
    output = [
        {
            "predictor": predictor,
            "coefficient": float(result.params[index]),
            "std_error": float(result.bse[index]),
            "p_value": float(result.pvalues[index]),
        }
        for index, predictor in enumerate(predictors, start=1)
    ]
    adjusted = benjamini_hochberg(np.asarray([row["p_value"] for row in output]))
    for row, q_value in zip(output, adjusted, strict=True):
        row["q_value_bh"] = float(q_value)
    return output
