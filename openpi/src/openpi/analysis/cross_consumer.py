from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from openpi.analysis.bns_mismatch import benjamini_hochberg
from openpi.analysis.bns_mismatch import linear_cka
from openpi.analysis.bns_mismatch import spearman_correlation

DOMAIN_ORDER = ("B", "N", "S")
REPRESENTATIONS = ("raw", "head")


@dataclass(frozen=True)
class ConsumerDomain:
    manifest: tuple[Mapping[str, Any], ...]
    raw: np.ndarray
    head: np.ndarray


@dataclass(frozen=True)
class CrossConsumerConfig:
    task_field: str = "task_id"
    anchor_field: str = "action_id"
    permutations: int = 1999
    bootstraps: int = 1999
    confidence: float = 0.95
    seed: int = 7


def _validate_domain(name: str, consumer: str, data: ConsumerDomain, config: CrossConsumerConfig) -> None:
    if not data.manifest:
        raise ValueError(f"{consumer}.{name} manifest is empty")
    for representation in REPRESENTATIONS:
        values = np.asarray(getattr(data, representation))
        if values.ndim != 2 or values.shape[0] != len(data.manifest):
            raise ValueError(
                f"{consumer}.{name}.{representation} must be [manifest rows, dimension], "
                f"got {values.shape} for {len(data.manifest)} rows"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{consumer}.{name}.{representation} contains non-finite values")
    for field in (config.task_field, config.anchor_field):
        if any(field not in row for row in data.manifest):
            raise KeyError(f"{consumer}.{name} manifest is missing {field!r}")
    anchors = [str(row[config.anchor_field]) for row in data.manifest]
    if len(anchors) != len(set(anchors)):
        raise ValueError(f"{consumer}.{name} has duplicate {config.anchor_field} values")


def _align_exact_anchors(
    pi: ConsumerDomain,
    smol: ConsumerDomain,
    representation: str,
    config: CrossConsumerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    pi_lookup = {str(row[config.anchor_field]): index for index, row in enumerate(pi.manifest)}
    smol_lookup = {str(row[config.anchor_field]): index for index, row in enumerate(smol.manifest)}
    anchors = sorted(set(pi_lookup) & set(smol_lookup))
    if len(anchors) < 2:
        raise ValueError(f"Cross-consumer alignment requires at least two exact shared anchors; found {len(anchors)}")
    pi_indices = np.asarray([pi_lookup[anchor] for anchor in anchors])
    smol_indices = np.asarray([smol_lookup[anchor] for anchor in anchors])
    pi_tasks = np.asarray([str(pi.manifest[index][config.task_field]) for index in pi_indices])
    smol_tasks = np.asarray([str(smol.manifest[index][config.task_field]) for index in smol_indices])
    inconsistent = [anchor for anchor, left, right in zip(anchors, pi_tasks, smol_tasks, strict=True) if left != right]
    if inconsistent:
        preview = inconsistent[:5]
        raise ValueError(f"Shared anchors have inconsistent task labels: {preview}")
    return (
        np.asarray(getattr(pi, representation))[pi_indices],
        np.asarray(getattr(smol, representation))[smol_indices],
        pi_tasks,
        anchors,
    )


def _normalized_centroids(values: np.ndarray, tasks: np.ndarray) -> tuple[list[str], np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Representation contains a zero-norm shared anchor")
    values = values / norms
    task_order = sorted(set(tasks))
    centroids = []
    for task in task_order:
        centroid = values[tasks == task].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm <= np.finfo(np.float64).eps:
            raise ValueError(f"Task {task!r} has a zero-norm centroid")
        centroids.append(centroid / norm)
    return task_order, np.stack(centroids)


def _rdm_upper(centroids: np.ndarray) -> np.ndarray:
    return (1.0 - centroids @ centroids.T)[np.triu_indices(len(centroids), k=1)]


def _rsa(pi_centroids: np.ndarray, smol_centroids: np.ndarray) -> float:
    return spearman_correlation(_rdm_upper(pi_centroids), _rdm_upper(smol_centroids))


def _rsa_inference(
    pi_centroids: np.ndarray,
    smol_centroids: np.ndarray,
    *,
    config: CrossConsumerConfig,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    observed = _rsa(pi_centroids, smol_centroids)
    exceedances = 0
    valid_permutations = 0
    for _ in range(config.permutations):
        value = _rsa(pi_centroids, smol_centroids[rng.permutation(len(smol_centroids))])
        if np.isfinite(value):
            valid_permutations += 1
            exceedances += abs(value) >= abs(observed)
    p_value = (exceedances + 1) / (valid_permutations + 1)

    bootstrap_values: list[float] = []
    for _ in range(config.bootstraps):
        indices = rng.integers(0, len(pi_centroids), size=len(pi_centroids))
        value = _rsa(pi_centroids[indices], smol_centroids[indices])
        if np.isfinite(value):
            bootstrap_values.append(value)
    alpha = 1.0 - config.confidence
    if bootstrap_values:
        ci_low, ci_high = np.quantile(bootstrap_values, [alpha / 2, 1 - alpha / 2])
    else:
        ci_low = ci_high = math.nan
    return {
        "rsa_rho": observed,
        "rsa_p_value": p_value,
        "rsa_ci_low": float(ci_low),
        "rsa_ci_high": float(ci_high),
        "valid_permutations": valid_permutations,
        "valid_bootstraps": len(bootstrap_values),
    }


def analyze_cross_consumer(
    pi_domains: Mapping[str, ConsumerDomain],
    smol_domains: Mapping[str, ConsumerDomain],
    config: CrossConsumerConfig | None = None,
) -> dict[str, Any]:
    """Compare Pi and Smol encodings per B/N/S domain using exact shared anchors."""
    config = config or CrossConsumerConfig()
    if set(pi_domains) != set(DOMAIN_ORDER) or set(smol_domains) != set(DOMAIN_ORDER):
        raise ValueError(f"Both consumers must provide exactly {DOMAIN_ORDER}")
    if config.permutations < 0 or config.bootstraps < 0:
        raise ValueError("Resampling counts must be non-negative")
    if not 0.0 < config.confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    rows: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(DOMAIN_ORDER):
        pi = pi_domains[domain]
        smol = smol_domains[domain]
        _validate_domain(domain, "pi", pi, config)
        _validate_domain(domain, "smol", smol, config)
        for representation_index, representation in enumerate(REPRESENTATIONS):
            pi_values, smol_values, tasks, anchors = _align_exact_anchors(pi, smol, representation, config)
            task_order, pi_centroids = _normalized_centroids(pi_values, tasks)
            smol_task_order, smol_centroids = _normalized_centroids(smol_values, tasks)
            if task_order != smol_task_order:
                raise RuntimeError("Internal task-centroid alignment failure")
            if len(task_order) < 3:
                raise ValueError(f"{domain} requires at least three shared task blocks; found {task_order}")
            rng = np.random.default_rng(config.seed + 101 * domain_index + 1009 * representation_index)
            inference = _rsa_inference(pi_centroids, smol_centroids, config=config, rng=rng)
            rows.append(
                {
                    "domain": domain,
                    "representation": representation,
                    "shared_anchors": len(anchors),
                    "shared_tasks": len(task_order),
                    "pi_dimension": int(pi_values.shape[1]),
                    "smol_dimension": int(smol_values.shape[1]),
                    "linear_cka": linear_cka(pi_values, smol_values),
                    **inference,
                }
            )

    adjusted = benjamini_hochberg(np.asarray([row["rsa_p_value"] for row in rows]))
    for row, q_value in zip(rows, adjusted, strict=True):
        row["rsa_q_value_bh"] = float(q_value)
    return {
        "scope": "Pi-versus-Smol alignment within each B/N/S domain; no B/N/S retrieval mismatch or checkpoint axis",
        "config": vars(config),
        "methodology": {
            "anchors": "exact action_id intersection within each domain",
            "cka": "linear CKA on row-aligned anchors; consumer feature dimensions may differ",
            "rsa": "Spearman correlation of task-centroid RDM upper triangles",
            "inference": "task-block permutation p-values, task-block bootstrap confidence intervals, BH-FDR",
        },
        "comparisons": rows,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_cross_consumer_report(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cross_consumer_alignment.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(report["comparisons"])
    with (output / "cross_consumer_alignment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_alignment(rows, output / "cross_consumer_alignment.png")


def _plot_alignment(rows: list[Mapping[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("PNG output requires matplotlib; install the project's dev dependency group") from exc

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(DOMAIN_ORDER), dtype=np.float64)
    width = 0.35
    colors = {"raw": "#287271", "head": "#D97706"}
    for offset_index, representation in enumerate(REPRESENTATIONS):
        selected = [
            next(row for row in rows if row["domain"] == domain and row["representation"] == representation)
            for domain in DOMAIN_ORDER
        ]
        offset = (offset_index - 0.5) * width
        axes[0].bar(
            x + offset,
            [row["linear_cka"] for row in selected],
            width,
            label=representation,
            color=colors[representation],
        )
        rho = np.asarray([row["rsa_rho"] for row in selected])
        low = np.asarray([row["rsa_ci_low"] for row in selected])
        high = np.asarray([row["rsa_ci_high"] for row in selected])
        axes[1].bar(x + offset, rho, width, label=representation, color=colors[representation])
        axes[1].errorbar(
            x + offset,
            rho,
            yerr=np.vstack([np.maximum(rho - low, 0.0), np.maximum(high - rho, 0.0)]),
            fmt="none",
            ecolor="#222222",
            capsize=3,
        )
    for axis, title in zip(axes, ("Linear CKA", "Task-centroid RSA"), strict=True):
        axis.set_xticks(x, DOMAIN_ORDER)
        axis.set_ylim(-1.0 if "RSA" in title else 0.0, 1.0)
        axis.set_title(title)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
