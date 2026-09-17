from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DOMAIN_ORDER = ("B", "N", "S")
CONSUMER_ORDER = ("pi", "smol")
SIGN_ORDER = ("positive", "negative")


@dataclass(frozen=True)
class DomainData:
    """One memory domain encoded in one fixed consumer representation space."""

    manifest: tuple[Mapping[str, Any], ...]
    raw: np.ndarray
    head: np.ndarray


@dataclass(frozen=True)
class AnalysisConfig:
    task_field: str = "task_id"
    anchor_field: str = "action_id"
    top_k: int = 8
    permutations: int = 1999
    bootstraps: int = 1999
    confidence: float = 0.95
    seed: int = 7


@dataclass(frozen=True)
class StudyUnit:
    consumer: str
    distribution: str
    sign: str
    manifest_path: Path
    manifest_sha256: str
    raw_spec: str
    raw_sha256: str
    head_spec: str
    head_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_path(spec: str) -> Path:
    path_text, separator, _key = spec.rpartition(":")
    return Path(path_text if separator and Path(path_text).suffix == ".npz" else spec)


def _verified_path(base: Path, value: Any, expected_sha256: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.path must be a non-empty string")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{label}.sha256 must be a frozen 64-character digest")
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return path


def load_study_manifest(path: str | Path) -> tuple[StudyUnit, ...]:
    """Load exactly 2 consumers x 3 distributions x 2 signs with frozen inputs."""
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("mismatch study schema_version must be 1")
    raw_units = payload.get("units")
    if not isinstance(raw_units, list):
        raise ValueError("mismatch study units must be a list")
    base = manifest_path.parent
    units: list[StudyUnit] = []
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(raw_units):
        if not isinstance(item, dict):
            raise ValueError(f"units[{index}] must be an object")
        consumer = str(item.get("consumer", ""))
        distribution = str(item.get("distribution", ""))
        sign = str(item.get("sign", ""))
        identity = (consumer, distribution, sign)
        if consumer not in CONSUMER_ORDER or distribution not in DOMAIN_ORDER or sign not in SIGN_ORDER:
            raise ValueError(f"Invalid mismatch unit identity: {identity}")
        if identity in seen:
            raise ValueError(f"Duplicate mismatch unit identity: {identity}")
        seen.add(identity)
        resolved: dict[str, tuple[str, str]] = {}
        for field in ("manifest", "raw", "head"):
            artifact = item.get(field)
            if not isinstance(artifact, dict):
                raise ValueError(f"units[{index}].{field} must contain path and sha256")
            raw_value = artifact.get("path")
            path_value, separator, key = str(raw_value).rpartition(":")
            file_value = (
                path_value if field != "manifest" and separator and Path(path_value).suffix == ".npz" else raw_value
            )
            verified = _verified_path(base, file_value, artifact.get("sha256"), f"units[{index}].{field}")
            spec = (
                f"{verified}:{key}"
                if field != "manifest" and separator and Path(path_value).suffix == ".npz"
                else str(verified)
            )
            resolved[field] = (spec, str(artifact["sha256"]))
        units.append(
            StudyUnit(
                consumer=consumer,
                distribution=distribution,
                sign=sign,
                manifest_path=Path(resolved["manifest"][0]),
                manifest_sha256=resolved["manifest"][1],
                raw_spec=resolved["raw"][0],
                raw_sha256=resolved["raw"][1],
                head_spec=resolved["head"][0],
                head_sha256=resolved["head"][1],
            )
        )
    expected = {
        (consumer, domain, sign) for consumer in CONSUMER_ORDER for domain in DOMAIN_ORDER for sign in SIGN_ORDER
    }
    if seen != expected:
        raise ValueError(f"mismatch study must contain exactly 12 units; missing={sorted(expected - seen)}")
    by_identity = {(unit.consumer, unit.distribution, unit.sign): unit for unit in units}
    for distribution in DOMAIN_ORDER:
        for sign in SIGN_ORDER:
            pi = by_identity[("pi", distribution, sign)]
            smol = by_identity[("smol", distribution, sign)]
            if pi.raw_spec == smol.raw_spec or pi.head_spec == smol.head_spec:
                raise ValueError(
                    f"{distribution}/{sign} must be independently encoded for Pi and Smol; "
                    "raw/head paths cannot be shared"
                )
    return tuple(units)


def load_manifest(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return tuple(rows)


def load_array(spec: str | Path) -> np.ndarray:
    """Load .npy or one array from .npz, using 'path.npz:key' when needed."""
    text = str(spec)
    path_text, separator, key = text.rpartition(":")
    path = Path(path_text if separator and Path(path_text).suffix == ".npz" else text)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix != ".npz":
        raise ValueError(f"Expected .npy or .npz array, got: {spec}")
    archive = np.load(path)
    if separator and path_text == str(path):
        if key not in archive:
            raise KeyError(f"Array {key!r} is absent from {path}; available={archive.files}")
        return archive[key]
    if len(archive.files) != 1:
        raise ValueError(f"{path} contains {archive.files}; select one with '{path}:KEY'")
    return archive[archive.files[0]]


def _validate_domains(domains: Mapping[str, DomainData], config: AnalysisConfig) -> None:
    if set(domains) != set(DOMAIN_ORDER):
        raise ValueError(f"domains must be exactly {DOMAIN_ORDER}, got {sorted(domains)}")
    dimensions: dict[str, set[int]] = {"raw": set(), "head": set()}
    for name in DOMAIN_ORDER:
        domain = domains[name]
        for representation in ("raw", "head"):
            values = np.asarray(getattr(domain, representation))
            if values.ndim != 2 or values.shape[0] != len(domain.manifest):
                raise ValueError(
                    f"{name}.{representation} must have shape [manifest rows, dimension], "
                    f"got {values.shape} for {len(domain.manifest)} rows"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name}.{representation} contains non-finite values")
            dimensions[representation].add(values.shape[1])
        for field in (config.task_field, config.anchor_field):
            if any(field not in row for row in domain.manifest):
                raise KeyError(f"{name} manifest is missing required field {field!r}")
        anchors = [str(row[config.anchor_field]) for row in domain.manifest]
        if len(anchors) != len(set(anchors)):
            raise ValueError(f"{name} manifest has duplicate {config.anchor_field} values")
    for representation, values in dimensions.items():
        if len(values) != 1:
            raise ValueError(f"All {representation} arrays must share a consumer-space dimension, got {values}")
    if config.top_k < 1 or config.permutations < 0 or config.bootstraps < 0:
        raise ValueError("top_k must be positive and resampling counts must be non-negative")
    if not 0.0 < config.confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Representation contains a zero-norm row")
    return values / norms


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranked


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).ravel()
    right = np.asarray(right).ravel()
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Spearman inputs must have equal shape and at least two values")
    ranked_left = _rank(left)
    ranked_right = _rank(right)
    if np.std(ranked_left) == 0 or np.std(ranked_right) == 0:
        return math.nan
    return float(np.corrcoef(ranked_left, ranked_right)[0, 1])


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear CKA for exact, row-aligned shared anchors."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Linear CKA requires two 2-D arrays with the same number of exact anchors")
    if left.shape[0] < 2:
        return math.nan
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(right.T @ right, ord="fro")
    return float(numerator / denominator) if denominator > 0 else math.nan


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite], kind="mergesort")]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _task_centroids(domain: DomainData, representation: str, task_field: str) -> dict[str, np.ndarray]:
    values = _normalize(getattr(domain, representation))
    tasks = np.asarray([str(row[task_field]) for row in domain.manifest])
    output: dict[str, np.ndarray] = {}
    for task in sorted(set(tasks)):
        centroid = values[tasks == task].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            output[task] = centroid / norm
    return output


def _matched_centroids(
    left: DomainData, right: DomainData, representation: str, task_field: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    left_centroids = _task_centroids(left, representation, task_field)
    right_centroids = _task_centroids(right, representation, task_field)
    tasks = sorted(set(left_centroids) & set(right_centroids))
    if len(tasks) < 3:
        raise ValueError(f"At least three shared tasks are required; found {tasks}")
    return tasks, np.stack([left_centroids[t] for t in tasks]), np.stack([right_centroids[t] for t in tasks])


def _upper_rdm(values: np.ndarray) -> np.ndarray:
    similarity = _normalize(values) @ _normalize(values).T
    return (1.0 - similarity)[np.triu_indices(len(values), k=1)]


def _rsa_stat(left: np.ndarray, right: np.ndarray) -> float:
    return spearman_correlation(_upper_rdm(left), _upper_rdm(right))


def _rsa_inference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int,
    bootstraps: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, float | int | None]:
    observed = _rsa_stat(left, right)
    exceedances = 0
    valid_permutations = 0
    for _ in range(permutations):
        permuted = _rsa_stat(left, right[rng.permutation(len(right))])
        if np.isfinite(permuted):
            valid_permutations += 1
            exceedances += abs(permuted) >= abs(observed)
    p_value = (exceedances + 1) / (valid_permutations + 1)
    samples: list[float] = []
    for _ in range(bootstraps):
        indices = rng.integers(0, len(left), size=len(left))
        value = _rsa_stat(left[indices], right[indices])
        if np.isfinite(value):
            samples.append(value)
    alpha = 1.0 - confidence
    if samples:
        low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    else:
        low = high = math.nan
    return {
        "rho": observed,
        "p_value": p_value,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_blocks": len(left),
        "valid_permutations": valid_permutations,
        "valid_bootstraps": len(samples),
    }


def _shared_anchor_values(
    left: DomainData, right: DomainData, representation: str, anchor_field: str
) -> tuple[np.ndarray, np.ndarray]:
    left_lookup = {str(row[anchor_field]): index for index, row in enumerate(left.manifest)}
    right_lookup = {str(row[anchor_field]): index for index, row in enumerate(right.manifest)}
    anchors = sorted(set(left_lookup) & set(right_lookup))
    return (
        np.asarray(getattr(left, representation))[[left_lookup[value] for value in anchors]],
        np.asarray(getattr(right, representation))[[right_lookup[value] for value in anchors]],
    )


def _retrieval_compatibility(
    query: DomainData,
    bank: DomainData,
    *,
    representation: str,
    task_field: str,
    anchor_field: str,
    top_k: int,
    chunk_size: int = 1024,
) -> dict[str, float | int]:
    queries = _normalize(getattr(query, representation))
    keys = _normalize(getattr(bank, representation))
    query_tasks = np.asarray([str(row[task_field]) for row in query.manifest])
    bank_tasks = np.asarray([str(row[task_field]) for row in bank.manifest])
    query_anchors = np.asarray([str(row[anchor_field]) for row in query.manifest])
    bank_anchors = np.asarray([str(row[anchor_field]) for row in bank.manifest])
    purities: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    recalls: list[np.ndarray] = []
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        similarities = queries[start:stop] @ keys.T
        similarities[query_anchors[start:stop, None] == bank_anchors[None, :]] = -np.inf
        available = np.isfinite(similarities).sum(axis=1)
        k = min(top_k, similarities.shape[1])
        if np.any(available < k):
            raise ValueError("A retrieval bank has fewer non-self anchors than top_k")
        selected = np.argpartition(similarities, -k, axis=1)[:, -k:]
        selected_tasks = bank_tasks[selected]
        matches = selected_tasks == query_tasks[start:stop, None]
        purities.append(matches.mean(axis=1))
        recalls.append(matches.any(axis=1))
        for row, task in zip(similarities, query_tasks[start:stop], strict=True):
            same = row[bank_tasks == task]
            different = row[bank_tasks != task]
            same = same[np.isfinite(same)]
            different = different[np.isfinite(different)]
            margins.append(
                np.array([same.max() - different.max()]) if same.size and different.size else np.array([np.nan])
            )
    purity = np.concatenate(purities)
    margin = np.concatenate(margins)
    recall = np.concatenate(recalls)
    return {
        "purity_at_k": float(purity.mean()),
        "recall_at_k": float(recall.mean()),
        "mean_margin": float(np.nanmean(margin)),
        "n_queries": len(queries),
        "top_k": top_k,
    }


def _clustering_metrics(domain: DomainData, representation: str, task_field: str) -> dict[str, float | int]:
    values = _normalize(getattr(domain, representation))
    tasks = np.asarray([str(row[task_field]) for row in domain.manifest])
    same_values: list[np.ndarray] = []
    different_values: list[np.ndarray] = []
    for row in range(len(values)):
        similarities = values[row] @ values.T
        candidates = np.arange(len(values)) != row
        same_values.append(similarities[candidates & (tasks == tasks[row])])
        different_values.append(similarities[candidates & (tasks != tasks[row])])
    same = np.concatenate([item for item in same_values if item.size])
    different = np.concatenate([item for item in different_values if item.size])
    centroids = _task_centroids(domain, representation, task_field)
    labels = sorted(centroids)
    centroid_values = np.stack([centroids[label] for label in labels])
    predicted = np.argmax(values @ centroid_values.T, axis=1)
    truth = np.asarray([labels.index(task) for task in tasks])
    return {
        "rows": len(values),
        "tasks": len(labels),
        "mean_intra_task_cosine": float(same.mean()),
        "mean_inter_task_cosine": float(different.mean()),
        "intra_inter_gap": float(same.mean() - different.mean()),
        "nearest_task_centroid_accuracy": float(np.mean(predicted == truth)),
    }


def _exact_anchor_cross_consumer(
    left: DomainData,
    right: DomainData,
    representation: str,
    config: AnalysisConfig,
) -> dict[str, float | int | None]:
    left_lookup = {str(row[config.anchor_field]): index for index, row in enumerate(left.manifest)}
    right_lookup = {str(row[config.anchor_field]): index for index, row in enumerate(right.manifest)}
    anchors = sorted(set(left_lookup) & set(right_lookup))
    if len(anchors) < 2:
        return {"exact_anchors": len(anchors), "linear_cka": None, "rsa_spearman": None}
    left_rows = np.asarray(getattr(left, representation))[[left_lookup[anchor] for anchor in anchors]]
    right_rows = np.asarray(getattr(right, representation))[[right_lookup[anchor] for anchor in anchors]]
    for anchor in anchors:
        left_task = str(left.manifest[left_lookup[anchor]][config.task_field])
        right_task = str(right.manifest[right_lookup[anchor]][config.task_field])
        if left_task != right_task:
            raise ValueError(f"Exact anchor {anchor!r} has conflicting task labels: {left_task!r} vs {right_task!r}")
    rsa = _rsa_stat(left_rows, right_rows) if len(anchors) >= 3 else math.nan
    return {
        "exact_anchors": len(anchors),
        "linear_cka": linear_cka(left_rows, right_rows),
        "rsa_spearman": None if not np.isfinite(rsa) else rsa,
    }


def analyze_bns_mismatch(domains: Mapping[str, DomainData], config: AnalysisConfig | None = None) -> dict[str, Any]:
    """Analyze B/N/S within one consumer space; no checkpoint-stage variable is accepted."""
    config = config or AnalysisConfig()
    _validate_domains(domains, config)
    report: dict[str, Any] = {"config": vars(config), "domains": {}, "representations": {}}
    report["methodology"] = {
        "analysis_unit": "one fixed consumer representation space with B/N/S data domains",
        "centroid": "mean cosine between matched task centroids",
        "rsa": "Spearman correlation of task-centroid RDM upper triangles",
        "rsa_inference": "task-block permutation p-values and task-block bootstrap confidence intervals",
        "cka": "linear CKA on exact shared anchors only; unavailable cross-domain pairs are null in JSON",
        "retrieval": "directional head-space cosine top-k with exact self anchors excluded",
        "multiplicity": "Benjamini-Hochberg FDR across B/N/S pairwise RSA tests",
        "excluded_methods": ["generic Mantel test", "training-checkpoint mismatch"],
    }
    for name in DOMAIN_ORDER:
        report["domains"][name] = {
            "rows": len(domains[name].manifest),
            "tasks": sorted({str(row[config.task_field]) for row in domains[name].manifest}),
            "raw_dimension": int(domains[name].raw.shape[1]),
            "head_dimension": int(domains[name].head.shape[1]),
            "clustering": {
                representation: _clustering_metrics(domains[name], representation, config.task_field)
                for representation in ("raw", "head")
            },
        }

    pairwise: list[dict[str, Any]] = []
    for representation_index, representation in enumerate(("raw", "head")):
        centroid = np.eye(3, dtype=np.float64)
        rsa = np.eye(3, dtype=np.float64)
        cka = np.eye(3, dtype=np.float64)
        rsa_results: dict[str, dict[str, Any]] = {}
        for left_index, left_name in enumerate(DOMAIN_ORDER):
            for right_index in range(left_index + 1, len(DOMAIN_ORDER)):
                right_name = DOMAIN_ORDER[right_index]
                tasks, left_centroids, right_centroids = _matched_centroids(
                    domains[left_name], domains[right_name], representation, config.task_field
                )
                centroid_value = float(np.mean(np.sum(left_centroids * right_centroids, axis=1)))
                rng = np.random.default_rng(config.seed + 1000 * representation_index + 31 * left_index + right_index)
                inference = _rsa_inference(
                    left_centroids,
                    right_centroids,
                    permutations=config.permutations,
                    bootstraps=config.bootstraps,
                    confidence=config.confidence,
                    rng=rng,
                )
                shared_left, shared_right = _shared_anchor_values(
                    domains[left_name], domains[right_name], representation, config.anchor_field
                )
                cka_value = linear_cka(shared_left, shared_right) if len(shared_left) >= 2 else math.nan
                centroid[left_index, right_index] = centroid[right_index, left_index] = centroid_value
                rsa[left_index, right_index] = rsa[right_index, left_index] = float(inference["rho"])
                cka[left_index, right_index] = cka[right_index, left_index] = cka_value
                key = f"{left_name}-{right_name}"
                inference.update({"tasks": tasks, "shared_anchors": len(shared_left)})
                rsa_results[key] = inference
                pairwise.append(
                    {
                        "representation": representation,
                        "left": left_name,
                        "right": right_name,
                        "centroid_cosine": centroid_value,
                        "rsa_rho": inference["rho"],
                        "rsa_p_value": inference["p_value"],
                        "rsa_ci_low": inference["ci_low"],
                        "rsa_ci_high": inference["ci_high"],
                        "linear_cka": cka_value,
                        "shared_anchors": len(shared_left),
                    }
                )
        report["representations"][representation] = {
            "centroid_cosine": centroid.tolist(),
            "rsa_spearman": rsa.tolist(),
            "linear_cka": cka.tolist(),
            "rsa_inference": rsa_results,
        }

    p_values = np.asarray([row["rsa_p_value"] for row in pairwise], dtype=np.float64)
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(pairwise, q_values, strict=True):
        row["rsa_q_value_bh"] = float(q_value)

    retrieval = np.empty((3, 3), dtype=np.float64)
    retrieval_details: dict[str, Any] = {}
    for query_index, query_name in enumerate(DOMAIN_ORDER):
        for bank_index, bank_name in enumerate(DOMAIN_ORDER):
            value = _retrieval_compatibility(
                domains[query_name],
                domains[bank_name],
                representation="head",
                task_field=config.task_field,
                anchor_field=config.anchor_field,
                top_k=config.top_k,
            )
            retrieval[query_index, bank_index] = float(value["purity_at_k"])
            retrieval_details[f"{query_name}->{bank_name}"] = value
    report["retrieval_compatibility"] = {
        "purity_at_k": retrieval.tolist(),
        "details": retrieval_details,
        "direction": "rows=query domain, columns=memory bank domain",
        "representation": "head",
    }
    report["pairwise"] = pairwise
    return report


def analyze_mismatch_study(units: tuple[StudyUnit, ...], config: AnalysisConfig | None = None) -> dict[str, Any]:
    """Analyze the 12 independent consumer x distribution x sign representation units."""
    config = config or AnalysisConfig()
    loaded: dict[tuple[str, str, str], DomainData] = {}
    provenance: list[dict[str, Any]] = []
    for unit in units:
        key = (unit.consumer, unit.distribution, unit.sign)
        if key in loaded:
            raise ValueError(f"Duplicate mismatch unit: {key}")
        loaded[key] = DomainData(
            manifest=load_manifest(unit.manifest_path),
            raw=load_array(unit.raw_spec),
            head=load_array(unit.head_spec),
        )
        provenance.append(
            {
                "consumer": unit.consumer,
                "distribution": unit.distribution,
                "sign": unit.sign,
                "manifest": {"path": str(unit.manifest_path), "sha256": unit.manifest_sha256},
                "raw": {"spec": unit.raw_spec, "sha256": unit.raw_sha256},
                "head": {"spec": unit.head_spec, "sha256": unit.head_sha256},
            }
        )
    expected = {
        (consumer, domain, sign) for consumer in CONSUMER_ORDER for domain in DOMAIN_ORDER for sign in SIGN_ORDER
    }
    if set(loaded) != expected:
        raise ValueError(f"Expected exactly 12 mismatch units; missing={sorted(expected - set(loaded))}")

    signs: dict[str, Any] = {}
    for sign in SIGN_ORDER:
        consumers = {
            consumer: analyze_bns_mismatch(
                {domain: loaded[(consumer, domain, sign)] for domain in DOMAIN_ORDER}, config
            )
            for consumer in CONSUMER_ORDER
        }
        cross_consumer: dict[str, Any] = {}
        for domain in DOMAIN_ORDER:
            cross_consumer[domain] = {
                representation: _exact_anchor_cross_consumer(
                    loaded[("pi", domain, sign)],
                    loaded[("smol", domain, sign)],
                    representation,
                    config,
                )
                for representation in ("raw", "head")
            }
        signs[sign] = {"consumers": consumers, "cross_consumer_exact_anchors": cross_consumer}
    return {
        "schema_version": 1,
        "analysis_unit": "2 consumers x 3 distributions x 2 signs; signs are never pooled",
        "config": vars(config),
        "inputs": provenance,
        "signs": signs,
    }


def write_study_report(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "mismatch_study.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for sign in SIGN_ORDER:
        for consumer in CONSUMER_ORDER:
            write_report(report["signs"][sign]["consumers"][consumer], output / sign / consumer)
        cross = report["signs"][sign]["cross_consumer_exact_anchors"]
        rows = [
            {"distribution": domain, "representation": representation, **cross[domain][representation]}
            for domain in DOMAIN_ORDER
            for representation in ("raw", "head")
        ]
        with (output / sign / "cross_consumer_exact_anchors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pairwise = list(report["pairwise"])
    with (output / "pairwise_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise[0]))
        writer.writeheader()
        writer.writerows(pairwise)
    matrices = {
        "raw_centroid_cosine": report["representations"]["raw"]["centroid_cosine"],
        "head_centroid_cosine": report["representations"]["head"]["centroid_cosine"],
        "raw_rsa_spearman": report["representations"]["raw"]["rsa_spearman"],
        "head_rsa_spearman": report["representations"]["head"]["rsa_spearman"],
        "raw_linear_cka": report["representations"]["raw"]["linear_cka"],
        "head_linear_cka": report["representations"]["head"]["linear_cka"],
        "head_retrieval_purity": report["retrieval_compatibility"]["purity_at_k"],
    }
    for name, matrix in matrices.items():
        np.savetxt(
            output / f"{name}.csv",
            np.asarray(matrix, dtype=np.float64),
            delimiter=",",
            header=",".join(DOMAIN_ORDER),
            comments="",
        )
    _plot_matrices(matrices, output / "bns_mismatch_heatmaps.png")


def _plot_matrices(matrices: Mapping[str, Any], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("PNG output requires matplotlib; install the project's dev dependency group") from exc
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axis, (name, values) in zip(axes.flat, matrices.items(), strict=False):
        matrix = np.asarray(values, dtype=np.float64)
        image = axis.imshow(matrix, vmin=-1.0 if "rsa" in name else 0.0, vmax=1.0, cmap="coolwarm")
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.set_xticks(range(3), DOMAIN_ORDER)
        axis.set_yticks(range(3), DOMAIN_ORDER)
        for row in range(3):
            for column in range(3):
                label = "NA" if not np.isfinite(matrix[row, column]) else f"{matrix[row, column]:.2f}"
                axis.text(column, row, label, ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes.flat[len(matrices) :]:
        axis.axis("off")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
