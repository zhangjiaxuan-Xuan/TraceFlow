from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


class MatchProtocolError(ValueError):
    pass


POLICIES = ("pi", "smol")
PROVIDERS = ("pi-mem", "smol-mem")


@dataclasses.dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str


@dataclasses.dataclass(frozen=True)
class FrozenSelection:
    consumer: str
    positive_count: int
    positive_top_k: int
    negative_count: int
    negative_top_k: int
    path: Path
    sha256: str


@dataclasses.dataclass(frozen=True)
class Policy:
    consumer: str
    checkpoint_dir: Path
    checkpoint_identity: Artifact
    head: Artifact
    selection: FrozenSelection
    python: str


@dataclasses.dataclass(frozen=True)
class BankEncoding:
    root: Path
    summary: Artifact
    positive_count_per_task: int
    negative_count_per_task: int
    positive: Mapping[str, Artifact]
    negative: Mapping[str, Artifact]


@dataclasses.dataclass(frozen=True)
class MatchExperiment:
    manifest_path: Path
    manifest_sha256: str
    run_root: Path
    gpu_pool: tuple[str, ...]
    ports: tuple[int, ...]
    resume: bool
    policies: Mapping[str, Policy]
    providers: Mapping[str, Mapping[str, BankEncoding]]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_digest(payload: Mapping[str, Any]) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _resolve(base: Path, value: Any, label: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise MatchProtocolError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise MatchProtocolError(f"missing {label} {kind}: {path}")
    return path


def _output_path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MatchProtocolError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _artifact(base: Path, payload: Any, label: str) -> Artifact:
    if not isinstance(payload, dict):
        raise MatchProtocolError(f"{label} must contain path and sha256")
    path = _resolve(base, payload.get("path"), label)
    expected = payload.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise MatchProtocolError(f"{label}.sha256 must be a frozen 64-character digest")
    actual = sha256_file(path)
    if actual != expected:
        raise MatchProtocolError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return Artifact(path, actual)


def _positive_int(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> int:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            value = None
            break
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MatchProtocolError(f"{label} must be a positive integer; got {value!r}")
    return value


def load_selection(base: Path, payload: Any, consumer: str) -> FrozenSelection:
    artifact = _artifact(base, payload, f"policies.{consumer}.selection")
    data = json.loads(artifact.path.read_text(encoding="utf-8"))
    if consumer == "pi":
        if data.get("schema_version") != 1:
            raise MatchProtocolError("Pi final selection schema_version must be 1")
        if data.get("consumer") != "pi":
            raise MatchProtocolError(f"selection consumer mismatch: expected pi, got {data.get('consumer')}")
        positive_count = _positive_int(data, ("positive_memory_per_task",), "positive_memory_per_task")
        positive_top_k = _positive_int(data, ("positive_top_k",), "positive_top_k")
        negative_count = _positive_int(data, ("negative_memory_per_task",), "negative_memory_per_task")
        negative_top_k = _positive_int(data, ("negative_top_k",), "negative_top_k")
    else:
        if data.get("schema_version") != 2:
            raise MatchProtocolError("Smol final selection schema_version must be 2")
        if data.get("selection") != "smol_final_selection":
            raise MatchProtocolError("Smol schema2 selection must be smol_final_selection")
        if data.get("consumer") not in (None, "smol"):
            raise MatchProtocolError(f"selection consumer mismatch: expected smol, got {data.get('consumer')}")
        positive_count = _positive_int(data, ("positive", "count_per_task"), "positive.count_per_task")
        positive_top_k = _positive_int(data, ("positive", "top_k"), "positive.top_k")
        negative_count = _positive_int(data, ("negative", "count_per_task"), "negative.count_per_task")
        negative_top_k = _positive_int(data, ("negative", "top_k"), "negative.top_k")
    return FrozenSelection(
        consumer,
        positive_count,
        positive_top_k,
        negative_count,
        negative_top_k,
        artifact.path,
        artifact.sha256,
    )


def _bank_encoding(base: Path, payload: Any, label: str) -> BankEncoding:
    if not isinstance(payload, dict):
        raise MatchProtocolError(f"{label} must be an object")
    root = _resolve(base, payload.get("root"), f"{label}.root", directory=True)
    summary = _artifact(base, payload.get("summary"), f"{label}.summary")
    positive_count = _positive_int(payload, ("positive_count_per_task",), f"{label}.positive_count_per_task")
    negative_count = _positive_int(payload, ("negative_count_per_task",), f"{label}.negative_count_per_task")
    outcomes: dict[str, dict[str, Artifact]] = {}
    required = {
        "positive": ("meta", "index", "actions"),
        "negative": ("meta", "index", "actions"),
    }
    for sign, names in required.items():
        group = payload.get(sign)
        if not isinstance(group, dict):
            raise MatchProtocolError(f"{label}.{sign} must be an artifact object")
        outcomes[sign] = {name: _artifact(base, group.get(name), f"{label}.{sign}.{name}") for name in names}
    contained = [summary, *outcomes["positive"].values(), *outcomes["negative"].values()]
    if any(root not in artifact.path.parents for artifact in contained):
        raise MatchProtocolError(f"all {label} artifacts must be contained by its root")
    return BankEncoding(root, summary, positive_count, negative_count, outcomes["positive"], outcomes["negative"])


def load_experiment(path: str | Path) -> MatchExperiment:
    manifest = Path(path).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise MatchProtocolError("match manifest schema_version must be 1")
    base = manifest.parent
    gpu_pool = tuple(str(value) for value in payload.get("gpu_pool", (0, 1, 2, 3)))
    ports = tuple(int(value) for value in payload.get("ports", (8200, 8300, 8400, 8500)))
    if len(gpu_pool) != 4 or len(set(gpu_pool)) != 4:
        raise MatchProtocolError("gpu_pool must contain exactly four unique GPU identifiers")
    if len(ports) != 4 or len(set(ports)) != 4:
        raise MatchProtocolError("ports must contain exactly four unique values")
    raw_policies = payload.get("policies")
    if not isinstance(raw_policies, dict) or set(raw_policies) != set(POLICIES):
        raise MatchProtocolError(f"policies must be exactly {POLICIES}")
    policies: dict[str, Policy] = {}
    for consumer in POLICIES:
        item = raw_policies[consumer]
        if not isinstance(item, dict):
            raise MatchProtocolError(f"policies.{consumer} must be an object")
        checkpoint_dir = _resolve(
            base, item.get("checkpoint_dir"), f"policies.{consumer}.checkpoint_dir", directory=True
        )
        identity = _artifact(base, item.get("checkpoint_identity"), f"policies.{consumer}.checkpoint_identity")
        if identity.path.parent != checkpoint_dir and checkpoint_dir not in identity.path.parents:
            raise MatchProtocolError(f"{consumer} checkpoint_identity must be inside checkpoint_dir")
        policies[consumer] = Policy(
            consumer,
            checkpoint_dir,
            identity,
            _artifact(base, item.get("head"), f"policies.{consumer}.head"),
            load_selection(base, item.get("selection"), consumer),
            str(item.get("python", "")),
        )
        if not policies[consumer].python:
            raise MatchProtocolError(f"policies.{consumer}.python is required")
    raw_providers = payload.get("providers")
    if not isinstance(raw_providers, dict) or set(raw_providers) != set(PROVIDERS):
        raise MatchProtocolError(f"providers must be exactly {PROVIDERS}")
    providers: dict[str, dict[str, BankEncoding]] = {}
    for provider in PROVIDERS:
        item = raw_providers[provider]
        encodings = item.get("encodings") if isinstance(item, dict) else None
        if not isinstance(encodings, dict) or set(encodings) != set(POLICIES):
            raise MatchProtocolError(f"providers.{provider}.encodings must be exactly {POLICIES}")
        providers[provider] = {
            consumer: _bank_encoding(base, encodings[consumer], f"providers.{provider}.encodings.{consumer}")
            for consumer in POLICIES
        }
    for provider in PROVIDERS:
        for consumer in POLICIES:
            selection = policies[consumer].selection
            encoding = providers[provider][consumer]
            if encoding.positive_count_per_task != selection.positive_count:
                raise MatchProtocolError(
                    f"{provider}/{consumer} positive count does not match frozen selection: "
                    f"{encoding.positive_count_per_task} != {selection.positive_count}"
                )
            if encoding.negative_count_per_task != selection.negative_count:
                raise MatchProtocolError(
                    f"{provider}/{consumer} negative count does not match frozen selection: "
                    f"{encoding.negative_count_per_task} != {selection.negative_count}"
                )
            if consumer == "smol":
                expected_paths = {
                    ("positive", "meta"): encoding.root / "positive/gpm_memory_meta.pt",
                    ("positive", "index"): encoding.root / "positive/gpm_memory.index",
                    ("positive", "actions"): encoding.root / "positive/gpm_memory_actions.npz",
                    ("negative", "meta"): encoding.root / "negative/gpm_negative_memory_meta.pt",
                    ("negative", "index"): encoding.root / "negative/gpm_negative_memory.index",
                    ("negative", "actions"): encoding.root / "negative/gpm_negative_memory_actions.npz",
                }
                for (sign, name), expected_path in expected_paths.items():
                    actual_path = getattr(encoding, sign)[name].path
                    if actual_path != expected_path:
                        raise MatchProtocolError(
                            f"Smol runtime requires conventional bank path {expected_path}; got {actual_path}"
                        )
    return MatchExperiment(
        manifest,
        sha256_file(manifest),
        _output_path(base, payload.get("run_root"), "run_root"),
        gpu_pool,
        ports,
        bool(payload.get("resume", True)),
        policies,
        providers,
    )


def job_identity(experiment: MatchExperiment, consumer: str, provider: str) -> dict[str, Any]:
    policy = experiment.policies[consumer]
    bank = experiment.providers[provider][consumer]
    payload = {
        "schema_version": 1,
        "suite": "libero_10",
        "policy_consumer": consumer,
        "memory_provider": provider,
        "guidance": "v0-positive-negative",
        "seed": 7,
        "episodes_per_task": 100,
        "batch_size": 8,
        "checkpoint_sha256": policy.checkpoint_identity.sha256,
        "head_sha256": policy.head.sha256,
        "selection_sha256": policy.selection.sha256,
        "positive_count": policy.selection.positive_count,
        "positive_top_k": policy.selection.positive_top_k,
        "negative_count": policy.selection.negative_count,
        "negative_top_k": policy.selection.negative_top_k,
        "bank_summary_sha256": bank.summary.sha256,
        "bank_positive_count_per_task": bank.positive_count_per_task,
        "bank_negative_count_per_task": bank.negative_count_per_task,
        "positive_artifacts": {key: value.sha256 for key, value in bank.positive.items()},
        "negative_artifacts": {key: value.sha256 for key, value in bank.negative.items()},
        "launcher_sha256": sha256_file(
            Path(__file__).resolve().parents[3] / "scripts/experiments/run_memory_provider_match.sh"
        ),
    }
    return {**payload, "identity_sha256": identity_digest(payload)}
