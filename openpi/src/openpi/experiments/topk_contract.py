from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


class TopKContractError(ValueError):
    """Raised when a frozen top-k selection cannot be trusted."""


@dataclasses.dataclass(frozen=True)
class TopKSelection:
    consumer: str
    positive_memory_per_task: int | None
    positive_top_k: int
    negative_memory_per_task: int | None
    negative_top_k: int
    prior_top_k: int | None
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(payload: dict[str, Any], key: str, *, required: bool) -> int | None:
    value = payload.get(key)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        requirement = "a positive integer" if required else "null or a positive integer"
        raise TopKContractError(f"{key} must be {requirement}; got {value!r}")
    return value


def load_topk_selection(
    path: str | Path,
    expected_sha256: str,
    *,
    consumer: str,
) -> TopKSelection:
    selection_path = Path(path).expanduser().resolve()
    if not selection_path.is_file():
        raise TopKContractError(f"missing top-k selection: {selection_path}")
    if not expected_sha256 or expected_sha256 == "REQUIRED":
        raise TopKContractError("top-k selection requires a frozen SHA-256 digest")
    actual_sha256 = sha256_file(selection_path)
    if actual_sha256 != expected_sha256:
        raise TopKContractError(f"top-k selection digest mismatch: expected {expected_sha256}, got {actual_sha256}")
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TopKContractError(f"invalid top-k selection JSON: {selection_path}") from error
    if not isinstance(payload, dict):
        raise TopKContractError("top-k selection must contain a JSON object")
    if payload.get("schema_version") not in (1, 2):
        raise TopKContractError("top-k selection schema_version must be 1 or 2")
    selected_consumer = str(payload.get("consumer", ""))
    if not selected_consumer and payload.get("selection") == "smol_final_selection":
        selected_consumer = "smol"
    if selected_consumer != consumer:
        raise TopKContractError(f"top-k selection consumer mismatch: expected {consumer!r}, got {selected_consumer!r}")
    values = payload
    if payload.get("selection") == "smol_final_selection":
        positive = payload.get("positive")
        negative = payload.get("negative", payload.get("positive_negative"))
        if not isinstance(positive, dict) or not isinstance(negative, dict):
            raise TopKContractError("Smol final selection is missing positive or negative results")
        values = {
            "positive_memory_per_task": positive.get("count_per_task", positive.get("success_count")),
            "positive_top_k": positive.get("top_k", positive.get("positive_top_k")),
            "negative_memory_per_task": negative.get("count_per_task", negative.get("failure_count")),
            "negative_top_k": negative.get("top_k", negative.get("negative_top_k")),
        }
    else:
        values = {
            **values,
            "positive_memory_per_task": payload.get("positive_memory_per_task", payload.get("positive_count")),
            "negative_memory_per_task": payload.get("negative_memory_per_task", payload.get("negative_count")),
        }
    prior_top_k = _positive_int(values, "prior_top_k", required=False)
    return TopKSelection(
        consumer=selected_consumer,
        positive_memory_per_task=_positive_int(values, "positive_memory_per_task", required=False),
        positive_top_k=int(_positive_int(values, "positive_top_k", required=True)),
        negative_memory_per_task=_positive_int(values, "negative_memory_per_task", required=False),
        negative_top_k=int(_positive_int(values, "negative_top_k", required=True)),
        prior_top_k=prior_top_k,
        path=selection_path,
        sha256=actual_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and print a frozen top-k selection")
    parser.add_argument("--path", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--consumer", choices=("pi", "smol"), required=True)
    parser.add_argument(
        "--field",
        choices=("positive", "negative", "positive-count", "negative-count", "prior", "tsv"),
        default="tsv",
    )
    args = parser.parse_args(argv)
    selection = load_topk_selection(args.path, args.sha256, consumer=args.consumer)
    values = {
        "positive": selection.positive_top_k,
        "negative": selection.negative_top_k,
        "positive-count": selection.positive_memory_per_task,
        "negative-count": selection.negative_memory_per_task,
        "prior": selection.prior_top_k,
    }
    if args.field == "tsv":
        prior = "" if selection.prior_top_k is None else str(selection.prior_top_k)
        print(f"{selection.positive_top_k}\t{selection.negative_top_k}\t{prior}")
    else:
        value = values[args.field]
        if value is None:
            raise TopKContractError(f"{args.consumer} selection does not define {args.field}_top_k")
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
