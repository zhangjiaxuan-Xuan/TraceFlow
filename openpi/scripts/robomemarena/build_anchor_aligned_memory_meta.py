from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-meta", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("anchor_forward_v1", "dense_frame_v3"),
        default="anchor_forward_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_meta.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output_meta}")
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    source = torch.load(args.source_meta, map_location="cpu", weights_only=False)
    if not isinstance(source, list) or not source:
        raise TypeError("Source memory metadata must be a non-empty list")

    by_action: dict[str, list[dict]] = {}
    for row in rows:
        by_action.setdefault(str(row["action_id"]), []).append(row)

    manifest_sha = _sha256(args.manifest)
    output = []
    used: set[tuple[str, int]] = set()
    for index, entry in enumerate(source):
        action_id = str(entry["action_id"])
        candidates = by_action.get(action_id, [])
        if not candidates:
            raise KeyError(f"Memory entry {index} has no manifest trajectory: {action_id}")
        stage = int(entry.get("stage_index", -1))
        progress = float(entry["anchor_progress"])
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                int(item[1]["stage_index"]) != stage,
                abs(float(item[1]["anchor_progress"]) - progress),
            ),
        )
        local_index, row = ranked[0]
        identity = (action_id, local_index)
        error = abs(float(row["anchor_progress"]) - progress)
        if identity in used or int(row["stage_index"]) != stage or error > 1e-6:
            raise RuntimeError(
                f"Ambiguous anchor mapping at memory entry {index}: "
                f"action={action_id} stage={stage} progress={progress} error={error}"
            )
        used.add(identity)
        upgraded = dict(entry)
        upgraded["anchor_frame"] = int(row["global_frame_index"])
        upgraded["action_alignment"] = args.protocol
        upgraded["anchor_manifest_sha256"] = manifest_sha
        output.append(upgraded)

    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_meta.with_name(f".{args.output_meta.name}.tmp")
    torch.save(output, temporary)
    temporary.replace(args.output_meta)
    record = {
        "protocol": args.protocol,
        "entries": len(output),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "source_meta": str(args.source_meta.resolve()),
        "source_meta_sha256": _sha256(args.source_meta),
        "output_meta": str(args.output_meta.resolve()),
        "output_meta_sha256": _sha256(args.output_meta),
    }
    args.output_meta.with_suffix(args.output_meta.suffix + ".json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
