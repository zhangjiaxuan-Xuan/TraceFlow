"""Small dependency-light CLI used by shell entrypoints and acceptance tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import assets, checkpoints, config
from .summary import summarize_libero


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(prog="traceflow")
    commands = parser.add_subparsers(dest="command", required=True)
    assets_parser = commands.add_parser("assets")
    assets_parser.add_argument("groups", nargs="+")
    assets_parser.add_argument("--revision")
    assets_parser.add_argument("--no-semantic", action="store_true")
    checkpoint_parser = commands.add_parser("checkpoints")
    checkpoint_parser.add_argument("model", choices=("pi05", "predimem-upper", "predimem-vla", "arena"))
    config_parser = commands.add_parser("config")
    config_parser.add_argument("name")
    summary_parser = commands.add_parser("summarize-libero")
    summary_parser.add_argument("run_root", type=Path)
    summary_parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.command == "assets":
        print(assets.resolve_snapshot(args.groups, revision=args.revision, semantic=not args.no_semantic))
    elif args.command == "checkpoints":
        if args.model == "pi05":
            print(checkpoints.resolve_pi05(_root()))
        elif args.model == "predimem-upper":
            print(checkpoints.resolve_predimem_upper())
        elif args.model == "predimem-vla":
            print(checkpoints.resolve_predimem_vla(_root()))
        else:
            print(json.dumps({"upper": str(checkpoints.resolve_predimem_upper()), "vla": str(checkpoints.resolve_predimem_vla(_root()))}))
    elif args.command == "config":
        print(json.dumps(config.load(args.name), indent=2, sort_keys=True))
    else:
        print(json.dumps(summarize_libero(args.run_root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

