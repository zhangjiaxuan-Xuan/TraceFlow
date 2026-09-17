#!/usr/bin/env python3
"""Fail on private paths, credentials, nested repositories, and model blobs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

FORBIDDEN = (
    "/" + "data/L202500340",
    "/" + "mnt/",
    "/" + "root/",
    "/" + "home/",
    "/" + "Users/",
    "C:" + "\\Users\\",
)
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
BINARY_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".index", ".npz"}


def _publish_files(root: Path) -> list[Path]:
    """Return exactly the tracked and non-ignored files Git would publish."""
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [root / value.decode() for value in result.stdout.split(b"\0") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    for nested_git in root.rglob(".git"):
        relative = nested_git.relative_to(root)
        if relative == Path(".git"):
            continue
        ignored = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--quiet", str(relative)],
            check=False,
        ).returncode == 0
        if not ignored:
            errors.append(f"nested Git repository: {relative}")
    for path in _publish_files(root):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            errors.append(f"model/data artifact committed to code repository: {relative}")
        if path.stat().st_size > 90 * 1024 * 1024:
            errors.append(f"file exceeds 90 MiB: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for value in FORBIDDEN:
            if value in text:
                errors.append(f"private absolute path {value!r}: {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible credential ({pattern.pattern}): {relative}")
    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print(f"acceptance scan passed: {root}")


if __name__ == "__main__":
    main()
