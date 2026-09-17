from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


EXPECTED_SHA256 = "9c7708761fccb9397fe64bbc0395abcae8c4bf7b0eac081e12b809bf47700d0b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    target = args.target.resolve()
    actual_digest = sha256(wheel)
    if actual_digest != EXPECTED_SHA256:
        raise RuntimeError(f"PyYAML wheel checksum mismatch: {actual_digest}")

    marker = target / ".wheel_sha256"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == actual_digest:
        print(target)
        return
    if target.exists():
        raise RuntimeError(f"Refusing incomplete LIBERO runtime target: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(temporary)
        (temporary / ".wheel_sha256").write_text(actual_digest + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(target)


if __name__ == "__main__":
    main()
