#!/usr/bin/env python3
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from openpi.experiments.topk_sweep import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
