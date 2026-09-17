from __future__ import annotations

import numpy as np
import pytest

from openpi.task_head.memory_init import PackedActionStore


def _write(path, *, ids, offsets=(0, 1, 2)) -> None:
    np.savez(
        path,
        actions=np.zeros((2, 32), dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        ids=np.asarray(ids),
    )


def test_packed_action_store_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "actions.npz"
    _write(path, ids=["same", "same"])
    with pytest.raises(ValueError, match="unique"):
        PackedActionStore(str(path))


def test_packed_action_store_rejects_nonmonotonic_offsets(tmp_path) -> None:
    path = tmp_path / "actions.npz"
    _write(path, ids=["a", "b"], offsets=(0, 2, 1))
    with pytest.raises(ValueError, match="nondecreasing"):
        PackedActionStore(str(path))
