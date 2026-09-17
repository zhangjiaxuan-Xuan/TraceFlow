from __future__ import annotations

import torch

import numpy as np

from scripts.memory.cache_prior_head_features import _deduplicate_observation_rows, _is_cuda_capacity_error


def test_cuda_capacity_error_recognizes_standard_oom() -> None:
    assert _is_cuda_capacity_error(torch.OutOfMemoryError("CUDA out of memory"))
    assert _is_cuda_capacity_error(RuntimeError("CUDA out of memory while allocating tensor"))


def test_cuda_capacity_error_recognizes_allocator_nvml_assert() -> None:
    assert _is_cuda_capacity_error(
        RuntimeError(
            'NVML_SUCCESS == r INTERNAL ASSERT FAILED at "pytorch/c10/cuda/CUDACachingAllocator.cpp":1016'
        )
    )


def test_cuda_capacity_error_rejects_unrelated_runtime_errors() -> None:
    assert not _is_cuda_capacity_error(RuntimeError("feature dimensions differ"))
    assert not _is_cuda_capacity_error(ValueError("out of memory"))


def test_observation_deduplication_preserves_prompt_conditioning_and_order() -> None:
    base = {
        "source_format": "official_hdf5",
        "trajectory_path": "/data/task.hdf5",
        "demo": "demo_0",
        "frame_index": 10,
        "prompt": "pick object",
    }
    rows = [base, {**base}, {**base, "frame_index": 15}, {**base, "prompt": "place object"}, base]
    unique, inverse = _deduplicate_observation_rows(rows)
    assert len(unique) == 3
    np.testing.assert_array_equal(inverse, np.asarray([0, 0, 1, 2, 0]))
    projected = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
    np.testing.assert_array_equal(projected[inverse].ravel(), np.asarray([1.0, 1.0, 2.0, 3.0, 1.0]))
