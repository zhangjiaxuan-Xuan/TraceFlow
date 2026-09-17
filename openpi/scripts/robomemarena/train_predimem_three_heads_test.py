import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.robomemarena.train_predimem_three_heads import BalancedProgressPairSampler
from scripts.robomemarena.train_predimem_three_heads import _exact_search
from scripts.robomemarena.train_predimem_three_heads import _positive_pair_mask
from scripts.robomemarena.train_predimem_three_heads import _pack_actions


def test_pack_actions_supports_traceflow_lerobot_parquet(tmp_path) -> None:
    source = tmp_path / "episode.parquet"
    values = np.arange(42, dtype=np.float32).reshape(3, 14)
    table = pa.table({"action": pa.array(values.tolist(), type=pa.list_(pa.float32(), 14))})
    pq.write_table(table, source)
    output = tmp_path / "actions.npz"

    _pack_actions(
        [
            {
                "action_id": "episode-0",
                "source_format": "lerobot_traceflow",
                "segment_paths": [str(source)],
            }
        ],
        output,
        workers=2,
    )

    with np.load(output) as packed:
        assert packed["actions"].shape == (3, 32)
        np.testing.assert_array_equal(packed["actions"][:, :14], values)
        np.testing.assert_array_equal(packed["actions"][:, 14:], 0.0)


def test_pack_actions_rejects_negative_workers(tmp_path) -> None:
    with unittest.TestCase().assertRaisesRegex(ValueError, "workers"):
        _pack_actions([], tmp_path / "actions.npz", workers=-1)


class ExactSearchTest(unittest.TestCase):
    def test_chunked_cpu_search_matches_direct_topk(self) -> None:
        rng = np.random.default_rng(7)
        keys = rng.normal(size=(31, 12)).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        query_indices = np.asarray([1, 7, 13, 19, 25], dtype=np.int64)
        key_indices = np.asarray([0, 2, 3, 5, 8, 11, 17, 23, 29], dtype=np.int64)
        actual = _exact_search(
            keys,
            query_indices,
            key_indices,
            4,
            device="cpu",
            query_batch_size=2,
        )
        expected = np.argsort(-(keys[query_indices] @ keys[key_indices].T), axis=1)[:, :4]
        np.testing.assert_array_equal(actual, expected)

    def test_empty_key_bank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one key"):
            _exact_search(
                np.ones((2, 2), dtype=np.float32),
                np.asarray([0]),
                np.asarray([], dtype=np.int64),
                1,
            )


class BalancedProgressPairSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        records = []
        for task_id in range(1, 27):
            for stage_id in range(3):
                for trajectory_id in range(8):
                    for progress in np.linspace(0.0, 1.0, 41):
                        records.append((task_id, stage_id, trajectory_id + task_id * 100, progress))
        values = np.asarray(records, dtype=np.float64)
        self.task = values[:, 0].astype(np.int64)
        self.stage = values[:, 1].astype(np.int64)
        self.trajectory = values[:, 2].astype(np.int64)
        self.progress = values[:, 3].astype(np.float32)
        self.indices = np.arange(len(records), dtype=np.int64)

    def test_every_anchor_has_cross_trajectory_positive(self) -> None:
        sampler = BalancedProgressPairSampler(
            self.indices,
            self.task,
            self.stage,
            self.progress,
            self.trajectory,
            batch_size=1024,
            positive_progress_radius=0.025,
        )
        batch = sampler.sample(np.random.default_rng(7))
        _, positive = _positive_pair_mask(
            torch.from_numpy(self.task[batch]),
            torch.from_numpy(self.stage[batch]),
            torch.from_numpy(self.progress[batch]),
            torch.from_numpy(self.trajectory[batch]),
            positive_progress_radius=0.025,
            require_cross_trajectory_positive=True,
        )
        self.assertEqual(len(batch), 1024)
        self.assertTrue(bool(positive.any(dim=1).all()))
        counts = np.unique(self.task[batch], return_counts=True)[1]
        self.assertLessEqual(int(counts.max() - counts.min()), 2)

    def test_sampling_is_deterministic(self) -> None:
        sampler = BalancedProgressPairSampler(
            self.indices,
            self.task,
            self.stage,
            self.progress,
            self.trajectory,
            batch_size=512,
            positive_progress_radius=0.025,
        )
        first = sampler.sample(np.random.default_rng(19))
        second = sampler.sample(np.random.default_rng(19))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
