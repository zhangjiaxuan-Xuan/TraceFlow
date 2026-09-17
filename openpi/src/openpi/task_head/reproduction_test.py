import numpy as np
import torch

from openpi.task_head.reproduction import TaskPairBatchSampler, supervised_contrastive_loss


def test_supervised_contrastive_loss_is_finite() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(embeddings, labels, temperature=0.07)
    assert torch.isfinite(loss)
    loss.backward()


def test_task_pair_batch_sampler_has_positive_for_every_anchor() -> None:
    indices = np.arange(24)
    labels = np.repeat(np.arange(6), 4)
    sampler = TaskPairBatchSampler(
        indices,
        labels,
        tasks_per_batch=4,
        episodes_per_task=2,
        batches_per_epoch=3,
        seed=7,
    )
    for batch in sampler:
        batch_labels = labels[batch]
        assert len(batch) == 8
        for label in batch_labels:
            assert np.count_nonzero(batch_labels == label) == 2
