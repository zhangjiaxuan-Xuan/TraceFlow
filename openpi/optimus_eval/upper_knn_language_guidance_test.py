import numpy as np
import torch

from optimus_eval.upper_knn_language_guidance import aggregate_subtask_candidates
from optimus_eval.upper_knn_language_guidance import KnnSubtaskLogitsProcessor
from optimus_eval.upper_knn_language_guidance import NumpyInnerProductIndex


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(value) for value in text]


def test_aggregate_merges_equivalent_subtasks() -> None:
    values = aggregate_subtask_candidates(
        ["pick cookies", " pick   cookies ", "place cookies"],
        np.asarray([1.0, 0.9, 0.0]),
        temperature=0.2,
    )
    assert len(values) == 2
    assert values[0][0] == "pick cookies"
    assert abs(sum(weight for _, weight in values) - 1.0) < 1e-6


def test_processor_changes_only_matching_prefix() -> None:
    tokenizer = CharacterTokenizer()
    processor = KnnSubtaskLogitsProcessor(
        tokenizer,
        [[("pick cookies", 1.0)]],
        prompt_width=2,
        interpolation=0.5,
    )
    scores = torch.zeros((1, 256))
    output = processor(torch.tensor([[7, 8]]), scores)
    expected = ord("{")
    assert output[0, expected] > output[0, ord("x")]

    diverged = processor(torch.tensor([[7, 8, ord("x")]]), scores)
    assert torch.equal(diverged, scores)


def test_processor_abstains_without_candidates() -> None:
    processor = KnnSubtaskLogitsProcessor(
        CharacterTokenizer(),
        [[]],
        prompt_width=1,
        interpolation=0.8,
    )
    scores = torch.randn((1, 256))
    assert torch.equal(processor(torch.tensor([[1]]), scores), scores)


def test_processor_accepts_per_sample_interpolation() -> None:
    processor = KnnSubtaskLogitsProcessor(
        CharacterTokenizer(),
        [[("pick cookies", 1.0)], [("pick butter", 1.0)]],
        prompt_width=1,
        interpolation=[0.8, 0.0],
    )
    scores = torch.zeros((2, 256))
    output = processor(torch.tensor([[1], [1]]), scores)
    assert output[0, ord("{")] > output[0, ord("x")]
    assert torch.equal(output[1], scores[1])


def test_numpy_inner_product_index_matches_exact_topk() -> None:
    keys = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    scores, indices = NumpyInnerProductIndex(keys).search(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        2,
    )
    assert indices.tolist() == [[0, 2], [1, 2]]
    np.testing.assert_allclose(scores, [[1.0, 0.8], [1.0, 0.2]])
