from torch import nn

from predictive_coding_head.predictive_coding_head import resolve_multimodal_base_model
from predictive_coding_head.predictive_coding_head import temporal_prediction_pairs


class _MultimodalBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = object()

    def get_image_features(self):
        return None


class _Wrapper(nn.Module):
    def __init__(self, child, attribute):
        super().__init__()
        setattr(self, attribute, child)


def test_resolve_multimodal_base_model_through_ddp_and_peft_shapes():
    base = _MultimodalBase()
    peft_like = _Wrapper(base, "model")
    ddp_like = _Wrapper(peft_like, "module")
    assert resolve_multimodal_base_model(ddp_like) is base


def test_three_view_pairs_stay_within_camera_and_skip_history() -> None:
    times = [-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    views = [0, 1, 2] * 4
    assert temporal_prediction_pairs(len(times), times, views) == [
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 11),
    ]
