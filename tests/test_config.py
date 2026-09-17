import pytest

from traceflow.config import load, validate


def test_all_release_configs_validate() -> None:
    for name in (
        "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_plus10",
        "arena_sequence", "arena_transferring", "arena_counting", "arena_occlusion",
        "tracebankstack_libero10", "tracebankstack_transferring",
    ):
        validate(load(name), name=name)


def test_envelope_is_explicit() -> None:
    rates = [load(name)["reported"]["success_rate"] for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10")]
    assert sum(rates) / len(rates) == pytest.approx(0.983)
