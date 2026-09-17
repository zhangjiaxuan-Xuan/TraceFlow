from __future__ import annotations

from types import SimpleNamespace

from scripts import serve_policy


def test_negative_only_does_not_construct_positive_provider(monkeypatch) -> None:
    args = serve_policy.Args(
        use_memory=True,
        use_positive_memory=False,
        use_memory_guidance=False,
        memory_guidance_only=True,
        use_negative_guidance=True,
        use_lcm=False,
        policy=serve_policy.Default(),
    )
    head = SimpleNamespace(variant="lower", retrieval_provenance={})
    negative = object()
    monkeypatch.setattr(serve_policy, "_require_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(serve_policy, "_load_task_head", lambda *_args, **_kwargs: head)
    monkeypatch.setattr(
        serve_policy,
        "MemoryInitProvider",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("positive provider was constructed")),
    )
    monkeypatch.setattr(serve_policy, "NegativeMemoryProvider", lambda **_kwargs: negative)
    model = SimpleNamespace(config=SimpleNamespace(action_horizon=10, action_dim=32))

    serve_policy._inject_gpm_lcm(model, args)

    assert model.memory_provider is None
    assert model.negative_memory_provider is negative
    assert model.use_memory is True
    assert model.use_memory_guidance is True
    assert model.memory_guidance_only is True
    assert model.use_negative_guidance is True
