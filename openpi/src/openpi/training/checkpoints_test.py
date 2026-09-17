import dataclasses

from flax import nnx
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import utils


class _TinyModel(nnx.Module):
    def __init__(self):
        self.vlm = nnx.Param(jnp.array([1.0, 2.0]))
        self.ae = nnx.Param(jnp.array([3.0, 4.0]))


def _copy_state(state: nnx.State) -> nnx.State:
    return state.map(lambda _, value: value.replace(value=jnp.array(value.value)))


def _train_state(*, ae_offset: float, step: int) -> utils.TrainState:
    graphdef, params = nnx.split(_TinyModel())
    params["ae"] = params["ae"].replace(value=params["ae"].value + ae_offset)
    tx = optax.adam(1e-3)
    trainable = params.filter(nnx_utils.PathRegex(".*ae.*"))
    return utils.TrainState(
        step=jnp.array(step),
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=tx.init(trainable),
        ema_decay=0.9,
        ema_params=_copy_state(params),
    )


def test_trainable_checkpoint_subset_and_merge_preserve_frozen_base():
    trainable_filter = nnx_utils.PathRegex(".*ae.*")
    trained = _train_state(ae_offset=10.0, step=17)
    subset = checkpoints._trainable_checkpoint_state(trained, trainable_filter)

    assert tuple(subset.params.flat_state()) == (("ae",),)
    assert tuple(subset.ema_params.flat_state()) == (("ae",),)

    restored = dataclasses.replace(subset, step=jnp.array(17))
    merged = checkpoints._merge_trainable_checkpoint_state(
        _train_state(ae_offset=0.0, step=0), restored
    )

    assert int(merged.step) == 17
    assert jnp.array_equal(merged.params["vlm"].value, jnp.array([1.0, 2.0]))
    assert jnp.array_equal(merged.params["ae"].value, jnp.array([13.0, 14.0]))
    assert jnp.array_equal(merged.ema_params["vlm"].value, jnp.array([1.0, 2.0]))
    assert jnp.array_equal(merged.ema_params["ae"].value, jnp.array([13.0, 14.0]))


def test_trainable_checkpoint_disk_roundtrip_preserves_frozen_base(tmp_path):
    trainable_filter = nnx_utils.PathRegex(".*ae.*")
    trained = _train_state(ae_offset=10.0, step=17)
    subset = checkpoints._trainable_checkpoint_state(trained, trainable_filter)
    train_state, params = checkpoints._split_params(subset)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(tmp_path / "train_state", train_state)
    checkpointer.save(tmp_path / "params", {"params": params})

    restored_train_state = checkpointer.restore(tmp_path / "train_state", item=train_state)
    restored_params = checkpointer.restore(tmp_path / "params", item={"params": params})
    restored_subset = checkpoints._merge_params(restored_train_state, restored_params)
    restored = checkpoints._merge_trainable_checkpoint_state(
        _train_state(ae_offset=0.0, step=0), restored_subset
    )

    assert int(restored.step) == 17
    assert jnp.array_equal(restored.params["vlm"].value, jnp.array([1.0, 2.0]))
    assert jnp.array_equal(restored.params["ae"].value, jnp.array([13.0, 14.0]))
    assert jnp.array_equal(restored.ema_params["vlm"].value, jnp.array([1.0, 2.0]))
    assert jnp.array_equal(restored.ema_params["ae"].value, jnp.array([13.0, 14.0]))
