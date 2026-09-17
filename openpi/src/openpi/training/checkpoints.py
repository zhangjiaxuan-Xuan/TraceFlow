from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
from typing import Protocol

from etils import epath
from flax import nnx
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    recent_checkpoints_to_keep: int = 1,
    overwrite: bool,
    resume: bool,
) -> tuple[ocp.CheckpointManager, bool]:
    if recent_checkpoints_to_keep < 1:
        raise ValueError("recent_checkpoints_to_keep must be at least 1")
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=recent_checkpoints_to_keep,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    trainable_filter: nnx.filterlib.Filter | None = None,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
        if trainable_filter is not None:
            (directory / "checkpoint_scope.json").write_text(
                json.dumps({"format": "openpi-trainable-only-v1", "compression": "none"})
            )

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        checkpoint_state = _trainable_checkpoint_state(state, trainable_filter)
        train_state, params = _split_params(checkpoint_state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader | None = None,
    step: int | None = None,
    trainable_filter: nnx.filterlib.Filter | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        checkpoint_state = _trainable_checkpoint_state(state, trainable_filter)
        train_state, params = _split_params(checkpoint_state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    restored_state = _merge_params(restored["train_state"], restored["params"])
    if trainable_filter is None:
        return restored_state
    return _merge_trainable_checkpoint_state(state, restored_state)


def _trainable_checkpoint_state(
    state: training_utils.TrainState,
    trainable_filter: nnx.filterlib.Filter | None,
) -> training_utils.TrainState:
    if trainable_filter is None:
        return state
    return dataclasses.replace(
        state,
        params=state.params.filter(trainable_filter),
        ema_params=None if state.ema_params is None else state.ema_params.filter(trainable_filter),
    )


def _merge_trainable_checkpoint_state(
    base_state: training_utils.TrainState,
    restored_state: training_utils.TrainState,
) -> training_utils.TrainState:
    base_state.params.replace_by_pure_dict(restored_state.params.to_pure_dict())
    ema_params = base_state.ema_params
    if restored_state.ema_params is not None:
        if ema_params is None:
            raise ValueError("Trainable-only checkpoint contains EMA but base state does not")
        ema_params.replace_by_pure_dict(restored_state.ema_params.to_pure_dict())
    return dataclasses.replace(
        base_state,
        step=restored_state.step,
        params=base_state.params,
        opt_state=restored_state.opt_state,
        ema_params=ema_params,
    )


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
