from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import pathlib
import time
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import datasets
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import lerobot.common.datasets.video_utils as lerobot_video_utils
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        metadata_keys: Sequence[str] = (),
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._metadata_keys = tuple(metadata_keys)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raw = self._dataset[index]
        transformed = self._transform(raw)
        for key in self._metadata_keys:
            if key not in raw:
                raise KeyError(f"Required training metadata key {key!r} is missing from the dataset sample")
            transformed[key] = raw[key]
        return transformed

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class _NoVideoLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobot numeric samples with placeholder images for norm statistics."""

    def _query_videos(self, query_timestamps, ep_idx):
        del ep_idx
        return {
            key: torch.zeros((len(timestamps), 3, 1, 1), dtype=torch.float32).squeeze(0)
            for key, timestamps in query_timestamps.items()
        }


def _is_retryable_video_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "invalid data found when processing input",
            "moov atom not found",
            "end of file",
            "input/output error",
            "resource temporarily unavailable",
        )
    )


class _RetryingVideoLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """Resolve subset links and retry reads without dropping or replacing samples."""

    def _query_videos(self, query_timestamps, ep_idx):
        item = {}
        for video_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, video_key)
            # Exact frame-range subsets reuse byte-identical source videos. Opening
            # the resolved object avoids transient FUSE symlink reads on AOSS.
            if video_path.is_symlink():
                video_path = video_path.resolve(strict=True)
            frames = _decode_video_frames_with_retry(
                video_path,
                query_ts,
                tolerance_s=self.tolerance_s,
                backend=self.video_backend,
                episode_index=ep_idx,
            )
            item[video_key] = frames.squeeze(0)
        return item


def _decode_video_frames_with_retry(
    video_path: pathlib.Path,
    query_timestamps,
    *,
    tolerance_s: float,
    backend: str | None,
    episode_index: int,
):
    attempts = max(1, int(os.environ.get("OPENPI_VIDEO_READ_ATTEMPTS", "5")))
    delay_s = max(0.0, float(os.environ.get("OPENPI_VIDEO_READ_RETRY_DELAY_S", "0.25")))
    for attempt in range(1, attempts + 1):
        try:
            return lerobot_video_utils.decode_video_frames(
                video_path, query_timestamps, tolerance_s, backend
            )
        except Exception as error:
            if attempt == attempts or not _is_retryable_video_error(error):
                raise
            logging.warning(
                "Transient video read failure for episode %s path=%s (attempt %s/%s): %s",
                episode_index,
                video_path,
                attempt,
                attempts,
                error,
            )
            if delay_s:
                time.sleep(delay_s * attempt)

    raise AssertionError("unreachable")


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    load_videos: bool = True,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_root = pathlib.Path(data_config.lerobot_root) if data_config.lerobot_root else None
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_root)
    dataset_class = _RetryingVideoLeRobotDataset if load_videos else _NoVideoLeRobotDataset
    arrow_dir = os.environ.get("OPENPI_LEROBOT_ARROW_DIR")
    if arrow_dir:

        class ArrowCachedLeRobotDataset(_RetryingVideoLeRobotDataset):
            def load_hf_dataset(self):
                shards = sorted(pathlib.Path(arrow_dir).glob("*.arrow"))
                if not shards:
                    raise FileNotFoundError(f"No Arrow shards found in {arrow_dir}")
                parts = [datasets.Dataset.from_file(str(path), in_memory=False) for path in shards]
                hf_dataset = datasets.concatenate_datasets(parts)
                hf_dataset.set_transform(lerobot_dataset.hf_transform_to_torch)
                return hf_dataset

        dataset_class = ArrowCachedLeRobotDataset

    dataset = dataset_class(
        data_config.repo_id,
        root=dataset_root,
        video_backend=os.environ.get("OPENPI_LEROBOT_VIDEO_BACKEND"),
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    metadata_keys: Sequence[str] = (),
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        metadata_keys=metadata_keys,
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    start_batch: int = 0,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        start_batch: Number of deterministic shuffled batches to skip.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        if start_batch:
            raise NotImplementedError("Strict batch-position resume is not supported for RLDS datasets.")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        start_batch=start_batch,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        metadata_keys=(config.guidance_aware.metadata_keys if config.guidance_aware.enabled else ()),
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    start_batch: int = 0,
    framework: str = "jax",
    metadata_keys: Sequence[str] = (),
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(
        dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
        metadata_keys=metadata_keys,
    )

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        start_batch=start_batch,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader, metadata_keys=metadata_keys)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class _StrictResumeRandomSampler(torch.utils.data.Sampler[int]):
    """Reconstruct the stock RandomSampler stream at an exact batch boundary."""

    def __init__(self, dataset, *, batch_size: int, seed: int, start_batch: int):
        self._size = len(dataset)
        self._batch_size = int(batch_size)
        self._batches_per_epoch = self._size // self._batch_size
        if self._batches_per_epoch < 1:
            raise ValueError("Dataset must contain at least one complete batch.")
        if start_batch < 0:
            raise ValueError("start_batch must be non-negative.")

        completed_epochs, batch_in_epoch = divmod(int(start_batch), self._batches_per_epoch)
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

        # Stock DataLoader consumes one random int for its base seed before
        # RandomSampler creates each epoch permutation.
        for _ in range(completed_epochs):
            torch.empty((), dtype=torch.int64).random_(generator=self._generator)
            torch.randperm(self._size, generator=self._generator)
            # RandomSampler generates a second permutation for its empty
            # num_samples % dataset_size slice when the iterator is exhausted.
            torch.randperm(self._size, generator=self._generator)

        self._first_sample_offset = batch_in_epoch * self._batch_size

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_(generator=self._generator)
        permutation = torch.randperm(self._size, generator=self._generator)
        offset = self._first_sample_offset
        self._first_sample_offset = 0
        yield from permutation[offset:].tolist()
        torch.randperm(self._size, generator=self._generator)

    def __len__(self) -> int:
        return self._size - self._first_sample_offset


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        start_batch: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
            start_batch: Exact shuffled batch position for deterministic resume.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        if start_batch:
            if sampler is not None:
                raise ValueError("start_batch cannot be combined with a custom sampler.")
            if not shuffle:
                raise ValueError("start_batch requires shuffle=True.")
            sampler = _StrictResumeRandomSampler(
                dataset,
                batch_size=local_batch_size,
                seed=seed,
                start_batch=start_batch,
            )

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
        *,
        metadata_keys: Sequence[str] = (),
    ):
        self._data_config = data_config
        self._data_loader = data_loader
        self._metadata_keys = tuple(metadata_keys)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            result = (_model.Observation.from_dict(batch), batch["actions"])
            if self._metadata_keys:
                metadata = {key: batch[key] for key in self._metadata_keys}
                yield (*result, metadata)
            else:
                yield result
