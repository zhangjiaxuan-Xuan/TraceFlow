from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def _timed_sample_actions(self, device, observation, **sample_kwargs):
        if (
            self._is_pytorch_model
            and torch.cuda.is_available()
            and torch.device(self._pytorch_device).type == "cuda"
        ):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            actions = self._sample_actions(device, observation, **sample_kwargs)
            end.record()
            end.synchronize()
            return actions, float(start.elapsed_time(end))
        start_time = time.monotonic()
        actions = self._sample_actions(device, observation, **sample_kwargs)
        return actions, (time.monotonic() - start_time) * 1000.0

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        actions, model_ms = self._timed_sample_actions(
            sample_rng_or_pytorch_device,
            observation,
            **sample_kwargs,
        )
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_ms,
        }
        lower_features = getattr(self._model, "_last_lower_retrieval_features", None)
        if lower_features is not None:
            outputs["lower_retrieval_feature"] = np.asarray(lower_features[0], dtype=np.float32)
        return outputs

    def infer_batch(self, observations: list[dict]) -> list[dict]:
        if not observations:
            return []
        if not self._is_pytorch_model:
            raise NotImplementedError("Batched inference is currently implemented for PyTorch policies only")
        transformed = [self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in observations]
        inputs = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *transformed)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(self._pytorch_device), inputs)
        observation = _model.Observation.from_dict(inputs)
        actions, model_ms = self._timed_sample_actions(
            self._pytorch_device,
            observation,
            **dict(self._sample_kwargs),
        )
        states = inputs["state"]
        component_timing = getattr(self._model, "_last_component_timing", None)
        lower_features = getattr(self._model, "_last_lower_retrieval_features", None)
        results = []
        for index in range(len(observations)):
            output = {
                "state": np.asarray(states[index].detach().cpu()),
                "actions": np.asarray(actions[index].detach().cpu()),
            }
            output = self._output_transform(output)
            output["policy_timing"] = {"infer_ms": model_ms, "batch_size": len(observations)}
            if isinstance(component_timing, dict):
                output["policy_timing"]["components"] = dict(component_timing)
            if lower_features is not None:
                output["lower_retrieval_feature"] = np.asarray(lower_features[index], dtype=np.float32)
            results.append(output)
        return results

    def probe_lower_retrieval_batch(self, observations: list[dict]) -> list[dict]:
        """Return Lower VLM retrieval states without sampling candidate actions."""
        if not observations or not self._is_pytorch_model:
            raise ValueError("Lower probing requires a non-empty PyTorch batch")
        transformed = [self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in observations]
        inputs = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *transformed)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(self._pytorch_device), inputs)
        observation = _model.Observation.from_dict(inputs)
        started = time.monotonic()
        features = self._model.encode_lower_retrieval_features(observation)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return [
            {
                "lower_retrieval_feature": np.asarray(features[index].detach().cpu(), dtype=np.float32),
                "policy_timing": {"probe_ms": elapsed_ms, "batch_size": len(observations)},
            }
            for index in range(len(observations))
        ]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
