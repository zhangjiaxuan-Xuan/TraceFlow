import copy
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time as _time

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.async_trace_writer import get_async_trace_writer
from openpi.task_head.memory_init import ActionMemorySession


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class GuidanceResidualAdapter(nn.Module):
    """Small zero-initialized correction around a frozen flow+guidance field."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * action_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x_t: torch.Tensor,
        v_base: torch.Tensor,
        guidance: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        time_features = time[:, None, None].expand(-1, x_t.shape[1], 1)
        features = torch.cat(
            [
                x_t.to(dtype=torch.float32),
                v_base.detach().to(dtype=torch.float32),
                guidance.detach().to(dtype=torch.float32),
                time_features.to(dtype=torch.float32),
            ],
            dim=-1,
        )
        return self.net(features)


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )
        self._capture_prefix_tokens = False
        self._last_prefix_tokens = None

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        compile_flag = os.environ.get("OPENPI_TORCH_COMPILE", "1").lower()
        if compile_flag not in {"0", "false", "no", "off"}:
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        else:
            logging.info("[PI0Pytorch] torch.compile disabled by OPENPI_TORCH_COMPILE=%s", compile_flag)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def configure_guidance_adapter(self, hidden_dim: int) -> None:
        """Attach the optional training adapter after loading a stock checkpoint."""
        if hasattr(self, "guidance_residual_adapter"):
            existing = self.guidance_residual_adapter.net[0].out_features
            if int(existing) != int(hidden_dim):
                raise ValueError(f"Guidance adapter already uses hidden_dim={existing}, requested {hidden_dim}")
            return
        self.guidance_residual_adapter = GuidanceResidualAdapter(self.config.action_dim, int(hidden_dim))

    def freeze_for_guidance_adapter_training(self) -> list[nn.Parameter]:
        """Freeze the stock policy and expose only residual-adapter parameters."""
        if not hasattr(self, "guidance_residual_adapter"):
            raise RuntimeError("configure_guidance_adapter must be called before freezing")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        trainable = list(self.guidance_residual_adapter.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
        return trainable

    def _apply_guidance_adapter(
        self,
        x_t: torch.Tensor,
        v_base: torch.Tensor,
        guidance: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "guidance_residual_adapter"):
            return v_base + guidance
        residual = self.guidance_residual_adapter(x_t, v_base, guidance, time).to(dtype=v_base.dtype)
        return v_base + guidance.to(dtype=v_base.dtype) + residual


    def enable_prefix_token_capture(self, flag: bool = True):
        """Enable or disable caching of VLM prefix tokens."""
        self._capture_prefix_tokens = bool(flag)
        if not flag:
            self._last_prefix_tokens = None

    @torch.no_grad()
    def get_last_prefix_tokens(self):
        """Return the latest cached VLM prefix tokens with shape [B, L, D]."""
        return self._last_prefix_tokens

    def _set_last_prefix_tokens(self, x: torch.Tensor | None):
        if x is None:
            self._last_prefix_tokens = None
            return
        if x.ndim == 2:
            x = x.unsqueeze(0)
        self._last_prefix_tokens = x.detach()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def predict_training_velocity(
        self,
        observation,
        actions,
        noise=None,
        time=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return base velocity and the sampled flow-matching training state."""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return v_t, u_t, x_t, time, prefix_embs

    def forward(
        self,
        observation,
        actions,
        noise=None,
        time=None,
        *,
        guidance_velocity: torch.Tensor | None = None,
        memory_blocks: torch.Tensor | None = None,
        memory_weights: torch.Tensor | None = None,
        guidance_parameters: dict | None = None,
        return_components: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute stock FM loss or loss under a persistent memory field."""
        v_t, u_t, x_t, time, _ = self.predict_training_velocity(observation, actions, noise=noise, time=time)

        if memory_blocks is not None:
            if memory_weights is None or guidance_parameters is None:
                raise ValueError("memory_blocks requires memory_weights and guidance_parameters")
            guidance_velocity = self.compute_training_memory_guidance(
                x_t,
                v_t,
                memory_blocks,
                memory_weights,
                time,
                **guidance_parameters,
            )

        residual = torch.zeros_like(v_t)
        guidance = torch.zeros_like(v_t)
        if guidance_velocity is not None:
            if guidance_velocity.shape != v_t.shape:
                raise ValueError(
                    f"Guidance shape {tuple(guidance_velocity.shape)} does not match velocity {tuple(v_t.shape)}"
                )
            guidance = guidance_velocity.detach().to(device=v_t.device, dtype=v_t.dtype)
            if hasattr(self, "guidance_residual_adapter"):
                residual = self.guidance_residual_adapter(x_t, v_t, guidance, time).to(dtype=v_t.dtype)

        actual_velocity = v_t + guidance + residual
        loss = F.mse_loss(u_t, actual_velocity, reduction="none")
        if return_components:
            return {
                "loss": loss,
                "v_base": v_t,
                "v_actual": actual_velocity,
                "target": u_t,
                "x_t": x_t,
                "time": time,
                "guidance": guidance,
                "residual": residual,
            }

        return loss

    def guidance_aware_loss_from_components(
        self,
        *,
        v_base: torch.Tensor,
        target: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
        guidance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the adapter to an already-computed frozen flow training state."""
        if not hasattr(self, "guidance_residual_adapter"):
            raise RuntimeError("Guidance residual adapter is not configured")
        guidance = guidance.detach().to(device=v_base.device, dtype=v_base.dtype)
        residual = self.guidance_residual_adapter(x_t, v_base, guidance, time).to(dtype=v_base.dtype)
        actual = v_base + guidance + residual
        return {
            "loss": F.mse_loss(target, actual, reduction="none"),
            "v_base": v_base,
            "v_actual": actual,
            "target": target,
            "x_t": x_t,
            "time": time,
            "guidance": guidance,
            "residual": residual,
        }

    @staticmethod
    def compute_training_memory_guidance(
        x_t: torch.Tensor,
        v_base: torch.Tensor,
        blocks: torch.Tensor,
        weights: torch.Tensor,
        time: torch.Tensor,
        *,
        version: str,
        lambda_max: float,
        t_cut: float,
        sigma: float,
        norm_cap: float,
    ) -> torch.Tensor:
        """Vectorized V0/V1 positive-memory field for independent training rows."""
        if blocks.ndim != 4 or blocks.shape[0] != x_t.shape[0]:
            raise ValueError("blocks must have shape [batch, top_k, horizon, action_dim]")
        if weights.shape != blocks.shape[:2]:
            raise ValueError("weights must have shape [batch, top_k]")
        if version not in {"v0", "v1"}:
            raise ValueError(f"Guidance-aware training only supports v0/v1, got {version!r}")
        if not 0.0 <= t_cut < 1.0:
            raise ValueError("t_cut must be in [0, 1)")

        x = x_t.to(dtype=torch.float32)
        blocks = blocks.to(device=x.device, dtype=torch.float32)
        weights = weights.to(device=x.device, dtype=torch.float32)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        sigma = max(float(sigma), 1e-6)
        diff = blocks - x.unsqueeze(1)
        dist2 = diff.square().sum(dim=(2, 3))
        logits = torch.log(weights.clamp_min(1e-12)) - dist2 / (2.0 * sigma * sigma)
        responsibilities = torch.softmax(logits, dim=1)
        attract = (responsibilities[:, :, None, None] * diff).sum(dim=1) / (sigma * sigma)
        scale = float(lambda_max) * ((time - float(t_cut)) / max(1e-6, 1.0 - float(t_cut))).clamp(0.0, 1.0)
        guidance = -scale[:, None, None] * attract
        if version == "v1" and float(norm_cap) > 0.0:
            guidance = PI0Pytorch._cap_guidance_relative(guidance, v_base, float(norm_cap))
        return guidance.to(dtype=v_base.dtype)

    def _temporal_memory_lower_features(self, lower: torch.Tensor) -> torch.Tensor:
        window = int(getattr(self.task_head, "temporal_window", 1))
        if window <= 1:
            return lower
        expected_dim = int(getattr(self.task_head, "lower_dim", 0))
        if expected_dim != int(lower.shape[1]) * window:
            raise ValueError(
                f"Temporal retrieval head expects lower_dim={expected_dim}, "
                f"but current feature dim {lower.shape[1]} x window {window} does not match"
            )

        histories = getattr(self, "_memory_feature_history", None)
        if not isinstance(histories, dict):
            histories = {}
            self._memory_feature_history = histories
        runtime = getattr(self, "_batch_runtime", None)
        if isinstance(runtime, list):
            if len(runtime) != int(lower.shape[0]):
                raise RuntimeError("Temporal retrieval runtime and feature batch sizes differ")
            keys = [str(record["environment_id"]) for record in runtime]
            resets = [bool(record.get("reset", False)) for record in runtime]
        else:
            if int(lower.shape[0]) != 1:
                raise RuntimeError("Temporal retrieval batches require per-environment runtime records")
            keys = ["__single__"] * int(lower.shape[0])
            resets = [False] * int(lower.shape[0])

        output = []
        for index, (key, reset) in enumerate(zip(keys, resets, strict=True)):
            if reset:
                histories.pop(key, None)
            state = histories.setdefault(key, {"values": [], "serial": None})
            values = state["values"]
            serial = getattr(self, "_memory_inference_serial", None)
            if state["serial"] != serial or not values:
                values.append(lower[index].detach())
                state["serial"] = serial
            else:
                values[-1] = lower[index].detach()
            del values[:-window]
            padded = [values[0]] * (window - len(values)) + values
            output.append(torch.cat(padded, dim=0))
        return torch.stack(output, dim=0)

    def _compute_memory_task_embeddings(self, prefix, *, device: torch.device) -> torch.Tensor:
        if self.task_head is None:
            raise RuntimeError("Memory retrieval requires task_head")
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        lower = prefix.mean(dim=1).to(dtype=torch.float32, device=device)
        variant = str(getattr(self.task_head, "variant", "lower"))
        if variant == "lower" and not hasattr(self.task_head, "lower_dim"):
            return F.normalize(self.task_head(lower), dim=-1)
        lower = self._temporal_memory_lower_features(lower)

        runtime = getattr(self, "_batch_runtime", None)
        if isinstance(runtime, list):
            upper_values = [record.get("upper_vlm_feature") for record in runtime]
            available = torch.tensor(
                [bool(record.get("upper_vlm_available", value is not None)) for record, value in zip(runtime, upper_values)],
                dtype=torch.float32,
                device=device,
            )
            upper_dim = int(getattr(self.task_head, "upper_dim", 0))
            upper = torch.zeros((len(runtime), upper_dim), dtype=torch.float32, device=device)
            for index, value in enumerate(upper_values):
                if value is not None:
                    # websocket deserialization can expose a read-only NumPy buffer.
                    row = torch.as_tensor(np.array(value, copy=True), dtype=torch.float32, device=device).flatten()
                    if row.numel() != upper_dim:
                        raise ValueError(
                            f"upper_vlm_feature dimension mismatch: expected {upper_dim}, got {row.numel()}"
                        )
                    upper[index] = row
            age = torch.tensor(
                [float(record.get("upper_vlm_age", 1.0)) for record in runtime],
                dtype=torch.float32,
                device=device,
            )
        else:
            value = getattr(self, "_external_upper_vlm_feature", None)
            upper = None if value is None else value.flatten().unsqueeze(0).to(device=device, dtype=torch.float32)
            age = torch.tensor(
                [float(getattr(self, "_external_upper_vlm_age", 1.0))], dtype=torch.float32, device=device
            )
            available = torch.tensor(
                [float(bool(getattr(self, "_external_upper_vlm_available", value is not None)))],
                dtype=torch.float32,
                device=device,
            )
        return self.task_head(lower, upper, age, available)

    @torch.inference_mode()
    def encode_lower_retrieval_features(self, observation) -> torch.Tensor:
        """Encode counterfactual prompts without denoising or mutating temporal memory."""
        if getattr(self, "task_head", None) is None:
            raise RuntimeError("Lower subtask probing requires a retrieval head")
        device = observation.state.device
        images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        attention_mask = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        prefix = outputs_embeds[0] if isinstance(outputs_embeds, (list, tuple)) else outputs_embeds
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        pooled = prefix.mean(dim=1).to(dtype=torch.float32, device=device)

        histories = getattr(self, "_memory_feature_history", None)
        snapshot = None
        if isinstance(histories, dict):
            snapshot = {
                key: {"values": list(value.get("values", [])), "serial": value.get("serial")}
                for key, value in histories.items()
            }
        try:
            return self._temporal_memory_lower_features(pooled).detach().to(dtype=torch.float32)
        finally:
            self._memory_feature_history = {} if snapshot is None else snapshot

    def _memory_search_batch(self, task_embs: torch.Tensor) -> list:
        search = getattr(getattr(self, "memory_provider", None), "search_batch", None)
        if callable(search):
            required_task_ids = None
            if bool(getattr(self, "memory_exact_task_gate", False)):
                runtime = getattr(self, "_batch_runtime", None)
                if not isinstance(runtime, list) or len(runtime) != int(task_embs.shape[0]):
                    raise RuntimeError("Exact-task memory retrieval requires batch runtime context")
                required_task_ids = []
                for record in runtime:
                    context = dict(record.get("context", {}) or {})
                    task_id = int(context.get("task_id", -1))
                    if task_id < 1:
                        raise ValueError("Exact-task memory retrieval requires a positive context task_id")
                    required_task_ids.append(task_id)
            return search(
                task_embs,
                int(getattr(self, "memory_top_k", 8)),
                required_task_ids=required_task_ids,
            )
        return [None] * int(task_embs.shape[0])

    def _component_timing_mark(self, device) -> float | None:
        if not bool(getattr(self, "_record_component_timing", False)):
            return None
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        return _time.perf_counter()

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        self._memory_inference_serial = int(getattr(self, "_memory_inference_serial", 0)) + 1
        bsize = observation.state.shape[0]

        prefix_started = self._component_timing_mark(device)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        outputs_embeds, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_finished = self._component_timing_mark(device)
        self._component_prefix_ms = (
            0.0
            if prefix_started is None or prefix_finished is None
            else (prefix_finished - prefix_started) * 1000.0
        )

        dbg = bool(getattr(self, "debug_memory", False))
        use_mem = bool(getattr(self, "use_memory", True))
        use_memory_guidance = bool(getattr(self, "use_memory_guidance", False))
        memory_guidance_only = bool(getattr(self, "memory_guidance_only", False))
        memory_prior_substep_guidance = bool(getattr(self, "memory_prior_substep_guidance", False))
        prefix_for_head = outputs_embeds[0] if isinstance(outputs_embeds, (list, tuple)) else outputs_embeds
        if getattr(self, "_export_lower_retrieval_feature", False) and getattr(self, "task_head", None) is not None:
            export_prefix = prefix_for_head.unsqueeze(0) if prefix_for_head.ndim == 2 else prefix_for_head
            pooled = export_prefix.mean(dim=1).to(dtype=torch.float32, device=device)
            exported = self._temporal_memory_lower_features(pooled)
            self._last_lower_retrieval_features = exported.detach().to(dtype=torch.float32, device="cpu")
        if memory_prior_substep_guidance and getattr(self, "_batch_runtime", None) is not None:
            return self._sample_actions_batched_prior_substep_guidance(
                device=device,
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                outputs_embeds=outputs_embeds,
                noise=noise,
            )
        if getattr(self, "_batch_runtime", None) is not None and memory_guidance_only:
            return self._sample_actions_batched_memory_guidance(
                device=device,
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                outputs_embeds=outputs_embeds,
                noise=noise,
                num_steps=int(getattr(self, "memory_guidance_num_steps", 10)),
            )
        if not use_mem:
            if noise is None:
                H = self.config.action_horizon
                A = self.config.action_dim
                actions_shape = (bsize, H, A)
                noise = self.sample_noise(actions_shape, device)
            dt = -1.0 / num_steps
            dt = torch.tensor(dt, dtype=torch.float32, device=device)
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
                x_t = x_t + dt * v_t
                time += dt
            return x_t

        if getattr(self, "_capture_prefix_tokens", False):
            self._set_last_prefix_tokens(prefix_for_head)

        task_emb = None
        if prefix_for_head is not None and getattr(self, "task_head", None) is not None:
            toks = prefix_for_head
            if toks.ndim == 2:
                toks = toks.unsqueeze(0)
            pooled = toks.mean(dim=1).to(dtype=torch.float32, device=device)
            task_emb = self._compute_memory_task_embeddings(prefix_for_head, device=device).squeeze(0)

        if getattr(self, "memory_provider", None) is not None and task_emb is not None:
            memory_top_k = int(getattr(self, "memory_top_k", 8))
            if getattr(self, "memory_session", None) is None:
                if dbg:
                    logging.info("Starting a new GPM memory session.")
                self.memory_session = ActionMemorySession(
                    provider=self.memory_provider,
                    init_task_emb=task_emb,
                    k=memory_top_k,
                    H=self.config.action_horizon,
                    progress=0.0,
                )
                self._sample_call_count = 0
                if dbg:
                    logging.info(
                        "GPM memory session created: k=%d horizon=%d memory_size=%d",
                        memory_top_k,
                        int(self.config.action_horizon),
                        int(getattr(self.memory_provider, "num_items", -1)),
                    )
            else:
                step_idx = getattr(self, "_sample_call_count", 0)
                did = self.memory_session.maybe_refresh(
                    new_task_emb=task_emb,
                    step_idx=step_idx,
                    refresh_every=int(getattr(self, "memory_refresh_every", 1)),
                    sim_threshold=float(getattr(self, "memory_refresh_sim_threshold", 0.0)),
                    k=memory_top_k,
                    advance_steps=int(getattr(self, "replan_steps_hint", self.config.action_horizon)),
                )
                if dbg:
                    logging.info("GPM memory refresh: refreshed=%s step_idx=%d", bool(did), step_idx)

        memory_prior_used = False
        lcm_prior_chunk = None
        progress_used = float(getattr(self, "_external_progress", 0.0))
        progress_src = "client"
        if getattr(self, "progress_mode", "client") == "memory" and getattr(self, "memory_session", None) is not None:
            step_idx = getattr(self, "_sample_call_count", 0)
            replan_hint = int(getattr(self, "replan_steps_hint", self.config.action_horizon))
            progress_used = float(self.memory_session.estimate_progress(step_idx=step_idx, replan_steps=replan_hint))
            progress_src = "memory"

        if noise is None:
            H = self.config.action_horizon
            A = self.config.action_dim
            if getattr(self, "memory_session", None) is not None and not memory_guidance_only:
                X_init, debug_info = self.memory_session.sample_chunk(progress_used, return_debug=True)
                noise = X_init[None, :, :].expand(bsize, H, A).contiguous()
                try:
                    lcm_mu = self.memory_session.prior_mean(progress_used)
                    lcm_prior_chunk = lcm_mu[None, :, :].expand(bsize, H, A).contiguous()
                except Exception:
                    lcm_prior_chunk = noise
                memory_prior_used = True
                num_steps = self.memory_session.nfe_adapt
                if dbg:
                    logging.info(
                        "GPM prior: progress=%.3f source=%s nfe=%d k=%d win=[%d,%d) "
                        "top_sims=%s prior_mean=%.4f prior_std=%.4f noise_sigma=%.4f",
                        progress_used,
                        progress_src,
                        int(num_steps),
                        debug_info.get("k", -1),
                        debug_info.get("win_start", -1),
                        debug_info.get("win_end", -1),
                        debug_info.get("top3_sims", []),
                        debug_info.get("prior_mean", float("nan")),
                        debug_info.get("prior_std", float("nan")),
                        debug_info.get("noise_sigma", float("nan")),
                    )
            else:
                actions_shape = (bsize, H, A)
                noise = self.sample_noise(actions_shape, device)
                if dbg:
                    reason = []
                    if getattr(self, "memory_provider", None) is None:
                        reason.append("no_provider")
                    if task_emb is None:
                        reason.append("no_task_emb")
                    if (
                        getattr(self, "memory_provider", None) is not None
                        and task_emb is not None
                        and getattr(self, "memory_session", None) is None
                    ):
                        reason.append("session_create_failed")
                    logging.info(
                        "Using Gaussian noise fallback: H=%d A=%d reason=%s",
                        H,
                        A,
                        "+".join(reason) or "unknown",
                    )

        guidance_blocks = None
        guidance_weights = None
        guidance_debug = {}
        if use_memory_guidance and getattr(self, "memory_session", None) is not None:
            min_sim = float(getattr(self, "memory_guidance_min_similarity", -1.0))
            sim = float(getattr(self.memory_session, "s_global", -1.0))
            if sim >= min_sim:
                guidance_blocks, guidance_weights, guidance_debug = self.memory_session.guidance_tensors(progress_used, device)
                if memory_guidance_only:
                    num_steps = int(getattr(self, "memory_guidance_num_steps", 10))
                if dbg:
                    logging.info(
                        "GPM flow guidance: progress=%.3f source=%s nfe=%d k=%d win=[%d,%d) "
                        "similarity=%.4f top_sims=%s",
                        progress_used,
                        progress_src,
                        int(num_steps),
                        guidance_debug.get("k", -1),
                        guidance_debug.get("win_start", -1),
                        guidance_debug.get("win_end", -1),
                        guidance_debug.get("similarity_global", float("nan")),
                        guidance_debug.get("top3_sims", []),
                    )
            elif dbg:
                logging.info("GPM flow guidance skipped: similarity %.4f < %.4f", sim, min_sim)

        negative_blocks = None
        negative_weights = None
        negative_debug = {}
        if (
            bool(getattr(self, "use_negative_guidance", False))
            and getattr(self, "negative_memory_provider", None) is not None
            and task_emb is not None
        ):
            negative_blocks, negative_weights, negative_debug = self.negative_memory_provider.query_blocks(
                task_emb,
                k=int(getattr(self, "negative_memory_top_k", 4)),
                horizon=int(self.config.action_horizon),
                progress=progress_used,
                min_similarity=float(getattr(self, "negative_memory_min_similarity", 0.975)),
                min_confidence=float(getattr(self, "negative_memory_min_confidence", 0.75)),
            )
            if dbg:
                logging.info(
                    "Negative GPM gate: reason=%s progress=%.3f available=%d selected=%d scores=%s memory_tasks=%s",
                    negative_debug.get("reason", "unknown"),
                    float(negative_debug.get("progress", progress_used)),
                    int(negative_debug.get("available", 0)),
                    int(negative_debug.get("selected", 0)),
                    negative_debug.get("scores", []),
                    negative_debug.get("memory_task_names", []),
                )

        use_lcm = (
            bool(getattr(self, "use_lcm", False))
            and getattr(self, "lcm", None) is not None
            and memory_prior_used
            and noise is not None
            and not memory_guidance_only
        )
        prev_lcm = getattr(self, "_lcm_prev_chunk", None)
        if use_lcm and prev_lcm is not None:
            prev_lcm = prev_lcm.to(device=device, dtype=torch.float32)
            if prev_lcm.ndim == 2:
                prev_lcm = prev_lcm.unsqueeze(0)
            if prev_lcm.shape[0] != bsize:
                if prev_lcm.shape[0] == 1:
                    prev_lcm = prev_lcm.expand(bsize, -1, -1).contiguous()
                else:
                    prev_lcm = None
            if prev_lcm is not None and prev_lcm.shape[1] != self.config.action_horizon:
                prev_lcm = F.interpolate(
                    prev_lcm.permute(0, 2, 1),
                    size=self.config.action_horizon,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)

            if prev_lcm is not None and prev_lcm.shape[-1] == self.config.action_dim:
                h_lcm = getattr(self, "_lcm_h", None)
                if h_lcm is not None:
                    h_lcm = h_lcm.to(device=device)
                context_lcm = lcm_prior_chunk if lcm_prior_chunk is not None else noise
                context_lcm = context_lcm.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    lcm_bias, h_new = self.lcm(prev_lcm, context_lcm, h_lcm)
                scale = float(getattr(self, "lcm_scale", 1.0))
                noise = noise + scale * lcm_bias.to(device=device, dtype=noise.dtype)
                self._lcm_h = h_new.detach() if h_new is not None else None
                if getattr(self, "debug_lcm", False):
                    logging.info(
                        "[LCM] applied action bias | scale=%.3f | norm(raw)=%.5f | norm(scaled)=%.5f",
                        scale,
                        float(lcm_bias.norm()),
                        float((scale * lcm_bias).norm()),
                    )
        elif getattr(self, "debug_lcm", False):
            reason = "no-prev" if prev_lcm is None else ("no-memory-prior" if not memory_prior_used else "disabled")
            logging.info("[LCM] skipped: %s", reason)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        trace_level = str(getattr(self, "memory_guidance_trace_level", "full")).lower()
        trace_requested = bool(getattr(self, "memory_guidance_trace_dir", ""))
        if trace_requested and trace_level not in ("full", "light", "bank"):
            raise ValueError(f"Unsupported memory guidance trace level: {trace_level}")
        trace_enabled = trace_requested and (
            guidance_blocks is not None or (trace_level == "light" and negative_blocks is not None)
        )
        full_trace = trace_enabled and trace_level == "full"
        trace_steps = []
        trace_x = [x_t.detach().to(dtype=torch.float32, device="cpu")] if full_trace else []
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            positive_guidance = torch.zeros_like(v_t)
            negative_guidance = torch.zeros_like(v_t)
            step_debug = {}
            if guidance_blocks is not None and guidance_weights is not None:
                positive_guidance, step_debug = self._memory_flow_guidance(
                    x_t,
                    v_t,
                    guidance_blocks,
                    guidance_weights,
                    float(time.item()),
                    return_debug=full_trace,
                )
            if negative_blocks is not None and negative_weights is not None:
                negative_guidance, negative_step_debug = self._negative_memory_flow_guidance(
                    x_t,
                    v_t,
                    negative_blocks,
                    negative_weights,
                    float(time.item()),
                    return_debug=full_trace,
                )
            else:
                negative_step_debug = {}
            guidance = self._cap_guidance_relative(
                positive_guidance + negative_guidance,
                v_t,
                float(getattr(self, "memory_guidance_total_norm_cap", 0.2)),
            )
            if full_trace:
                trace_record = {
                    **step_debug,
                    "v_base": v_t.detach().to(dtype=torch.float32, device="cpu"),
                    "guidance": guidance.detach().to(dtype=torch.float32, device="cpu"),
                    "guidance_positive": positive_guidance.detach().to(dtype=torch.float32, device="cpu"),
                    "guidance_negative": negative_guidance.detach().to(dtype=torch.float32, device="cpu"),
                }
                if negative_step_debug:
                    trace_record["negative_responsibilities"] = negative_step_debug["responsibilities"]
                    trace_record["negative_dist2"] = negative_step_debug["dist2"]
                trace_steps.append(trace_record)
            elif trace_enabled:
                trace_steps.append(
                    self._summarize_memory_guidance_step(
                        v_base=v_t,
                        guidance_positive=positive_guidance,
                        guidance_negative=negative_guidance,
                        guidance_final=guidance,
                        time=float(time.item()),
                    )
                )
            v_t = self._apply_guidance_adapter(x_t, v_t, guidance, expanded_time)
            x_t = x_t + dt * v_t
            if full_trace:
                trace_x.append(x_t.detach().to(dtype=torch.float32, device="cpu"))
            time += dt

        if trace_enabled:
            self._write_memory_guidance_trace(
                trace_x=trace_x,
                trace_steps=trace_steps,
                blocks=guidance_blocks,
                weights=guidance_weights,
                retrieval_debug=guidance_debug,
                negative_weights=negative_weights,
                negative_debug=negative_debug,
                task_emb=task_emb,
                progress=progress_used,
                progress_source=progress_src,
                num_steps=int(num_steps),
                final_action=x_t,
            )

        self._lcm_prev_chunk = x_t.detach()
        if getattr(self, "memory_session", None) is not None:
            self._sample_call_count = getattr(self, "_sample_call_count", 0) + 1
        return x_t

    def _sample_actions_batched_prior_substep_guidance(
        self,
        *,
        device,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        outputs_embeds,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        """Memory-prior sampling with dynamic-NFE batched recursive substep guidance."""
        version = str(getattr(self, "memory_prior_guidance_version", "v1")).lower()
        if version == "v2":
            return self._sample_actions_batched_prior_substep_guidance_v2(
                device=device,
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                outputs_embeds=outputs_embeds,
                noise=noise,
            )
        if version in (
            "v3_prior_only",
            "v3_prior_decay",
            "v3_1_prior_decay",
            "v3_prior_decay_joint",
            "v3_re_prior_decay",
        ):
            return self._sample_actions_batched_prior_anchor_guidance(
                device=device,
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                outputs_embeds=outputs_embeds,
                noise=noise,
                use_flow_decay=version != "v3_prior_only",
                use_joint_guidance=version == "v3_prior_decay_joint",
                compact_active=version == "v3_re_prior_decay",
                flow_guidance_norm_cap=(
                    float(getattr(self, "memory_prior_guidance_norm_cap", 0.5))
                    if version == "v3_1_prior_decay"
                    else None
                ),
            )
        if version != "v1":
            raise ValueError(f"Unknown memory prior guidance version: {version}")

        bsize = int(state.shape[0])
        runtime = getattr(self, "_batch_runtime", None)
        if not isinstance(runtime, list) or len(runtime) != bsize:
            raise RuntimeError(f"Prior-substep guidance requires {bsize} runtime records")
        if self.memory_provider is None or self.task_head is None:
            raise RuntimeError("Prior-substep guidance requires the positive GPM memory bank")

        prefix = outputs_embeds[0] if isinstance(outputs_embeds, (list, tuple)) else outputs_embeds
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        pooled = prefix.mean(dim=1).to(dtype=torch.float32, device=device)
        task_embs = self._compute_memory_task_embeddings(prefix, device=device)
        memory_search_results = self._memory_search_batch(task_embs)
        states = getattr(self, "_batched_memory_states", None)
        if states is None:
            states = {}
            self._batched_memory_states = states

        horizon = int(self.config.action_horizon)
        action_dim = int(self.config.action_dim)
        positive: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        negative: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        prior_rows: list[torch.Tensor] = []
        nfe_rows: list[int] = []
        active_ids: list[str] = []

        for index, record in enumerate(runtime):
            env_id = str(record["environment_id"])
            if bool(record.get("reset")) or env_id not in states:
                states[env_id] = {"session": None, "call_count": 0, "seed": int(record["seed"])}
            slot = states[env_id]
            call_count = int(slot["call_count"])
            progress = float(np.clip(float(record.get("progress", 0.0)), 0.0, 1.0))

            # ActionMemorySession currently samples through NumPy's global RNG. Isolate
            # each environment's deterministic stream without leaking RNG state.
            numpy_state = np.random.get_state()
            np.random.seed((int(slot["seed"]) + call_count) % (2**32 - 1))
            try:
                session = slot.get("session")
                if session is None:
                    session = ActionMemorySession(
                        provider=self.memory_provider,
                        init_task_emb=task_embs[index],
                        k=int(getattr(self, "memory_top_k", 8)),
                        H=horizon,
                        progress=progress,
                        search_result=memory_search_results[index],
                    )
                    slot["session"] = session
                else:
                    session.maybe_refresh(
                        new_task_emb=task_embs[index],
                        step_idx=call_count,
                        refresh_every=int(getattr(self, "memory_refresh_every", 1)),
                        sim_threshold=float(getattr(self, "memory_refresh_sim_threshold", 0.0)),
                        k=int(getattr(self, "memory_top_k", 8)),
                        advance_steps=int(getattr(self, "replan_steps_hint", horizon)),
                        search_result=memory_search_results[index],
                    )
                prior_rows.append(session.sample_chunk(progress))
            finally:
                np.random.set_state(numpy_state)

            nfe_rows.append(max(1, int(session.nfe_adapt)))
            if float(session.s_global) >= float(getattr(self, "memory_guidance_min_similarity", -1.0)):
                blocks, weights, _ = session.guidance_tensors(progress, device)
                positive.append((blocks, weights))
            else:
                positive.append(None)

            neg = None
            if bool(getattr(self, "use_negative_guidance", False)) and self.negative_memory_provider is not None:
                blocks, weights, _ = self.negative_memory_provider.query_blocks(
                    task_embs[index],
                    k=int(getattr(self, "negative_memory_top_k", 4)),
                    horizon=horizon,
                    progress=progress,
                    min_similarity=float(getattr(self, "negative_memory_min_similarity", 0.975)),
                    min_confidence=float(getattr(self, "negative_memory_min_confidence", 0.75)),
                )
                if blocks is not None and weights is not None:
                    neg = (blocks, weights)
            negative.append(neg)
            active_ids.append(env_id)

        x_t = noise if noise is not None else torch.stack(prior_rows, dim=0)
        if tuple(x_t.shape) != (bsize, horizon, action_dim):
            raise ValueError(
                f"Expected prior/noise shape {(bsize, horizon, action_dim)}, got {tuple(x_t.shape)}"
            )
        nfe = torch.tensor(nfe_rows, dtype=torch.long, device=device)

        for macro_step in range(max(nfe_rows)):
            active_rows = [macro_step < steps for steps in nfe_rows]
            guided_rows = [macro_step < steps - 2 for steps in nfe_rows]
            frozen_rows = [is_active and not is_guided for is_active, is_guided in zip(active_rows, guided_rows)]
            active = macro_step < nfe
            time = torch.where(
                active,
                1.0 - macro_step / nfe.to(dtype=torch.float32),
                torch.zeros(bsize, dtype=torch.float32, device=device),
            )
            v_model = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
            guided = active & (macro_step < (nfe - 2))
            frozen = active & ~guided

            if any(frozen_rows):
                macro_dt = (1.0 / nfe.to(dtype=torch.float32)).view(-1, 1, 1)
                frozen_mask = frozen.view(-1, 1, 1)
                x_t = torch.where(frozen_mask, x_t - macro_dt * v_model, x_t)

            if any(guided_rows):
                recursive_velocity = v_model
                micro_dt = (0.5 / nfe.to(dtype=torch.float32)).view(-1, 1, 1)
                guided_mask = guided.view(-1, 1, 1)
                for micro_step in range(2):
                    guidance_rows = []
                    for index in range(bsize):
                        if not guided_rows[index]:
                            guidance_rows.append(torch.zeros_like(recursive_velocity[index : index + 1]))
                            continue
                        positive_velocity = torch.zeros_like(recursive_velocity[index : index + 1])
                        negative_velocity = torch.zeros_like(positive_velocity)
                        if positive[index] is not None:
                            positive_velocity, _ = self._memory_flow_guidance(
                                x_t[index : index + 1],
                                recursive_velocity[index : index + 1],
                                *positive[index],
                                1.0 - (macro_step + 0.5 * micro_step) / nfe_rows[index],
                            )
                        if negative[index] is not None:
                            negative_velocity, _ = self._negative_memory_flow_guidance(
                                x_t[index : index + 1],
                                recursive_velocity[index : index + 1],
                                *negative[index],
                                1.0 - (macro_step + 0.5 * micro_step) / nfe_rows[index],
                            )
                        guidance_rows.append(positive_velocity + negative_velocity)
                    guidance = self._cap_guidance_relative(
                        torch.cat(guidance_rows, dim=0),
                        recursive_velocity,
                        float(getattr(self, "memory_guidance_total_norm_cap", 0.2)),
                    )
                    recursive_velocity = recursive_velocity + guidance
                    x_t = torch.where(guided_mask, x_t - micro_dt * recursive_velocity, x_t)

        for env_id in active_ids:
            states[env_id]["call_count"] = int(states[env_id]["call_count"]) + 1
        return x_t

    @staticmethod
    def _full_interval_schedule(nfe: int, nfe_max: int) -> tuple[int, np.ndarray]:
        """Return the original full-interval Euler schedule for a dynamic NFE."""
        if int(nfe_max) <= 0:
            raise ValueError("nfe_max must be positive")
        steps = max(1, min(int(nfe_max), int(nfe)))
        return steps, np.full(steps, 1.0 / steps, dtype=np.float64)

    @staticmethod
    def _mix_guidance_direction_first(
        v_base: torch.Tensor,
        guidance: torch.Tensor,
        magnitude_cap: float,
    ) -> torch.Tensor:
        """Choose the guided direction first, then bound speed around the base velocity norm."""
        base = v_base.to(dtype=torch.float32)
        mixed = base + guidance.to(dtype=torch.float32)
        base_norm = base.flatten(1).norm(dim=1).clamp_min(1e-12)
        mixed_norm_raw = mixed.flatten(1).norm(dim=1)
        mixed_norm = mixed_norm_raw.clamp_min(1e-12)
        mixed_direction = mixed / mixed_norm.view(-1, 1, 1)
        base_direction = base / base_norm.view(-1, 1, 1)
        direction = torch.where(
            (mixed_norm_raw > 1e-12).view(-1, 1, 1),
            mixed_direction,
            base_direction,
        )
        cap = max(0.0, float(magnitude_cap))
        if cap == 0.0:
            speed = base_norm
        else:
            speed = torch.clamp(mixed_norm, min=(1.0 - cap) * base_norm, max=(1.0 + cap) * base_norm)
        return (direction * speed.view(-1, 1, 1)).to(dtype=v_base.dtype)

    @staticmethod
    def _cap_guidance_relative_to_base(
        v_base: torch.Tensor,
        guidance: torch.Tensor,
        cap_ratio: float,
    ) -> torch.Tensor:
        """Bound full-chunk guidance norm relative to the current base velocity."""
        ratio = float(cap_ratio)
        if ratio < 0.0:
            raise ValueError("Guidance norm cap ratio must be non-negative")
        base_norm = v_base.to(dtype=torch.float32).flatten(1).norm(dim=1)
        guidance_float = guidance.to(dtype=torch.float32)
        guidance_norm = guidance_float.flatten(1).norm(dim=1).clamp_min(1e-12)
        factor = torch.clamp(ratio * base_norm / guidance_norm, max=1.0)
        return (guidance_float * factor.view(-1, 1, 1)).to(dtype=guidance.dtype)

    @staticmethod
    def _prior_anchor_guidance_scale(
        step_index: int,
        num_steps: int,
        start_scale: float,
        final_scale: float,
    ) -> float:
        """Linearly anneal guidance over the actual adaptive-NFE schedule."""
        steps = max(1, int(num_steps))
        if not 0 <= int(step_index) < steps:
            raise ValueError(f"step_index must be in [0, {steps}), got {step_index}")
        start = float(start_scale)
        final = float(final_scale)
        if start < 0.0 or final < 0.0 or final > start:
            raise ValueError("Prior-anchor guidance scales must satisfy 0 <= final <= start")
        if steps == 1:
            return final
        fraction = float(step_index) / float(steps - 1)
        return start + fraction * (final - start)

    @staticmethod
    def _apply_prior_velocity_correction(
        noisy_prior: torch.Tensor,
        guidance_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Apply velocity-coordinate memory guidance as an action-space prior shift."""
        # Sampling integrates x <- x - dt * v. The corresponding endpoint
        # correction therefore has the opposite sign from guidance velocity.
        return noisy_prior - guidance_velocity.to(device=noisy_prior.device, dtype=noisy_prior.dtype)

    @staticmethod
    def _select_dynamic_cache_batch(past_key_values, indices):
        """Return a cache sharing metadata but containing only selected batch rows."""
        if not hasattr(past_key_values, "key_cache") or not hasattr(past_key_values, "value_cache"):
            raise TypeError(
                "V3-re requires a Transformers DynamicCache with key_cache/value_cache batch tensors"
            )
        selected = copy.copy(past_key_values)

        def select_rows(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.numel() == 0:
                return tensor
            if isinstance(indices, slice):
                return tensor[indices]
            return tensor.index_select(0, indices.to(device=tensor.device))

        selected.key_cache = [select_rows(tensor) for tensor in past_key_values.key_cache]
        selected.value_cache = [select_rows(tensor) for tensor in past_key_values.value_cache]
        return selected

    def _sample_actions_batched_prior_anchor_guidance(
        self,
        *,
        device,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        outputs_embeds,
        noise: torch.Tensor | None,
        use_flow_decay: bool,
        use_joint_guidance: bool = False,
        compact_active: bool = False,
        flow_guidance_norm_cap: float | None = None,
    ) -> torch.Tensor:
        """Correct the clean memory prior once, then run base or annealed direction-only flow."""
        bsize = int(state.shape[0])
        runtime = getattr(self, "_batch_runtime", None)
        if not isinstance(runtime, list) or len(runtime) != bsize:
            raise RuntimeError(f"Prior-anchor guidance requires {bsize} runtime records")
        if self.memory_provider is None or self.task_head is None:
            raise RuntimeError("Prior-anchor guidance requires the positive GPM memory bank")
        if str(getattr(self.memory_provider, "mixture_mode", "gaussian")) != "gaussian":
            raise ValueError("Prior-anchor guidance currently requires mixture_mode=gaussian")

        start_scale = float(getattr(self, "memory_guidance_lambda_max", 0.20))
        final_scale = float(getattr(self, "memory_prior_guidance_final_scale", 0.01))
        if start_scale < 0.0 or final_scale < 0.0 or final_scale > start_scale:
            raise ValueError("Prior-anchor guidance requires 0 <= final_scale <= lambda_max")

        prefix = outputs_embeds[0] if isinstance(outputs_embeds, (list, tuple)) else outputs_embeds
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        pooled = prefix.mean(dim=1).to(dtype=torch.float32, device=device)
        task_embs = self._compute_memory_task_embeddings(prefix, device=device)
        memory_search_results = self._memory_search_batch(task_embs)

        states = getattr(self, "_batched_memory_states", None)
        if states is None:
            states = {}
            self._batched_memory_states = states

        horizon = int(self.config.action_horizon)
        action_dim = int(self.config.action_dim)
        nfe_max = int(getattr(self, "memory_nfe_max", 10))
        positive: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        negative: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        noisy_prior_rows: list[torch.Tensor] = []
        clean_prior_rows: list[torch.Tensor] = []
        step_rows: list[int] = []
        raw_nfe_rows: list[float] = []
        effective_nfe_rows: list[float] = []
        prior_noise_scale_rows: list[float] = []
        dt_schedule_rows: list[np.ndarray] = []
        active_ids: list[str] = []

        for index, record in enumerate(runtime):
            env_id = str(record["environment_id"])
            if bool(record.get("reset")) or env_id not in states:
                states[env_id] = {"session": None, "call_count": 0, "seed": int(record["seed"])}
            slot = states[env_id]
            call_count = int(slot["call_count"])
            progress = float(np.clip(float(record.get("progress", 0.0)), 0.0, 1.0))

            numpy_state = np.random.get_state()
            np.random.seed((int(slot["seed"]) + call_count) % (2**32 - 1))
            try:
                session = slot.get("session")
                if session is None:
                    session = ActionMemorySession(
                        provider=self.memory_provider,
                        init_task_emb=task_embs[index],
                        k=int(getattr(self, "memory_top_k", 8)),
                        H=horizon,
                        progress=progress,
                        search_result=memory_search_results[index],
                    )
                    slot["session"] = session
                else:
                    session.maybe_refresh(
                        new_task_emb=task_embs[index],
                        step_idx=call_count,
                        refresh_every=int(getattr(self, "memory_refresh_every", 1)),
                        sim_threshold=float(getattr(self, "memory_refresh_sim_threshold", 0.0)),
                        k=int(getattr(self, "memory_top_k", 8)),
                        advance_steps=int(getattr(self, "replan_steps_hint", horizon)),
                        search_result=memory_search_results[index],
                    )
                noisy_prior_rows.append(session.sample_chunk(progress))
                clean_prior_rows.append(session.sample_clean_chunk(progress))
            finally:
                np.random.set_state(numpy_state)

            steps, dt_schedule = self._full_interval_schedule(session.nfe_adapt, nfe_max)
            step_rows.append(steps)
            raw_nfe_rows.append(float(getattr(session, "nfe_continuous_raw", session.nfe_adapt)))
            effective_nfe_rows.append(float(getattr(session, "nfe_continuous", session.nfe_adapt)))
            prior_noise_scale_rows.append(float(getattr(session, "lambda_noise", 0.0)))
            dt_schedule_rows.append(dt_schedule)

            if float(session.s_global) >= float(getattr(self, "memory_guidance_min_similarity", -1.0)):
                blocks, weights, _ = session.guidance_tensors(progress, device)
                positive.append((blocks, weights))
            else:
                positive.append(None)

            neg = None
            if bool(getattr(self, "use_negative_guidance", False)) and self.negative_memory_provider is not None:
                blocks, weights, _ = self.negative_memory_provider.query_blocks(
                    task_embs[index],
                    k=int(getattr(self, "negative_memory_top_k", 4)),
                    horizon=horizon,
                    progress=progress,
                    min_similarity=float(getattr(self, "negative_memory_min_similarity", 0.975)),
                    min_confidence=float(getattr(self, "negative_memory_min_confidence", 0.75)),
                )
                if blocks is not None and weights is not None:
                    neg = (blocks, weights)
            negative.append(neg)
            active_ids.append(env_id)
            slot["prior_anchor_schedule"] = {
                "mode": (
                    "prior_decay_joint"
                    if use_joint_guidance
                    else ("prior_decay" if use_flow_decay else "prior_only")
                ),
                "nfe": int(steps),
                "nfe_continuous_raw": raw_nfe_rows[-1],
                "nfe_continuous_effective": effective_nfe_rows[-1],
                "prior_noise_scale": prior_noise_scale_rows[-1],
                "prior_scale": start_scale,
                "flow_scales": (
                    [
                        self._prior_anchor_guidance_scale(step, steps, start_scale, final_scale)
                        for step in range(steps)
                    ]
                    if use_flow_decay
                    else []
                ),
                "direction_first_magnitude_cap": 0.0,
                "flow_guidance_norm_cap": flow_guidance_norm_cap,
                "joint_failure_prior": (
                    float(getattr(self, "memory_joint_failure_prior", 0.5))
                    if use_joint_guidance
                    else None
                ),
            }

        active_batch_sizes = [
            int(sum(step_index < count for count in step_rows))
            for step_index in range(max(step_rows))
        ]
        dense_denoise_rows = int(bsize * max(step_rows))
        actual_denoise_rows = int(sum(active_batch_sizes)) if compact_active else dense_denoise_rows
        self._last_adaptive_nfe_batch = {
            "sampler": "v3_re_prior_anchor" if compact_active else "v3_prior_anchor",
            "logical_nfe": [int(value) for value in step_rows],
            "raw_nfe_continuous": raw_nfe_rows,
            "effective_nfe_continuous": effective_nfe_rows,
            "prior_noise_scale": prior_noise_scale_rows,
            "executed_batch_nfe": int(max(step_rows)),
            "fixed_nfe_baseline": int(nfe_max),
            "active_batch_sizes": active_batch_sizes,
            "actual_denoise_rows": actual_denoise_rows,
            "dense_denoise_rows": dense_denoise_rows,
            "fixed_denoise_rows": int(bsize * nfe_max),
            "active_row_fraction": (
                float(actual_denoise_rows / dense_denoise_rows) if dense_denoise_rows else 0.0
            ),
            "environment_ids": list(active_ids),
            "contexts": [dict(record.get("context", {}) or {}) for record in runtime],
        }
        sampled_prior = torch.stack(noisy_prior_rows, dim=0).to(device=device, dtype=torch.float32)
        clean_prior = torch.stack(clean_prior_rows, dim=0).to(device=device, dtype=torch.float32)
        x_t = noise.to(device=device, dtype=torch.float32) if noise is not None else sampled_prior
        expected_shape = (bsize, horizon, action_dim)
        if tuple(x_t.shape) != expected_shape or tuple(clean_prior.shape) != expected_shape:
            raise ValueError(
                f"Expected prior shapes {expected_shape}, got noisy={tuple(x_t.shape)} clean={tuple(clean_prior.shape)}"
            )

        prior_guidance_rows = []
        for index in range(bsize):
            zero_velocity = torch.zeros_like(clean_prior[index : index + 1])
            if use_joint_guidance:
                if positive[index] is not None and negative[index] is not None:
                    guidance_velocity, _ = self._joint_outcome_memory_flow_guidance(
                        clean_prior[index : index + 1],
                        *positive[index],
                        *negative[index],
                        1.0,
                        scale_override=start_scale,
                    )
                else:
                    guidance_velocity = zero_velocity
            else:
                positive_velocity = torch.zeros_like(zero_velocity)
                negative_velocity = torch.zeros_like(zero_velocity)
                if positive[index] is not None:
                    positive_velocity, _ = self._memory_flow_guidance(
                        clean_prior[index : index + 1],
                        zero_velocity,
                        *positive[index],
                        1.0,
                        apply_norm_cap=False,
                        scale_override=start_scale,
                    )
                if negative[index] is not None:
                    negative_velocity, _ = self._negative_memory_flow_guidance(
                        clean_prior[index : index + 1],
                        zero_velocity,
                        *negative[index],
                        1.0,
                        apply_norm_cap=False,
                        scale_override=start_scale,
                    )
                guidance_velocity = positive_velocity + negative_velocity
            prior_guidance_rows.append(guidance_velocity)
        prior_guidance = torch.cat(prior_guidance_rows, dim=0)
        x_t = self._apply_prior_velocity_correction(x_t, prior_guidance)
        if not bool(torch.isfinite(x_t).all()):
            raise FloatingPointError("Prior-anchor correction produced a non-finite initial state")

        if compact_active:
            order_indices = sorted(range(bsize), key=lambda index: (-step_rows[index], index))
            order = torch.tensor(order_indices, dtype=torch.long, device=device)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(bsize, dtype=torch.long, device=device)
            x_sorted = x_t.index_select(0, order)
            state_sorted = state.index_select(0, order)
            prefix_masks_sorted = prefix_pad_masks.index_select(0, order)
            cache_sorted = self._select_dynamic_cache_batch(past_key_values, order)
            step_rows_sorted = [step_rows[index] for index in order_indices]
            schedules_sorted = [dt_schedule_rows[index] for index in order_indices]
            positive_sorted = [positive[index] for index in order_indices]
            negative_sorted = [negative[index] for index in order_indices]

            record_step_timing = bool(getattr(self, "_record_active_nfe_timing", False))
            cuda_step_events = []
            cpu_step_ms = []
            for step_index, active_count in enumerate(active_batch_sizes):
                if record_step_timing and x_t.is_cuda:
                    step_start = torch.cuda.Event(enable_timing=True)
                    step_end = torch.cuda.Event(enable_timing=True)
                    step_start.record()
                else:
                    step_start = _time.perf_counter() if record_step_timing else None
                active_x = x_sorted[:active_count]
                time_rows = [
                    float(schedules_sorted[index][step_index:].sum())
                    for index in range(active_count)
                ]
                dt_rows = [
                    float(schedules_sorted[index][step_index])
                    for index in range(active_count)
                ]
                time = torch.tensor(time_rows, dtype=torch.float32, device=device)
                dt = torch.tensor(dt_rows, dtype=torch.float32, device=device).view(-1, 1, 1)
                active_cache = self._select_dynamic_cache_batch(cache_sorted, slice(0, active_count))
                v_flow = self.denoise_step(
                    state_sorted[:active_count],
                    prefix_masks_sorted[:active_count],
                    active_cache,
                    active_x,
                    time,
                )
                velocity = v_flow

                if use_flow_decay:
                    guidance_rows = []
                    for index in range(active_count):
                        scale = self._prior_anchor_guidance_scale(
                            step_index,
                            step_rows_sorted[index],
                            start_scale,
                            final_scale,
                        )
                        positive_velocity = torch.zeros_like(v_flow[index : index + 1])
                        negative_velocity = torch.zeros_like(positive_velocity)
                        if positive_sorted[index] is not None:
                            positive_velocity, _ = self._memory_flow_guidance(
                                active_x[index : index + 1],
                                v_flow[index : index + 1],
                                *positive_sorted[index],
                                time_rows[index],
                                apply_norm_cap=False,
                                scale_override=scale,
                            )
                        if negative_sorted[index] is not None:
                            negative_velocity, _ = self._negative_memory_flow_guidance(
                                active_x[index : index + 1],
                                v_flow[index : index + 1],
                                *negative_sorted[index],
                                time_rows[index],
                                apply_norm_cap=False,
                                scale_override=scale,
                            )
                        guidance_rows.append(positive_velocity + negative_velocity)
                    guidance_batch = torch.cat(guidance_rows, dim=0)
                    if flow_guidance_norm_cap is not None:
                        guidance_batch = self._cap_guidance_relative_to_base(
                            v_flow,
                            guidance_batch,
                            flow_guidance_norm_cap,
                        )
                    velocity = self._mix_guidance_direction_first(
                        v_flow, guidance_batch, magnitude_cap=0.0
                    )

                updated = active_x - dt * velocity
                x_sorted = (
                    updated
                    if active_count == bsize
                    else torch.cat([updated, x_sorted[active_count:]], dim=0)
                )
                if not bool(torch.isfinite(updated).all()):
                    raise FloatingPointError("V3-re prior-anchor flow produced a non-finite state")
                if record_step_timing and x_t.is_cuda:
                    step_end.record()
                    cuda_step_events.append((step_start, step_end))
                elif record_step_timing:
                    cpu_step_ms.append((_time.perf_counter() - step_start) * 1000.0)
            x_t = x_sorted.index_select(0, inverse)
            if cuda_step_events:
                cuda_step_events[-1][1].synchronize()
                self._last_adaptive_nfe_batch["denoise_step_ms"] = [
                    float(start.elapsed_time(end)) for start, end in cuda_step_events
                ]
            elif cpu_step_ms:
                self._last_adaptive_nfe_batch["denoise_step_ms"] = cpu_step_ms
        else:
            steps_tensor = torch.tensor(step_rows, dtype=torch.long, device=device)
            for step_index in range(max(step_rows)):
                active_rows = [step_index < count for count in step_rows]
                active = step_index < steps_tensor
                dt_rows = [
                    float(schedule[step_index]) if step_index < count else 0.0
                    for schedule, count in zip(dt_schedule_rows, step_rows)
                ]
                time_rows = [
                    float(schedule[step_index:].sum()) if step_index < count else 0.0
                    for schedule, count in zip(dt_schedule_rows, step_rows)
                ]
                dt = torch.tensor(dt_rows, dtype=torch.float32, device=device).view(-1, 1, 1)
                time = torch.tensor(time_rows, dtype=torch.float32, device=device)
                v_flow = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
                velocity = v_flow

                if use_flow_decay:
                    guidance_rows = []
                    for index in range(bsize):
                        if not active_rows[index]:
                            guidance_rows.append(torch.zeros_like(v_flow[index : index + 1]))
                            continue
                        scale = self._prior_anchor_guidance_scale(
                            step_index,
                            step_rows[index],
                            start_scale,
                            final_scale,
                        )
                        if use_joint_guidance:
                            if positive[index] is not None and negative[index] is not None:
                                guidance_velocity, _ = self._joint_outcome_memory_flow_guidance(
                                    x_t[index : index + 1],
                                    *positive[index],
                                    *negative[index],
                                    time_rows[index],
                                    scale_override=scale,
                                )
                            else:
                                guidance_velocity = torch.zeros_like(v_flow[index : index + 1])
                        else:
                            positive_velocity = torch.zeros_like(v_flow[index : index + 1])
                            negative_velocity = torch.zeros_like(positive_velocity)
                            if positive[index] is not None:
                                positive_velocity, _ = self._memory_flow_guidance(
                                    x_t[index : index + 1],
                                    v_flow[index : index + 1],
                                    *positive[index],
                                    time_rows[index],
                                    apply_norm_cap=False,
                                    scale_override=scale,
                                )
                            if negative[index] is not None:
                                negative_velocity, _ = self._negative_memory_flow_guidance(
                                    x_t[index : index + 1],
                                    v_flow[index : index + 1],
                                    *negative[index],
                                    time_rows[index],
                                    apply_norm_cap=False,
                                    scale_override=scale,
                                )
                            guidance_velocity = positive_velocity + negative_velocity
                        guidance_rows.append(guidance_velocity)
                    guidance_batch = torch.cat(guidance_rows, dim=0)
                    if flow_guidance_norm_cap is not None:
                        guidance_batch = self._cap_guidance_relative_to_base(
                            v_flow,
                            guidance_batch,
                            flow_guidance_norm_cap,
                        )
                    velocity = self._mix_guidance_direction_first(
                        v_flow, guidance_batch, magnitude_cap=0.0
                    )

                x_t = torch.where(active.view(-1, 1, 1), x_t - dt * velocity, x_t)
                if not bool(torch.isfinite(x_t).all()):
                    raise FloatingPointError("Prior-anchor flow produced a non-finite state")

        for env_id in active_ids:
            states[env_id]["call_count"] = int(states[env_id]["call_count"]) + 1
        return x_t

    def _sample_actions_batched_prior_substep_guidance_v2(
        self,
        *,
        device,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        outputs_embeds,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        """V1 memory prior, NFE, and time integration with direction-first guidance."""
        bsize = int(state.shape[0])
        runtime = getattr(self, "_batch_runtime", None)
        if not isinstance(runtime, list) or len(runtime) != bsize:
            raise RuntimeError(f"Prior-substep guidance v2 requires {bsize} runtime records")
        if self.memory_provider is None or self.task_head is None:
            raise RuntimeError("Prior-substep guidance v2 requires the positive GPM memory bank")

        prefix = outputs_embeds[0] if isinstance(outputs_embeds, (list, tuple)) else outputs_embeds
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        pooled = prefix.mean(dim=1).to(dtype=torch.float32, device=device)
        task_embs = self._compute_memory_task_embeddings(prefix, device=device)
        memory_search_results = self._memory_search_batch(task_embs)

        states = getattr(self, "_batched_memory_states", None)
        if states is None:
            states = {}
            self._batched_memory_states = states

        horizon = int(self.config.action_horizon)
        action_dim = int(self.config.action_dim)
        nfe_max = int(getattr(self, "memory_nfe_max", 10))
        positive: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        negative: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        prior_rows: list[torch.Tensor] = []
        step_rows: list[int] = []
        dt_schedule_rows: list[np.ndarray] = []
        active_ids: list[str] = []

        for index, record in enumerate(runtime):
            env_id = str(record["environment_id"])
            if bool(record.get("reset")) or env_id not in states:
                states[env_id] = {"session": None, "call_count": 0, "seed": int(record["seed"])}
            slot = states[env_id]
            call_count = int(slot["call_count"])
            progress = float(np.clip(float(record.get("progress", 0.0)), 0.0, 1.0))

            numpy_state = np.random.get_state()
            np.random.seed((int(slot["seed"]) + call_count) % (2**32 - 1))
            try:
                session = slot.get("session")
                if session is None:
                    session = ActionMemorySession(
                        provider=self.memory_provider,
                        init_task_emb=task_embs[index],
                        k=int(getattr(self, "memory_top_k", 8)),
                        H=horizon,
                        progress=progress,
                        search_result=memory_search_results[index],
                    )
                    slot["session"] = session
                else:
                    session.maybe_refresh(
                        new_task_emb=task_embs[index],
                        step_idx=call_count,
                        refresh_every=int(getattr(self, "memory_refresh_every", 1)),
                        sim_threshold=float(getattr(self, "memory_refresh_sim_threshold", 0.0)),
                        k=int(getattr(self, "memory_top_k", 8)),
                        advance_steps=int(getattr(self, "replan_steps_hint", horizon)),
                        search_result=memory_search_results[index],
                    )
                prior = session.sample_chunk(progress)
                prior_rows.append(prior)
            finally:
                np.random.set_state(numpy_state)

            steps, dt_schedule = self._full_interval_schedule(session.nfe_adapt, nfe_max)
            step_rows.append(steps)
            dt_schedule_rows.append(dt_schedule)
            slot["v2_schedule"] = {
                "nfe": int(steps),
                "integration": "v1_full_interval",
                "executed_length": 1.0,
                "dt": [float(value) for value in dt_schedule],
            }
            if bool(getattr(self, "debug_memory", False)):
                logging.info("V2 memory schedule env=%s %s", env_id, slot["v2_schedule"])

            if float(session.s_global) >= float(getattr(self, "memory_guidance_min_similarity", -1.0)):
                blocks, weights, _ = session.guidance_tensors(progress, device)
                positive.append((blocks, weights))
            else:
                positive.append(None)

            neg = None
            if bool(getattr(self, "use_negative_guidance", False)) and self.negative_memory_provider is not None:
                blocks, weights, _ = self.negative_memory_provider.query_blocks(
                    task_embs[index],
                    k=int(getattr(self, "negative_memory_top_k", 4)),
                    horizon=horizon,
                    progress=progress,
                    min_similarity=float(getattr(self, "negative_memory_min_similarity", 0.975)),
                    min_confidence=float(getattr(self, "negative_memory_min_confidence", 0.75)),
                )
                if blocks is not None and weights is not None:
                    neg = (blocks, weights)
            negative.append(neg)
            active_ids.append(env_id)

        x_t = noise.to(device=device, dtype=torch.float32) if noise is not None else torch.stack(prior_rows, dim=0)
        if tuple(x_t.shape) != (bsize, horizon, action_dim):
            raise ValueError(f"Expected prior/noise shape {(bsize, horizon, action_dim)}, got {tuple(x_t.shape)}")

        steps = torch.tensor(step_rows, dtype=torch.long, device=device)

        for macro_step in range(max(step_rows)):
            active_rows = [macro_step < count for count in step_rows]
            guided_rows = [macro_step < count - 2 for count in step_rows]
            frozen_rows = [is_active and not is_guided for is_active, is_guided in zip(active_rows, guided_rows)]
            active = macro_step < steps
            current_dt_rows = [
                float(schedule[macro_step]) if macro_step < count else 0.0
                for schedule, count in zip(dt_schedule_rows, step_rows)
            ]
            current_time_rows = [
                float(schedule[macro_step:].sum()) if macro_step < count else 0.0
                for schedule, count in zip(dt_schedule_rows, step_rows)
            ]
            macro_dt = torch.tensor(current_dt_rows, dtype=torch.float32, device=device)
            time = torch.tensor(current_time_rows, dtype=torch.float32, device=device)
            v_model = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
            guided = active & (macro_step < (steps - 2))
            frozen = active & ~guided

            if any(frozen_rows):
                frozen_mask = frozen.view(-1, 1, 1)
                x_t = torch.where(frozen_mask, x_t - macro_dt.view(-1, 1, 1) * v_model, x_t)

            if any(guided_rows):
                recursive_velocity = v_model
                micro_dt = (0.5 * macro_dt).view(-1, 1, 1)
                guided_mask = guided.view(-1, 1, 1)
                for micro_step in range(2):
                    guidance_rows = []
                    for index in range(bsize):
                        if not guided_rows[index]:
                            guidance_rows.append(torch.zeros_like(recursive_velocity[index : index + 1]))
                            continue
                        t_micro = current_time_rows[index] - 0.5 * micro_step * current_dt_rows[index]
                        positive_velocity = torch.zeros_like(recursive_velocity[index : index + 1])
                        negative_velocity = torch.zeros_like(positive_velocity)
                        if positive[index] is not None:
                            positive_velocity, _ = self._memory_flow_guidance(
                                x_t[index : index + 1],
                                recursive_velocity[index : index + 1],
                                *positive[index],
                                t_micro,
                            )
                        if negative[index] is not None:
                            negative_velocity, _ = self._negative_memory_flow_guidance(
                                x_t[index : index + 1],
                                recursive_velocity[index : index + 1],
                                *negative[index],
                                t_micro,
                            )
                        guidance_rows.append(positive_velocity + negative_velocity)
                    recursive_velocity = self._mix_guidance_direction_first(
                        recursive_velocity,
                        torch.cat(guidance_rows, dim=0),
                        float(getattr(self, "memory_guidance_v2_magnitude_cap", 0.10)),
                    )
                    x_t = torch.where(guided_mask, x_t - micro_dt * recursive_velocity, x_t)

        for env_id in active_ids:
            states[env_id]["call_count"] = int(states[env_id]["call_count"]) + 1
        return x_t

    def _sample_actions_batched_memory_guidance(
        self,
        *,
        device,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        outputs_embeds,
        noise: torch.Tensor | None,
        num_steps: int,
    ) -> torch.Tensor:
        """Fixed-NFE batched guidance with isolated memory and selectable time schedule."""
        bsize = int(state.shape[0])
        runtime = getattr(self, "_batch_runtime", None)
        if not isinstance(runtime, list) or len(runtime) != bsize:
            raise RuntimeError(f"Batched guidance requires {bsize} runtime records")
        if int(num_steps) != 10:
            raise ValueError("Batched guidance currently requires fixed 10 NFE")

        retrieval_started = self._component_timing_mark(device)
        prefix = outputs_embeds[0] if isinstance(outputs_embeds, (list, tuple)) else outputs_embeds
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        pooled = prefix.mean(dim=1).to(dtype=torch.float32, device=device)
        task_embs = self._compute_memory_task_embeddings(prefix, device=device)
        memory_search_results = self._memory_search_batch(task_embs)
        lower_retrieval_features = self._temporal_memory_lower_features(pooled)
        if getattr(self, "_export_lower_retrieval_feature", False):
            self._last_lower_retrieval_features = lower_retrieval_features.detach().to(
                dtype=torch.float32, device="cpu"
            )
        upper_dim = int(getattr(self.task_head, "upper_dim", 0))
        upper_retrieval_features = torch.zeros((bsize, upper_dim), dtype=torch.float32, device=device)
        upper_available_rows = []
        upper_age_rows = []
        for index, record in enumerate(runtime):
            upper_value = record.get("upper_vlm_feature")
            upper_available_rows.append(bool(record.get("upper_vlm_available", upper_value is not None)))
            upper_age_rows.append(float(record.get("upper_vlm_age", 1.0)))
            if upper_value is not None:
                upper_row = torch.as_tensor(
                    np.array(upper_value, copy=True), dtype=torch.float32, device=device
                ).flatten()
                if upper_row.numel() != upper_dim:
                    raise ValueError(
                        f"upper_vlm_feature dimension mismatch: expected {upper_dim}, got {upper_row.numel()}"
                    )
                upper_retrieval_features[index] = upper_row

        states = getattr(self, "_batched_memory_states", None)
        if states is None:
            states = {}
            self._batched_memory_states = states

        H = int(self.config.action_horizon)
        A = int(self.config.action_dim)
        positive: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        negative: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        trace_requested = bool(getattr(self, "memory_guidance_trace_dir", ""))
        trace_level = str(getattr(self, "memory_guidance_trace_level", "full")).lower()
        if trace_requested and trace_level not in ("full", "light", "bank"):
            raise ValueError(f"Unsupported memory guidance trace level: {trace_level}")
        full_trace = trace_requested and trace_level == "full"
        light_trace = trace_requested and trace_level == "light"
        positive_debug: list[dict] | None = [] if trace_requested else None
        negative_debug: list[dict] | None = [] if trace_requested else None
        light_trace_steps: list[list[dict]] | None = [[] for _ in range(bsize)] if light_trace else None
        full_trace_steps: list[list[dict]] | None = [[] for _ in range(bsize)] if full_trace else None
        static_mode = str(getattr(self, "memory_static_action_mode", "off")).lower()
        static_rows: list[dict] = []
        noise_rows = []
        active_ids = []
        guidance_gate_rows: list[dict | None] = []
        for index, record in enumerate(runtime):
            env_id = str(record["environment_id"])
            if bool(record.get("reset")) or env_id not in states:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(record["seed"]))
                states[env_id] = {"session": None, "call_count": 0, "generator": generator}
            slot = states[env_id]
            guidance_gate_rows.append(
                self._memory_guidance_suite_profile(
                    dict(record.get("context", {}) or {}),
                    str(getattr(self, "memory_guidance_time_version", "v1")),
                )
            )
            head_variant = str(getattr(self.task_head, "variant", "lower"))
            if head_variant == "upper" and not bool(record.get("upper_vlm_available", False)):
                positive.append(None)
                negative.append(None)
                if trace_requested:
                    assert positive_debug is not None and negative_debug is not None
                    positive_debug.append({"reason": "upper_unavailable"})
                    negative_debug.append({"reason": "upper_unavailable"})
                if noise is None:
                    noise_rows.append(
                        torch.randn((H, A), generator=slot["generator"], device=device)
                    )
                active_ids.append(env_id)
                continue
            session = slot.get("session")
            if session is None and self.memory_provider is not None:
                session = ActionMemorySession(
                    provider=self.memory_provider,
                    init_task_emb=task_embs[index],
                    k=int(getattr(self, "memory_top_k", 8)),
                    H=H,
                    progress=0.0,
                    search_result=memory_search_results[index],
                )
                slot["session"] = session
            elif session is not None:
                session.maybe_refresh(
                    new_task_emb=task_embs[index],
                    step_idx=int(slot["call_count"]),
                    refresh_every=int(getattr(self, "memory_refresh_every", 1)),
                    sim_threshold=float(getattr(self, "memory_refresh_sim_threshold", 0.0)),
                    k=int(getattr(self, "memory_top_k", 8)),
                    advance_steps=int(getattr(self, "replan_steps_hint", H)),
                    search_result=memory_search_results[index],
                )
                session = slot["session"]

            progress = float(np.clip(float(record.get("progress", 0.0)), 0.0, 1.0))
            similarity_accepted = session is not None and float(session.s_global) >= float(
                getattr(self, "memory_guidance_min_similarity", -1.0)
            )
            static_debug = None
            static_gate_applied = False
            if static_mode != "off" and session is not None:
                static_debug = session.static_action_diagnostics(
                    progress,
                    arm_dim=int(getattr(self, "memory_static_action_arm_dim", 6)),
                    action_threshold=float(
                        getattr(self, "memory_static_action_threshold", 1e-8)
                    ),
                    gate_fraction=float(
                        getattr(self, "memory_static_action_gate_fraction", 0.5)
                    ),
                )
                static_gate_applied = bool(
                    static_mode == "gate" and similarity_accepted and static_debug["gate"]
                )
                context = dict(record.get("context", {}) or {})
                static_rows.append(
                    {
                        "environment_id": env_id,
                        "task_id": int(context.get("task_id", -1)),
                        "episode_idx": int(context.get("episode_idx", -1)),
                        "episode_seed": int(context.get("episode_seed", -1)),
                        "inference_call": int(
                            context.get("inference_call", slot["call_count"])
                        ),
                        "retrieval_progress": float(
                            session.init_info.get("retrieval_progress", progress)
                        ),
                        "similarity_global": float(session.s_global),
                        "similarity_accepted": bool(similarity_accepted),
                        "mode": static_mode,
                        "gate_applied": static_gate_applied,
                        "estimated_frames": [
                            int(source["estimated_frame"])
                            for source in session.init_info.get("sources", [])
                            if "estimated_frame" in source
                        ],
                        **static_debug,
                    }
                )

            if session is not None and similarity_accepted and not static_gate_applied:
                blocks, weights, debug = session.guidance_tensors(progress, device)
                positive.append((blocks, weights))
                if trace_requested:
                    assert positive_debug is not None
                    if static_debug is not None:
                        debug["static_action"] = static_debug
                    positive_debug.append(debug)
            else:
                positive.append(None)
                if trace_requested:
                    assert positive_debug is not None
                    positive_debug.append(
                        {
                            "reason": (
                                "static_action_gate"
                                if static_gate_applied
                                else "below_similarity"
                            ),
                            "static_action": static_debug,
                        }
                    )

            neg = None
            neg_debug = {"reason": "disabled"}
            if bool(getattr(self, "use_negative_guidance", False)) and self.negative_memory_provider is not None:
                blocks, weights, neg_debug = self.negative_memory_provider.query_blocks(
                    task_embs[index],
                    k=int(getattr(self, "negative_memory_top_k", 4)),
                    horizon=H,
                    progress=progress,
                    min_similarity=float(getattr(self, "negative_memory_min_similarity", 0.975)),
                    min_confidence=float(getattr(self, "negative_memory_min_confidence", 0.75)),
                )
                if blocks is not None and weights is not None:
                    neg = (blocks, weights)
            negative.append(neg)
            if trace_requested:
                assert negative_debug is not None
                negative_debug.append(neg_debug)
            if noise is None:
                noise_rows.append(torch.randn((H, A), generator=slot["generator"], device=device))
            active_ids.append(env_id)

        self._last_static_action_batch = (
            {
                "schema_version": 1,
                "mode": static_mode,
                "rows": static_rows,
            }
            if static_mode != "off"
            else None
        )

        x_t = noise if noise is not None else torch.stack(noise_rows, dim=0)
        retrieval_finished = self._component_timing_mark(device)
        retrieval_ms = (
            0.0
            if retrieval_started is None or retrieval_finished is None
            else (retrieval_finished - retrieval_started) * 1000.0
        )
        base_denoise_ms = 0.0
        guidance_ms = 0.0
        time_version = str(getattr(self, "memory_guidance_time_version", "v1")).lower()
        if time_version not in ("v0", "v0_5", "v1", "v2"):
            raise ValueError(f"Unknown memory_guidance_time_version: {time_version}")
        if time_version == "v0_5":
            return self._sample_actions_batched_memory_guidance_v05(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                positive=positive,
                negative=negative,
                active_ids=active_ids,
                states=states,
                num_steps=int(num_steps),
            )
        unbounded_guidance = time_version == "v0"
        _, dt_schedule = self._full_interval_schedule(int(num_steps), int(num_steps))
        full_trace_x: list[list[torch.Tensor]] | None = (
            [
                [x_t[index : index + 1].detach().to(dtype=torch.float32, device="cpu")]
                for index in range(bsize)
            ]
            if full_trace
            else None
        )

        for step_index, dt_value in enumerate(dt_schedule):
            time_value = float(dt_schedule[step_index:].sum())
            time = torch.full((bsize,), time_value, dtype=torch.float32, device=device)
            base_started = self._component_timing_mark(device)
            v_base = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
            base_finished = self._component_timing_mark(device)
            if base_started is not None and base_finished is not None:
                base_denoise_ms += (base_finished - base_started) * 1000.0
            guidance_started = self._component_timing_mark(device)
            positive_rows = []
            negative_rows = []
            positive_step_debug: list[dict] = []
            negative_step_debug: list[dict] = []
            for index in range(bsize):
                pos = torch.zeros_like(v_base[index : index + 1])
                neg = torch.zeros_like(pos)
                pos_debug = {
                    "time": time_value,
                    "lambda": 0.0,
                    "responsibilities": torch.empty((1, 0), dtype=torch.float32),
                    "dist2": torch.empty((1, 0), dtype=torch.float32),
                    "guidance_raw": torch.zeros_like(pos, dtype=torch.float32, device="cpu"),
                    "cap_factor": torch.ones((1,), dtype=torch.float32),
                }
                neg_debug = {}
                if positive[index] is not None:
                    gate = guidance_gate_rows[index]
                    pos, pos_debug = self._memory_flow_guidance(
                        x_t[index : index + 1],
                        v_base[index : index + 1],
                        *positive[index],
                        time_value,
                        apply_norm_cap=not unbounded_guidance,
                        lambda_max_override=(None if gate is None else gate["lambda_max"]),
                        t_cut_override=(None if gate is None else gate["t_cut"]),
                        norm_cap_override=(None if gate is None else gate["norm_cap"]),
                        return_debug=full_trace,
                    )
                if negative[index] is not None:
                    neg, neg_debug = self._negative_memory_flow_guidance(
                        x_t[index : index + 1],
                        v_base[index : index + 1],
                        *negative[index],
                        time_value,
                        apply_norm_cap=not unbounded_guidance,
                        return_debug=full_trace,
                    )
                positive_rows.append(pos)
                negative_rows.append(neg)
                positive_step_debug.append(pos_debug)
                negative_step_debug.append(neg_debug)
            positive_guidance = torch.cat(positive_rows, dim=0)
            negative_guidance = torch.cat(negative_rows, dim=0)
            guidance = positive_guidance + negative_guidance
            if time_version == "v0":
                if not bool(torch.isfinite(guidance).all()):
                    raise FloatingPointError("V0 produced non-finite unbounded memory guidance")
                velocity = v_base + guidance
            elif time_version == "v2":
                velocity = self._mix_guidance_direction_first(
                    v_base,
                    guidance,
                    float(getattr(self, "memory_guidance_v2_magnitude_cap", 0.10)),
                )
            else:
                guidance = self._cap_guidance_relative(
                    guidance,
                    v_base,
                    float(getattr(self, "memory_guidance_total_norm_cap", 0.2)),
                )
                velocity = v_base + guidance
            if time_version in {"v0", "v1"} and hasattr(self, "guidance_residual_adapter"):
                velocity = self._apply_guidance_adapter(x_t, v_base, velocity - v_base, time)
            guidance_finished = self._component_timing_mark(device)
            if guidance_started is not None and guidance_finished is not None:
                guidance_ms += (guidance_finished - guidance_started) * 1000.0
            if light_trace:
                assert light_trace_steps is not None
                guidance_final = velocity - v_base
                for index in range(bsize):
                    light_trace_steps[index].append(
                        self._summarize_memory_guidance_step(
                            v_base=v_base[index : index + 1],
                            guidance_positive=positive_guidance[index : index + 1],
                            guidance_negative=negative_guidance[index : index + 1],
                            guidance_final=guidance_final[index : index + 1],
                            time=time_value,
                            step_index=step_index,
                        )
                    )
            elif full_trace:
                assert full_trace_steps is not None
                guidance_final = velocity - v_base
                for index in range(bsize):
                    trace_record = {
                        **positive_step_debug[index],
                        "v_base": v_base[index : index + 1].detach().to(dtype=torch.float32, device="cpu"),
                        "guidance": guidance_final[index : index + 1]
                        .detach()
                        .to(dtype=torch.float32, device="cpu"),
                        "guidance_positive": positive_guidance[index : index + 1]
                        .detach()
                        .to(dtype=torch.float32, device="cpu"),
                        "guidance_negative": negative_guidance[index : index + 1]
                        .detach()
                        .to(dtype=torch.float32, device="cpu"),
                    }
                    if negative_step_debug[index]:
                        trace_record["negative_responsibilities"] = negative_step_debug[index][
                            "responsibilities"
                        ]
                        trace_record["negative_dist2"] = negative_step_debug[index]["dist2"]
                    full_trace_steps[index].append(trace_record)
            x_t = x_t - float(dt_value) * velocity
            if full_trace:
                assert full_trace_x is not None
                for index in range(bsize):
                    full_trace_x[index].append(
                        x_t[index : index + 1].detach().to(dtype=torch.float32, device="cpu")
                    )
            if time_version == "v0" and not bool(torch.isfinite(x_t).all()):
                raise FloatingPointError("V0 produced a non-finite flow state")

        if trace_requested:
            assert positive_debug is not None
            assert negative_debug is not None
            for index, record in enumerate(runtime):
                positive_item = positive[index]
                negative_item = negative[index]
                self._write_memory_guidance_trace(
                    trace_x=(full_trace_x[index] if full_trace_x is not None else []),
                    trace_steps=(
                        full_trace_steps[index]
                        if full_trace_steps is not None
                        else light_trace_steps[index]
                        if light_trace_steps is not None
                        else []
                    ),
                    blocks=positive_item[0] if positive_item is not None else None,
                    weights=positive_item[1] if positive_item is not None else None,
                    retrieval_debug=positive_debug[index],
                    negative_weights=negative_item[1] if negative_item is not None else None,
                    negative_debug=negative_debug[index],
                    task_emb=task_embs[index],
                    progress=float(np.clip(float(record.get("progress", 0.0)), 0.0, 1.0)),
                    progress_source="client",
                    num_steps=int(num_steps),
                    trace_context=dict(record.get("context", {}) or {}),
                    lower_retrieval_feature=lower_retrieval_features[index],
                    upper_retrieval_feature=upper_retrieval_features[index],
                    upper_vlm_age=upper_age_rows[index],
                    upper_vlm_available=upper_available_rows[index],
                    final_action=x_t[index],
                )

        for env_id in active_ids:
            states[env_id]["call_count"] = int(states[env_id]["call_count"]) + 1
        self._last_component_timing = {
            "prefix_ms": float(getattr(self, "_component_prefix_ms", 0.0)),
            "base_denoise_ms": float(base_denoise_ms),
            "base_vla_ms": float(getattr(self, "_component_prefix_ms", 0.0) + base_denoise_ms),
            "retrieval_ms": float(retrieval_ms),
            "guidance_ms": float(guidance_ms),
            "nfe": int(num_steps),
            "batch_size": int(bsize),
        }
        return x_t

    @staticmethod
    def _v05_extra_skip(v_actual: torch.Tensor, v_flow: torch.Tensor) -> torch.Tensor:
        """Return conservative extra virtual steps represented by a mixed velocity."""
        actual_norm = v_actual.to(dtype=torch.float32).flatten(1).norm(dim=1)
        flow_norm = v_flow.to(dtype=torch.float32).flatten(1).norm(dim=1).clamp_min(1e-12)
        return (torch.floor(actual_norm / flow_norm).to(dtype=torch.int64) - 1).clamp_min(0)

    def _sample_actions_batched_memory_guidance_v05(
        self,
        *,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        positive: list[tuple[torch.Tensor, torch.Tensor] | None],
        negative: list[tuple[torch.Tensor, torch.Tensor] | None],
        active_ids: list[str],
        states: dict,
        num_steps: int,
    ) -> torch.Tensor:
        """Run raw velocity-gated guidance with optional conservative dynamic NFE."""
        bsize = int(x_t.shape[0])
        dynamic_nfe = bool(getattr(self, "memory_guidance_v05_dynamic_nfe", False))
        fine_ratio = float(getattr(self, "memory_guidance_v05_fine_ratio", 1.2))
        fine_scale = float(getattr(self, "memory_guidance_v05_fine_scale", 0.2))
        confirm_steps = int(getattr(self, "memory_guidance_v05_fine_confirm_steps", 2))
        if fine_ratio <= 0.0 or not 0.0 <= fine_scale <= 1.0 or confirm_steps <= 0:
            raise ValueError("Invalid V0.5 fine-mode configuration")
        if dynamic_nfe:
            return self._sample_actions_batched_memory_guidance_v05_dynamic_v2(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                positive=positive,
                negative=negative,
                active_ids=active_ids,
                states=states,
                num_steps=num_steps,
            )

        dynamic_denominator = [int(num_steps)] * bsize
        fine_mode = [False] * bsize
        below_count = [0] * bsize
        time_remaining = torch.ones(bsize, dtype=torch.float32, device=x_t.device)
        executed_steps = [0] * bsize
        step_index = 0

        while True:
            active_rows = [
                executed_steps[index] < num_steps and float(time_remaining[index].item()) > 1e-7
                for index in range(bsize)
            ]
            if not any(active_rows):
                break

            model_time = torch.where(
                torch.tensor(active_rows, dtype=torch.bool, device=x_t.device),
                time_remaining,
                torch.zeros_like(time_remaining),
            )
            v_flow = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, model_time)
            guidance_rows = []
            for index in range(bsize):
                positive_raw = torch.zeros_like(v_flow[index : index + 1])
                negative_raw = torch.zeros_like(positive_raw)
                if active_rows[index] and positive[index] is not None:
                    positive_raw, _ = self._memory_flow_guidance(
                        x_t[index : index + 1],
                        v_flow[index : index + 1],
                        *positive[index],
                        float(time_remaining[index].item()),
                        apply_norm_cap=False,
                        scale_override=1.0,
                    )
                if active_rows[index] and negative[index] is not None:
                    negative_raw, _ = self._negative_memory_flow_guidance(
                        x_t[index : index + 1],
                        v_flow[index : index + 1],
                        *negative[index],
                        float(time_remaining[index].item()),
                        apply_norm_cap=False,
                        scale_override=1.0,
                    )
                guidance_rows.append(positive_raw + negative_raw)
            guidance_raw = torch.cat(guidance_rows, dim=0)

            flow_norm = v_flow.to(dtype=torch.float32).flatten(1).norm(dim=1).clamp_min(1e-12)
            guidance_norm = guidance_raw.to(dtype=torch.float32).flatten(1).norm(dim=1)
            coarse_velocity = v_flow + guidance_raw
            extra_skip = self._v05_extra_skip(coarse_velocity, v_flow)
            scales = torch.ones(bsize, dtype=torch.float32, device=x_t.device)

            dt_rows = []
            virtual_skip_rows = []
            for index in range(bsize):
                if not active_rows[index]:
                    dt_rows.append(0.0)
                    virtual_skip_rows.append(0)
                    continue
                dt_value = min(
                    1.0 / dynamic_denominator[index],
                    float(time_remaining[index].item()),
                )
                dt_rows.append(dt_value)

                has_guidance = positive[index] is not None or negative[index] is not None
                fine_candidate = has_guidance and (
                    float(guidance_norm[index].item()) < fine_ratio * float(flow_norm[index].item())
                )
                if not fine_mode[index] and fine_candidate:
                    below_count[index] += 1
                elif not fine_mode[index]:
                    below_count[index] = 0
                if not fine_mode[index] and below_count[index] >= confirm_steps:
                    fine_mode[index] = True

                if fine_mode[index]:
                    scales[index] = fine_scale
                    virtual_skip_rows.append(0)
                elif dynamic_nfe and not fine_candidate:
                    requested_skip = int(extra_skip[index].item())
                    # Never claim more virtual progress than fits before the
                    # residual interval. Floor keeps the fast-forward
                    # conservative with respect to the mixed-velocity ratio.
                    time_slots = max(1, int(math.floor(float(time_remaining[index].item()) / dt_value + 1e-7)))
                    effective_skip = min(
                        requested_skip,
                        dynamic_denominator[index] - 1,
                        time_slots - 1,
                    )
                    virtual_skip_rows.append(max(0, effective_skip))
                    dynamic_denominator[index] -= max(0, effective_skip)
                else:
                    virtual_skip_rows.append(0)

            dt = torch.tensor(dt_rows, dtype=torch.float32, device=x_t.device)
            velocity = v_flow + scales.view(-1, 1, 1).to(dtype=v_flow.dtype) * guidance_raw
            if not bool(torch.isfinite(velocity).all()):
                raise FloatingPointError("V0.5 produced non-finite mixed velocity")
            active_mask = torch.tensor(active_rows, dtype=torch.bool, device=x_t.device).view(-1, 1, 1)
            x_t = torch.where(active_mask, x_t - dt.view(-1, 1, 1) * velocity, x_t)
            if not bool(torch.isfinite(x_t).all()):
                raise FloatingPointError("V0.5 produced a non-finite flow state")

            virtual_multiplier = 1.0 + torch.tensor(
                virtual_skip_rows, dtype=torch.float32, device=x_t.device
            )
            time_remaining = torch.clamp(time_remaining - dt * virtual_multiplier, min=0.0)
            for index, active in enumerate(active_rows):
                if active:
                    executed_steps[index] += 1
            step_index += 1
            if step_index > num_steps:
                raise RuntimeError("V0.5 exceeded its configured maximum NFE")

        if bool(getattr(self, "debug_memory", False)):
            logging.info(
                "V0.5 sampler: dynamic_nfe=%s executed=%s denominator=%s fine=%s remaining_t=%s",
                dynamic_nfe,
                executed_steps,
                dynamic_denominator,
                fine_mode,
                [round(float(value), 6) for value in time_remaining.detach().cpu().tolist()],
            )
        stats = getattr(self, "_v05_nfe_stats", None)
        if not isinstance(stats, dict):
            stats = {
                "samples": 0,
                "fine_samples": 0,
                "individual_hist": [0] * (num_steps + 1),
                "batch_hist": [0] * (num_steps + 1),
                "denominator_hist": [0] * (num_steps + 1),
                "next_log": 1000,
            }
            self._v05_nfe_stats = stats
        stats["samples"] += bsize
        stats["fine_samples"] += sum(fine_mode)
        for count in executed_steps:
            stats["individual_hist"][count] += 1
        for count in dynamic_denominator:
            stats["denominator_hist"][count] += 1
        stats["batch_hist"][step_index] += 1
        if stats["samples"] >= stats["next_log"]:
            logging.info(
                "V0.5 NFE stats: dynamic=%s samples=%d fine=%d individual_hist=%s "
                "denominator_hist=%s batch_hist=%s",
                dynamic_nfe,
                stats["samples"],
                stats["fine_samples"],
                stats["individual_hist"][1:],
                stats["denominator_hist"][1:],
                stats["batch_hist"][1:],
            )
            while stats["next_log"] <= stats["samples"]:
                stats["next_log"] += 1000
        for env_id in active_ids:
            states[env_id]["call_count"] = int(states[env_id]["call_count"]) + 1
        return x_t

    @staticmethod
    def _v05_dynamic_next_grid(grid: int, fine_region: bool) -> int:
        if grid <= 0:
            return 0
        if fine_region:
            return grid - 1
        return max(0, grid // 2 - 1)

    def _sample_actions_batched_memory_guidance_v05_dynamic_v2(
        self,
        *,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        positive: list[tuple[torch.Tensor, torch.Tensor] | None],
        negative: list[tuple[torch.Tensor, torch.Tensor] | None],
        active_ids: list[str],
        states: dict,
        num_steps: int,
    ) -> torch.Tensor:
        """Arc-length V0.5: quantized NFE grid with normalized v*dt budget."""
        bsize = int(x_t.shape[0])
        fine_ratio = float(getattr(self, "memory_guidance_v05_fine_ratio", 1.2))
        guidance_scale = float(getattr(self, "memory_guidance_v05_dynamic_guidance_scale", 0.5))
        if fine_ratio <= 0.0 or guidance_scale < 0.0:
            raise ValueError("Invalid V0.5 dynamic-v2 configuration")

        grid = [int(num_steps)] * bsize
        executed_steps = [0] * bsize
        fine_seen = [False] * bsize
        step_index = 0
        while True:
            active_rows = [value > 0 for value in grid]
            if not any(active_rows):
                break

            model_time = torch.tensor(
                [value / num_steps if active else 0.0 for value, active in zip(grid, active_rows)],
                dtype=torch.float32,
                device=x_t.device,
            )
            v_flow = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, model_time)
            guidance_rows = []
            for index in range(bsize):
                positive_raw = torch.zeros_like(v_flow[index : index + 1])
                negative_raw = torch.zeros_like(positive_raw)
                if active_rows[index] and positive[index] is not None:
                    positive_raw, _ = self._memory_flow_guidance(
                        x_t[index : index + 1],
                        v_flow[index : index + 1],
                        *positive[index],
                        float(model_time[index].item()),
                        apply_norm_cap=False,
                        scale_override=1.0,
                    )
                if active_rows[index] and negative[index] is not None:
                    negative_raw, _ = self._negative_memory_flow_guidance(
                        x_t[index : index + 1],
                        v_flow[index : index + 1],
                        *negative[index],
                        float(model_time[index].item()),
                        apply_norm_cap=False,
                        scale_override=1.0,
                    )
                guidance_rows.append(positive_raw + negative_raw)

            guidance_raw = torch.cat(guidance_rows, dim=0)
            guidance = guidance_scale * guidance_raw
            velocity = v_flow + guidance.to(dtype=v_flow.dtype)
            if not bool(torch.isfinite(velocity).all()):
                raise FloatingPointError("V0.5 dynamic-v2 produced non-finite mixed velocity")

            flow_norm = v_flow.to(dtype=torch.float32).flatten(1).norm(dim=1).clamp_min(1e-12)
            guidance_norm = guidance.to(dtype=torch.float32).flatten(1).norm(dim=1)
            next_grid = list(grid)
            progress_rows = []
            for index in range(bsize):
                if not active_rows[index]:
                    progress_rows.append(0.0)
                    continue
                has_guidance = positive[index] is not None or negative[index] is not None
                fine_region = has_guidance and (
                    float(guidance_norm[index].item()) < fine_ratio * float(flow_norm[index].item())
                )
                fine_seen[index] = fine_seen[index] or fine_region
                next_grid[index] = self._v05_dynamic_next_grid(grid[index], fine_region)
                progress_rows.append((grid[index] - next_grid[index]) / num_steps)

            velocity_float = velocity.to(dtype=torch.float32)
            speed_rms = velocity_float.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-12)
            progress = torch.tensor(progress_rows, dtype=torch.float32, device=x_t.device)
            dt = progress / speed_rms
            active_mask = torch.tensor(active_rows, dtype=torch.bool, device=x_t.device).view(-1, 1, 1)
            x_t = torch.where(active_mask, x_t - dt.view(-1, 1, 1) * velocity, x_t)
            if not bool(torch.isfinite(x_t).all()):
                raise FloatingPointError("V0.5 dynamic-v2 produced a non-finite flow state")

            for index, active in enumerate(active_rows):
                if active:
                    executed_steps[index] += 1
            grid = next_grid
            step_index += 1
            if step_index > num_steps:
                raise RuntimeError("V0.5 dynamic-v2 exceeded its 0.1 progress grid")

        stats = getattr(self, "_v05_nfe_stats", None)
        if not isinstance(stats, dict) or stats.get("algorithm") != "arc_length_v2":
            stats = {
                "algorithm": "arc_length_v2",
                "samples": 0,
                "fine_samples": 0,
                "individual_hist": [0] * (num_steps + 1),
                "batch_hist": [0] * (num_steps + 1),
                "next_log": 1000,
            }
            self._v05_nfe_stats = stats
        stats["samples"] += bsize
        stats["fine_samples"] += sum(fine_seen)
        for count in executed_steps:
            stats["individual_hist"][count] += 1
        stats["batch_hist"][step_index] += 1
        if stats["samples"] >= stats["next_log"]:
            logging.info(
                "V0.5 dynamic-v2 NFE stats: samples=%d fine=%d individual_hist=%s batch_hist=%s",
                stats["samples"],
                stats["fine_samples"],
                stats["individual_hist"][1:],
                stats["batch_hist"][1:],
            )
            while stats["next_log"] <= stats["samples"]:
                stats["next_log"] += 1000
        for env_id in active_ids:
            states[env_id]["call_count"] = int(states[env_id]["call_count"]) + 1
        return x_t

    def _memory_flow_guidance(
        self,
        x_t: torch.Tensor,
        v_base: torch.Tensor,
        blocks: torch.Tensor,
        weights: torch.Tensor,
        t_denoise: float,
        *,
        apply_norm_cap: bool = True,
        scale_override: float | None = None,
        lambda_max_override: float | None = None,
        t_cut_override: float | None = None,
        norm_cap_override: float | None = None,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Compute early-phase GPM mixture guidance in velocity coordinates."""
        lambda_max = (
            float(getattr(self, "memory_guidance_lambda_max", 0.2))
            if lambda_max_override is None
            else float(lambda_max_override)
        )
        t_cut = (
            float(getattr(self, "memory_guidance_t_cut", 0.3))
            if t_cut_override is None
            else float(t_cut_override)
        )
        sigma = max(float(getattr(self, "memory_guidance_sigma", 0.3)), 1e-6)
        if scale_override is not None:
            scale = float(scale_override)
        elif float(t_denoise) <= t_cut + 1e-6:
            scale = 0.0
        else:
            scale = lambda_max * (float(t_denoise) - t_cut) / max(1e-6, 1.0 - t_cut)
        blocks = blocks.to(device=x_t.device, dtype=torch.float32)
        weights = weights.to(device=x_t.device, dtype=torch.float32)
        weights = weights / (weights.sum() + 1e-12)

        x = x_t.to(dtype=torch.float32)
        diff = blocks.unsqueeze(0) - x.unsqueeze(1)
        dist2 = diff.square().sum(dim=(2, 3))
        log_w = torch.log(weights.clamp_min(1e-12)).unsqueeze(0)
        logits = log_w - dist2 / (2.0 * sigma * sigma)
        resp = torch.softmax(logits, dim=1)
        attract = (resp[:, :, None, None] * diff).sum(dim=1) / (sigma * sigma)

        # The sampler integrates with negative dt, so an attraction direction in action
        # space must be subtracted from the velocity to move x_t toward memory modes.
        guidance_raw = -scale * attract
        guidance = guidance_raw
        cap_ratio = (
            float(getattr(self, "memory_guidance_norm_cap", 0.2))
            if norm_cap_override is None
            else float(norm_cap_override)
        )
        factor = torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
        if apply_norm_cap and cap_ratio > 0.0:
            g_norm = guidance.flatten(1).norm(dim=1).clamp_min(1e-12)
            v_norm = v_base.to(dtype=torch.float32).flatten(1).norm(dim=1)
            cap = cap_ratio * v_norm
            factor = torch.clamp(cap / g_norm, max=1.0)
            guidance = guidance * factor.view(-1, 1, 1)
        guidance = guidance.to(dtype=v_base.dtype)
        if not return_debug:
            return guidance, {}
        debug = {
            "time": float(t_denoise),
            "lambda": float(scale),
            "responsibilities": resp.detach().to(dtype=torch.float32, device="cpu"),
            "dist2": dist2.detach().to(dtype=torch.float32, device="cpu"),
            "guidance_raw": guidance_raw.detach().to(dtype=torch.float32, device="cpu"),
            "cap_factor": factor.detach().to(dtype=torch.float32, device="cpu"),
            "suite_gate": {
                "lambda_max": lambda_max,
                "t_cut": t_cut,
                "norm_cap": cap_ratio,
            },
        }
        return guidance, debug

    def _memory_guidance_suite_profile(self, context: dict, version: str) -> dict | None:
        profiles = getattr(self, "memory_guidance_suite_gate", None)
        if not profiles:
            return None
        task_id = str(int(context.get("task_id", -1)))
        profile = profiles.get(task_id)
        if profile is None:
            raise KeyError(f"No suite guidance gate profile for task_id={task_id}")
        t_cut = float(profile["t_cut"])
        if not 0.0 <= t_cut < 1.0:
            raise ValueError(f"Invalid suite guidance t_cut for task {task_id}: {t_cut}")
        if version == "v0":
            # V0 is intentionally unbounded. Its gate scales the raw field and
            # changes its active interval; it must never introduce a norm cap.
            return {
                "lambda_max": float(profile["v0_lambda_max"]),
                "t_cut": t_cut,
                "norm_cap": float(getattr(self, "memory_guidance_norm_cap", 0.2)),
            }
        if version == "v1":
            # V1 keeps its raw field scale and gates the relative-to-flow cap.
            return {
                "lambda_max": float(getattr(self, "memory_guidance_lambda_max", 0.2)),
                "t_cut": t_cut,
                "norm_cap": float(profile["v1_norm_cap"]),
            }
        raise ValueError(f"Suite guidance gate only supports v0/v1, got {version}")

    def _negative_memory_flow_guidance(
        self,
        x_t: torch.Tensor,
        v_base: torch.Tensor,
        blocks: torch.Tensor,
        weights: torch.Tensor,
        t_denoise: float,
        *,
        apply_norm_cap: bool = True,
        scale_override: float | None = None,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Compute early-phase repulsion from task-gated failure action modes."""
        lambda_max = float(getattr(self, "memory_guidance_lambda_max", 0.2))
        beta = float(getattr(self, "negative_guidance_beta", 0.1))
        t_cut = float(getattr(self, "memory_guidance_t_cut", 0.3))
        sigma = max(float(getattr(self, "negative_guidance_sigma", 0.3)), 1e-6)
        if scale_override is not None:
            scale = float(scale_override) * beta
        else:
            scale = 0.0 if float(t_denoise) <= t_cut + 1e-6 else (
                lambda_max * beta * (float(t_denoise) - t_cut) / max(1e-6, 1.0 - t_cut)
            )
        blocks = blocks.to(device=x_t.device, dtype=torch.float32)
        weights = weights.to(device=x_t.device, dtype=torch.float32)
        weights = weights / (weights.sum() + 1e-12)
        x = x_t.to(dtype=torch.float32)
        diff = blocks.unsqueeze(0) - x.unsqueeze(1)
        dist2 = diff.square().sum(dim=(2, 3))
        logits = torch.log(weights.clamp_min(1e-12)).unsqueeze(0) - dist2 / (2.0 * sigma * sigma)
        responsibilities = torch.softmax(logits, dim=1)
        attract_negative = (responsibilities[:, :, None, None] * diff).sum(dim=1) / (sigma * sigma)

        # Positive velocity sign repels in action space because the sampler integrates with negative dt.
        guidance_raw = scale * attract_negative
        if apply_norm_cap:
            guidance = self._cap_guidance_relative(
                guidance_raw,
                v_base,
                float(getattr(self, "negative_guidance_norm_cap", 0.10)),
            )
        else:
            guidance = guidance_raw
        guidance = guidance.to(dtype=v_base.dtype)
        if not return_debug:
            return guidance, {}
        return guidance, {
            "responsibilities": responsibilities.detach().to(dtype=torch.float32, device="cpu"),
            "dist2": dist2.detach().to(dtype=torch.float32, device="cpu"),
        }

    def _joint_outcome_memory_flow_guidance(
        self,
        x_t: torch.Tensor,
        positive_blocks: torch.Tensor,
        positive_weights: torch.Tensor,
        negative_blocks: torch.Tensor,
        negative_weights: torch.Tensor,
        t_denoise: float,
        *,
        scale_override: float | None = None,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Compute one success-posterior field from positive and failure KDEs."""
        lambda_max = float(getattr(self, "memory_guidance_lambda_max", 0.2))
        t_cut = float(getattr(self, "memory_guidance_t_cut", 0.3))
        if scale_override is not None:
            scale = float(scale_override)
        elif float(t_denoise) <= t_cut + 1e-6:
            scale = 0.0
        else:
            scale = lambda_max * (float(t_denoise) - t_cut) / max(1e-6, 1.0 - t_cut)

        failure_prior = float(getattr(self, "memory_joint_failure_prior", 0.5))
        if not 0.0 < failure_prior < 1.0:
            raise ValueError("memory_joint_failure_prior must be strictly between 0 and 1")
        positive_sigma = max(float(getattr(self, "memory_guidance_sigma", 0.3)), 1e-6)
        negative_sigma = max(float(getattr(self, "negative_guidance_sigma", 0.3)), 1e-6)
        x = x_t.to(dtype=torch.float32)

        def component(
            blocks: torch.Tensor,
            weights: torch.Tensor,
            sigma: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            blocks = blocks.to(device=x.device, dtype=torch.float32)
            weights = weights.to(device=x.device, dtype=torch.float32).flatten()
            weights = weights / (weights.sum() + 1e-12)
            diff = blocks.unsqueeze(0) - x.unsqueeze(1)
            dist2 = diff.square().sum(dim=(2, 3))
            logits = torch.log(weights.clamp_min(1e-12)).unsqueeze(0) - dist2 / (2.0 * sigma * sigma)
            responsibilities = torch.softmax(logits, dim=1)
            score = (responsibilities[:, :, None, None] * diff).sum(dim=1) / (sigma * sigma)
            dimensions = int(x.shape[1] * x.shape[2])
            log_normalizer = 0.5 * dimensions * math.log(2.0 * math.pi * sigma * sigma)
            log_density = torch.logsumexp(logits, dim=1) - log_normalizer
            return score, log_density, responsibilities

        positive_score, positive_log_density, positive_resp = component(
            positive_blocks,
            positive_weights,
            positive_sigma,
        )
        negative_score, negative_log_density, negative_resp = component(
            negative_blocks,
            negative_weights,
            negative_sigma,
        )
        log_positive_joint = math.log1p(-failure_prior) + positive_log_density
        log_negative_joint = math.log(failure_prior) + negative_log_density
        failure_posterior = torch.sigmoid(log_negative_joint - log_positive_joint)

        # In action coordinates this is grad(log p(success | x)). The sampler uses
        # x <- x - dt*v, so the velocity correction has the opposite sign.
        action_score = failure_posterior.view(-1, 1, 1) * (positive_score - negative_score)
        guidance = (-scale * action_score).to(dtype=x_t.dtype)
        if not return_debug:
            return guidance, {}
        return guidance, {
            "failure_prior": failure_prior,
            "failure_posterior": failure_posterior.detach().to(dtype=torch.float32, device="cpu"),
            "positive_log_density": positive_log_density.detach().to(dtype=torch.float32, device="cpu"),
            "negative_log_density": negative_log_density.detach().to(dtype=torch.float32, device="cpu"),
            "positive_responsibilities": positive_resp.detach().to(dtype=torch.float32, device="cpu"),
            "negative_responsibilities": negative_resp.detach().to(dtype=torch.float32, device="cpu"),
        }

    @staticmethod
    def _cap_guidance_relative(
        guidance: torch.Tensor,
        v_base: torch.Tensor,
        cap_ratio: float,
    ) -> torch.Tensor:
        if cap_ratio <= 0.0:
            return guidance
        guidance_float = guidance.to(dtype=torch.float32)
        g_norm = guidance_float.flatten(1).norm(dim=1).clamp_min(1e-12)
        v_norm = v_base.to(dtype=torch.float32).flatten(1).norm(dim=1)
        factor = torch.clamp((float(cap_ratio) * v_norm) / g_norm, max=1.0).view(-1, 1, 1)
        return (guidance_float * factor).to(dtype=guidance.dtype)

    def _write_memory_guidance_trace(
        self,
        *,
        trace_x: list[torch.Tensor],
        trace_steps: list[dict],
        blocks: torch.Tensor | None,
        weights: torch.Tensor | None,
        retrieval_debug: dict,
        negative_weights: torch.Tensor | None,
        negative_debug: dict,
        task_emb: torch.Tensor | None,
        progress: float,
        progress_source: str,
        num_steps: int,
        trace_context: dict | None = None,
        lower_retrieval_feature: torch.Tensor | None = None,
        upper_retrieval_feature: torch.Tensor | None = None,
        upper_vlm_age: float | None = None,
        upper_vlm_available: bool | None = None,
        final_action: torch.Tensor | None = None,
    ) -> None:
        """Persist a versioned, pickle-free full or light trace for offline analysis."""
        trace_dir = Path(str(getattr(self, "memory_guidance_trace_dir", "")))
        trace_dir.mkdir(parents=True, exist_ok=True)
        context = dict(trace_context if trace_context is not None else (getattr(self, "_trace_context", {}) or {}))
        episode = int(context.get("episode_idx", -1))
        call = int(context.get("policy_call_idx", getattr(self, "_sample_call_count", 0)))
        task_id = int(context.get("task_id", -1))
        task_suite = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in str(context.get("task_suite", "unknown"))
        )
        stem = f"{task_suite}_task_{task_id:03d}_episode_{episode:03d}_call_{call:04d}"
        trace_level = str(getattr(self, "memory_guidance_trace_level", "full")).lower()
        positive_retrieval = self._trace_retrieval_summary(retrieval_debug, weights, positive=True)
        negative_retrieval = self._trace_retrieval_summary(negative_debug, negative_weights, positive=False)
        if trace_level == "bank":
            if final_action is None:
                raise ValueError("Bank traces require the final action chunk")

            def cpu_array(value: torch.Tensor | None) -> np.ndarray:
                if value is None:
                    return np.empty((0,), dtype=np.float32)
                return value.detach().to(dtype=torch.float32, device="cpu").numpy()

            action = cpu_array(final_action)
            if action.ndim == 2:
                action = action[None, None, ...]
            elif action.ndim == 3:
                action = action[None, ...]
            if action.ndim != 4:
                raise ValueError(f"Unexpected final action shape for bank trace: {action.shape}")
            arrays = {
                "schema_version": np.asarray(3, dtype=np.int64),
                "x_trajectory": action,
                "task_embedding": cpu_array(task_emb),
                "lower_retrieval_feature": cpu_array(lower_retrieval_feature),
                "upper_retrieval_feature": cpu_array(upper_retrieval_feature),
                "upper_vlm_age": np.asarray(
                    np.nan if upper_vlm_age is None else upper_vlm_age, dtype=np.float32
                ),
                "upper_vlm_available": np.asarray(
                    False if upper_vlm_available is None else upper_vlm_available, dtype=np.bool_
                ),
                "retrieval_weights": np.asarray(positive_retrieval["weights"], dtype=np.float32),
                "retrieval_scores": np.asarray(positive_retrieval["scores"], dtype=np.float32),
                "memory_indices": np.asarray(positive_retrieval["indices"], dtype=np.int64),
                "window_starts": np.asarray(positive_retrieval["window_starts"], dtype=np.int64),
                "negative_retrieval_weights": np.asarray(negative_retrieval["weights"], dtype=np.float32),
                "negative_retrieval_scores": np.asarray(negative_retrieval["scores"], dtype=np.float32),
                "negative_memory_indices": np.asarray(negative_retrieval["indices"], dtype=np.int64),
            }
            record = {
                "schema_version": 3,
                "trace_level": "bank",
                **context,
                "task_id": task_id,
                "episode_idx": episode,
                "policy_call_idx": call,
                "progress": float(progress),
                "progress_source": progress_source,
                "num_steps": int(num_steps),
                "bank_trace_schema_complete": True,
                "task_embedding_dim": int(arrays["task_embedding"].size),
                "lower_feature_dim": int(arrays["lower_retrieval_feature"].size),
                "upper_feature_dim": int(arrays["upper_retrieval_feature"].size),
                "upper_vlm_available": bool(arrays["upper_vlm_available"]),
                "positive_selected": len(positive_retrieval["indices"]),
                "negative_selected": len(negative_retrieval["indices"]),
            }
            get_async_trace_writer(self, trace_dir).submit(stem, arrays, record)
            return
        if trace_level == "light":
            final_path = trace_dir / f"{stem}.light.jsonl"
            if final_path.exists():
                raise FileExistsError(f"Refusing to overwrite an existing guidance trace: {final_path}")
            light_record = {
                "schema_version": 2,
                "trace_level": "light",
                "trace_id": stem,
                **context,
                "task_id": task_id,
                "episode_idx": episode,
                "policy_call_idx": call,
                "progress": float(progress),
                "progress_source": progress_source,
                "num_steps": int(num_steps),
                "positive_retrieval": positive_retrieval,
                "negative_retrieval": negative_retrieval,
                "steps": trace_steps,
            }
            self._atomic_write_json_line(final_path, light_record)
            self._append_memory_guidance_manifest(
                trace_dir,
                {
                    "schema_version": 2,
                    "trace_level": "light",
                    "trace": final_path.name,
                    **context,
                    "progress": float(progress),
                    "progress_source": progress_source,
                    "num_steps": int(num_steps),
                    "positive_selected": len(positive_retrieval["indices"]),
                    "negative_selected": len(negative_retrieval["indices"]),
                },
            )
            return
        if trace_level != "full":
            raise ValueError(f"Unsupported memory guidance trace level: {trace_level}")

        final_path = trace_dir / f"{stem}.npz"
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing guidance trace: {final_path}")

        arrays = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "x_trajectory": torch.stack(trace_x).numpy(),
            "v_base": torch.stack([step["v_base"] for step in trace_steps]).numpy(),
            "guidance_raw": torch.stack([step["guidance_raw"] for step in trace_steps]).numpy(),
            "guidance_clipped": torch.stack([step["guidance"] for step in trace_steps]).numpy(),
            "guidance_positive": torch.stack([step["guidance_positive"] for step in trace_steps]).numpy(),
            "guidance_negative": torch.stack([step["guidance_negative"] for step in trace_steps]).numpy(),
            "responsibilities": torch.stack([step["responsibilities"] for step in trace_steps]).numpy(),
            "dist2": torch.stack([step["dist2"] for step in trace_steps]).numpy(),
            "cap_factor": torch.stack([step["cap_factor"] for step in trace_steps]).numpy(),
            "time": np.asarray([step["time"] for step in trace_steps], dtype=np.float32),
            "lambda": np.asarray([step["lambda"] for step in trace_steps], dtype=np.float32),
            "memory_blocks": (
                blocks.detach().to(dtype=torch.float32, device="cpu").numpy()
                if blocks is not None
                else np.empty((0,), dtype=np.float32)
            ),
            "retrieval_weights": (
                weights.detach().to(dtype=torch.float32, device="cpu").numpy()
                if weights is not None
                else np.empty((0,), dtype=np.float32)
            ),
            "retrieval_scores": np.asarray(retrieval_debug.get("scores", []), dtype=np.float32),
            "memory_indices": np.asarray(retrieval_debug.get("memory_indices", []), dtype=np.int64),
            "window_starts": np.asarray(retrieval_debug.get("window_starts", []), dtype=np.int64),
            "task_embedding": (
                task_emb.detach().to(dtype=torch.float32, device="cpu").numpy()
                if task_emb is not None
                else np.empty((0,), dtype=np.float32)
            ),
            "lower_retrieval_feature": (
                lower_retrieval_feature.detach().to(dtype=torch.float32, device="cpu").numpy()
                if lower_retrieval_feature is not None
                else np.empty((0,), dtype=np.float32)
            ),
            "upper_retrieval_feature": (
                upper_retrieval_feature.detach().to(dtype=torch.float32, device="cpu").numpy()
                if upper_retrieval_feature is not None
                else np.empty((0,), dtype=np.float32)
            ),
            "upper_vlm_age": np.asarray(
                np.nan if upper_vlm_age is None else upper_vlm_age, dtype=np.float32
            ),
            "upper_vlm_available": np.asarray(
                False if upper_vlm_available is None else upper_vlm_available, dtype=np.bool_
            ),
            "positive_memory_task_names": np.asarray(positive_retrieval["task_names"], dtype=np.str_),
            "negative_retrieval_weights": np.asarray(negative_retrieval["weights"], dtype=np.float32),
            "negative_retrieval_scores": np.asarray(negative_retrieval["scores"], dtype=np.float32),
            "negative_memory_indices": np.asarray(negative_retrieval["indices"], dtype=np.int64),
            "negative_memory_task_names": np.asarray(negative_retrieval["task_names"], dtype=np.str_),
        }
        if trace_steps and "negative_responsibilities" in trace_steps[0]:
            arrays["negative_responsibilities"] = torch.stack(
                [step["negative_responsibilities"] for step in trace_steps]
            ).numpy()
            arrays["negative_dist2"] = torch.stack([step["negative_dist2"] for step in trace_steps]).numpy()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=trace_dir, prefix=f".{stem}.", suffix=".tmp", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                np.savez_compressed(tmp, **arrays)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise

        record = {
            "schema_version": 2,
            "trace": final_path.name,
            **context,
            "progress": float(progress),
            "progress_source": progress_source,
            "num_steps": int(num_steps),
            "similarity_global": float(retrieval_debug.get("similarity_global", float("nan"))),
            "lambda_max": float(getattr(self, "memory_guidance_lambda_max", 0.2)),
            "t_cut": float(getattr(self, "memory_guidance_t_cut", 0.3)),
            "sigma": float(getattr(self, "memory_guidance_sigma", 0.3)),
            "norm_cap": float(getattr(self, "memory_guidance_norm_cap", 0.2)),
            # The NPZ is atomically published before this manifest record.
            # These fields let the offline indexer avoid reopening every large
            # trace on high-latency object-storage mounts.
            "full_trace_schema_complete": True,
            "task_embedding_dim": int(arrays["task_embedding"].size),
            "lower_feature_dim": int(arrays["lower_retrieval_feature"].size),
            "upper_feature_dim": int(arrays["upper_retrieval_feature"].size),
            "upper_vlm_available": bool(arrays["upper_vlm_available"]),
        }
        self._append_memory_guidance_manifest(trace_dir, record)

    def _trace_retrieval_summary(
        self,
        debug: dict,
        weights: torch.Tensor | None,
        *,
        positive: bool,
    ) -> dict:
        indices = [int(value) for value in debug.get("memory_indices", [])]
        if weights is None:
            weight_values = [float(value) for value in debug.get("weights", [])]
        else:
            weight_values = weights.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
        task_names = [str(value) for value in debug.get("memory_task_names", [])]
        if positive and len(task_names) < len(indices):
            memory = getattr(getattr(self, "memory_provider", None), "memory", [])
            task_names = [
                str(memory[index].get("task_name", "")) if 0 <= index < len(memory) else "" for index in indices
            ]
        return {
            "indices": indices,
            "scores": [float(value) for value in debug.get("scores", [])],
            "weights": [float(value) for value in weight_values],
            "task_names": task_names,
            "window_starts": [int(value) for value in debug.get("window_starts", [])],
            "action_alignment": str(debug.get("action_alignment", "unknown")),
            "retrieval_progress": float(debug.get("retrieval_progress", float("nan"))),
            "estimated_frames": [int(value) for value in debug.get("estimated_frames", [])],
            "estimated_frames_float": [
                float(value) for value in debug.get("estimated_frames_float", [])
            ],
            "reason": str(debug.get("reason", "ok" if indices else "not_selected")),
        }

    @staticmethod
    def _summarize_memory_guidance_step(
        *,
        v_base: torch.Tensor,
        guidance_positive: torch.Tensor,
        guidance_negative: torch.Tensor,
        guidance_final: torch.Tensor,
        time: float,
        step_index: int | None = None,
    ) -> dict:
        base = v_base.detach().to(dtype=torch.float32).flatten(1)

        def metrics(value: torch.Tensor) -> dict:
            flat = value.detach().to(dtype=torch.float32).flatten(1)
            norm = flat.norm(dim=1)
            base_norm = base.norm(dim=1)
            cosine = torch.nn.functional.cosine_similarity(flat, base, dim=1, eps=1e-12)
            cosine = torch.where((norm > 0) & (base_norm > 0), cosine, torch.zeros_like(cosine))
            return {
                "norm": norm.cpu().tolist(),
                "cosine_to_v_base": cosine.cpu().tolist(),
            }

        v_actual = v_base + guidance_final
        record = {
            "time": float(time),
            "v_base": metrics(v_base),
            "guidance_positive": metrics(guidance_positive),
            "guidance_negative": metrics(guidance_negative),
            "guidance_final": metrics(guidance_final),
            "v_actual": metrics(v_actual),
        }
        if step_index is not None:
            record["step_index"] = int(step_index)
        return record

    @staticmethod
    def _atomic_write_json_line(path: Path, record: dict) -> None:
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _append_memory_guidance_manifest(trace_dir: Path, record: dict) -> None:
        manifest_path = trace_dir / "manifest.jsonl"
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(manifest_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            written = os.write(fd, payload)
            if written != len(payload):
                raise OSError(f"Short append to {manifest_path}: {written}/{len(payload)} bytes")
            os.fsync(fd)
        finally:
            os.close(fd)

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
