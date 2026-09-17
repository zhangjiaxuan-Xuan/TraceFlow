from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
from pathlib import Path
import socket
import threading

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.task_head.local_consistency_memory import LocalConsistencyMemory
from openpi.task_head.dual_tower_head import DualTowerRetrievalHead
from openpi.task_head.memory_init import MemoryInitProvider
from openpi.task_head.memory_init import NegativeMemoryProvider
from openpi.task_head.task_head_mlp import TaskHeadMLP
from openpi.training import config as _config

_REPO_ROOT = Path(__file__).resolve().parents[1]


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.LIBERO
    default_prompt: str | None = None
    port: int = 8000
    inference_batch_size: int = 1
    inference_batch_wait_ms: float = 5.0
    inference_batch_group_size: int = 0
    separate_lower_probe_batches: bool = False
    seed: int = 7
    record: bool = False
    export_lower_retrieval_feature: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    use_memory: bool = True
    use_positive_memory: bool = True
    task_head_ckpt: str = str(_REPO_ROOT / "checkpoints" / "gpm_task_head.pt")
    memory_meta_path: str = str(_REPO_ROOT / "memory" / "gpm_memory_meta.pt")
    faiss_index_path: str = str(_REPO_ROOT / "memory" / "gpm_memory.index")
    memory_actions_path: str = str(_REPO_ROOT / "memory" / "gpm_memory_actions.npz")
    action_norm_stats_path: str = ""
    action_use_quantile_norm: bool = True

    align_mode: str = "hybrid"
    mixture_mode: str = "gaussian"
    temperature: float = 10.0
    sigma_min: float = 0.05
    noise_min: float = 0.20
    noise_max: float = 1.00
    nfe_min: int = 1
    nfe_max: int = 10
    nfe_floor: int = 1
    memory_top_k: int = 8
    memory_refresh_every: int = 1
    memory_refresh_sim_threshold: float = 0.0
    memory_progress_window: float = 0.20
    memory_action_alignment: str = "auto"
    memory_retrieval_backend: str = "auto"
    memory_allowed_task_ids: str = ""
    memory_exact_task_gate: bool = False
    allow_ae_only_retrieval_head_reuse: bool = False
    progress_mode: str = "client"
    replan_steps: int = 10
    debug_memory: bool = False

    use_memory_guidance: bool = False
    memory_guidance_only: bool = False
    memory_guidance_time_version: str = "v1"
    memory_prior_substep_guidance: bool = False
    memory_prior_guidance_version: str = "v1"
    memory_prior_guidance_final_scale: float = 0.01
    memory_prior_guidance_norm_cap: float = 0.50
    memory_joint_failure_prior: float = 0.50
    memory_guidance_v2_magnitude_cap: float = 0.10
    memory_guidance_v05_dynamic_nfe: bool = False
    memory_guidance_v05_fine_ratio: float = 1.20
    memory_guidance_v05_fine_scale: float = 0.20
    memory_guidance_v05_fine_confirm_steps: int = 2
    memory_guidance_v05_dynamic_guidance_scale: float = 0.50
    memory_guidance_num_steps: int = 10
    memory_guidance_lambda_max: float = 0.20
    memory_guidance_t_cut: float = 0.30
    memory_guidance_sigma: float = 0.30
    memory_guidance_norm_cap: float = 0.20
    memory_guidance_suite_gate_path: str = ""
    memory_guidance_min_similarity: float = -1.0
    memory_guidance_trace_dir: str = ""
    memory_guidance_trace_level: str = "full"
    memory_guidance_trace_workers: int = 8
    memory_static_action_mode: str = "off"
    memory_static_action_arm_dim: int = 6
    memory_static_action_threshold: float = 1e-8
    memory_static_action_gate_fraction: float = 0.50

    use_negative_guidance: bool = False
    negative_memory_meta_path: str = str(_REPO_ROOT / "memory" / "negative" / "gpm_negative_memory_meta.pt")
    negative_faiss_index_path: str = str(_REPO_ROOT / "memory" / "negative" / "gpm_negative_memory.index")
    negative_memory_actions_path: str = str(_REPO_ROOT / "memory" / "negative" / "gpm_negative_memory_actions.npz")
    negative_memory_top_k: int = 4
    negative_memory_min_similarity: float = 0.975
    negative_memory_min_confidence: float = 0.75
    negative_guidance_beta: float = 0.10
    negative_guidance_sigma: float = 0.30
    negative_guidance_norm_cap: float = 0.10
    memory_guidance_total_norm_cap: float = 0.20
    guidance_adapter_path: str = ""

    use_lcm: bool = False
    lcm_ckpt: str = str(_REPO_ROOT / "checkpoints" / "lcm.pt")
    lcm_hidden: int = 256
    lcm_layers: int = 1
    lcm_heads: int = 4
    lcm_dropout: float = 0.0
    lcm_mamba_impl: str = "auto"
    lcm_mamba_state: int = 16
    lcm_mamba_conv: int = 4
    lcm_mamba_expand: int = 2
    lcm_scale: float = 0.10
    lcm_debug: bool = False


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    checkpoint = DEFAULT_CHECKPOINT.get(env)
    if checkpoint is None:
        raise ValueError(f"Unsupported environment mode: {env}")
    return _policy_config.create_trained_policy(
        _config.get_config(checkpoint.config),
        checkpoint.dir,
        default_prompt=default_prompt,
    )


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _require_file(name: str, path: str) -> None:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_guidance_adapter(model, path_value: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    weights_path = path / "guidance_adapter.safetensors" if path.is_dir() else path
    metadata_path = weights_path.with_name("guidance_adapter.json")
    _require_file("guidance_adapter_path", str(weights_path))
    _require_file("guidance_adapter_metadata", str(metadata_path))
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != "openpi_guidance_residual_adapter_v1":
        raise ValueError(f"Unsupported guidance adapter schema: {metadata.get('schema')}")
    model.configure_guidance_adapter(int(metadata["hidden_dim"]))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True) if weights_path.suffix == ".pt" else None
    if state_dict is None:
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
    model.guidance_residual_adapter.load_state_dict(state_dict, strict=True)
    device = next(model.parameters()).device
    model.guidance_residual_adapter.to(device=device).eval()
    logging.info("Loaded guidance-aware residual adapter: %s", weights_path)


def _load_task_head(path: str, device: str) -> TaskHeadMLP | DualTowerRetrievalHead:
    ckpt = _torch_load_cpu(path)
    if ckpt.get("head_type") == "dual_tower":
        head = DualTowerRetrievalHead(
            variant=str(ckpt["variant"]),
            lower_dim=int(ckpt["lower_dim"]),
            upper_dim=int(ckpt["upper_dim"]),
            hidden=int(ckpt["hidden"]),
            out_dim=int(ckpt["out_dim"]),
        )
    else:
        head = TaskHeadMLP(
            in_dim=int(ckpt["in_dim"]),
            hidden=int(ckpt["hidden"]),
            out_dim=int(ckpt["out_dim"]),
        )
    state_dict = ckpt.get("state_dict") or ckpt.get("task_head")
    if state_dict is None:
        raise KeyError(f"Cannot find task head weights in {path}")
    head.load_state_dict(state_dict, strict=True)
    head.retrieval_provenance = {
        "lower_producer_policy_dir": ckpt.get("lower_producer_policy_dir"),
        "lower_producer_config_name": ckpt.get("lower_producer_config_name"),
        "upper_producer_checkpoint": ckpt.get("upper_producer_checkpoint"),
        "feature_manifest_sha256": ckpt.get("feature_manifest_sha256"),
        "joint_conditioned": ckpt.get("joint_conditioned"),
        "lower_conditioning_protocol": ckpt.get("lower_conditioning_protocol"),
        "upper_feature_protocol": ckpt.get("upper_feature_protocol"),
        "subtask_records_sha256": ckpt.get("subtask_records_sha256"),
        "lower_feature_protocol": ckpt.get("lower_feature_protocol"),
        "temporal_window": int(ckpt.get("temporal_window", 1)),
        "temporal_offsets": list(ckpt.get("temporal_offsets", [0])),
    }
    head.temporal_window = int(ckpt.get("temporal_window", 1))
    head.temporal_offsets = tuple(int(value) for value in ckpt.get("temporal_offsets", [0]))
    if len(head.temporal_offsets) != head.temporal_window or head.temporal_offsets[-1] != 0:
        raise ValueError("Invalid temporal retrieval metadata in task-head checkpoint")
    return head.eval().float().to(device)


def _clear_runtime_state(model) -> None:
    model.memory_session = None
    model._sample_call_count = 0
    model._external_progress = 0.0
    model._trace_context = {}
    model._batch_runtime = None
    model._external_upper_vlm_feature = None
    model._external_upper_vlm_age = 1.0
    model._external_upper_vlm_available = False
    model._batched_memory_states = {}
    model._memory_feature_history = {}
    model._memory_inference_serial = 0
    model._record_component_timing = os.environ.get("PREDIMEM_COMPONENT_TIMING", "0") == "1"
    model._last_component_timing = None
    model._last_lower_retrieval_features = None
    for attr in ("_last_action_chunk", "_lcm_prev_chunk", "_lcm_h"):
        if hasattr(model, attr):
            setattr(model, attr, None)


def _attach_runtime_helpers(model, args: Args) -> None:
    def set_progress(progress: float) -> None:
        model._external_progress = float(max(0.0, min(1.0, progress)))

    def reset_memory_session() -> None:
        _clear_runtime_state(model)

    model.set_progress = set_progress
    model.reset_memory_session = reset_memory_session
    model.progress_mode = args.progress_mode
    model.replan_steps_hint = int(args.replan_steps)
    model.eval_seed = int(args.seed)


def _inject_gpm_lcm(model, args: Args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prior_guidance_version = str(args.memory_prior_guidance_version).lower()
    guidance_time_version = str(args.memory_guidance_time_version).lower()
    if prior_guidance_version not in (
        "v1",
        "v2",
        "v3_prior_only",
        "v3_prior_decay",
        "v3_1_prior_decay",
        "v3_prior_decay_joint",
        "v3_re_prior_decay",
    ):
        raise ValueError(
            "memory_prior_guidance_version must be v1, v2, v3_prior_only, "
            "v3_prior_decay, v3_1_prior_decay, v3_prior_decay_joint, or v3_re_prior_decay"
        )
    if guidance_time_version not in ("v0", "v0_5", "v1", "v2"):
        raise ValueError("memory_guidance_time_version must be v0, v0_5, v1, or v2")
    if float(args.memory_guidance_v2_magnitude_cap) < 0.0:
        raise ValueError("memory_guidance_v2_magnitude_cap must be non-negative")
    if not 0.0 <= float(args.memory_prior_guidance_final_scale) <= float(args.memory_guidance_lambda_max):
        raise ValueError("memory_prior_guidance_final_scale must be in [0, memory_guidance_lambda_max]")
    if float(args.memory_prior_guidance_norm_cap) < 0.0:
        raise ValueError("memory_prior_guidance_norm_cap must be non-negative")
    if not 0.0 < float(args.memory_joint_failure_prior) < 1.0:
        raise ValueError("memory_joint_failure_prior must be strictly between 0 and 1")
    if prior_guidance_version.startswith("v3_") and float(args.memory_guidance_v2_magnitude_cap) != 0.0:
        raise ValueError("v3 prior-anchor guidance requires memory_guidance_v2_magnitude_cap=0")
    if prior_guidance_version == "v3_prior_decay_joint" and not args.use_negative_guidance:
        raise ValueError("v3_prior_decay_joint requires use_negative_guidance")
    if float(args.memory_guidance_v05_fine_ratio) <= 0.0:
        raise ValueError("memory_guidance_v05_fine_ratio must be positive")
    if not 0.0 <= float(args.memory_guidance_v05_fine_scale) <= 1.0:
        raise ValueError("memory_guidance_v05_fine_scale must be in [0, 1]")
    if int(args.memory_guidance_v05_fine_confirm_steps) <= 0:
        raise ValueError("memory_guidance_v05_fine_confirm_steps must be positive")
    if float(args.memory_guidance_v05_dynamic_guidance_scale) < 0.0:
        raise ValueError("memory_guidance_v05_dynamic_guidance_scale must be non-negative")
    trace_level = str(args.memory_guidance_trace_level).lower()
    if trace_level not in ("full", "light", "bank"):
        raise ValueError("memory_guidance_trace_level must be full, light, or bank")
    if int(args.memory_guidance_trace_workers) < 1:
        raise ValueError("memory_guidance_trace_workers must be positive")
    static_action_mode = str(args.memory_static_action_mode).lower()
    if static_action_mode not in ("off", "observe", "gate"):
        raise ValueError("memory_static_action_mode must be off, observe, or gate")
    if int(args.memory_static_action_arm_dim) <= 0:
        raise ValueError("memory_static_action_arm_dim must be positive")
    if float(args.memory_static_action_threshold) < 0.0:
        raise ValueError("memory_static_action_threshold must be non-negative")
    if not 0.0 <= float(args.memory_static_action_gate_fraction) <= 1.0:
        raise ValueError("memory_static_action_gate_fraction must be in [0, 1]")
    if static_action_mode != "off" and not args.memory_guidance_only:
        raise ValueError("Static-action diagnostics currently require memory_guidance_only")
    if args.memory_prior_substep_guidance and (args.use_memory_guidance or args.memory_guidance_only):
        raise ValueError("memory_prior_substep_guidance is an independent guidance mode")
    if args.memory_prior_substep_guidance and not args.use_memory:
        raise ValueError("memory_prior_substep_guidance requires use_memory")
    if args.use_negative_guidance and not (
        args.use_memory_guidance or args.memory_guidance_only or args.memory_prior_substep_guidance
    ):
        raise ValueError("Negative guidance requires positive memory guidance to be enabled")
    if not args.use_positive_memory and not args.use_negative_guidance:
        raise ValueError("Disabling positive memory requires negative guidance")
    if not args.use_positive_memory and args.memory_prior_substep_guidance:
        raise ValueError("Prior-substep guidance requires positive memory")
    model.use_memory = bool(args.use_memory)
    model._export_lower_retrieval_feature = bool(args.export_lower_retrieval_feature)
    _attach_runtime_helpers(model, args)
    _clear_runtime_state(model)

    if not args.use_memory:
        if args.export_lower_retrieval_feature:
            _require_file("task_head_ckpt", args.task_head_ckpt)
            model.task_head = _load_task_head(args.task_head_ckpt, device)
            if str(getattr(model.task_head, "variant", "")) != "fusion":
                raise ValueError("Lower feature export requires a fusion task-head checkpoint")
        else:
            model.task_head = None
        model.memory_provider = None
        model.lcm = None
        model.use_lcm = False
        model.negative_memory_provider = None
        model.use_negative_guidance = False
        logging.info(
            "GPM is disabled; serving the base policy%s.",
            " with lower-feature export" if args.export_lower_retrieval_feature else "",
        )
        return

    _require_file("task_head_ckpt", args.task_head_ckpt)
    if args.use_positive_memory:
        _require_file("memory_meta_path", args.memory_meta_path)
        _require_file("faiss_index_path", args.faiss_index_path)
        _require_file("memory_actions_path", args.memory_actions_path)
    if args.action_norm_stats_path:
        _require_file("action_norm_stats_path", args.action_norm_stats_path)

    model.task_head = _load_task_head(args.task_head_ckpt, device)
    provenance = getattr(model.task_head, "retrieval_provenance", {})
    head_variant = str(getattr(model.task_head, "variant", "lower"))
    lower_producer = provenance.get("lower_producer_policy_dir")
    if head_variant in ("lower", "fusion"):
        if lower_producer and isinstance(args.policy, Checkpoint):
            expected = Path(lower_producer).resolve()
            actual = Path(args.policy.dir).resolve()
            if actual != expected:
                if not args.allow_ae_only_retrieval_head_reuse:
                    raise ValueError(
                        "Retrieval head/model identity mismatch: "
                        f"head was trained from {expected}, policy is {actual}. "
                        "For a verified AE-only fine-tune with an unchanged VLM, explicitly pass "
                        "--allow-ae-only-retrieval-head-reuse."
                    )
                logging.warning(
                    "Explicitly reusing retrieval head across AE-only policy paths: producer=%s consumer=%s",
                    expected,
                    actual,
                )
        elif isinstance(model.task_head, DualTowerRetrievalHead):
            logging.warning(
                "Dual-tower retrieval checkpoint has no lower-model provenance; "
                "treat this as a legacy ablation, not a valid self-model evaluation."
            )
    model.memory_provider = (
        MemoryInitProvider(
            memory_meta_path=args.memory_meta_path,
            faiss_index_path=args.faiss_index_path,
            memory_actions_path=args.memory_actions_path,
            align_mode=args.align_mode,
            mixture_mode=args.mixture_mode,
            temperature=args.temperature,
            sigma_min=args.sigma_min,
            noise_scale_range=(args.noise_min, args.noise_max),
            nfe_range=(args.nfe_min, args.nfe_max),
            nfe_floor=args.nfe_floor,
            device=device,
            action_norm_stats_path=args.action_norm_stats_path or None,
            action_use_quantile_norm=args.action_use_quantile_norm,
            progress_window=args.memory_progress_window,
            action_alignment=args.memory_action_alignment,
            retrieval_backend=args.memory_retrieval_backend,
            allowed_task_ids=(
                {int(value) for value in args.memory_allowed_task_ids.split(",") if value.strip()}
                if args.memory_allowed_task_ids
                else None
            ),
        )
        if args.use_positive_memory
        else None
    )
    model.memory_top_k = int(args.memory_top_k)
    model.memory_exact_task_gate = bool(args.memory_exact_task_gate)
    model.memory_refresh_every = int(args.memory_refresh_every)
    model.memory_refresh_sim_threshold = float(args.memory_refresh_sim_threshold)
    model.debug_memory = bool(args.debug_memory)
    model.use_memory_guidance = bool(
        args.use_memory_guidance or args.memory_guidance_only or args.memory_prior_substep_guidance
    )
    model.memory_guidance_only = bool(args.memory_guidance_only)
    model.memory_guidance_time_version = guidance_time_version
    model.memory_prior_substep_guidance = bool(args.memory_prior_substep_guidance)
    model.memory_prior_guidance_version = prior_guidance_version
    model.memory_prior_guidance_final_scale = float(args.memory_prior_guidance_final_scale)
    model.memory_prior_guidance_norm_cap = float(args.memory_prior_guidance_norm_cap)
    model.memory_joint_failure_prior = float(args.memory_joint_failure_prior)
    model.memory_guidance_v2_magnitude_cap = float(args.memory_guidance_v2_magnitude_cap)
    model.memory_guidance_v05_dynamic_nfe = bool(args.memory_guidance_v05_dynamic_nfe)
    model.memory_guidance_v05_fine_ratio = float(args.memory_guidance_v05_fine_ratio)
    model.memory_guidance_v05_fine_scale = float(args.memory_guidance_v05_fine_scale)
    model.memory_guidance_v05_fine_confirm_steps = int(args.memory_guidance_v05_fine_confirm_steps)
    model.memory_guidance_v05_dynamic_guidance_scale = float(args.memory_guidance_v05_dynamic_guidance_scale)
    model.memory_nfe_max = int(args.nfe_max)
    model.memory_guidance_num_steps = int(args.memory_guidance_num_steps)
    model.memory_guidance_lambda_max = float(args.memory_guidance_lambda_max)
    model.memory_guidance_t_cut = float(args.memory_guidance_t_cut)
    model.memory_guidance_sigma = float(args.memory_guidance_sigma)
    model.memory_guidance_norm_cap = float(args.memory_guidance_norm_cap)
    model.memory_guidance_suite_gate = None
    if args.memory_guidance_suite_gate_path:
        _require_file("memory_guidance_suite_gate_path", args.memory_guidance_suite_gate_path)
        gate_path = Path(args.memory_guidance_suite_gate_path)
        gate_payload = json.loads(gate_path.read_text())
        if gate_payload.get("schema") != "arena_suite_guidance_gate_v1":
            raise ValueError(f"Unsupported guidance gate schema: {gate_payload.get('schema')}")
        profiles = gate_payload.get("task_profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("Guidance gate requires non-empty task_profiles")
        model.memory_guidance_suite_gate = profiles
    model.memory_guidance_min_similarity = float(args.memory_guidance_min_similarity)
    model.memory_guidance_trace_dir = str(args.memory_guidance_trace_dir)
    model.memory_guidance_trace_level = trace_level
    model.memory_guidance_trace_workers = int(args.memory_guidance_trace_workers)
    model.memory_static_action_mode = static_action_mode
    model.memory_static_action_arm_dim = int(args.memory_static_action_arm_dim)
    model.memory_static_action_threshold = float(args.memory_static_action_threshold)
    model.memory_static_action_gate_fraction = float(args.memory_static_action_gate_fraction)
    model.use_negative_guidance = bool(args.use_negative_guidance)
    model.negative_memory_top_k = int(args.negative_memory_top_k)
    model.negative_memory_min_similarity = float(args.negative_memory_min_similarity)
    model.negative_memory_min_confidence = float(args.negative_memory_min_confidence)
    model.negative_guidance_beta = float(args.negative_guidance_beta)
    model.negative_guidance_sigma = float(args.negative_guidance_sigma)
    model.negative_guidance_norm_cap = float(args.negative_guidance_norm_cap)
    model.memory_guidance_total_norm_cap = float(args.memory_guidance_total_norm_cap)
    if args.use_negative_guidance:
        _require_file("negative_memory_meta_path", args.negative_memory_meta_path)
        _require_file("negative_faiss_index_path", args.negative_faiss_index_path)
        _require_file("negative_memory_actions_path", args.negative_memory_actions_path)
        model.negative_memory_provider = NegativeMemoryProvider(
            memory_meta_path=args.negative_memory_meta_path,
            faiss_index_path=args.negative_faiss_index_path,
            memory_actions_path=args.negative_memory_actions_path,
            action_norm_stats_path=args.action_norm_stats_path or None,
            action_use_quantile_norm=args.action_use_quantile_norm,
            temperature=args.temperature,
            device=device,
        )
    else:
        model.negative_memory_provider = None
    logging.info(
        "Loaded GPM: task_head=%s memory_items=%d retrieval_items=%d allowed_tasks=%s action_alignment=%s "
        "top_k=%d refresh_every=%d guidance=%s "
        "guidance_only=%s guidance_version=%s guidance_runtime=%s v05_dynamic_nfe=%s "
        "v05_fine_ratio=%.3f v05_fine_scale=%.3f v05_confirm=%d "
        "v05_dynamic_scale=%.3f v05_dynamic_algorithm=%s prior_substep=%s "
        "prior_version=%s prior_runtime=%s prior_final_scale=%.3f "
        "nfe_floor=%d floor_noise_scale=%.6f joint_failure_prior=%.3f negative=%s "
        "static_action=%s threshold=%.3g gate_fraction=%.3f",
        args.task_head_ckpt,
        int(model.memory_provider.num_items) if model.memory_provider is not None else 0,
        int(len(model.memory_provider._retrieval_indices)) if model.memory_provider is not None else 0,
        (
            sorted(model.memory_provider.allowed_task_ids)
            if model.memory_provider is not None and model.memory_provider.allowed_task_ids is not None
            else "all"
        ),
        model.memory_provider.action_alignment if model.memory_provider is not None else "disabled",
        model.memory_top_k,
        model.memory_refresh_every,
        bool(model.use_memory_guidance),
        bool(model.memory_guidance_only),
        model.memory_guidance_time_version,
        {
            "v0": "full_interval_direct_unbounded",
            "v0_5": "velocity_gated_raw_guidance",
            "v1": "full_interval_additive_capped",
            "v2": "full_interval_direction_first",
        }[model.memory_guidance_time_version],
        model.memory_guidance_v05_dynamic_nfe,
        model.memory_guidance_v05_fine_ratio,
        model.memory_guidance_v05_fine_scale,
        model.memory_guidance_v05_fine_confirm_steps,
        model.memory_guidance_v05_dynamic_guidance_scale,
        "arc_length_v2" if model.memory_guidance_v05_dynamic_nfe else "disabled",
        bool(model.memory_prior_substep_guidance),
        model.memory_prior_guidance_version,
        {
            "v1": "v1_reference",
            "v2": "full_interval_direction_first",
            "v3_prior_only": "clean_prior_correction_then_base_flow",
            "v3_prior_decay": "clean_prior_correction_then_annealed_direction_first",
            "v3_1_prior_decay": "clean_prior_then_capped_annealed_direction_first",
            "v3_prior_decay_joint": "clean_prior_joint_posterior_then_annealed_direction_first",
            "v3_re_prior_decay": "v3_prior_decay_with_compact_active_set",
        }[model.memory_prior_guidance_version],
        model.memory_prior_guidance_final_scale,
        int(model.memory_provider.nfe_floor) if model.memory_provider is not None else int(args.nfe_floor),
        (
            float(model.memory_provider._lambda_from_similarity(1.0))
            if model.memory_provider is not None
            else 0.0
        ),
        model.memory_joint_failure_prior,
        bool(model.use_negative_guidance),
        model.memory_static_action_mode,
        model.memory_static_action_threshold,
        model.memory_static_action_gate_fraction,
    )

    guidance_without_lcm = args.memory_guidance_only or args.memory_prior_substep_guidance
    if args.use_lcm and guidance_without_lcm:
        logging.info("LCM disabled because the selected guidance mode excludes LCM.")

    if args.use_lcm and not guidance_without_lcm:
        _require_file("lcm_ckpt", args.lcm_ckpt)
        lcm_ckpt = _torch_load_cpu(args.lcm_ckpt)
        meta = lcm_ckpt.get("meta", {})
        horizon = int(model.config.action_horizon)
        action_dim = int(model.config.action_dim)
        if int(meta.get("H", horizon)) != horizon or int(meta.get("A", action_dim)) != action_dim:
            raise RuntimeError(
                f"LCM checkpoint shape mismatch: checkpoint H/A={meta.get('H')}/{meta.get('A')} "
                f"model H/A={horizon}/{action_dim}"
            )
        lcm = LocalConsistencyMemory(
            action_dim=action_dim,
            horizon=horizon,
            hidden=int(meta.get("hidden", args.lcm_hidden)),
            n_layers=int(meta.get("layers", args.lcm_layers)),
            n_heads=int(meta.get("heads", args.lcm_heads)),
            dropout=float(meta.get("dropout", args.lcm_dropout)),
            use_tanh=bool(meta.get("use_tanh", False)),
            mamba_impl=str(meta.get("mamba_impl", args.lcm_mamba_impl)),
            mamba_state=int(meta.get("mamba_state", args.lcm_mamba_state)),
            mamba_conv=int(meta.get("mamba_conv", args.lcm_mamba_conv)),
            mamba_expand=int(meta.get("mamba_expand", args.lcm_mamba_expand)),
        ).to(device).eval()
        key = "lcm" if "lcm" in lcm_ckpt else "state_dict"
        lcm.load_state_dict(lcm_ckpt[key], strict=True)
        model.lcm = lcm
        model.use_lcm = True
        model.lcm_scale = float(args.lcm_scale)
        model.debug_lcm = bool(args.lcm_debug)
        logging.info("Loaded LCM: ckpt=%s scale=%.3f", args.lcm_ckpt, model.lcm_scale)
    else:
        model.lcm = None
        model.use_lcm = False
        model.lcm_scale = 1.0
        model.debug_lcm = False


class ProgressAwarePolicy:
    """Submit per-episode progress to the underlying PyTorch model before inference."""

    def __init__(self, base_policy: _policy.Policy):
        self._base = base_policy
        trace_path = os.environ.get("PREDIMEM_NFE_STATS_PATH", "").strip()
        self._nfe_stats_path = Path(trace_path) if trace_path else None
        self._base._model._record_active_nfe_timing = self._nfe_stats_path is not None
        self._nfe_stats_lock = threading.Lock()
        self._nfe_batch_index = 0
        static_path = os.environ.get("MEMORY_STATIC_ACTION_STATS_PATH", "").strip()
        self._static_action_stats_path = Path(static_path) if static_path else None
        self._static_action_stats_lock = threading.Lock()
        self._static_action_batch_index = 0

    def _write_nfe_stats(self, stats: dict, results: list[dict]) -> None:
        if self._nfe_stats_path is None:
            return
        record = dict(stats)
        self._nfe_batch_index += 1
        record["batch_index"] = self._nfe_batch_index
        record["batch_size"] = len(results)
        record["infer_ms"] = (
            float(results[0].get("policy_timing", {}).get("infer_ms", float("nan")))
            if results
            else float("nan")
        )
        self._nfe_stats_path.parent.mkdir(parents=True, exist_ok=True)
        with self._nfe_stats_lock:
            with self._nfe_stats_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()

    def _write_static_action_stats(self, stats: dict, results: list[dict]) -> None:
        if self._static_action_stats_path is None:
            return
        record = dict(stats)
        self._static_action_batch_index += 1
        record["batch_index"] = self._static_action_batch_index
        record["batch_size"] = len(results)
        record["infer_ms"] = (
            float(results[0].get("policy_timing", {}).get("infer_ms", float("nan")))
            if results
            else float("nan")
        )
        self._static_action_stats_path.parent.mkdir(parents=True, exist_ok=True)
        with self._static_action_stats_lock:
            with self._static_action_stats_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()

    @property
    def _model(self):
        return self._base._model

    @property
    def metadata(self):
        return self._base.metadata

    def infer(self, element: dict):
        trace_context = dict(element.pop("trace_context", None) or {})
        trace_context.setdefault("task_name", str(element.get("prompt", "")))
        reset_flag = element.pop("episode_reset", None)
        if reset_flag and hasattr(self._model, "reset_memory_session"):
            episode_seed = (
                int(getattr(self._model, "eval_seed", 7))
                + 1000 * int(trace_context.get("task_id", 0))
                + int(trace_context.get("episode_idx", 0))
            )
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
            self._model.reset_memory_session()
        self._model._trace_context = trace_context

        progress = element.pop("progress", None)
        upper_feature = element.pop("upper_vlm_feature", None)
        model_device = next(self._model.parameters()).device
        self._model._external_upper_vlm_feature = (
            None
            if upper_feature is None
            else torch.as_tensor(upper_feature, dtype=torch.float32, device=model_device)
        )
        self._model._external_upper_vlm_age = float(element.pop("upper_vlm_age", 1.0))
        self._model._external_upper_vlm_available = bool(element.pop("upper_vlm_available", False))
        if getattr(self._model, "progress_mode", "client") == "client" and progress is not None:
            if hasattr(self._model, "set_progress"):
                self._model.set_progress(float(progress))

        return self._base.infer(element)

    def infer_batch(self, elements: list[dict]):
        clean = []
        runtime = []
        probe_flags = []
        for slot, source in enumerate(elements):
            element = dict(source)
            probe_flags.append(bool(element.pop("_lower_probe_only", False)))
            context = dict(element.pop("trace_context", None) or {})
            context.setdefault("task_name", str(element.get("prompt", "")))
            context.setdefault("policy_call_idx", int(context.get("inference_call", 0)))
            env_id = str(context.get("environment_id", f"slot-{slot}"))
            reset = bool(element.pop("episode_reset", False))
            progress = float(element.pop("progress", 0.0))
            upper_feature = element.pop("upper_vlm_feature", None)
            upper_age = float(element.pop("upper_vlm_age", 1.0))
            upper_available = bool(element.pop("upper_vlm_available", upper_feature is not None))
            seed = (
                int(getattr(self._model, "eval_seed", 7))
                + 1000 * int(context.get("task_id", 0))
                + int(context.get("episode_idx", 0))
            )
            runtime.append(
                {
                    "environment_id": env_id,
                    "reset": reset,
                    "progress": progress,
                    "seed": seed,
                    "context": context,
                    "upper_vlm_feature": upper_feature,
                    "upper_vlm_age": upper_age,
                    "upper_vlm_available": upper_available,
                }
            )
            clean.append(element)
        self._model._batch_runtime = runtime
        self._model._last_adaptive_nfe_batch = None
        self._model._last_static_action_batch = None
        try:
            if any(probe_flags):
                if not all(probe_flags):
                    raise RuntimeError("Lower probe batches cannot mix probe and action requests")
                return self._base.probe_lower_retrieval_batch(clean)
            results = self._base.infer_batch(clean)
            stats = getattr(self._model, "_last_adaptive_nfe_batch", None)
            if isinstance(stats, dict):
                self._write_nfe_stats(stats, results)
            static_stats = getattr(self._model, "_last_static_action_batch", None)
            if isinstance(static_stats, dict):
                self._write_static_action_stats(static_stats, results)
            return results
        finally:
            self._model._batch_runtime = None


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy = create_policy(args)
    policy_metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    _load_guidance_adapter(policy._model, args.guidance_adapter_path)
    _inject_gpm_lcm(policy._model, args)
    policy = ProgressAwarePolicy(policy)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server on %s (%s), port %d", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        batch_size=args.inference_batch_size,
        batch_wait_ms=args.inference_batch_wait_ms,
        batch_group_size=args.inference_batch_group_size,
        separate_lower_probe_batches=args.separate_lower_probe_batches,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
