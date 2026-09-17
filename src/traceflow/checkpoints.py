"""Download official model weights and convert VLA checkpoints on demand."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

PREDIMEM_REPO = "huashuolei/PrediMem"
PREDIMEM_REVISION = "e645741c2f34f27e1596bfa89e856d6f3560ed90"


def _cache_root() -> Path:
    explicit = os.environ.get("TRACEFLOW_CHECKPOINT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (xdg / "traceflow" / "official").resolve()


def _override(name: str, marker: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not (path / marker).is_file():
        raise FileNotFoundError(f"{name} does not contain {marker}: {path}")
    return path


def _hf_snapshot(patterns: list[str]) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to obtain official PrediMem weights") from error
    return Path(snapshot_download(PREDIMEM_REPO, revision=PREDIMEM_REVISION, allow_patterns=patterns)).resolve()


def _convert(root: Path, source: Path, output: Path, config: str) -> Path:
    marker = output / "model.safetensors"
    if marker.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    script = root / "openpi" / "examples" / "convert_jax_model_to_pytorch.py"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "openpi" / "src"), str(root / "openpi" / "packages" / "openpi-client" / "src"), os.environ.get("PYTHONPATH", "")))}
    subprocess.run(
        [sys.executable, str(script), "--checkpoint-dir", str(source), "--config-name", config,
         "--output-path", str(output), "--precision", "bfloat16"],
        check=True,
        env=env,
    )
    if not marker.is_file():
        raise RuntimeError(f"checkpoint conversion did not create {marker}")
    return output


def resolve_pi05(root: Path) -> Path:
    overridden = _override("PI05_CHECKPOINT", "model.safetensors")
    if overridden:
        return overridden
    output = _cache_root() / "pi05_libero_pytorch"
    if (output / "model.safetensors").is_file():
        return output
    sys.path.insert(0, str(root / "openpi" / "src"))
    try:
        from openpi.shared.download import maybe_download
    except ImportError as error:
        raise RuntimeError("OpenPI dependencies are required to obtain Pi0.5-LIBERO") from error
    os.environ.setdefault("OPENPI_DATA_HOME", str(_cache_root() / "openpi-downloads"))
    source = Path(maybe_download("gs://openpi-assets/checkpoints/pi05_libero", token="anon"))
    if not (source / "params" / "_METADATA").is_file():
        source = Path(
            maybe_download(
                "gs://openpi-assets/checkpoints/pi05_libero",
                force_download=True,
                token="anon",
            )
        )
    if not (source / "params" / "_METADATA").is_file():
        raise RuntimeError(f"official Pi0.5 checkpoint is incomplete: {source}")
    return _convert(root, source, output, "pi05_libero")


def resolve_predimem_upper() -> Path:
    overridden = _override("PREDIMEM_UPPER_CHECKPOINT", "model.safetensors")
    if overridden:
        return overridden
    snapshot = _hf_snapshot(["vlm_tasks1to26_ckpt74500/**"])
    result = snapshot / "vlm_tasks1to26_ckpt74500"
    if not (result / "model.safetensors").is_file():
        raise RuntimeError(f"official PrediMem Upper snapshot is incomplete: {result}")
    return result


def resolve_predimem_vla(root: Path) -> Path:
    overridden = _override("PREDIMEM_VLA_CHECKPOINT", "model.safetensors")
    if overridden:
        return overridden
    output = _cache_root() / "predimem" / "checkpoints" / "vla_alltask_pytorch"
    if not (output / "model.safetensors").is_file():
        snapshot = _hf_snapshot(["vla_alltask/**"])
        _convert(root, snapshot / "vla_alltask", output, "pi05_robomemarena_all26_reactive")
        norm_source = snapshot / "vla_alltask" / "assets" / "policy_assets" / "norm_stats.json"
        for asset_id in ("all26_pi05_reactive", "extra8_pi05_reactive"):
            destination = output / "assets" / "robomemarena" / asset_id / "norm_stats.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(norm_source, destination)
    return output
