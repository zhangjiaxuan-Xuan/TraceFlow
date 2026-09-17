from __future__ import annotations

import json
from pathlib import Path

import faiss
import h5py
import numpy as np
import pytest
import torch

from openpi.task_head.memory_init import MemoryInitProvider
from openpi.task_head.memory_init import NegativeMemoryProvider
from openpi.task_head.task_head_mlp import TaskHeadMLP
from openpi.task_head.reproduction import manifest_digest
from scripts.build_cl_memory_bank import Args
from scripts.build_cl_memory_bank import main


def _write_baseline(root: Path, label: str, embedding: np.ndarray) -> tuple[str, str, str]:
    root.mkdir(parents=True)
    action_id = f"m0_{label}"
    metadata = [
        {
            "label": label,
            "task_name": f"baseline {label}",
            "task_id": -1,
            "task_emb": torch.from_numpy(embedding.copy()),
            "action_id": action_id,
            "failure_confidence": 1.0 if label == "negative" else 0.0,
            "chunk_meta": {"chunk_len": 10, "stride": 10, "num_chunks": 1, "T": 10},
        }
    ]
    meta_path = root / "meta.pt"
    index_path = root / "index.faiss"
    actions_path = root / "actions.npz"
    torch.save(metadata, meta_path)
    index = faiss.IndexFlatIP(embedding.shape[0])
    index.add(embedding.reshape(1, -1))
    faiss.write_index(index, str(index_path))
    np.savez_compressed(
        actions_path,
        actions=np.zeros((10, 32), dtype=np.float32),
        offsets=np.asarray([0, 10], dtype=np.int64),
        ids=np.asarray([action_id]),
    )
    return str(meta_path), str(index_path), str(actions_path)


def _artifacts(tmp_path: Path) -> tuple[dict[str, str], list[dict]]:
    eval_path = tmp_path / "success.hdf5"
    with h5py.File(eval_path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.arange(84, dtype=np.float32).reshape(12, 7))
    smol_path = tmp_path / "failure.npz"
    np.savez_compressed(smol_path, actions=-np.ones((11, 7), dtype=np.float32))
    rows = [
        {
            "action_id": "pi_action_3",
            "trajectory_path": str(eval_path),
            "source_format": "eval_hdf5",
            "success": True,
            "prompt": "put object away",
            "task_id": 2,
            "episode_idx": 3,
            "source_run": "eval-a",
        },
        {
            "action_id": "smol_action_5",
            "trajectory_path": str(smol_path),
            "source_format": "smol_npz",
            "success": False,
            "prompt": "open drawer",
            "task_id": 4,
            "episode_idx": 5,
            "source_run": "smol-b",
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "pooled_prefix.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(feature_dir / "completed.npy", np.ones(2, dtype=np.bool_))
    (feature_dir / "cache_state.json").write_text(
        json.dumps({"manifest_sha256": manifest_digest(manifest), "rows": 2}), encoding="utf-8"
    )

    head = TaskHeadMLP(in_dim=2, hidden=2, out_dim=2)
    with torch.no_grad():
        head.net[0].weight.copy_(torch.eye(2))
        head.net[0].bias.zero_()
        head.net[2].weight.copy_(torch.eye(2))
        head.net[2].bias.zero_()
    checkpoint = tmp_path / "head.pt"
    torch.save({"state_dict": head.state_dict(), "in_dim": 2, "hidden": 2, "out_dim": 2}, checkpoint)

    pos = _write_baseline(tmp_path / "m0-positive", "positive", np.asarray([1.0, 0.0], dtype=np.float32))
    neg = _write_baseline(tmp_path / "m0-negative", "negative", np.asarray([0.0, 1.0], dtype=np.float32))
    paths = {
        "manifest": str(manifest),
        "feature_dir": str(feature_dir),
        "checkpoint": str(checkpoint),
        "baseline_positive_meta": pos[0],
        "baseline_positive_index": pos[1],
        "baseline_positive_actions": pos[2],
        "baseline_negative_meta": neg[0],
        "baseline_negative_index": neg[1],
        "baseline_negative_actions": neg[2],
    }
    return paths, rows


@pytest.mark.parametrize(
    ("admission", "positive_count", "negative_count"),
    [("success", 2, 1), ("failure", 1, 2), ("both", 2, 2)],
)
def test_build_cl_bank_admission_and_provider_loading(
    tmp_path: Path, admission: str, positive_count: int, negative_count: int
) -> None:
    paths, rows = _artifacts(tmp_path)
    output = tmp_path / f"out-{admission}"
    main(Args(group="BN", **paths, output_dir=str(output), admission=admission, device="cpu"))  # type: ignore[arg-type]

    positive_meta = torch.load(output / "positive/gpm_memory_meta.pt", weights_only=False)
    negative_meta = torch.load(output / "negative/gpm_negative_memory_meta.pt", weights_only=False)
    assert len(positive_meta) == positive_count
    assert len(negative_meta) == negative_count
    assert len({entry["action_id"] for entry in positive_meta + negative_meta}) == positive_count + negative_count
    if admission in {"success", "both"}:
        assert positive_meta[-1]["provenance"] == rows[0]
        with np.load(output / "positive/gpm_memory_actions.npz") as packed:
            assert packed["actions"].shape == (22, 32)
            np.testing.assert_array_equal(packed["actions"][-12:, 7:], 0.0)
    if admission in {"failure", "both"}:
        assert negative_meta[-1]["failure_confidence"] == 1.0
        assert negative_meta[-1]["provenance"] == rows[1]

    positive_provider = MemoryInitProvider(
        str(output / "positive/gpm_memory_meta.pt"),
        str(output / "positive/gpm_memory.index"),
        memory_actions_path=str(output / "positive/gpm_memory_actions.npz"),
        device="cpu",
    )
    sample, _, _ = positive_provider.query(torch.tensor([1.0, 0.0]), k=1, action_horizon=10)
    assert sample.shape == (10, 32)
    negative_provider = NegativeMemoryProvider(
        memory_meta_path=str(output / "negative/gpm_negative_memory_meta.pt"),
        faiss_index_path=str(output / "negative/gpm_negative_memory.index"),
        memory_actions_path=str(output / "negative/gpm_negative_memory_actions.npz"),
        action_norm_stats_path=None,
        device="cpu",
    )
    blocks, weights, debug = negative_provider.query_blocks(
        torch.tensor([0.0, 1.0]),
        k=1,
        horizon=10,
        progress=0.0,
        min_similarity=-1.0,
        min_confidence=0.75,
    )
    assert blocks is not None
    assert blocks.shape == (1, 10, 32)
    assert weights is not None
    assert debug["reason"] == "ok"
    summary = json.loads((output / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["admission"] == admission
    assert summary["group"] == "BN"
    assert (output / "manifest.jsonl").read_text(encoding="utf-8") == Path(paths["manifest"]).read_text(
        encoding="utf-8"
    )


def test_failure_admission_starts_from_empty_negative_baseline(tmp_path: Path) -> None:
    paths, _ = _artifacts(tmp_path)
    for key in ("baseline_negative_meta", "baseline_negative_index", "baseline_negative_actions"):
        paths.pop(key)
    output = tmp_path / "empty-negative-parent"
    main(Args(group="BS", **paths, output_dir=str(output), admission="failure", device="cpu"))

    summary = json.loads((output / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_items"]["negative"] == 0
    assert summary["admitted_items"] == {"positive": 0, "negative": 1}
    negative = torch.load(output / "negative/gpm_negative_memory_meta.pt", weights_only=False)
    assert [entry["provenance"]["source_format"] for entry in negative] == ["smol_npz"]
