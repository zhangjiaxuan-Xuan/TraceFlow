from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_retrieval_head import train  # noqa: E402


def test_generic_retrieval_head_training_smoke(tmp_path: Path) -> None:
    rows = []
    generator = np.random.default_rng(17)
    lower = []
    for label in range(3):
        center = generator.normal(size=8).astype(np.float32)
        for sample in range(8):
            split = "train" if sample < 6 else "validation"
            rows.append({"task_id": label, "action_id": f"{label}-{sample}", "split": split})
            lower.append(center + 0.01 * generator.normal(size=8).astype(np.float32))
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    features = tmp_path / "lower.npy"
    np.save(features, np.asarray(lower, dtype=np.float32))
    output = tmp_path / "head"
    args = argparse.Namespace(
        metadata=metadata,
        lower_features=features,
        upper_features=None,
        upper_age=None,
        upper_available=None,
        variant="lower",
        output_dir=output,
        label_field="task_id",
        split_field="split",
        group_field="action_id",
        validation_fraction=0.2,
        hidden_dim=16,
        out_dim=8,
        epochs=1,
        steps_per_epoch=2,
        labels_per_batch=3,
        samples_per_label=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.07,
        projection_batch_size=32,
        device="cpu",
        seed=7,
        resume=False,
    )
    best = train(args)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert checkpoint["head_type"] == "dual_tower"
    assert checkpoint["variant"] == "lower"
    assert checkpoint["training_provenance"]["train_rows"] == 18
    assert (output / "metrics.jsonl").is_file()
