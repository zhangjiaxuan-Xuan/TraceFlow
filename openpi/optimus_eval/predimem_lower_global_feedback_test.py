from __future__ import annotations

import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoProcessor = object
transformers_stub.LogitsProcessor = object
transformers_stub.LogitsProcessorList = list
transformers_stub.Qwen3VLForConditionalGeneration = object
sys.modules["transformers"] = transformers_stub

from optimus_eval.predimem_upper_server import BatchedUpperPlanner
from optimus_eval.predimem_upper_server import UpperLanguageGuidance
from optimus_eval.upper_knn_language_guidance import NumpyInnerProductIndex


class _IdentityFusionHead:
    variant = "fusion"

    def __call__(self, lower, upper, upper_age, upper_available):
        del upper, upper_age, upper_available
        return F.normalize(lower.float(), dim=-1)


def _guidance() -> UpperLanguageGuidance:
    keys = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    return UpperLanguageGuidance(
        head=_IdentityFusionHead(),
        index=NumpyInnerProductIndex(keys),
        task_ids=np.asarray([1, 1, 1, 1, 2], dtype=np.int64),
        subtasks=np.asarray(["stage a", "stage a", "stage b", "stage b", "other"], dtype=object),
        stage_indices=np.asarray([0, 0, 1, 1, 0], dtype=np.int64),
        anchor_progress=np.asarray([0.1, 0.2, 0.6, 0.7, 0.1], dtype=np.float32),
        stage_progress=np.asarray([0.1, 0.2, 0.1, 0.2, 0.1], dtype=np.float32),
        action_ids=np.asarray(["traj-a", "traj-a", "traj-b", "traj-b", "other"], dtype=object),
        top_k=2,
        temperature=0.07,
        min_task_purity=0.0,
        min_subtask_confidence=0.0,
        interpolation=0.8,
    )


def test_global_probe_search_does_not_pre_filter_by_candidate_label() -> None:
    result = _guidance().score_lower_subtask_probes_global(
        raw_upper_feature=np.zeros(2, dtype=np.float32),
        task_id=1,
        probes=[
            {"candidate": "stage a", "feature": np.asarray([0.0, 1.0], dtype=np.float32)},
            {"candidate": "stage b", "feature": np.asarray([0.0, 1.0], dtype=np.float32)},
        ],
        top_k=2,
        current_stage=None,
        current_progress=None,
        progress_scale=0.15,
    )

    assert result["bank_scope"] == "current-task-global"
    assert result["bank_size"] == 4
    assert result["best_candidate"] == "stage b"
    rows = {row["candidate"]: row for row in result["scores"]}
    assert rows["stage a"]["inferred_subtask"] == "stage b"
    assert rows["stage a"]["candidate_posterior"] == 0.0
    assert rows["stage b"]["candidate_posterior"] > 0.99
    assert all(neighbor["subtask"] == "stage b" for neighbor in rows["stage a"]["top_neighbors"])


def test_probe_candidates_fill_retrieval_collapse_from_task_stages() -> None:
    candidates = _guidance().build_probe_candidates(
        task_id=1,
        main_subtask="stage a",
        retrieved_candidates=["stage a", " stage a "],
        center_stage=0,
        max_count=4,
    )
    assert candidates == ["stage a", "stage b"]


def test_global_feedback_shadow_records_but_control_rewrites() -> None:
    diagnostic = {
        "available": True,
        "bank_scope": "current-task-global",
        "bank_size": 4,
        "top_k": 2,
        "best_candidate": "stage b",
        "best_score": 0.9,
        "best_candidate_posterior": 0.95,
        "best_inferred_subtask": "stage b",
        "best_inferred_stage": 1,
        "best_inferred_progress": 0.6,
        "best_agrees_with_posterior": True,
        "margin": 0.2,
        "scores": [],
    }
    planner = BatchedUpperPlanner.__new__(BatchedUpperPlanner)
    planner.upper_guidance_lower_global_min_posterior = 0.35
    planner.upper_guidance_lower_global_min_score = 0.45
    planner.upper_guidance_lower_global_min_margin = 0.03
    original = ([('stage a', 1.0)], 0.8, {"candidate": "stage a"})

    planner.upper_guidance_lower_global_mode = "shadow"
    candidates, _, shadow_meta = planner._apply_lower_global_feedback(original, diagnostic)
    assert candidates == original[0]
    assert shadow_meta["lower_global_decision"] == "shadow-accept"
    assert not shadow_meta.get("lower_global_applied", False)

    planner.upper_guidance_lower_global_mode = "control"
    candidates, _, control_meta = planner._apply_lower_global_feedback(original, diagnostic)
    assert candidates == [("stage b", 1.0)]
    assert control_meta["lower_global_applied"] is True
    assert control_meta["candidate_stage"] == 1
    assert control_meta["candidate_progress"] == 0.6
