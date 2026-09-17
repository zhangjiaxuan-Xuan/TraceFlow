# 26-Task Reference Evaluation

This folder provides reference VLM/VLA integration code for RoboMemArena.

The benchmark should be understood as one 1-26 task evaluation setting. A VLM/VLA system is still an adapter-style model stack: the evaluator creates the environment, loads the BDDL task, obtains the current observation and prompt, calls the planner/policy stack, executes the returned actions, and computes CSR/TSR.

For the user-facing full 1-26 task reference runner, use:

```text
evaluation_benchmark/async_vlm26_reference/
```

The source tree may keep historical helper files internally for compatibility with earlier experiments. Those helper names do not define separate public benchmark modes.

The required Python/runtime environment is based on the official OpenPI environment. For the VLM/Qwen3-VL side, use a newer `transformers` environment instead of relying on the original OpenPI venv with `transformers==4.48.1`, which does not provide `Qwen3VLForConditionalGeneration`.

The model-agnostic adapter interface remains in `evaluation_benchmark/scripts/`.
For external model integration, see [Evaluate Your Model on RoboMemArena](../docs/evaluate_your_model.md).

## Weight Interface (HF-ready)

Use interface variables to connect weights (HF or local) instead of hardcoding URLs:

```bash
# unified weight source
export WEIGHT_SOURCE=hf            # hf or local

# HF repo entry (example)
export HF_REPO_ID=<your_hf_repo_id>
export VLM_HF_SUBDIR=<vlm_subdir>
export VLA_HF_SUBDIR=<vla_subdir>

# local resolved paths used by scripts
export VLM_CKPT=/abs/path/to/vlm_checkpoint
export VLA_CKPT=/abs/path/to/vla_checkpoint
```

For local-only usage:

```bash
export WEIGHT_SOURCE=local
export VLM_CKPT=/abs/path/to/vlm_checkpoint
export VLA_CKPT=/abs/path/to/vla_checkpoint
```

## Files

```text
reference_evaluation/
  README.md
  legacy reference helper files
```

These files are kept to preserve compatibility with the original reference implementation. For a single 1-26 task command, prefer `evaluation_benchmark/async_vlm26_reference/run_fullvlm26_async_vlm_vla_csr_tsr.sh`.

## 1-26 Reference Runner

OpenPI source interface:

- Default: use the bundled minimal runtime at `third_party/openpi_minimal`.
- Optional: point `OPENPI_ROOT` to your own OpenPI checkout. No code changes are needed.

Required local inputs:

```bash
export OPENPI_ROOT=/abs/path/to/openpi  # optional; defaults to repo-bundled third_party/openpi_minimal
export OPENPI_INFERENCE_ROOT=/abs/path/to/openpi_inference
export TARGET_LIBERO_PATH=/abs/path/to/LIBERO/libero
export VLM_CKPT=/abs/path/to/vlm_checkpoint
export VLA_CKPT=/abs/path/to/vla_checkpoint
# optional override; default in runner is pi05_robomemarena
export VLA_CONFIG=<your_vla_config_name>
```

Run the full 1-26 task set:

```bash
cd evaluation_benchmark/async_vlm26_reference
export TASKS_JSON='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]'
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

Run a subset for debugging:

```bash
export TASKS_JSON='[1,2,3]'
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

## Outputs

The runner writes outputs under `OUT_ROOT`:

```text
${OUT_ROOT}/summary.tsv
${OUT_ROOT}/summary.json
${OUT_ROOT}/aggregate.json
${OUT_ROOT}/prompt_trace.tsv
${OUT_ROOT}/videos/task*/...
${OUT_ROOT}/task*/ep*/sync_vlm_trace.jsonl
```

Metric names:

- `CSR`: average stage completion rate. In output files this is stored in the
  legacy-compatible `goal_success_rate` field.
- `stage_success_rate`: strict required-stage completion rate. An episode has
  TSR=1 only when all required stages are completed.
- `stage_score_pct`: partial stage/process completion score.

The evaluator uses stage-based scoring for all tasks. Counting-pour tasks 6, 7,
8, 9, 10, 15, 16, and 22 additionally monitor 30 environment steps after the
second completed pour and reject a third pour. Microwave tasks do not require
the final microwave-close stage.
Set `FAIL_ON_EXTRA_POUR=0` only for diagnostic runs; the default is `1`.
Set `POST_STAGE_STEPS` to change the post-stage monitoring window.

Example HF weight repository: `https://huggingface.co/huashuolei/PrediMem`
