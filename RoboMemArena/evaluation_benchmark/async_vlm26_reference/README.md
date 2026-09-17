# Async VLM/VLA 26-Task Evaluation Reference

This folder contains one reference implementation of asynchronous VLM/VLA evaluation on the 26 RoboMemArena tasks.
It is provided as example code for inspection and adaptation, not as the only way to run the benchmark.

Use this folder if you want to reproduce or inspect the 26-task async evaluation logic in this reference implementation, especially:

- async VLM planning with a single-slot subtask buffer
- dual-camera VLM input: `agentview_rgb` and `eye_in_hand_rgb`
- unlimited historical keyframes (`K_MAX=0`)
- task-conditioned VLM prompting with historical keyframes and recent visual context
- corrected metric naming:
  - `CSR`: average stage/process completion percentage
  - `TSR`: strict required-stage completion rate

This is one reference integration path. The model-agnostic benchmark interface remains in `evaluation_benchmark/scripts/`.

Users may adjust the prompt wording for different model families, languages, context lengths, or deployment needs. However, the task semantics should stay as close as possible to the official 26-task table: object identities, target receptacles, ordering constraints, counting requirements, occlusion/memory requirements, and final goals should not be changed. If you rewrite prompts, use the task table and the BDDL files as the source of truth.

## Files

```text
async_vlm26_reference/
  README.md
  eval_fullvlm26_async_vlm_vla.py
  run_fullvlm26_async_vlm_vla_csr_tsr.sh
  fullvlm_v2_26_memory_tasks.json
```

## Required Local Inputs

Set these paths before running:

OpenPI source interface:

- Default: use the bundled minimal runtime at `third_party/openpi_minimal`.
- Optional: use your own OpenPI source tree via `OPENPI_ROOT`.
- Required entries under `OPENPI_ROOT`:
  - `scripts/serve_policy.py`
  - `packages/openpi/src`
  - `packages/openpi-client/src`

```bash
export OPENPI_ROOT=/abs/path/to/openpi  # optional; defaults to repo-bundled third_party/openpi_minimal
export OPENPI_INFERENCE_ROOT=/abs/path/to/openpi_inference
export TARGET_LIBERO_PATH=/abs/path/to/LIBERO/libero
export VLM_CKPT=/abs/path/to/vlm/checkpoint
export VLA_CKPT=/abs/path/to/vla/checkpoint
```

If your policy stack uses a named VLA config, you can set it explicitly:

```bash
export VLA_CONFIG=<your_vla_config_name>

# if not set, the runner default is:
# VLA_CONFIG=pi05_robomemarena
```

When using external processor files together with a VLM checkpoint, set:

```bash
export VLM_PROCESSOR_DIR=/abs/path/to/vlm_processor_dir
```

## Run All 26 Tasks

From the repository root:

```bash
cd evaluation_benchmark/async_vlm26_reference

export TASKS_JSON='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]'
export NUM_TRIALS=1
export SEED=0

export ASYNC_VLM=1
export VLM_QUEUE_SIZE=1
export VLM_INTERVAL=5
export N_RECENT=5
export K_MAX=0
export VLM_USE_WRIST=1
export VLM_USE_KEYFRAME_MEMORY=1
export VLM_INPUT_PROFILE=<your_vlm_input_profile>

bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

To run a subset:

```bash
export TASKS_JSON='[1,2,3]'
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

To run 10 trials per task:

```bash
export NUM_TRIALS=10
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

`summary.tsv` reports:

- `csr`: average stage/process completion percentage for the task
- `tsr`: strict required-stage completion rate for the task

`aggregate.json` reports macro averages over completed tasks:

- `macro_csr`
- `macro_tsr`
