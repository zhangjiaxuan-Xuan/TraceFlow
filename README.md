# TraceFlow

TraceFlow is a reproducible release for memory-guided vision-language-action evaluation on LIBERO, LIBERO-Plus, and RoboMemArena. It includes the evaluation code, release-locked configurations, retrieval-head training code, and TraceBankStack continual-learning workflows.

TraceFlow-owned retrieval heads and memory banks are published separately at [ModelScope: JoeyXuan/traceflow-models](https://modelscope.ai/models/JoeyXuan/traceflow-models). Official Pi0.5 and PrediMem checkpoints are third-party resources and are not redistributed by this project.

## Released results and configurations

| Benchmark | Released configuration | Result |
| --- | --- | ---: |
| LIBERO-Spatial | B6500 positive-only, V1, `k+=8` | 494 / 500 (98.8%) |
| LIBERO-Object | B50 positive + N+C_pi negative, V1, `k+=16`, `k-=8` | 500 / 500 (100.0%) |
| LIBERO-Goal | B6500 positive-only, V1, `k+=8` | 491 / 500 (98.2%) |
| LIBERO-10 | 96.2% raw-best bank, V1, `k+=16`, `k-=8` | 481 / 500 (96.2%) |
| LIBERO-Plus-10 | LIBERO-10 positive/negative memory, V1 | 2043 / 2519 (81.1%) |

The reported **98.3% LIBERO score is a best-of-suite envelope**: it combines the four independently selected suite configurations above (1966 / 2000). It is not the result of one shared configuration or one unified 1966 / 2000 run. Machine-readable settings and provenance are under `configs/release/` and `provenance/`.

The RoboMemArena release contains four frozen protocols:

| Suite | Tasks | Memory and head |
| --- | --- | --- |
| Sequence | 1, 2, 3, 22 | Extra8 joint memory, Upper-V0 |
| Transferring | 18, 19, 25, 26 | Extra8 memory, Fusion-V1 |
| Counting | 6, 7, 8, 9, 10, 15, 16 | Suite-only memory, Fusion-V1.1 |
| Occlusion | 4, 5, 11, 12, 13, 14, 17, 20, 21, 23, 24 | Suite-only memory, Fusion-V1.1 |

## 1. Clone and prerequisites

```bash
git clone https://github.com/zhangjiaxuan-Xuan/TraceFlow.git
cd TraceFlow
```

Required system software:

- Linux with an NVIDIA driver compatible with the selected PyTorch/CUDA packages;
- Conda or Mamba;
- [`uv`](https://docs.astral.sh/uv/) on `PATH`;
- Git, `curl`, `unzip`, and a working EGL/MuJoCo runtime.

The release was smoke-tested on a 40 GB H100 MIG instance. Full evaluation defaults are substantially larger than smoke settings; adjust worker and batch variables to the available GPU memory.

## 2. Rebuild the three environments

The installer creates environments in the user's normal Conda root. It does not create a project-local virtual environment.

```bash
bash scripts/setup/create_envs.sh
```

This creates:

- `traceflow-openpi` (Python 3.11): OpenPI policy runtime, PyTorch, JAX, FAISS, ModelScope, and the TraceFlow package;
- `traceflow-libero` (Python 3.10): LIBERO, robosuite, MuJoCo, and evaluation dependencies;
- `traceflow-predimem` (Python 3.10): PrediMem Upper-VLM and Qwen3-VL dependencies.

Every package is installed with an explicit interpreter through `uv pip install --python ...`. Useful installation overrides are:

```bash
UV_INDEX_URL=https://pypi.org/simple bash scripts/setup/create_envs.sh

# If Conda cannot be discovered automatically:
CONDA_ROOT="$HOME/miniforge3" bash scripts/setup/create_envs.sh
```

`create_envs.sh` fetches the pinned LIBERO source checkout, but it does **not** download Pi0.5 or PrediMem model weights. Verify the reconstructed environments with:

```bash
bash scripts/doctor.sh
bash scripts/acceptance/run_static.sh
```

The doctor checks the three interpreters, CUDA visibility, MuJoCo/EGL, and the pinned LIBERO checkout. After assets and checkpoints are available, enable the extended checks:

```bash
TRACEFLOW_DOCTOR_ASSETS=1 \
TRACEFLOW_DOCTOR_CHECKPOINTS=1 \
bash scripts/doctor.sh
```

## 3. TraceFlow assets on ModelScope

The ModelScope repository contains only TraceFlow-produced artifacts:

- retrieval heads;
- positive and negative memory metadata;
- FAISS indexes and referenced actions;
- task/suite scopes, gate configurations, provenance, hashes, sizes, and sample counts.

Its top-level organization is `common/`, `libero/`, `arena/`, and `cl/`. The release lock currently exposes these logical groups:

```text
common.libero_head
libero.b50_positive
libero.b6500_positive
libero.ncpi_negative
libero.libero10_positive
libero.libero10_negative
arena.extra8
arena.counting
arena.occlusion
cl.libero10
cl.transferring
```

No manual cache path construction is needed. Entry scripts call `modelscope.snapshot_download()`, use the snapshot path returned by ModelScope, and verify it against `src/traceflow/manifest.lock.json` (SHA-256, size, task scope, vector count, and action references).

The default endpoint is the ModelScope international site. The downloader respects ModelScope's normal cache configuration, including `MODELSCOPE_CACHE`:

```bash
MODELSCOPE_CACHE="$HOME/.cache/modelscope" \
bash scripts/eval/run_libero_best_98_3.sh
```

For an already downloaded snapshot or a fully offline machine, point directly to the snapshot root containing `manifest.json`:

```bash
export TRACEFLOW_ASSET_ROOT=/path/to/traceflow-models-snapshot
```

You can resolve and verify selected groups without starting an evaluation:

```bash
conda run -n traceflow-openpi traceflow assets \
  common.libero_head libero.b50_positive libero.ncpi_negative
```

## 4. Official third-party checkpoints

TraceFlow does not redistribute model weights. Obtain them under their original licenses and terms from the official projects:

- Pi0.5-LIBERO: the official OpenPI checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`);
- PrediMem Upper VLM and VLA: the official [`huashuolei/PrediMem`](https://huggingface.co/huashuolei/PrediMem) release.

There are two supported workflows.

### Use existing converted checkpoints

Each override must point to a directory containing `model.safetensors`:

```bash
export PI05_CHECKPOINT=/path/to/pi05_libero_pytorch
export PREDIMEM_UPPER_CHECKPOINT=/path/to/vlm_tasks1to26_ckpt74500
export PREDIMEM_VLA_CHECKPOINT=/path/to/vla_alltask_pytorch
```

This is the recommended mode on managed clusters or wherever checkpoints are already present.

### Let the resolver obtain official checkpoints

If an override is absent, the resolver downloads from the official source and performs the required JAX-to-PyTorch conversion. Converted files are placed under `${XDG_CACHE_HOME:-$HOME/.cache}/traceflow/official`, or under an explicit location:

```bash
export TRACEFLOW_CHECKPOINT_ROOT=/path/to/checkpoint-cache
conda run -n traceflow-openpi traceflow checkpoints pi05
conda run -n traceflow-openpi traceflow checkpoints predimem-upper
conda run -n traceflow-openpi traceflow checkpoints predimem-vla
```

The evaluation launchers use the same resolver, so these commands are optional prefetch checks rather than required manual steps.

## 5. Benchmark sources and assets

LIBERO is pinned and installed automatically during environment creation. It can also be fetched independently:

```bash
bash scripts/setup/fetch_benchmarks.sh libero
```

LIBERO-Plus source and its official asset archive are fetched on the first Plus run, or explicitly with:

```bash
bash scripts/setup/fetch_benchmarks.sh plus
```

To keep third-party checkouts outside the repository, set `TRACEFLOW_THIRD_PARTY`. To place the resumable LIBERO-Plus archive cache elsewhere, set `TRACEFLOW_DOWNLOAD_CACHE`.

## 6. Evaluation entrypoints

All paths below are repository-relative. Outputs default to `outputs/<run>_<UTC timestamp>/`; set `TRACEFLOW_OUTPUT_ROOT` globally or `RUN_ROOT` for one run. Every release launcher supports a dependency-free configuration inspection:

```bash
bash scripts/eval/run_libero_best_98_3.sh --print-config
bash scripts/eval/run_arena_counting.sh --print-config
```

### LIBERO best-of-suite envelope

This command runs Spatial, Object, Goal, and LIBERO-10 sequentially and writes an aggregate `summary.json`:

```bash
GPU=0 \
EPISODES_PER_TASK=50 \
BATCH_SIZE=16 \
ENV_WORKERS=32 \
bash scripts/eval/run_libero_best_98_3.sh
```

Common overrides are `SEED`, `SAVE_VIDEOS`, `RESUME`, `RUN_ROOT`, `OPENPI_PYTHON`, and `LIBERO_PYTHON`.

### LIBERO-Plus-10

```bash
GPU=0 \
EPISODES_PER_TASK=1 \
BATCH_SIZE=8 \
ENV_WORKERS=16 \
bash scripts/eval/run_libero_plus10.sh
```

The full published protocol evaluates all 2,519 variants. `LIBERO_PLUS_ROOT` can point to an existing checkout with its asset directory already installed.

### RoboMemArena

The official protocols are dual-GPU by default:

```bash
UPPER_GPU=0 LOWER_GPU=1 bash scripts/eval/run_arena_sequence.sh
UPPER_GPU=0 LOWER_GPU=1 bash scripts/eval/run_arena_transferring.sh
UPPER_GPU=0 LOWER_GPU=1 bash scripts/eval/run_arena_counting.sh
UPPER_GPU=0 LOWER_GPU=1 bash scripts/eval/run_arena_occlusion.sh
```

The launchers accept `TASK_IDS`, `EPISODES_PER_TASK`, `UPPER_BATCH_SIZE`, `LOWER_BATCH_SIZE`, `ENV_WORKERS`, `SEED`, `RUN_ROOT`, `SAVE_VIDEO`, and `RESUME`. For a small single-GPU path check:

```bash
UPPER_GPU=0 LOWER_GPU=0 \
TASK_IDS=1 EPISODES_PER_TASK=1 \
UPPER_BATCH_SIZE=2 LOWER_BATCH_SIZE=2 ENV_WORKERS=2 \
bash scripts/eval/run_arena_sequence.sh
```

`PREFLIGHT_ONLY=1` resolves and validates the selected configuration, assets, and model paths without running episodes.

## 7. TraceBankStack continual learning

Both workflows support `collect`, `success`, `failure`, `joint`, or `all`. Positive and negative banks remain physically separate; inherited memory is retained, rounds are recorded independently, and existing campaigns can resume.

### LIBERO-10

```bash
bash scripts/eval/run_tracebankstack_libero10.sh collect
bash scripts/eval/run_tracebankstack_libero10.sh success
bash scripts/eval/run_tracebankstack_libero10.sh failure
bash scripts/eval/run_tracebankstack_libero10.sh joint

# Run the complete sequence:
bash scripts/eval/run_tracebankstack_libero10.sh all
```

Important overrides include `CAMPAIGN_ROOT`, `ROUNDS`, `EPISODES_PER_TASK`, `TASK_IDS_CSV`, `GPU`, `PORT`, `BATCH_SIZE`, `ENV_WORKERS`, and `SEED`. The thin `*_success.sh`, `*_failure.sh`, and `*_joint.sh` wrappers are suitable for independent scheduling.

### RoboMemArena Transferring

```bash
UPPER_GPU=0 LOWER_GPU=1 \
bash scripts/eval/run_tracebankstack_transferring.sh all
```

The same individual modes and thin wrappers are available. Common controls are `CAMPAIGN_ROOT`, `ROUNDS`, `START_ROUND`, `EPISODES_PER_TASK`, `UPPER_BATCH_SIZE`, `LOWER_BATCH_SIZE`, `ENV_WORKERS`, and `SAVE_VIDEO`.

## 8. Train a retrieval head

The public trainer consumes cached features rather than binding training to a particular VLA/VLM implementation. Inputs are:

- a JSONL metadata file with one object per feature row;
- lower-policy and/or upper-model feature matrices in `[N, D]` NumPy `.npy` format;
- labels (default: `task_id`) and leakage-safe grouping IDs (default: `action_id`);
- optional explicit `split`, normalized upper-feature age, and upper-feature availability arrays.

Example fusion-head training:

```bash
conda run -n traceflow-openpi python scripts/train/train_retrieval_head.py \
  --metadata /path/to/metadata.jsonl \
  --lower-features /path/to/lower_features.npy \
  --upper-features /path/to/upper_features.npy \
  --variant fusion \
  --output-dir /path/to/output/head \
  --epochs 30 \
  --steps-per-epoch 100 \
  --labels-per-batch 8 \
  --samples-per-label 4
```

Use `--variant lower` or `--variant upper` for a single tower, and `--resume` to continue from `last.pt`. The output includes runtime-compatible `best.pt` and `last.pt`, `metrics.jsonl`, deterministic split indices, and a SHA-256 provenance manifest.

## Validation and licensing

Run the complete static release gate after modifying code or documentation:

```bash
bash scripts/acceptance/run_static.sh
```

TraceFlow code is released under Apache-2.0. Embedded or fetched third-party components, benchmarks, assets, and official checkpoints retain their original licenses and notices; see `THIRD_PARTY_NOTICES.md` before redistribution.
