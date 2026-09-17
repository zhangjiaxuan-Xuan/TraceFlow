# Training a retrieval head

TraceFlow publishes a model-agnostic trainer at
`scripts/train/train_retrieval_head.py`. It consumes cached features rather than
raw observations, so users can connect any lower policy or upper vision-language
model without modifying the training loop.

## Input contract

The metadata is JSONL with one object per feature row. By default each object
contains:

```json
{"task_id": 3, "action_id": "trajectory-0042", "split": "train"}
```

`task_id` is the positive-pair label. `action_id` is the leakage-prevention
group: one group must never occur in both splits. `split` accepts `train` and
`validation` (also `val`, `valid`, or `test`). If the split field is omitted,
the script deterministically assigns complete groups using
`--validation-fraction` and `--seed`.

Feature files are NumPy `.npy` matrices with shape `[N, D]` in exactly the same
row order as the JSONL file:

- lower head: `--lower-features lower.npy`
- upper head: `--upper-features upper.npy`
- fusion head: both matrices, with optional `[N]` `upper_age.npy` and
  `upper_available.npy`

Age values are normalized to `[0, 1]`; availability contains only zero or one.

## Example

```bash
python scripts/train/train_retrieval_head.py \
  --metadata cache/metadata.jsonl \
  --lower-features cache/lower.npy \
  --upper-features cache/upper.npy \
  --upper-age cache/upper_age.npy \
  --upper-available cache/upper_available.npy \
  --variant fusion \
  --output-dir outputs/my-fusion-head \
  --labels-per-batch 8 \
  --samples-per-label 4 \
  --epochs 30 \
  --steps-per-epoch 100 \
  --device cuda:0
```

The output contains `best.pt`, `last.pt`, `metrics.jsonl`, deterministic split
indices, and `training_manifest.json`. The checkpoints use the same
`dual_tower` schema loaded by TraceFlow evaluation. `last.pt` also stores the
optimizer state and can be continued with `--resume`.

To adapt the objective or sampling policy, the main extension points are
`supervised_contrastive_loss`, `TaskPairBatchSampler`, and the epoch loop in the
training script. Input file hashes and label mappings are retained in every
checkpoint for provenance.
