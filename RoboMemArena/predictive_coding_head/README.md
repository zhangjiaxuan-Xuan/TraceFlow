# PrediMem S2 Training: Predictive Coding Head (Qwen3-VL)

This folder provides the core PrediMem S2 training add-on for a Qwen3-VL style model with a **Predictive Coding Head**.

It focuses on the Predictive Coding Head integration logic (head + loss computation), so users can:

1. Clone their own upstream Qwen/Transformers training code.
2. Drop in this module.
3. Add a few lines to their existing trainer.

For the S1 low-level policy, both the environment setup and the training logic
can directly follow the official OpenPI repository:
https://github.com/Physical-Intelligence/openpi

For VLM training data construction, users may follow the official Qwen3-VL
multimodal `messages` / `content` format described in the official repository:
https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#using-transformers-to-chat

## Files

- `predictive_coding_head.py`: standalone utilities
  - resolve multimodal base model under wrappers
  - initialize the Predictive Coding Head
  - compute Predictive Coding Head next-image losses (MSE + cosine)
  - combine CE + predictive losses

## Minimal Integration Steps

In your training script:

1. Import utilities:

```python
from predictive_coding_head.predictive_coding_head import (
    init_predictive_coding_head,
    compute_predictive_coding_losses,
    combine_main_and_predictive_losses,
)
```

2. After loading your model:

```python
init_predictive_coding_head(model)
```

3. In your training step / custom Trainer:
- Forward pass with `output_hidden_states=True`
- Compute CE as usual
- Compute predictive losses from batch inputs and model outputs
- Combine them with weights

```python
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
    pixel_values=batch.get("pixel_values"),
    image_grid_thw=batch.get("image_grid_thw"),
    output_hidden_states=True,
    return_dict=True,
)

ce_loss = outputs.loss
mse_loss, cosine_loss = compute_predictive_coding_losses(model, batch, outputs)
loss = combine_main_and_predictive_losses(
    ce_loss,
    mse_loss,
    cosine_loss,
    mse_weight=0.1,
    cosine_weight=0.1,
)
```

4. Save head parameters (optional but recommended):

```python
import os
import torch
base = model.get_base_model() if hasattr(model, "get_base_model") else model
head = getattr(base, "predictive_coding_head", None)
if head is not None:
    torch.save(head.state_dict(), os.path.join(output_dir, "predictive_coding_head.pt"))
```

## Batch/Input Expectations

The helper expects these fields in each training batch:

- `input_ids`
- `pixel_values`
- `image_grid_thw`
- `num_images` (number of images per sample in sequence order)

It also expects the model config to expose `image_token_id`, and the model to expose `get_image_features(...)`.

## Notes

- Recommended initial weights: `mse_weight=0.1`, `cosine_weight=0.1`.
- Keep CE as the main loss; predictive losses are auxiliary.
- If token counts between hidden states and image features mismatch, the utility raises an explicit error.

## Qwen3-VL-8B Quick Wiring

This module can be directly integrated into a Qwen3-VL-8B training codebase.

1. Load Qwen3-VL-8B in your existing training stack:

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
```

2. Initialize the predictive head right after model creation:

```python
init_predictive_coding_head(model)
```

3. Keep your normal CE training path, and add the predictive losses in the same step:

```python
outputs = model(..., output_hidden_states=True, return_dict=True)
ce_loss = outputs.loss
mse_loss, cosine_loss = compute_predictive_coding_losses(model, batch, outputs)
loss = combine_main_and_predictive_losses(
    ce_loss, mse_loss, cosine_loss, mse_weight=0.1, cosine_weight=0.1
)
```

4. Ensure your batch includes:

- `input_ids`
- `pixel_values`
- `image_grid_thw`
- `num_images`

5. Save the predictive head together with model checkpoints to simplify resume/inference packaging.

## Evaluation Example

For evaluation, this add-on can be paired with tasks in a VLM+VLA pipeline.

One practical reference is a single-task benchmark run (for example, Task1), and the full 26-task async reference benchmark in `evaluation_benchmark/` can also be used as a reference integration path.

Evaluation environment wiring, policy serving, and task assets can be organized together with your benchmark stack.
