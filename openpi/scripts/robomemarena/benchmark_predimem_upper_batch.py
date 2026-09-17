from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from optimus_eval.predimem_upper_server import BatchedUpperPlanner, _load_tasks
from scripts.robomemarena.cache_predimem_upper_features import _load_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PrediMem Upper generation batches after one model load.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="12,16,20,24,28,32")
    parser.add_argument("--n-recent", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("batch sizes must be positive")

    rows = [json.loads(line) for line in args.manifest.open() if line.strip()]
    if len(rows) < max(batch_sizes):
        raise ValueError(f"Manifest has {len(rows)} rows but batch {max(batch_sizes)} was requested")
    tasks = _load_tasks(args.task_config)
    contexts = [_load_context(row, args.n_recent, []) for row in rows[: max(batch_sizes)]]

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    def prepare(count: int) -> tuple[dict, int]:
        messages: list[str] = []
        flat_images: list[Image.Image] = []
        for row, context in zip(rows[:count], contexts[:count], strict=True):
            memory_main_raw, memory_wrist_raw, context_main_raw, context_wrist_raw = context
            memory_main = [Image.open(BytesIO(value)).convert("RGB") for value in memory_main_raw]
            memory_wrist = [Image.open(BytesIO(value)).convert("RGB") for value in memory_wrist_raw]
            context_main = [Image.open(BytesIO(value)).convert("RGB") for value in context_main_raw]
            context_wrist = [Image.open(BytesIO(value)).convert("RGB") for value in context_wrist_raw]
            prompt = BatchedUpperPlanner._messages(
                tasks[int(row["task_id"])], memory_main, memory_wrist, context_main, context_wrist
            )
            messages.append(processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
            for main, wrist in [
                *zip(memory_main, memory_wrist, strict=True),
                *zip(context_main, context_wrist, strict=True),
            ]:
                flat_images.extend((main, wrist))
        inputs = processor(text=messages, images=flat_images, return_tensors="pt", padding=True)
        return ({key: value.to(args.device) if hasattr(value, "to") else value for key, value in inputs.items()}, len(flat_images))

    results = []
    for batch_size in batch_sizes:
        inputs, image_count = prepare(batch_size)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated_tokens = int(generated.sequences.shape[1] - inputs["input_ids"].shape[1])
            record = {
                "batch_size": batch_size,
                "images": image_count,
                "elapsed_seconds": elapsed,
                "rows_per_second": batch_size / elapsed,
                "generated_tokens_per_row_max": generated_tokens,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "free_gib": torch.cuda.mem_get_info()[0] / 2**30,
                "status": "ok",
            }
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            record = {"batch_size": batch_size, "status": "oom", "error": str(exc).splitlines()[0]}
            torch.cuda.empty_cache()
        results.append(record)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.output)
        print(json.dumps(record, sort_keys=True), flush=True)
        if record["status"] == "oom":
            break
    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
