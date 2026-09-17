from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import random
import re
import sys

OPENPI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPENPI_ROOT))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import h5py
import numpy as np
from PIL import Image
from safetensors import safe_open
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from optimus_eval.predimem_upper_server import _load_upper_guidance
from optimus_eval.upper_knn_language_guidance import subtask_prefix
from scripts.robomemarena.cache_predimem_upper_features import _load_context, _messages, _read_rows

ARENA_ROOT = Path(__file__).resolve().parents[3] / "RoboMemArena"
sys.path.insert(0, str(ARENA_ROOT))
from predictive_coding_head.predictive_coding_head import (  # noqa: E402
    combine_main_and_predictive_losses,
    compute_predictive_coding_losses,
    init_predictive_coding_head,
    resolve_multimodal_base_model,
)


def primitive_for_row(row: dict) -> str:
    stage = int(row["stage_index"])
    stem = Path(row["segment_paths"][stage]).stem
    stem = re.sub(r"_\d+_seed\d+_task\d+$", "", stem)
    return " ".join(stem.replace("_", " ").replace("-", " ").lower().split())


def keyframe_code(row: dict, n_recent: int) -> list[int]:
    stage = int(row["stage_index"])
    local_frame = int(row["stage_frame_index"])
    start = max(0, int(row["global_frame_index"]) - n_recent + 1)
    stage_offset = sum(int(value) for value in row["segment_lengths"][:stage])
    with h5py.File(row["segment_paths"][stage], "r") as stream:
        values = np.asarray(stream["data/demo_0/keyframe_indices"], dtype=np.int64)
    current_global = stage_offset + local_frame
    return [int(stage_offset + value - start + 1) for value in values if start <= stage_offset + value <= current_global]


class UpperDataset(Dataset):
    def __init__(self, rows: list[dict], n_recent: int):
        self.rows = rows
        self.n_recent = n_recent

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        encoded = _load_context(row, self.n_recent)
        images = [[Image.open(BytesIO(value)).convert("RGB") for value in group] for group in encoded]
        messages = _messages(row, *images)
        target = json.dumps(
            {
                "current_primitive": primitive_for_row(row),
                "keyframe_positions": keyframe_code(row, self.n_recent),
            },
            ensure_ascii=False,
        )
        return {
            "row_index": int(row["row_index"]),
            "task_id": int(row["task_id"]),
            "messages": messages,
            "images": [image for group in images for image in group],
            "target": target,
        }


class UpperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples: list[dict]) -> dict:
        texts = [
            self.processor.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True)
            for sample in samples
        ]
        flat_images = [image for sample in samples for image in sample["images"]]
        prompt = self.processor(text=texts, images=flat_images, return_tensors="pt", padding=True)
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        sequences, labels, token_types, prompt_lengths, target_ids = [], [], [], [], []
        token_type_key = "mm_token_type_ids" if "mm_token_type_ids" in prompt else "token_type_ids"
        prompt_token_types = prompt.get(token_type_key)
        for sample_index, (row, mask, sample) in enumerate(
            zip(prompt["input_ids"], prompt["attention_mask"], samples, strict=True)
        ):
            prefix = row[mask.bool()]
            target = tokenizer.encode(sample["target"], add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                target.append(tokenizer.eos_token_id)
            target_tensor = torch.tensor(target, dtype=prefix.dtype)
            sequences.append(torch.cat((prefix, target_tensor)))
            labels.append(torch.cat((torch.full_like(prefix, -100), target_tensor)))
            if prompt_token_types is not None:
                prefix_types = prompt_token_types[sample_index][mask.bool()]
                token_types.append(torch.cat((prefix_types, torch.zeros_like(target_tensor))))
            prompt_lengths.append(len(prefix))
            target_ids.append(target)
        width = max(map(len, sequences))
        input_ids = torch.full((len(samples), width), pad_id, dtype=sequences[0].dtype)
        attention = torch.zeros((len(samples), width), dtype=torch.long)
        label_batch = torch.full((len(samples), width), -100, dtype=torch.long)
        token_type_batch = torch.zeros((len(samples), width), dtype=torch.long)
        for index, (sequence, label) in enumerate(zip(sequences, labels, strict=True)):
            input_ids[index, : len(sequence)] = sequence
            attention[index, : len(sequence)] = 1
            label_batch[index, : len(sequence)] = label
            if token_types:
                token_type_batch[index, : len(sequence)] = token_types[index]
        prompt["input_ids"] = input_ids
        prompt["attention_mask"] = attention
        prompt["labels"] = label_batch
        if token_types:
            prompt[token_type_key] = token_type_batch
        prompt["num_images"] = torch.tensor([len(sample["images"]) for sample in samples], dtype=torch.long)
        prompt["prompt_lengths"] = torch.tensor(prompt_lengths, dtype=torch.long)
        prompt["row_indices"] = torch.tensor([sample["row_index"] for sample in samples], dtype=torch.long)
        prompt["task_ids"] = torch.tensor([sample["task_id"] for sample in samples], dtype=torch.long)
        prompt["target_ids"] = target_ids
        return prompt


def mixture_ce(logits, target_ids, prompt_lengths, candidates, tokenizer) -> torch.Tensor:
    losses = []
    for row, (tokens, start, candidate_pack) in enumerate(
        zip(target_ids, prompt_lengths.tolist(), candidates, strict=True)
    ):
        encoded = []
        for primitive, weight in candidate_pack:
            ids = tokenizer.encode(subtask_prefix(primitive), add_special_tokens=False)
            if ids and weight > 0:
                encoded.append((ids, float(weight)))
        for offset, target in enumerate(tokens):
            base_log_probs = torch.log_softmax(logits[row, start + offset - 1].float(), dim=-1)
            alive = [(ids, weight) for ids, weight in encoded if offset < len(ids) and ids[:offset] == tokens[:offset]]
            if not alive:
                losses.append(-base_log_probs[target])
                continue
            total = sum(weight for _, weight in alive)
            memory_probability = sum(weight for ids, weight in alive if ids[offset] == target) / total
            interpolation = float(candidate_pack.interpolation)
            probability = (1.0 - interpolation) * base_log_probs[target].exp() + interpolation * memory_probability
            losses.append(-torch.log(probability.clamp_min(1e-12)))
    return torch.stack(losses).mean()


class CandidatePack(list):
    def __init__(self, values, interpolation: float):
        super().__init__(values)
        self.interpolation = interpolation


def restore_official_lfp_head(model, checkpoint: Path) -> None:
    base = resolve_multimodal_base_model(model)
    init_predictive_coding_head(model, head_attr="lfp_head")
    standalone = checkpoint / "lfp_head.pt"
    if standalone.is_file():
        state = torch.load(standalone, map_location="cpu", weights_only=True)
    else:
        archive = checkpoint / "model.safetensors"
        with safe_open(archive, framework="pt", device="cpu") as stream:
            names = [name for name in stream.keys() if name.startswith("lfp_head.")]
            if not names:
                raise FileNotFoundError(f"No official lfp_head weights in {checkpoint}")
            state = {name.removeprefix("lfp_head."): stream.get_tensor(name) for name in names}
    base.lfp_head.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--upper-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-save", action="store_true", help="Smoke only; do not use for formal training")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = _read_rows(args.manifest)
    if args.max_items:
        rows = rows[: args.max_items]
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    restore_official_lfp_head(model, args.checkpoint)
    model.cuda()
    base = resolve_multimodal_base_model(model)
    frozen = 0
    for name, parameter in model.named_parameters():
        if name.startswith("visual.") or ".visual." in name:
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    if frozen == 0:
        raise RuntimeError("freeze_vision_tower requested but no visual parameters were found")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    guidance, = _load_upper_guidance(
        head_path=args.head,
        manifest_path=args.manifest,
        upper_features=args.upper_features,
        top_k=16,
        temperature=0.07,
        min_task_purity=0.5,
        min_subtask_confidence=0.8,
        interpolation=0.8,
        bank_per_primitive=16,
        bank_seed=17,
        device="cuda:0",
    )
    loader = DataLoader(
        UpperDataset(rows, 5), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=UpperCollator(processor), pin_memory=True, persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    micro_step = 0
    step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while step < args.steps:
        for batch in loader:
            metadata = {key: batch.pop(key) for key in ("prompt_lengths", "row_indices", "task_ids", "target_ids")}
            batch = {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            prompt_lengths = metadata["prompt_lengths"].cuda()
            context_features = torch.stack(
                [outputs.hidden_states[-1][i, int(length) - 1] for i, length in enumerate(prompt_lengths)]
            )
            retrieved = guidance.retrieve(
                context_features.detach().float().cpu().numpy(), metadata["task_ids"].numpy()
            )
            packs = [CandidatePack(values, interpolation) for values, interpolation, _ in retrieved]
            ce = mixture_ce(outputs.logits, metadata["target_ids"], prompt_lengths, packs, processor.tokenizer)
            mse, cosine = compute_predictive_coding_losses(model, batch, outputs, head_attr="lfp_head")
            loss = combine_main_and_predictive_losses(ce, mse, cosine, mse_weight=0.1, cosine_weight=0.1)
            (loss / args.gradient_accumulation).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            else:
                continue
            if step % 10 == 0:
                print(json.dumps({"step": step, "loss": float(loss.detach()), "text_loss": float(ce.detach())}), flush=True)
            if step % args.save_every == 0 or (step == args.steps and not args.skip_final_save):
                checkpoint = args.output / f"step_{step}"
                model.save_pretrained(checkpoint, safe_serialization=True)
                processor.save_pretrained(checkpoint)
                torch.save(base.lfp_head.state_dict(), checkpoint / "lfp_head.pt")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
