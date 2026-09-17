from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch

from openpi.task_head.dual_tower_head import DualTowerRetrievalHead


class TorchFlatIP:
    """GPU fallback for environments where faiss-cpu is installed."""

    def __init__(self, index, device: torch.device, task_ids: np.ndarray, *, copy_rows: int = 65_536):
        flat = faiss.downcast_index(index)
        if not hasattr(flat, "get_xb"):
            raise TypeError("Torch GPU fallback requires a flat FAISS index")
        vectors = faiss.rev_swig_ptr(flat.get_xb(), flat.ntotal * flat.d).reshape(flat.ntotal, flat.d)
        source = torch.from_numpy(vectors)
        self.bank = torch.empty((flat.ntotal, flat.d), device=device, dtype=torch.float16)
        for start in range(0, flat.ntotal, copy_rows):
            stop = min(flat.ntotal, start + copy_rows)
            self.bank[start:stop].copy_(source[start:stop])
            print(f"FAISS bank uploaded to GPU: {stop}/{flat.ntotal}", flush=True)

        if len(task_ids) != flat.ntotal:
            raise ValueError("Task labels must align with every FAISS bank row")
        self.task_rows = {
            int(task): torch.from_numpy(np.flatnonzero(task_ids == task)).to(device=device, dtype=torch.long)
            for task in np.unique(task_ids)
        }

    def search(
        self,
        queries: torch.Tensor,
        top_k: int,
        query_task_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search each query only inside its known task bank."""
        queries = queries.to(device=self.bank.device, dtype=self.bank.dtype)
        query_task_ids = np.asarray(query_task_ids, dtype=np.int32)
        scores_out = np.empty((len(queries), top_k), dtype=np.float32)
        indices_out = np.empty((len(queries), top_k), dtype=np.int64)
        for task in np.unique(query_task_ids):
            query_rows_np = np.flatnonzero(query_task_ids == task)
            bank_rows = self.task_rows.get(int(task))
            if bank_rows is None or len(bank_rows) < top_k:
                raise RuntimeError(f"Task {int(task)} has fewer than {top_k} memory anchors")
            query_rows = torch.from_numpy(query_rows_np).to(device=self.bank.device, dtype=torch.long)
            similarity = queries.index_select(0, query_rows) @ self.bank.index_select(0, bank_rows).T
            local_scores, local_indices = torch.topk(
                similarity,
                k=top_k,
                dim=1,
                largest=True,
                sorted=True,
            )
            global_indices = bank_rows.index_select(0, local_indices.reshape(-1)).reshape_as(local_indices)
            scores_out[query_rows_np] = local_scores.float().cpu().numpy()
            indices_out[query_rows_np] = global_indices.cpu().numpy()
        return scores_out, indices_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leave-one-trajectory-out guidance training candidates")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lower-features", type=Path, required=True)
    parser.add_argument("--upper-features", type=Path, required=True)
    parser.add_argument("--upper-age", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--search-k", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Smoke-test limit for query rows; bank remains complete",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_columns(path: Path, *, progress_every: int = 50_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream the large manifest and retain only retrieval-addressing columns."""
    action_ids: list[str] = []
    task_ids: list[int] = []
    episodes: list[int] = []
    frames: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for expected_index, line in enumerate(stream):
            row = json.loads(line)
            if int(row["row_index"]) != expected_index:
                raise ValueError(
                    f"Manifest row_index must be dense and ordered; "
                    f"row {expected_index} contains {row['row_index']}"
                )
            action_ids.append(str(row["action_id"]))
            task_ids.append(int(row["task_id"]))
            episodes.append(int(row["episode_index"]))
            frames.append(int(row["global_frame_index"]))
            if progress_every > 0 and (expected_index + 1) % progress_every == 0:
                print(f"manifest rows loaded: {expected_index + 1}", flush=True)
    return (
        np.asarray(action_ids),
        np.asarray(task_ids, dtype=np.int32),
        np.asarray(episodes, dtype=np.int32),
        np.asarray(frames, dtype=np.int32),
    )


def action_chunk(actions: np.ndarray, offsets: np.ndarray, trajectory: int, frame: int, horizon: int) -> np.ndarray:
    start, stop = int(offsets[trajectory]), int(offsets[trajectory + 1])
    trajectory_actions = actions[start:stop]
    frame = int(np.clip(frame, 0, len(trajectory_actions) - 1))
    chunk = np.asarray(trajectory_actions[frame : frame + horizon], dtype=np.float32)
    if len(chunk) < horizon:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)], axis=0)
    return chunk


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.search_k <= args.top_k:
        raise ValueError("search-k must be greater than top-k")
    output = args.output
    state_path = output / "build_state.json"
    if output.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Loading query manifest: {args.manifest}", flush=True)
    action_ids, task_ids, episode, frame = load_columns(args.manifest)
    source_rows = len(action_ids)
    print(f"Loaded {source_rows} query anchors", flush=True)
    lower = np.load(args.lower_features, mmap_mode="r")
    upper = np.load(args.upper_features, mmap_mode="r")
    upper_age = np.load(args.upper_age, mmap_mode="r")
    if not (source_rows == len(lower) == len(upper) == len(upper_age)):
        raise ValueError("Manifest and feature caches have inconsistent row counts")

    print(f"Loading retrieval head: {args.head}", flush=True)
    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    head = DualTowerRetrievalHead(
        variant=str(payload["variant"]),
        lower_dim=int(payload["lower_dim"]),
        upper_dim=int(payload["upper_dim"]),
        hidden=int(payload["hidden"]),
        out_dim=int(payload["out_dim"]),
    )
    head.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device(args.device)
    head.to(device).eval()

    # AOSS/FUSE does not provide reliable mmap semantics for large FAISS
    # indexes; read_index is slower but avoids process-level SIGBUS exits.
    print(f"Loading FAISS index: {args.index}", flush=True)
    index = faiss.read_index(str(args.index))
    print(f"Loaded FAISS index: rows={index.ntotal} dim={index.d}", flush=True)
    if index.d != int(payload["out_dim"]) or index.ntotal != source_rows:
        raise ValueError(f"FAISS index shape ({index.ntotal}, {index.d}) does not match rows/head")
    torch_index = None
    if device.type == "cuda":
        # The task identity is known in both training and evaluation. A Torch
        # backend lets us search the exact per-task subset instead of retrieving
        # globally and hoping enough candidates survive the task gate.
        print("Loading the flat bank into the task-filtered Torch GPU search backend", flush=True)
        torch_index = TorchFlatIP(index, device, task_ids)

    packed = np.load(args.actions, mmap_mode="r", allow_pickle=False)
    actions = packed["actions"]
    offsets = packed["offsets"]
    ids = [str(value) for value in packed["ids"].tolist()]
    id_to_trajectory = {value: index for index, value in enumerate(ids)}
    query_rows = source_rows if args.max_rows <= 0 else min(source_rows, int(args.max_rows))

    completed = 0
    if state_path.is_file():
        completed = int(json.loads(state_path.read_text())["completed_rows"])

    shape = (query_rows, args.top_k, 10, actions.shape[1])
    mode = "r+" if completed else "w+"
    blocks = np.lib.format.open_memmap(output / "blocks.npy", mode=mode, dtype=np.float16, shape=shape)
    weights = np.lib.format.open_memmap(
        output / "weights.npy", mode=mode, dtype=np.float32, shape=(query_rows, args.top_k)
    )
    candidate_indices = np.lib.format.open_memmap(
        output / "candidate_indices.npy", mode=mode, dtype=np.int32, shape=(query_rows, args.top_k)
    )
    candidate_scores = np.lib.format.open_memmap(
        output / "candidate_scores.npy", mode=mode, dtype=np.float16, shape=(query_rows, args.top_k)
    )
    if not completed:
        np.save(output / "episode_index.npy", episode[:query_rows])
        np.save(output / "frame_index.npy", frame[:query_rows])

    available = torch.ones(args.batch_size, device=device)
    with torch.inference_mode():
        for start in range(completed, query_rows, args.batch_size):
            stop = min(query_rows, start + args.batch_size)
            indices = np.arange(start, stop)
            lower_batch = torch.from_numpy(np.asarray(lower[indices], dtype=np.float32)).to(device)
            upper_batch = torch.from_numpy(np.asarray(upper[indices], dtype=np.float32)).to(device)
            age_batch = torch.from_numpy(np.asarray(upper_age[indices], dtype=np.float32)).to(device)
            embeddings = head(lower_batch, upper_batch, age_batch, available[: len(indices)])
            if torch_index is None:
                scores, neighbors = index.search(embeddings.float().cpu().numpy(), args.search_k)
            else:
                scores, neighbors = torch_index.search(embeddings, args.search_k, task_ids[indices])
            for local, query_index in enumerate(indices):
                valid_neighbors = [int(candidate) for candidate in neighbors[local] if candidate >= 0]
                if not valid_neighbors:
                    raise RuntimeError(f"FAISS returned no candidates for row {query_index}")
                # Training labels provide the true task identity, and runtime
                # evaluation also knows the task. Never let a cross-task top-1
                # retrieval redefine the candidate pool for the whole row.
                top_task = int(task_ids[query_index])
                selected_pairs = []
                selected_actions: set[str] = set()
                for candidate, score in zip(neighbors[local], scores[local], strict=True):
                    candidate = int(candidate)
                    if candidate < 0 or int(task_ids[candidate]) != top_task:
                        continue
                    action_id = action_ids[candidate]
                    if action_id == action_ids[query_index] or action_id in selected_actions:
                        continue
                    selected_actions.add(action_id)
                    selected_pairs.append((candidate, float(score)))
                    if len(selected_pairs) == args.top_k:
                        break
                selected = [candidate for candidate, _ in selected_pairs]
                if len(selected) != args.top_k:
                    raise RuntimeError(f"Insufficient cross-trajectory candidates for row {query_index}")
                selected_scores = np.asarray([score for _, score in selected_pairs], dtype=np.float32)
                candidate_indices[query_index] = np.asarray(selected, dtype=np.int32)
                candidate_scores[query_index] = selected_scores.astype(np.float16)
                logits = args.temperature * (selected_scores - selected_scores.max())
                row_weights = np.exp(logits)
                weights[query_index] = row_weights / row_weights.sum()
                for rank, candidate in enumerate(selected):
                    trajectory = id_to_trajectory[action_ids[candidate]]
                    blocks[query_index, rank] = action_chunk(actions, offsets, trajectory, frame[candidate], 10)
            blocks.flush()
            weights.flush()
            candidate_indices.flush()
            candidate_scores.flush()
            state_path.write_text(json.dumps({"completed_rows": stop, "rows": query_rows}, indent=2) + "\n")
            print(f"guidance cache {stop}/{query_rows}", flush=True)

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "predimem_guidance_training_cache_leave_one_trajectory_out_v1",
                "rows": query_rows,
                "source_rows": source_rows,
                "top_k": args.top_k,
                "search_k": args.search_k,
                "head": str(args.head.resolve()),
                "index": str(args.index.resolve()),
                "source_manifest": str(args.manifest.resolve()),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
