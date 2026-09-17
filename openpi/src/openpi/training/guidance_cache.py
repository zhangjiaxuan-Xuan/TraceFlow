"""Frame-addressed memory candidates for guidance-aware policy training."""

from pathlib import Path

import numpy as np
import torch


class GuidanceTrainingCache:
    """Read a sorted, pickle-free cache of per-frame memory blocks and weights."""

    REQUIRED_KEYS = ("episode_index", "frame_index", "blocks", "weights")

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Guidance training cache does not exist: {self.path}")
        if self.path.is_dir():
            arrays = {key: self.path / f"{key}.npy" for key in self.REQUIRED_KEYS}
            missing = [str(value) for value in arrays.values() if not value.is_file()]
            if missing:
                raise ValueError(f"Guidance cache is missing arrays: {missing}")
            self._archive = None
            loaded = {key: np.load(value, mmap_mode="r", allow_pickle=False) for key, value in arrays.items()}
        else:
            archive = np.load(self.path, mmap_mode="r", allow_pickle=False)
            missing = [key for key in self.REQUIRED_KEYS if key not in archive]
            if missing:
                raise ValueError(f"Guidance cache is missing arrays: {missing}")
            self._archive = archive
            loaded = {key: archive[key] for key in self.REQUIRED_KEYS}
        self.episode_index = np.asarray(loaded["episode_index"], dtype=np.int64)
        self.frame_index = np.asarray(loaded["frame_index"], dtype=np.int64)
        self.blocks = loaded["blocks"]
        self.weights = loaded["weights"]
        rows = self.episode_index.shape[0]
        if self.frame_index.shape != (rows,) or self.blocks.shape[0] != rows or self.weights.shape[0] != rows:
            raise ValueError("Guidance cache arrays have inconsistent row counts")
        if self.blocks.ndim != 4 or self.weights.shape != self.blocks.shape[:2]:
            raise ValueError("Expected blocks [N,K,H,A] and weights [N,K]")
        if np.any(self.episode_index < 0) or np.any(self.frame_index < 0):
            raise ValueError("Guidance cache episode/frame indices must be non-negative")
        if np.any(self.episode_index >= 2**31) or np.any(self.frame_index >= 2**32):
            raise ValueError("Guidance cache episode/frame indices exceed packed-key limits")
        self._keys = (self.episode_index.astype(np.uint64) << np.uint64(32)) | self.frame_index.astype(np.uint64)
        if rows > 1 and np.any(self._keys[1:] < self._keys[:-1]):
            raise ValueError("Guidance cache keys must be sorted by episode_index, frame_index")

    def lookup(self, metadata: dict[str, torch.Tensor], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        episodes = metadata["episode_index"].detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        frames = metadata["frame_index"].detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        if np.any(episodes < 0) or np.any(frames < 0):
            raise KeyError("Guidance cache lookup received a negative episode/frame index")
        query = (episodes.astype(np.uint64) << np.uint64(32)) | frames.astype(np.uint64)
        positions = np.searchsorted(self._keys, query)
        exact = positions < len(self._keys)
        if np.any(exact):
            exact[exact] &= self._keys[positions[exact]] == query[exact]
        if not np.all(exact):
            # Training data contains every frame while dense retrieval caches may
            # use a stride. Use the closest cached frame from the same episode.
            for index in np.flatnonzero(~exact):
                insertion = int(positions[index])
                candidates = [value for value in (insertion - 1, insertion) if 0 <= value < len(self._keys)]
                candidates = [value for value in candidates if int(self.episode_index[value]) == int(episodes[index])]
                if not candidates:
                    raise KeyError(
                        f"Guidance cache has no anchor for episode={int(episodes[index])}, frame={int(frames[index])}"
                    )
                positions[index] = min(
                    candidates,
                    key=lambda value: abs(int(self.frame_index[value]) - int(frames[index])),
                )
        blocks = torch.from_numpy(np.asarray(self.blocks[positions], dtype=np.float32)).to(device=device)
        weights = torch.from_numpy(np.asarray(self.weights[positions], dtype=np.float32)).to(device=device)
        return blocks, weights
