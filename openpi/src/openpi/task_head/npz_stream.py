from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import BinaryIO
import zipfile

import numpy as np


@dataclasses.dataclass(frozen=True)
class NpyHeader:
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool


def _member_name(key: str) -> str:
    return key if key.endswith(".npy") else f"{key}.npy"


def _read_header(stream: BinaryIO) -> NpyHeader:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY format version: {version}")
    return NpyHeader(tuple(int(value) for value in shape), np.dtype(dtype), bool(fortran_order))


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"NPY member ended with {remaining} bytes missing")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class NpzStreamReader:
    """Read NPZ metadata and individual C-order frames without inflating full arrays."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._archive = zipfile.ZipFile(self.path, mode="r")
        self.files = tuple(
            name[:-4] if name.endswith(".npy") else name for name in self._archive.namelist()
        )

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> NpzStreamReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def header(self, key: str) -> NpyHeader:
        with self._archive.open(_member_name(key), mode="r") as stream:
            return _read_header(stream)

    def array(self, key: str, *, max_bytes: int | None = None) -> np.ndarray:
        with self._archive.open(_member_name(key), mode="r") as stream:
            header = _read_header(stream)
            size = int(np.prod(header.shape, dtype=np.int64)) * header.dtype.itemsize
            if max_bytes is not None and size > int(max_bytes):
                raise ValueError(f"Refusing to inflate {key!r}: {size} bytes exceeds {max_bytes}")
            raw = _read_exact(stream, size)
        order = "F" if header.fortran_order else "C"
        return np.frombuffer(raw, dtype=header.dtype).reshape(header.shape, order=order).copy()

    def scalar(self, key: str):
        value = self.array(key, max_bytes=1024 * 1024)
        if value.shape != ():
            raise ValueError(f"NPZ field {key!r} must be scalar, got shape {value.shape}")
        return value.item()

    def frame(self, key: str, index: int) -> np.ndarray:
        with self._archive.open(_member_name(key), mode="r") as stream:
            header = _read_header(stream)
            if header.fortran_order:
                raise ValueError(f"Streaming frames requires a C-order array: {self.path}:{key}")
            if not header.shape:
                raise ValueError(f"Cannot select a frame from scalar field: {self.path}:{key}")
            frame_index = int(index)
            if frame_index < 0 or frame_index >= header.shape[0]:
                raise IndexError(f"frame_index={frame_index} is invalid for {self.path}:{key}{header.shape}")
            frame_shape = header.shape[1:]
            frame_size = int(np.prod(frame_shape, dtype=np.int64)) * header.dtype.itemsize
            skip = frame_index * frame_size
            while skip:
                chunk = stream.read(min(skip, 1024 * 1024))
                if not chunk:
                    raise EOFError(f"NPY member ended while seeking frame {frame_index}: {self.path}:{key}")
                skip -= len(chunk)
            raw = _read_exact(stream, frame_size)
        return np.frombuffer(raw, dtype=header.dtype).reshape(frame_shape).copy()
