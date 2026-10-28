"""Bounded copies of numeric arrays, including private disk-backed results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

COPY_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class H5Array:
    """A closed-file array reference used by deferred commits; slicing only."""

    path: Path
    key: str
    shape: tuple[int, ...]
    dtype: Any

    def __getitem__(self, selection):
        import h5py
        with h5py.File(self.path, "r") as h5:
            return h5[self.key][selection]


def is_disk_array(value: Any) -> bool:
    import h5py
    return isinstance(value, (h5py.Dataset, H5Array))


def copy_array(group, name, values, *, chunk_bytes=COPY_BYTES):
    """Create a compressed dataset without asking a lazy source for a full array."""
    import numpy as np
    shape, dtype = values.shape, values.dtype
    if not shape:
        return group.create_dataset(name, data=values[()])
    dataset = group.create_dataset(name, shape=shape, dtype=dtype, compression="gzip", shuffle=True)
    row_bytes = max(1, int(np.prod(shape[1:])) * dtype.itemsize)
    rows = max(1, int(chunk_bytes) // row_bytes)
    for start in range(0, shape[0], rows):
        dataset[start:start + rows] = values[start:start + rows]
    return dataset
