"""Private append-only scratch storage for surface transport results."""
from __future__ import annotations

from pathlib import Path


class MotionSpool:
    """Buffer at most one link beyond the byte budget; flush at frame changes.

    The caller owns the lifetime. This file is never a canonical analysis H5.
    """
    def __init__(self, path: Path, *, buffer_bytes: int = 8 * 1024 * 1024):
        import h5py
        self.h5 = h5py.File(path, "w")
        self.buffer_bytes = int(buffer_bytes)
        self.buffer = {}
        self.pending_bytes = 0
        self.peak_buffer_bytes = 0
        self.flush_count = 0
        self.frame = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.buffer.clear()
        self.h5.close()

    def append(self, frame, transport, source_summary, target_summary):
        if self.frame is not None and self.frame != frame:
            self.flush()
        self.frame = frame
        groups = {"transport": transport, "source_summary": source_summary, "target_summary": target_summary}
        incoming = sum(value.nbytes for cols in groups.values() for value in cols.values())
        if self.pending_bytes + incoming > self.buffer_bytes:
            self.flush()
        for group, columns in groups.items():
            for key, value in columns.items():
                self.buffer.setdefault(f"{group}/{key}", []).append(value)
        self.pending_bytes += incoming
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.pending_bytes)
        if self.pending_bytes >= self.buffer_bytes:
            self.flush()

    def flush(self):
        # Append one block at a time: no duplicate concatenated buffer.
        for key, blocks in self.buffer.items():
            first = blocks[0]
            if key not in self.h5:
                self.h5.create_dataset(key, shape=(0, *first.shape[1:]),
                                       maxshape=(None, *first.shape[1:]), dtype=first.dtype,
                                       chunks=True)
            dataset = self.h5[key]
            start = dataset.shape[0]
            dataset.resize(start + sum(value.shape[0] for value in blocks), axis=0)
            for value in blocks:
                end = start + value.shape[0]
                dataset[start:end] = value
                start = end
        if self.buffer:
            self.flush_count += 1
            self.buffer.clear()
            self.pending_bytes = 0
            self.h5.flush()

    def finish(self, empty_transport):
        self.flush()
        for key, value in empty_transport.items():
            if f"transport/{key}" not in self.h5:
                self.h5.create_dataset(f"transport/{key}", data=value)
        def columns(group):
            return dict(self.h5[group].items()) if group in self.h5 else {}
        return columns("transport"), {name: columns(name) for name in ("source_summary", "target_summary")}
