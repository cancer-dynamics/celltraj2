"""Compact, reusable topology and frame-window references (zero-based rows).

Chains partition the observations: shared ancestry is never duplicated at a
division. Source graph edges remain in parent/child CSR arrays for inspection.
Population restrictions are applied after indexing, independently of topology.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class GraphIndex:
    order: np.ndarray
    offsets: np.ndarray
    parent: np.ndarray
    indptr: np.ndarray
    children: np.ndarray

    @classmethod
    def from_graph(cls, graph):
        n = graph.observation_count
        indptr = np.asarray(graph.adjacency.indptr, dtype=np.int64)
        children = np.asarray(graph.adjacency.indices, dtype=np.int64)
        degree = np.diff(indptr)
        parent = np.full(n, -1, dtype=np.int64)
        parent[children] = np.repeat(np.arange(n), degree)
        if not np.array_equal(parent + 1, graph.assignments['parent_observation_id']):
            raise ValueError('Graph parent assignments disagree with adjacency')
        roots = np.flatnonzero(parent < 0)
        # Traverse every edge once, including division edges, to reject cycles.
        pending = list(roots)
        visited = 0
        while pending:
            row = pending.pop()
            visited += 1
            pending.extend(children[indptr[row]:indptr[row + 1]])
        if visited != n:
            raise ValueError('Cycle detected in track graph')
        starts = np.flatnonzero((parent < 0) | (degree[np.maximum(parent, 0)] != 1))
        order = np.empty(n, dtype=np.int64)
        offsets = [0]
        cursor = 0
        for row in starts:
            while True:
                order[cursor] = row
                cursor += 1
                if degree[row] != 1:
                    break
                row = children[indptr[row]]
            offsets.append(cursor)
        return cls(order, np.asarray(offsets, dtype=np.int64), parent, indptr, children)

    def trace(self, row):
        """One selected history plus unambiguous future, computed on demand.

        Ancestors can cross a division along the selected child's path. Future
        traversal stops at the next division; it never chooses a child silently.
        """
        if row < 0 or row >= len(self.parent):
            raise ValueError('Observation row outside graph index')
        history = [row]
        current = row
        while self.parent[current] >= 0:
            current = int(self.parent[current])
            history.append(current)
        history.reverse()
        current = row
        while self.indptr[current+1] - self.indptr[current] == 1:
            current = int(self.children[self.indptr[current]])
            history.append(current)
        return np.asarray(history, dtype=np.int64)

    def project(self, source_to_snapshot, frames, eligible, scopes=None):
        """Disjoint runs; exclusions, gaps and scope switches always break a run.

        A dividing mother may end a chain; no delay or pair crosses a division.
        scopes may encode Type, split and group together. Never bridge a missing
        intermediate observation by filtering a chain and then rejoining it.
        """
        mapped = np.asarray(source_to_snapshot, dtype=np.int64)[self.order]
        present = mapped >= 0
        present[present] &= np.asarray(eligible)[mapped[present]]
        positions = np.flatnonzero(present)
        rows = mapped[positions]
        if not len(rows):
            return Runs(rows, np.asarray([0], dtype=np.int64))
        chain = np.searchsorted(self.offsets[1:], positions, side='right')
        cut = (np.diff(positions) != 1) | (np.diff(chain) != 0)
        cut |= np.diff(np.asarray(frames)[rows]) != 1
        if scopes is not None:
            values = np.asarray(scopes)[rows]
            cut |= values[1:] != values[:-1]
        return Runs(rows, np.r_[0, np.flatnonzero(cut) + 1, len(rows)].astype(np.int64))


@dataclass
class Runs:
    order: np.ndarray
    offsets: np.ndarray

    def windows(self, length=1):
        if isinstance(length, bool) or int(length) != length or length < 1:
            raise ValueError('Delay length must be a positive number of frames')
        length = int(length)
        positions = np.arange(len(self.order), dtype=np.int64)
        run = np.searchsorted(self.offsets[1:], positions, side='right')
        starts = positions[positions + length <= self.offsets[run + 1]]
        return FrameWindows(self, starts, length)


@dataclass
class FrameWindows:
    runs: Runs
    starts: np.ndarray
    length: int

    @property
    def anchors(self):
        return self.runs.order[self.starts + self.length - 1]

    def members(self, index):
        start = self.starts[index]
        return self.runs.order[start:start + self.length]

    def summarize(self, values, mode='final'):
        values = np.asarray(values, dtype=float)
        if mode in ('initial', 'final'):
            return values[self.runs.order[self.starts + (self.length - 1 if mode == 'final' else 0)]]
        if mode != 'mean':
            raise ValueError('Unknown delay summary')
        ordered = values[self.runs.order]
        finite = np.isfinite(ordered)
        sums = np.r_[0., np.cumsum(np.where(finite, ordered, 0.))]
        counts = np.r_[0, np.cumsum(finite)]
        ends = self.starts + self.length
        result = (sums[ends] - sums[self.starts]) / self.length
        result[counts[ends] - counts[self.starts] != self.length] = np.nan
        return result

    def batches(self, columns, *, batch_size=2048, indices=None, max_batch_bytes=32*1024**2):
        """Gather bounded chronological delay vectors only when requested."""
        width = len(columns) * self.length
        if width * 8 > max_batch_bytes:
            raise ValueError('One delay vector exceeds the fitting batch budget')
        size = min(batch_size, max(1, max_batch_bytes // max(1, width * 8)))
        chosen = np.arange(len(self.starts)) if indices is None else np.asarray(indices)
        for offset in range(0, len(chosen), size):
            ids = chosen[offset:offset + size]
            rows = self.runs.order[self.starts[ids, None] + np.arange(self.length)]
            yield ids, np.stack([np.asarray(c)[rows] for c in columns], axis=-1).reshape(len(ids), width)

    def pairs(self, times_s, *, lag_frames=1):
        """Window-index pairs in one run, with actual elapsed seconds.

        Unknown/nonpositive time does not erase frame pairs. Physical drift
        callers must explicitly require finite positive elapsed times.
        """
        if isinstance(lag_frames, bool) or int(lag_frames) != lag_frames or lag_frames < 1:
            raise ValueError('Pair lag must be positive frames')
        target = np.searchsorted(self.starts, self.starts + lag_frames)
        source = np.flatnonzero(target < len(self.starts))
        target = target[source]
        same = self.starts[target] == self.starts[source] + lag_frames
        runs = np.searchsorted(self.runs.offsets[1:], self.starts, side='right')
        same &= runs[source] == runs[target]
        source, target = source[same], target[same]
        anchors = self.anchors
        dt = np.asarray(times_s)[anchors[target]] - np.asarray(times_s)[anchors[source]]
        return source, target, dt
