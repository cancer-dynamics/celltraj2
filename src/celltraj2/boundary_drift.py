"""Directed boundary displacement, independent of storage and plotting.

The moving point is FROM. Its nearest OTHER object's TO boundary is measured
at the same source time. Positive displacement is toward that boundary. A
normal filter is applied *after* nearest-neighbor selection. Zero movement is
valid; absent motion and zero separation are not zero-valued measurements.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


VALIDITY_CODES = {
    0: 'valid', 1: 'invalid_coordinate', 2: 'missing_motion',
    3: 'missing_neighbor', 4: 'zero_separation', 5: 'outside_distance_domain',
    6: 'invalid_normal', 7: 'angle_filter', 8: 'invalid_weight',
    9: 'low_matched_fraction', 10: 'high_variance', 11: 'invalid_duration',
    12: 'interval_mismatch', 13: 'branch_or_merge', 14: 'gap',
    15: 'motion_quality', 16: 'unknown_cohort', 17: 'unsupported_registration',
    18: 'mixed_displacement_intervals', 19: 'unverified_weighting',
}


def _cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError('Boundary drift analysis cancelled')


def distance_bins(edges):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or not 2 <= len(edges) <= 100001 or
            not np.isfinite(edges).all() or edges[0] < 0 or
            np.any(np.diff(edges) <= 0)):
        raise ValueError('Distance edges must be finite, nonnegative, strictly increasing (at most 100000 bins)')
    return edges


class BoundaryTargetIndex:
    """One reusable, finite target point index for a frame and TO cohort.

    Query memory is bounded by 65536 candidate entries. Exceptionally large
    same-object neighborhoods fall back to a streaming single-point scan.
    Equidistant ties select the lowest target point ID deterministically.
    """
    def __init__(self, positions, object_ids, point_ids=None):
        positions = np.asarray(positions, dtype=float)
        object_ids = np.asarray(object_ids, dtype=np.int64)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError('Boundary positions must have shape (N, 2) or (N, 3)')
        if object_ids.shape != (len(positions),):
            raise ValueError('Target object IDs must align with positions')
        point_ids = np.arange(len(positions)) if point_ids is None else np.asarray(point_ids, dtype=np.int64)
        if point_ids.shape != object_ids.shape or len(np.unique(point_ids)) != len(point_ids):
            raise ValueError('Target point IDs must be unique and aligned')
        finite = np.isfinite(positions).all(axis=1)
        self.positions, self.object_ids, self.point_ids = positions[finite], object_ids[finite], point_ids[finite]
        self.original_rows = np.flatnonzero(finite)
        self.dimension = positions.shape[1]
        self.tree = cKDTree(self.positions) if len(self.positions) else None

    def nearest_other(self, positions, object_ids, *, cancelled=None):
        positions = np.asarray(positions, dtype=float)
        object_ids = np.asarray(object_ids, dtype=np.int64)
        result = np.full(len(positions), -1, dtype=np.int64)
        distance = np.full(len(positions), np.nan)
        if self.tree is None:
            return result, distance
        count = len(self.positions)
        pending = np.flatnonzero(np.isfinite(positions).all(axis=1))
        k = min(4, count)
        while pending.size:
            _cancel(cancelled)
            unresolved = []
            for start in range(0, len(pending), max(1, 65536 // k)):
                _cancel(cancelled)
                rows = pending[start:start + max(1, 65536 // k)]
                dd, ii = self.tree.query(positions[rows], k=k, workers=1)
                dd, ii = np.asarray(dd).reshape(len(rows), k), np.asarray(ii).reshape(len(rows), k)
                for row, ds, ix in zip(rows, dd, ii):
                    eligible = self.object_ids[ix] != object_ids[row]
                    if eligible.any():
                        minimum = ds[eligible].min()
                        # Resolve all exact distance ties before assigning identity/direction.
                        if k == count or ds[-1] > minimum:
                            ties = ix[eligible & (ds == minimum)]
                            winner = ties[np.argmin(self.point_ids[ties])]
                            result[row], distance[row] = winner, minimum
                            continue
                    if k < count:
                        unresolved.append(row)
            pending = np.asarray(unresolved, dtype=np.int64)
            if not len(pending):
                break
            if k >= 65536:
                for row in pending:
                    best, winner = np.inf, -1
                    for start in range(0, count, 65536):
                        _cancel(cancelled)
                        stop = min(start + 65536, count)
                        ds = np.linalg.norm(self.positions[start:stop] - positions[row], axis=1)
                        ds[self.object_ids[start:stop] == object_ids[row]] = np.inf
                        minimum = ds.min()
                        if np.isfinite(minimum) and minimum <= best:
                            ties = np.flatnonzero(ds == minimum) + start
                            candidate = ties[np.argmin(self.point_ids[ties])]
                            if minimum < best or winner < 0 or self.point_ids[candidate] < self.point_ids[winner]:
                                best, winner = minimum, candidate
                    if winner >= 0:
                        result[row], distance[row] = winner, best
                break
            k = min(count, 2 * k, 65536)
        return result, distance


def summarize_boundary_samples(distance, values, weights, validity, distance_edges, *, min_pairs=2):
    """Finite weighted sufficient statistics; support counts are not biological n."""
    edges = distance_bins(distance_edges)
    if int(min_pairs) < 1:
        raise ValueError('min_pairs must be at least one')
    distance, values, weights = (np.asarray(v, dtype=float) for v in (distance, values, weights))
    bins = np.searchsorted(edges, distance, side='right') - 1
    bins[distance == edges[-1]] = len(edges) - 2
    valid = ((np.asarray(validity) == 0) & np.isfinite(values) & np.isfinite(distance) &
             np.isfinite(weights) & (weights > 0) & (bins >= 0) & (bins < len(edges) - 1))
    n = len(edges) - 1
    b, v, w = bins[valid], values[valid], weights[valid]
    counts = np.bincount(b, minlength=n).astype(np.int64)
    sums = [np.bincount(b, weights=entry, minlength=n) for entry in (w, w*v, w*v*v, w*w)]
    means = np.divide(sums[1], sums[0], out=np.full(n, np.nan), where=(counts >= min_pairs) & (sums[0] > 0))
    status = np.full(n, 'ok', dtype='U20')
    status[counts < min_pairs] = 'insufficient_support'
    status[counts == 0] = 'no_pairs'
    return dict(distance_edges=edges, values=means, counts=counts, weight_sums=sums[0],
                weighted_sums=sums[1], weighted_square_sums=sums[2], squared_weight_sums=sums[3], statuses=status)


def project_boundary_drift(source_positions, displacements, source_object_ids, target_index, *,
        distance_edges, weights=None, normals=None, angle_min=None, validity=None,
        min_pairs=2, cancelled=None):
    """Project forward FROM motion onto the nearest-other TO direction.

    ``angle_min`` is the minimum outward-normal cosine in [-1,1]. Prior
    validity codes are retained, so absent/low-quality motion never becomes 0.
    The returned target_rows address the supplied index's original input rows.
    """
    _cancel(cancelled)
    edges = distance_bins(distance_edges)
    p, u = np.asarray(source_positions, dtype=float), np.asarray(displacements, dtype=float)
    ids = np.asarray(source_object_ids, dtype=np.int64)
    n = len(p)
    if p.ndim != 2 or p.shape[1] != target_index.dimension or u.shape != p.shape or ids.shape != (n,):
        raise ValueError('Moving boundary positions, vectors, and object IDs must align')
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    reason = np.zeros(n, dtype=np.int16) if validity is None else np.asarray(validity, dtype=np.int16).copy()
    if w.shape != (n,) or reason.shape != (n,):
        raise ValueError('Weights and validity must align with moving points')
    def reject(mask, code):
        reason[(reason == 0) & mask] = code
    reject(~np.isfinite(p).all(axis=1), 1)
    reject(~np.isfinite(u).all(axis=1), 2)
    reject(~np.isfinite(w) | (w <= 0), 8)
    target, distance = target_index.nearest_other(p, ids, cancelled=cancelled)
    reject(target < 0, 3)
    reject(distance == 0, 4)
    reject((distance < edges[0]) | (distance > edges[-1]), 5)
    direction = np.full_like(p, np.nan)
    found = (target >= 0) & (distance > 0)
    direction[found] = (target_index.positions[target[found]] - p[found]) / distance[found, None]
    if angle_min is not None:
        if not np.isfinite(angle_min) or not -1 <= angle_min <= 1:
            raise ValueError('angle_min must be a cosine between -1 and 1')
        if normals is None:
            raise ValueError('An enabled normal-angle filter requires saved normals')
        normals = np.asarray(normals, dtype=float)
        if normals.shape != p.shape:
            raise ValueError('Normals must align with moving points')
        norm = np.linalg.norm(normals, axis=1)
        reject(~np.isfinite(norm) | (norm <= 0), 6)
        cosine = np.divide(np.sum(normals * direction, axis=1), norm, out=np.full(n, np.nan), where=norm > 0)
        reject(cosine < angle_min, 7)
    projected = np.einsum('ij,ij->i', u, direction)
    projected[reason != 0] = np.nan
    target_rows = np.full(n, -1, dtype=np.int64)
    target_rows[target >= 0] = target_index.original_rows[target[target >= 0]]
    return {**summarize_boundary_samples(distance, projected, w, reason, edges, min_pairs=min_pairs),
            'distance': distance, 'displacement': projected, 'weight': w,
            'validity': reason, 'target_rows': target_rows}
