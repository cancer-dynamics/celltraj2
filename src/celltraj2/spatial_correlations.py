"""Directed finite-pair motion surfaces within one supplied coordinate system.

The caller resolves clocks, calibration, registration, topology, and inference
units. This module never joins motion intervals across supplied runs. Pair
support is not a count of biological replicates.
"""

import numpy as np
from scipy.spatial import cKDTree

from .temporal_correlations import _matrix, _validate_runs

_MAX_SURFACE_BINS = 100000
DEFAULT_MAX_SUPPORT_ENTRIES = 20_000_000


class SpatialSupportLimitExceeded(ValueError):
    """Exact support accounting reached the caller's configured entry budget."""


class SpatialCorrelationCancelled(RuntimeError):
    """Raised when a spatial calculation is cooperatively cancelled."""


def _check_cancelled(cancelled):
    if cancelled is not None:
        requested = cancelled() if callable(cancelled) else cancelled.is_set()
        if requested:
            raise SpatialCorrelationCancelled("Spatial correlation calculation cancelled")


def _edges(values, name, nonnegative=False):
    edges = np.asarray(values, dtype=float)
    if (edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0) or (nonnegative and edges[0] < 0)):
        raise ValueError(f"{name} must have finite, strictly increasing"
                         + (", nonnegative" if nonnegative else "") + " edges")
    return edges


def _mask(values, size, name):
    if values is None:
        return np.ones(size, dtype=bool)
    result = np.asarray(values)
    if result.shape != (size,) or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean observation array")
    return result


def _ids(values, order, offsets, size, name):
    result = np.asarray(values)
    if (result.shape != (size,) or not np.issubdtype(result.dtype, np.integer)
            or np.any(result[order] < 0)):
        raise ValueError(f"{name} must be nonnegative integer IDs on selected observations")
    for start, stop in zip(offsets[:-1], offsets[1:]):
        ids = result[order[start:stop]]
        if np.any(ids != ids[0]):
            raise ValueError(f"{name} must be constant within each supplied run")
    return result


def _bin_indices(edges, values):
    bins = np.searchsorted(edges, values, side="right") - 1
    bins[values == edges[-1]] = len(edges) - 2
    return bins


def _grouped_moments(a, b, groups, first, number):
    """Stable, jointly centered trace moments for all occupied bins at once.

    Subtract a representative pair before summing, retaining small variations
    on large baselines. A second centering pass avoids raw-square cancellation.
    """
    mean_a = a[first].copy()
    mean_b = b[first].copy()
    for component in range(a.shape[1]):
        mean_a[:, component] += np.bincount(
            groups, weights=a[:, component] - mean_a[groups, component],
        ) / number
        mean_b[:, component] += np.bincount(
            groups, weights=b[:, component] - mean_b[groups, component],
        ) / number
    da, db = a - mean_a[groups], b - mean_b[groups]
    return (mean_a, mean_b,
            np.bincount(groups, weights=np.einsum("ij,ij->i", da, da)),
            np.bincount(groups, weights=np.einsum("ij,ij->i", db, db)),
            np.bincount(groups, weights=np.einsum("ij,ij->i", da, db)))


def compute_motion_surface(
    runs, times, positions, *, distance_edges, lag_edges,
    displacement_interval, interval_tolerance=0, method="radial_response",
    normalization="raw", distance_convention="cross_time", radial_sign="toward",
    transverse_mode="sum", source_mask=None, receiver_mask=None, cell_ids=None,
    lineage_ids=None, exclude_same_cell=True, exclude_related_lineage=False,
    min_pairs=2, progress=None, cancelled=None, pair_search="auto",
    max_support_entries=DEFAULT_MAX_SUPPORT_ENTRIES,
):
    """Compute motion versus directed origin lag and pair separation.

    Rows of ``positions`` and ``times`` are observation aligned. ``runs`` is a
    disjoint :class:`celltraj2.trajectory_index.Runs`, usually projected from a
    graph with gaps and divisions cut. A displacement is **forward**, from its
    origin at t to the unique nearest later observation to t+delta, provided its
    elapsed-time error is at most ``interval_tolerance``. Exact ties are rejected;
    A machine-roundoff allowance of 8 eps * max(1, |origin time|, |delta|) is
    added to the explicit tolerance and used to recognize ambiguous ties. Thus
    zero tolerance accepts nominally exact matches on decimal clocks. There is
    no interpolation.
    Supplied run clocks must be finite and strictly increasing.

    Positive lag is receiver origin time minus source origin time. All directed
    eligible pairs are weighted equally. Axes have shape (lag, distance). Bins
    are left closed, right open, with the final right edge included. Actual mean
    lags, distances, and realized source/receiver displacement intervals are
    reported separately from nominal centers and delta.

    ``cross_time`` separation uses the two origin positions. ``source_time``
    uses the receiver's position at the **exact source clock**, in the same run
    as its motion; it does not choose the nearest frame or interpolate. The axis
    points from that receiver reference position toward the source position.
    ``radial_sign='toward'`` makes approach positive; ``'away'`` reverses radial
    values. Zero separation is excluded for every method with an explicit count.

    * ``radial_response``: receiver displacement projected on the pair axis.
      Only the source position is used; a singleton source can contribute.
    * ``relative_radial``: (receiver - source displacement) projected on the axis.
    * ``full_dot``: source dot receiver displacement.
    * ``longitudinal``: product of their projections on the same pair axis.
    * ``transverse``: dot product of their orthogonal vector projections. ``sum``
      sums the D-1 transverse components; ``mean`` divides by D-1.

    Radial methods support raw means. Two-vector methods support raw products,
    bin-local joint-pair covariance, and Pearson correlation using the projected
    vector trace variances on those same pairs. This covariance is centered
    within each distance/lag bin, not against a fixed ROI reference. Coefficients
    of different projections have different denominators; they do not add up to
    the full-vector coefficient. Only the raw longitudinal + transverse sum is
    guaranteed to equal the full dot product; centered projections need not
    decompose when axes vary across pairs. Zero motion remains a valid raw observation;
    zero variance makes normalized values undefined.

    Cohort masks apply independently. All observations along a used displacement
    interval must remain in that role's cohort, even when both endpoints match.
    For source-time geometry the receiver cohort must also hold along the path
    joining its reference position and motion interval. A radial source only
    requires source membership at its anchor. Finite positions are required at
    every used endpoint/reference; an unused source endpoint is never required.

    ``cell_ids`` are original cell/chain IDs, constant within each run but allowed
    to repeat across split runs; they drive same-cell exclusion and track support.
    When absent, supplied run IDs are used and that fallback is recorded. Optional
    ``lineage_ids`` identify connected lineages; excluding related lineages
    requires these nonnegative integer IDs. IDs outside selected runs are ignored.

    ``pair_search`` selects ``auto`` (spatial index when useful),
    ``spatial_index``, or ``brute_force``. This is an execution setting only:
    candidate pruning is exact and all finite-pair/moment/support definitions
    are identical. Extremely large coordinates use the overflow-safe direct
    geometry calculation. Receiver references and lag assignments are reused
    across sources at the same clock; no references persist across calls.

    Scratch pair arrays are bounded chunks, never an N by N matrix. Support sets
    retain distinct contributing receiver origins and cell IDs per occupied bin,
    so their memory scales with unique support, not pair multiplicity. Work still
    scales with accepted neighbor pairs and unique support. At most 100,000 surface bins
    and 2,000,000 bin-component entries are allowed. ``max_support_entries``
    limits the combined source-cell, receiver-cell and receiver-origin set
    memberships across all bins in this call (default 20,000,000; 0 = unlimited).
    The same identity in several bins counts separately. This is an entry budget,
    not a measurement of available RAM; concurrent calls each have their own
    budget. Reaching it raises SpatialSupportLimitExceeded without truncating
    support or returning a partial surface. Changing it does not change estimates.
    ``progress`` receives strings;
    ``cancelled`` is a predicate or event and raises SpatialCorrelationCancelled.
    """
    _check_cancelled(cancelled)
    if (isinstance(max_support_entries, (bool, np.bool_))
            or not isinstance(max_support_entries, (int, np.integer)) or max_support_entries < 0):
        raise ValueError("max_support_entries must be a nonnegative integer (0 = unlimited)")
    if pair_search not in {"auto", "spatial_index", "brute_force"}:
        raise ValueError("pair_search must be auto, spatial_index or brute_force")
    radial = method in {"radial_response", "relative_radial"}
    needs_source_motion = method != "radial_response"
    if method not in {"radial_response", "relative_radial", "full_dot", "longitudinal", "transverse"}:
        raise ValueError("Unknown spatial motion method")
    if normalization not in {"raw", "covariance", "coefficient"} or (radial and normalization != "raw"):
        raise ValueError("Radial methods require raw normalization; vector methods support raw/covariance/coefficient")
    if distance_convention not in {"cross_time", "source_time"}:
        raise ValueError("distance_convention must be cross_time or source_time")
    if radial_sign not in {"toward", "away"}:
        raise ValueError("radial_sign must be toward or away")
    if transverse_mode not in {"sum", "mean"}:
        raise ValueError("transverse_mode must be sum or mean")
    if not np.isfinite(displacement_interval) or displacement_interval <= 0:
        raise ValueError("displacement_interval must be finite and positive")
    if not np.isfinite(interval_tolerance) or interval_tolerance < 0:
        raise ValueError("interval_tolerance must be finite and nonnegative")
    if isinstance(min_pairs, (bool, np.bool_)) or int(min_pairs) != min_pairs or min_pairs < 1:
        raise ValueError("min_pairs must be a positive integer")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional observation array")
    positions = _matrix(positions, len(times), "positions")
    dimension = positions.shape[1]
    if method == "transverse" and dimension < 2:
        raise ValueError("Transverse motion requires at least two position components")
    r_edges = _edges(distance_edges, "distance_edges", nonnegative=True)
    t_edges = _edges(lag_edges, "lag_edges")
    shape = (len(t_edges) - 1, len(r_edges) - 1)
    bins = shape[0] * shape[1]
    moment_dimension = 1 if radial or method == "longitudinal" else dimension
    if bins > _MAX_SURFACE_BINS or bins * moment_dimension > 2000000:
        raise ValueError("Surface grid exceeds the bounded accumulator limit; use fewer bins/components")
    order, offsets = _validate_runs(runs, times)
    size = len(times)
    source_mask = _mask(source_mask, size, "source_mask")
    receiver_mask = _mask(receiver_mask, size, "receiver_mask")
    batch_size = min(4096, max(1, 1024 * 1024 // (8 * dimension * 12)))
    run_ids = np.full(size, -1, dtype=np.int64)
    local_ids = np.full(size, -1, dtype=np.int64)
    endpoint_rows = np.full(size, -1, dtype=np.int64)
    endpoint_local = np.full(size, -1, dtype=np.int64)
    endpoint_reason = np.full(size, "missing_endpoint", dtype="U24")
    motion = np.full(positions.shape, np.nan)
    interval = np.full(size, np.nan)
    source_ok = np.zeros(size, dtype=bool)
    receiver_ok = np.zeros(size, dtype=bool)
    run_rows, run_clocks, receiver_bad_prefixes = [], [], []
    exclusion_counts = {}

    def excluded(name, count):
        exclusion_counts[name] = exclusion_counts.get(name, 0) + int(count)

    finite_position = np.all(np.isfinite(positions), axis=1)
    for run, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        _check_cancelled(cancelled)
        rows = order[start:stop]
        clock = times[rows]
        run_rows.append(rows)
        run_clocks.append(clock)
        run_ids[rows] = run
        local_ids[rows] = np.arange(len(rows))
        source_bad = np.r_[0, np.cumsum(~source_mask[rows])]
        receiver_bad = np.r_[0, np.cumsum(~receiver_mask[rows])]
        receiver_bad_prefixes.append(receiver_bad)
        if progress is not None:
            progress(f"Spatial motion endpoints: run {run + 1}/{len(offsets) - 1}")
        for begin in range(0, len(rows), batch_size):
            _check_cancelled(cancelled)
            origins = np.arange(begin, min(begin + batch_size, len(rows)))
            origin_rows = rows[origins]
            with np.errstate(over="ignore", invalid="ignore"):
                right = np.maximum(np.searchsorted(clock, clock[origins] + displacement_interval), origins + 1)
                left = right - 1
                left_valid = (left > origins) & (left < len(rows))
                right_valid = (right > origins) & (right < len(rows))
                left = np.clip(left, 0, len(rows) - 1)
                right = np.clip(right, 0, len(rows) - 1)
                left_error = np.where(left_valid, np.abs(clock[left] - clock[origins] - displacement_interval), np.inf)
                right_error = np.where(right_valid, np.abs(clock[right] - clock[origins] - displacement_interval), np.inf)
            best_error = np.minimum(left_error, right_error)
            roundoff = 8 * np.finfo(float).eps * np.maximum(np.maximum(np.abs(clock[origins]), abs(displacement_interval)), 1.0)
            close = np.isfinite(best_error) & (best_error <= interval_tolerance + roundoff)
            with np.errstate(invalid="ignore"):
                tied = close & np.isfinite(left_error) & np.isfinite(right_error) & (np.abs(left_error - right_error) <= roundoff)
            endpoint_reason[origin_rows[tied]] = "ambiguous_endpoint"
            matched = close & ~tied
            ends = np.where(left_error < right_error, left, right)[matched]
            matched_rows = origin_rows[matched]
            endpoint_rows[matched_rows] = rows[ends]
            endpoint_local[matched_rows] = ends
            endpoint_reason[matched_rows] = "ok"
            interval[matched_rows] = clock[ends] - times[matched_rows]
            with np.errstate(over="ignore", invalid="ignore"):
                motion[matched_rows] = positions[rows[ends]] - positions[matched_rows]
            finite_motion = np.all(np.isfinite(motion[matched_rows]), axis=1)
            starts = origins[matched]
            source_continuous = source_bad[ends + 1] == source_bad[starts]
            receiver_continuous = receiver_bad[ends + 1] == receiver_bad[starts]
            source_ok[matched_rows] = source_continuous & finite_motion
            receiver_ok[matched_rows] = receiver_continuous & finite_motion
            for role, mask, continuous in (("source", source_mask, source_continuous), ("receiver", receiver_mask, receiver_continuous)):
                if role == "source" and not needs_source_motion:
                    continue
                excluded(role + "_cohort_crossing", np.count_nonzero(mask[matched_rows] & ~continuous))
                excluded(role + "_nonfinite_motion", np.count_nonzero(continuous & ~finite_motion))
        for role, mask in (("source", source_mask), ("receiver", receiver_mask)):
            excluded(role + "_masked_origins", np.count_nonzero(~mask[rows]))
            if role == "source" and not needs_source_motion:
                continue
            for reason in ("missing_endpoint", "ambiguous_endpoint"):
                excluded(role + "_" + reason, np.count_nonzero(mask[rows] & (endpoint_reason[rows] == reason)))
    if not needs_source_motion:
        source_ok[order] = source_mask[order] & finite_position[order]
        excluded("source_nonfinite_position", np.count_nonzero(source_mask[order] & ~finite_position[order]))
    identity_definition = "provided_cell_ids" if cell_ids is not None else "supplied_disjoint_runs"
    cells = run_ids if cell_ids is None else _ids(cell_ids, order, offsets, size, "cell_ids")
    lineages = None if lineage_ids is None else _ids(lineage_ids, order, offsets, size, "lineage_ids")
    if exclude_related_lineage and lineages is None:
        raise ValueError("exclude_related_lineage requires lineage_ids")

    counts = np.zeros(bins, dtype=np.int64)
    source_counts = np.zeros(bins, dtype=np.int64)
    sums, lag_sums, distance_sums = (np.zeros(bins) for _ in range(3))
    source_interval_sums, receiver_interval_sums = (np.zeros(bins) for _ in range(2))
    means_a, means_b = (np.zeros((bins, moment_dimension)) for _ in range(2))
    m2_a, m2_b, cross_m2 = (np.zeros(bins) for _ in range(3))
    source_cells, receiver_cells, receiver_origins = {}, {}, {}
    support_entries = 0

    def add_support(mapping, ibin, identities):
        nonlocal support_entries
        members = mapping.setdefault(int(ibin), set())
        additions = {int(x) for x in identities}.difference(members)
        required = support_entries + len(additions)
        if max_support_entries and required > max_support_entries:
            raise SpatialSupportLimitExceeded(
                f"Spatial support membership limit exceeded: at least {required:,} combined "
                f"bin memberships needed; configured limit {max_support_entries:,} per surface. "
                "Increase max_support_entries or set it to 0 for unlimited support accounting")
        members.update(additions)
        support_entries += len(additions)

    # Accepted pairs from many sources share one reduction. This avoids a tiny
    # Python/moment operation for every occupied bin of every source origin.
    pending, pending_count = [], 0
    last_source = np.full(bins, -1, dtype=np.int64)

    def flush_pairs():
        nonlocal pending_count
        if not pending:
            return
        _check_cancelled(cancelled)
        a, b, targets, origins, lag, distance, combined_bins = (
            np.concatenate(parts) for parts in zip(*pending)
        )
        occupied, first, groups, number = np.unique(
            combined_bins, return_index=True, return_inverse=True, return_counts=True,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            ca, cb, va, vb, cab = _grouped_moments(a, b, groups, first, number)
            previous = counts[occupied]
            total = previous + number
            shift_a, shift_b = ca - means_a[occupied], cb - means_b[occupied]
            weight = previous * (number / total)
            m2_a[occupied] += va + np.einsum("ij,ij->i", shift_a, shift_a) * weight
            m2_b[occupied] += vb + np.einsum("ij,ij->i", shift_b, shift_b) * weight
            cross_m2[occupied] += cab + np.einsum("ij,ij->i", shift_a, shift_b) * weight
            means_a[occupied] += shift_a * (number / total)[:, None]
            means_b[occupied] += shift_b * (number / total)[:, None]
            sums[occupied] += np.bincount(groups, weights=np.einsum("ij,ij->i", a, b))
        counts[occupied] += number
        lag_sums[occupied] += np.bincount(groups, weights=lag)
        distance_sums[occupied] += np.bincount(groups, weights=distance)
        if needs_source_motion:
            source_interval_sums[occupied] += np.bincount(groups, weights=interval[source_rows[origins]])
        receiver_interval_sums[occupied] += np.bincount(groups, weights=interval[targets])
        # Stable grouping preserves source traversal order, allowing exact
        # unique source counts with one last-source integer per bin.
        grouping = np.argsort(groups, kind="stable")
        bounds = np.r_[0, np.cumsum(number)]
        for index, ibin in enumerate(occupied):
            selected = grouping[bounds[index]:bounds[index + 1]]
            source_numbers = origins[selected]
            source_counts[ibin] += int(source_numbers[0] != last_source[ibin]) + np.count_nonzero(np.diff(source_numbers))
            last_source[ibin] = source_numbers[-1]
            add_support(source_cells, ibin, cells[source_rows[source_numbers]])
            add_support(receiver_cells, ibin, cells[targets[selected]])
            add_support(receiver_origins, ibin, targets[selected])
        pending.clear()
        pending_count = 0

    receiver_rows = order[receiver_ok[order]]
    receiver_rows = receiver_rows[np.argsort(times[receiver_rows], kind="stable")]
    receiver_clock = times[receiver_rows]
    source_rows = order[source_ok[order]]
    # Clock order lets source-time reference searches be reused across all
    # sources in a frame, without retaining references for the whole movie.
    source_rows = source_rows[np.argsort(times[source_rows], kind="stable")]
    cache_clock = None
    sign = 1.0 if radial_sign == "toward" else -1.0

    def rejection_histogram(identities, categories):
        identifiers, inverse = np.unique(identities, return_inverse=True, axis=0)
        grouped = np.bincount(inverse * 5 + categories, minlength=len(identifiers) * 5).reshape(-1, 5)
        return {(tuple(identity) if np.ndim(identity) else int(identity)): row
                for identity, row in zip(identifiers, grouped)}

    def frame_candidates(source_time):
        """One receiver lookup/index per source clock, with O(observations) memory.

        Category histograms retain the original *sequential* rejection counts
        even when distant pairs never enter the numerical geometry calculation.
        """
        lower = np.searchsorted(receiver_clock, np.nextafter(source_time + t_edges[0], -np.inf), side="left")
        upper = np.searchsorted(receiver_clock, np.nextafter(source_time + t_edges[-1], np.inf), side="right")
        targets = receiver_rows[lower:upper]
        lag = times[targets] - source_time
        lag_bin = _bin_indices(t_edges, lag)
        category = np.where((lag_bin >= 0) & (lag_bin < shape[0]), 4, 0).astype(np.int8)
        reference_rows = targets.copy()
        if distance_convention == "source_time" and len(targets):
            target_runs = run_ids[targets]
            grouping = np.argsort(target_runs, kind="stable")
            grouped_runs = target_runs[grouping]
            bounds = np.r_[0, np.flatnonzero(np.diff(grouped_runs)) + 1, len(targets)]
            for start, stop in zip(bounds[:-1], bounds[1:]):
                _check_cancelled(cancelled)
                selected = grouping[start:stop]
                selected = selected[category[selected] == 4]
                if not len(selected):
                    continue
                run = int(grouped_runs[start])
                reference = int(np.searchsorted(run_clocks[run], source_time))
                if reference == len(run_clocks[run]) or run_clocks[run][reference] != source_time:
                    category[selected] = 1
                    continue
                reference_rows[selected] = run_rows[run][reference]
                begin_path = np.minimum(local_ids[targets[selected]], reference)
                end_path = np.maximum(endpoint_local[targets[selected]], reference)
                prefix = receiver_bad_prefixes[run]
                continuous = prefix[end_path + 1] == prefix[begin_path]
                category[selected[~continuous]] = 2
        category[(category == 4) & ~finite_position[reference_rows]] = 3
        totals = np.bincount(category, minlength=5)
        by_cell = rejection_histogram(cells[targets], category) if exclude_same_cell else {}
        by_lineage = rejection_histogram(lineages[targets], category) if exclude_related_lineage else {}
        by_both = rejection_histogram(np.c_[cells[targets], lineages[targets]], category) if exclude_same_cell and exclude_related_lineage else {}
        candidate_indices = np.flatnonzero(category == 4)
        tree = None
        if (pair_search != "brute_force" and len(candidate_indices)
                and (pair_search == "spatial_index" or (len(candidate_indices) >= 64 and dimension <= 3))
                and r_edges[-1] < 1e100
                and np.all(np.abs(positions[reference_rows[candidate_indices]]) < 1e100)):
            _check_cancelled(cancelled)
            tree = cKDTree(positions[reference_rows[candidate_indices]])
        return (targets, lag, lag_bin, reference_rows, totals, by_cell,
                by_lineage, by_both, candidate_indices, tree)

    for origin_number, source_row in enumerate(source_rows):
        _check_cancelled(cancelled)
        if progress is not None and origin_number % 256 == 0:
            progress(f"Spatial motion pairs: {origin_number}/{len(source_rows)} source origins")
        source_time = times[source_row]
        if source_time != cache_clock:
            cached = frame_candidates(source_time)
            cache_clock = source_time
        (frame_targets, frame_lag, frame_lag_bin, frame_references, totals,
         by_cell, by_lineage, by_both, candidate_indices, tree) = cached
        if not len(frame_targets):
            continue
        excluded("pair_outside_lag", totals[0])
        remaining = totals.copy()
        if exclude_same_cell:
            same_counts = by_cell.get(int(cells[source_row]), np.zeros(5, dtype=np.int64))
            excluded("pair_same_cell", np.sum(same_counts[1:]))
            remaining -= same_counts
        if exclude_related_lineage:
            related_counts = by_lineage.get(int(lineages[source_row]), np.zeros(5, dtype=np.int64)).copy()
            if exclude_same_cell:
                related_counts -= by_both.get((int(cells[source_row]), int(lineages[source_row])), np.zeros(5, dtype=np.int64))
            excluded("pair_related_lineage", np.sum(related_counts[1:]))
            remaining -= related_counts
        if distance_convention == "source_time":
            excluded("pair_missing_source_time_reference", remaining[1])
            excluded("pair_receiver_reference_cohort_crossing", remaining[2])
        excluded("pair_nonfinite_geometry", remaining[3])
        query_indices = candidate_indices
        indexed = tree is not None and np.all(np.abs(positions[source_row]) < 1e100)
        if indexed:
            # A slightly expanded Chebyshev ball is a conservative Euclidean
            # candidate set. Exact hypot/bin checks below decide every boundary.
            neighbors = tree.query_ball_point(positions[source_row], np.nextafter(r_edges[-1], np.inf),
                                             p=np.inf, return_sorted=True)
            query_indices = candidate_indices[neighbors]
        kept_neighbors = 0
        for begin in range(0, len(query_indices), batch_size):
            _check_cancelled(cancelled)
            selected = query_indices[begin:begin + batch_size]
            targets = frame_targets[selected]
            valid = np.ones(len(targets), dtype=bool)
            if exclude_same_cell:
                valid &= cells[targets] != cells[source_row]
            if exclude_related_lineage:
                valid &= lineages[targets] != lineages[source_row]
            selected = selected[valid]
            targets, lag, lag_bin, reference_rows = (x[selected] for x in
                (frame_targets, frame_lag, frame_lag_bin, frame_references))
            kept_neighbors += len(targets)
            if not len(targets):
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                separation = positions[source_row] - positions[reference_rows]
                distance = np.hypot.reduce(separation, axis=1)
            finite = np.all(np.isfinite(separation), axis=1) & np.isfinite(distance)
            excluded("pair_nonfinite_geometry", np.count_nonzero(~finite))
            zero = distance == 0
            excluded("pair_zero_separation", np.count_nonzero(finite & zero))
            rbin = _bin_indices(r_edges, distance)
            inside = (rbin >= 0) & (rbin < shape[1])
            excluded("pair_outside_distance", np.count_nonzero(finite & ~zero & ~inside))
            valid = finite & ~zero & inside
            targets, lag, lag_bin, distance, separation, rbin = (
                x[valid] for x in (targets, lag, lag_bin, distance, separation, rbin)
            )
            if not len(targets):
                continue
            axis = separation / distance[:, None]
            b = motion[targets]
            with np.errstate(over="ignore", invalid="ignore"):
                if radial:
                    displacement = b if not needs_source_motion else b - motion[source_row]
                    b = (sign * np.einsum("ij,ij->i", displacement, axis))[:, None]
                    a = np.ones(b.shape)
                else:
                    a = np.broadcast_to(motion[source_row], b.shape)
                    if method in {"longitudinal", "transverse"}:
                        a_radial = np.einsum("ij,ij->i", a, axis)
                        b_radial = np.einsum("ij,ij->i", b, axis)
                        if method == "longitudinal":
                            a, b = a_radial[:, None], b_radial[:, None]
                        else:
                            a = a - a_radial[:, None] * axis
                            b = b - b_radial[:, None] * axis
                            if transverse_mode == "mean":
                                a, b = a / np.sqrt(dimension - 1), b / np.sqrt(dimension - 1)
            valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
            excluded("pair_nonfinite_projection", np.count_nonzero(~valid))
            a, b, targets, lag, distance, lag_bin, rbin = (
                x[valid] for x in (a, b, targets, lag, distance, lag_bin, rbin)
            )
            combined_bins = lag_bin * shape[1] + rbin
            if len(targets):
                pending.append((a, b, targets, np.full(len(targets), origin_number, dtype=np.int64),
                                lag, distance, combined_bins))
                pending_count += len(targets)
                if pending_count >= batch_size:
                    flush_pairs()
        if indexed:
            excluded("pair_outside_distance", int(remaining[4]) - kept_neighbors)

    flush_pairs()
    _check_cancelled(cancelled)
    values = np.full(bins, np.nan)
    status = np.full(bins, "no_pairs", dtype="U32")
    status[(counts > 0) & (counts < min_pairs)] = "insufficient_pairs"
    enough = counts >= min_pairs
    status[enough] = "ok"
    if normalization in {"raw", "covariance"}:
        np.divide(sums if normalization == "raw" else cross_m2, counts, out=values, where=enough)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            denominator = np.sqrt(m2_a) * np.sqrt(m2_b)
        defined = enough & np.isfinite(denominator) & (denominator > 0)
        np.divide(cross_m2, denominator, out=values, where=defined)
        values[defined] = np.clip(values[defined], -1.0, 1.0)
        status[enough & (denominator == 0)] = "zero_variance"
        status[enough & ~np.isfinite(denominator)] = "numerical_error"
    status[enough & (status == "ok") & ~np.isfinite(values)] = "numerical_error"
    values[status != "ok"] = np.nan

    def average(total):
        return np.divide(total, counts, out=np.full(bins, np.nan), where=counts > 0).reshape(shape)

    def memberships(mapping):
        result = np.zeros(bins, dtype=np.int64)
        for ibin, members in mapping.items():
            result[ibin] = len(members)
        return result.reshape(shape)

    if progress is not None:
        progress(f"Spatial motion complete: {int(np.sum(counts))} finite directed pairs")
    return {
        "values": values.reshape(shape), "status": status.reshape(shape),
        "pair_counts": counts.reshape(shape), "source_counts": source_counts.reshape(shape),
        "receiver_counts": memberships(receiver_origins),
        "source_track_counts": memberships(source_cells), "receiver_track_counts": memberships(receiver_cells),
        "mean_lags": average(lag_sums), "mean_distances": average(distance_sums),
        "mean_source_intervals": average(source_interval_sums) if needs_source_motion else np.full(shape, np.nan),
        "mean_receiver_intervals": average(receiver_interval_sums),
        "lag_edges": t_edges.copy(), "distance_edges": r_edges.copy(),
        "lag_centers": t_edges[:-1] + np.diff(t_edges) / 2,
        "distance_centers": r_edges[:-1] + np.diff(r_edges) / 2,
        "method": method, "normalization": normalization,
        "centering": "none" if normalization == "raw" else "bin_local_joint_pairs",
        "distance_convention": distance_convention, "radial_sign": radial_sign,
        "transverse_mode": transverse_mode, "components": dimension,
        "displacement_interval": float(displacement_interval), "interval_tolerance": float(interval_tolerance),
        "exclude_same_cell": bool(exclude_same_cell), "exclude_related_lineage": bool(exclude_related_lineage),
        "min_pairs": int(min_pairs), "pair_weighting": "pooled_finite_directed_pairs",
        "track_count_definition": identity_definition, "origin_count_definition": "unique_observation_rows",
        "lag_convention": "receiver_origin_time_minus_source_origin_time",
        "displacement_convention": "forward_from_origin",
        "radial_axis": "receiver_reference_toward_source",
        "bin_convention": "left_closed_right_open_final_right_closed",
        "endpoint_policy": "unique_nearest_later_endpoint_within_explicit_tolerance_no_interpolation",
        "endpoint_roundoff_policy": "8eps_times_max_1_abs_origin_clock_abs_delta",
        "cohort_policy": "continuous_membership_along_each_used_role_path",
        "source_time_reference_policy": "exact_clock_in_receiver_run",
        "eligible_source_origins": int(len(source_rows)), "eligible_receiver_origins": int(len(receiver_rows)),
        "exclusion_counts": exclusion_counts,
        "exclusion_count_definition": "source/receiver: selected origins; pair: sequentially rejected candidate pairs",
        "moments": {
            "value_sum": sums.reshape(shape), "lag_sum": lag_sums.reshape(shape),
            "distance_sum": distance_sums.reshape(shape),
            "source_interval_sum": source_interval_sums.reshape(shape),
            "receiver_interval_sum": receiver_interval_sums.reshape(shape),
            "pair_centered_cross_sum": cross_m2.reshape(shape),
            "pair_centered_square_sum_a": m2_a.reshape(shape),
            "pair_centered_square_sum_b": m2_b.reshape(shape),
            "pair_reference_mean_a": means_a.reshape(shape + (moment_dimension,)),
            "pair_reference_mean_b": means_b.reshape(shape + (moment_dimension,)),
        },
    }
