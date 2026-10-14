"""Finite-pair temporal curves over disjoint, observation-indexed runs.

This module does not discover tracks, infer clocks, register positions, or choose
experimental units. Callers supply those decisions. Pair counts measure support,
not biological replication. No pair joins two supplied runs.
"""

import numpy as np


class TemporalCorrelationCancelled(RuntimeError):
    """Raised when a temporal calculation is cooperatively cancelled."""


def _check_cancelled(cancelled):
    if cancelled is not None:
        requested = cancelled() if callable(cancelled) else cancelled.is_set()
        if requested:
            raise TemporalCorrelationCancelled("Temporal correlation calculation cancelled")


def _matrix(values, size, name):
    if values is None:
        raise ValueError(f"{name} is required")
    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] != size or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape (observations,) or (observations, components)")
    return result


def _validate_runs(runs, times):
    order_input = np.asarray(runs.order)
    offsets_input = np.asarray(runs.offsets)
    if order_input.ndim != 1 or offsets_input.ndim != 1:
        raise ValueError("Run order and offsets must be one-dimensional")
    if not np.issubdtype(order_input.dtype, np.integer) and order_input.size:
        raise ValueError("Run observation indices must be integers")
    if not np.issubdtype(offsets_input.dtype, np.integer):
        raise ValueError("Run offsets must be integers")
    order = order_input.astype(np.int64, copy=False)
    offsets = offsets_input.astype(np.int64, copy=False)
    if (not offsets.size or offsets[0] != 0 or offsets[-1] != order.size
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Run offsets must partition observations into nonempty runs")
    if np.any(order < 0) or np.any(order >= times.size):
        raise ValueError("Run observation index outside the time/feature arrays")
    if np.unique(order).size != order.size:
        raise ValueError("Runs must be disjoint; repeated observation indices are not allowed")
    for run in range(offsets.size - 1):
        clock = times[order[offsets[run]:offsets[run + 1]]]
        if not np.all(np.isfinite(clock)) or np.any(np.diff(clock) <= 0):
            raise ValueError(f"Run {run} times must be finite and strictly increasing; do not sort or compress gaps")
    return order, offsets


def _feature_summary(values, rows, cancelled, batch_size):
    """Stable vector mean and trace sum of squares without a full gather."""
    count = 0
    mean = np.zeros(values.shape[1])
    m2 = 0.0
    for start in range(0, rows.size, batch_size):
        _check_cancelled(cancelled)
        batch = values[rows[start:start + batch_size]]
        batch = batch[np.all(np.isfinite(batch), axis=1)]
        n = len(batch)
        if not n:
            continue
        # Center before taking a mean to avoid losing small fluctuations on a
        # large common baseline; the parallel-moment merge is similarly stable.
        chunk_mean = batch[0] + np.mean(batch - batch[0], axis=0)
        residual = batch - chunk_mean
        chunk_m2 = float(np.einsum("ij,ij->", residual, residual))
        delta = chunk_mean - mean
        total = count + n
        m2 += chunk_m2 + float(delta @ delta) * count * n / total
        mean += delta * n / total
        count = total
    return count, mean, m2


def _references(values, order, offsets, centering, cancelled, batch_size):
    if centering == "roi":
        n, mean, m2 = _feature_summary(values, order, cancelled, batch_size)
        return mean[None, :], m2 / n if n else np.nan, n
    means = np.zeros((offsets.size - 1, values.shape[1]))
    count = 0
    m2 = 0.0
    for run in range(offsets.size - 1):
        n, means[run], run_m2 = _feature_summary(
            values, order[offsets[run]:offsets[run + 1]], cancelled, batch_size
        )
        count += n
        m2 += run_m2
    return means, m2 / count if count else np.nan, count


def compute_temporal_curve(
    runs, times, *, lag_edges, method="autocorrelation", feature_a=None,
    feature_b=None, positions=None, normalization="coefficient", centering="roi",
    min_pairs=2, progress=None, cancelled=None, mi_options=None,
):
    """Compute a pooled finite-endpoint curve without materializing all pairs.

    ``times`` and numeric inputs are aligned to observation rows; ``runs`` is a
    :class:`celltraj2.trajectory_index.Runs`, normally from ``GraphIndex.project``.
    Times inside each run must be finite and strictly increasing. Invalid clocks
    raise rather than silently sorting rows or reconnecting gaps. Feature NaNs
    exclude endpoint pairs but do not alter timestamps or remove interior rows.

    Positive lag means A now, B later. Every ordered eligible pair (including
    self pairs at zero lag) contributes once. Bins are [left, right), except the
    final right edge is included. A bin containing several elapsed times reports
    their pooled result; it is not forcibly normalized to one merely because its
    center is zero. ``mean_lags`` records actual contributing elapsed times.

    Scalar inputs are (N,) or (N, 1); vector inputs are (N, D) with the same D
    for A/B. Vector products are dot products; variances are covariance traces,
    giving rotationally invariant normalization under a shared rotation.

    * ``raw``: mean A dot B, without centering.
    * ``covariance``: mean (A - reference_A) dot (B - reference_B).
    * ``coefficient``: joint-pair, lag-local centered dot product divided by
      sqrt(trace variance A * trace variance B), using the same valid pairs.
    * ``fixed``: covariance divided by fixed zero-lag reference trace standard
      deviations. No clipping: changing support can produce magnitudes above 1.

    ``centering='roi'`` obtains separate A/B reference means from their complete
    finite observations selected by runs. ``'track'`` subtracts each run's mean
    before pooling; fixed variances then pool within-run squared fluctuations.
    The coefficient in track mode is a correlation of pooled detrended pairs,
    not an average of per-track coefficients. Raw products ignore centering.
    A/B fixed reference supports can differ and are reported separately.

    ``method='msd'`` uses mean squared Euclidean displacement of supplied
    positions and ignores normalization/centering (reported as raw/none).
    Position units/calibration and selected components are the caller's choice.
    Zero displacement is valid; no unit-direction normalization is applied.

    ``method='mutual_information'`` dispatches scalar A/B inputs to
    ``compute_delayed_mutual_information``. Its explicit feature kinds, nearest
    neighbor count, nats/bits, seed, optional circular null and storage budget
    are passed in ``mi_options``. Correlation normalization/centering are unused.

    ``track_counts`` counts contributing supplied runs, not unique physical
    lineages or experimental replicates. Counts remain available when a value
    is masked by ``min_pairs``. Optional ``progress`` receives strings and
    ``cancelled`` is a predicate (or event with is_set); cancellation raises
    :class:`TemporalCorrelationCancelled`. Scratch pair memory is bounded by a
    target chunk, while work scales with candidate pairs inside the lag range.
    """
    if method == "mutual_information":
        from .mutual_information import compute_delayed_mutual_information
        return compute_delayed_mutual_information(
            runs, times, lag_edges=lag_edges, feature_a=feature_a, feature_b=feature_b,
            min_pairs=min_pairs, progress=progress, cancelled=cancelled, **(mi_options or {}))
    _check_cancelled(cancelled)
    if method not in {"msd", "autocorrelation", "crosscorrelation"}:
        raise ValueError("Unknown temporal method")
    if normalization not in {"raw", "covariance", "coefficient", "fixed"}:
        raise ValueError("Unknown temporal normalization")
    if centering not in {"roi", "track"}:
        raise ValueError("Centering must be roi or track")
    if isinstance(min_pairs, (bool, np.bool_)) or int(min_pairs) != min_pairs or min_pairs < 1:
        raise ValueError("min_pairs must be a positive integer")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional observation array")
    edges = np.asarray(lag_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("lag_edges must contain at least two finite, strictly increasing edges")
    order, offsets = _validate_runs(runs, times)
    if method == "msd":
        a = b = _matrix(positions, times.size, "positions")
    else:
        a = _matrix(feature_a, times.size, "feature_a")
        b = a if method == "autocorrelation" else _matrix(feature_b, times.size, "feature_b")
        if a.shape[1] != b.shape[1]:
            raise ValueError("A and B must have the same number of vector components")
    dimension = a.shape[1]
    # At most 4096 receiver rows; bound vector scratch as dimension increases.
    batch_size = min(4096, max(1, 1024 * 1024 // (8 * dimension * 6)))
    bins = edges.size - 1
    counts = np.zeros(bins, dtype=np.int64)
    track_counts = np.zeros(bins, dtype=np.int64)
    sums = np.zeros(bins)
    lag_sums = np.zeros(bins)
    fixed_sums = np.zeros(bins)
    means_a = np.zeros((bins, dimension))
    means_b = np.zeros((bins, dimension))
    m2_a = np.zeros(bins)
    m2_b = np.zeros(bins)
    cross_m2 = np.zeros(bins)
    reference_a = reference_b = np.zeros((1, dimension))
    reference_var_a = reference_var_b = np.nan
    reference_count_a = reference_count_b = 0
    if method != "msd" and normalization != "raw":
        if progress is not None:
            progress("Computing finite temporal reference statistics")
        reference_a, reference_var_a, reference_count_a = _references(
            a, order, offsets, centering, cancelled, batch_size
        )
        if b is a:
            reference_b, reference_var_b, reference_count_b = reference_a, reference_var_a, reference_count_a
        else:
            reference_b, reference_var_b, reference_count_b = _references(
                b, order, offsets, centering, cancelled, batch_size
            )

    for run in range(offsets.size - 1):
        _check_cancelled(cancelled)
        rows = order[offsets[run]:offsets[run + 1]]
        clock = times[rows]
        touched = np.zeros(bins, dtype=bool)
        reference_index = run if centering == "track" and normalization != "raw" and method != "msd" else 0
        center_a = reference_a[reference_index]
        center_b = reference_b[reference_index]
        if progress is not None:
            progress(f"Temporal run {run + 1}/{offsets.size - 1}: {rows.size} observations")
        for origin, source_row in enumerate(rows):
            _check_cancelled(cancelled)
            av = a[source_row]
            if not np.all(np.isfinite(av)):
                continue
            av_centered = av - center_a
            lower = np.searchsorted(clock, np.nextafter(clock[origin] + edges[0], -np.inf), side="left")
            upper = np.searchsorted(clock, np.nextafter(clock[origin] + edges[-1], np.inf), side="right")
            for start in range(lower, upper, batch_size):
                _check_cancelled(cancelled)
                target_rows = rows[start:min(start + batch_size, upper)]
                delta_t = times[target_rows] - clock[origin]
                ibins = np.searchsorted(edges, delta_t, side="right") - 1
                ibins[delta_t == edges[-1]] = bins - 1
                bv = b[target_rows]
                valid = (ibins >= 0) & (ibins < bins) & np.all(np.isfinite(bv), axis=1)
                ibins, delta_t, bv = ibins[valid], delta_t[valid], bv[valid]
                for ibin in np.unique(ibins):
                    selected = ibins == ibin
                    targets = bv[selected]
                    n = targets.shape[0]
                    old_count = counts[ibin]
                    total = old_count + n
                    counts[ibin] = total
                    touched[ibin] = True
                    lag_sums[ibin] += np.sum(delta_t[selected])
                    if method == "msd":
                        displacement = targets - av
                        sums[ibin] += np.einsum("ij,ij->", displacement, displacement)
                        continue
                    if normalization == "raw":
                        sums[ibin] += np.sum(targets @ av)
                        continue
                    targets = targets - center_b
                    fixed_sums[ibin] += np.sum(targets @ av_centered)
                    # Within this chunk A is constant. Merge chunk means and
                    # centered moments, avoiding unstable E[X^2] - E[X]^2.
                    chunk_mean_b = targets[0] + np.mean(targets - targets[0], axis=0)
                    centered_b = targets - chunk_mean_b
                    difference_a = av_centered - means_a[ibin]
                    difference_b = chunk_mean_b - means_b[ibin]
                    merge_weight = old_count * n / total
                    m2_a[ibin] += (difference_a @ difference_a) * merge_weight
                    m2_b[ibin] += np.einsum("ij,ij->", centered_b, centered_b) + (difference_b @ difference_b) * merge_weight
                    cross_m2[ibin] += (difference_a @ difference_b) * merge_weight
                    means_a[ibin] += difference_a * n / total
                    means_b[ibin] += difference_b * n / total
            if progress is not None and (origin + 1) % 256 == 0:
                progress(f"Temporal run {run + 1}/{offsets.size - 1}: {origin + 1}/{rows.size} origins")
        track_counts += touched

    _check_cancelled(cancelled)
    values = np.full(bins, np.nan)
    mean_lags = np.divide(lag_sums, counts, out=np.full(bins, np.nan), where=counts > 0)
    status = np.full(bins, "no_pairs", dtype="U32")
    status[(counts > 0) & (counts < min_pairs)] = "insufficient_pairs"
    enough = counts >= min_pairs
    status[enough] = "ok"
    if method == "msd" or normalization == "raw":
        np.divide(sums, counts, out=values, where=enough)
    elif normalization == "covariance":
        np.divide(fixed_sums, counts, out=values, where=enough)
    elif normalization == "fixed":
        denominator = np.sqrt(reference_var_a) * np.sqrt(reference_var_b)
        if np.isfinite(denominator) and denominator > 0:
            np.divide(fixed_sums, counts, out=values, where=enough)
            values[enough] /= denominator
        else:
            status[enough] = "zero_reference_variance" if denominator == 0 else "invalid_reference"
    else:
        denominator = np.sqrt(m2_a) * np.sqrt(m2_b)
        defined = enough & np.isfinite(denominator) & (denominator > 0)
        np.divide(cross_m2, denominator, out=values, where=defined)
        # The coefficient is mathematically bounded; remove only round-off.
        values[defined] = np.clip(values[defined], -1.0, 1.0)
        status[enough & (denominator == 0)] = "zero_variance"
        status[enough & ~np.isfinite(denominator)] = "numerical_error"
    failed_numeric = enough & (status == "ok") & ~np.isfinite(values)
    status[failed_numeric] = "numerical_error"
    values[status != "ok"] = np.nan
    return {
        "lag_edges": edges.copy(),
        "lag_centers": edges[:-1] + np.diff(edges) / 2,
        "mean_lags": mean_lags,
        "values": values,
        "pair_counts": counts,
        "track_counts": track_counts,
        "status": status,
        "method": method,
        "normalization": "raw" if method == "msd" else normalization,
        "centering": "none" if method == "msd" or normalization == "raw" else centering,
        "components": dimension,
        "min_pairs": int(min_pairs),
        "pair_weighting": "pooled_finite_pairs",
        "track_count_definition": "contributing_disjoint_runs",
        "lag_convention": "A(t), B(t+lag)",
        "bin_convention": "left_closed_right_open_final_right_closed",
        "endpoint_policy": "finite_endpoints_within_supplied_run",
        "reference_variance_a": reference_var_a,
        "reference_variance_b": reference_var_b,
        "reference_count_a": reference_count_a,
        "reference_count_b": reference_count_b,
        "moments": {
            "value_sum": sums if method == "msd" or normalization == "raw" else fixed_sums,
            "lag_sum": lag_sums,
            "pair_centered_cross_sum": cross_m2,
            "pair_centered_square_sum_a": m2_a,
            "pair_centered_square_sum_b": m2_b,
            "pair_reference_mean_a": means_a,
            "pair_reference_mean_b": means_b,
        },
    }
