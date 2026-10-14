"""Scalar delayed mutual information on the same disjoint runs as temporal curves.

Continuous estimates use KSG estimator 1 (Kraskov et al., PRE 69, 066138,
2004); mixed estimates use the fixed-k Ross estimator (PLoS ONE 9, e87357,
2014). Discrete estimates are empirical contingency-table MI. Values are
nonnegative estimates in nats or bits, never correlation coefficients.

Pairs are dependent observations, not independent experimental units. Optional
circular-shift diagnostics assume stationary, regularly sampled finite segments.
They describe the chosen surrogate distribution, not a calibrated significance
test or an automatic bias correction. No row shuffle or hidden subsampling occurs.
"""

import numpy as np

from .temporal_correlations import _check_cancelled, _matrix, _validate_runs


DEFAULT_MAX_MI_PAIRS = 1_000_000
JITTER_SCALE = 1e-10


class MutualInformationPairLimitExceeded(ValueError):
    """The explicit per-bin pair storage budget was exceeded."""


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _standardize(values, rng):
    # Offset first: large common baselines must not set the jitter scale.
    with np.errstate(over='ignore'):
        centered = values - values[0]
    if not np.isfinite(centered).all():
        scaled = values / np.max(np.abs(values))
        centered = scaled - scaled[0]
    values = centered
    scale = np.max(np.abs(values))
    values = values / scale
    values = (values - np.mean(values)) / np.std(values)
    return values + JITTER_SCALE * rng.standard_normal(len(values))


def _radii(tree, points, k, cancelled):
    radius = np.empty(len(points))
    for start in range(0, len(points), 4096):
        _check_cancelled(cancelled)
        block = points[start:start + 4096]
        # Request only rank k+1, including self, to avoid N-by-k storage.
        radius[start:start + len(block)] = tree.query(block, k=[k + 1], p=np.inf)[0][:, 0]
    return np.nextafter(radius, 0.)


def _counts(tree, points, radius, cancelled):
    result = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 4096):
        _check_cancelled(cancelled)
        block = points[start:start + 4096]
        result[start:start + len(block)] = tree.query_ball_point(
            block, radius[start:start + len(block)], p=np.inf, return_length=True)
    return result


def estimate_mutual_information(a, b, *, kind_a='continuous', kind_b='continuous',
                                neighbors=3, units='nats', seed=0, cancelled=None):
    """Estimate MI for finite scalar pairs and report unsupported inputs.

    Input kinds are explicit; numeric category labels are never inferred from
    unique-value counts. Continuous inputs receive seeded 1e-10-SD jitter after
    centering/scaling. Ties and the unclipped estimator are reported. A constant
    variable has MI zero. Mixed bins require at least k+1 samples in EVERY
    category; rare categories are not silently discarded or assigned smaller k.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    _check_cancelled(cancelled)
    if kind_a not in {'continuous', 'discrete'} or kind_b not in {'continuous', 'discrete'}:
        raise ValueError('MI feature kinds must be continuous or discrete')
    if units not in {'nats', 'bits'}:
        raise ValueError('MI units must be nats or bits')
    k = _integer(neighbors, 'neighbors', 1)
    seed = _integer(seed, 'seed')
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('MI requires equal-length finite scalar pair arrays')
    n = len(a)
    estimator = ('discrete_plugin' if kind_a == kind_b == 'discrete' else
                 'ksg1' if kind_a == kind_b else 'ross_fixed_k')
    diagnostic = dict(estimator=estimator, status='ok', value=np.nan, raw_estimate=np.nan,
                      sample_count=n, units=units, neighbors=k,
                      ties_a=n - len(np.unique(a)), ties_b=n - len(np.unique(b)),
                      jitter_sd=JITTER_SCALE if estimator != 'discrete_plugin' else 0.)
    if not n:
        return dict(diagnostic, status='no_pairs')
    if np.all(a == a[0]) or np.all(b == b[0]):
        return dict(diagnostic, value=0., raw_estimate=0., constant_input=True)
    if estimator != 'discrete_plugin' and n <= k:
        return dict(diagnostic, status='insufficient_neighbors')
    rng = np.random.default_rng(seed)
    if estimator == 'discrete_plugin':
        _, ia, ca = np.unique(a, return_inverse=True, return_counts=True)
        _, ib, cb = np.unique(b, return_inverse=True, return_counts=True)
        # Sparse occupied pairs only: no category-A by category-B dense table.
        pairs, counts = np.unique(np.column_stack((ia, ib)), axis=0, return_counts=True)
        raw = np.sum(counts / n * np.log(counts * float(n) / (ca[pairs[:, 0]] * cb[pairs[:, 1]])))
    elif estimator == 'ksg1':
        x, y = _standardize(a, rng), _standardize(b, rng)
        xy = np.column_stack((x, y))
        radii = _radii(cKDTree(xy), xy, k, cancelled)
        nx = _counts(cKDTree(x[:, None]), x[:, None], radii, cancelled)
        ny = _counts(cKDTree(y[:, None]), y[:, None], radii, cancelled)
        raw = digamma(n) + digamma(k) - np.mean(digamma(nx) + digamma(ny))
    else:
        continuous, discrete = (a, b) if kind_a == 'continuous' else (b, a)
        _, labels, label_counts = np.unique(discrete, return_inverse=True, return_counts=True)
        diagnostic['minimum_category_count'] = int(label_counts.min())
        if np.any(label_counts <= k):
            return dict(diagnostic, status='insufficient_category_support')
        c = _standardize(continuous, rng)[:, None]
        radii = np.empty(n)
        for label in range(len(label_counts)):
            _check_cancelled(cancelled)
            members = np.flatnonzero(labels == label)
            points = c[members]
            radii[members] = _radii(cKDTree(points), points, k, cancelled)
        counts = _counts(cKDTree(c), c, radii, cancelled)
        raw = digamma(n) + digamma(k) - np.mean(digamma(label_counts[labels]) + digamma(counts))
    _check_cancelled(cancelled)
    raw = float(raw) / (np.log(2.) if units == 'bits' else 1.)
    if not np.isfinite(raw):
        return dict(diagnostic, status='numerical_error')
    return dict(diagnostic, value=max(0., raw), raw_estimate=raw)


class _PairBuffer:
    def __init__(self, maximum):
        self.maximum = maximum
        self.size = 0
        self.rows = np.empty((min(maximum or 16384, 16384), 2), dtype=np.int64)

    def append(self, source, targets):
        required = self.size + len(targets)
        if self.maximum and required > self.maximum:
            raise MutualInformationPairLimitExceeded(
                f'Delayed MI pair storage exceeded {self.maximum:,} pairs in one lag bin. '
                'Increase the MI pair limit (0 means unlimited), narrow the lag bins, or reduce the cohort; '
                'no observations were subsampled.')
        if required > len(self.rows):
            capacity = max(required, 2 * len(self.rows))
            if self.maximum:
                capacity = min(capacity, self.maximum)
            replacement = np.empty((capacity, 2), dtype=np.int64)
            replacement[:self.size] = self.rows[:self.size]
            self.rows = replacement
        self.rows[self.size:required, 0] = source
        self.rows[self.size:required, 1] = targets
        self.size = required


def _null_segments(order, offsets, times, b, minimum_shift):
    segments, supported = [], np.zeros(len(b), dtype=bool)
    reasons = set()
    for run in range(len(offsets) - 1):
        rows = order[offsets[run]:offsets[run + 1]]
        finite = np.isfinite(b[rows])
        boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
        for start, stop in boundaries.reshape(-1, 2):
            segment = rows[start:stop]
            if len(segment) < 2 * minimum_shift:
                reasons.add('short_finite_segments')
                continue
            deltas = np.diff(times[segment])
            if not np.allclose(deltas, deltas[0], rtol=1e-6, atol=1e-12):
                reasons.add('irregular_sampling')
                continue
            segments.append(segment)
            supported[segment] = True
    return segments, supported, sorted(reasons)


def compute_delayed_mutual_information(
    runs, times, *, lag_edges, feature_a, feature_b, kind_a='continuous', kind_b='continuous',
    neighbors=3, units='nats', min_pairs=2, seed=0, null_draws=0, null_min_shift=1,
    max_pairs=DEFAULT_MAX_MI_PAIRS, progress=None, cancelled=None,
):
    """I(A(t); B(t+lag)), pooling finite pairs within supplied disjoint runs.

    Bins use the temporal kernel's [left,right) convention with a closed final
    edge. One lag's exact pairs are stored at a time, with a configurable hard
    budget and no approximation. Counts remain support, never replicate N.

    Optional nulls circularly shift B inside contiguous finite segments of each
    supplied run, by at least ``null_min_shift`` observations in either cyclic
    direction. Missing rows remain missing. A bin touching a short or irregular
    segment has unavailable null diagnostics; its observed estimate is retained.
    The null mean and observed-minus-null are diagnostics, not substituted MI.
    """
    _check_cancelled(cancelled)
    min_pairs = _integer(min_pairs, 'min_pairs', 1)
    neighbors = _integer(neighbors, 'neighbors', 1)
    seed = _integer(seed, 'seed')
    null_draws = _integer(null_draws, 'null_draws')
    null_min_shift = _integer(null_min_shift, 'null_min_shift', 1)
    max_pairs = _integer(max_pairs, 'max_pairs')
    if kind_a not in {'continuous', 'discrete'} or kind_b not in {'continuous', 'discrete'}:
        raise ValueError('MI feature kinds must be continuous or discrete')
    if units not in {'nats', 'bits'}:
        raise ValueError('MI units must be nats or bits')
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError('times must be one dimensional')
    edges = np.asarray(lag_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('lag_edges must be finite and strictly increasing')
    order, offsets = _validate_runs(runs, times)
    a, b = _matrix(feature_a, len(times), 'feature_a'), _matrix(feature_b, len(times), 'feature_b')
    if a.shape[1] != 1 or b.shape[1] != 1:
        raise ValueError('Delayed MI currently requires one scalar component per feature')
    a, b = a[:, 0], b[:, 0]
    bins = len(edges) - 1
    values, mean_lags = np.full(bins, np.nan), np.full(bins, np.nan)
    pair_counts, track_counts = np.zeros(bins, dtype=np.int64), np.zeros(bins, dtype=np.int64)
    statuses = np.full(bins, 'no_pairs', dtype='U40')
    diagnostics = []
    segments, null_supported, null_reasons = _null_segments(order, offsets, times, b, null_min_shift) if null_draws else ([], None, [])
    for ibin in range(bins):
        _check_cancelled(cancelled)
        if progress:
            progress(f'Delayed MI lag {ibin + 1}/{bins}: collecting finite pairs')
        buffer = _PairBuffer(max_pairs)
        lag_sum = 0.
        for run in range(len(offsets) - 1):
            rows = order[offsets[run]:offsets[run + 1]]
            clock = times[rows]
            old_count = buffer.size
            for origin, source in enumerate(rows):
                _check_cancelled(cancelled)
                if not np.isfinite(a[source]):
                    continue
                lower = np.searchsorted(clock, np.nextafter(clock[origin] + edges[ibin], -np.inf), side='left')
                upper = np.searchsorted(clock, np.nextafter(clock[origin] + edges[ibin + 1], np.inf), side='right')
                for start in range(lower, upper, 4096):
                    _check_cancelled(cancelled)
                    targets = rows[start:min(start + 4096, upper)]
                    lag = times[targets] - times[source]
                    valid = (lag >= edges[ibin]) & ((lag <= edges[ibin + 1]) if ibin == bins - 1 else (lag < edges[ibin + 1])) & np.isfinite(b[targets])
                    buffer.append(source, targets[valid])
                    lag_sum += np.sum(lag[valid])
            track_counts[ibin] += buffer.size > old_count
        n = buffer.size
        pair_counts[ibin] = n
        if n:
            mean_lags[ibin] = lag_sum / n
        diagnostic = dict(sample_count=n, null_status='disabled', null_draws_requested=null_draws)
        if n < min_pairs:
            statuses[ibin] = 'insufficient_pairs' if n else 'no_pairs'
            diagnostics.append(dict(diagnostic, status=str(statuses[ibin])))
            continue
        pairs = buffer.rows[:n]
        av, bv = a[pairs[:, 0]], b[pairs[:, 1]]
        estimator_seed = int(np.random.SeedSequence([seed, ibin, 0]).generate_state(1)[0])
        estimate = estimate_mutual_information(av, bv, kind_a=kind_a, kind_b=kind_b,
            neighbors=neighbors, units=units, seed=estimator_seed, cancelled=cancelled)
        diagnostic.update(estimate)
        statuses[ibin], values[ibin] = estimate['status'], estimate['value']
        if null_draws and estimate['status'] == 'ok':
            targets = pairs[:, 1]
            if not null_supported[targets].all():
                diagnostic.update(null_status='unsupported_segments', null_unshiftable_pair_count=int((~null_supported[targets]).sum()),
                                  null_reasons=null_reasons)
            else:
                null_values = []
                null_failures = {}
                used = np.zeros(len(b), dtype=bool)
                used[targets] = True
                active = [segment for segment in segments if used[segment].any()]
                shifted = b.copy()
                for draw in range(null_draws):
                    _check_cancelled(cancelled)
                    rng = np.random.default_rng(np.random.SeedSequence([seed, ibin, draw + 1]))
                    for segment in active:
                        shift = int(rng.integers(null_min_shift, len(segment) - null_min_shift + 1))
                        shifted[segment] = np.roll(b[segment], shift)
                    null = estimate_mutual_information(av, shifted[targets], kind_a=kind_a, kind_b=kind_b,
                        neighbors=neighbors, units=units, seed=estimator_seed, cancelled=cancelled)
                    if null['status'] == 'ok':
                        null_values.append(null['value'])
                    else:
                        null_failures[null['status']] = null_failures.get(null['status'], 0) + 1
                    if progress:
                        progress(f'Delayed MI lag {ibin + 1}/{bins}: circular-shift diagnostic {draw + 1}/{null_draws}')
                # Do not summarize only successful surrogates: that conditions
                # the diagnostic on a different category-support distribution.
                complete = len(null_values) == null_draws
                diagnostic.update(null_status='ok' if complete else 'unsupported_surrogates',
                    null_draws_valid=len(null_values), null_failures=null_failures,
                    null_mean=float(np.mean(null_values)) if complete else np.nan,
                    null_sd=float(np.std(null_values, ddof=1)) if complete and null_draws > 1 else np.nan,
                    observed_minus_null=float(values[ibin] - np.mean(null_values)) if complete else np.nan)
        diagnostics.append(diagnostic)
    _check_cancelled(cancelled)
    return dict(lag_edges=edges.copy(), lag_centers=edges[:-1] + np.diff(edges) / 2,
        mean_lags=mean_lags, values=values, pair_counts=pair_counts, track_counts=track_counts, status=statuses,
        method='mutual_information', normalization='none', centering='none', components=1,
        min_pairs=min_pairs, kind_a=kind_a, kind_b=kind_b, neighbors=neighbors, value_units=units,
        seed=seed, max_pairs=max_pairs, jitter_sd=JITTER_SCALE,
        pair_weighting='pooled_finite_pairs', track_count_definition='contributing_disjoint_runs',
        lag_convention='A(t), B(t+lag)', bin_convention='left_closed_right_open_final_right_closed',
        endpoint_policy='finite_endpoints_within_supplied_run', diagnostics=diagnostics,
        null_method='within_finite_segment_circular_shift' if null_draws else 'disabled',
        null_draws=null_draws, null_min_shift=null_min_shift,
        null_assumptions='Stationarity and uniform cadence within finite B segments; circular boundary; shifts chosen in observation units. Diagnostic association, not causal influence or a significance p-value.',
        estimator_references=['https://doi.org/10.1103/PhysRevE.69.066138', 'https://doi.org/10.1371/journal.pone.0087357'])
