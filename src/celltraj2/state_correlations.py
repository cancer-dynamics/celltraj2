"""Finite, ROI-local state/feature and frozen-event distance–time surfaces.

Observations are array rows, never biological IDs. Event callers must resolve
their stable focal identities to ``anchor_rows`` before entering this kernel.
No source survival is required; only source-time receiver geometry uses runs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .spatial_correlations import (
    _edges, _mask, _bin_indices, _check_cancelled, SpatialSupportLimitExceeded,
    DEFAULT_MAX_SUPPORT_ENTRIES,
)
from .temporal_correlations import _validate_runs

METHODS = frozenset({'state_conditional', 'state_joint', 'feature_correlation',
                     'event_state', 'event_feature'})


def compute_state_surface(runs, times, positions, feature_b, *, distance_edges,
        lag_edges, feature_a=None, method='state_conditional', normalization='raw',
        response_mode='response', weighting=None, distance_convention=None,
        source_mask=None, receiver_mask=None, cell_ids=None, lineage_ids=None,
        anchor_rows=None, anchor_times=None, anchor_ids=None,
        exclude_same_cell=True, exclude_related_lineage=False, min_pairs=2,
        max_support_entries=DEFAULT_MAX_SUPPORT_ENTRIES, progress=None, cancelled=None):
    """Compute directed pairs with explicit exposure and missingness.

    Conditional responses use source ON anchors; joint occupancy uses all known
    source/receiver states. State values must be 0/1/NaN. Continuous products,
    bin-local covariance and Pearson use jointly finite pairs. Events use the
    supplied frozen focal row position at ``anchor_times`` (not later survival).
    ``source_time`` requires an exact receiver reference clock in the same
    unbroken, receiver-eligible run; ``cross_time`` needs only target exposure.

    Equal-anchor weighting averages each anchor's finite receivers within each
    bin, then supported anchors. Equal-frame weighting pools pairs per source
    clock, then averages supported clocks. ``pairs`` pools finite pairs. Joint
    occupancy and feature correlation require pairs. Receivers can contribute
    to several event neighborhoods; these are descriptive weights, not N.

    Baseline is the mean of all finite receivers in the same ROI, target clock
    and eligible cohort, using the same self/lineage and geometry eligibility.
    It is matched to each response pair's target clock, then aggregated with
    exactly the response weighting. It is population exposure, NOT event-free
    control exposure. No review coverage is inferred. Response and baseline are
    always retained separately. Zero enrichment denominator is undefined.

    KD trees are reused per target clock (cross-time), or per source/target clock
    (source-time). Scratch arrays are at most one receiver frame; no all-pairs
    table is created. Sparse per-anchor sufficient sums and distinct support
    memberships share the explicit support-entry budget (0 = unlimited).
    """
    _check_cancelled(cancelled)
    if method not in METHODS:
        raise ValueError('Unknown state/event surface method')
    event = method.startswith('event_')
    conditional = method in {'state_conditional', 'event_state', 'event_feature'}
    weighting = weighting or ('equal_anchor' if conditional else 'pairs')
    if weighting not in {'pairs', 'equal_anchor', 'equal_frame'} or (not conditional and weighting != 'pairs'):
        raise ValueError('Joint occupancy and feature correlations require pair weighting')
    if normalization not in {'raw', 'covariance', 'coefficient'} or (method != 'feature_correlation' and normalization != 'raw'):
        raise ValueError('Only continuous feature correlations support covariance/coefficient normalization')
    if response_mode not in {'response', 'baseline', 'excess', 'enrichment'} or (not conditional and response_mode != 'response'):
        raise ValueError('Baseline/excess/enrichment require a conditional or event response')
    distance_convention = distance_convention or ('cross_time' if event else 'source_time')
    if distance_convention not in {'cross_time', 'source_time'}:
        raise ValueError('Choose source_time or cross_time geometry')
    if isinstance(min_pairs, (bool, np.bool_)) or int(min_pairs) != min_pairs or min_pairs < 1:
        raise ValueError('min_pairs must be a positive integer')
    if isinstance(max_support_entries, (bool, np.bool_)) or not isinstance(max_support_entries, (int, np.integer)) or max_support_entries < 0:
        raise ValueError('max_support_entries must be a nonnegative integer')
    t = np.asarray(times, float)
    p = np.asarray(positions, float)
    b = np.asarray(feature_b, float)
    n = len(t)
    if t.ndim != 1 or b.shape != (n,) or p.ndim != 2 or p.shape[0] != n or not p.shape[1]:
        raise ValueError('Times, positions and scalar features must be observation aligned')
    order, offsets = _validate_runs(runs, t)
    selected = np.zeros(n, bool); selected[order] = True
    geometry = selected & np.isfinite(t) & np.isfinite(p).all(axis=1)
    sm = _mask(source_mask, n, 'source_mask') & geometry
    rm = _mask(receiver_mask, n, 'receiver_mask') & geometry
    run_ids = np.full(n, -1, int)
    for k, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        run_ids[order[start:stop]] = k
    cells = run_ids if cell_ids is None else np.asarray(cell_ids)
    lineage = cells if lineage_ids is None else np.asarray(lineage_ids)
    for ids in (cells, lineage):
        if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer) or (ids[order] < 0).any():
            raise ValueError('Cell/lineage identities must be nonnegative observation-aligned integers')
    if exclude_related_lineage and lineage_ids is None:
        raise ValueError('Related-lineage exclusion requires lineage IDs')
    a = np.ones(n) if event else np.asarray(feature_a, float)
    if a.shape != (n,):
        raise ValueError('feature_a must be an observation-aligned scalar')
    for values in ([b] if method == 'event_state' else [a, b] if method.startswith('state_') else []):
        known = values[np.isfinite(values)]
        if not np.isin(known, [0., 1.]).all():
            raise ValueError('Empirical states must be 0/1/NaN; model scores are continuous inputs')
    sm &= np.isfinite(a)
    if method == 'state_conditional':
        sm &= a == 1
    if event:
        rows = np.asarray(anchor_rows)
        clocks = np.asarray(anchor_times, float)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or clocks.shape != rows.shape or np.any((rows < 0) | (rows >= n)) or not np.isfinite(clocks).all():
            raise ValueError('Event anchors require valid row indices and finite occurrence times')
        ids = list(anchor_ids) if anchor_ids is not None else list(range(len(rows)))
        if len(ids) != len(rows) or len(set(ids)) != len(ids):
            raise ValueError('Event anchor IDs must be unique')
        keep = sm[rows]
        rows, clocks = rows[keep], clocks[keep]
        ids = [v for v, ok in zip(ids, keep) if ok]
    else:
        rows = np.flatnonzero(sm); clocks = t[rows]; ids = rows.tolist()
    re = _edges(distance_edges, 'distance_edges', nonnegative=True)
    le = _edges(lag_edges, 'lag_edges')
    shape = (len(le)-1, len(re)-1); size = int(np.prod(shape))
    if size > 100000:
        raise ValueError('Choose at most 100,000 distance–time cells')
    # Moments use fixed shifts before accumulation to avoid large-offset loss.
    a0 = float(a[rows[0]]) if len(rows) else 0.
    finite_b = b[np.isfinite(b) & rm]
    b0 = float(finite_b[0]) if len(finite_b) else 0.
    counts = np.zeros(size, np.int64)
    moments = np.zeros((7, size))  # A, B, AA, BB, AB, matched baseline, lag
    # Bin-local centered moments must not inherit an unrelated anchor's offset.
    centered = np.zeros((6, size))  # N, mean A, mean B, M2 A, M2 B, co-moment
    distance_sum = np.zeros(size)
    response_sum = np.zeros(size); baseline_sum = np.zeros(size); weights = np.zeros(size)
    support = {name: [set() for _ in range(size)] for name in ('sources', 'receivers', 'source_cells', 'receiver_cells', 'exposure')}
    memberships = 0
    anchor_sums = []
    frame_sums = {}
    def budget(add):
        nonlocal memberships
        memberships += add
        if max_support_entries and memberships > max_support_entries:
            raise SpatialSupportLimitExceeded(f'Spatial support membership exceeded configured limit {max_support_entries:,}; increase the support limit or use 0 for unlimited')
    def remember(name, slot, values):
        old = len(support[name][slot]); support[name][slot].update(values)
        budget(len(support[name][slot])-old)
    # Role-filtered runs ensure source-time geometry cannot reconnect a cohort gap.
    reference_lookup = {}
    if distance_convention == 'source_time':
        for rid, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            segment = 0
            for row in order[start:stop]:
                if not rm[row]:
                    segment += 1; continue
                reference_lookup[(rid, segment, float(t[row]))] = int(row)
        row_segment = np.full(n, -1, int)
        for (rid, segment, clock), row in reference_lookup.items():
            row_segment[row] = segment
    receiver_order = np.flatnonzero(rm & np.isfinite(b))
    receiver_order = receiver_order[np.argsort(t[receiver_order], kind='stable')]
    splits = np.flatnonzero(np.diff(t[receiver_order]))+1
    frames = [(float(t[group[0]]), group) for group in np.split(receiver_order, splits) if len(group)]
    frame_clocks = np.array([clock for clock, _ in frames])
    trees = {}
    for ai, (row, clock) in enumerate(zip(rows, clocks)):
        _check_cancelled(cancelled)
        local = {}  # occupied bin -> [N, sums A/B/AA/BB/AB/baseline/lag/distance]
        first_frame, last_frame = np.searchsorted(frame_clocks, [clock+le[0], clock+le[-1]], side='left')
        last_frame = np.searchsorted(frame_clocks, clock+le[-1], side='right')
        for target_clock, target in frames[first_frame:last_frame]:
            lag = target_clock-clock
            li = int(_bin_indices(le, np.array([lag]))[0])
            if li < 0 or li >= shape[0]:
                continue
            _check_cancelled(cancelled)
            cache_key = (float(clock), target_clock) if distance_convention == 'source_time' else target_clock
            if cache_key not in trees:
                refs = target if distance_convention == 'cross_time' else np.array([
                    reference_lookup.get((int(run_ids[r]), int(row_segment[r]), float(clock)), -1) for r in target], int)
                accepted = refs >= 0
                target_rows, reference_rows = target[accepted], refs[accepted]
                trees[cache_key] = (target_rows, reference_rows, cKDTree(p[reference_rows]) if len(reference_rows) else None)
            target_rows, reference_rows, tree = trees[cache_key]
            if tree is None:
                continue
            eligible = np.ones(len(target_rows), bool)
            if exclude_same_cell:
                eligible &= cells[target_rows] != cells[row]
            if exclude_related_lineage:
                eligible &= lineage[target_rows] != lineage[row]
            exposure = target_rows[eligible]
            if not len(exposure):
                continue
            baseline = float(np.mean(b[exposure]-b0)) if conditional else 0.
            near = np.asarray(tree.query_ball_point(p[row], float(re[-1])), int)
            near = near[eligible[near]]
            if not len(near):
                continue
            targets = target_rows[near]
            distances = np.linalg.norm(p[reference_rows[near]]-p[row], axis=1)
            rb = _bin_indices(re, distances)
            for ri in np.unique(rb[(rb >= 0) & (rb < shape[1])]):
                take = rb == ri; rr = targets[take]; d = distances[take]
                slot = li*shape[1]+int(ri); number = len(rr)
                da, db = a[row]-a0, b[rr]-b0
                if method == 'feature_correlation':
                    values_b = b[rr]
                    local_b_mean = float(values_b[0]+np.mean(values_b-values_b[0]))
                    residual_b = values_b-local_b_mean
                    old_n, old_a, old_b, old_aa, old_bb, old_ab = centered[:, slot]
                    total_n = old_n+number
                    delta_a, delta_b = a[row]-old_a, local_b_mean-old_b
                    merge_weight = old_n*number/total_n
                    centered[:, slot] = [total_n, old_a+delta_a*number/total_n,
                        old_b+delta_b*number/total_n, old_aa+delta_a*delta_a*merge_weight,
                        old_bb+float(np.dot(residual_b, residual_b))+delta_b*delta_b*merge_weight,
                        old_ab+delta_a*delta_b*merge_weight]
                vector = np.array([number, da*number, db.sum(), da*da*number,
                    np.dot(db, db), da*db.sum(), baseline*number, lag*number, d.sum()])
                local[slot] = local.get(slot, np.zeros(9)) + vector
                remember('sources', slot, [ai]); remember('receivers', slot, rr.tolist())
                remember('source_cells', slot, [int(cells[row])]); remember('receiver_cells', slot, cells[rr].tolist())
                if conditional:
                    remember('exposure', slot, exposure.tolist())
        for slot, vector in local.items():
            number = int(vector[0]); counts[slot] += number
            moments[:, slot] += vector[1:8]; distance_sum[slot] += vector[8]
            budget(1)
            anchor_sums.append([ai, slot, *vector.tolist()])
            if weighting == 'equal_anchor':
                response_sum[slot] += vector[2]/number+b0
                baseline_sum[slot] += vector[6]/number+b0; weights[slot] += 1
            elif weighting == 'equal_frame':
                key = (float(clock), slot)
                frame_sums[key] = frame_sums.get(key, np.zeros(3))+vector[[0, 2, 6]]
        if distance_convention == 'source_time':
            # Cache only one source clock; repeated event/source anchors reuse it.
            if ai+1 == len(rows) or clocks[ai+1] != clock:
                trees.clear()
        if progress and (ai % 100 == 0 or ai+1 == len(rows)):
            progress(f'State/event anchors {ai+1}/{len(rows)}; {int(counts.sum())} finite directed pairs')
    if weighting == 'equal_frame':
        for (_, slot), (number, value, base) in frame_sums.items():
            response_sum[slot] += value/number+b0; baseline_sum[slot] += base/number+b0; weights[slot] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_a, mean_b = moments[0]/counts, moments[1]/counts
        if method == 'feature_correlation':
            cov = centered[5]/centered[0]
            va = centered[3]/centered[0]
            vb = centered[4]/centered[0]
            response = (moments[4]/counts+a0*mean_b+b0*mean_a+a0*b0 if normalization == 'raw'
                        else cov if normalization == 'covariance' else cov/np.sqrt(va*vb))
        elif method == 'state_joint':
            response = moments[4]/counts+a0*mean_b+b0*mean_a+a0*b0
        else:
            response = mean_b+b0 if weighting == 'pairs' else response_sum/weights
        baseline = moments[5]/counts+b0 if weighting == 'pairs' else baseline_sum/weights
        values = (response if response_mode == 'response' else baseline if response_mode == 'baseline'
                  else response-baseline if response_mode == 'excess' else response/baseline)
    status = np.full(size, 'ok', dtype='<U32')
    status[counts < min_pairs] = 'insufficient_pairs'
    if method == 'feature_correlation' and normalization == 'coefficient':
        status[(counts >= min_pairs) & ((va <= 0) | (vb <= 0))] = 'zero_variance'
    if response_mode == 'enrichment':
        status[(counts >= min_pairs) & (baseline == 0)] = 'zero_baseline'
    bad = status != 'ok'; values[bad] = np.nan
    response[counts < min_pairs] = np.nan; baseline[counts < min_pairs] = np.nan
    if not conditional:
        baseline[:] = np.nan
    result = dict(values=values.reshape(shape), response_values=response.reshape(shape), baseline_values=baseline.reshape(shape),
        status=status.reshape(shape), pair_counts=counts.reshape(shape),
        lag_edges=le, distance_edges=re, lag_centers=(le[:-1]+le[1:])/2, distance_centers=(re[:-1]+re[1:])/2,
        weighting=weighting, distance_convention=distance_convention, normalization=normalization, method=method,
        baseline_definition='same ROI, target clock and eligible receiver cohort; matched response weighting; not event-free controls',
        source_anchor_count=len(rows), receiver_reuse='allowed across distinct anchors',
        support_details=dict(anchor_ids=ids, anchor_rows=rows, anchor_times=clocks,
            anchor_sums=np.asarray(anchor_sums, float).reshape((-1, 11)),
            anchor_sum_columns=['anchor_index', 'flat_bin', 'pairs', 'sum_a_shifted', 'sum_b_shifted', 'sum_aa_shifted', 'sum_bb_shifted', 'sum_ab_shifted', 'sum_baseline_shifted', 'sum_lag', 'sum_distance'],
            feature_shifts=[a0, b0], weighting_counts=weights.reshape(shape),
            centered_moments=centered.reshape((6, *shape)),
            centered_moment_columns=['pairs', 'mean_a', 'mean_b', 'm2_a', 'm2_b', 'co_moment']))
    for name, field in [('sources', 'source_counts'), ('receivers', 'receiver_counts'), ('source_cells', 'source_track_counts'), ('receiver_cells', 'receiver_track_counts'), ('exposure', 'exposure_counts')]:
        result[field] = np.array([len(s) for s in support[name]], np.int64).reshape(shape)
    result['anchor_counts'] = result['source_counts'].copy()
    for name, numerator in [('mean_lags', moments[6]), ('mean_distances', distance_sum)]:
        result[name] = np.divide(numerator, counts, out=np.full(size, np.nan), where=counts > 0).reshape(shape)
    return result
