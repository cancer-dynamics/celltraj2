"""Reviewed event/detection and prospective fate target kernels.

Topology supplies admissible support, never event truth. All references are
final-window anchors; outcome support is kept separately from causal inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
from typing import Mapping

import numpy as np

from .state_supervision import (
    _Contract, _choice, _integer, _key, _mask, _snapshot_keys, _text,
    _validate_runs, EventDefinition, EventRecord, EventSet, SupervisionTask,
    project_state_references,
)


@dataclass(frozen=True)
class TargetSupportSpan(_Contract):
    """Inclusive/exclusive offsets into the projection's immutable Runs.order."""
    SCHEMA = 'celltraj2.target_support_span.v1'
    run_index: int
    start_offset: int
    stop_offset: int

    def __post_init__(self):
        for name in ('run_index', 'start_offset', 'stop_offset'):
            _integer(getattr(self, name), name)
        if self.stop_offset <= self.start_offset:
            raise ValueError('Support span must be nonempty')


@dataclass(frozen=True)
class TemporalTargetRow(_Contract):
    SCHEMA = 'celltraj2.temporal_target_row.v1'
    anchor_key: object
    snapshot_index: int
    window_index: int
    class_id: str | None
    status: str
    reason: str
    evidence_ids: tuple[str, ...] = ()
    source_unit_ids: tuple[str, ...] = ()
    input_span: TargetSupportSpan | None = None
    outcome_span: TargetSupportSpan | None = None
    evidence_spans: tuple[TargetSupportSpan, ...] = ()
    support_keys: tuple = ()
    origin_time_s: float | None = None
    outcome_end_s: float | None = None
    input_span_s: float | None = None
    weight: float = 0.0
    origin: str = 'reviewed_event'

    def __post_init__(self):
        object.__setattr__(self, 'anchor_key', _key(self.anchor_key))
        for name in ('snapshot_index', 'window_index'):
            _integer(getattr(self, name), name)
        _choice(self.status, {'eligible', 'unreviewed', 'excluded', 'censored', 'ambiguous', 'conflict'}, 'target status')
        for name in ('evidence_ids', 'source_unit_ids'):
            object.__setattr__(self, name, tuple(_text(x, name) for x in getattr(self, name)))
        object.__setattr__(self, 'support_keys', tuple(_key(k) for k in self.support_keys))
        for name in ('input_span', 'outcome_span'):
            value = getattr(self, name)
            if value is not None and not isinstance(value, TargetSupportSpan):
                object.__setattr__(self, name, TargetSupportSpan.from_dict(value))
        object.__setattr__(self, 'evidence_spans', tuple(x if isinstance(x, TargetSupportSpan)
            else TargetSupportSpan.from_dict(x) for x in self.evidence_spans))
        for name in ('origin_time_s', 'outcome_end_s', 'input_span_s'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f'{name} must be finite or unavailable')
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('Target weight must be finite and nonnegative')
        if self.status == 'eligible':
            _text(self.class_id, 'class_id')
            if not self.source_unit_ids or not self.evidence_ids or self.input_span is None:
                raise ValueError('Eligible targets require reviewed evidence, source units and input support')
        elif self.class_id is not None or self.weight:
            raise ValueError('Unavailable targets cannot have labels or training weight')


@dataclass(frozen=True)
class TemporalTargetProjection(_Contract):
    SCHEMA = 'celltraj2.temporal_target_projection.v1'
    task: SupervisionTask
    rows: tuple[TemporalTargetRow, ...]
    event_set_digest: str
    graph_fingerprint: str | None = None
    detection_range_frames: tuple[int, int] = (0, 0)
    require_complete_participants: bool = True
    competing_event_digests: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.task, SupervisionTask):
            object.__setattr__(self, 'task', SupervisionTask.from_dict(self.task))
        object.__setattr__(self, 'rows', tuple(x if isinstance(x, TemporalTargetRow) else TemporalTargetRow.from_dict(x) for x in self.rows))
        if len({x.anchor_key for x in self.rows}) != len(self.rows):
            raise ValueError('Duplicate temporal target anchor')
        _text(self.event_set_digest, 'event_set_digest')
        object.__setattr__(self, 'detection_range_frames', tuple(self.detection_range_frames))
        object.__setattr__(self, 'competing_event_digests', tuple(self.competing_event_digests))
        if self.task.family not in {'event_detection', 'fixed_horizon_fate'}:
            raise ValueError('Temporal projection requires event or fate semantics')
        if any(row.class_id is not None and row.class_id not in self.task.class_ids for row in self.rows):
            raise ValueError('Target class is outside the frozen task vocabulary')
        if not isinstance(self.require_complete_participants, bool):
            raise ValueError('require_complete_participants must be Boolean')

    @property
    def counts(self):
        return dict(Counter(row.status for row in self.rows))


def project_current_state_targets(task, evidence, snapshot_keys, windows, **kwargs):
    """Task-checked adapter over the existing final-anchor State projection."""
    if not isinstance(task, SupervisionTask):
        task = SupervisionTask.from_dict(task)
    if task.family != 'current_state' and not (task.family == 'event_detection' and task.detection_protocol == 'associated_state'):
        raise ValueError('This adapter requires current State or associated-State evidence')
    projection = project_state_references(evidence, snapshot_keys, windows,
        population_id=task.population_id, recipe_id=task.recipe_id, **kwargs)
    if any(row.class_id is not None and row.class_id not in task.class_ids for row in projection.rows):
        raise ValueError('Reference class is outside the frozen task vocabulary')
    return projection


def iter_target_support_keys(row, snapshot_keys, runs):
    """Yield all consumed support lazily for the shared grouped-split validator.

    Repetition is intentional: memory is bounded independently of delay/outcome
    length. No split is assigned here; callers must validate every yielded key.
    """
    for span in (row.input_span, row.outcome_span, *row.evidence_spans):
        if span is None:
            continue
        if (span.run_index >= len(runs.offsets) - 1 or
                span.start_offset < runs.offsets[span.run_index] or
                span.stop_offset > runs.offsets[span.run_index + 1]):
            raise ValueError('Target support does not fit the supplied run recipe')
        for offset in range(span.start_offset, span.stop_offset):
            yield snapshot_keys[int(runs.order[offset])]
    yield from row.support_keys


def _active_records(records):
    ids = [record.event_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate event identity')
    by_id = {record.event_id: record for record in records}
    complete = set()
    for event_id in ids:
        active, pending = set(), [(event_id, False)]
        while pending:
            current, finished = pending.pop()
            if finished:
                active.remove(current)
                complete.add(current)
            elif current in active:
                raise ValueError('Cyclic event supersession')
            elif current in by_id and current not in complete:
                active.add(current)
                pending.append((current, True))
                pending.extend((other, False) for other in by_id[current].supersedes)
    superseded = {x for record in records for x in record.supersedes}
    return tuple(record for record in records if record.event_id not in superseded)


def _bounds(record, physical):
    if hasattr(record, 'occurrence_start_frame'):
        return ((record.occurrence_start_s, record.occurrence_end_s) if physical else
                (record.occurrence_start_frame, record.occurrence_end_frame))
    return (record.start_s, record.end_s) if physical else (record.start_frame, record.end_frame)


def _coverage(exposures, begin, end, physical):
    """Find a deterministic reviewed event-free interval cover, with no gaps."""
    available = sorted(((_bounds(x, physical), x) for x in exposures
                        if x.status == 'event_free' and _bounds(x, physical)[0] is not None),
                       key=lambda x: (x[0][0], x[0][1], x[1].exposure_id))
    cursor, used = begin, []
    # A point target still needs explicit exposure at that point.
    for (left, right), exposure in available:
        if left <= cursor <= right and (right > cursor or begin == end):
            cursor = max(cursor, right)
            used.append(exposure)
            if cursor >= end:
                return tuple(used)
    return ()


def project_event_targets(task, snapshot_keys, frames, windows, event_set, *,
                          event_definition=None, times_s=None, reviewed_at_risk=None,
                          competing_events=(), detection_range_frames=(0, 0),
                          require_complete_participants=True, eligible=None,
                          feature_valid=None, graph_fingerprint=None,
                          source_fingerprints=None):
    """Build binary detection or (t, t+H] fate targets from reviewed truth.

    ``task.class_ids`` must explicitly order (negative, positive). Forecasts
    require ``reviewed_at_risk`` mapping portable anchor keys to review IDs, plus
    valid actual per-observation clocks. Event-free review must cover from the
    origin through the last observed frame before the first target event's
    earliest bound, or the whole negative horizon. The confirmed occurrence
    review supplies its endpoint. Continuous observed support reaches the latest event bound or
    horizon, including a bracketing observation for between-frame endpoints.

    Detection ranges are inclusive offsets (pre, post) from occurrence frames.
    Only the intersection valid for *every* possible occurrence is positive;
    the rest of the possible range is ambiguous. Events are never expanded into
    multiple occurrences. Superseded reviews do not count as independent truth.
    The caller pins source/run recipes and validates returned support using the
    shared split service before fitting. Frames may be pooled across cadences;
    forecasts never substitute frame counts for physical time.
    """
    task = task if isinstance(task, SupervisionTask) else SupervisionTask.from_dict(task)
    if task.family not in {'event_detection', 'fixed_horizon_fate'} or task.detection_protocol == 'associated_state':
        raise ValueError('Use the State evidence adapter for current/associated State targets')
    if len(task.class_ids) != 2:
        raise ValueError('Declare binary class_ids in (negative, positive) order')
    event_set = event_set if isinstance(event_set, EventSet) else EventSet.from_dict(event_set)
    if task.definition_id != event_set.definition_id:
        raise ValueError('Task and EventSet definition mismatch')
    if not isinstance(require_complete_participants, bool):
        raise ValueError('require_complete_participants must be Boolean')
    offsets = tuple(detection_range_frames)
    if len(offsets) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in offsets) or offsets[0] > offsets[1]:
        raise ValueError('Detection range requires ordered integer frame offsets')
    if task.detection_protocol == 'reviewed_anchor' and offsets != (0, 0):
        raise ValueError('Reviewed-anchor detection cannot use a target range')
    keys = _snapshot_keys(snapshot_keys)
    # Existing projection validates window bounds, duplicate anchors and masks.
    project_state_references((), keys, windows, population_id=task.population_id, recipe_id=task.recipe_id)
    frames = np.asarray(frames)
    if frames.shape != (len(keys),) or frames.dtype.kind not in 'iu' or np.any(frames < 0):
        raise ValueError('frames must be nonnegative integers aligned to snapshot keys')
    times = np.full(len(keys), np.nan) if times_s is None else np.asarray(times_s, dtype=float)
    if times.shape != (len(keys),):
        raise ValueError('times_s must align to snapshot keys')
    membership = _mask(eligible, len(keys), 'eligible')
    features = _mask(feature_valid, len(windows.starts), 'feature_valid')
    if reviewed_at_risk is not None and not isinstance(reviewed_at_risk, Mapping):
        raise ValueError('reviewed_at_risk must map anchor keys to review evidence IDs')
    at_risk = {_key(k): _text(v, 'at-risk review ID') for k, v in (reviewed_at_risk or {}).items()}
    physical = task.family == 'fixed_horizon_fate'
    runs = windows.runs
    lookup = {key: row for row, key in enumerate(keys)}
    position = {int(row): offset for offset, row in enumerate(runs.order)}
    run_of = {int(row): int(np.searchsorted(runs.offsets[1:], offset, side='right'))
              for row, offset in position.items()}
    # Reject malformed continuity rather than quietly treating a graph gap as follow-up.
    for run_index in range(len(runs.offsets) - 1):
        part = runs.order[runs.offsets[run_index]:runs.offsets[run_index + 1]]
        if np.any(np.diff(frames[part]) != 1) or any(keys[int(i)].partition != keys[int(part[0])].partition for i in part):
            raise ValueError('Target runs must stop at frame gaps and partition boundaries')
    # Prefix/segment indices keep support validation O(N + W log N), without
    # materializing observation-by-delay or observation-by-followup arrays.
    ordered_times = times[runs.order]
    bad_membership = np.r_[0, np.cumsum(~membership[runs.order])]
    bad_time = np.r_[0, np.cumsum(~np.isfinite(ordered_times))]
    bad_edge = np.r_[False, (~np.isfinite(np.diff(ordered_times))) | (np.diff(ordered_times) <= 0)]
    if len(bad_edge):
        bad_edge[runs.offsets[:-1]] = False
    bad_edge_prefix = np.r_[0, np.cumsum(bad_edge)]
    clock_stop = np.empty(len(runs.order), dtype=np.int64)
    for ri in range(len(runs.offsets) - 1):
        begin, stop = int(runs.offsets[ri]), int(runs.offsets[ri + 1])
        end = stop
        for position_index in range(stop - 1, begin - 1, -1):
            if position_index + 1 < stop and bad_edge[position_index + 1]:
                end = position_index + 1
            clock_stop[position_index] = end
    definition = event_definition
    if definition is not None:
        definition = definition if isinstance(definition, EventDefinition) else EventDefinition.from_dict(definition)
        if definition.definition_id != task.definition_id:
            raise ValueError('Event definition revision mismatch')
    competing = tuple(x if isinstance(x, EventRecord) else EventRecord.from_dict(x) for x in competing_events)
    if any(x.definition_id == task.definition_id for x in competing):
        raise ValueError('Target events cannot also be competing events')
    records = _active_records(event_set.records)
    competitors = _active_records(competing)
    by_run, free_by_run, competing_by_run = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in records:
        if definition is not None:
            record.validate_definition(definition)
        row = lookup.get(record.focal_key)
        if row in run_of:
            by_run[run_of[row]].append(record)
    for record in event_set.exposures:
        row = lookup.get(record.focal_key)
        if row in run_of:
            free_by_run[run_of[row]].append(record)
    for record in competitors:
        row = lookup.get(record.focal_key)
        if row in run_of:
            competing_by_run[run_of[row]].append(record)

    def stale(record):
        return ((source_fingerprints is not None and source_fingerprints.get(record.focal_key) != record.source_fingerprint) or
                (record.graph_fingerprint is not None and record.graph_fingerprint != graph_fingerprint))

    def participants_ok(record):
        return record.participant_status not in {'conflicting'} and (not require_complete_participants or record.participant_status in {'complete', 'not_applicable'})

    result = []
    for wi, anchor in enumerate(windows.anchors):
        anchor = int(anchor)
        key = keys[anchor]
        run_index = run_of[anchor]
        start, origin_offset = int(windows.starts[wi]), position[anchor]
        run_start, run_stop = int(runs.offsets[run_index]), int(runs.offsets[run_index + 1])
        input_membership_ok = bad_membership[origin_offset + 1] == bad_membership[start]
        input_clock_ok = (bad_time[origin_offset + 1] == bad_time[start] and
                          bad_edge_prefix[origin_offset + 1] == bad_edge_prefix[start + 1])
        row = TemporalTargetRow(key, anchor, wi, None, 'unreviewed', 'no_reviewed_target',
            input_span=TargetSupportSpan(run_index, start, origin_offset + 1),
            origin_time_s=float(times[anchor]) if np.isfinite(times[anchor]) else None,
            input_span_s=float(times[anchor] - ordered_times[start]) if input_clock_ok else None)
        events, exposures, rivals = by_run[run_index], free_by_run[run_index], competing_by_run[run_index]
        if not input_membership_ok or not features[wi]:
            result.append(replace(row, status='excluded', reason='outside_membership' if not input_membership_ok else 'unavailable_model_features'))
            continue
        used_events, used_exposures = [], []
        status, reason, label, endpoint = 'unreviewed', 'no_reviewed_target', None, None
        if physical:
            if key not in at_risk:
                result.append(replace(row, status='excluded', reason='at_risk_not_reviewed'))
                continue
            if not input_clock_ok:
                result.append(replace(row, status='excluded', reason='invalid_physical_input_time'))
                continue
            origin, horizon = float(times[anchor]), float(times[anchor] + task.horizon_s)
            endpoint = horizon
            confirmed = [e for e in events if e.occurrence_status == 'confirmed']
            unresolved = [e for e in events if e.occurrence_status in {'uncertain', 'unreviewed'}]
            if any(_bounds(e, True)[0] is None for e in confirmed + unresolved):
                status, reason = 'excluded', 'event_physical_time_unavailable'
            elif any(e.occurrence_end_s <= origin for e in confirmed):
                status, reason = 'excluded', 'prior_event_not_at_risk'
            else:
                candidates = sorted([e for e in confirmed if e.occurrence_end_s > origin and e.occurrence_start_s <= horizon], key=lambda e: (e.occurrence_start_s, e.occurrence_end_s, e.event_id))
                overlaps = [e for e in unresolved if e.occurrence_end_s > origin and e.occurrence_start_s <= horizon]
                event = candidates[0] if candidates else None
                used_events = [event] if event else []
                if event is not None and (event.occurrence_start_s <= origin or event.occurrence_end_s > horizon):
                    status, reason = 'ambiguous', 'occurrence_crosses_horizon_boundary'
                elif event is not None and not participants_ok(event):
                    status, reason = 'excluded', 'required_participants_unresolved'
                elif overlaps and (event is None or any(e.occurrence_start_s <= event.occurrence_end_s for e in overlaps)):
                    used_events += overlaps
                    status, reason = 'ambiguous', 'unresolved_occurrence_in_followup'
                else:
                    endpoint = event.occurrence_end_s if event else horizon
                    review_end = horizon
                    if event is not None:
                        clock = ordered_times[origin_offset:int(clock_stop[origin_offset])]
                        preceding = max(0, int(np.searchsorted(clock, event.occurrence_start_s, side='left')) - 1)
                        review_end = float(clock[min(preceding, len(clock) - 1)])
                    used_exposures = list(_coverage(exposures, origin, review_end, True))
                    if not used_exposures:
                        status, reason = 'censored', 'incomplete_reviewed_followup'
                    else:
                        status, reason = 'eligible', ''
                        label = task.class_ids[1 if event else 0]
                        # For a positive, reviewed absence reaches up to the
                        # earliest event boundary. A review extending beyond
                        # that boundary contradicts the occurrence evidence.
                        contradictory = [e for e in exposures if event is not None and e.status == 'event_free'
                            and e.start_s is not None and e.start_s <= event.occurrence_start_s < e.end_s]
                        if contradictory:
                            used_exposures += [e for e in contradictory if e not in used_exposures]
                            status, reason, label = 'conflict', 'event_and_event_free_review_disagree', None
                # Competing events censor outcomes unless target truth is strictly earlier.
                applicable_rivals = [e for e in rivals if e.occurrence_status != 'rejected']
                if any(_bounds(e, True)[0] is None for e in applicable_rivals):
                    status, reason, label = 'excluded', 'competing_event_time_unavailable', None
                else:
                    blockers = [e for e in applicable_rivals if e.occurrence_start_s <= endpoint]
                    if blockers:
                        used_events += blockers
                        status, reason, label = 'censored', 'competing_event', None
                censoring = [e for e in exposures if e.status == 'censored' and e.start_s is not None
                             and e.start_s <= origin <= e.end_s < endpoint]
                if censoring:
                    used_exposures += censoring
                    status, reason, label = 'censored', 'reviewed_followup_censored', None
        else:
            frame = int(frames[anchor])
            possible = [e for e in events if e.occurrence_status != 'rejected' and e.occurrence_start_frame + offsets[0] <= frame <= e.occurrence_end_frame + offsets[1]]
            positive = [e for e in possible if e.occurrence_status == 'confirmed' and
                        e.occurrence_end_frame + offsets[0] <= frame <= e.occurrence_start_frame + offsets[1]]
            used_events = possible
            if possible:
                if not positive or any(e.occurrence_status != 'confirmed' for e in possible):
                    status, reason = 'ambiguous', 'uncertain_occurrence_target_range'
                elif any(not participants_ok(e) for e in positive):
                    status, reason = 'excluded', 'required_participants_unresolved'
                else:
                    status, reason, label = 'eligible', '', task.class_ids[1]
                    contradictory = list(_coverage(exposures, frame, frame, False))
                    if contradictory:
                        used_exposures = contradictory
                        status, reason, label = 'conflict', 'event_and_event_free_review_disagree', None
            else:
                used_exposures = list(_coverage(exposures, frame, frame, False))
                if used_exposures:
                    status, reason, label = 'eligible', '', task.class_ids[0]
        # Include all inspected event/exposure focal and participant support.
        used = used_events + used_exposures
        if status == 'eligible' and any(stale(e) for e in used):
            status, reason, label = 'excluded', 'review_source_or_graph_changed', None
        outcome_span = None
        if physical and endpoint is not None and np.isfinite(times[anchor]):
            safe_stop = int(clock_stop[origin_offset])
            clock = ordered_times[origin_offset:safe_stop]
            relative_end = int(np.searchsorted(clock, endpoint, side='left'))
            reaches = relative_end < len(clock)
            stop = origin_offset + relative_end + 1 if reaches else safe_stop
            outcome_span = TargetSupportSpan(run_index, origin_offset, stop)
            if status == 'eligible' and (not reaches or bad_membership[stop] != bad_membership[origin_offset]):
                status, reason, label = 'censored', 'incomplete_observed_followup', None
        evidence_ids = tuple(dict.fromkeys(([at_risk[key]] if physical and key in at_risk else []) +
                             [getattr(e, 'event_id', None) or e.exposure_id for e in used]))
        units = tuple(dict.fromkeys([e.event_id for e in used_events if e.definition_id == task.definition_id]
                    if label == task.class_ids[1] else [e.exposure_id for e in used_exposures]))
        support = tuple(dict.fromkeys(k for e in used for k in (e.focal_key,) + tuple(p.key for p in getattr(e, 'participants', ()))))
        evidence_spans = []
        run_frames = frames[runs.order[run_start:run_stop]]
        for evidence in used:
            # Retain the complete reviewed interval as split support, including
            # its provenance path before an anchor and uncertainty after it.
            lo, hi = _bounds(evidence, False)
            begin = run_start + int(np.searchsorted(run_frames, lo, side='left'))
            end = run_start + int(np.searchsorted(run_frames, hi, side='right'))
            if begin < end:
                span = TargetSupportSpan(run_index, begin, end)
                if span not in evidence_spans:
                    evidence_spans.append(span)
        result.append(replace(row, status=status, reason=reason, class_id=label,
            evidence_ids=evidence_ids, source_unit_ids=units, support_keys=support,
            evidence_spans=tuple(evidence_spans),
            outcome_span=outcome_span, outcome_end_s=float(endpoint) if endpoint is not None else None,
            origin='reviewed_event' if used_events else 'reviewed_interval'))
    counts = Counter(unit for row in result if row.status == 'eligible' for unit in row.source_unit_ids)
    result = tuple(replace(row, weight=sum(1 / counts[u] for u in row.source_unit_ids)) if row.status == 'eligible' else row for row in result)
    return TemporalTargetProjection(task, result, event_set.digest, graph_fingerprint, offsets,
        require_complete_participants, tuple(e.digest for e in competitors))


def reviewed_at_risk_from_exposure(snapshot_keys, times_s, runs, event_set, *,
                                   source_fingerprints=None, graph_fingerprint=None):
    """Opt-in eligibility protocol: reviewed event-free exposure at each origin.

    Callers must explicitly choose and persist this policy. It does not establish
    recurrent-event eligibility or survival beyond the reviewed interval; the
    fate projector still checks first-event semantics and full outcome support.
    """
    keys = _snapshot_keys(snapshot_keys)
    _validate_runs(runs, len(keys))
    times = np.asarray(times_s, dtype=float)
    if times.shape != (len(keys),):
        raise ValueError('times_s must align to snapshot keys')
    event_set = event_set if isinstance(event_set, EventSet) else EventSet.from_dict(event_set)
    run_of = {keys[int(row)]: ri for ri in range(len(runs.offsets) - 1)
              for row in runs.order[runs.offsets[ri]:runs.offsets[ri + 1]]}
    by_run = defaultdict(list)
    for exposure in event_set.exposures:
        if (exposure.status != 'event_free' or exposure.start_s is None or
                exposure.focal_key not in run_of or
                (source_fingerprints is not None and source_fingerprints.get(exposure.focal_key) != exposure.source_fingerprint) or
                (exposure.graph_fingerprint is not None and exposure.graph_fingerprint != graph_fingerprint)):
            continue
        by_run[run_of[exposure.focal_key]].append(exposure)
    result = {}
    for i, key in enumerate(keys):
        if not np.isfinite(times[i]):
            continue
        choices = [e for e in by_run.get(run_of.get(key), ()) if e.start_s <= times[i] <= e.end_s]
        if choices:
            result[key] = min(choices, key=lambda e: e.exposure_id).exposure_id
    return result
