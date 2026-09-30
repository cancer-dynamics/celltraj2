"""Immutable checkpoint-C truth and compact temporal-reference contracts.

These kernels do not infer event truth from graph topology, fit estimators, or
publish releases. State evidence belongs to an observation, never to a delay.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
import math
from typing import ClassVar, Mapping, Sequence

import numpy as np

from .interpretation import ProjectObservationKey, _json_safe, canonical_json_digest
from .trajectory_index import FrameWindows, Runs


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def _optional_text(value, name):
    if value is not None:
        _text(value, name)


def _notes(instance, *names):
    for name in names:
        if not isinstance(getattr(instance, name), str):
            raise ValueError(f'{name} must be text')


def _key(value):
    return value if isinstance(value, ProjectObservationKey) else ProjectObservationKey.from_dict(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _choice(value, allowed, name):
    if value not in allowed:
        raise ValueError(f'Unsupported {name}: {value!r}')


def _interval(start, end, name):
    _integer(start, f'{name} start')
    _integer(end, f'{name} end')
    if end < start:
        raise ValueError(f'{name} end precedes start')


def _time_interval(start, end):
    if (start is None) != (end is None):
        raise ValueError('Both physical occurrence bounds must be supplied together')
    if start is not None:
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                   for x in (start, end)) or end < start:
            raise ValueError('Physical occurrence bounds must be finite and ordered')


class _Contract:
    SCHEMA: ClassVar[str]

    def to_dict(self):
        def plain(value):
            # Scientific callers routinely supply NumPy frame/count scalars.
            # Keep the public payload ordinary JSON without modifying evidence.
            if isinstance(value, np.generic):
                return plain(value.item())
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value
        return {'schema': self.SCHEMA, **plain(_json_safe(self))}

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if values.pop('schema', cls.SCHEMA) != cls.SCHEMA:
            raise ValueError(f'Unsupported {cls.__name__} schema')
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown {cls.__name__} fields: {sorted(unknown)}')
        return cls(**values)

    @property
    def digest(self):
        return canonical_json_digest(self.to_dict())


@dataclass(frozen=True)
class EventRole(_Contract):
    SCHEMA = 'celltraj2.event_role.v1'
    name: str
    minimum: int = 0
    maximum: int | None = None
    allow_outside_membership: bool = False

    def __post_init__(self):
        _text(self.name, 'role name')
        _integer(self.minimum, 'minimum participants')
        if self.maximum is not None and _integer(self.maximum, 'maximum participants') < self.minimum:
            raise ValueError('Maximum participants is less than minimum')
        if not isinstance(self.allow_outside_membership, bool):
            raise ValueError('allow_outside_membership must be Boolean')


@dataclass(frozen=True)
class EventDefinition(_Contract):
    SCHEMA = 'celltraj2.event_definition.v1'
    definition_id: str
    family_id: str
    name: str
    roles: tuple[EventRole, ...] = ()
    criteria: str = ''
    timing_convention: str = 'reviewed_interval'
    focal_semantics: str = 'annotated cell'

    def __post_init__(self):
        for name in ('definition_id', 'family_id', 'name', 'focal_semantics'):
            _text(getattr(self, name), name)
        _choice(self.timing_convention, {'reviewed_interval', 'reviewed_anchor'}, 'timing convention')
        _notes(self, 'criteria')
        roles = tuple(x if isinstance(x, EventRole) else EventRole.from_dict(x) for x in self.roles)
        if len({x.name for x in roles}) != len(roles):
            raise ValueError('Duplicate event participant role')
        object.__setattr__(self, 'roles', roles)


@dataclass(frozen=True)
class EventParticipant(_Contract):
    SCHEMA = 'celltraj2.event_participant.v1'
    role: str
    key: ProjectObservationKey

    def __post_init__(self):
        _text(self.role, 'participant role')
        object.__setattr__(self, 'key', _key(self.key))


@dataclass(frozen=True)
class EventRecord(_Contract):
    SCHEMA = 'celltraj2.event_record.v1'
    event_id: str
    definition_id: str
    focal_key: ProjectObservationKey
    occurrence_start_frame: int
    occurrence_end_frame: int
    source_fingerprint: str
    occurrence_status: str = 'unreviewed'
    participant_status: str = 'not_applicable'
    participants: tuple[EventParticipant, ...] = ()
    occurrence_start_s: float | None = None
    occurrence_end_s: float | None = None
    graph_fingerprint: str | None = None
    review_revision: str = ''
    note: str = ''
    reviewer: str = ''
    proposal_id: str | None = None
    supersedes: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('event_id', 'definition_id', 'source_fingerprint'):
            _text(getattr(self, name), name)
        _notes(self, 'review_revision', 'note', 'reviewer')
        _optional_text(self.graph_fingerprint, 'graph_fingerprint')
        _optional_text(self.proposal_id, 'proposal_id')
        object.__setattr__(self, 'focal_key', _key(self.focal_key))
        _interval(self.occurrence_start_frame, self.occurrence_end_frame, 'occurrence')
        _time_interval(self.occurrence_start_s, self.occurrence_end_s)
        _choice(self.occurrence_status, {'confirmed', 'rejected', 'uncertain', 'unreviewed'}, 'occurrence status')
        _choice(self.participant_status, {'complete', 'partial', 'conflicting', 'not_applicable'}, 'participant status')
        participants = tuple(x if isinstance(x, EventParticipant) else EventParticipant.from_dict(x)
                             for x in self.participants)
        if len(set((x.role, x.key) for x in participants)) != len(participants):
            raise ValueError('Duplicate event participant')
        if any(x.key.project_uuid != self.focal_key.project_uuid for x in participants):
            raise ValueError('Event participants must belong to the focal project')
        if self.participant_status == 'not_applicable' and participants:
            raise ValueError('Participants cannot accompany not_applicable status')
        object.__setattr__(self, 'participants', participants)
        supersedes = tuple(_text(x, 'superseded event ID') for x in self.supersedes)
        if len(set(supersedes)) != len(supersedes) or self.event_id in supersedes:
            raise ValueError('Invalid event supersession')
        object.__setattr__(self, 'supersedes', supersedes)

    def validate_definition(self, definition, *, eligible_keys=None):
        """Check declared roles and optional membership; never infer participants."""
        if self.definition_id != definition.definition_id:
            raise ValueError('Event definition revision mismatch')
        if definition.timing_convention == 'reviewed_anchor' and self.occurrence_start_frame != self.occurrence_end_frame:
            raise ValueError('Reviewed-anchor definition requires one occurrence frame')
        roles = {r.name: r for r in definition.roles}
        counts = Counter(p.role for p in self.participants)
        if set(counts) - set(roles):
            raise ValueError('Event contains an undeclared participant role')
        for name, role in roles.items():
            if role.maximum is not None and counts[name] > role.maximum:
                raise ValueError(f'Too many participants for role {name}')
            if self.participant_status == 'complete' and counts[name] < role.minimum:
                raise ValueError(f'Incomplete participants for role {name}')
            if self.participant_status == 'not_applicable' and role.minimum:
                raise ValueError('Required participant roles are unresolved, not not_applicable')
        if eligible_keys is not None:
            eligible = set(eligible_keys)
            if self.focal_key not in eligible:
                raise ValueError('Event focal observation is outside channel membership')
            if any(p.key not in eligible and not roles[p.role].allow_outside_membership for p in self.participants):
                raise ValueError('Participant role does not allow outside-membership context')


@dataclass(frozen=True)
class ReviewedExposure(_Contract):
    """A deliberate review interval; rejecting a proposal creates no exposure."""
    SCHEMA = 'celltraj2.reviewed_exposure.v1'
    exposure_id: str
    definition_id: str
    focal_key: ProjectObservationKey
    start_frame: int
    end_frame: int
    source_fingerprint: str
    status: str = 'event_free'
    graph_fingerprint: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    review_revision: str = ''
    note: str = ''
    censor_reason: str | None = None

    def __post_init__(self):
        for name in ('exposure_id', 'definition_id', 'source_fingerprint'):
            _text(getattr(self, name), name)
        _notes(self, 'review_revision', 'note')
        _optional_text(self.graph_fingerprint, 'graph_fingerprint')
        object.__setattr__(self, 'focal_key', _key(self.focal_key))
        _interval(self.start_frame, self.end_frame, 'exposure')
        _time_interval(self.start_s, self.end_s)
        _choice(self.status, {'event_free', 'censored', 'unreviewed'}, 'exposure status')
        if self.status == 'censored':
            _text(self.censor_reason, 'censor reason')
        elif self.censor_reason is not None:
            raise ValueError('Only censored exposure may have a censor reason')


@dataclass(frozen=True)
class EventSet(_Contract):
    """One immutable review revision, including separately reviewed exposure."""
    SCHEMA = 'celltraj2.event_set.v1'
    event_set_id: str
    definition_id: str
    records: tuple[EventRecord, ...] = ()
    exposures: tuple[ReviewedExposure, ...] = ()
    review_revision: str = ''

    def __post_init__(self):
        _text(self.event_set_id, 'event_set_id')
        _text(self.definition_id, 'definition_id')
        _notes(self, 'review_revision')
        records = tuple(x if isinstance(x, EventRecord) else EventRecord.from_dict(x) for x in self.records)
        exposures = tuple(x if isinstance(x, ReviewedExposure) else ReviewedExposure.from_dict(x) for x in self.exposures)
        if len({x.event_id for x in records}) != len(records):
            raise ValueError('EventSet contains duplicate occurrence IDs')
        if len({x.exposure_id for x in exposures}) != len(exposures):
            raise ValueError('EventSet contains duplicate exposure IDs')
        if any(x.definition_id != self.definition_id for x in records + exposures):
            raise ValueError('EventSet definition revision mismatch')
        if len({x.focal_key.project_uuid for x in records + exposures}) > 1:
            raise ValueError('EventSet cannot combine different projects')
        object.__setattr__(self, 'records', records)
        object.__setattr__(self, 'exposures', exposures)


@dataclass(frozen=True)
class SupervisionTask(_Contract):
    """Task semantics only; no estimator or release-binding completion claim."""
    SCHEMA = 'celltraj2.state_supervision_task.v1'
    task_id: str
    channel_id: str
    definition_id: str
    family: str
    population_id: str
    recipe_id: str
    class_ids: tuple[str, ...] = ()
    horizon_s: float | None = None
    temporal_direction: str = 'causal'
    detection_protocol: str | None = None
    annotation_anchor: str = 'final'
    negative_policy: str = 'explicit_review'
    at_risk_policy: str = 'reviewed_eligible'
    competing_event_policy: str = 'censor'

    def __post_init__(self):
        for name in ('task_id', 'channel_id', 'definition_id', 'population_id', 'recipe_id'):
            _text(getattr(self, name), name)
        _choice(self.family, {'current_state', 'event_detection', 'fixed_horizon_fate'}, 'task family')
        _choice(self.temporal_direction, {'causal', 'retrospective'}, 'temporal direction')
        if self.annotation_anchor != 'final' or self.negative_policy != 'explicit_review':
            raise ValueError('Only final-anchor annotations and explicitly reviewed negatives are supported')
        classes = tuple(_text(c, 'class ID') for c in self.class_ids)
        if len(set(classes)) != len(classes):
            raise ValueError('Duplicate task class ID')
        object.__setattr__(self, 'class_ids', classes)
        if self.family == 'current_state':
            if len(classes) < 2:
                raise ValueError('A supervised State task requires at least two exclusive classes')
        elif classes and len(classes) != 2:
            raise ValueError('Initial event/fate tasks are binary')
        if self.family == 'fixed_horizon_fate':
            if self.temporal_direction != 'causal':
                raise ValueError('A prospective fate task requires causal predictors')
            if isinstance(self.horizon_s, bool) or not isinstance(self.horizon_s, (int, float)) or not math.isfinite(self.horizon_s) or self.horizon_s <= 0:
                raise ValueError('A fixed-horizon fate task requires positive finite horizon_s')
        elif self.horizon_s is not None:
            raise ValueError('Only fate tasks have a future horizon')
        if self.family == 'event_detection':
            _choice(self.detection_protocol, {'associated_state', 'reviewed_anchor', 'reviewed_range'}, 'detection protocol')
        elif self.detection_protocol is not None:
            raise ValueError('Detection protocol belongs only to event detection')
        if self.at_risk_policy != 'reviewed_eligible' or self.competing_event_policy != 'censor':
            raise ValueError('Unsupported at-risk or competing-event policy')


@dataclass(frozen=True)
class StateAnnotation(_Contract):
    SCHEMA = 'celltraj2.state_annotation_reference.v1'
    evidence_id: str
    key: ProjectObservationKey
    class_id: str | None = None
    status: str = 'known'
    origin: str = 'direct'
    source_unit_id: str | None = None
    source_fingerprint: str | None = None
    graph_fingerprint: str | None = None
    seed_evidence_id: str | None = None

    def __post_init__(self):
        _text(self.evidence_id, 'evidence_id')
        for name in ('source_fingerprint', 'graph_fingerprint', 'seed_evidence_id'):
            _optional_text(getattr(self, name), name)
        object.__setattr__(self, 'key', _key(self.key))
        _choice(self.status, {'known', 'uncertain', 'unreviewed', 'excluded', 'censored'}, 'annotation status')
        _choice(self.origin, {'direct', 'reviewed_interval', 'derived_extension', 'rule', 'model'}, 'annotation origin')
        if self.status == 'known':
            _text(self.class_id, 'known annotation class_id')
        elif self.class_id is not None:
            raise ValueError('Uncertain/unreviewed/excluded/censored status is not a class label')
        object.__setattr__(self, 'source_unit_id', self.source_unit_id or self.evidence_id)
        _text(self.source_unit_id, 'source_unit_id')
        if self.origin == 'derived_extension':
            _text(self.seed_evidence_id, 'extension seed ID')
            _text(self.graph_fingerprint, 'extension graph fingerprint')


@dataclass(frozen=True)
class StateReferenceRow(_Contract):
    SCHEMA = 'celltraj2.state_reference_row.v1'
    anchor_key: ProjectObservationKey
    snapshot_index: int | None
    window_index: int | None
    class_id: str | None
    status: str
    reason: str
    evidence_ids: tuple[str, ...]
    source_unit_ids: tuple[str, ...]
    origin: str
    weight: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, 'anchor_key', _key(self.anchor_key))
        object.__setattr__(self, 'evidence_ids', tuple(self.evidence_ids))
        object.__setattr__(self, 'source_unit_ids', tuple(self.source_unit_ids))
        if self.snapshot_index is not None:
            _integer(self.snapshot_index, 'snapshot_index')
        if self.window_index is not None:
            _integer(self.window_index, 'window_index')
        _choice(self.status, {'eligible', 'unmapped', 'conflict', 'uncertain', 'unreviewed', 'excluded', 'censored'}, 'reference status')
        _optional_text(self.class_id, 'class_id')
        _notes(self, 'reason', 'origin')
        if not self.evidence_ids or not self.source_unit_ids:
            raise ValueError('Reference rows require evidence and source units')
        for value in self.evidence_ids + self.source_unit_ids:
            _text(value, 'evidence/source unit ID')
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('Reference weight must be nonnegative and finite')
        if self.status == 'eligible' and (self.window_index is None or self.class_id is None):
            raise ValueError('Eligible reference needs a window and class')
        if self.status != 'eligible' and self.weight:
            raise ValueError('Ineligible references cannot carry training weight')


@dataclass(frozen=True)
class StateReferenceProjection(_Contract):
    SCHEMA = 'celltraj2.state_reference_projection.v1'
    population_id: str
    recipe_id: str
    rows: tuple[StateReferenceRow, ...]
    graph_fingerprint: str | None = None
    annotation_anchor: str = 'final'

    def __post_init__(self):
        _text(self.population_id, 'population_id')
        _text(self.recipe_id, 'recipe_id')
        if self.annotation_anchor != 'final':
            raise ValueError('State references require final anchors')
        rows = tuple(x if isinstance(x, StateReferenceRow) else StateReferenceRow.from_dict(x) for x in self.rows)
        if len(set(x.anchor_key for x in rows)) != len(rows):
            raise ValueError('Duplicate projection anchor')
        object.__setattr__(self, 'rows', rows)


def _snapshot_keys(snapshot_keys):
    keys = tuple(_key(k) for k in snapshot_keys)
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate snapshot observation identity')
    return keys


def _mask(values, size, name):
    if values is None:
        return np.ones(size, dtype=bool)
    result = np.asarray(values)
    if result.shape != (size,) or result.dtype.kind != 'b':
        raise ValueError(f'{name} must be a Boolean vector of length {size}')
    return result


def _validate_runs(runs, count):
    order, offsets = np.asarray(runs.order), np.asarray(runs.offsets)
    if order.ndim != 1 or offsets.ndim != 1 or order.dtype.kind not in 'iu' or offsets.dtype.kind not in 'iu':
        raise ValueError('Runs require one-dimensional integer references')
    if not len(offsets) or offsets[0] != 0 or offsets[-1] != len(order) or np.any(np.diff(offsets) <= 0):
        if not (len(offsets) == 1 and offsets[0] == 0 and len(order) == 0):
            raise ValueError('Invalid run offsets')
    if np.any(order < 0) or np.any(order >= count) or len(np.unique(order)) != len(order):
        raise ValueError('Runs contain duplicate/out-of-range snapshot rows')


def project_state_references(evidence, snapshot_keys, windows: FrameWindows, *,
                             population_id, recipe_id, eligible=None,
                             feature_valid=None, source_fingerprints=None,
                             graph_fingerprint=None):
    """Map only final anchors, retaining unmatched evidence with explicit reasons.

    ``feature_valid`` is a per-window mask for the chosen modeling recipe, not
    the plot axes. No window members/features are copied. Membership and source
    checks are independent of Type leaf identity. Each source-evidence unit has
    total weight one across its eligible derived anchors.
    """
    keys = _snapshot_keys(snapshot_keys)
    _validate_runs(windows.runs, len(keys))
    _integer(windows.length, 'window length', 1)
    starts = np.asarray(windows.starts)
    if starts.ndim != 1 or starts.dtype.kind not in 'iu' or np.any(starts < 0):
        raise ValueError('Invalid window starts')
    run = np.searchsorted(windows.runs.offsets[1:], starts, side='right')
    if np.any(run >= len(windows.runs.offsets) - 1):
        raise ValueError('Window starts outside runs')
    if np.any(starts + windows.length > windows.runs.offsets[run + 1]):
        raise ValueError('A window crosses a run boundary')
    anchors = windows.anchors
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError('Duplicate window final anchor')
    membership = _mask(eligible, len(keys), 'eligible')
    valid = _mask(feature_valid, len(starts), 'feature_valid')
    # Whole-window membership via a prefix sum, O(N+W), not O(N*delay).
    bad = np.r_[0, np.cumsum(~membership[windows.runs.order])]
    window_eligible = bad[starts + windows.length] == bad[starts]
    snapshot_lookup = {key: i for i, key in enumerate(keys)}
    anchor_lookup = {int(row): i for i, row in enumerate(anchors)}
    claims = defaultdict(list)
    ids = set()
    for value in evidence:
        claim = value if isinstance(value, StateAnnotation) else StateAnnotation.from_dict(value)
        if claim.evidence_id in ids:
            raise ValueError('Duplicate evidence ID in reference input')
        ids.add(claim.evidence_id)
        claims[claim.key].append(claim)
    pending = []
    priority = {'direct': 0, 'reviewed_interval': 0, 'derived_extension': 1, 'rule': 2, 'model': 2}
    for key, all_claims in claims.items():
        rank = min(priority[c.origin] for c in all_claims)
        chosen = [c for c in all_claims if priority[c.origin] == rank]
        row = snapshot_lookup.get(key)
        window = anchor_lookup.get(row)
        labels = {c.class_id for c in chosen if c.status == 'known'}
        statuses = {c.status for c in chosen}
        status, reason, label = 'eligible', '', next(iter(labels)) if len(labels) == 1 else None
        if rank == 2:
            status, reason = 'excluded', 'origin_requires_explicit_task_policy'
        elif row is None:
            status, reason = 'unmapped', 'outside_current_population'
        elif source_fingerprints is not None and any(
                c.source_fingerprint is not None and c.source_fingerprint != source_fingerprints.get(key)
                for c in chosen):
            status, reason = 'unmapped', 'source_identity_changed'
        elif any(c.origin == 'derived_extension' and c.graph_fingerprint != graph_fingerprint for c in chosen):
            status, reason = 'unmapped', 'extension_graph_changed'
        elif not membership[row] or (window is not None and not window_eligible[window]):
            status, reason = 'unmapped', 'outside_membership'
        elif window is None:
            status, reason = 'unmapped', 'no_eligible_final_anchor_window'
        elif not valid[window]:
            status, reason = 'unmapped', 'unavailable_model_features'
        elif len(labels) > 1 or len(statuses) > 1:
            status, reason, label = 'conflict', 'opposing_annotation_claims', None
        elif statuses != {'known'}:
            status, reason, label = next(iter(statuses)), 'not_a_reviewed_class', None
        pending.append(dict(anchor_key=key, snapshot_index=row, window_index=window,
                            class_id=label, status=status, reason=reason,
                            evidence_ids=tuple(c.evidence_id for c in chosen),
                            source_unit_ids=tuple(sorted({c.source_unit_id for c in chosen})),
                            origin=chosen[0].origin))
    units = Counter(unit for row in pending if row['status'] == 'eligible' for unit in row['source_unit_ids'])
    rows = tuple(StateReferenceRow(**row, weight=sum(1 / units[u] for u in row['source_unit_ids'])
                                   if row['status'] == 'eligible' else 0.0) for row in pending)
    return StateReferenceProjection(population_id, recipe_id, rows, graph_fingerprint)


@dataclass(frozen=True)
class ExtensionPolicy(_Contract):
    SCHEMA = 'celltraj2.state_extension_policy.v1'
    backward_frames: int = 0
    forward_frames: int = 0
    start_frame: int | None = None
    end_frame: int | None = None
    max_time_s: float | None = None

    def __post_init__(self):
        _integer(self.backward_frames, 'backward_frames')
        _integer(self.forward_frames, 'forward_frames')
        if self.start_frame is not None:
            _integer(self.start_frame, 'start_frame')
        if self.end_frame is not None:
            _integer(self.end_frame, 'end_frame')
        if self.start_frame is not None and self.end_frame is not None:
            _interval(self.start_frame, self.end_frame, 'extension')
        if self.max_time_s is not None and (isinstance(self.max_time_s, bool) or
                not isinstance(self.max_time_s, (int, float)) or not math.isfinite(self.max_time_s) or self.max_time_s < 0):
            raise ValueError('max_time_s must be finite and nonnegative')


@dataclass(frozen=True)
class ExtensionSpan(_Contract):
    """Support is a compact inclusive/exclusive slice of an immutable run recipe."""
    SCHEMA = 'celltraj2.state_extension_span.v1'
    seed_evidence_id: str
    run_index: int | None
    start_offset: int | None
    stop_offset: int | None
    backward_stop: str
    forward_stop: str

    def __post_init__(self):
        _text(self.seed_evidence_id, 'seed_evidence_id')
        _text(self.backward_stop, 'backward_stop')
        _text(self.forward_stop, 'forward_stop')
        bounds = (self.run_index, self.start_offset, self.stop_offset)
        if any(value is None for value in bounds):
            if not all(value is None for value in bounds):
                raise ValueError('Unavailable extension span must omit all run bounds')
        else:
            for name, value in zip(('run_index', 'start_offset', 'stop_offset'), bounds):
                _integer(value, name)
            if self.stop_offset <= self.start_offset:
                raise ValueError('Extension span must include its seed')


@dataclass(frozen=True)
class ExtensionProjection(_Contract):
    SCHEMA = 'celltraj2.state_extension_projection.v1'
    annotations: tuple[StateAnnotation, ...]
    spans: tuple[ExtensionSpan, ...]
    policy: ExtensionPolicy
    graph_fingerprint: str

    def __post_init__(self):
        _text(self.graph_fingerprint, 'graph_fingerprint')
        object.__setattr__(self, 'annotations', tuple(x if isinstance(x, StateAnnotation) else StateAnnotation.from_dict(x) for x in self.annotations))
        object.__setattr__(self, 'spans', tuple(x if isinstance(x, ExtensionSpan) else ExtensionSpan.from_dict(x) for x in self.spans))
        if not isinstance(self.policy, ExtensionPolicy):
            object.__setattr__(self, 'policy', ExtensionPolicy.from_dict(self.policy))


def extend_state_annotations(evidence, snapshot_keys, frames, runs: Runs, *, policy,
                             graph_fingerprint, source_fingerprint=None,
                             times_s=None, eligible=None, splits=None):
    """Finite extension inside prevalidated graph runs, never across branches.

    The caller builds runs with GraphIndex.project and pins their graph/source
    recipe. This also checks partitions, adjacent frames, eligibility and an
    optional complete split map. Return originals plus derived claims; only
    direct/reviewed known evidence seeds extensions. A derived claim never seeds
    another extension. Bounds default to the source observation (zero extension).
    """
    _text(graph_fingerprint, 'graph_fingerprint')
    if not isinstance(policy, ExtensionPolicy):
        policy = ExtensionPolicy.from_dict(policy)
    keys = _snapshot_keys(snapshot_keys)
    _validate_runs(runs, len(keys))
    frames = np.asarray(frames)
    if frames.shape != (len(keys),) or frames.dtype.kind not in 'iu' or np.any(frames < 0):
        raise ValueError('frames must be nonnegative integers aligned to snapshot keys')
    membership = _mask(eligible, len(keys), 'eligible')
    if times_s is not None:
        times_s = np.asarray(times_s, dtype=float)
        if times_s.shape != (len(keys),):
            raise ValueError('times_s must align to snapshot keys')
    if policy.max_time_s is not None and times_s is None:
        raise ValueError('Physical-time extension requires time metadata')
    if splits is not None:
        if isinstance(splits, Mapping):
            if any(key not in splits or splits[key] in (None, '') for key in keys):
                raise ValueError('Supervised extension requires a complete split map')
            split_values = [splits[k] for k in keys]
        else:
            split_values = list(splits)
            if len(split_values) != len(keys) or any(x in (None, '') for x in split_values):
                raise ValueError('Supervised extension requires a complete split map')
    else:
        split_values = None
    originals = tuple(x if isinstance(x, StateAnnotation) else StateAnnotation.from_dict(x) for x in evidence)
    if len({x.evidence_id for x in originals}) != len(originals):
        raise ValueError('Duplicate evidence ID in extension input')
    lookup = {key: i for i, key in enumerate(keys)}
    position = {int(row): i for i, row in enumerate(runs.order)}
    derived, spans = [], []
    for seed in originals:
        if seed.origin not in {'direct', 'reviewed_interval'} or seed.status != 'known':
            continue
        row = lookup.get(seed.key)
        pos = position.get(row)
        if pos is None or not membership[row]:
            spans.append(ExtensionSpan(seed.evidence_id, None, None, None, 'outside_current_runs', 'outside_current_runs'))
            continue
        if source_fingerprint is not None and seed.source_fingerprint is not None and seed.source_fingerprint != source_fingerprint:
            spans.append(ExtensionSpan(seed.evidence_id, None, None, None, 'source_identity_changed', 'source_identity_changed'))
            continue
        if policy.max_time_s is not None and not np.isfinite(times_s[row]):
            spans.append(ExtensionSpan(seed.evidence_id, None, None, None, 'unknown_physical_time', 'unknown_physical_time'))
            continue
        run_index = int(np.searchsorted(runs.offsets[1:], pos, side='right'))
        low, high = int(runs.offsets[run_index]), int(runs.offsets[run_index + 1])
        start = policy.start_frame if policy.start_frame is not None else int(frames[row]) - policy.backward_frames
        end = policy.end_frame if policy.end_frame is not None else int(frames[row]) + policy.forward_frames
        if not start <= frames[row] <= end:
            raise ValueError('Extension interval must contain its seed observation')
        def walk(direction):
            last = pos
            while True:
                candidate = last + direction
                if candidate < low or candidate >= high:
                    return last, 'run_boundary'
                current_row, next_row = int(runs.order[last]), int(runs.order[candidate])
                next_key = keys[next_row]
                if next_key.project_uuid != seed.key.project_uuid or next_key.partition != seed.key.partition:
                    return last, 'partition_boundary'
                if frames[next_row] - frames[current_row] != direction:
                    return last, 'frame_gap'
                if not membership[next_row]:
                    return last, 'membership_boundary'
                if split_values is not None and split_values[next_row] != split_values[row]:
                    return last, 'split_boundary'
                if not start <= frames[next_row] <= end:
                    return last, 'reviewed_bound'
                if policy.max_time_s is not None:
                    if not np.isfinite(times_s[next_row]) or (times_s[next_row] - times_s[current_row]) * direction <= 0:
                        return last, 'unknown_or_nonmonotone_time'
                    if abs(times_s[next_row] - times_s[row]) > policy.max_time_s:
                        return last, 'physical_time_bound'
                last = candidate
        first, backward_stop = walk(-1)
        last, forward_stop = walk(1)
        spans.append(ExtensionSpan(seed.evidence_id, run_index, first, last + 1, backward_stop, forward_stop))
        for offset in range(first, last + 1):
            if offset == pos:
                continue
            key = keys[int(runs.order[offset])]
            eid = canonical_json_digest({'seed': seed.evidence_id, 'key': key.to_dict(),
                                         'policy': policy.to_dict(), 'graph': graph_fingerprint})
            derived.append(StateAnnotation(eid, key, seed.class_id, origin='derived_extension',
                                            source_unit_id=seed.source_unit_id,
                                            source_fingerprint=source_fingerprint or seed.source_fingerprint,
                                            graph_fingerprint=graph_fingerprint, seed_evidence_id=seed.evidence_id))
    return ExtensionProjection(originals + tuple(derived), tuple(spans), policy, graph_fingerprint)
