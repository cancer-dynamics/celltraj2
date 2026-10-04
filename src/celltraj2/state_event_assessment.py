"""Frozen score-to-episode proposals and reviewed one-to-one event assessment.

These functions never confirm proposals or infer events/participants from graph
branches. Policy authoring/development and held-out assessment are separate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .state_supervision import _Contract, _choice, _integer, _key, _snapshot_keys, _text, _validate_runs, EventSet
from .state_supervised_targets import _active_records


@dataclass(frozen=True)
class EpisodePolicy(_Contract):
    SCHEMA = 'celltraj2.event_episode_policy.v1'
    threshold: float = 0.5
    minimum_support: int = 1
    maximum_gap_frames: int = 0
    minimum_duration_frames: int = 0
    refractory_frames: int = 0
    event_time: str = 'peak'
    tuning_role: str = 'development'
    tuning_evidence_digest: str = ''

    def __post_init__(self):
        if isinstance(self.threshold, bool) or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('Episode threshold must be in [0, 1]')
        _integer(self.minimum_support, 'minimum_support', 1)
        for name in ('maximum_gap_frames', 'minimum_duration_frames', 'refractory_frames'):
            _integer(getattr(self, name), name)
        _choice(self.event_time, {'peak', 'first', 'last'}, 'event-time convention')
        _choice(self.tuning_role, {'development', 'predeclared', 'training'}, 'episode tuning role')
        _text(self.tuning_evidence_digest, 'tuning evidence digest')


@dataclass(frozen=True)
class EventProposal(_Contract):
    SCHEMA = 'celltraj2.event_proposal.v1'
    proposal_id: str
    definition_id: str
    focal_key: object
    run_index: int
    start_frame: int
    end_frame: int
    event_frame: int
    score: float
    support_count: int
    policy_digest: str
    participant_keys: tuple = ()
    occurrence_status: str = 'unreviewed'

    def __post_init__(self):
        for name in ('proposal_id', 'definition_id', 'policy_digest'):
            _text(getattr(self, name), name)
        object.__setattr__(self, 'focal_key', _key(self.focal_key))
        object.__setattr__(self, 'participant_keys', tuple(_key(k) for k in self.participant_keys))
        for name in ('run_index', 'start_frame', 'end_frame', 'event_frame'):
            _integer(getattr(self, name), name)
        _integer(self.support_count, 'support_count', 1)
        if not self.start_frame <= self.event_frame <= self.end_frame:
            raise ValueError('Proposal event time must lie within its episode')
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError('Proposal score must be a finite probability')
        if self.occurrence_status != 'unreviewed':
            raise ValueError('Scored proposals are unreviewed, never confirmed truth')


@dataclass(frozen=True)
class EpisodeExtraction(_Contract):
    SCHEMA = 'celltraj2.event_episode_extraction.v1'
    policy: EpisodePolicy
    proposals: tuple[EventProposal, ...]
    suppressed: tuple[EventProposal, ...]
    missing_scores: int
    graph_fingerprint: str

    def __post_init__(self):
        if not isinstance(self.policy, EpisodePolicy):
            object.__setattr__(self, 'policy', EpisodePolicy.from_dict(self.policy))
        for name in ('proposals', 'suppressed'):
            object.__setattr__(self, name, tuple(x if isinstance(x, EventProposal) else EventProposal.from_dict(x) for x in getattr(self, name)))
        _integer(self.missing_scores, 'missing_scores')
        _text(self.graph_fingerprint, 'graph_fingerprint')


def extract_event_episodes(scores, snapshot_keys, frames, runs, *, definition_id,
                           policy, graph_fingerprint, participant_keys=None):
    """Group finite scores in safe unbranched runs; NaN always breaks continuity.

    Subthreshold observations may bridge at most maximum_gap_frames. Refractory
    handling greedily retains the strongest episode (earliest on ties), storing
    suppressed episodes separately. The supplied policy is immutable and must
    identify its training/development or predeclared provenance; test tuning is
    forbidden by the policy contract.
    """
    keys = _snapshot_keys(snapshot_keys)
    _validate_runs(runs, len(keys))
    _text(graph_fingerprint, 'graph_fingerprint')
    _text(definition_id, 'definition_id')
    policy = policy if isinstance(policy, EpisodePolicy) else EpisodePolicy.from_dict(policy)
    scores, frames = np.asarray(scores, dtype=float), np.asarray(frames)
    if scores.shape != (len(keys),) or frames.shape != scores.shape or frames.dtype.kind not in 'iu':
        raise ValueError('Scores and integer frames must align to snapshot keys')
    if np.any(np.isinf(scores)) or np.any((scores[np.isfinite(scores)] < 0) | (scores[np.isfinite(scores)] > 1)):
        raise ValueError('Scores must be probabilities or NaN missing values')
    participant_keys = participant_keys or {}
    proposals, suppressed = [], []
    for ri in range(len(runs.offsets) - 1):
        order = runs.order[runs.offsets[ri]:runs.offsets[ri + 1]]
        if np.any(np.diff(frames[order]) != 1) or any(keys[int(i)].partition != keys[int(order[0])].partition for i in order):
            raise ValueError('Episode runs must stop at frame gaps and partition boundaries')
        episodes, current = [], []
        for raw in order:
            row = int(raw)
            if not np.isfinite(scores[row]):
                if current:
                    episodes.append(current)
                    current = []
                continue
            if scores[row] >= policy.threshold:
                if current and frames[row] - frames[current[-1]] > policy.maximum_gap_frames + 1:
                    episodes.append(current)
                    current = []
                current.append(row)
        if current:
            episodes.append(current)
        candidates = []
        for rows in episodes:
            if len(rows) < policy.minimum_support or frames[rows[-1]] - frames[rows[0]] < policy.minimum_duration_frames:
                continue
            peak = max(rows, key=lambda i: (scores[i], -int(frames[i])))
            event = {'first': rows[0], 'last': rows[-1], 'peak': peak}[policy.event_time]
            # ID includes policy and portable identity through its stable digest.
            from .interpretation import canonical_json_digest
            proposal_id = 'episode-' + canonical_json_digest({'policy': policy.digest,
                'definition': definition_id, 'first': keys[rows[0]].to_dict(), 'last': keys[rows[-1]].to_dict()})
            candidates.append(EventProposal(proposal_id, definition_id, keys[event], ri,
                int(frames[rows[0]]), int(frames[rows[-1]]), int(frames[event]), float(scores[peak]), len(rows),
                policy.digest, tuple(participant_keys.get(keys[event], ()))))
        kept = []
        for proposal in sorted(candidates, key=lambda p: (-p.score, p.event_frame, p.proposal_id)):
            if any(abs(proposal.event_frame - other.event_frame) <= policy.refractory_frames for other in kept):
                suppressed.append(proposal)
            else:
                kept.append(proposal)
        proposals.extend(sorted(kept, key=lambda p: (p.event_frame, p.proposal_id)))
    return EpisodeExtraction(policy, tuple(proposals), tuple(suppressed),
        int(np.count_nonzero(~np.isfinite(scores[runs.order]))), graph_fingerprint)


@dataclass(frozen=True)
class EventMatch(_Contract):
    SCHEMA = 'celltraj2.event_match.v1'
    proposal_id: str
    event_id: str
    earliest_error_frames: int
    latest_error_frames: int
    earliest_error_s: float | None = None
    latest_error_s: float | None = None


@dataclass(frozen=True)
class EventAssessment(_Contract):
    SCHEMA = 'celltraj2.event_assessment.v1'
    event_set_digest: str
    matches: tuple[EventMatch, ...]
    false_positive_ids: tuple[str, ...]
    false_negative_ids: tuple[str, ...]
    duplicate_proposal_ids: tuple[str, ...]
    unreviewed_proposal_ids: tuple[str, ...]
    excluded_event_ids: tuple[str, ...]
    reviewed_exposure_frames: int | None
    timing_tolerance_frames: int
    participant_policy: str
    reviewed_exposure_s: float | None = None

    def __post_init__(self):
        object.__setattr__(self, 'matches', tuple(x if isinstance(x, EventMatch) else EventMatch.from_dict(x) for x in self.matches))
        for name in ('false_positive_ids', 'false_negative_ids', 'duplicate_proposal_ids', 'unreviewed_proposal_ids', 'excluded_event_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def metrics(self):
        tp, fp, fn = len(self.matches), len(self.false_positive_ids), len(self.false_negative_ids)
        return {'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
                'precision': tp / (tp + fp) if tp + fp else None,
                'recall': tp / (tp + fn) if tp + fn else None,
                'false_positives_per_reviewed_frame': fp / self.reviewed_exposure_frames if self.reviewed_exposure_frames else None,
                'false_positives_per_reviewed_second': fp / self.reviewed_exposure_s if self.reviewed_exposure_s else None,
                'unreviewed_proposals': len(self.unreviewed_proposal_ids),
                'duplicate_proposals': len(self.duplicate_proposal_ids),
                'excluded_events': len(self.excluded_event_ids)}


def assess_event_proposals(proposals, event_set, snapshot_keys, runs, *,
                           timing_tolerance_frames=0, participant_policy='require_complete',
                           frames=None, times_s=None):
    """Maximum-cardinality, deterministic one-to-one matching within same run.

    Interval truth yields an error interval, never an invented exact timestamp.
    Unmatched proposals count as false positives only in explicitly event-free
    exposure or in a truth event's tolerance region; other proposals are unknown.
    Rejected/uncertain proposals alone never supply negative exposure. Frame
    exposure is unioned independently per run and clipped to observed frames,
    including both reviewed bounds. Without frame metadata the exposure-rate
    metric is unavailable (proposal/event matching remains possible).
    """
    _integer(timing_tolerance_frames, 'timing_tolerance_frames')
    _choice(participant_policy, {'require_complete', 'occurrence_only', 'exact_keys'}, 'participant policy')
    keys = _snapshot_keys(snapshot_keys)
    _validate_runs(runs, len(keys))
    frame_values = None if frames is None else np.asarray(frames)
    if frame_values is not None and (frame_values.shape != (len(keys),) or frame_values.dtype.kind not in 'iu'):
        raise ValueError('Integer frames must align to snapshot keys')
    clock = None if times_s is None else np.asarray(times_s, dtype=float)
    if clock is not None and clock.shape != (len(keys),):
        raise ValueError('Physical time must align to snapshot keys')
    event_set = event_set if isinstance(event_set, EventSet) else EventSet.from_dict(event_set)
    proposals = tuple(x if isinstance(x, EventProposal) else EventProposal.from_dict(x) for x in proposals)
    if len({p.proposal_id for p in proposals}) != len(proposals):
        raise ValueError('Duplicate proposal identity')
    lookup = {key: i for i, key in enumerate(keys)}
    run_of = {keys[int(row)]: ri for ri in range(len(runs.offsets) - 1)
              for row in runs.order[runs.offsets[ri]:runs.offsets[ri + 1]]}
    if any(p.definition_id != event_set.definition_id or p.focal_key not in lookup or run_of.get(p.focal_key) != p.run_index for p in proposals):
        raise ValueError('Proposal definition or run identity mismatch')
    if frame_values is not None and any(int(frame_values[lookup[p.focal_key]]) != p.event_frame for p in proposals):
        raise ValueError('Proposal focal frame differs from its declared event time')
    active = _active_records(event_set.records)
    truth, excluded = [], []
    for event in active:
        if event.occurrence_status != 'confirmed':
            continue
        if (event.focal_key not in run_of or event.participant_status == 'conflicting' or
                (participant_policy != 'occurrence_only' and event.participant_status not in {'complete', 'not_applicable'})):
            excluded.append(event.event_id)
        else:
            truth.append(event)
    truth.sort(key=lambda e: (e.occurrence_start_frame, e.event_id))
    proposals = tuple(sorted(proposals, key=lambda p: (p.event_frame, p.proposal_id)))
    adjacency, reviewed_regions = [], []
    for proposal in proposals:
        candidates, in_reviewed_region = [], False
        for ti, event in enumerate(truth):
            if run_of[event.focal_key] != proposal.run_index:
                continue
            distance = max(event.occurrence_start_frame - proposal.event_frame,
                           proposal.event_frame - event.occurrence_end_frame, 0)
            if distance > timing_tolerance_frames:
                continue
            in_reviewed_region = True
            if participant_policy == 'exact_keys' and set(proposal.participant_keys) != {p.key for p in event.participants}:
                continue
            candidates.append((distance, event.event_id, ti))
        adjacency.append([ti for _, _, ti in sorted(candidates)])
        reviewed_regions.append(in_reviewed_region)
    # Augmenting paths maximize number of unique occurrences matched. A greedy
    # nearest match can undercount when tolerance intervals overlap.
    owner, assigned = {}, {}
    def augment(start):
        pending, visited, parent = deque([start]), set(), {}
        while pending:
            pi = pending.popleft()
            for ti in adjacency[pi]:
                if ti in visited:
                    continue
                visited.add(ti)
                parent[ti] = pi
                if ti not in owner:
                    while True:
                        pi = parent[ti]
                        previous = assigned.get(pi)
                        owner[ti], assigned[pi] = pi, ti
                        if previous is None:
                            return True
                        ti = previous
                pending.append(owner[ti])
        return False
    for pi in range(len(proposals)):
        augment(pi)
    matched = set(owner.values())
    matches = []
    for ti, pi in sorted(owner.items()):
        proposal, event = proposals[pi], truth[ti]
        time = None if clock is None else clock[lookup[proposal.focal_key]]
        physical = time is not None and np.isfinite(time) and event.occurrence_start_s is not None
        matches.append(EventMatch(proposal.proposal_id, event.event_id,
            proposal.event_frame - event.occurrence_end_frame,
            proposal.event_frame - event.occurrence_start_frame,
            float(time - event.occurrence_end_s) if physical else None,
            float(time - event.occurrence_start_s) if physical else None))
    intervals = {}
    for exposure in event_set.exposures:
        if exposure.status == 'event_free' and exposure.focal_key in run_of:
            intervals.setdefault(run_of[exposure.focal_key], []).append((exposure.start_frame, exposure.end_frame))
    exposure_frames = 0 if frame_values is not None else None
    for ri, bounds in intervals.items():
        merged = []
        for start, end in sorted(bounds):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        intervals[ri] = merged
        if frame_values is not None:
            observed = frame_values[runs.order[runs.offsets[ri]:runs.offsets[ri + 1]]]
            exposure_frames += sum(any(start <= int(frame) <= end for start, end in merged) for frame in observed)
    exposure_s = 0.0 if clock is not None else None
    if clock is not None:
        for ri in intervals:
            observed = clock[runs.order[runs.offsets[ri]:runs.offsets[ri + 1]]]
            reviews = [e for e in event_set.exposures if e.status == 'event_free' and run_of.get(e.focal_key) == ri]
            if (not np.isfinite(observed).all() or np.any(np.diff(observed) <= 0) or
                    any(e.start_s is None for e in reviews)):
                exposure_s = None
                break
            merged = []
            for start, end in sorted((max(float(observed[0]), e.start_s), min(float(observed[-1]), e.end_s)) for e in reviews):
                if end < start:
                    continue
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            exposure_s += sum(end - start for start, end in merged)
    fp, duplicate, unknown = [], [], []
    for pi, proposal in enumerate(proposals):
        if pi in matched:
            continue
        if adjacency[pi]:
            fp.append(proposal.proposal_id)
            duplicate.append(proposal.proposal_id)
        elif reviewed_regions[pi]:
            # A wrong participant assignment near reviewed truth is a false
            # proposal, not unreviewed territory or a duplicate match.
            fp.append(proposal.proposal_id)
        elif any(start <= proposal.event_frame <= end for start, end in intervals.get(proposal.run_index, ())):
            fp.append(proposal.proposal_id)
        else:
            unknown.append(proposal.proposal_id)
    return EventAssessment(event_set.digest, tuple(matches), tuple(fp),
        tuple(e.event_id for ti, e in enumerate(truth) if ti not in owner), tuple(duplicate), tuple(unknown),
        tuple(excluded), exposure_frames, timing_tolerance_frames, participant_policy, exposure_s)
