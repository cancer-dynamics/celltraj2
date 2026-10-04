import json
import unittest
from dataclasses import replace
from uuid import UUID

import numpy as np

from celltraj2.interpretation import ProjectObservationKey
from celltraj2.state_supervision import EventParticipant, EventRecord, EventSet, ReviewedExposure
from celltraj2.state_event_assessment import (EpisodePolicy, EpisodeExtraction, EventProposal,
    EventAssessment, extract_event_episodes, assess_event_proposals)
from celltraj2.trajectory_index import Runs


def key(row):
    return ProjectObservationKey(str(UUID(int=1)), str(UUID(int=2)), str(UUID(int=3)), 'cells', row + 1)


class EventAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.keys = tuple(key(i) for i in range(12))
        self.frames = np.arange(12)
        self.runs = Runs(np.arange(12), np.array([0, 12]))
        self.policy = EpisodePolicy(tuning_evidence_digest='frozen-development')

    def extract(self, scores, **kwargs):
        return extract_event_episodes(scores, self.keys, self.frames, self.runs,
            definition_id='custom', policy=kwargs.pop('policy', self.policy), graph_fingerprint='graph', **kwargs)

    def proposal(self, frame, name=None, **kwargs):
        return EventProposal(name or f'p{frame}', 'custom', key(frame), 0, frame, frame, frame, .8, 1, self.policy.digest, **kwargs)

    def event(self, start, end=None, name='event', **kwargs):
        return EventRecord(name, 'custom', key(start), start, start if end is None else end,
            'source', occurrence_status='confirmed', **kwargs)

    def assess(self, proposals, records=(), exposures=(), **kwargs):
        return assess_event_proposals(proposals, EventSet('review', 'custom', records, exposures),
            self.keys, self.runs, frames=kwargs.pop('frames', self.frames), **kwargs)

    def test_nan_breaks_continuity_and_minimum_support_is_not_duration(self):
        scores = [.8, .9, np.nan, .8, .2, .8, .1, .1, .8, .1, .1, .1]
        result = self.extract(scores, policy=replace(self.policy, minimum_support=2, maximum_gap_frames=1))
        self.assertEqual([(p.start_frame, p.end_frame, p.event_frame) for p in result.proposals], [(0, 1, 1), (3, 5, 3)])
        self.assertEqual(result.missing_scores, 1)
        self.assertTrue(all(p.occurrence_status == 'unreviewed' for p in result.proposals))
        self.assertEqual(EpisodeExtraction.from_dict(json.loads(json.dumps(result.to_dict()))), result)
        duration = self.extract(scores, policy=replace(self.policy, minimum_support=2, maximum_gap_frames=1, minimum_duration_frames=2))
        self.assertEqual(len(duration.proposals), 1)

    def test_frozen_policy_cannot_have_test_tuning_or_invent_truth(self):
        with self.assertRaisesRegex(ValueError, 'tuning role'):
            replace(self.policy, tuning_role='test')
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            replace(self.proposal(3), occurrence_status='confirmed')
        with self.assertRaisesRegex(ValueError, 'graph_fingerprint'):
            extract_event_episodes(np.ones(12), self.keys, self.frames, self.runs,
                definition_id='custom', policy=self.policy, graph_fingerprint='')

    def test_refractory_keeps_best_proposal_and_saves_suppression(self):
        scores = [.1, .8, .1, .95, .1, .1, .1, .1, .9, .1, .1, .1]
        result = self.extract(scores, policy=replace(self.policy, refractory_frames=3))
        self.assertEqual([p.event_frame for p in result.proposals], [3, 8])
        self.assertEqual([p.event_frame for p in result.suppressed], [1])

    def test_run_boundaries_never_join_graph_branches(self):
        self.runs = Runs(np.arange(12), np.array([0, 5, 12]))
        result = self.extract(np.ones(12), policy=replace(self.policy, maximum_gap_frames=100))
        self.assertEqual(len(result.proposals), 2)
        self.assertEqual([p.support_count for p in result.proposals], [5, 7])

    def test_duplicate_frames_match_one_occurrence_once_and_unknown_not_false(self):
        truth = self.event(3, 5)
        exposure = ReviewedExposure('free', 'custom', key(0), 0, 2, 'source')
        result = self.assess([self.proposal(3), self.proposal(4), self.proposal(1), self.proposal(9)], (truth,), (exposure,))
        self.assertEqual(result.metrics['true_positives'], 1)
        self.assertEqual(result.metrics['false_positives'], 2)
        self.assertEqual(result.metrics['duplicate_proposals'], 1)
        self.assertEqual(result.metrics['unreviewed_proposals'], 1)
        self.assertEqual(result.matches[0].earliest_error_frames, -2)
        self.assertEqual(result.matches[0].latest_error_frames, 0)
        self.assertAlmostEqual(result.metrics['false_positives_per_reviewed_frame'], 2/3)
        self.assertEqual(EventAssessment.from_dict(json.loads(json.dumps(result.to_dict()))), result)

    def test_matching_is_maximum_cardinality_in_overlapping_tolerances(self):
        # First proposal can match both; second only the earlier occurrence.
        # A nearest-first greedy pass would miss the augmenting reassignment.
        e1, e2 = self.event(3, name='a'), self.event(5, name='b')
        result = self.assess([self.proposal(3), self.proposal(1)], (e1, e2), timing_tolerance_frames=2)
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(result.false_negative_ids, ())

    def test_rejected_and_partial_occurrences_do_not_create_truth(self):
        rejected = replace(self.event(3), occurrence_status='rejected')
        result = self.assess([self.proposal(3)], (rejected,))
        self.assertEqual(result.false_positive_ids, ())
        self.assertEqual(result.unreviewed_proposal_ids, ('p3',))
        self.assertIsNone(result.metrics['recall'])
        partial = replace(self.event(3), participant_status='partial')
        result = self.assess([self.proposal(3)], (partial,))
        self.assertEqual(result.excluded_event_ids, ('event',))
        occurrence_only = self.assess([self.proposal(3)], (partial,), participant_policy='occurrence_only')
        self.assertEqual(len(occurrence_only.matches), 1)

    def test_union_exposure_and_supersession_do_not_double_count(self):
        exposures = (ReviewedExposure('a', 'custom', key(0), 0, 5, 'source'),
                     ReviewedExposure('b', 'custom', key(4), 4, 8, 'source'))
        e1, e2 = self.event(3, name='old'), self.event(4, name='new', supersedes=('old',))
        result = self.assess([self.proposal(4)], (e1, e2), exposures)
        self.assertEqual(result.reviewed_exposure_frames, 9)
        self.assertEqual(result.matches[0].event_id, 'new')
        self.assertEqual(result.false_negative_ids, ())

    def test_exposure_denominator_uses_observed_frames_or_is_unavailable(self):
        exposure = ReviewedExposure('a', 'custom', key(0), 0, 100, 'source')
        result = self.assess([self.proposal(4)], exposures=(exposure,))
        self.assertEqual(result.reviewed_exposure_frames, 12)
        self.assertAlmostEqual(result.metrics['false_positives_per_reviewed_frame'], 1 / 12)
        unknown = self.assess([self.proposal(4)], exposures=(exposure,), frames=None)
        self.assertIsNone(unknown.metrics['false_positives_per_reviewed_frame'])

    def test_participant_mismatch_is_false_in_reviewed_event_region(self):
        record = self.event(4, participant_status='complete',
            participants=(EventParticipant('daughter', key(5)),))
        result = self.assess([self.proposal(4)], (record,), participant_policy='exact_keys')
        self.assertEqual(result.false_positive_ids, ('p4',))
        self.assertEqual(result.false_negative_ids, ('event',))
        self.assertEqual(result.duplicate_proposal_ids, ())

    def test_physical_exposure_and_timing_use_actual_clock(self):
        exposure = ReviewedExposure('free', 'custom', key(0), 0, 2, 'source', start_s=0., end_s=20.)
        event = self.event(3, 5, occurrence_start_s=30., occurrence_end_s=50.)
        result = self.assess([self.proposal(4), self.proposal(1)], (event,), (exposure,), times_s=self.frames * 10.)
        self.assertEqual(result.reviewed_exposure_s, 20.)
        self.assertEqual(result.matches[0].earliest_error_s, -10.)
        self.assertEqual(result.matches[0].latest_error_s, 10.)
        self.assertEqual(result.metrics['false_positives_per_reviewed_second'], .05)
        missing = np.asarray(self.frames * 10., dtype=float)
        missing[1] = np.nan
        result = self.assess([self.proposal(4)], (event,), (exposure,), times_s=missing)
        self.assertIsNone(result.reviewed_exposure_s)


if __name__ == '__main__':
    unittest.main()
