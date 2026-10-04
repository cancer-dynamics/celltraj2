import json
import unittest
from dataclasses import replace
from uuid import UUID

import numpy as np

from celltraj2.interpretation import ProjectObservationKey
from celltraj2.state_supervision import EventDefinition, EventRole, EventRecord, EventParticipant, EventSet, ReviewedExposure, SupervisionTask, StateAnnotation
from celltraj2.state_supervised_targets import (TemporalTargetProjection, project_event_targets,
    project_current_state_targets, iter_target_support_keys, reviewed_at_risk_from_exposure)
from celltraj2.trajectory_index import Runs


def key(row, dataset=2):
    return ProjectObservationKey(str(UUID(int=1)), str(UUID(int=dataset)), str(UUID(int=3)), 'cells', row + 1)


def event(i=4, end=None, **kwargs):
    return EventRecord('event', 'custom', key(i), i, i if end is None else end, 'source',
        occurrence_status='confirmed', occurrence_start_s=float(i), occurrence_end_s=float(i if end is None else end), **kwargs)


def exposure(end=8, **kwargs):
    return ReviewedExposure('exposure', 'custom', key(0), 0, end, 'source', start_s=0., end_s=float(end), **kwargs)


class TemporalTargetTests(unittest.TestCase):
    def setUp(self):
        self.keys = tuple(key(i) for i in range(9))
        self.frames = np.arange(9)
        self.runs = Runs(np.arange(9), np.array([0, 9]))
        self.task = SupervisionTask('fate', 'channel', 'custom', 'fixed_horizon_fate', 'population', 'recipe', ('no', 'yes'), horizon_s=4.)

    def project(self, records=(), exposures=(), length=1, **kwargs):
        return project_event_targets(kwargs.pop('task', self.task), self.keys, self.frames, self.runs.windows(length),
            EventSet('review', 'custom', records, exposures), times_s=kwargs.pop('times_s', self.frames),
            reviewed_at_risk=kwargs.pop('reviewed_at_risk', {k: 'risk-review' for k in self.keys}), **kwargs)

    def test_current_state_adapter_preserves_final_anchors_and_vocabulary(self):
        task = replace(self.task, family='current_state', horizon_s=None)
        refs = project_current_state_targets(task, [StateAnnotation('e', key(3), 'yes')], self.keys, self.runs.windows(3))
        self.assertEqual(refs.rows[0].window_index, 1)
        self.assertEqual(refs.rows[0].anchor_key, key(3))
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            project_current_state_targets(task, [StateAnnotation('e', key(3), 'other')], self.keys, self.runs.windows())

    def test_forecast_open_origin_closed_horizon_first_event_and_weights(self):
        projection = self.project((event(4),), (exposure(4),))
        self.assertEqual([r.class_id for r in projection.rows[:4]], ['yes'] * 4)
        self.assertEqual(projection.rows[4].reason, 'prior_event_not_at_risk')
        self.assertAlmostEqual(sum(r.weight for r in projection.rows), 1.)
        self.assertEqual(projection.rows[0].outcome_end_s, 4.)
        self.assertEqual(TemporalTargetProjection.from_dict(json.loads(json.dumps(projection.to_dict()))), projection)

    def test_negative_needs_complete_review_and_observed_followup(self):
        result = self.project(exposures=(exposure(8),))
        self.assertEqual([r.class_id for r in result.rows[:5]], ['no'] * 5)
        self.assertEqual(result.rows[5].status, 'censored')
        self.assertAlmostEqual(sum(r.weight for r in result.rows), 1.)
        no_review = self.project()
        self.assertTrue(all(r.status == 'censored' for r in no_review.rows))
        no_risk = self.project(exposures=(exposure(8),), reviewed_at_risk={})
        self.assertTrue(all(r.reason == 'at_risk_not_reviewed' for r in no_risk.rows))

    def test_intervals_crossing_origin_or_horizon_are_ambiguous(self):
        result = self.project((event(3, 5),), (exposure(3),))
        self.assertEqual(result.rows[0].status, 'ambiguous')
        self.assertEqual(result.rows[1].class_id, 'yes')
        self.assertEqual(result.rows[3].status, 'ambiguous')
        self.assertEqual(result.rows[4].status, 'ambiguous')

    def test_positive_loss_after_event_is_allowed_loss_before_event_is_not(self):
        self.runs = Runs(np.arange(5), np.array([0, 5]))
        good = self.project((event(4),), (exposure(4),))
        self.assertEqual(good.rows[0].class_id, 'yes')
        before = self.project((replace(event(4), occurrence_start_s=4.5, occurrence_end_s=4.5),), (replace(exposure(5), end_s=4.5),), task=replace(self.task, horizon_s=5.))
        self.assertEqual(before.rows[0].reason, 'incomplete_observed_followup')

    def test_missing_time_and_reversed_followup_do_not_become_negatives(self):
        missing = self.project(exposures=(exposure(8),), times_s=np.full(9, np.nan))
        self.assertEqual(missing.rows[0].reason, 'invalid_physical_input_time')
        times = np.arange(9, dtype=float)
        times[2] = 0.5
        result = self.project(exposures=(exposure(8),), times_s=times)
        self.assertEqual(result.rows[0].reason, 'incomplete_observed_followup')

    def test_competing_division_censors_target_division_is_positive(self):
        division = replace(event(2), event_id='division', definition_id='division')
        result = self.project((event(4),), (exposure(4),), competing_events=(division,))
        self.assertEqual(result.rows[0].reason, 'competing_event')
        later = self.project((event(4),), (exposure(4),), competing_events=(replace(division, occurrence_start_s=5., occurrence_end_s=5.),))
        self.assertEqual(later.rows[0].class_id, 'yes')
        targeted = self.project((event(2),), (exposure(2),))
        self.assertEqual(targeted.rows[0].class_id, 'yes')

    def test_participants_and_outcome_support_are_exposed_for_split_validation(self):
        participant = EventParticipant('recipient', key(7, dataset=7))
        record = event(4, participant_status='complete', participants=(participant,))
        definition = EventDefinition('custom', 'family', 'Transfer', (EventRole('recipient', 1, 1, True),))
        result = self.project((record,), (exposure(4),), length=3, event_definition=definition)
        row = result.rows[0]
        self.assertEqual(row.input_span.stop_offset, 3)
        self.assertEqual(row.outcome_span.stop_offset, 5)
        support = list(iter_target_support_keys(row, self.keys, self.runs))
        self.assertIn(key(7, dataset=7), support)
        self.assertIn(key(0), support)
        self.assertIn(key(4), support)
        self.assertIn(key(1), support)
        self.assertNotIn(key(5), support)
        partial = self.project((replace(record, participant_status='partial'),), (exposure(4),), event_definition=definition)
        self.assertEqual(partial.rows[0].reason, 'required_participants_unresolved')

    def test_detection_target_carries_complete_reviewed_occurrence_interval_support(self):
        task = replace(self.task, family='event_detection', horizon_s=None, detection_protocol='reviewed_range')
        result = self.project((event(3, 5),), task=task, detection_range_frames=(-4, 0))
        row = result.rows[1]
        self.assertEqual(row.status, 'eligible')
        support = set(iter_target_support_keys(row, self.keys, self.runs))
        self.assertTrue({key(3), key(4), key(5)} <= support)

    def test_detection_interval_intersection_and_explicit_negatives(self):
        task = replace(self.task, family='event_detection', horizon_s=None, detection_protocol='reviewed_range')
        result = self.project((event(3, 5),), (exposure(1),), task=task, detection_range_frames=(-1, 1))
        self.assertEqual(result.rows[0].class_id, 'no')
        self.assertEqual(result.rows[2].status, 'ambiguous')
        self.assertEqual(result.rows[4].class_id, 'yes')
        self.assertEqual(result.rows[6].status, 'ambiguous')
        self.assertEqual(result.rows[7].status, 'unreviewed')
        anchored = self.project((event(3, 5),), task=replace(task, detection_protocol='reviewed_anchor'))
        self.assertTrue(all(anchored.rows[i].status == 'ambiguous' for i in (3, 4, 5)))

    def test_rejected_event_does_not_supply_negative_truth_and_supersession_is_counted_once(self):
        task = replace(self.task, family='event_detection', horizon_s=None, detection_protocol='reviewed_anchor')
        rejected = self.project((replace(event(3), occurrence_status='rejected'),), task=task)
        self.assertEqual(rejected.rows[3].status, 'unreviewed')
        new = replace(event(4), event_id='new', supersedes=('event',))
        result = self.project((event(3), new), (exposure(4),))
        self.assertEqual(result.rows[0].evidence_ids, ('risk-review', 'new', 'exposure'))
        self.assertEqual(result.rows[3].class_id, 'yes')

    def test_changed_sources_and_graphs_do_not_supply_truth(self):
        result = self.project((event(4),), (exposure(4),), source_fingerprints={k: 'new' for k in self.keys})
        self.assertEqual(result.rows[0].reason, 'review_source_or_graph_changed')
        result = self.project((event(4, graph_fingerprint='old'),), (exposure(4),), graph_fingerprint='new')
        self.assertEqual(result.rows[0].status, 'excluded')

    def test_mixed_cadence_is_actual_seconds_and_at_risk_helper_is_explicit(self):
        review = EventSet('review', 'custom', (), (exposure(8),))
        at_risk = reviewed_at_risk_from_exposure(self.keys, self.frames * 2, self.runs, review)
        self.assertIn(key(4), at_risk)
        self.assertNotIn(key(5), at_risk)
        result = self.project(exposures=(exposure(8),), times_s=self.frames * 2, reviewed_at_risk=at_risk, length=3)
        self.assertEqual(result.rows[0].input_span_s, 4.)
        self.assertEqual(result.rows[0].outcome_span.stop_offset, 5)
        self.assertEqual(result.rows[0].class_id, 'no')

    def test_membership_and_run_boundaries_end_followup(self):
        membership = np.ones(9, dtype=bool)
        membership[3] = False
        result = self.project(exposures=(exposure(8),), eligible=membership)
        self.assertEqual(result.rows[0].status, 'censored')
        self.runs = Runs(np.arange(9), np.array([0, 3, 9]))
        result = self.project((event(4),), (exposure(8),))
        self.assertNotEqual(result.rows[0].status, 'eligible')

    def test_reviewed_censoring_truncates_even_with_overlapping_free_review(self):
        censored = replace(exposure(3, status='censored', censor_reason='tracking_loss'), exposure_id='loss')
        result = self.project((event(4),), (exposure(8), censored))
        self.assertEqual(result.rows[0].reason, 'reviewed_followup_censored')
        self.assertIn('loss', result.rows[0].evidence_ids)
        early = self.project((event(2),), (exposure(2), censored))
        self.assertEqual(early.rows[0].class_id, 'yes')

    def test_opposing_event_review_and_cyclic_supersession_cannot_supply_labels(self):
        task = replace(self.task, family='event_detection', horizon_s=None, detection_protocol='reviewed_anchor')
        result = self.project((event(4),), (exposure(8),), task=task)
        self.assertEqual(result.rows[4].status, 'conflict')
        future = self.project((event(4),), (exposure(8),))
        self.assertEqual(future.rows[0].status, 'conflict')
        first = replace(event(3), supersedes=('second',))
        second = replace(event(4), event_id='second', supersedes=('event',))
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            self.project((first, second), (exposure(8),))


if __name__ == '__main__':
    unittest.main()
