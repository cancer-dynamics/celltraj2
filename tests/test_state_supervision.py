import json
import unittest
from dataclasses import FrozenInstanceError, replace
from uuid import UUID

import numpy as np

from celltraj2.interpretation import ProjectObservationKey
from celltraj2.state_supervision import (
    EventDefinition, EventParticipant, EventRecord, EventRole, EventSet, ReviewedExposure,
    SupervisionTask, StateAnnotation, StateReferenceProjection, ExtensionPolicy,
    ExtensionProjection, extend_state_annotations, project_state_references,
)
from celltraj2.trajectory_index import Runs, GraphIndex
from test_trajectory_index import graph


def key(row, dataset=2):
    return ProjectObservationKey(str(UUID(int=1)), str(UUID(int=dataset)), str(UUID(int=3)), 'cells', row + 1)


def keys(n):
    return tuple(key(i) for i in range(n))


def annotation(row, label='active', **kwargs):
    return StateAnnotation(f'e-{row}-{label}', key(row), label, **kwargs)


class EventContractTests(unittest.TestCase):
    def test_numpy_frame_and_count_scalars_roundtrip_as_ordinary_json(self):
        event = EventRecord('event', 'definition', key(1), np.int64(2), np.int64(3), 'spine')
        role = EventRole('participant', np.int64(1), np.int64(2))
        policy = ExtensionPolicy(np.int64(1), np.int64(2), start_frame=np.int64(0))
        for contract in (event, role, policy):
            serialized = json.loads(json.dumps(contract.to_dict(), allow_nan=False))
            self.assertEqual(type(contract).from_dict(serialized), contract)
            self.assertEqual(contract.digest, type(contract).from_dict(serialized).digest)

    def test_event_set_retains_distinct_review_and_exposure_truth(self):
        rejected = EventRecord('proposal', 'custom', key(1), 1, 1, 'spine', occurrence_status='rejected')
        pending = EventSet('events', 'custom', (rejected,))
        self.assertEqual(pending.exposures, ())
        exposure = ReviewedExposure('reviewed', 'custom', key(1), 1, 5, 'spine')
        reviewed = replace(pending, exposures=(exposure,), review_revision='review2')
        self.assertEqual(EventSet.from_dict(json.loads(json.dumps(reviewed.to_dict()))), reviewed)
        with self.assertRaisesRegex(ValueError, 'duplicate occurrence'):
            replace(pending, records=(rejected, rejected))
        with self.assertRaisesRegex(ValueError, 'revision mismatch'):
            replace(pending, definition_id='changed')
        with self.assertRaisesRegex(ValueError, 'note must be text'):
            replace(rejected, note={'mutable': 'not text'})

    def test_custom_definition_occurrence_and_participants_are_separate(self):
        definition = EventDefinition('custom', 'family', 'Contact transfer',
                                     (EventRole('recipient', 1, 1, True),))
        occurrence = EventRecord('e', 'custom', key(2), 3, 5, 'spine',
                                 occurrence_status='confirmed', participant_status='partial')
        occurrence.validate_definition(definition, eligible_keys={key(2)})
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            replace(occurrence, participant_status='complete').validate_definition(definition)
        confirmed = replace(occurrence, participant_status='complete',
                            participants=(EventParticipant('recipient', key(9, dataset=8)),))
        confirmed.validate_definition(definition, eligible_keys={key(2)})
        self.assertEqual(EventRecord.from_dict(json.loads(json.dumps(confirmed.to_dict()))), confirmed)
        self.assertEqual(EventDefinition.from_dict(definition.to_dict()), definition)
        with self.assertRaises(FrozenInstanceError):
            confirmed.note = 'mutate'
        with self.assertRaisesRegex(ValueError, 'outside channel'):
            confirmed.validate_definition(definition, eligible_keys=set())

    def test_required_roles_and_unknown_roles_do_not_confirm_themselves(self):
        definition = EventDefinition('division', 'family', 'Division', (EventRole('daughter', 2, 2),))
        proposal = EventRecord('e', 'division', key(1), 2, 2, 'spine')
        self.assertEqual(proposal.occurrence_status, 'unreviewed')
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            proposal.validate_definition(definition)
        bad = replace(proposal, participant_status='partial', participants=(EventParticipant('neighbor', key(2)),))
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            bad.validate_definition(definition)
        partial = replace(proposal, participant_status='partial')
        partial.validate_definition(definition)
        self.assertEqual(partial.occurrence_status, 'unreviewed')

    def test_exposure_requires_explicit_review_and_censor_reason(self):
        free = ReviewedExposure('x', 'death', key(1), 1, 8, 'spine')
        self.assertEqual(ReviewedExposure.from_dict(free.to_dict()), free)
        with self.assertRaisesRegex(ValueError, 'censor reason'):
            replace(free, status='censored')
        censored = replace(free, status='censored', censor_reason='lost_tracking')
        self.assertEqual(censored.status, 'censored')
        with self.assertRaisesRegex(ValueError, 'precedes'):
            replace(free, end_frame=0)
        with self.assertRaisesRegex(ValueError, 'together'):
            replace(free, start_s=0)

    def test_task_family_cannot_relabel_risk_as_current_state(self):
        current = SupervisionTask('t', 'channel', 'state', 'current_state', 'pop', 'L1', ('off', 'on'))
        self.assertEqual(SupervisionTask.from_dict(current.to_dict()), current)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            replace(current, class_ids=('one',))
        with self.assertRaisesRegex(ValueError, 'Only fate'):
            replace(current, horizon_s=3600)
        fate = replace(current, family='fixed_horizon_fate', horizon_s=3600)
        with self.assertRaisesRegex(ValueError, 'causal'):
            replace(fate, temporal_direction='retrospective')
        with self.assertRaisesRegex(ValueError, 'detection protocol'):
            replace(current, family='event_detection')
        self.assertEqual(replace(current, family='event_detection', detection_protocol='reviewed_anchor').annotation_anchor, 'final')
        with self.assertRaisesRegex(ValueError, 'schema'):
            SupervisionTask.from_dict({**current.to_dict(), 'schema': 'future.unknown.v2'})


class StateProjectionTests(unittest.TestCase):
    def setUp(self):
        self.keys = keys(4)
        self.runs = Runs(np.arange(4), np.array([0, 4]))

    def project(self, evidence, length=3, **kwargs):
        return project_state_references(evidence, self.keys, self.runs.windows(length),
                                        population_id='population', recipe_id=f'length-{length}', **kwargs)

    def test_only_final_member_is_labeled_and_missing_delay_retains_evidence(self):
        result = self.project([annotation(2), annotation(0)])
        by_key = {r.anchor_key: r for r in result.rows}
        self.assertEqual(by_key[key(2)].window_index, 0)
        self.assertEqual(by_key[key(2)].status, 'eligible')
        self.assertEqual(by_key[key(0)].reason, 'no_eligible_final_anchor_window')
        self.assertNotIn(key(3), by_key)
        singleton = self.project([annotation(2), annotation(0)], length=1)
        self.assertTrue(all(r.status == 'eligible' for r in singleton.rows))
        self.assertEqual(StateReferenceProjection.from_dict(json.loads(json.dumps(result.to_dict()))), result)

    def test_cross_file_ids_do_not_collide_and_exclusions_are_retained(self):
        outside = replace(annotation(2), evidence_id='other-file', key=key(2, dataset=7))
        result = self.project([outside, annotation(2)], feature_valid=np.array([False, True]))
        self.assertEqual([r.reason for r in result.rows], ['outside_current_population', 'unavailable_model_features'])
        membership = np.array([True, False, True, True])
        self.assertEqual(self.project([annotation(2)], eligible=membership).rows[0].reason, 'outside_membership')

    def test_direct_precedence_conflicts_and_unknown_are_not_negatives(self):
        derived = StateAnnotation('derived', key(2), 'off', origin='derived_extension',
                                  graph_fingerprint='graph', seed_evidence_id='seed')
        result = self.project([derived, annotation(2)], graph_fingerprint='graph')
        self.assertEqual(result.rows[0].class_id, 'active')
        conflict = self.project([annotation(2), replace(annotation(2, 'off'), evidence_id='other')])
        self.assertEqual(conflict.rows[0].status, 'conflict')
        self.assertEqual(conflict.rows[0].weight, 0)
        unknown = StateAnnotation('unknown', key(2), status='uncertain')
        self.assertEqual(self.project([derived, unknown], graph_fingerprint='graph').rows[0].status, 'uncertain')
        model = replace(annotation(2), origin='model')
        self.assertEqual(self.project([model]).rows[0].reason, 'origin_requires_explicit_task_policy')

    def test_changing_states_along_track_are_valid(self):
        result = self.project([annotation(2, 'on'), annotation(3, 'off')])
        self.assertEqual([r.status for r in result.rows], ['eligible', 'eligible'])

    def test_source_and_graph_changes_keep_unmapped_evidence(self):
        source = annotation(2, source_fingerprint='old')
        self.assertEqual(self.project([source], source_fingerprints={key(2): 'new'}).rows[0].reason, 'source_identity_changed')
        derived = replace(source, origin='derived_extension', seed_evidence_id='source', graph_fingerprint='oldgraph')
        self.assertEqual(self.project([derived], graph_fingerprint='newgraph').rows[0].reason, 'extension_graph_changed')
        self.assertEqual(self.project([source], graph_fingerprint='newgraph').rows[0].status, 'eligible')

    def test_window_cannot_cross_runs_even_if_final_anchor_exists(self):
        bad = self.runs.windows(3)
        bad.runs = Runs(np.arange(4), np.array([0, 2, 4]))
        with self.assertRaisesRegex(ValueError, 'crosses'):
            project_state_references([annotation(2)], self.keys, bad, population_id='p', recipe_id='r')


class StateExtensionTests(unittest.TestCase):
    def extend(self, evidence, *, n=5, runs=None, policy=None, **kwargs):
        return extend_state_annotations(evidence, keys(n), np.arange(n),
                                        runs or Runs(np.arange(n), np.array([0, n])),
                                        policy=policy or ExtensionPolicy(2, 2), graph_fingerprint='graph', **kwargs)

    def test_default_is_zero_and_explicit_extension_does_not_recurse(self):
        seed = annotation(2)
        empty = self.extend([seed], policy=ExtensionPolicy())
        self.assertEqual(empty.annotations, (seed,))
        extended = self.extend([seed], policy=ExtensionPolicy(1, 1))
        self.assertEqual({x.key for x in extended.annotations}, {key(1), key(2), key(3)})
        only_derived = tuple(x for x in extended.annotations if x.origin == 'derived_extension')
        again = self.extend(only_derived, policy=ExtensionPolicy(9, 9))
        self.assertEqual(again.annotations, only_derived)
        self.assertEqual(again.spans, ())
        self.assertEqual(ExtensionProjection.from_dict(extended.to_dict()), extended)

    def test_division_and_membership_split_boundaries(self):
        index = GraphIndex.from_graph(graph([0, 1, 2, 2, 3, 4]))
        frame = np.array([0, 1, 2, 2, 3, 3])
        runs = index.project(np.arange(6), frame, np.ones(6, dtype=bool))
        extended = extend_state_annotations([annotation(1)], keys(6), frame, runs,
                                            policy=ExtensionPolicy(20, 20), graph_fingerprint='graph')
        self.assertEqual({x.key for x in extended.annotations}, {key(0), key(1)})
        membership = np.array([True, True, True, False, True])
        bounded = self.extend([annotation(2)], eligible=membership, splits=['fit', 'fit', 'fit', 'fit', 'test'])
        self.assertEqual({x.key for x in bounded.annotations}, {key(0), key(1), key(2)})
        self.assertEqual(bounded.spans[0].forward_stop, 'membership_boundary')
        split = self.extend([annotation(2)], splits=['fit', 'fit', 'fit', 'test', 'test'])
        self.assertEqual(split.spans[0].forward_stop, 'split_boundary')
        with self.assertRaisesRegex(ValueError, 'complete split'):
            self.extend([annotation(2)], splits={key(2): 'fit'})

    def test_explicit_time_bound_and_missing_time(self):
        bounded = self.extend([annotation(2)], policy=ExtensionPolicy(4, 4, max_time_s=11),
                              times_s=[0, 10, 20, 30, 50])
        self.assertEqual({x.key for x in bounded.annotations}, {key(1), key(2), key(3)})
        with self.assertRaisesRegex(ValueError, 'time metadata'):
            self.extend([annotation(2)], policy=ExtensionPolicy(2, 2, max_time_s=20))
        unknown = self.extend([annotation(2)], policy=ExtensionPolicy(2, 2, max_time_s=20),
                              times_s=[0, 10, np.nan, 30, 40])
        self.assertEqual(len(unknown.annotations), 1)
        self.assertEqual(unknown.spans[0].forward_stop, 'unknown_physical_time')

    def test_frame_gap_and_partition_boundaries_do_not_bridge(self):
        mixed = (key(0), key(1), key(2), key(0, dataset=7))
        malformed_runs = Runs(np.arange(4), np.array([0, 4]))
        projected = extend_state_annotations([annotation(2)], mixed, np.arange(4), malformed_runs,
                                             policy=ExtensionPolicy(0, 10), graph_fingerprint='graph')
        self.assertEqual(len(projected.annotations), 1)
        self.assertEqual(projected.spans[0].forward_stop, 'partition_boundary')
        gap = extend_state_annotations([annotation(2)], keys(4), np.array([0, 1, 2, 4]), malformed_runs,
                                       policy=ExtensionPolicy(0, 10), graph_fingerprint='graph')
        self.assertEqual(gap.spans[0].forward_stop, 'frame_gap')

    def test_source_unit_weight_does_not_expand_with_derived_frames(self):
        extended = self.extend([annotation(2)])
        refs = project_state_references(extended.annotations, keys(5), Runs(np.arange(5), np.array([0, 5])).windows(),
                                        population_id='p', recipe_id='r', graph_fingerprint='graph')
        self.assertEqual(len(refs.rows), 5)
        self.assertAlmostEqual(sum(row.weight for row in refs.rows), 1.0)
        self.assertTrue(all(row.source_unit_ids == ('e-2-active',) for row in refs.rows))

    def test_long_delay_projection_stores_no_member_expansion(self):
        n = 10000
        sample_keys = keys(n)
        windows = Runs(np.arange(n), np.array([0, n])).windows(1000)
        refs = project_state_references([StateAnnotation('last', sample_keys[-1], 'on')], sample_keys, windows,
                                        population_id='p', recipe_id='r')
        self.assertEqual(len(refs.rows), 1)
        self.assertEqual(refs.rows[0].window_index, n - 1000)
        self.assertLess(len(json.dumps(refs.to_dict())), 1500)


if __name__ == '__main__':
    unittest.main()
