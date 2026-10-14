"""Analytic state/event surfaces, exposure, weighting and topology gates."""
import unittest
from unittest.mock import patch
import numpy as np
from celltraj2.trajectory_index import Runs
from celltraj2.state_correlations import compute_state_surface
from celltraj2.spatial_correlations import SpatialCorrelationCancelled, SpatialSupportLimitExceeded


def runs(*rows):
    return Runs(np.concatenate(rows).astype(int) if rows else np.array([], int),
                np.r_[0, np.cumsum([len(row) for row in rows])].astype(int))


class StateSurfaceTests(unittest.TestCase):
    def fixture(self, **updates):
        return dict(runs=runs([0], [1], [2], [3]), times=np.zeros(4),
            positions=np.array([[0., 0], [1, 0], [2, 0], [3, 0]]),
            feature_a=np.array([1., 0, np.nan, 1]), feature_b=np.array([0., 1, np.nan, 1]),
            source_mask=np.array([True, False, False, False]), distance_edges=[0, 5],
            lag_edges=[-.1, .1], min_pairs=1, **updates)

    def test_conditional_known_off_and_unknown(self):
        result = compute_state_surface(**self.fixture())
        self.assertEqual(result['values'][0, 0], 1.)
        self.assertEqual(result['pair_counts'][0, 0], 2)
        self.assertEqual(result['baseline_values'][0, 0], 1)
        self.assertEqual(result['anchor_counts'][0, 0], 1)
        args = self.fixture(); args['source_mask'] = np.array([False, True, True, False])
        empty = compute_state_surface(**args)
        self.assertTrue(np.isnan(empty['values']).all())
        self.assertEqual(empty['source_anchor_count'], 0)

    def test_joint_occupancy_includes_off_source(self):
        args = self.fixture(); args['source_mask'] = np.ones(4, bool)
        args['feature_b'] = np.array([1., 0., np.nan, 1.])
        result = compute_state_surface(**args, method='state_joint')
        self.assertEqual(result['pair_counts'][0, 0], 6)
        self.assertAlmostEqual(result['values'][0, 0], 1/3)

    def test_terminating_event_and_receiver_late_appearance(self):
        args = dict(runs=runs([0], [1]), times=[0., 2.], positions=[[0., 0], [2., 0]],
            feature_b=[np.nan, 1.], method='event_state', anchor_rows=[0], anchor_times=[0.],
            anchor_ids=['event-971'], lag_edges=[1.5, 2.5], distance_edges=[0, 3], min_pairs=1)
        result = compute_state_surface(**args)
        self.assertEqual(result['values'][0, 0], 1)
        self.assertEqual(result['support_details']['anchor_ids'], ['event-971'])
        source_time = compute_state_surface(**args, distance_convention='source_time')
        self.assertTrue(np.isnan(source_time['values']).all())

    def test_source_time_requires_unbroken_receiver_cohort(self):
        args = dict(runs=runs([0], [1, 2, 3]), times=[0., 0., 1., 2.],
            positions=[[0., 0], [2., 0], [8., 0], [9., 0]], feature_a=[1., 0, 0, 0], feature_b=[0., 0, 0, 1],
            source_mask=np.array([True, False, False, False]), lag_edges=[1.5, 2.5], distance_edges=[0, 3, 10], min_pairs=1)
        reference = compute_state_surface(**args)
        np.testing.assert_equal(reference['pair_counts'], [[1, 0]])
        cross = compute_state_surface(**args, distance_convention='cross_time')
        np.testing.assert_equal(cross['pair_counts'], [[0, 1]])
        args['receiver_mask'] = np.array([False, True, False, True])
        self.assertTrue(np.isnan(compute_state_surface(**args)['values']).all())

    def test_equal_anchor_pair_and_frame_weights_are_explicit(self):
        # Two colocated sources at t=0 and one source at t=1. Receivers differ
        # in count and ON frequency; both time groups land in the same lag bin.
        args = dict(runs=runs(*[[i] for i in range(8)]), times=[0, 0, 0, 0, 0, 1, 1, 1],
            positions=[[0, 0], [0, 0], [1, 0], [2, 0], [3, 0], [0, 0], [1, 0], [8, 0]],
            source_mask=np.array([True, True, False, False, False, True, False, False]),
            receiver_mask=np.array([False, False, True, True, True, False, True, True]),
            feature_a=np.ones(8), feature_b=[0, 0, 1, 1, 1, 0, 0, 0], distance_edges=[0, 4], lag_edges=[-.1, .1], min_pairs=1)
        self.assertAlmostEqual(compute_state_surface(**args)['values'][0, 0], 2/3)
        self.assertAlmostEqual(compute_state_surface(**args, weighting='pairs')['values'][0, 0], 6/7)
        self.assertAlmostEqual(compute_state_surface(**args, weighting='equal_frame')['values'][0, 0], .5)

    def test_time_matched_baseline_and_zero_enrichment(self):
        args = self.fixture(); args['feature_b'] = np.array([1., 1, 0, 0])
        args['distance_edges'] = [0, 1.5]
        result = compute_state_surface(**args, response_mode='excess')
        self.assertAlmostEqual(result['response_values'][0, 0], 1)
        self.assertAlmostEqual(result['baseline_values'][0, 0], 1/3)
        self.assertAlmostEqual(result['values'][0, 0], 2/3)
        args['feature_b'][:] = 0
        missing = compute_state_surface(**args, response_mode='enrichment')
        self.assertTrue(np.isnan(missing['values']).all())
        self.assertEqual(missing['status'][0, 0], 'zero_baseline')

    def test_continuous_pair_moments_and_constant_undefined(self):
        args = dict(runs=runs(*[[i] for i in range(6)]), times=[0, 0, 1, 1, 2, 2],
            positions=[[0, 0], [1, 0]]*3, source_mask=np.array([True, False]*3),
            receiver_mask=np.array([False, True]*3), feature_a=[1e9+1, 0, 1e9+2, 0, 1e9+3, 0],
            feature_b=[0, 8, 0, 6, 0, 4], method='feature_correlation', distance_edges=[0, 2], lag_edges=[-.1, .1], min_pairs=1)
        result = compute_state_surface(**args, normalization='coefficient')
        self.assertAlmostEqual(result['values'][0, 0], -1)
        self.assertAlmostEqual(compute_state_surface(**args, normalization='covariance')['values'][0, 0], -4/3)
        args['feature_a'] = np.ones(6)
        zero = compute_state_surface(**args, normalization='coefficient')
        self.assertEqual(zero['status'][0, 0], 'zero_variance')

    def test_empty_unknown_validation_cancellation_and_budget(self):
        args = self.fixture(); args['feature_b'][:] = np.nan
        self.assertTrue(np.isnan(compute_state_surface(**args)['values']).all())
        with self.assertRaises(SpatialCorrelationCancelled):
            compute_state_surface(**self.fixture(), cancelled=lambda: True)
        with self.assertRaises(SpatialSupportLimitExceeded):
            compute_state_surface(**self.fixture(), max_support_entries=1)
        np.testing.assert_equal(compute_state_surface(**self.fixture(), max_support_entries=0)['values'], [[1]])
        args = self.fixture(); args['feature_b'][1] = .3
        with self.assertRaisesRegex(ValueError, '0/1/NaN'):
            compute_state_surface(**args)

    def test_unsupported_anchor_cannot_destabilize_supported_pearson(self):
        args = dict(runs=runs(*[[i] for i in range(7)]), times=[0, 1, 1, 2, 2, 3, 3],
            positions=[[1000, 0], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 0]],
            source_mask=np.array([True, True, False, True, False, True, False]),
            receiver_mask=np.array([False, False, True, False, True, False, True]),
            feature_a=[0, 1e9+1, 0, 1e9+2, 0, 1e9+3, 0], feature_b=[0, 0, 8, 0, 6, 0, 4],
            method='feature_correlation', normalization='coefficient', distance_edges=[0, 2], lag_edges=[-.1, .1], min_pairs=1)
        self.assertAlmostEqual(compute_state_surface(**args)['values'][0, 0], -1)

    def test_spatial_index_reused_and_one_frame_scratch(self):
        import celltraj2.state_correlations as module
        args = self.fixture(); args['source_mask'] = np.ones(4, bool)
        with patch.object(module, 'cKDTree', wraps=module.cKDTree) as trees:
            result = compute_state_surface(**args, distance_convention='cross_time')
        self.assertEqual(trees.call_count, 1)
        self.assertLessEqual(result['support_details']['anchor_sums'].shape[0], 4)


if __name__ == '__main__':
    unittest.main()
