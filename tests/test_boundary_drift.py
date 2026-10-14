import unittest

import numpy as np

from celltraj2.boundary_drift import BoundaryTargetIndex, project_boundary_drift, summarize_boundary_samples


class BoundaryDriftTests(unittest.TestCase):
    def project(self, p, u, targets, *, source_ids=None, target_ids=None, **kwargs):
        p, targets = np.asarray(p, float), np.asarray(targets, float)
        return project_boundary_drift(p, u, np.zeros(len(p), int) if source_ids is None else source_ids,
            BoundaryTargetIndex(targets, np.ones(len(targets), int) if target_ids is None else target_ids),
            distance_edges=[0, 5, 20], min_pairs=1, **kwargs)

    def test_positive_toward_and_zero_are_real_measurements(self):
        result = self.project([[0, 0], [0, 1]], [[2, 0], [0, 0]], [[10, 0], [10, 1]])
        np.testing.assert_allclose(result['displacement'], [2, 0])
        np.testing.assert_allclose(result['values'][1], 1)
        self.assertEqual(result['counts'][1], 2)

    def test_nearest_other_skips_arbitrarily_many_self_points(self):
        target = np.c_[np.arange(30), np.zeros(30)]
        result = self.project([[0, 0]], [[1, 0]], target,
            target_ids=np.r_[np.zeros(20, int), np.ones(10, int)])
        self.assertEqual(result['distance'][0], 20)
        self.assertEqual(result['target_rows'][0], 20)

    def test_only_self_has_no_neighbor(self):
        result = self.project([[0, 0]], [[1, 0]], [[0, 0]], target_ids=[0])
        self.assertEqual(result['validity'][0], 3)
        self.assertTrue(np.isnan(result['values']).all())

    def test_zero_separation_is_not_a_direction(self):
        result = self.project([[0, 0]], [[1, 0]], [[0, 0]])
        self.assertEqual(result['validity'][0], 4)

    def test_normals_are_ignored_when_disabled(self):
        result = self.project([[0, 0]], [[1, 0]], [[10, 0]], normals=[[np.nan, np.nan]])
        self.assertEqual(result['displacement'][0], 1)
        result = self.project([[0, 0]], [[1, 0]], [[10, 0]], normals=[[np.nan, np.nan]], angle_min=0)
        self.assertEqual(result['validity'][0], 6)

    def test_angle_filters_nearest_before_searching_farther(self):
        result = self.project([[0, 0]], [[1, 0]], [[-2, 0], [10, 0]], normals=[[1, 0]], angle_min=0)
        self.assertEqual(result['distance'][0], 2)
        self.assertEqual(result['validity'][0], 7)

    def test_nan_motion_is_not_zero(self):
        result = self.project([[0, 0], [0, 1]], [[np.nan, 0], [0, 0]], [[10, 0], [10, 1]])
        np.testing.assert_equal(result['validity'], [2, 0])
        self.assertEqual(result['counts'][1], 1)
        self.assertEqual(result['values'][1], 0)

    def test_ties_choose_stable_smallest_point_identity(self):
        index = BoundaryTargetIndex([[2, 0], [-2, 0]], [2, 3], [30, 10])
        result = project_boundary_drift([[0, 0]], [[1, 0]], [1], index, distance_edges=[0, 3], min_pairs=1)
        self.assertEqual(result['displacement'][0], -1)

    def test_vector_rotation_preserves_projection(self):
        p, u, q = np.array([[1., 2, 3]]), np.array([[2., -1, 0]]), np.array([[4., 6, 8]])
        rotation, _ = np.linalg.qr(np.array([[1., 2, 1], [2, 0, 3], [0, 1, 2]]))
        a, b = self.project(p, u, q), self.project(p@rotation, u@rotation, q@rotation)
        np.testing.assert_allclose(a['displacement'], b['displacement'])

    def test_weighted_sufficient_statistics_and_minimum_support(self):
        result = summarize_boundary_samples([1, 1, 5], [2, 4, 9], [3, 1, 1], [0, 0, 2], [0, 2, 6], min_pairs=3)
        np.testing.assert_equal(result['counts'], [2, 0])
        self.assertEqual(result['weighted_sums'][0], 10)
        self.assertEqual(result['weighted_square_sums'][0], 28)
        self.assertEqual(result['squared_weight_sums'][0], 10)
        self.assertTrue(np.isnan(result['values']).all())

    def test_cancellation_and_invalid_edges(self):
        with self.assertRaisesRegex(InterruptedError, 'cancelled'):
            self.project([[0, 0]], [[1, 0]], [[2, 0]], cancelled=lambda: True)
        with self.assertRaises(ValueError):
            summarize_boundary_samples([], [], [], [], [0, 0])


if __name__ == '__main__':
    unittest.main()
