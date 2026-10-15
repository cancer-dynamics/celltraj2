import unittest
import numpy as np

from celltraj2.boundaries import boundary_motion_link_dtype
from celltraj2.motion_diagnostics import add_centroid_motion_diagnostics, registered_centroid_lookup


class MotionDiagnosticsTests(unittest.TestCase):
    def fixture(self):
        links = np.zeros(1, boundary_motion_link_dtype())
        links['source_observation_id'], links['target_observation_id'] = 17, 93
        links['source_summary_count'], links['solver_converged'] = 2, 1
        points = dict(barycentric_displacement_zyx=np.array([[0., 0., .4], [0., 0., 1.2]]),
            point_mass=np.array([3., 1.]), matched_fraction=np.array([1., .5]), quality_flags=np.zeros(2, int))
        graph = np.array([(17, 93)], dtype=[('parent_observation_id', 'i8'), ('child_observation_id', 'i8')])
        centers = {17: np.zeros(3), 93: np.array([0., 0., 2.])}
        return links, points, centers, graph

    def test_represented_mass_weighting_and_nan_support(self):
        args = self.fixture()
        links, schema, event = add_centroid_motion_diagnostics(*args)
        np.testing.assert_allclose(links['boundary_mean_displacement_zyx'][0], [0, 0, .6])
        self.assertAlmostEqual(links['centroid_motion_recovery'][0], .3)
        self.assertAlmostEqual(links['centroid_motion_error'][0], 1.4)
        self.assertEqual(event['comparison_link_count'], 1)
        self.assertEqual(schema['weighting'], 'represented_point_mass')
        args[1]['barycentric_displacement_zyx'][1] = np.nan
        links, _, _ = add_centroid_motion_diagnostics(*args)
        self.assertAlmostEqual(links['centroid_motion_recovery'][0], .2)
        self.assertAlmostEqual(links['centroid_motion_support'][0], .75)

    def test_stationary_and_unsupported_ratios_are_unavailable(self):
        args = self.fixture()
        args[2][93] = np.zeros(3)
        links, _, event = add_centroid_motion_diagnostics(*args)
        self.assertTrue(np.isnan(links['centroid_motion_recovery'][0]))
        self.assertAlmostEqual(links['centroid_motion_error'][0], .6)
        self.assertIsNone(event['median_centroid_recovery'])
        args[1]['matched_fraction'][:] = 0
        links, _, event = add_centroid_motion_diagnostics(*args)
        self.assertEqual(links['centroid_motion_support'][0], 0)
        self.assertTrue(np.isnan(links['boundary_mean_displacement_zyx']).all())

    def test_branch_merge_and_nonconvergence_do_not_enter_ratio_summary(self):
        links, points, centers, graph = self.fixture()
        for extra in ((17, 94), (18, 93)):
            branched = np.r_[graph, np.array([extra], dtype=graph.dtype)]
            result, _, event = add_centroid_motion_diagnostics(links, points, centers, branched)
            self.assertEqual(event['comparison_link_count'], 0)
            self.assertEqual(result['centroid_comparison_valid'][0], 0)
        links['solver_converged'] = 0
        _, _, event = add_centroid_motion_diagnostics(links, points, centers, graph)
        self.assertEqual(event['comparison_link_count'], 0)

    def test_centroids_use_ids_calibration_and_registration(self):
        observations = np.array([(93, 2, 1., 2., 3.), (17, 1, 1., 2., 2.)], dtype=[
            ('observation_id','i8'), ('frame','i4'), ('centroid_z','f8'), ('centroid_y','f8'), ('centroid_x','f8')])
        class Registration:
            def apply_zyx(self, points, frames):
                return points + frames[:, None]*np.array([0., 0., .5])
        result = registered_centroid_lookup(observations, [3., .325, .325], Registration())
        np.testing.assert_allclose(result[93]-result[17], [0, 0, .825])
        np.testing.assert_allclose(result[17], [3, .65, 1.15])


if __name__ == '__main__':
    unittest.main()
