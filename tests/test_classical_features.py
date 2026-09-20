from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from skimage.measure import label, regionprops

from celltraj2.centroid_features import collective_values, voronoi_neighbors
from celltraj2.feature_catalog import component_metrics, motion_metrics
from celltraj2.features import _mask_component_values
from celltraj2.image_features import masked_glcm, normalize_intensity
from celltraj2.schema import ChannelSpec, ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore
from celltraj2.surface_flow_features import radial_shape_magnitudes_2d, surface_flow_fields
from celltraj2.trajectory import Trajectory


class ClassicalFeatureTests(unittest.TestCase):
    def test_clipped_voronoi_2d_and_3d(self):
        for ndim, expected in ((2, 5), (3, 25)):
            from itertools import product
            points = np.array(list(product([2., 8.], repeat=ndim)))
            neighbors = voronoi_neighbors(points, np.zeros(ndim), np.full(ndim, 10.))
            for row in neighbors:
                self.assertEqual(len(row), ndim)
                np.testing.assert_allclose([weight for _, weight in row], expected)
        with self.assertRaisesRegex(ValueError, "Coincident"):
            voronoi_neighbors(np.ones((2, 2)), [0, 0], [3, 3])

    def test_collective_legacy_sign_and_weighting(self):
        points = np.array([[1., 1.], [2., 1.], [1., 2.]])
        vectors = np.array([[-1., 0.], [1., 0.], [0., 1.]])
        values = collective_values(points, vectors, vectors, [[(1, 3.), (2, 1.)], [], []])
        self.assertAlmostEqual(values["beta"][0], -.75)
        self.assertAlmostEqual(values["alpha"][0], 1.75)
        self.assertAlmostEqual(values["nematic_alignment"][0], .5)
        stationary = collective_values(points, np.zeros_like(vectors), np.zeros_like(vectors), [[(1, 1.)], [], []])
        self.assertTrue(np.isnan(stationary["beta"][0]))
        self.assertEqual(stationary["relative_speed"][0], 0)

    def test_planar_surface_derivatives(self):
        y, x = np.mgrid[-3:4, -3:4]
        p = np.column_stack([np.zeros(x.size), y.ravel(), x.ravel()])
        normals = np.tile([1., 0., 0.], (len(p), 1))
        v = np.column_stack([(x + 2*y).ravel(), (3*x + 4*y).ravel(), (2*x - 3*y).ravel()])
        fields = surface_flow_fields(p, v, normals, neighbors=16)
        for field, expected in (("normal_gradient_squared", 5), ("surface_divergence", 6),
                                ("tangential_vorticity", 6), ("tangential_strain_rate", np.sqrt(20))):
            np.testing.assert_allclose(fields[field], expected, atol=1e-10)
        # Scaling coordinates and velocity together preserves spatial velocity gradients.
        scaled = surface_flow_fields(p * 5, v * 5, normals)
        np.testing.assert_allclose(scaled["surface_divergence"], fields["surface_divergence"])

    def test_curve_has_no_intrinsic_vorticity(self):
        x = np.arange(10.)
        p = np.column_stack([x*0, x*0, x])
        n = np.tile([0., 1., 0.], (len(x), 1))
        v = np.column_stack([x*0, 2*x, 3*x])
        fields = surface_flow_fields(p, v, n, spatial_ndim=2, neighbors=5)
        np.testing.assert_allclose(fields["surface_divergence"], 3, atol=1e-10)
        np.testing.assert_allclose(fields["normal_gradient_squared"], 4, atol=1e-10)
        self.assertTrue(np.isnan(fields["tangential_vorticity"]).all())

    def test_radial_shape_2d_rotation_and_sampling(self):
        theta = np.linspace(0, 2*np.pi, 1200, endpoint=False)
        biased = np.r_[theta, np.linspace(0, .6, 1500)]
        def ellipse(t, rotation=0):
            radius = 1 / np.sqrt((np.cos(t)/3)**2 + np.sin(t)**2)
            return np.column_stack([t*0, radius*np.sin(t+rotation), radius*np.cos(t+rotation)])
        original = radial_shape_magnitudes_2d(ellipse(theta), center_zyx=[0,0,0], order=4)
        rotated = radial_shape_magnitudes_2d(ellipse(biased, .37), center_zyx=[0,0,0], order=4)
        np.testing.assert_allclose(rotated, original, atol=5e-5)
        self.assertGreater(original[2], .2)
        circle = np.column_stack([theta*0, np.sin(theta), np.cos(theta)])
        np.testing.assert_allclose(radial_shape_magnitudes_2d(circle, center_zyx=[0,0,0], order=4), [1,0,0,0,0], atol=1e-12)

    def test_mask_largest_component_is_not_max_roundness(self):
        mask = np.zeros((20, 20), bool)
        mask[1:3, 1:12] = True
        mask[10:13, 10:13] = True
        components = regionprops(label(mask))
        metrics = component_metrics({"metrics": ["component_count"], "fields": ["roundness", "area"], "statistics": ["mean", "min", "max", "largest"]})
        values = _mask_component_values(np.ones_like(mask), mask, components, metrics=metrics, voxel_measure=1, np=np)
        self.assertEqual(values["component_count"], 2)
        self.assertLess(values["component_roundness_largest"], values["component_roundness_max"])
        self.assertEqual(values["component_area_largest"], 22)
        self.assertEqual(motion_metrics({"metrics": ["normal_mean"]}), ["normal_mean"])

    def test_glcm_excludes_background_pairs(self):
        image = np.arange(64.).reshape(8,8)
        mask = np.zeros((8,8), bool)
        mask[2:6,2:6] = True
        first, _ = masked_glcm(image, mask)
        image[~mask] = -1e12
        second, _ = masked_glcm(image, mask)
        self.assertEqual(first, second)
        constant, _ = masked_glcm(np.ones((5,5)), np.ones((5,5), bool))
        self.assertEqual(constant["contrast"], 0)
        self.assertEqual(constant["energy"], 1)
        isolated = np.eye(3, dtype=bool)
        empty, qc = masked_glcm(np.ones((3,3)), isolated, distances=[5])
        self.assertTrue(np.isnan(empty["contrast"]))
        self.assertEqual(qc["pair_count"], 0)

    def _movie(self, path):
        metadata = TrajectoryMetadata(roi_id="sample", dataset_id="sample", frame_count=3,
            channels=[ChannelSpec(raw_index=0, display_name="Signal")],
            image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Y", "X", "C")),
            acquisition={"micron_per_pixel": 2., "time_interval_s": 1800.})
        with TrajectoryStore.create(path, metadata=metadata) as store:
            for frame in range(1,4):
                labels = np.zeros((16,16), np.uint16)
                labels[3:6,frame:frame+3] = 1
                labels[9:12,frame:frame+3] = 2
                store.write_label_frame("cells", frame, labels)
                store.write_raw_frame(frame, (np.arange(256.).reshape(16,16,1) + 1) * frame)
        with Trajectory(path, mode="r+") as trajectory:
            trajectory.index_observations("cells")
            trajectory.track_minimum_centroid_distance("cells", track_set="tracks", max_distance=5., coordinate_scale=(2,2,2))

    def test_row_aligned_centroid_and_normalization_persistence(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"movie.h5"
            self._movie(path)
            with Trajectory(path, mode="r+") as trajectory:
                result = trajectory.extract_features({"feature_set": "classical", "object_set": "cells", "features": [
                    {"kind": "centroid_motility", "name": "move", "track_set": "tracks", "registration_set": "",
                     "metrics": ["speed", "displacement", "persistence", "acceleration", "beta", "alpha", "neighbor_count"]},
                    {"kind": "intensity", "name": "normalized", "stats": ["mean"], "channel": 0,
                     "normalization": {"method": "match_mean", "scope": "per_frame"}},
                    {"kind": "texture", "name": "texture", "channel": 0, "metrics": ["contrast", "asm"]},
                ]})
                np.testing.assert_allclose(result.values["move_speed"][2:], 4)
                np.testing.assert_allclose(result.values["move_displacement"][2:], 2)
                np.testing.assert_allclose(result.values["move_beta"][2:], 1)
                np.testing.assert_allclose(result.values["move_alpha"][2:], 0)
                np.testing.assert_allclose(result.values["move_acceleration"][4:], 0)
                np.testing.assert_allclose(result.values["move_persistence"][4:], 1)
                self.assertTrue(np.isnan(result.values["move_speed"][:2]).all())
                columns = result.schema["columns"]
                normalized = next(column for column in columns if column["name"] == "normalized")
                self.assertEqual(len(normalized["normalization"]["frame_parameters"]), 3)
                self.assertAlmostEqual(normalized["normalization"]["frame_parameters"]["3"]["scale"], 1/3)
                self.assertIn("texture_slice_z", result.values.dtype.names)
                self.assertTrue(trajectory.store.has_feature_set("cells", "classical"))

    def test_normalization_pooled_and_empty_reference(self):
        images = [np.arange(1., 5.).reshape(2,2), np.arange(1., 5.).reshape(2,2)*2]
        trajectory = SimpleNamespace(metadata=SimpleNamespace(frame_count=2), get_image_data=lambda frame, channels: images[frame-1])
        channel = {"local_index": 0}
        feature = {"normalization": {"method": "zscore", "scope": "all_frames"}}
        from unittest.mock import patch
        with patch("celltraj2.features._read_channel_image", side_effect=lambda t, f, c, np: images[f-1]):
            cache = {}
            combined = []
            for frame, image in enumerate(images, 1):
                normalized, _, _ = normalize_intensity(trajectory, image, frame=frame, channel=channel, feature=feature, cache=cache)
                combined.extend(normalized.ravel())
            self.assertAlmostEqual(np.mean(combined), 0)
            self.assertAlmostEqual(np.std(combined), 1)
        normalized, _, warnings = normalize_intensity(trajectory, np.zeros((2,2)), frame=1, channel=channel,
            feature={"normalization": {"method": "zscore"}}, cache={})
        self.assertTrue(np.isnan(normalized).all())
        self.assertTrue(warnings)

    def test_texture_3d_uses_largest_xy_slice(self):
        from celltraj2.image_features import compute_texture_frame
        from unittest.mock import patch
        labels = np.zeros((3,6,6), np.uint16)
        labels[0,2:4,2:4] = 1
        labels[1,1:5,1:5] = 1
        labels[2,2:4,2:4] = 1
        image = np.zeros_like(labels, float)
        image[1] = np.indices((6,6)).sum(axis=0) % 2
        trajectory = SimpleNamespace()
        with patch("celltraj2.features._resolve_channel", return_value={"schema": {}}), patch("celltraj2.features._read_channel_image", return_value=image):
            result = compute_texture_frame(trajectory, labels, frame=1, source_label_set="cells",
                                            feature={"name": "tex", "compartment": {}, "levels": 2})
        self.assertEqual(result["values_by_label"][1]["tex_slice_z"], 1)
        self.assertEqual(result["values_by_label"][1]["tex_slice_area"], 16)
        self.assertGreater(result["values_by_label"][1]["tex_contrast"], 0)

    def test_stored_surface_motion_fields_and_shape(self):
        from test_features import FeatureExtractionTests
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"surfaces.h5"
            fixture = FeatureExtractionTests()
            fixture.np = np
            fixture._create_boundary_feature_h5(path)
            with Trajectory(path, mode="r+") as trajectory:
                result = trajectory.extract_features({"feature_set": "flows", "object_set": "cells", "features": [
                    {"kind": "boundary_motion", "name": "flow", "boundary_set": "cell_surfaces", "boundary_source_name": "tracked_cells",
                     "motion_set": "surface_ot", "geometry_set": "geometry", "as_velocity": True,
                     "fields": ["magnitude", "normal_gradient_squared", "surface_divergence", "tangential_vorticity"],
                     "statistics": ["mean", "max"], "metrics": ["mapped_fraction", "derivative_valid_fraction"]},
                    {"kind": "boundary_multipole", "name": "shape", "boundary_set": "cell_surfaces", "signal": "shape_radial_deviation", "order": 4},
                ]}, save_outputs=False)
                self.assertIn("flow_surface_divergence_mean", result.values.dtype.names)
                self.assertTrue(np.isnan(result.values["flow_tangential_vorticity_mean"]).all())
                self.assertTrue(np.isfinite(result.values["flow_magnitude_mean"][2:]).all())
                np.testing.assert_allclose(result.values["shape_l0"], 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
