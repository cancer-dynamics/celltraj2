"""Physical and serialization checks for motion alignment, independent of tracking scores."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from celltraj2.tracking import _transport_pair_samples
from celltraj2.boundaries import optimal_transport_plan
from celltraj2.tracking_batch import TrackingFileJob


class TransportAlignmentTests(unittest.TestCase):
    def test_registered_calibrated_motion_with_capped_surfaces(self):
        from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
        from celltraj2.store import TrajectoryStore
        from celltraj2.trajectory import Trajectory
        from celltraj2.registration import RegistrationSet, pairwise_result_dtype
        scale = np.array([2., .5, .5])
        delta = np.array([1.5, .2, .7])  # native translation [2,0,1] plus registration
        z, y, x = np.indices((15, 21, 21))
        first = (((z-6)/3)**2 + ((y-9)/4)**2 + ((x-8)/4)**2 <= 1).astype("uint16")
        second = np.roll(first, (1, 0, 2), axis=(0, 1, 2))
        with TemporaryDirectory() as directory:
            path = Path(directory)/"translated.h5"
            metadata = TrajectoryMetadata(roi_id="synthetic", dataset_id="synthetic", frame_count=2,
                image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
                acquisition={"micron_per_pixel": .5, "voxel_size_um": dict(zip("ZYX", scale))})
            with TrajectoryStore.create(path, metadata=metadata) as store:
                store.write_label_frame("cells", 1, first)
                store.write_label_frame("cells", 2, second)
            with Trajectory(path, mode="r+") as trajectory:
                trajectory.index_observations("cells", run_id="index")
                trajectory.object_set("cells").build_boundary_library("surface", point_spacing=None)
                transforms = np.repeat(np.eye(4)[None], 2, axis=0)
                transforms[1, :3, -1] = [-.5, .2, -.3]
                registration = RegistrationSet(name="registered", frames=np.array([1, 2]),
                    transforms=transforms, frame_status=np.array([1, 2], dtype="uint8"),
                    pairwise_results=np.empty(0, dtype=pairwise_result_dtype()), canvas={},
                    schema={"method":"test_translation", "spatial_axes":list("ZYX"),
                            "coordinate_scale_zyx":scale.tolist()})
                trajectory.store.write_registration_set(registration)
                trajectory.store.set_active_registration("registered", reason="physical regression")
                result = trajectory.object_set("cells").track_minimum_boundary_ot_cost(
                    boundary_set="surface", max_distance=10., track_set="tracks",
                    winner_cost_prealign="cell_centroid", max_boundary_points=32,
                    max_transport_distance=.1)
                motion = result.motion_result
                self.assertEqual(len(motion.links), 1)
                self.assertGreater(motion.links["sampling_voxel_spacing"][0], 0)
                self.assertLessEqual(motion.links["source_summary_count"][0], 32)
                np.testing.assert_allclose(motion.links["centroid_displacement_zyx"][0], delta, atol=1e-8)
                np.testing.assert_allclose(motion.links["target_cost_origin_zyx"][0]
                                           - motion.links["source_cost_origin_zyx"][0], delta, atol=1e-8)
                vectors = motion.transport["registered_displacement_zyx"]
                np.testing.assert_allclose(vectors, np.broadcast_to(delta, vectors.shape), atol=1e-6)
                independent = trajectory.compute_boundary_motion("cells", "tracks", boundary_set="surface",
                    motion_set="independent", cost_prealign="cell_centroid", max_boundary_points=32,
                    max_transport_distance=.1)
                np.testing.assert_array_equal(independent.transport["registered_displacement_zyx"], vectors)

    def pair(self, source, target, mode="cell_centroid", cap=64):
        return _transport_pair_samples(source, target, max_boundary_points=cap,
            mass_mode="probability", point_measure=1., cost_prealign=mode, np=np)

    def test_translated_unequal_clouds_preserve_measure_and_full_motion(self):
        rng = np.random.default_rng(42)
        p = rng.normal(size=(150, 3))*[2., 4., 3.]
        delta = np.array([3., -2., 5.])
        # Duplicate target sampling eightfold without changing the surface measure.
        source = dict(registered=p, cell_centroid=np.zeros(3))
        target = dict(registered=np.repeat(p+delta, 8, axis=0), cell_centroid=delta)
        pair = self.pair(source, target)
        np.testing.assert_allclose(pair["source_cost"], pair["target_cost"], atol=1e-12)
        np.testing.assert_allclose(pair["source"].weights, pair["target"].weights)
        plan = optimal_transport_plan(pair["source_cost"], pair["target_cost"],
            method="unbalanced", max_transport_distance=.01,
            source_weights=pair["source"].weights, target_weights=pair["target"].weights)
        # A residual-distance gate can be smaller than the actual translation.
        vectors = (target["registered"][pair["target"].rows[plan.target_rows]]
                   - source["registered"][pair["source"].rows[plan.source_rows]])
        np.testing.assert_allclose(vectors, np.broadcast_to(delta, vectors.shape), atol=1e-12)
        self.assertLessEqual(len(pair["source"].rows), 64)

    def test_invalid_centroids_fail_only_when_requested(self):
        p = np.array([[0., 1., 2.], [0., 3., 4.]])
        for center in (None, [0., np.nan, 0.], [0., 1.]):
            points = dict(registered=p, cell_centroid=center)
            with self.assertRaisesRegex(ValueError, "finite registered observation centroids"):
                self.pair(points, points)
            np.testing.assert_array_equal(self.pair(points, points, "none")["source_cost"], p)
        with self.assertRaisesRegex(ValueError, "cost_prealign"):
            self.pair(dict(registered=p), dict(registered=p), "typo")

    def test_saved_jobs_preserve_new_and_legacy_semantics(self):
        base = dict(h5_path="synthetic.h5", object_set="cells", cost_prealign="centroid")
        old = TrackingFileJob.from_dict(base)
        self.assertIsNone(old.winner_cost_prealign)
        for mode in (None, "none", "cell_centroid", "centroid"):
            job = TrackingFileJob.from_dict(dict(base, winner_cost_prealign=mode))
            replay = TrackingFileJob.from_dict(job.to_dict())
            self.assertEqual(replay.winner_cost_prealign, mode)
            self.assertEqual(replay.cost_prealign, "centroid")
        with self.assertRaisesRegex(ValueError, "winner_cost_prealign"):
            TrackingFileJob.from_dict(dict(base, winner_cost_prealign="typo"))


if __name__ == "__main__":
    unittest.main()
