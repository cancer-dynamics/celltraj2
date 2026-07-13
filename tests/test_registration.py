from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from celltraj2.registration import (
    FRAME_STATUS,
    estimate_pair_translation,
    identity_registration,
    register_global_translation,
)
from celltraj2.registration_batch import RegistrationBatchJob, run_batch_registration
from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore
from celltraj2.trajectory import Trajectory


class GlobalRegistrationTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        self.np = np

    def require_scipy(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy is not installed")

    def require_h5py(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py is not installed")

    def test_grid_plus_continuous_refinement_recovers_subpixel_shift(self):
        self.require_scipy()
        reference = self.np.asarray([[0.0, 0.0], [10.0, 1.0], [3.0, 8.0], [8.0, 6.0]])
        moving = reference + self.np.asarray([1.25, -2.5])
        result = estimate_pair_translation(reference, moving, max_shift=4.0, grid_step=1.0)
        self.assertTrue(result["success"])
        self.np.testing.assert_allclose(result["shift"], [-1.25, 2.5], atol=1e-3)
        self.assertLessEqual(result["objective_score"], result["coarse_score"])

    def test_identity_registration_is_a_complete_native_coordinate_map(self):
        metadata = TrajectoryMetadata(
            roi_id="roi",
            dataset_id="sample",
            frame_count=2,
            image_source=ImageSourceSpec(
                source_type="embedded_h5",
                axes=("T", "Y", "X", "C"),
                sizes={"T": 2, "Y": 8, "X": 9, "C": 1},
            ),
        )
        registration = identity_registration(metadata)
        self.assertTrue(registration.schema["registration_complete"])
        points = self.np.asarray([[0.0, 2.5, 3.5], [0.0, 6.0, 7.0]])
        self.np.testing.assert_allclose(registration.apply_zyx(points, [1, 2]), points)

    def test_failed_pair_does_not_become_the_next_registration_anchor(self):
        observations = self.np.zeros(
            3,
            dtype=[
                ("frame", "<i4"),
                ("centroid_z", "<f8"),
                ("centroid_y", "<f8"),
                ("centroid_x", "<f8"),
            ],
        )
        observations["frame"] = [1, 2, 3]
        observations["centroid_x"] = [1.0, 2.0, 3.0]

        class Store:
            @staticmethod
            def read_observations(_object_set):
                return observations

        class FakeTrajectory:
            metadata = TrajectoryMetadata(
                roi_id="roi",
                dataset_id="sample",
                frame_count=3,
                image_source=ImageSourceSpec(
                    source_type="embedded_h5",
                    axes=("T", "Y", "X", "C"),
                    sizes={"T": 3, "Y": 8, "X": 9, "C": 1},
                ),
            )
            store = Store()
            path = Path("sample.ct2.h5")

        failed = {
            "shift": self.np.asarray([0.0, 0.0]),
            "success": False,
            "optimizer_method": "L-BFGS-B",
            "coarse_score": 1.0,
            "refined_score": 1.0,
            "objective_score": 1.0,
            "optimizer_nit": 1,
        }
        recovered = {
            "shift": self.np.asarray([0.0, -2.0]),
            "success": True,
            "optimizer_method": "L-BFGS-B",
            "coarse_score": 0.0,
            "refined_score": 0.0,
            "objective_score": 0.0,
            "optimizer_nit": 1,
        }
        with patch(
            "celltraj2.registration.estimate_pair_translation",
            side_effect=[failed, recovered],
        ):
            result = register_global_translation(
                FakeTrajectory(),
                "cells",
                save_outputs=False,
            )

        pairs = result.registration.pairwise_results
        self.assertEqual(pairs[0]["source_frame"], 1)
        self.assertEqual(pairs[1]["source_frame"], 1)
        self.assertEqual(pairs[1]["target_frame"], 3)
        self.assertEqual(int(result.registration.frame_status[1]), FRAME_STATUS["failed"])
        self.assertEqual(int(result.registration.frame_status[2]), FRAME_STATUS["estimated"])

    def _create_drift_h5(self, path: Path, *, partial: bool = False) -> None:
        self.require_h5py()
        metadata = TrajectoryMetadata(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            frame_count=4 if partial else 2,
            image_source=ImageSourceSpec(
                source_type="embedded_h5",
                axes=("T", "Y", "X", "C"),
                sizes={"T": 4 if partial else 2, "Y": 16, "X": 16, "C": 1},
            ),
            acquisition={
                "micron_per_pixel": 1.0,
                "voxel_size_um": {"Y": 1.0, "X": 1.0},
            },
        )
        with TrajectoryStore.create(path, metadata=metadata) as store:
            frame_1 = self.np.zeros((16, 16), dtype=self.np.uint16)
            frame_1[2, 2] = 1
            frame_1[7, 4] = 2
            frame_1[11, 10] = 3
            target_frame = 3 if partial else 2
            frame_target = self.np.zeros((16, 16), dtype=self.np.uint16)
            frame_target[2, 5] = 1
            frame_target[7, 7] = 2
            frame_target[11, 13] = 3
            store.write_label_frame("cells", 1, frame_1)
            store.write_label_frame("cells", target_frame, frame_target)
        with Trajectory(path) as trajectory:
            trajectory.index_observations("cells", frames=[1, target_frame], run_id="index_cells")

    def test_identity_registration_is_initialized_for_every_new_h5(self):
        self.require_h5py()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            metadata = TrajectoryMetadata(
                roi_id="roi",
                dataset_id="sample",
                frame_count=3,
                image_source=ImageSourceSpec(
                    source_type="embedded_h5",
                    axes=("T", "Y", "X", "C"),
                    sizes={"T": 3, "Y": 8, "X": 9, "C": 1},
                ),
            )
            with TrajectoryStore.create(path, metadata=metadata) as store:
                self.assertEqual(store.list_registration_sets(), ["identity"])
                self.assertEqual(store.active_registration_name(), "identity")
                identity = store.read_active_registration()
                self.assertEqual(identity.transforms.shape, (3, 3, 3))
                self.assertEqual(identity.canvas["output_shape"], [8, 9])

    def test_registration_cancels_drift_for_centroid_tracking(self):
        self.require_h5py()
        self.require_scipy()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_drift_h5(path)
            with Trajectory(path) as trajectory:
                native = trajectory.track_minimum_centroid_distance(
                    "cells",
                    max_distance=1.0,
                    coordinate_scale=(1.0, 1.0, 1.0),
                    registration_set="identity",
                    save_outputs=False,
                    metadata={"distance_unit": "um"},
                )
                self.assertEqual(native.link_count, 0)
                registered = trajectory.register_global_translation(
                    "cells",
                    max_shift_per_frame=5.0,
                    grid_step=1.0,
                    run_id="register_cells",
                )
                self.np.testing.assert_allclose(
                    registered.registration.translation_zyx(2),
                    [0.0, 0.0, -3.0],
                    atol=1e-6,
                )
                tracked = trajectory.track_minimum_centroid_distance(
                    "cells",
                    max_distance=1.0,
                    coordinate_scale=(1.0, 1.0, 1.0),
                    track_set="registered",
                    metadata={"distance_unit": "um"},
                )
                self.assertEqual(tracked.link_count, 3)
                dependency = tracked.graph.schema["registration_dependency"]
                self.assertEqual(dependency["registration_set"], "global_registration")
                self.assertEqual(dependency["registration_digest"], registered.registration.digest)

    def test_partial_frames_inherit_previous_absolute_transform(self):
        self.require_h5py()
        self.require_scipy()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_drift_h5(path, partial=True)
            with Trajectory(path) as trajectory:
                result = trajectory.register_global_translation(
                    "cells",
                    max_shift_per_frame=2.0,
                    grid_step=1.0,
                    run_id="register_partial",
                )
                registration = result.registration
                self.assertEqual(int(registration.frame_status[1]), FRAME_STATUS["inherited"])
                self.assertEqual(int(registration.frame_status[2]), FRAME_STATUS["estimated"])
                self.assertEqual(int(registration.frame_status[3]), FRAME_STATUS["inherited"])
                self.np.testing.assert_allclose(
                    registration.translation_zyx(4),
                    registration.translation_zyx(3),
                )
                self.assertFalse(registration.schema["registration_complete"])

    def test_registration_batch_dry_run_and_saved_run(self):
        self.require_h5py()
        self.require_scipy()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_drift_h5(path)
            job = RegistrationBatchJob.from_dict(
                {
                    "job_id": "registration_preview",
                    "save_outputs": False,
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "cells",
                            "max_shift_per_frame": 5.0,
                            "grid_step": 1.0,
                            "coordinate_scale": [1.0, 1.0, 1.0],
                            "distance_unit": "um",
                        }
                    ],
                }
            )
            preview = run_batch_registration(job)
            self.assertEqual(preview.completed, 1)
            with Trajectory(path) as trajectory:
                self.assertEqual(trajectory.registration_sets(), ["identity"])
            saved = run_batch_registration(
                {
                    "job_id": "registration_saved",
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "cells",
                            "max_shift_per_frame": 5.0,
                            "grid_step": 1.0,
                            "coordinate_scale": [1.0, 1.0, 1.0],
                            "distance_unit": "um",
                        }
                    ],
                }
            )
            self.assertEqual(saved.completed, 1)
            with Trajectory(path) as trajectory:
                self.assertEqual(trajectory.active_registration_name(), "global_registration")
                self.assertEqual(trajectory.registration_runs(), ["registration_saved"])


if __name__ == "__main__":
    unittest.main()
