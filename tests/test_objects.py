from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from celltraj2.object_indexing import run_batch_object_indexing
from celltraj2.schema import ImageSourceSpec, RoiSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore
from celltraj2.trajectory import Trajectory


class ObjectIndexingTests(unittest.TestCase):
    def setUp(self):
        try:
            import h5py  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("h5py/numpy are not installed")
        self.np = np

    def _create_labeled_h5(self, path: Path) -> None:
        metadata = TrajectoryMetadata(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            frame_count=2,
            roi=RoiSpec(roi_id="sample_XY001_ROI001", dataset_id="sample", time_start=4, time_stop=6),
            image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
        )
        with TrajectoryStore.create(path, metadata=metadata) as store:
            labels_1 = self.np.array([[0, 2, 2], [1, 0, 0]], dtype=self.np.uint16)
            labels_2 = self.np.array([[0, 3, 3], [0, 0, 3]], dtype=self.np.uint16)
            store.write_label_frame("tumor", 1, labels_1)
            store.write_label_frame("tumor", 2, labels_2)

    def test_index_observations_are_one_based_and_row_aligned(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_labeled_h5(path)

            with Trajectory(path) as trajectory:
                result = trajectory.object_set("tumor").index_observations(run_id="index_tumor")
                observations = trajectory.object_set("tumor").read_observations()
                lookup_1 = trajectory.object_set("tumor").read_lookup_frame(1)
                lookup_2 = trajectory.object_set("tumor").read_lookup_frame(2)

                self.assertEqual(result.observation_count, 3)
                self.assertEqual(observations["observation_id"].tolist(), [1, 2, 3])
                self.assertEqual(observations["label_id"].tolist(), [1, 2, 3])
                self.assertEqual(observations["frame"].tolist(), [1, 1, 2])
                self.assertEqual(observations["parent_time_index"].tolist(), [4, 4, 5])
                self.assertEqual(int(observations[0]["z_min"]), 0)
                self.assertEqual(int(observations[0]["z_max"]), 1)
                self.assertEqual(int(observations[0]["y_min"]), 1)
                self.assertEqual(int(observations[0]["y_max"]), 2)
                self.assertEqual(int(observations[0]["x_min"]), 0)
                self.assertEqual(int(observations[0]["x_max"]), 1)
                self.assertEqual(int(observations[1]["voxel_count"]), 2)
                self.assertEqual(int(lookup_1[0]), 0)
                self.assertEqual(int(lookup_1[1]), 1)
                self.assertEqual(int(lookup_1[2]), 2)
                self.assertEqual(int(lookup_2[3]), 3)
                self.assertEqual(trajectory.object_set("tumor").observation_id_for_label(frame=1, label_id=2), 2)
                self.assertEqual(trajectory.object_set("tumor").observation_id_for_label(frame=1, label_id=99), 0)
                self.assertEqual(trajectory.object_sets(), ["tumor"])
                self.assertEqual(trajectory.object_indexing_runs(), ["index_tumor"])

    def test_existing_observation_index_requires_overwrite(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_labeled_h5(path)

            with Trajectory(path) as trajectory:
                trajectory.index_observations("tumor", run_id="first")
                with self.assertRaises(FileExistsError):
                    trajectory.index_observations("tumor", run_id="second")
                trajectory.index_observations("tumor", frames=[1], overwrite=True, run_id="third")
                self.assertEqual(trajectory.object_set("tumor").observation_count(), 2)
                self.assertEqual(trajectory.object_set("tumor").lookup_frames(), [1])

    def test_batch_object_indexing_writes_run_metadata(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_labeled_h5(path)

            events = []
            summary = run_batch_object_indexing(
                {
                    "job_id": "obj_index_test",
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "tumor",
                            "frames": {"mode": "all"},
                        }
                    ],
                },
                reporter=lambda event: events.append(dict(event)),
            )

            self.assertEqual(summary.completed, 2)
            self.assertEqual(summary.observations, 3)
            with TrajectoryStore.open(path, mode="r") as store:
                self.assertEqual(store.list_object_indexing_runs(), ["obj_index_test"])
                run_record = store.read_object_indexing_run("obj_index_test")
                self.assertEqual(run_record["status"], "completed")
                self.assertEqual(run_record["observation_count"], 3)
                frame_record = store.read_object_indexing_frame_result("obj_index_test", 1)
                self.assertEqual(frame_record["observation_count"], 2)
            self.assertIn("job_completed", [event.get("event") for event in events])

    def test_batch_object_indexing_dry_run_does_not_write_h5_outputs(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_labeled_h5(path)

            summary = run_batch_object_indexing(
                {
                    "job_id": "obj_index_preview",
                    "save_outputs": False,
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "tumor",
                            "frames": {"mode": "range", "frame_start": 1, "frame_stop": 1},
                        }
                    ],
                },
            )

            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.observations, 2)
            with TrajectoryStore.open(path, mode="r") as store:
                self.assertEqual(store.list_object_sets(), [])
                self.assertEqual(store.list_object_indexing_runs(), [])


if __name__ == "__main__":
    unittest.main()
