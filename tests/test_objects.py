from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from celltraj2.object_indexing import ObjectIndexFileJob, _mask_to_labels, run_batch_object_indexing
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

            with Trajectory(path, mode="r+") as trajectory:
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

            with Trajectory(path, mode="r+") as trajectory:
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
                self.assertNotIn("frames", store.h5["runs/object_indexing/obj_index_test"])
            frame_record = next(event for event in events if event.get("event") == "frame_completed")
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

    def test_mask_source_job_round_trips_typed_source_and_conversion(self):
        job = ObjectIndexFileJob.from_dict(
            {
                "h5_path": "sample.ct2.h5",
                "object_set": "basement_objects",
                "source_kind": "mask_set",
                "source_mask_set": "basement",
                "derived_label_set": "basement_object_labels",
                "mask_conversion": "connected_components",
            }
        )

        self.assertEqual(job.source_name, "basement")
        self.assertEqual(job.source_labels, "basement_object_labels")
        self.assertEqual(job.to_dict()["source_kind"], "mask_set")
        self.assertEqual(job.to_dict()["mask_conversion"], "connected_components")

    def test_mask_whole_foreground_promotion_materializes_labels_and_provenance(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_labeled_h5(path)
            mask_1 = self.np.zeros((5, 6), dtype=bool)
            mask_1[1, 1] = True
            mask_1[3, 4] = True
            mask_2 = self.np.zeros((5, 6), dtype=bool)
            mask_2[2, 1:5] = True
            with TrajectoryStore.open(path, mode="r+") as store:
                store.write_mask_frame("basement", 1, mask_1)
                store.write_mask_frame("basement", 2, mask_2)

            dry_summary = run_batch_object_indexing(
                {
                    "job_id": "promote_mask_preview",
                    "save_outputs": False,
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "basement_objects",
                            "source_kind": "mask_set",
                            "source_mask_set": "basement",
                            "source_label_set": "basement_object_labels",
                            "mask_conversion": "whole_foreground",
                        }
                    ],
                }
            )
            self.assertEqual(dry_summary.observations, 2)
            with TrajectoryStore.open(path, mode="r") as store:
                self.assertNotIn("labels/basement_object_labels", store.h5)
                self.assertNotIn("object_sets/basement_objects", store.h5)

            events = []
            saved_summary = run_batch_object_indexing(
                {
                    "job_id": "promote_mask",
                    "save_outputs": True,
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "basement_objects",
                            "source_kind": "mask_set",
                            "source_mask_set": "basement",
                            "source_label_set": "basement_object_labels",
                            "mask_conversion": "whole_foreground",
                        }
                    ],
                },
                reporter=lambda event: events.append(dict(event)),
            )
            self.assertEqual(saved_summary.observations, 2, events)
            with TrajectoryStore.open(path, mode="r") as store:
                derived = store.h5["labels/basement_object_labels/frame_1"][()]
                self.assertEqual(sorted(self.np.unique(derived).tolist()), [0, 1])
                self.assertEqual(store.observation_count("basement_objects"), 2)
                metadata = store.read_json(
                    "/object_sets/basement_objects/object_set.json"
                )["metadata"]
                self.assertEqual(metadata["source_kind"], "mask_set")
                self.assertEqual(metadata["source_mask_set"], "basement")
                self.assertEqual(metadata["derived_label_set"], "basement_object_labels")

    def test_connected_component_mask_promotion_assigns_one_label_per_region(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy is not installed")
        mask = self.np.zeros((6, 6), dtype=bool)
        mask[1:3, 1:3] = True
        mask[4, 4] = True

        labels = _mask_to_labels(mask, conversion="connected_components")

        self.assertEqual(sorted(self.np.unique(labels).tolist()), [0, 1, 2])

    def test_mask_promotion_refuses_existing_derived_frames_before_commit(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_labeled_h5(path)
            mask = self.np.zeros((4, 4), dtype=bool)
            mask[1:3, 1:3] = True
            sentinel = self.np.full((4, 4), 7, dtype=self.np.uint16)
            with TrajectoryStore.open(path, mode="r+") as store:
                store.write_mask_frame("basement", 1, mask)
                store.write_mask_frame("basement", 2, mask)
                store.write_label_frame("basement_object_labels", 1, sentinel)

            events = []
            summary = run_batch_object_indexing(
                {
                    "job_id": "promote_mask_collision",
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "basement_objects",
                            "source_kind": "mask_set",
                            "source_mask_set": "basement",
                            "source_label_set": "basement_object_labels",
                        }
                    ],
                },
                reporter=lambda event: events.append(dict(event)),
            )

            self.assertEqual(summary.failed, 1)
            self.assertTrue(any("Derived label frames already exist" in str(event) for event in events))
            with TrajectoryStore.open(path, mode="r") as store:
                self.np.testing.assert_array_equal(
                    store.read_label_frame("basement_object_labels", 1), sentinel
                )
                self.assertNotIn("labels/basement_object_labels/frame_2", store.h5)
                self.assertNotIn("object_sets/basement_objects", store.h5)


if __name__ == "__main__":
    unittest.main()
