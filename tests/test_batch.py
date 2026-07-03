from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from celltraj2.batch import SegmentationResult, run_batch_segmentation
from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore


class BatchSegmentationTests(unittest.TestCase):
    def setUp(self):
        try:
            import h5py  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("h5py/numpy are not installed")
        self.np = np

    def test_batch_segmentation_writes_labels_and_run_metadata(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            metadata = TrajectoryMetadata(
                roi_id="sample_XY001_ROI001",
                dataset_id="sample",
                frame_count=2,
                image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
            )
            with TrajectoryStore.create(path, metadata=metadata) as store:
                store.write_raw_frame(1, self.np.zeros((1, 3, 4, 1), dtype=self.np.uint16))
                store.write_raw_frame(2, self.np.ones((1, 3, 4, 1), dtype=self.np.uint16))

            def fake_segmenter(image, _file_job, frame):
                self.assertEqual(image.shape, (1, 3, 4))
                labels = self.np.full((1, 3, 4), frame, dtype=self.np.uint16)
                return SegmentationResult(labels=labels, metadata={"engine": "fake"})

            events = []
            summary = run_batch_segmentation(
                {
                    "job_id": "seg_test",
                    "files": [
                        {
                            "h5_path": str(path),
                            "label_set": "cyto",
                            "overwrite": True,
                            "frames": {"mode": "range", "frame_start": 1, "frame_stop": 2},
                            "backend": {"backend_id": "fake", "parameters": {"do_3D": True}},
                            "model_input": {
                                "channel_specs": [
                                    {"channel_indices": [0], "normalization": "raw", "combination": "single"}
                                ]
                            },
                        }
                    ],
                },
                fake_segmenter,
                reporter=lambda event: events.append(dict(event)),
            )

            self.assertEqual(summary.completed, 2)
            with TrajectoryStore.open(path, mode="r") as store:
                self.assertEqual(store.list_label_frames("cyto"), [1, 2])
                self.assertEqual(store.read_label_frame("cyto", 2).max(), 2)
                self.assertEqual(store.list_segmentation_runs(), ["seg_test"])
                run_record = store.read_segmentation_run("seg_test")
                self.assertEqual(run_record["status"], "completed")
                frame_record = store.read_segmentation_frame_result("seg_test", 1)
                self.assertEqual(frame_record["backend_metadata"]["engine"], "fake")
            self.assertIn("job_completed", [event.get("event") for event in events])

    def test_batch_segmentation_skips_existing_labels_without_overwrite(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            metadata = TrajectoryMetadata(
                roi_id="sample_XY001_ROI001",
                dataset_id="sample",
                frame_count=1,
                image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
            )
            with TrajectoryStore.create(path, metadata=metadata) as store:
                store.write_raw_frame(1, self.np.zeros((1, 3, 4, 1), dtype=self.np.uint16))
                store.write_label_frame("cyto", 1, self.np.ones((1, 3, 4), dtype=self.np.uint16))

            def should_not_run(_image, _file_job, _frame):
                raise AssertionError("existing labels should be skipped")

            summary = run_batch_segmentation(
                {
                    "job_id": "seg_skip",
                    "files": [
                        {
                            "h5_path": str(path),
                            "label_set": "cyto",
                            "frames": {"mode": "all"},
                            "backend": {"backend_id": "fake", "parameters": {"do_3D": True}},
                            "model_input": {
                                "channel_specs": [
                                    {"channel_indices": [0], "normalization": "raw", "combination": "single"}
                                ]
                            },
                        }
                    ],
                },
                should_not_run,
            )

            self.assertEqual(summary.skipped, 1)
            with TrajectoryStore.open(path, mode="r") as store:
                frame_record = store.read_segmentation_frame_result("seg_skip", 1)
                self.assertEqual(frame_record["status"], "skipped")

    def test_batch_segmentation_dry_run_does_not_write_h5_outputs(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            metadata = TrajectoryMetadata(
                roi_id="sample_XY001_ROI001",
                dataset_id="sample",
                frame_count=1,
                image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
            )
            with TrajectoryStore.create(path, metadata=metadata) as store:
                store.write_raw_frame(1, self.np.zeros((1, 3, 4, 1), dtype=self.np.uint16))

            def fake_segmenter(image, _file_job, frame):
                self.assertEqual(frame, 1)
                return self.np.ones(image.shape, dtype=self.np.uint16)

            events = []
            summary = run_batch_segmentation(
                {
                    "job_id": "seg_preview",
                    "save_outputs": False,
                    "files": [
                        {
                            "h5_path": str(path),
                            "label_set": "cyto",
                            "frames": {"mode": "all"},
                            "backend": {"backend_id": "fake", "parameters": {"do_3D": True}},
                            "model_input": {
                                "channel_specs": [
                                    {"channel_indices": [0], "normalization": "raw", "combination": "single"}
                                ]
                            },
                        }
                    ],
                },
                fake_segmenter,
                reporter=lambda event: events.append(dict(event)),
            )

            self.assertEqual(summary.completed, 1)
            frame_events = [event for event in events if event.get("event") == "frame_completed"]
            self.assertEqual(frame_events[0]["saved"], False)
            with TrajectoryStore.open(path, mode="r") as store:
                self.assertEqual(store.list_label_frames("cyto"), [])
                self.assertEqual(store.list_segmentation_runs(), [])

    def test_batch_segmentation_can_write_bool_masks(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            metadata = TrajectoryMetadata(
                roi_id="sample_XY001_ROI001",
                dataset_id="sample",
                frame_count=1,
                image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
            )
            with TrajectoryStore.create(path, metadata=metadata) as store:
                store.write_raw_frame(1, self.np.zeros((1, 3, 4, 1), dtype=self.np.uint16))

            def fake_segmenter(image, _file_job, _frame):
                labels = self.np.zeros(image.shape, dtype=self.np.uint16)
                labels[..., 1:3] = 5
                return labels

            summary = run_batch_segmentation(
                {
                    "job_id": "seg_mask",
                    "files": [
                        {
                            "h5_path": str(path),
                            "output_name": "cyto_mask",
                            "output_kind": "masks",
                            "overwrite": True,
                            "frames": {"mode": "all"},
                            "backend": {"backend_id": "fake", "parameters": {"do_3D": True}},
                            "model_input": {
                                "channel_specs": [
                                    {"channel_indices": [0], "normalization": "raw", "combination": "single"}
                                ]
                            },
                        }
                    ],
                },
                fake_segmenter,
            )

            self.assertEqual(summary.completed, 1)
            with TrajectoryStore.open(path, mode="r") as store:
                self.assertEqual(store.list_mask_frames("cyto_mask"), [1])
                self.assertEqual(store.read_mask_frame("cyto_mask", 1).dtype, self.np.bool_)
                frame_record = store.read_segmentation_frame_result("seg_mask", 1)
                self.assertEqual(frame_record["output_kind"], "masks")
                self.assertEqual(frame_record["output_h5_path"], "/masks/cyto_mask")

    def test_batch_segmentation_preview_npz_contains_input_and_labels(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            preview_path = Path(tmp) / "preview.npz"
            metadata = TrajectoryMetadata(
                roi_id="sample_XY001_ROI001",
                dataset_id="sample",
                frame_count=1,
                image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
            )
            with TrajectoryStore.create(path, metadata=metadata) as store:
                store.write_raw_frame(1, self.np.ones((1, 3, 4, 1), dtype=self.np.uint16))

            def fake_segmenter(image, _file_job, _frame):
                return self.np.full(image.shape, 7, dtype=self.np.uint16)

            run_batch_segmentation(
                {
                    "job_id": "seg_preview_arrays",
                    "save_outputs": False,
                    "preview_output_path": str(preview_path),
                    "files": [
                        {
                            "h5_path": str(path),
                            "output_name": "cyto",
                            "output_kind": "labels",
                            "frames": {"mode": "all"},
                            "backend": {"backend_id": "fake", "parameters": {"do_3D": True}},
                            "model_input": {
                                "channel_specs": [
                                    {"channel_indices": [0], "normalization": "raw", "combination": "single"}
                                ]
                            },
                        }
                    ],
                },
                fake_segmenter,
            )

            self.assertTrue(preview_path.exists())
            with self.np.load(preview_path) as preview:
                self.assertEqual(preview["model_input"].shape, (1, 3, 4))
                self.assertEqual(preview["labels"].max(), 7)
                self.assertEqual(str(preview["output_h5_path"].item()), "/labels/cyto")

    def test_batch_segmentation_preview_output_dir_writes_one_npz_per_frame(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            preview_dir = Path(tmp) / "previews"
            metadata = TrajectoryMetadata(
                roi_id="sample_XY001_ROI001",
                dataset_id="sample",
                frame_count=2,
                image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
            )
            with TrajectoryStore.create(path, metadata=metadata) as store:
                store.write_raw_frame(1, self.np.ones((1, 3, 4, 1), dtype=self.np.uint16))
                store.write_raw_frame(2, self.np.ones((1, 3, 4, 1), dtype=self.np.uint16) * 2)

            def fake_segmenter(image, _file_job, frame):
                return self.np.full(image.shape, frame, dtype=self.np.uint16)

            events = []
            run_batch_segmentation(
                {
                    "job_id": "seg_preview_dir",
                    "save_outputs": False,
                    "preview_output_dir": str(preview_dir),
                    "files": [
                        {
                            "h5_path": str(path),
                            "output_name": "cyto",
                            "output_kind": "labels",
                            "frames": {"mode": "all"},
                            "backend": {"backend_id": "fake", "parameters": {"do_3D": True}},
                            "model_input": {
                                "channel_specs": [
                                    {"channel_indices": [0], "normalization": "raw", "combination": "single"}
                                ]
                            },
                        }
                    ],
                },
                fake_segmenter,
                reporter=lambda event: events.append(dict(event)),
            )

            outputs = sorted(preview_dir.glob("*.npz"))
            self.assertEqual(len(outputs), 2)
            frame_events = [event for event in events if event.get("event") == "frame_completed"]
            self.assertEqual(len(frame_events), 2)
            self.assertTrue(all(Path(str(event["preview_output_path"])).exists() for event in frame_events))


if __name__ == "__main__":
    unittest.main()
