from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore
from celltraj2.trajectory import Trajectory


class StoreTests(unittest.TestCase):
    def setUp(self):
        try:
            import h5py  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("h5py/numpy are not installed")
        self.np = np

    def test_store_writes_frame_based_labels_and_masks(self):
        metadata = TrajectoryMetadata(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            frame_count=2,
            image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            with TrajectoryStore.create(path, metadata=metadata) as store:
                labels = self.np.ones((2, 3, 4), dtype=self.np.uint16)
                mask = labels > 0
                store.write_label_frame("epithelial", 1, labels)
                store.write_mask_frame("nuclear", 1, mask)

                self.assertEqual(store.list_label_frames("epithelial"), [1])
                self.assertTrue(store.has_label_frame("epithelial", 1))
                self.assertEqual(store.read_label_frame("epithelial", 1).shape, (2, 3, 4))
                self.assertEqual(store.read_mask_frame("nuclear", 1).dtype, self.np.bool_)

    def test_trajectory_facade_roundtrip(self):
        metadata = TrajectoryMetadata(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            frame_count=1,
            image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Z", "Y", "X", "C")),
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            with TrajectoryStore.create(path, metadata=metadata) as store:
                store.write_raw_frame(1, self.np.zeros((2, 3, 4, 1), dtype=self.np.uint16))
            with Trajectory(path) as traj:
                traj.write_label_frame("epithelial", 1, self.np.ones((2, 3, 4), dtype=self.np.uint16))
                self.assertEqual(traj.get_image_data(frame=1).shape, (2, 3, 4, 1))
                self.assertEqual(traj.label_frames("epithelial"), [1])


if __name__ == "__main__":
    unittest.main()
