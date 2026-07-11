from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore
from celltraj2.tracking import track_minimum_centroid_distance
from celltraj2.tracking_batch import TrackingBatchJob, TrackingFileJob, run_batch_tracking
from celltraj2.trajectory import Trajectory


class SparseTrackingTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        self.np = np

    def _create_indexed_h5(self, path: Path) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py is not installed")
        metadata = TrajectoryMetadata(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            frame_count=3,
            image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Y", "X", "C")),
        )
        with TrajectoryStore.create(path, metadata=metadata) as store:
            frames = []
            frame_1 = self.np.zeros((10, 10), dtype=self.np.uint16)
            frame_1[1, 1] = 1
            frame_1[8, 8] = 2
            frames.append(frame_1)
            frame_2 = self.np.zeros((10, 10), dtype=self.np.uint16)
            frame_2[1, 0] = 1
            frame_2[1, 2] = 2
            frame_2[8, 8] = 3
            frame_2[5, 5] = 4
            frames.append(frame_2)
            frame_3 = self.np.zeros((10, 10), dtype=self.np.uint16)
            frame_3[1, 3] = 1
            frame_3[5, 6] = 2
            frames.append(frame_3)
            for frame, labels in enumerate(frames, start=1):
                store.write_label_frame("cells", frame, labels)
        with Trajectory(path) as trajectory:
            trajectory.index_observations("cells", run_id="index_cells")

    def test_centroid_tracker_preserves_unique_parent_and_forward_branching(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed_h5(path)

            with Trajectory(path) as trajectory:
                result = trajectory.track_minimum_centroid_distance(
                    "cells", max_distance=2.0, track_set="nearest"
                )
                graph = result.graph

                self.assertEqual(result.link_count, 5)
                self.assertEqual(graph.parent(3), 1)
                self.assertEqual(graph.parent(4), 1)
                self.assertIsNone(graph.parent(6))
                self.assertEqual(graph.children(1).tolist(), [3, 4])
                self.assertEqual(graph.history(7).tolist(), [1, 4, 7])
                self.assertEqual(graph.descendants(1).tolist(), [3, 4, 7])
                self.assertEqual(graph.selection_tree(4).tolist(), [1, 4, 7])
                self.assertEqual(graph.lineage(4).tolist(), [1, 3, 4, 7])
                self.assertEqual(
                    [trajectory.tolist() for trajectory in graph.maximal_trajectories()],
                    [[1, 3], [2, 5], [1, 4, 7], [6, 8]],
                )
                self.assertEqual(graph.assignments["tracklet_id"].tolist(), [1, 2, 3, 4, 2, 5, 4, 5])

    def test_sparse_graph_round_trips_through_h5(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed_h5(path)

            with Trajectory(path) as trajectory:
                trajectory.object_set("cells").track_minimum_centroid_distance(
                    max_distance=2.0, track_set="nearest"
                )
                self.assertEqual(trajectory.track_sets("cells"), ["nearest"])
                loaded = trajectory.read_tracks("cells", "nearest")
                self.assertEqual(loaded.adjacency.shape, (8, 8))
                self.assertEqual(loaded.adjacency.indptr.tolist(), [0, 2, 3, 3, 4, 4, 5, 5, 5])
                self.assertEqual(loaded.children(1).tolist(), [3, 4])
                self.assertEqual(loaded.schema["method"], "minimum_centroid_distance")
                with self.assertRaises(FileExistsError):
                    trajectory.track_minimum_centroid_distance(
                        "cells", max_distance=2.0, track_set="nearest"
                    )

    def test_coordinate_scale_changes_distance_cutoff(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed_h5(path)

            with Trajectory(path) as trajectory:
                result = trajectory.track_minimum_centroid_distance(
                    "cells",
                    max_distance=2.0,
                    coordinate_scale=(1.0, 3.0, 3.0),
                    save_outputs=False,
                )
                self.assertEqual(result.link_count, 1)
                self.assertFalse(result.saved)

    def test_tracker_graph_queries_without_h5_or_scipy(self):
        observations = self.np.zeros(
            5,
            dtype=[
                ("observation_id", "<i8"),
                ("frame", "<i4"),
                ("centroid_z", "<f8"),
                ("centroid_y", "<f8"),
                ("centroid_x", "<f8"),
            ],
        )
        observations["observation_id"] = [1, 2, 3, 4, 5]
        observations["frame"] = [1, 1, 2, 2, 3]
        observations["centroid_y"] = [0, 10, -1, 1, 2]

        class FakeStore:
            def read_observations(self, object_set):
                return observations

        class FakeTrajectory:
            store = FakeStore()

        result = track_minimum_centroid_distance(
            FakeTrajectory(), "cells", max_distance=2.0, save_outputs=False
        )
        graph = result.graph
        self.assertEqual(graph.children(1).tolist(), [3, 4])
        self.assertEqual(graph.parent(5), 4)
        self.assertEqual(graph.history(5).tolist(), [1, 4, 5])
        self.assertEqual(graph.lineage(3).tolist(), [1, 3, 4, 5])

    def test_tracking_job_parses_site_payload(self):
        job = TrackingBatchJob.from_dict(
            {
                "job_id": "track_test",
                "project_root": "project",
                "save_outputs": False,
                "files": [
                    {
                        "h5_path": "cell_files/sample.ct2.h5",
                        "object_set": "cells",
                        "track_set": "nearest",
                        "method": "mindist",
                        "distcut": 7.5,
                        "coordinate_scale": [2, 1, 1],
                    }
                ],
            }
        )
        self.assertFalse(job.save_outputs)
        self.assertEqual(job.files[0].method, "minimum_centroid_distance")
        self.assertEqual(job.files[0].max_distance, 7.5)
        self.assertEqual(job.files[0].coordinate_scale, (2.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            TrackingFileJob.from_dict(
                {"h5_path": "sample.h5", "object_set": "cells", "method": "optimal_transport"}
            )

    def test_batch_tracking_dry_run_and_saved_provenance(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed_h5(path)
            events = []
            dry = run_batch_tracking(
                {
                    "job_id": "track_preview",
                    "save_outputs": False,
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "cells",
                            "track_set": "nearest",
                            "max_distance": 2.0,
                        }
                    ],
                },
                reporter=lambda event: events.append(dict(event)),
            )
            self.assertEqual(dry.completed, 1)
            self.assertEqual(dry.links, 5)
            self.assertIn("tracking_frame_summary", [event.get("event") for event in events])
            with Trajectory(path) as trajectory:
                self.assertEqual(trajectory.track_sets("cells"), [])

            saved = run_batch_tracking(
                {
                    "job_id": "track_saved",
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "cells",
                            "track_set": "nearest",
                            "max_distance": 2.0,
                        }
                    ],
                }
            )
            self.assertEqual(saved.completed, 1)
            with Trajectory(path) as trajectory:
                self.assertEqual(trajectory.track_sets("cells"), ["nearest"])
                self.assertEqual(trajectory.tracking_runs(), ["track_saved"])
                run = trajectory.store.read_tracking_run("track_saved")
                self.assertEqual(run["link_count"], 5)
                frame = trajectory.store.read_tracking_frame_result("track_saved", 2)
                self.assertEqual(frame["linked_count"], 3)


if __name__ == "__main__":
    unittest.main()
