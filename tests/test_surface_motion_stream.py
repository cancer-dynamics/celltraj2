"""Parity, bounded IO and lifecycle checks for surface-motion disk results."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from celltraj2.array_stream import COPY_BYTES, H5Array, copy_array
from celltraj2.boundaries import BoundarySourceSpec
from celltraj2.deferred_commit import apply_deferred_job, read_bundle
from celltraj2.h5_access import H5AccessTimeout, H5DependencyChangedError
from celltraj2.motion_stream import MotionSpool
from celltraj2.reporting import JobCancelledError
from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore
from celltraj2.surface_motion_batch import run_batch_surface_motion
from celltraj2.tracking import compute_boundary_motion
from celltraj2.trajectory import Trajectory


def fixture(path, frames=4):
    metadata = TrajectoryMetadata(roi_id="roi", dataset_id="data", frame_count=frames,
        image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Y", "X", "C")))
    with TrajectoryStore.create(path, metadata=metadata) as store:
        for frame in range(frames):
            labels = np.zeros((32, 32), np.uint16)
            labels[2:7, 2 + frame:7 + frame] = 1
            # Split the second object after the first frame to exercise branches.
            labels[20:25, 15:20] = 2
            if frame:
                labels[20:25, 21:25] = 3
            store.write_label_frame("cells", frame + 1, labels)
    with Trajectory(path, mode="r+") as trajectory:
        trajectory.index_observations("cells")
        trajectory.build_boundary_library("surface", sources=[
            BoundarySourceSpec(kind="object_set", name="cells", object_set="cells")])
        trajectory.track_minimum_centroid_distance("cells", max_distance=10, track_set="nearest")


def config(root, **kwargs):
    return dict(project_root=str(root), job_id="stream_test", fail_fast=True, files=[dict(
        h5_path="sample.ct2.h5", object_set="cells", track_set="nearest", boundary_set="surface",
        motion_set="motion", max_boundary_points=32, ot_method="unbalanced", **kwargs)])


class SurfaceMotionStreamTests(unittest.TestCase):
    def assert_columns(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertEqual(left[key].dtype, right[key].dtype)
            np.testing.assert_array_equal(left[key][...], right[key][...])

    def test_exact_numerical_parity_diagnostics_offsets_and_empty_graph(self):
        for frames in (1, 4):
            with self.subTest(frames=frames), TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root / "sample.ct2.h5", frames=frames)
                with Trajectory(root / "sample.ct2.h5") as trajectory:
                    for method in ("emd", "sinkhorn", "unbalanced", "partial"):
                        args = dict(boundary_set="surface", max_boundary_points=32, ot_method=method,
                                    cost_prealign="cell_centroid", save_outputs=False)
                        expected = compute_boundary_motion(trajectory, "cells", "nearest", **args)
                        with MotionSpool(root / "scratch.h5", buffer_bytes=1) as spool:
                            actual = compute_boundary_motion(trajectory, "cells", "nearest", result_spool=spool, **args)
                            self.assert_columns(expected.transport, actual.transport)
                            for name in expected.point_summaries:
                                self.assert_columns(expected.point_summaries[name], actual.point_summaries[name])
                            self.assertEqual(expected.links.dtype, actual.links.dtype)
                            for name in expected.links.dtype.names:
                                np.testing.assert_array_equal(expected.links[name], actual.links[name])
                            self.assertEqual(expected.frame_counts, actual.frame_counts)
                            self.assertEqual({k:v for k,v in expected.schema.items() if k != "created_at"},
                                             {k:v for k,v in actual.schema.items() if k != "created_at"})
                            self.assertEqual(spool.pending_bytes, 0)
                            if frames > 1:
                                self.assertGreater(spool.flush_count, 1)

    def test_batch_writes_readonly_test_and_failure_cleanup(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.ct2.h5"
            fixture(path)
            events = []
            summary = run_batch_surface_motion(config(root), reporter=events.append)
            self.assertEqual(summary.completed, 1)
            links = [e for e in events if e['event'] == 'surface_motion_link_summary']
            self.assertEqual(len(links), summary.links)
            self.assertTrue(any(e['result_flush_count'] > 0 for e in links))
            self.assertTrue(all(e['boundary_cache_bytes'] <= 32 * 1024**2 for e in links))
            frame_events = [e for e in events if e['event'] == 'surface_motion_frame_summary']
            self.assertEqual(len(frame_events), 3)
            self.assertLess(events.index(frame_events[0]), events.index(links[-1]))
            with Trajectory(path) as trajectory:
                stored = trajectory.store.read_boundary_motion("surface", "motion")
                expected = compute_boundary_motion(trajectory, "cells", "nearest", boundary_set="surface",
                                                   max_boundary_points=32, save_outputs=False)
                self.assert_columns(stored['transport'], expected.transport)
                for name in expected.links.dtype.names:
                    np.testing.assert_array_equal(stored['links'][name], expected.links[name])
            before = path.read_bytes()
            run_batch_surface_motion(config(root, save_outputs=False))
            self.assertEqual(path.read_bytes(), before)
            def cancel(event):
                if event['event'] == 'surface_motion_link_summary':
                    raise JobCancelledError("cancel")
            with self.assertRaises(JobCancelledError):
                run_batch_surface_motion(config(root, overwrite=True), reporter=cancel)
            self.assertEqual(path.read_bytes(), before)
            with patch("celltraj2.boundaries.optimal_transport_plan", side_effect=RuntimeError("solver failed")):
                with self.assertRaisesRegex(RuntimeError, "solver failed"):
                    run_batch_surface_motion(config(root, overwrite=True))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list((root / "cache/surface_motion").iterdir()), [])

    def test_deferred_bundle_remains_lazy_on_retry_and_preserves_values(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.ct2.h5"
            fixture(path)
            events = []
            def busy(*args, **kwargs):
                if kwargs.get("mode") == "r+":
                    raise H5AccessTimeout("busy")
                return Trajectory(*args, **kwargs)
            with patch("celltraj2.surface_motion_batch.Trajectory", side_effect=busy):
                summary = run_batch_surface_motion(config(root), reporter=events.append)
            self.assertEqual(summary.deferred, 1)
            event = next(e for e in events if e['event'] == 'deferred_commit_created')
            job = json.loads(Path(event['deferred_job_path']).read_text())
            bundle = Path(job['bundle_path'])
            self.assertTrue(h5py.is_hdf5(bundle))
            plan = read_bundle(bundle, expected_sha256=job['bundle_sha256'])
            args = plan['operations'][0]['args']
            self.assertIsInstance(args['transport']['mass'], H5Array)
            self.assertEqual(list((root / "cache/surface_motion").iterdir()), [])
            with patch("celltraj2.deferred_commit.Trajectory", side_effect=H5AccessTimeout("still busy")):
                self.assertTrue(apply_deferred_job(job).deferred)
            self.assertTrue(bundle.exists())
            self.assertEqual(apply_deferred_job(job).status, 'committed')
            self.assertFalse(bundle.exists())
            with Trajectory(path) as trajectory:
                stored = trajectory.store.read_boundary_motion("surface", "motion")
                expected = compute_boundary_motion(trajectory, "cells", "nearest", boundary_set="surface",
                                                   max_boundary_points=32, save_outputs=False)
                self.assert_columns(stored['transport'], expected.transport)
                for name in expected.links.dtype.names:
                    np.testing.assert_array_equal(stored['links'][name], expected.links[name])

    def test_stale_commit_recalculates_with_fresh_scratch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "sample.ct2.h5")
            events = []
            with patch("celltraj2.surface_motion_batch.validate_revisions",
                       side_effect=[H5DependencyChangedError("changed"), None]):
                self.assertEqual(run_batch_surface_motion(config(root), reporter=events.append).completed, 1)
            attempts = [e['scratch_directory'] for e in events if e['event'] == 'surface_motion_storage']
            self.assertEqual(len(set(attempts)), 2)
            self.assertTrue(all(not Path(p).exists() for p in attempts))

    def test_deferred_stale_inputs_rejected_and_bundle_kept(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'sample.ct2.h5'
            fixture(path)
            events = []
            def busy(*args, **kwargs):
                if kwargs.get('mode') == 'r+':
                    raise H5AccessTimeout('busy')
                return Trajectory(*args, **kwargs)
            with patch('celltraj2.surface_motion_batch.Trajectory', side_effect=busy):
                run_batch_surface_motion(config(root), reporter=events.append)
            event = next(e for e in events if e['event'] == 'deferred_commit_created')
            job = json.loads(Path(event['deferred_job_path']).read_text())
            with Trajectory(path, mode='r+') as trajectory:
                trajectory.store._mark_mutation('/object_sets/cells/tracks/nearest')
            with self.assertRaises(H5DependencyChangedError):
                apply_deferred_job(job)
            self.assertTrue(Path(job['bundle_path']).exists())
            with Trajectory(path) as trajectory:
                self.assertNotIn('motion', trajectory.boundary_library('surface').motion_sets())

    def test_spool_buffer_budget_and_frame_flush_preserve_order(self):
        with TemporaryDirectory() as tmp:
            with MotionSpool(Path(tmp) / 'scratch.h5', buffer_bytes=4096) as spool:
                for index in range(100):
                    values = np.full(128, index, dtype='f8')
                    spool.append(index // 7, {'mass': values}, {}, {})
                    self.assertLessEqual(spool.pending_bytes, 4096)
                transport, _ = spool.finish({})
                self.assertLessEqual(spool.peak_buffer_bytes, 4096)
                self.assertGreater(spool.flush_count, 10)
                np.testing.assert_array_equal(transport['mass'][::128], np.arange(100))
                self.assertEqual(spool.pending_bytes, 0)

    def test_copy_never_materializes_full_array_and_failed_overwrite_keeps_old(self):
        class LargeArray:
            shape, dtype = (3 * COPY_BYTES // 8,), np.dtype('f8')
            def __array__(self, *args, **kwargs):
                raise AssertionError("unbounded read")
            def __getitem__(self, selection):
                self_test.assertIsInstance(selection, slice)
                count = min(self.shape[0], selection.stop) - selection.start
                self_test.assertLessEqual(count * 8, COPY_BYTES)
                return np.ones(count)
        self_test = self
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with h5py.File(root / 'copy.h5', 'w') as h5:
                result = copy_array(h5, 'array', LargeArray())
                self.assertEqual(result.shape, LargeArray.shape)
                self.assertEqual(result[-1], 1)
            path = root / "sample.ct2.h5"
            fixture(path)
            run_batch_surface_motion(config(root))
            with Trajectory(path, mode='r+') as trajectory:
                saved = trajectory.store.read_boundary_motion("surface", "motion")
                with patch("celltraj2.array_stream.copy_array", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        trajectory.store.write_boundary_motion('surface', 'motion', **saved, overwrite=True)
                restored = trajectory.store.read_boundary_motion('surface', 'motion')
                self.assert_columns(saved['transport'], restored['transport'])
                self.assertFalse(any(key.startswith('_pending_motion') for key in trajectory.store.h5))


if __name__ == "__main__":
    unittest.main()
