from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from celltraj2.interpretation import compile_gate_memberships, observation_spine_digest
from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.source_dependencies import (
    capture_source_dependencies, check_source_dependencies, merge_source_dependencies,
)
from celltraj2.store import TrajectoryStore


class SourceDependencyTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import h5py  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        self.np = np
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "source.ct2.h5"
        self.store = TrajectoryStore.create(self.path, metadata=TrajectoryMetadata(
            "roi", "dataset", image_source=ImageSourceSpec(source_type="embedded_h5", axes=("T", "Y", "X"))))
        self.addCleanup(self.store.close)
        self.observations = np.asarray([(1, 1, 1, 2.0), (2, 1, 2, 3.0)], dtype=[
            ("observation_id", "<i8"), ("frame", "<i4"), ("label_id", "<i8"), ("centroid_x", "<f8")])
        self.store.write_observations("cells", self.observations, {"schema": "test"})
        self.store.write_label_frame("cells", 1, np.asarray([[1, 2]]))
        self.store.write_feature_set("cells", "markers", np.asarray([(1, 1.), (2, 2.)],
            dtype=[("observation_id", "<i8"), ("marker", "<f8")]), {"columns": [{"name": "marker", "units": "intensity"}]})
        self.store.write_json("/object_sets/cells/tracks/accepted/schema.json", {
            "registration_dependency": {"registration_set": "motion"}})
        self.store.h5["object_sets/cells/tracks/accepted"].create_dataset("assignments", data=np.asarray([1, 1]))
        self.store.write_json("/registrations/motion/schema.json", {"units": "um"})
        self.store.h5["registrations/motion"].create_dataset("transforms", data=np.eye(3))

    def capture(self):
        return capture_source_dependencies(self.store.h5, object_set="cells", feature_sets=["markers"], track_set="accepted")

    def test_selected_measurements_geometry_tracks_and_registration_are_hashed(self):
        for path, mutate in (
            ("/object_sets/cells/observations", lambda a: a.__setitem__("centroid_x", [7, 8])),
            ("/object_sets/cells/features/markers/values", lambda a: a.__setitem__("marker", [7, 8])),
            ("/object_sets/cells/tracks/accepted/assignments", lambda a: a.__setitem__(0, 2)),
            ("/registrations/motion/transforms", lambda a: a.__setitem__((0, 2), 9)),
        ):
            with self.subTest(path=path):
                before = self.capture()
                array = self.store.h5[path][()]
                mutate(array)
                self.store.h5[path][...] = array  # Also detects direct table writes without revision bumps.
                report = check_source_dependencies(self.store.h5, before)
                self.assertEqual(report["status"], "stale")
                self.assertTrue(any(path.startswith(change["path"]) for change in report["changes"]))
        self.assertEqual(observation_spine_digest(self.store.read_observations("cells"), object_set="cells"),
            observation_spine_digest(self.observations, object_set="cells"))

    def test_schemas_channel_meaning_and_segmentation_versions_are_dependencies(self):
        before = self.capture()
        self.store.write_json("/object_sets/cells/features/markers/schema.json", {"units": "normalized"}, overwrite=True)
        self.assertEqual(check_source_dependencies(self.store.h5, before)["status"], "stale")
        before = self.capture()
        metadata = self.store.read_json("/metadata/celltraj2.json")
        metadata["channels"] = [{"raw_index": 0, "target": "different marker"}]
        self.store.write_json("/metadata/celltraj2.json", metadata, overwrite=True)
        self.assertEqual(check_source_dependencies(self.store.h5, before)["status"], "stale")
        before = self.capture()
        self.store.write_label_frame("cells", 1, self.np.asarray([[2, 1]]), overwrite=True)
        self.assertEqual(check_source_dependencies(self.store.h5, before)["changes"][0]["path"], "/labels/cells")

    def test_unrelated_outputs_viewer_state_and_source_relocation_do_not_invalidate(self):
        before = self.capture()
        self.store.write_json("/metadata/site_manifest.json", {"viewer_state": {"lut": [0, 9]}}, overwrite=True)
        self.store.write_feature_set("cells", "other", self.np.asarray([1., 2.]), {})
        self.store.write_json("/runs/interpretation/other/run.json", {"status": "completed"})
        metadata = self.store.read_json("/metadata/celltraj2.json")
        metadata["notes"] = "New note"
        metadata["image_source"]["path"] = "/relocated/image.h5"
        self.store.write_json("/metadata/celltraj2.json", metadata, overwrite=True)
        self.assertEqual(check_source_dependencies(self.store.h5, before)["status"], "current")

    def test_legacy_missing_and_mixed_versions_are_explicit(self):
        self.assertEqual(check_source_dependencies(self.store.h5, None)["status"], "unverified")
        before = self.capture()
        merged = merge_source_dependencies(before, None)
        self.assertEqual(check_source_dependencies(self.store.h5, merged)["status"], "unverified")
        del self.store.h5["object_sets/cells/features/markers"]
        after = self.capture()
        mixed = merge_source_dependencies(before, after)
        self.assertEqual(check_source_dependencies(self.store.h5, mixed)["status"], "stale")
        damaged = deepcopy(after)
        damaged["resources"] = []
        with self.assertRaisesRegex(ValueError, "digest"):
            check_source_dependencies(self.store.h5, damaged)

    def test_commit_plan_revalidates_sources_before_writing(self):
        from celltraj2.deferred_commit import PLAN_SCHEMA, execute_commit_plan, read_bundle, write_bundle
        dependencies = self.capture()
        raw = compile_gate_memberships(self.np.asarray([1, 2]), self.np.ones((2, 1)))
        artifact_id = str(uuid4())
        args = {"object_set": "cells", "classification_set": artifact_id,
            "values": raw.values, "probabilities": raw.probabilities,
            "schema": {"class_ids": [str(uuid4())]}, "source_manifest": {"artifact_id": artifact_id},
            "expected_source_dependencies": dependencies}
        plan = {"schema": PLAN_SCHEMA, "h5_path": str(self.path), "job_id": "source-check",
            "operation": "materialize_interpretation", "dependencies": {},
            "operations": [{"op": "write_classification_set", "args": args}]}
        bundle, digest = write_bundle(Path(self.tmp.name) / "commit.npz", plan)
        self.store.write_json("/object_sets/cells/features/markers/schema.json", {"units": "changed"}, overwrite=True)
        self.store.close()
        with self.assertRaisesRegex(ValueError, "Stale interpretation source"):
            execute_commit_plan(read_bundle(bundle, expected_sha256=digest))
        with TrajectoryStore.open(self.path, "r") as store:
            self.assertEqual(store.list_classification_sets("cells"), [])
