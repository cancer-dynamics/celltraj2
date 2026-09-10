from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from celltraj2.deferred_commit import (
    PLAN_SCHEMA,
    CommitOutcome,
    apply_deferred_job,
    defer_commit_plan,
    execute_commit_plan,
    read_bundle,
    write_bundle,
)
from celltraj2.h5_access import H5AccessTimeout


class _FakeStore:
    def __init__(self) -> None:
        self.revisions = {"/input": 4}
        self.json_writes: list[tuple[str, object, bool]] = []

    def resource_revision(self, path: str) -> int:
        return int(self.revisions.get(path, -1))

    def write_json(self, path, data, *, overwrite=False):
        self.json_writes.append((str(path), data, bool(overwrite)))


class _FakeTrajectory:
    store = _FakeStore()

    def __init__(self, *_args, **_kwargs) -> None:
        self.store = type(self).store

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class DeferredCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        self.np = np

    def _plan(self, h5_path: Path) -> dict:
        return {
            "schema": PLAN_SCHEMA,
            "job_id": "source_job",
            "h5_path": str(h5_path),
            "operation": "test_commit",
            "dependencies": {"/input": 4},
            "operations": [
                {
                    "op": "write_json",
                    "args": {
                        "path": "/runs/test/run.json",
                        "data": {"array": self.np.arange(4, dtype=self.np.int16)},
                        "overwrite": True,
                    },
                }
            ],
        }

    def test_bundle_roundtrip_uses_non_pickle_arrays(self):
        with TemporaryDirectory() as tmp:
            bundle, digest = write_bundle(Path(tmp) / "result.npz", self._plan(Path(tmp) / "target.h5"))
            restored = read_bundle(bundle, expected_sha256=digest)

            self.assertEqual(restored["schema"], PLAN_SCHEMA)
            self.np.testing.assert_array_equal(
                restored["operations"][0]["args"]["data"]["array"],
                self.np.arange(4, dtype=self.np.int16),
            )
            with self.np.load(bundle, allow_pickle=False) as archive:
                self.assertIn("__manifest__", archive.files)

    def test_execute_plan_validates_and_runs_only_allowlisted_operations(self):
        _FakeTrajectory.store = _FakeStore()
        with TemporaryDirectory() as tmp, patch(
            "celltraj2.deferred_commit.Trajectory", _FakeTrajectory
        ):
            outcome = execute_commit_plan(self._plan(Path(tmp) / "target.h5"))

        self.assertEqual(outcome.status, "committed")
        self.assertEqual(len(_FakeTrajectory.store.json_writes), 1)
        self.assertEqual(_FakeTrajectory.store.json_writes[0][0], "/runs/test/run.json")

    def test_successful_apply_removes_bundle(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "workflow_jobs.jsonl"
            environment = {
                "CELLTRAJ2_DEFERRED_ROOT": str(root / "deferred"),
                "CELLTRAJ2_WORKFLOW_QUEUE_PATH": str(queue),
                "SITELAB_INSTANCE_ID": "instance-a",
                "SITELAB_WORKSPACE_ID": "workspace-a",
                "SITELAB_INSTANCE_LABEL": "test:1:a",
            }
            with patch.dict(os.environ, environment, clear=False):
                deferred = defer_commit_plan(self._plan(root / "target.h5"))
                record = json.loads(queue.read_text(encoding="utf-8").splitlines()[0])
                job = json.loads(Path(record["job_path"]).read_text(encoding="utf-8"))
                with patch(
                    "celltraj2.deferred_commit.execute_commit_plan",
                    return_value=CommitOutcome(status="committed", operation_results={}),
                ):
                    outcome = apply_deferred_job(job)

            self.assertEqual(outcome.status, "committed")
            self.assertFalse(Path(str(deferred.bundle_path)).exists())
            self.assertEqual(
                record["submitted_by"],
                {
                    "instance_id": "instance-a",
                    "workspace_id": "workspace-a",
                    "instance_label": "test:1:a",
                },
            )

    def test_timeout_requeues_same_durable_bundle(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "workflow_jobs.jsonl"
            environment = {
                "CELLTRAJ2_DEFERRED_ROOT": str(root / "deferred"),
                "CELLTRAJ2_WORKFLOW_QUEUE_PATH": str(queue),
            }
            with patch.dict(os.environ, environment, clear=False):
                first = defer_commit_plan(self._plan(root / "target.h5"))
                first_record = json.loads(queue.read_text(encoding="utf-8").splitlines()[0])
                first_job = json.loads(Path(first_record["job_path"]).read_text(encoding="utf-8"))
                with patch(
                    "celltraj2.deferred_commit.execute_commit_plan",
                    side_effect=H5AccessTimeout("still busy"),
                ):
                    outcome = apply_deferred_job(first_job)

            records = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(outcome.status, "deferred")
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["retry_number"], 2)
            self.assertEqual(records[1]["bundle_path"], first.bundle_path)
            self.assertTrue(Path(str(first.bundle_path)).exists())


if __name__ == "__main__":
    unittest.main()
