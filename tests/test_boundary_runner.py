from types import SimpleNamespace
import unittest
from unittest.mock import patch

from celltraj2.runners import build_boundaries


class BoundaryRunnerTests(unittest.TestCase):
    @patch.object(build_boundaries, "load_boundary_job", return_value=object())
    @patch.object(build_boundaries, "run_batch_boundaries")
    def test_runner_returns_nonzero_when_any_file_failed(self, run_batch, _load_job):
        run_batch.return_value = SimpleNamespace(failed=1)

        self.assertEqual(build_boundaries.main(["boundary_job.json"]), 1)

    @patch.object(build_boundaries, "load_boundary_job", return_value=object())
    @patch.object(build_boundaries, "run_batch_boundaries")
    def test_runner_returns_zero_when_batch_completed_without_failures(self, run_batch, _load_job):
        run_batch.return_value = SimpleNamespace(failed=0)

        self.assertEqual(build_boundaries.main(["boundary_job.json"]), 0)


if __name__ == "__main__":
    unittest.main()
