from types import SimpleNamespace
import unittest
from unittest.mock import patch

from celltraj2.runners import index_objects


class ObjectRunnerTests(unittest.TestCase):
    @patch.object(index_objects, "load_object_index_job", return_value=object())
    @patch.object(index_objects, "run_batch_object_indexing")
    def test_runner_returns_nonzero_when_any_file_failed(self, run_batch, _load_job):
        run_batch.return_value = SimpleNamespace(failed=1)

        self.assertEqual(index_objects.main(["object_job.json"]), 1)

    @patch.object(index_objects, "load_object_index_job", return_value=object())
    @patch.object(index_objects, "run_batch_object_indexing")
    def test_runner_returns_zero_when_batch_completed_without_failures(self, run_batch, _load_job):
        run_batch.return_value = SimpleNamespace(failed=0)

        self.assertEqual(index_objects.main(["object_job.json"]), 0)


if __name__ == "__main__":
    unittest.main()
