from multiprocessing import get_context
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from celltraj2.h5_access import H5AccessTimeout, file_lease, lock_path
from celltraj2.reporting import JsonlReporter


def _hold_lease(path: str, exclusive: bool, ready, duration: float) -> None:
    with file_lease(path, exclusive=exclusive, timeout=2.0):
        ready.set()
        time.sleep(duration)


class H5AccessTests(unittest.TestCase):
    def test_lock_sidecar_is_hidden_and_distinct(self):
        path = Path("sample.ct2.h5")
        self.assertEqual(lock_path(path).name, ".sample.ct2.h5.sitelab.lock")

    def test_writer_waits_for_reader_in_another_process(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sample.ct2.h5")
            context = get_context("spawn")
            ready = context.Event()
            process = context.Process(target=_hold_lease, args=(path, False, ready, 0.35))
            process.start()
            try:
                self.assertTrue(ready.wait(10.0))
                started = time.monotonic()
                with file_lease(path, exclusive=True, timeout=3.0):
                    waited = time.monotonic() - started
                self.assertGreaterEqual(waited, 0.15)
            finally:
                process.join(3.0)
                if process.is_alive():
                    process.terminate()
                    process.join(1.0)
            self.assertEqual(process.exitcode, 0)

    def test_readers_can_share_access_across_processes(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sample.ct2.h5")
            context = get_context("spawn")
            ready = context.Event()
            process = context.Process(target=_hold_lease, args=(path, False, ready, 0.5))
            process.start()
            try:
                self.assertTrue(ready.wait(10.0))
                started = time.monotonic()
                with file_lease(path, exclusive=False, timeout=1.0):
                    waited = time.monotonic() - started
                self.assertLess(waited, 0.25)
            finally:
                process.join(3.0)
                if process.is_alive():
                    process.terminate()
                    process.join(1.0)
            self.assertEqual(process.exitcode, 0)

    def test_wait_timeout_is_bounded(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sample.ct2.h5")
            context = get_context("spawn")
            ready = context.Event()
            process = context.Process(target=_hold_lease, args=(path, True, ready, 0.5))
            process.start()
            try:
                self.assertTrue(ready.wait(10.0))
                with self.assertRaises(H5AccessTimeout):
                    with file_lease(path, exclusive=False, timeout=0.05):
                        pass
            finally:
                process.join(3.0)
                if process.is_alive():
                    process.terminate()
                    process.join(1.0)

    def test_jsonl_reporter_flushes_stdout_and_durable_event_file(self):
        with TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            stream = StringIO()
            reporter = JsonlReporter(stream, events_path=events_path)

            reporter({"event": "frame_completed", "frame": 3})

            stdout_event = json.loads(stream.getvalue())
            stored_event = json.loads(events_path.read_text(encoding="utf-8"))
            self.assertEqual(stdout_event["event"], "frame_completed")
            self.assertEqual(stored_event["frame"], 3)
            self.assertIn("timestamp", stored_event)


if __name__ == "__main__":
    unittest.main()
