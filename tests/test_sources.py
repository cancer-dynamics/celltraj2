import unittest

from celltraj2.sources import InMemoryImageSource


class SourceTests(unittest.TestCase):
    def test_in_memory_source_reads_one_based_frames(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        data = np.arange(2 * 3 * 4 * 5 * 2).reshape(2, 3, 4, 5, 2)
        source = InMemoryImageSource(data, axes=("T", "Z", "Y", "X", "C"))

        frame = source.read_frame(frame=2, channels=[1], z=slice(1, 3), y=slice(0, 2), x=slice(2, 5))

        self.assertEqual(frame.shape, (2, 2, 3, 1))
        self.assertEqual(frame[0, 0, 0, 0], data[1, 1, 0, 2, 1])

    def test_static_snapshot_is_frame_one(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        data = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        source = InMemoryImageSource(data, axes=("Z", "Y", "X"))

        frame = source.read_frame(frame=1)

        self.assertEqual(frame.shape, (3, 4, 5))
        self.assertEqual(frame.tolist(), data.tolist())


if __name__ == "__main__":
    unittest.main()
