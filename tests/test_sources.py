import tempfile
import unittest
from pathlib import Path

from celltraj2.sources import (
    InMemoryImageSource,
    _axes_for_array,
    _ome_zarr_channel_index_map,
    _ome_zarr_layout_hint,
    _open_zarr_array_for_read,
    _open_zarr_group_for_read,
)


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

    def test_in_memory_2d_multichannel_frame_axes_are_yxc(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        data = np.zeros((2, 2, 3, 4), dtype=np.uint16)
        source = InMemoryImageSource(data, axes=("T", "C", "Y", "X"))

        frame = source.read_frame(frame=1)

        self.assertEqual(frame.shape, (3, 4, 2))
        self.assertEqual(source.frame_axes(frame.ndim), ("Y", "X", "C"))

    def test_zarr_group_reader_tries_explicit_format_arguments(self):
        class FakeZarr:
            def __init__(self):
                self.calls = []

            def open_group(self, path, mode="r", **kwargs):
                self.calls.append(dict(kwargs))
                if kwargs.get("zarr_format") == 2:
                    return "group"
                raise RuntimeError("not this attempt")

        fake = FakeZarr()

        result = _open_zarr_group_for_read(fake, Path("roi.ome.zarr"))

        self.assertEqual(result, "group")
        self.assertIn({"zarr_format": 2}, fake.calls)

    def test_zarr_array_reader_can_fallback_to_root_array(self):
        class FakeZarr:
            def __init__(self):
                self.calls = []

            def open_array(self, path, mode="r", **kwargs):
                self.calls.append(dict(kwargs))
                if kwargs.get("zarr_version") == 2:
                    return "array"
                raise RuntimeError("not this attempt")

        fake = FakeZarr()

        result = _open_zarr_array_for_read(fake, Path("roi.ome.zarr"))

        self.assertEqual(result, "array")
        self.assertIn({"zarr_version": 2}, fake.calls)

    def test_zarr_layout_hint_identifies_v3_store_for_v2_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roi.ome.zarr"
            path.mkdir()
            (path / "zarr.json").write_text("{}", encoding="utf-8")

            hint = _ome_zarr_layout_hint(path)

        self.assertIsNotNone(hint)
        self.assertIn("Zarr v3", hint or "")

    def test_ome_zarr_channel_index_map_uses_site_channel_indices(self):
        class Root:
            attrs = {"site": {"channel_indices": [2, 5]}}

        mapping = _ome_zarr_channel_index_map(Root(), axes=("T", "C", "Z", "Y", "X"), shape=(1, 2, 3, 4, 5))

        self.assertEqual(mapping, {2: 0, 5: 1})

    def test_ome_zarr_channel_index_map_rejects_mismatched_channel_count(self):
        class Root:
            attrs = {"site": {"channel_indices": [2]}}

        with self.assertRaisesRegex(ValueError, "does not match C axis size"):
            _ome_zarr_channel_index_map(Root(), axes=("T", "C", "Z", "Y", "X"), shape=(1, 2, 3, 4, 5))

    def test_axes_for_array_repairs_stale_2d_site_ome_zarr_spec(self):
        axes = _axes_for_array(4, ("T", "C", "Z", "Y", "X"))

        self.assertEqual(axes, ("T", "C", "Y", "X"))

    def test_axes_for_array_prefers_actual_ome_zarr_attrs_over_stale_h5_spec(self):
        axes = _axes_for_array(4, ("T", "C", "Y", "X"), ("T", "C", "Z", "Y", "X"))

        self.assertEqual(axes, ("T", "C", "Y", "X"))


if __name__ == "__main__":
    unittest.main()
