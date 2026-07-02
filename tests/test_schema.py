from pathlib import Path
import unittest

from celltraj2.schema import (
    ChannelSpec,
    ImageSourceSpec,
    RoiBounds,
    RoiSpec,
    TrajectoryMetadata,
    channels_from_site,
)


class SchemaTests(unittest.TestCase):
    def test_roi_bounds_shape(self):
        bounds = RoiBounds(z_start=1, z_stop=4, y_start=10, y_stop=20, x_start=30, x_stop=45)

        self.assertEqual(bounds.shape_zyx, (3, 10, 15))

    def test_metadata_roundtrip_keeps_paths_and_frame_map(self):
        roi = RoiSpec(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            time_start=5,
            time_stop=8,
            artifact_path=Path("roi_files/sample/sample_XY001_ROI001.ome.zarr"),
        )
        source = ImageSourceSpec(source_type="roi_ome_zarr", path=roi.artifact_path, axes=("T", "C", "Z", "Y", "X"), roi=roi)
        metadata = TrajectoryMetadata(
            roi_id=roi.roi_id,
            dataset_id="sample",
            frame_count=3,
            channels=[ChannelSpec(raw_index=0, display_name="DAPI")],
            roi=roi,
            image_source=source,
        )

        payload = metadata.to_dict()
        loaded = TrajectoryMetadata.from_dict(payload)

        self.assertEqual(payload["frame_map"], [
            {"frame": 1, "parent_time_index": 5},
            {"frame": 2, "parent_time_index": 6},
            {"frame": 3, "parent_time_index": 7},
        ])
        self.assertEqual(loaded.image_source.path, Path("roi_files/sample/sample_XY001_ROI001.ome.zarr"))
        self.assertEqual(loaded.channels[0].display_name, "DAPI")

    def test_channels_from_site(self):
        channels = channels_from_site(
            [
                {
                    "raw_index": 5,
                    "raw_name": "pYFP",
                    "display_name": "ERK",
                    "role": "signaling_reporter",
                    "target": "ERK",
                }
            ]
        )

        self.assertEqual(channels[0].raw_index, 5)
        self.assertEqual(channels[0].target, "ERK")
        self.assertEqual(channels[0].metadata["raw_name"], "pYFP")


if __name__ == "__main__":
    unittest.main()
