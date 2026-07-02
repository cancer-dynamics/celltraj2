from pathlib import Path
import unittest

from celltraj2.sitelab import (
    create_metadata_from_site_roi,
    dataset_id_from_roi_json,
    default_cell_file_path,
    frame_count_from_roi,
    image_source_from_site_roi,
)


class SitelabHandoffTests(unittest.TestCase):
    def test_default_cell_file_path(self):
        self.assertEqual(
            default_cell_file_path(Path("/project"), "sample", "sample_XY001_ROI001"),
            Path("/project/cell_files/sample/sample_XY001_ROI001.ct2.h5"),
        )

    def test_dataset_id_from_roi_json(self):
        self.assertEqual(dataset_id_from_roi_json(Path("/project/rois/sample.rois.json")), "sample")
        self.assertEqual(dataset_id_from_roi_json(Path("/project/rois/sample.rois.json"), {"dataset_id": "sample_2"}), "sample_2")

    def test_frame_count_from_roi_defaults_snapshot_to_one(self):
        self.assertEqual(frame_count_from_roi({}, {"source_sizes": {}}), 1)
        self.assertEqual(frame_count_from_roi({"time_start": 2, "time_stop": 5}, {"source_sizes": {"T": 10}}), 3)
        self.assertEqual(frame_count_from_roi({"time_start": 0}, {"source_sizes": {"T": 7}}), 7)

    def test_image_source_from_roi_ome_zarr(self):
        roi_set = {
            "dataset_id": "sample",
            "source_path": "raw/sample.nd2",
            "source_axes": ["T", "P", "Z", "C", "Y", "X"],
            "source_sizes": {"T": 3, "P": 1, "Z": 2, "C": 2, "Y": 4, "X": 5},
        }
        roi = {
            "roi_id": "sample_XY001_ROI001",
            "position_index": 0,
            "bounds": {"z_start": 0, "z_stop": 2, "y_start": 0, "y_stop": 4, "x_start": 0, "x_stop": 5},
            "storage_mode": "roi_ome_zarr",
            "artifact_path": "roi_files/sample/sample_XY001_ROI001.ome.zarr",
        }

        spec = image_source_from_site_roi(roi_set=roi_set, roi_record=roi, project_root=Path("/project"))

        self.assertEqual(spec.source_type, "roi_ome_zarr")
        self.assertEqual(spec.path, Path("/project/roi_files/sample/sample_XY001_ROI001.ome.zarr"))
        self.assertEqual(spec.axes, ("T", "C", "Z", "Y", "X"))

    def test_create_metadata_from_site_roi(self):
        roi_set = {
            "dataset_id": "sample",
            "source_path": "raw/sample.nd2",
            "source_axes": ["T", "P", "Z", "C", "Y", "X"],
            "source_sizes": {"T": 3, "P": 1, "Z": 2, "C": 1, "Y": 4, "X": 5},
            "rois": [
                {
                    "roi_id": "sample_XY001_ROI001",
                    "position_index": 0,
                    "time_start": 1,
                    "time_stop": 3,
                    "bounds": {"z_start": 0, "z_stop": 2, "y_start": 0, "y_stop": 4, "x_start": 0, "x_stop": 5},
                    "storage_mode": "linked_nd2",
                }
            ],
        }
        manifest = {
            "images": [
                {
                    "channels": [{"raw_index": 0, "display_name": "DAPI"}],
                    "acquisition": {"zscale": 4.0},
                    "treatments": [{"position_index": 0, "treatment": "EGF"}],
                }
            ]
        }

        metadata, _roi_set, _roi, root, dataset_id = create_metadata_from_site_roi(
            roi_json_path=Path("/project/rois/sample.rois.json"),
            roi_id="sample_XY001_ROI001",
            roi_set=roi_set,
            manifest=manifest,
        )

        self.assertEqual(root, Path("/project"))
        self.assertEqual(dataset_id, "sample")
        self.assertEqual(metadata.frame_count, 2)
        self.assertEqual(metadata.channels[0].display_name, "DAPI")
        self.assertEqual(metadata.frame_map()[0], {"frame": 1, "parent_time_index": 1})


if __name__ == "__main__":
    unittest.main()
