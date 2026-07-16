from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from celltraj2.boundaries import BoundarySourceSpec
from celltraj2.boundary_features import boundary_multipole_magnitudes
from celltraj2.feature_extraction import run_batch_feature_extraction
from celltraj2.features import regionprops_v1_spec, site_signaling_v1_spec
from celltraj2.schema import ChannelSpec, ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore
from celltraj2.trajectory import Trajectory


class FeatureExtractionTests(unittest.TestCase):
    def setUp(self):
        try:
            import h5py  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("h5py/numpy are not installed")
        self.np = np

    def _create_feature_h5(self, path: Path) -> None:
        metadata = TrajectoryMetadata(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            frame_count=1,
            channels=[ChannelSpec(raw_index=0, display_name="ERK", readout="erk")],
            image_source=ImageSourceSpec(source_type="embedded_h5", axes=("Y", "X", "C")),
        )
        labels = self.np.array(
            [
                [1, 1, 0],
                [0, 2, 2],
                [0, 0, 2],
            ],
            dtype=self.np.uint16,
        )
        image = self.np.array(
            [
                [[4.0], [2.0], [1.0]],
                [[1.0], [10.0], [5.0]],
                [[1.0], [1.0], [5.0]],
            ],
            dtype=self.np.float32,
        )
        nuc_mask = self.np.array(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        nuc_labels = nuc_mask.astype(self.np.uint16)
        background_mask = labels == 0
        foreground_labels = (labels > 0).astype(self.np.uint16)
        with TrajectoryStore.create(path, metadata=metadata) as store:
            store.write_raw_frame(1, image)
            store.write_label_frame("cyto", 1, labels)
            store.write_label_frame("nuc_label", 1, nuc_labels)
            store.write_label_frame("foreground", 1, foreground_labels)
            store.write_mask_frame("nuc", 1, nuc_mask)
            store.write_mask_frame("background", 1, background_mask)
        with Trajectory(path) as trajectory:
            trajectory.index_observations("cyto", run_id="index_cyto")

    def _create_boundary_feature_h5(self, path: Path) -> None:
        metadata = TrajectoryMetadata(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            frame_count=2,
            image_source=ImageSourceSpec(
                source_type="embedded_h5",
                axes=("T", "Y", "X", "C"),
                sizes={"T": 2, "Y": 18, "X": 18, "C": 1},
            ),
            acquisition={"micron_per_pixel": 1.0, "voxel_size_um": {"Y": 1.0, "X": 1.0}},
        )
        frame_1 = self.np.zeros((18, 18), dtype=self.np.uint16)
        frame_1[4:10, 2:7] = 1
        frame_1[4:10, 8:13] = 2
        frame_2 = self.np.zeros_like(frame_1)
        frame_2[4:10, 3:8] = 1
        frame_2[4:10, 9:14] = 2
        with TrajectoryStore.create(path, metadata=metadata) as store:
            store.write_label_frame("cells", 1, frame_1)
            store.write_label_frame("cells", 2, frame_2)
        with Trajectory(path) as trajectory:
            trajectory.index_observations("cells", run_id="index_cells")
            trajectory.build_boundary_library(
                "cell_surfaces",
                sources=[
                    BoundarySourceSpec(
                        kind="object_set",
                        name="tracked_cells",
                        object_set="cells",
                        role="cell",
                    )
                ],
            )
            trajectory.compute_boundary_geometry(
                "cell_surfaces", geometry_set="geometry", backend="local", knn=6
            )
            trajectory.compute_boundary_neighbors(
                "cell_surfaces", neighbor_set="cell_contacts", k=1
            )
            trajectory.track_minimum_centroid_distance(
                "cells", max_distance=3.0, track_set="centroid"
            )
            trajectory.compute_boundary_motion(
                "cells",
                "centroid",
                boundary_set="cell_surfaces",
                boundary_source_name="tracked_cells",
                motion_set="surface_ot",
                ot_method="sinkhorn",
                sinkhorn_regularization=0.05,
            )

    def test_centered_boundary_multipoles_are_translation_and_rotation_invariant(self):
        theta = self.np.linspace(0.0, 2.0 * self.np.pi, 32, endpoint=False)
        points = self.np.column_stack(
            [self.np.zeros(theta.size), self.np.sin(theta), self.np.cos(theta)]
        )
        charge = 1.0 + 0.5 * self.np.cos(2.0 * theta)
        rotation = self.np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        first = boundary_multipole_magnitudes(points, charge, order=3, spatial_ndim=2)
        second = boundary_multipole_magnitudes(
            points @ rotation.T + self.np.asarray([4.0, -3.0, 7.0]),
            charge,
            order=3,
            spatial_ndim=2,
        )
        self.np.testing.assert_allclose(first, second, atol=1e-10)

    def test_boundary_object_features_combine_geometry_interaction_motion_and_multipoles(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "boundary_features.ct2.h5"
            self._create_boundary_feature_h5(path)
            with Trajectory(path) as trajectory:
                result = trajectory.extract_features(
                    {
                        "feature_set": "boundary_v1",
                        "object_set": "cells",
                        "source_label_set": "cells",
                        "features": [
                            {
                                "kind": "boundary_geometry",
                                "name": "curvature",
                                "boundary_set": "cell_surfaces",
                                "boundary_source_name": "tracked_cells",
                                "geometry_set": "geometry",
                                "fields": ["mean_curvature"],
                                "statistics": ["mean", "std"],
                            },
                            {
                                "kind": "boundary_interaction",
                                "name": "contact",
                                "boundary_set": "cell_surfaces",
                                "boundary_source_name": "tracked_cells",
                                "neighbor_set": "cell_contacts",
                                "contact_distance": 2.0,
                                "metrics": [
                                    "contact_fraction",
                                    "distance_mean",
                                    "neighbor_entity_count",
                                ],
                            },
                            {
                                "kind": "boundary_motion",
                                "name": "mapped",
                                "boundary_set": "cell_surfaces",
                                "boundary_source_name": "tracked_cells",
                                "motion_set": "surface_ot",
                                "geometry_set": "geometry",
                                "direction": "incoming",
                                "metrics": [
                                    "displacement_x_mean",
                                    "magnitude_mean",
                                    "normal_mean",
                                    "mapped_fraction",
                                    "ot_cost_mean",
                                ],
                            },
                            {
                                "kind": "boundary_multipole",
                                "name": "neighbor_pattern",
                                "boundary_set": "cell_surfaces",
                                "boundary_source_name": "tracked_cells",
                                "signal": "neighbor_distance",
                                "neighbor_set": "cell_contacts",
                                "distance_transform": "inverse_distance",
                                "order": 2,
                            },
                        ],
                    },
                    run_id="boundary_features",
                )
                values = trajectory.object_set("cells").read_features("boundary_v1")
                schema = trajectory.object_set("cells").read_feature_schema("boundary_v1")

            self.assertEqual(result.feature_count, 13)
            self.assertEqual(values.shape[0], 4)
            self.assertTrue(self.np.all(self.np.isfinite(values["curvature_mean"])))
            self.assertTrue(self.np.all(values["contact_contact_fraction"] > 0.0))
            self.assertTrue(self.np.all(values["contact_neighbor_entity_count"] == 1.0))
            self.assertTrue(self.np.all(self.np.isnan(values["mapped_magnitude_mean"][:2])))
            self.assertTrue(self.np.all(self.np.isfinite(values["mapped_magnitude_mean"][2:])))
            self.assertTrue(self.np.all(self.np.isfinite(values["neighbor_pattern_l2"])))
            motion_column = next(
                item for item in schema["columns"] if item["name"] == "mapped_magnitude_mean"
            )
            self.assertEqual(
                motion_column["registration_dependency"],
                motion_column["motion_dependency"]["schema"]["registration_dependency"],
            )
            self.assertEqual(
                motion_column["boundary_dependency"]["boundary_set"], "cell_surfaces"
            )

    def test_boundary_feature_batch_test_is_read_only_and_reports_frame_summaries(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "boundary_feature_preview.ct2.h5"
            self._create_boundary_feature_h5(path)
            events = []
            summary = run_batch_feature_extraction(
                {
                    "job_id": "boundary_feature_preview",
                    "save_outputs": False,
                    "files": [
                        {
                            "h5_path": str(path),
                            "feature_spec": {
                                "feature_set": "boundary_preview",
                                "object_set": "cells",
                                "source_label_set": "cells",
                                "features": [
                                    {
                                        "kind": "boundary_interaction",
                                        "name": "contact",
                                        "boundary_set": "cell_surfaces",
                                        "boundary_source_name": "tracked_cells",
                                        "neighbor_set": "cell_contacts",
                                        "contact_distance": 2.0,
                                        "metrics": ["contact_fraction", "distance_mean"],
                                    }
                                ],
                            },
                        }
                    ],
                },
                reporter=events.append,
            )
            self.assertEqual(summary.completed, 2)
            self.assertEqual(summary.features, 2)
            self.assertEqual(
                len([event for event in events if event.get("event") == "feature_frame_summary"]),
                4,
            )
            with Trajectory(path, mode="r") as trajectory:
                self.assertNotIn("boundary_preview", trajectory.object_set("cells").feature_sets())
                self.assertNotIn("boundary_feature_preview", trajectory.feature_extraction_runs())

    def test_regionprops_feature_set_is_named_and_row_aligned(self):
        try:
            import skimage  # noqa: F401
        except ImportError:
            self.skipTest("scikit-image is not installed")

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_feature_h5(path)

            with Trajectory(path) as trajectory:
                spec = regionprops_v1_spec("cyto", properties=["area"])
                result = trajectory.extract_features(spec, run_id="features_regionprops")
                values = trajectory.object_set("cyto").read_features("regionprops_v1")
                schema = trajectory.object_set("cyto").read_feature_schema("regionprops_v1")

                self.assertEqual(result.feature_count, 1)
                self.assertEqual(values["observation_id"].tolist(), [1, 2])
                self.assertEqual(values["regionprops_area"].tolist(), [2.0, 3.0])
                self.assertEqual(schema["row_alignment"], "/object_sets/cyto/observations")
                self.assertEqual(trajectory.object_set("cyto").feature_sets(), ["regionprops_v1"])
                self.assertEqual(trajectory.feature_extraction_runs(), ["features_regionprops"])

    def test_site_signaling_feature_set_defaults_to_cyto_over_nuc_ratio(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_feature_h5(path)

            with Trajectory(path) as trajectory:
                spec = site_signaling_v1_spec(
                    "cyto",
                    signal_channel={"readout": "erk"},
                    nuclear_mask_set="nuc",
                )
                result = trajectory.extract_features(spec, run_id="features_signaling")
                values = trajectory.object_set("cyto").read_features("site_v1")
                schema = trajectory.object_set("cyto").read_feature_schema("site_v1")

                ratio_column = "site_ratio"
                self.assertEqual(result.feature_count, 3)
                self.assertIn("site_cyto", values.dtype.names)
                self.assertIn("site_nuc", values.dtype.names)
                self.assertIn(ratio_column, values.dtype.names)
                self.np.testing.assert_allclose(values["site_cyto"], [4.0, 5.0])
                self.np.testing.assert_allclose(values["site_nuc"], [2.0, 10.0])
                self.np.testing.assert_allclose(values[ratio_column], [2.0, 0.5])
                ratio_schema = [column for column in schema["columns"] if column["name"] == ratio_column][0]
                self.assertEqual(ratio_schema["numerator"]["name"], "cyto_excluding_nuc")
                self.assertEqual(ratio_schema["denominator"]["name"], "nuc")

    def test_intensity_can_subtract_mean_background_from_mask_region(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_feature_h5(path)

            with Trajectory(path) as trajectory:
                spec = {
                    "feature_set": "intensity_background",
                    "object_set": "cyto",
                    "source_label_set": "cyto",
                    "features": [
                        {
                            "kind": "intensity",
                            "channel": {"readout": "erk"},
                            "compartment": {"label_set": "cyto", "name": "cell"},
                            "stats": ["mean"],
                            "background": {
                                "source_kind": "mask",
                                "source_name": "background",
                                "region": "inside",
                                "mode": "mean",
                            },
                        }
                    ],
                }
                trajectory.extract_features(spec, run_id="features_intensity_background")
                values = trajectory.object_set("cyto").read_features("intensity_background")
                schema = trajectory.object_set("cyto").read_feature_schema("intensity_background")

                self.np.testing.assert_allclose(values["erk_cell_mean"], [2.0, 17.0 / 3.0])
                column_schema = [column for column in schema["columns"] if column["name"] == "erk_cell_mean"][0]
                self.assertEqual(column_schema["background"]["source_kind"], "mask")
                self.assertEqual(column_schema["background"]["source_name"], "background")
                self.assertEqual(column_schema["background"]["region"], "inside")
                self.assertEqual(column_schema["background"]["mode"], "mean")

    def test_ratio_can_subtract_mean_background_from_inverse_label_region(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_feature_h5(path)

            with Trajectory(path) as trajectory:
                spec = site_signaling_v1_spec(
                    "cyto",
                    signal_channel={"readout": "erk"},
                    nuclear_mask_set="nuc",
                    background={
                        "source_kind": "label",
                        "source_name": "foreground",
                        "region": "inverse",
                        "mode": "mean",
                    },
                )
                trajectory.extract_features(spec, run_id="features_signaling_background")
                values = trajectory.object_set("cyto").read_features("site_v1")
                schema = trajectory.object_set("cyto").read_feature_schema("site_v1")

                ratio_column = "site_ratio"
                self.np.testing.assert_allclose(values[ratio_column], [3.0, 4.0 / 9.0])
                ratio_schema = [column for column in schema["columns"] if column["name"] == ratio_column][0]
                self.assertEqual(ratio_schema["background"]["source_kind"], "label")
                self.assertEqual(ratio_schema["background"]["source_name"], "foreground")
                self.assertEqual(ratio_schema["background"]["region"], "inverse")

    def test_site_signaling_can_use_label_set_as_nuclear_source(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_feature_h5(path)

            with Trajectory(path) as trajectory:
                spec = site_signaling_v1_spec(
                    "cyto",
                    signal_channel={"readout": "erk"},
                    nuclear_mask_set="nuc_label",
                    nuclear_source_kind="label",
                )
                trajectory.extract_features(spec, run_id="features_signaling_label_source")
                values = trajectory.object_set("cyto").read_features("site_v1")

                ratio_column = "site_ratio"
                self.np.testing.assert_allclose(values[ratio_column], [2.0, 0.5])

    def test_batch_feature_extraction_dry_run_does_not_write_h5_outputs(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_feature_h5(path)
            spec = site_signaling_v1_spec("cyto", signal_channel={"readout": "erk"}, nuclear_mask_set="nuc")

            summary = run_batch_feature_extraction(
                {
                    "job_id": "features_preview",
                    "save_outputs": False,
                    "files": [
                        {
                            "h5_path": str(path),
                            "feature_spec": spec.to_dict(),
                        }
                    ],
                },
            )

            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.features, 3)
            with Trajectory(path, mode="r") as trajectory:
                self.assertEqual(trajectory.object_set("cyto").feature_sets(), [])
                self.assertEqual(trajectory.feature_extraction_runs(), [])


if __name__ == "__main__":
    unittest.main()
