from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from celltraj2.boundaries import (
    GEOMETRY_QUALITY_NOT_SELECTED,
    BoundarySourceSpec,
    optimal_transport_plan,
)
from celltraj2.boundary_batch import run_batch_boundaries
from celltraj2.registration import (
    FRAME_STATUS,
    RegistrationSet,
    pairwise_result_dtype,
    registration_digest,
)
from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.surface_motion_batch import run_batch_surface_motion
from celltraj2.store import TrajectoryStore
from celltraj2.trajectory import Trajectory


class BoundaryLibraryTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        self.np = np

    def require_h5py(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py is not installed")

    def _metadata(self, frame_count: int) -> TrajectoryMetadata:
        return TrajectoryMetadata(
            roi_id="sample_XY001_ROI001",
            dataset_id="sample",
            frame_count=frame_count,
            image_source=ImageSourceSpec(
                source_type="embedded_h5",
                axes=("T", "Y", "X", "C"),
                sizes={"T": frame_count, "Y": 16, "X": 16, "C": 1},
            ),
            acquisition={
                "micron_per_pixel": 1.0,
                "voxel_size_um": {"Y": 1.0, "X": 1.0},
            },
        )

    def _create_indexed(self, path: Path, frames: list):
        self.require_h5py()
        with TrajectoryStore.create(path, metadata=self._metadata(len(frames))) as store:
            for frame, labels in enumerate(frames, start=1):
                store.write_label_frame("cells", frame, labels)
        with Trajectory(path, mode="r+") as trajectory:
            trajectory.index_observations("cells", run_id="index_cells")

    def test_sinkhorn_plan_preserves_uniform_mass(self):
        source = self.np.asarray([[0.0, 0.0], [1.0, 0.0]])
        target = self.np.asarray([[0.0, 1.0], [1.0, 1.0]])
        plan = optimal_transport_plan(
            source,
            target,
            method="sinkhorn",
            regularization=0.01,
            mass_tolerance=1e-8,
        )
        self.assertEqual(plan.method, "numpy.sinkhorn")
        self.assertAlmostEqual(float(self.np.sum(plan.mass)), 1.0, places=7)
        self.assertAlmostEqual(plan.total_cost, 1.0, places=5)

    def test_pcdiff_shape_operator_recovers_unit_sphere_curvature(self):
        try:
            import pcdiff  # noqa: F401
        except ImportError:
            self.skipTest("pcdiff is not installed")
        from celltraj2.boundaries import _pcdiff_geometry

        count = 128
        index = self.np.arange(count)
        z = 1.0 - 2.0 * (index + 0.5) / count
        angle = self.np.pi * (3.0 - self.np.sqrt(5.0)) * index
        radius = self.np.sqrt(1.0 - z * z)
        points = self.np.column_stack([z, radius * self.np.sin(angle), radius * self.np.cos(angle)])
        geometry = _pcdiff_geometry(points, points, knn=16, np=self.np)
        self.assertGreater(float(self.np.mean(self.np.sum(geometry["normals_zyx"] * points, axis=1))), 0.95)
        self.assertAlmostEqual(float(self.np.mean(geometry["principal_curvature_1"])), -1.0, delta=0.2)
        self.assertAlmostEqual(float(self.np.mean(geometry["principal_curvature_2"])), -1.0, delta=0.2)

    def test_native_library_roundtrip_supports_objects_and_mask_surfaces(self):
        frame = self.np.zeros((12, 12), dtype=self.np.uint16)
        frame[2:5, 2:5] = 1
        frame[6:10, 7:11] = 2
        basement = self.np.zeros_like(frame, dtype=bool)
        basement[10, 1:11] = True
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed(path, [frame])
            with Trajectory(path, mode="r+") as trajectory:
                trajectory.write_mask_frame("basement", 1, basement)
                result = trajectory.build_boundary_library(
                    "cells_and_matrix",
                    sources=[
                        BoundarySourceSpec(kind="object_set", name="cells", object_set="cells"),
                        BoundarySourceSpec(
                            kind="mask_set",
                            name="basement",
                            label_set="basement",
                            role="basement_membrane",
                        ),
                    ],
                )
                self.assertEqual(result.entity_count, 3)
                self.assertGreater(result.point_count, 0)
                self.assertFalse(result.schema["registration_applied"])
                self.assertEqual(len(result.schema["boundary_digest"]), 64)
                view = trajectory.boundary_library("cells_and_matrix")
                self.assertEqual(trajectory.boundary_sets(), ["cells_and_matrix"])
                self.assertEqual(view.entity_id_for_observation(1), 1)
                first = view.read_points(1)
                self.assertTrue(self.np.array_equal(first["point_id"], self.np.arange(1, 9)))
                self.assertTrue(self.np.all(first["native_index_zyx"][:, 0] == 0))
                self.np.testing.assert_allclose(
                    first["native_position_zyx"], first["native_index_zyx"].astype(float)
                )
                self.assertEqual(view.sources[1]["role"], "basement_membrane")
                attributes = self.np.zeros(
                    result.entity_count,
                    dtype=[("state", "<i2"), ("cell_type", "S16")],
                )
                attributes["state"] = [1, 2, 0]
                attributes["cell_type"] = [b"epithelial", b"epithelial", b"matrix"]
                trajectory.write_boundary_entity_attributes(
                    "cells_and_matrix",
                    "biology",
                    attributes,
                    {"schema": "example.boundary_attributes.v1"},
                )
                self.assertEqual(view.entity_attribute_sets(), ["biology"])
                self.assertEqual(view.entity_attributes("biology")["state"].tolist(), [1, 2, 0])
                self.assertEqual(
                    [(item.start, item.stop) for item in view.point_spans([3, 1])],
                    [(0, 8), (20, 30)],
                )

    def test_geometry_and_external_neighbor_sets_are_point_row_aligned(self):
        frame = self.np.zeros((14, 14), dtype=self.np.uint16)
        frame[3:8, 2:6] = 1
        frame[3:8, 7:11] = 2
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed(path, [frame])
            with Trajectory(path, mode="r+") as trajectory:
                trajectory.object_set("cells").build_boundary_library("native")
                geometry = trajectory.compute_boundary_geometry(
                    "native", geometry_set="surface", knn=6, backend="local"
                )
                self.assertEqual(geometry.values["normals_zyx"].shape[1], 3)
                self.assertGreater(
                    int(self.np.sum(self.np.isfinite(geometry.values["mean_curvature"]))), 0
                )
                neighbors = trajectory.compute_boundary_neighbors(
                    "native", neighbor_set="contacts", k=1
                )
                self.assertEqual(
                    neighbors.indptr.shape[0], geometry.values["quality_flags"].shape[0] + 1
                )
                self.assertEqual(neighbors.edge_count, geometry.values["quality_flags"].shape[0])
                view = trajectory.boundary_library("native")
                edges = view.neighbor_edges("contacts", 1)
                point_entities = view.read_points(fields=("boundary_entity_id",))["boundary_entity_id"]
                self.assertTrue(
                    self.np.all(
                        point_entities[edges["source_point_rows"]]
                        != point_entities[edges["target_point_rows"]]
                    )
                )
                self.assertIn("surface", view.geometry_sets())
                self.assertIn("contacts", view.neighbor_sets())
                topology = view.geometry_topology("surface", 1)
                self.assertGreater(topology["target_point_rows"].shape[0], 0)

    def test_registered_boundary_ot_tracking_keeps_native_points_and_stores_dependency(self):
        frame_1 = self.np.zeros((16, 16), dtype=self.np.uint16)
        frame_1[4:9, 3:8] = 1
        frame_2 = self.np.zeros((16, 16), dtype=self.np.uint16)
        frame_2[4:9, 5:10] = 1
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed(path, [frame_1, frame_2])
            with Trajectory(path, mode="r+") as trajectory:
                trajectory.object_set("cells").build_boundary_library("native")
                native_before = trajectory.boundary_library("native").native_positions(2).copy()
                frames = self.np.asarray([1, 2], dtype=self.np.int32)
                transforms = self.np.repeat(self.np.eye(3)[None, :, :], 2, axis=0)
                transforms[1, 1, -1] = -2.0
                status = self.np.asarray(
                    [FRAME_STATUS["reference"], FRAME_STATUS["estimated"]], dtype=self.np.uint8
                )
                digest = registration_digest(frames, transforms, status)
                registration = RegistrationSet(
                    name="drift_corrected",
                    frames=frames,
                    transforms=transforms,
                    frame_status=status,
                    pairwise_results=self.np.empty(0, dtype=pairwise_result_dtype()),
                    schema={
                        "schema": "celltraj2.registration.v1",
                        "method": "pairwise_symmetric_nearest_neighbor_translation",
                        "spatial_axes": ["Y", "X"],
                        "coordinate_scale_zyx": [1.0, 1.0, 1.0],
                        "registration_digest": digest,
                    },
                    canvas={"output_shape": [16, 18], "canvas_offset": [0.0, 2.0]},
                )
                trajectory.store.write_registration_set(registration)
                trajectory.store.set_active_registration("drift_corrected", reason="test")
                result = trajectory.track_minimum_boundary_ot_cost(
                    "cells",
                    boundary_set="native",
                    max_distance=0.5,
                    ot_cost_cutoff=0.05,
                    track_set="boundary_ot",
                    ot_method="sinkhorn",
                    sinkhorn_regularization=0.01,
                    mass_tolerance=1e-7,
                    max_boundary_points=None,
                )
                self.assertEqual(result.link_count, 1)
                self.assertIsNotNone(result.motion_path)
                self.assertEqual(
                    result.graph.schema["registration_dependency"]["registration_digest"], digest
                )
                native_after = trajectory.boundary_library("native").native_positions(2)
                self.np.testing.assert_array_equal(native_after, native_before)
                motion = trajectory.store.read_boundary_motion("native", "boundary_ot")
                self.assertEqual(
                    motion["schema"]["registration_dependency"]["registration_digest"], digest
                )
                self.assertEqual(
                    motion["schema"]["displacement_definition"],
                    "T_target(q_native)-T_source(p_native)",
                )
                self.assertLess(float(motion["links"][0]["ot_cost"]), 0.05)
                self.assertLess(
                    float(
                        self.np.average(
                            self.np.linalg.norm(
                                motion["transport"]["registered_displacement_zyx"], axis=1
                            ),
                            weights=motion["transport"]["mass"],
                        )
                    ),
                    0.05,
                )

    def test_boundary_batch_test_is_read_only_and_run_saves_source_scoped_products(self):
        frame = self.np.zeros((14, 14), dtype=self.np.uint16)
        frame[3:8, 2:7] = 1
        basement = self.np.zeros_like(frame, dtype=bool)
        basement[10, 1:12] = True
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed(path, [frame])
            with Trajectory(path, mode="r+") as trajectory:
                trajectory.write_mask_frame("basement", 1, basement)
            payload = {
                "job_id": "boundary_batch_test",
                "save_outputs": False,
                "files": [
                    {
                        "h5_path": str(path),
                        "boundary_set": "cells_and_matrix",
                        "sources": [
                            {
                                "kind": "object_set",
                                "name": "cells",
                                "object_set": "cells",
                                "role": "cell",
                            },
                            {
                                "kind": "mask_set",
                                "name": "basement",
                                "label_set": "basement",
                                "role": "basement_membrane",
                            },
                        ],
                        "geometries": [
                            {
                                "geometry_set": "cell_surface",
                                "backend": "local",
                                "knn": 6,
                                "source_roles": ["cell"],
                            }
                        ],
                        "neighbors": [
                            {
                                "neighbor_set": "cell_to_matrix",
                                "source_roles": ["cell"],
                                "target_roles": ["basement_membrane"],
                                "k": 1,
                            }
                        ],
                    }
                ],
            }
            events = []
            dry_summary = run_batch_boundaries(payload, reporter=events.append)
            self.assertEqual(dry_summary.completed, 1)
            self.assertEqual(dry_summary.geometry_sets, 1)
            self.assertEqual(dry_summary.neighbor_sets, 1)
            self.assertTrue(any(event.get("event") == "boundary_frame_summary" for event in events))
            with Trajectory(path, mode="r") as trajectory:
                self.assertEqual(trajectory.boundary_sets(), [])

            payload["save_outputs"] = True
            payload["files"][0]["save_outputs"] = True
            saved_summary = run_batch_boundaries(payload)
            self.assertEqual(saved_summary.completed, 1)
            with Trajectory(path, mode="r") as trajectory:
                view = trajectory.boundary_library("cells_and_matrix")
                self.assertEqual(view.geometry_sets(), ["cell_surface"])
                self.assertEqual(view.neighbor_sets(), ["cell_to_matrix"])
                geometry = view.geometry("cell_surface")
                mask_entity = int(view.entities[view.entities["source_id"] == 2][0]["boundary_entity_id"])
                mask_span = view.point_slice(mask_entity)
                self.assertTrue(
                    self.np.all(
                        geometry["quality_flags"][mask_span] & GEOMETRY_QUALITY_NOT_SELECTED
                    )
                )
                neighbor_schema = trajectory.store.read_json(
                    "/boundaries/cells_and_matrix/neighbors/cell_to_matrix/schema.json"
                )
                self.assertEqual(neighbor_schema["source_ids"], [1])
                self.assertEqual(neighbor_schema["target_ids"], [2])

    def test_transient_boundary_tracking_and_independent_surface_motion(self):
        frame_1 = self.np.zeros((16, 16), dtype=self.np.uint16)
        frame_1[4:9, 3:8] = 1
        frame_2 = self.np.zeros((16, 16), dtype=self.np.uint16)
        frame_2[4:9, 4:9] = 1
        basement = self.np.zeros_like(frame_1, dtype=bool)
        basement[12, 1:14] = True
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            self._create_indexed(path, [frame_1, frame_2])
            with Trajectory(path, mode="r+") as trajectory:
                transient = trajectory.track_minimum_boundary_ot_cost(
                    "cells",
                    boundary_set=None,
                    max_distance=3.0,
                    track_set="transient_ot",
                    ot_method="sinkhorn",
                    sinkhorn_regularization=0.05,
                    save_motion=False,
                    save_outputs=False,
                )
                self.assertEqual(transient.link_count, 1)
                self.assertFalse(transient.graph.schema["boundary_dependency"]["stored"])
                self.assertEqual(trajectory.boundary_sets(), [])

                trajectory.write_mask_frame("basement", 1, basement)
                trajectory.write_mask_frame("basement", 2, basement)
                trajectory.build_boundary_library(
                    "interaction_domain",
                    sources=[
                        BoundarySourceSpec(
                            kind="object_set",
                            name="tracked_cells",
                            object_set="cells",
                            role="cell",
                        ),
                        BoundarySourceSpec(
                            kind="mask_set",
                            name="basement",
                            label_set="basement",
                            role="basement_membrane",
                        ),
                    ],
                )
                tracked = trajectory.track_minimum_centroid_distance(
                    "cells", max_distance=3.0, track_set="centroid"
                )
                self.assertEqual(tracked.link_count, 1)
                motion = trajectory.compute_boundary_motion(
                    "cells",
                    "centroid",
                    boundary_set="interaction_domain",
                    boundary_source_name="tracked_cells",
                    motion_set="centroid_surface_ot",
                    ot_method="sinkhorn",
                    sinkhorn_regularization=0.05,
                )
                self.assertEqual(motion.link_count, 1)
                self.assertGreater(motion.transport_edge_count, 0)
                self.assertEqual(
                    motion.schema["track_dependency"]["track_digest"],
                    tracked.graph.schema["track_digest"],
                )
                stored = trajectory.store.read_boundary_motion(
                    "interaction_domain", "centroid_surface_ot"
                )
                self.assertEqual(stored["links"].shape[0], 1)
            motion_events = []
            motion_summary = run_batch_surface_motion(
                {
                    "job_id": "surface_motion_test",
                    "save_outputs": False,
                    "files": [
                        {
                            "h5_path": str(path),
                            "object_set": "cells",
                            "track_set": "centroid",
                            "boundary_set": "interaction_domain",
                            "boundary_source_name": "tracked_cells",
                            "motion_set": "dry_motion",
                            "ot_method": "sinkhorn",
                        }
                    ],
                },
                reporter=motion_events.append,
            )
            self.assertEqual(motion_summary.completed, 1)
            self.assertEqual(motion_summary.links, 1)
            self.assertTrue(
                any(event.get("event") == "surface_motion_link_summary" for event in motion_events)
            )
            with Trajectory(path, mode="r") as trajectory:
                self.assertNotIn("dry_motion", trajectory.boundary_library("interaction_domain").motion_sets())


if __name__ == "__main__":
    unittest.main()
