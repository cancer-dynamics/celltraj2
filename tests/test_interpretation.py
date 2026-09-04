from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from celltraj2.interpretation import (
    CLASSIFICATION_STATUS_CODES,
    QC_POSTERIOR_SWITCH,
    QC_SUSPICIOUS_BRANCH,
    BiologyRelease,
    ProjectObservationKey,
    StateNode,
    StateSpace,
    TypeNode,
    TypeTaxonomy,
    classification_content_digest,
    compile_gate_memberships,
    enforce_track_type_constancy,
    observation_spine_digest,
)
from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore


class InterpretationSchemaTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        self.np = np

    def test_project_observation_key_is_stable_and_strict(self):
        key = ProjectObservationKey(
            project_uuid=str(uuid4()),
            dataset_uuid=str(uuid4()),
            roi_uuid=str(uuid4()),
            object_set="cells",
            observation_id=7,
        )
        self.assertEqual(ProjectObservationKey.from_dict(key.to_dict()), key)
        self.assertEqual(key.partition[2], "cells")
        with self.assertRaisesRegex(ValueError, "one-based"):
            ProjectObservationKey(
                project_uuid=key.project_uuid,
                dataset_uuid=key.dataset_uuid,
                roi_uuid=key.roi_uuid,
                object_set="cells",
                observation_id=0,
            )

    def test_taxonomy_and_state_space_validate_hierarchies(self):
        root_id, leaf_id = str(uuid4()), str(uuid4())
        taxonomy = TypeTaxonomy(
            taxonomy_id=str(uuid4()),
            name="Tissue cells",
            nodes=(
                TypeNode(root_id, "Immune", "immune", "#3366AA"),
                TypeNode(leaf_id, "Myeloid", "myeloid", "#AA6633", parent_type_id=root_id),
            ),
        )
        self.assertEqual([node.type_id for node in taxonomy.leaf_nodes], [leaf_id])
        restored = TypeTaxonomy.from_dict(taxonomy.to_dict())
        self.assertEqual(restored, taxonomy)

        ordinary_id, terminal_id = str(uuid4()), str(uuid4())
        state_space = StateSpace(
            state_space_id=str(uuid4()),
            name="Myeloid states",
            type_taxonomy_id=taxonomy.taxonomy_id,
            scope_type_id=leaf_id,
            nodes=(
                StateNode(ordinary_id, "Motile", "motile", "#11AA44"),
                StateNode(
                    terminal_id,
                    "Dead",
                    "dead",
                    "#444444",
                    terminal=True,
                    absorbing=True,
                ),
            ),
        )
        self.assertTrue(StateSpace.from_dict(state_space.to_dict()).nodes[1].absorbing)

        with self.assertRaisesRegex(ValueError, "cycle"):
            TypeTaxonomy(
                taxonomy_id=str(uuid4()),
                name="Bad",
                nodes=(
                    TypeNode(root_id, "A", "a", "#000000", parent_type_id=leaf_id),
                    TypeNode(leaf_id, "B", "b", "#FFFFFF", parent_type_id=root_id),
                ),
            )

    def test_release_keeps_parallel_named_classification_roles(self):
        release = BiologyRelease(
            release_id=str(uuid4()),
            project_uuid=str(uuid4()),
            name="Live interpretation",
            type_taxonomy_id=str(uuid4()),
            type_classifications={"live_gate": str(uuid4()), "manual_live": str(uuid4())},
        )
        self.assertEqual(BiologyRelease.from_dict(release.to_dict()), release)
        self.assertEqual(set(release.type_classifications), {"live_gate", "manual_live"})

    def test_observation_spine_digest_depends_on_ordered_identity_columns(self):
        observations = self.np.asarray(
            [(1, 1, 5), (2, 2, 6)],
            dtype=[("observation_id", "<i8"), ("frame", "<i4"), ("label_id", "<i8")],
        )
        first = observation_spine_digest(observations, object_set="cells")
        changed = observations.copy()
        changed[1]["label_id"] = 7
        second = observation_spine_digest(changed, object_set="cells")
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)

    def test_gate_compiler_marks_unknown_overlap_and_exclusion(self):
        result = compile_gate_memberships(
            self.np.arange(1, 5),
            self.np.asarray(
                [
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [0, 0],
                ],
                dtype=self.np.float32,
            ),
            excluded=self.np.asarray([False, False, False, True]),
        )
        self.assertEqual(
            result.values["assignment_status"].tolist(),
            [
                CLASSIFICATION_STATUS_CODES["assigned"],
                CLASSIFICATION_STATUS_CODES["assigned"],
                CLASSIFICATION_STATUS_CODES["ambiguous"],
                CLASSIFICATION_STATUS_CODES["excluded"],
            ],
        )
        self.np.testing.assert_allclose(result.probabilities[2], [0.5, 0.5])
        self.assertEqual(result.values["class_index"].tolist(), [1, 2, 0, 0])

    def test_track_constancy_flags_switch_and_suspicious_branch(self):
        initial = compile_gate_memberships(
            self.np.arange(1, 4),
            self.np.asarray([[1, 0], [0, 1], [1, 0]], dtype=self.np.float32),
        )
        assignments = self.np.asarray(
            [(1, 1, 2), (2, 1, 0), (3, 1, 0)],
            dtype=[("observation_id", "<i8"), ("lineage_id", "<i8"), ("n_children", "<i4")],
        )
        links = self.np.asarray(
            [(1, 2), (1, 3)],
            dtype=[("parent_observation_id", "<i8"), ("child_observation_id", "<i8")],
        )
        result = enforce_track_type_constancy(
            initial.values,
            initial.probabilities,
            assignments,
            links=links,
        )
        self.assertEqual(
            result.values["assignment_status"].tolist(),
            [CLASSIFICATION_STATUS_CODES["ambiguous"]] * 3,
        )
        self.assertTrue(self.np.all(result.values["qc_flags"] & QC_SUSPICIOUS_BRANCH))
        self.assertTrue(self.np.any(result.values["qc_flags"] & QC_POSTERIOR_SWITCH))
        branch = [item for item in result.review_records if item["kind"] == "suspicious_branch"]
        self.assertEqual(branch[0]["biological_event_created"], False)

    def test_track_constancy_propagates_normalized_posteriors_to_unsupported_rows(self):
        initial = compile_gate_memberships(
            self.np.arange(1, 4),
            self.np.asarray([[1, 0], [0, 0], [0, 0]], dtype=self.np.float32),
        )
        assignments = self.np.asarray(
            [(1, 7, 0), (2, 7, 0), (3, 7, 0)],
            dtype=[("observation_id", "<i8"), ("lineage_id", "<i8"), ("n_children", "<i4")],
        )
        result = enforce_track_type_constancy(
            initial.values,
            initial.probabilities,
            assignments,
        )
        self.assertEqual(
            result.values["assignment_status"].tolist(),
            [CLASSIFICATION_STATUS_CODES["assigned"]] * 3,
        )
        self.np.testing.assert_allclose(result.probabilities, [[1, 0], [1, 0], [1, 0]])


class InterpretationStoreTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("numpy/h5py are not installed")
        self.np = np

    def _metadata(self) -> TrajectoryMetadata:
        return TrajectoryMetadata(
            roi_id="roi-1",
            dataset_id="dataset-1",
            frame_count=2,
            image_source=ImageSourceSpec(source_type="embedded_h5", axes=("T", "Y", "X")),
        )

    def _observations(self):
        return self.np.asarray(
            [(1, 1, 1), (2, 2, 1)],
            dtype=[("observation_id", "<i8"), ("frame", "<i4"), ("label_id", "<i8")],
        )

    def test_classification_h5_roundtrip_is_dense_immutable_and_idempotent(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ct2.h5"
            with TrajectoryStore.create(path, metadata=self._metadata()) as store:
                store.write_observations(
                    "cells",
                    self._observations(),
                    {"schema": "celltraj2.observations.v1"},
                )
                result = compile_gate_memberships(
                    self.np.asarray([1, 2]),
                    self.np.asarray([[1, 0], [0, 1]], dtype=self.np.float32),
                )
                artifact_id = str(uuid4())
                schema = {
                    "schema": "celltraj2.classification_values.v1",
                    "classification_kind": "cell_type",
                    "class_ids": [str(uuid4()), str(uuid4())],
                }
                digest = classification_content_digest(result.values, result.probabilities, schema)
                manifest = {"artifact_id": artifact_id, "content_digest": digest}
                first = store.write_classification_set(
                    "cells",
                    artifact_id,
                    values=result.values,
                    probabilities=result.probabilities,
                    schema=schema,
                    source_manifest=manifest,
                )
                second = store.write_classification_set(
                    "cells",
                    artifact_id,
                    values=result.values,
                    probabilities=result.probabilities,
                    schema=schema,
                    source_manifest=manifest,
                )
                self.assertEqual(first, second)
                loaded = store.read_classification_set("cells", artifact_id)
                self.assertEqual(loaded["probabilities"].dtype, self.np.dtype("float32"))
                self.np.testing.assert_array_equal(loaded["values"], result.values)
                self.assertEqual(store.list_classification_sets("cells"), [artifact_id])
                release_id = str(uuid4())
                release_manifest = {
                    "schema": "site.biology_release.v1",
                    "release_id": release_id,
                    "name": "Working release",
                }
                store.write_interpretation_release("cells", release_id, release_manifest)
                store.set_active_interpretation(
                    "cells",
                    active_release_ids=[release_id],
                    selected_release_id=release_id,
                )
                self.assertEqual(
                    store.read_json("/object_sets/cells/interpretation_active.json"),
                    {
                        "schema": "celltraj2.interpretation_active.v1",
                        "active_release_ids": [release_id],
                        "selected_release_id": release_id,
                    },
                )
                entities = self.np.asarray(
                    [(1,), (3,)],
                    dtype=[("observation_id", "<u8")],
                )
                store.write_boundary_library(
                    "cell_surfaces",
                    entities=entities,
                    points={"point_id": self.np.asarray([1, 2], dtype=self.np.uint64)},
                    sources=[{"object_set": "cells"}],
                    schema={"schema": "celltraj2.boundary_library.v1"},
                )
                store.project_classification_to_boundary(
                    "cell_surfaces",
                    "working_type",
                    object_set="cells",
                    classification_set=artifact_id,
                    schema={"display_name": "Working cell type"},
                )
                projected = store.read_boundary_entity_attributes(
                    "cell_surfaces", "working_type"
                )
                self.assertEqual(projected["class_index"].tolist(), [1, 0])
                projection_schema = store.read_json(
                    "/boundaries/cell_surfaces/entity_attributes/working_type/schema.json"
                )
                self.assertEqual(projection_schema["missing_observation_count"], 1)
                with self.assertRaisesRegex(ValueError, "another content digest"):
                    store.write_classification_set(
                        "cells",
                        artifact_id,
                        values=result.values,
                        probabilities=result.probabilities,
                        schema=schema,
                        source_manifest={"artifact_id": artifact_id, "content_digest": "0" * 64},
                    )


if __name__ == "__main__":
    unittest.main()
