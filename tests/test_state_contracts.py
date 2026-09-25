from __future__ import annotations

from dataclasses import replace
import json
import unittest
from uuid import uuid4

from celltraj2.interpretation import (
    BiologyRelease, ProjectObservationKey, TypeNode, TypeTaxonomy, canonical_json_digest,
)
from celltraj2.state_contracts import (
    BIOLOGY_RELEASE_V2_SCHEMA, KineticModelBinding, MembershipExpression,
    StateComponentBinding, StateDependency, StateMembershipSpec,
    kinetic_cohort_digest, validate_release_dependency_closure,
    validate_release_state_context,
)


def uid():
    return str(uuid4())


class StateContractTests(unittest.TestCase):
    def setUp(self):
        self.immune, self.cd11b, self.cd3, self.tumor = (uid() for _ in range(4))
        self.taxonomy = TypeTaxonomy(uid(), "Types", (
            TypeNode(self.immune, "Immune", "immune", "#111111"),
            TypeNode(self.cd11b, "CD11B+", "cd11b", "#222222", self.immune),
            TypeNode(self.cd3, "CD3+", "cd3", "#333333", self.immune),
            TypeNode(self.tumor, "Tumor", "tumor", "#444444"),
        ))
        self.project, self.dataset, self.roi = uid(), uid(), uid()
        self.keys = tuple(ProjectObservationKey(self.project, self.dataset, self.roi, "cells", i)
                          for i in range(1, 5))
        self.expression = MembershipExpression("or", children=(
            MembershipExpression("and", children=(
                MembershipExpression("in_subtree", self.immune),
                MembershipExpression("not", children=(MembershipExpression("in_subtree", self.cd11b),)),
            )),
            MembershipExpression("in_subtree", self.tumor),
        ))
        self.membership = StateMembershipSpec(uid(), self.taxonomy.taxonomy_id, uid(), uid(),
            "primary", self.expression, self.keys, (self.keys[0].partition,), treatment_ids=("treated",))
        self.binding = StateComponentBinding(uid(), "fate/death", self.membership.membership_id,
            uid(), uid(), self.membership.conditioning_digest)

    def release(self, **overrides):
        params = dict(
            release_id=uid(), project_uuid=self.project, name="State", schema=BIOLOGY_RELEASE_V2_SCHEMA,
            type_taxonomy_id=self.taxonomy.taxonomy_id,
            type_classifications={"primary": self.membership.type_classification_id},
            state_memberships=(self.membership,), state_bindings=(self.binding,),
        )
        params.update(overrides)
        ids = {self.taxonomy.taxonomy_id, self.membership.type_classification_id,
               self.membership.type_release_id, self.binding.definition_id, self.binding.assignment_id}
        for binding in params["state_bindings"]:
            ids.update(binding.artifact_ids)
        for binding in params.get("kinetic_bindings", ()):
            ids.update(binding.artifact_ids)
        params.setdefault("dependency_closure", tuple(StateDependency("artifact", item, "a" * 64)
                                                     for item in sorted(ids)))
        return BiologyRelease(**params)

    def kinetic(self, leaf=None, observations=None, **overrides):
        params = dict(binding_id=uid(), state_binding_id=self.binding.binding_id,
                      leaf_type_id=leaf or self.tumor, observation_cohort=observations or self.keys[:2],
                      timebase={"unit": "s", "source": "recorded_timestamps"})
        params.update(overrides)
        params["cohort_digest"] = kinetic_cohort_digest(params["leaf_type_id"], params["observation_cohort"],
            params.get("window_ids", ()), params.get("pair_ids", ()), params["timebase"])
        return KineticModelBinding(**params)

    def test_boolean_truth_table_parent_calls_and_unknown_not_universe(self):
        expected = {self.immune: True, self.cd11b: False, self.cd3: True, self.tumor: True}
        for type_id, result in expected.items():
            self.assertEqual(self.expression.matches(type_id, self.taxonomy), result)
        self.assertFalse(self.expression.matches(None, self.taxonomy))
        not_tumor = MembershipExpression("not", children=(MembershipExpression("assigned_exact", self.tumor),))
        self.assertFalse(not_tumor.matches(uid(), self.taxonomy))
        self.assertTrue(not_tumor.matches(self.immune, self.taxonomy))
        exact_parent = MembershipExpression("assigned_exact", self.immune)
        self.assertFalse(exact_parent.matches(self.cd3, self.taxonomy))
        for status in ("unknown", "ambiguous", "excluded", "not_evaluated"):
            self.assertFalse(self.membership.eligible_assignment(self.immune, status, self.taxonomy))
        self.assertFalse(self.membership.eligible_assignment(self.immune, "assigned", self.taxonomy, incompatible=True))
        self.assertFalse(replace(self.membership, resolved_leaves_only=True).eligible_assignment(
            self.immune, "assigned", self.taxonomy))

    def test_ast_rejects_malformed_or_unknown_nodes_and_tampered_readable_expression(self):
        for expression in ({"op": "eval", "type_id": self.immune}, {"op": "not", "children": []},
                           {"op": "or", "children": []}, {"op": "assigned_exact", "type_id": "bad"}):
            with self.assertRaises(ValueError):
                MembershipExpression.from_dict(expression)
        with self.assertRaisesRegex(ValueError, "missing taxonomy"):
            MembershipExpression("in_subtree", uid()).validate_taxonomy(self.taxonomy)
        payload = self.membership.to_dict()
        payload["canonical_expression"] = "True"
        with self.assertRaisesRegex(ValueError, "canonical expression"):
            StateMembershipSpec.from_dict(payload)

    def test_frozen_file_scope_and_membership_digest_are_independent_of_type_retyping(self):
        restored = StateMembershipSpec.from_dict(json.loads(json.dumps(self.membership.to_dict())))
        self.assertEqual(restored, self.membership)
        self.assertEqual(replace(self.membership, type_classification_id=uid()).conditioning_digest,
                         self.membership.conditioning_digest)
        with self.assertRaisesRegex(ValueError, "frozen resolved file set"):
            replace(self.membership, resolved_partitions=())
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            replace(self.membership, resolved_observations=(self.keys[0], self.keys[0]))
        with self.assertRaisesRegex(ValueError, "hard calls"):
            replace(self.membership, allowed_assignment_statuses=("unknown",))

    def test_shared_state_accepts_separate_exact_leaf_kinetics_and_parent_is_rejected(self):
        assignments = dict(zip(self.keys, (self.tumor, self.tumor, self.cd3, self.cd3)))
        tumor = self.kinetic()
        immune = self.kinetic(self.cd3, self.keys[2:])
        release = self.release(kinetic_bindings=(tumor, immune))
        validate_release_state_context(release, self.taxonomy, assignments)
        self.assertEqual(len(release.kinetic_bindings), 2)
        self.assertNotIn("leaf_type_id", self.binding.to_dict())
        with self.assertRaisesRegex(ValueError, "exact Type leaf"):
            self.kinetic(self.immune).validate_context(self.taxonomy, self.membership, assignments)
        with self.assertRaisesRegex(ValueError, "mixed Type leaves"):
            self.kinetic(observations=self.keys).validate_context(self.taxonomy, self.membership, assignments)

    def test_window_and_entire_pair_segment_validate_leaf_file_and_physical_time(self):
        assignments = {key: self.tumor for key in self.keys}
        window_id, pair_id = uid(), uid()
        kinetic = self.kinetic(observations=self.keys, window_ids=(window_id,), pair_ids=(pair_id,),
                               operator_id=uid(), partition_id=uid(), assessment_id=uid())
        windows = {window_id: {"anchor": self.keys[1], "members": self.keys[:2]}}
        pairs = {pair_id: {"source_anchor": self.keys[0], "target_anchor": self.keys[2],
                          "segment": self.keys[:3], "elapsed_time": 2, "time_unit": "s",
                          "split_ids": ("train",) * 3, "graph_components": ("track_1",) * 3,
                          "accepted_edges": tuple(zip(self.keys[:2], self.keys[1:3]))}}
        kinetic.validate_context(self.taxonomy, self.membership, assignments, windows=windows, pairs=pairs)
        narrow = self.kinetic(observations=self.keys[:2], window_ids=(window_id,))
        with self.assertRaisesRegex(ValueError, "Entire kinetic window segment"):
            narrow.validate_context(self.taxonomy, self.membership, assignments,
                windows={window_id: {**windows[window_id], "segment": self.keys[:3]}})
        other_file = replace(self.keys[2], roi_uuid=uid())
        shared_cohort = (*self.keys, other_file)
        shared_membership = replace(self.membership, resolved_observations=shared_cohort,
            resolved_partitions=(self.keys[0].partition, other_file.partition), conditioning_digest=None)
        shared_kinetic = self.kinetic(observations=shared_cohort, window_ids=(window_id,))
        with self.assertRaisesRegex(ValueError, "Kinetic window crosses files"):
            shared_kinetic.validate_context(self.taxonomy, shared_membership,
                {**assignments, other_file: self.tumor},
                windows={window_id: {**windows[window_id],
                    "segment": (self.keys[0], other_file, self.keys[1])}})
        assignments[self.keys[1]] = self.cd3
        with self.assertRaisesRegex(ValueError, "mixed Type leaves"):
            kinetic.validate_context(self.taxonomy, self.membership, assignments, windows=windows, pairs=pairs)
        assignments[self.keys[1]] = self.tumor
        pairs[pair_id]["split_ids"] = ("train", "test", "test")
        with self.assertRaisesRegex(ValueError, "crosses split"):
            kinetic.validate_context(self.taxonomy, self.membership, assignments, windows=windows, pairs=pairs)
        pairs[pair_id]["split_ids"] = ("train",) * 3
        pairs[pair_id]["time_unit"] = "frame"
        with self.assertRaisesRegex(ValueError, "physical elapsed"):
            kinetic.validate_context(self.taxonomy, self.membership, assignments, windows=windows, pairs=pairs)
        with self.assertRaisesRegex(ValueError, "Missing kinetic window"):
            kinetic.validate_context(self.taxonomy, self.membership, assignments)
        with self.assertRaisesRegex(ValueError, "physical timebase"):
            self.kinetic(timebase={"unit": "frame", "source": "frame_index"})
        with self.assertRaisesRegex(ValueError, "lag-pair cohort"):
            self.kinetic(operator_id=uid())
        with self.assertRaisesRegex(ValueError, "mapping and assessment"):
            self.kinetic(operator_id=uid(), pair_ids=(pair_id,))
        pairs[pair_id]["time_unit"] = "s"
        pairs[pair_id]["accepted_edges"] = ()
        with self.assertRaisesRegex(ValueError, "every accepted graph edge"):
            kinetic.validate_context(self.taxonomy, self.membership, assignments, windows=windows, pairs=pairs)

    def test_release_v2_json_roundtrip_and_v1_serialization_stays_historical(self):
        release = self.release()
        self.assertEqual(BiologyRelease.from_dict(json.loads(json.dumps(release.to_dict()))), release)
        legacy = BiologyRelease(uid(), self.project, "Type only", self.taxonomy.taxonomy_id,
                                {"primary": uid()}, state_spaces=(uid(),))
        payload = legacy.to_dict()
        self.assertNotIn("state_memberships", payload)
        self.assertNotIn("dependency_closure", payload)
        self.assertEqual(BiologyRelease.from_dict(payload), legacy)
        self.assertEqual(canonical_json_digest(legacy), canonical_json_digest(payload))
        with self.assertRaisesRegex(ValueError, "require BiologyRelease v2"):
            replace(legacy, state_bindings=(self.binding,))
        with self.assertRaisesRegex(ValueError, "explicit validated binding adoption"):
            self.release(state_classifications={"legacy": uid()})

    def test_release_rejects_missing_membership_closure_and_wrong_conditioning(self):
        with self.assertRaisesRegex(ValueError, "missing membership"):
            self.release(state_memberships=())
        with self.assertRaisesRegex(ValueError, "missing required artifacts"):
            self.release(dependency_closure=())
        with self.assertRaisesRegex(ValueError, "conditioning digest"):
            self.release(state_bindings=(replace(self.binding, conditioning_digest="b" * 64),))
        with self.assertRaisesRegex(ValueError, "different project"):
            self.release(project_uuid=uid())

    def test_same_channel_overlaps_rejected_but_disjoint_coverage_and_channels_work(self):
        second = replace(self.binding, binding_id=uid())
        with self.assertRaisesRegex(ValueError, "Overlapping authoritative"):
            self.release(state_bindings=(self.binding, second))
        self.release(state_bindings=(replace(self.binding, coverage_observations=self.keys[:2]),
                                    replace(second, coverage_observations=self.keys[2:])))
        self.release(state_bindings=(self.binding, replace(second, channel="fate/division")))
        outside = ProjectObservationKey(self.project, self.dataset, uid(), "cells", 1)
        with self.assertRaisesRegex(ValueError, "coverage escapes"):
            self.release(state_bindings=(replace(self.binding, coverage_observations=(outside,)),))

    def test_external_closure_verifies_existence_digest_and_recursive_references(self):
        release = self.release()
        digests = {dep.artifact_id: dep.digest for dep in release.dependency_closure}
        graph = {artifact_id: () for artifact_id in digests}
        validate_release_dependency_closure(release, digests, graph)
        first = next(iter(digests))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_release_dependency_closure(release, {k: v for k, v in digests.items() if k != first}, graph)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_release_dependency_closure(release, dict(digests, **{first: "b" * 64}), graph)
        with self.assertRaisesRegex(ValueError, "graph is required"):
            validate_release_dependency_closure(release, digests)
        graph[first] = (uid(),)
        with self.assertRaisesRegex(ValueError, "Transitive"):
            validate_release_dependency_closure(release, digests, graph)

    def test_type_role_pin_and_saved_membership_decisions_are_checked(self):
        with self.assertRaisesRegex(ValueError, "Type classification/role"):
            self.release(type_classifications={"primary": uid()})
        with self.assertRaisesRegex(ValueError, "ineligible or unresolved"):
            validate_release_state_context(self.release(), self.taxonomy, {key: self.cd11b for key in self.keys})


if __name__ == "__main__":
    unittest.main()
