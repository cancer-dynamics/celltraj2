from __future__ import annotations

from dataclasses import replace
import json
import unittest
from uuid import uuid4

from celltraj2.interpretation import ProjectObservationKey, TypeNode, TypeTaxonomy
from celltraj2.trajectory_representations import (
    TimebaseSpec, TrajectoryWindowResult, TrajectoryWindowSpec,
    build_trajectory_windows, resolve_physical_time,
)


class TrajectoryRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.parent, self.leaf, self.other = (str(uuid4()) for _ in range(3))
        self.taxonomy = TypeTaxonomy(str(uuid4()), "Types", (
            TypeNode(self.parent, "Immune", "immune", "#112233"),
            TypeNode(self.leaf, "T cell", "t-cell", "#223344", self.parent),
            TypeNode(self.other, "Tumor", "tumor", "#334455"),
        ))
        project, dataset, roi = (str(uuid4()) for _ in range(3))
        self.keys = [ProjectObservationKey(project, dataset, roi, "cells", i + 1) for i in range(7)]
        self.rows = [{**key.to_dict(), "frame": i + 1, "time_s": float(i), "type_id": self.leaf,
                      "assignment_status": "assigned", "group_id": "roi-a", "split_id": "train",
                      "feature_valid": True} for i, key in enumerate(self.keys)]
        self.edges = [{"edge_id": f"edge-{i}", "source": source, "target": target, "accepted": True}
                      for i, (source, target) in enumerate(zip(self.keys, self.keys[1:]))]
        self.timebases = {self.keys[0].partition: TimebaseSpec("recorded_timestamps", provenance={"field": "time_s"})}

    def build(self, spec=None, *, rows=None, edges=None, **kwargs):
        return build_trajectory_windows(self.rows if rows is None else rows,
            self.edges if edges is None else edges, spec or TrajectoryWindowSpec(length=3, sample_interval_s=1, lag_s=1),
            taxonomy=self.taxonomy, timebases=kwargs.pop("timebases", self.timebases), **kwargs)

    def test_long_lag_expansion_has_a_cumulative_budget(self):
        self.build(TrajectoryWindowSpec(lag_s=1), max_expanded_members=100)
        with self.assertRaisesRegex(ValueError, 'expansion exceeds max_expanded_members'):
            self.build(TrajectoryWindowSpec(lag_s=6), max_expanded_members=100)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            self.build(max_expanded_members=0)

    def test_delay_anchors_members_pairs_and_full_spine_roundtrip(self):
        result = self.build()
        self.assertEqual(len(result.observations), 7)
        self.assertEqual(len(result.windows), 7)
        self.assertEqual([window["valid"] for window in result.windows], [False, False, True, True, True, True, True])
        self.assertEqual(result.windows[2]["members"], [key.to_dict() for key in self.keys[:3]])
        self.assertEqual(result.windows[2]["anchor"], self.keys[2].to_dict())
        self.assertEqual(result.windows[2]["times_s"], [0., 1., 2.])
        self.assertEqual(len([pair for pair in result.lag_pairs if pair["valid"]]), 4)
        self.assertEqual(result.lag_pairs[2]["elapsed_time"], 1.)
        self.assertEqual(result.lag_pairs[2]["leaf_type_id"], self.leaf)
        self.assertEqual(len(result.segments), 1)
        payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        self.assertEqual(TrajectoryWindowResult.from_dict(payload).to_dict(), result.to_dict())

    def test_determinism_source_order_and_overlapping_window_dependence(self):
        first = self.build()
        second = self.build(rows=list(reversed(self.rows)), edges=list(reversed(self.edges)))
        by_anchor = lambda result: {row["anchor"]["observation_id"]: row["window_id"] for row in result.windows}
        self.assertEqual(by_anchor(first), by_anchor(second))
        self.assertEqual(second.windows[0]["anchor"], self.keys[-1].to_dict())
        self.assertEqual(first.windows[3]["members"][:2], first.windows[2]["members"][1:])
        self.assertEqual(first.windows[3]["split_id"], "train")
        stride = self.build(TrajectoryWindowSpec(length=3, sample_interval_s=1, stride=2))
        self.assertEqual([row["anchor"]["observation_id"] for row in stride.windows if row["valid"]], [3, 5, 7])
        self.assertEqual(stride.windows[3]["exclusion_reason"], "stride_excluded")

    def test_declared_interval_and_recorded_preference_do_not_invent_seconds(self):
        timebase = TimebaseSpec("declared_interval", unit="min", interval=2, frame_origin=1,
                                provenance={"source": "acquisition.time_interval_min"})
        self.assertEqual(resolve_physical_time({"frame": 4}, timebase), (360., ""))
        self.assertEqual(resolve_physical_time({"frame": 0}, timebase), (None, "frame_before_time_origin"))
        zero_based = replace(timebase, frame_origin=0)
        self.assertEqual(resolve_physical_time({"frame": 0}, zero_based), (0., ""))
        recorded = TimebaseSpec("recorded_timestamps", unit="ms", interval=100)
        self.assertEqual(resolve_physical_time({"physical_time": 1200}, recorded), (1.2, ""))
        self.assertEqual(resolve_physical_time({"time_s": 7, "frame": 100}, recorded), (7., ""))
        self.assertEqual(resolve_physical_time({"frame": 100}, recorded), (None, "missing_recorded_time"))
        self.assertEqual(resolve_physical_time({"physical_time": 1e308},
            TimebaseSpec("recorded_timestamps", unit="h")), (None, "nonfinite_physical_time"))
        with self.assertRaisesRegex(ValueError, "frames are not seconds"):
            TimebaseSpec("declared_interval", unit="frame", interval=1)

    def test_snapshot_static_validity_survives_missing_time_but_delays_and_pairs_do_not(self):
        static = self.build(TrajectoryWindowSpec(), timebases={})
        self.assertTrue(all(row["valid"] and row["time_s"] is None for row in static.observations))
        self.assertTrue(all(window["valid"] for window in static.windows))
        temporal = self.build(timebases={})
        self.assertTrue(all(not window["valid"] for window in temporal.windows))
        self.assertTrue(all(not pair["valid"] for pair in temporal.lag_pairs))
        self.assertEqual(temporal.windows[2]["exclusion_reason"], "timebase_unavailable")
        self.assertEqual(temporal.windows[2]["members"], [self.keys[2].to_dict()])

    def test_physical_tolerance_no_interpolation_and_complete_intermediate_segment(self):
        rows = [dict(row) for row in self.rows]
        rows[2]["time_s"] += .02
        strict = self.build(rows=rows)
        self.assertFalse(strict.windows[2]["valid"])
        tolerant = self.build(TrajectoryWindowSpec(length=3, sample_interval_s=1, tolerance_s=.05), rows=rows)
        self.assertTrue(tolerant.windows[2]["valid"])
        subsampled = self.build(TrajectoryWindowSpec(length=3, sample_interval_s=2))
        self.assertEqual(subsampled.windows[4]["members"], [self.keys[i].to_dict() for i in (0, 2, 4)])
        self.assertEqual(subsampled.windows[4]["segment"], [key.to_dict() for key in self.keys[:5]])
        self.assertEqual(len(subsampled.windows[4]["accepted_edge_ids"]), 4)
        rows[1]["feature_valid"] = False
        blocked = self.build(TrajectoryWindowSpec(length=3, sample_interval_s=2), rows=rows)
        self.assertFalse(blocked.windows[4]["valid"])

    def test_branch_merge_and_scope_boundary_do_not_disappear_after_filtering(self):
        fork_key = replace(self.keys[3], roi_uuid=str(uuid4()))
        branched = self.build(edges=[*self.edges, {"edge_id": "branch-outside", "source": self.keys[2],
                                                  "target": fork_key, "accepted": True}])
        self.assertTrue(any("branch_or_merge" in row["exclusion_reasons"] for row in branched.edge_reviews))
        self.assertFalse(branched.windows[3]["valid"])
        merged = self.build(edges=[*self.edges, {"edge_id": "merge", "source": self.keys[0],
                                                "target": self.keys[3], "accepted": True}])
        self.assertFalse(merged.windows[4]["valid"])
        self.assertGreater(len(merged.segments), 1)

    def test_gaps_type_switches_split_groups_censoring_and_required_data_stop_paths(self):
        cases = (("frame", 10, "frame_gap_or_reverse"), ("type_id", self.other, "type_conflict"),
                 ("split_id", "test", "cross_split_id"), ("group_id", "another-roi", "cross_group_id"),
                 ("censoring_barrier", True, "censoring_barrier"), ("feature_valid", False, "ineligible_edge_member"))
        for field, value, reason in cases:
            with self.subTest(field=field):
                rows = [dict(row) for row in self.rows]
                rows[2][field] = value
                result = self.build(rows=rows)
                self.assertTrue(any(reason in review["exclusion_reasons"] for review in result.edge_reviews))
                self.assertFalse(result.windows[3]["valid"])
                self.assertFalse(result.lag_pairs[2]["valid"])

    def test_parent_only_windows_can_be_shared_but_pairs_require_exact_leaf(self):
        rows = [{**row, "type_id": self.parent} for row in self.rows]
        result = self.build(rows=rows)
        self.assertTrue(result.windows[3]["valid"])
        self.assertTrue(all(not pair["valid"] and pair["leaf_type_id"] is None for pair in result.lag_pairs))
        with self.assertRaisesRegex(ValueError, "exact Type leaf"):
            self.build(kinetic_leaf_id=self.parent)
        leaf_scope = self.build(kinetic_leaf_id=self.other)
        self.assertEqual(leaf_scope.windows, self.build().windows)
        self.assertTrue(all(not pair["valid"] for pair in leaf_scope.lag_pairs))

    def test_duplicate_numeric_ids_across_files_never_merge_and_cross_file_edges_stop(self):
        other_roi = str(uuid4())
        other_keys = [replace(key, roi_uuid=other_roi) for key in self.keys]
        other_rows = [{**row, **key.to_dict(), "group_id": "roi-b", "type_id": self.other}
                      for row, key in zip(self.rows, other_keys)]
        other_edges = [{"edge_id": "second-" + edge["edge_id"], "source": other_keys[i],
                        "target": other_keys[i+1], "accepted": True} for i, edge in enumerate(self.edges)]
        result = self.build(rows=[*self.rows, *other_rows], edges=[*self.edges, *other_edges],
            timebases={**self.timebases, other_keys[0].partition: TimebaseSpec("recorded_timestamps")})
        self.assertEqual(len(result.observations), 14)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual({pair["leaf_type_id"] for pair in result.lag_pairs if pair["valid"]}, {self.leaf, self.other})
        cross = self.build(rows=[*self.rows, *other_rows], edges=[*self.edges, *other_edges,
            {"edge_id": "cross", "source": self.keys[-1], "target": other_keys[0], "accepted": True}],
            timebases={**self.timebases, other_keys[0].partition: TimebaseSpec("recorded_timestamps")})
        self.assertTrue(any("cross_file_edge" in edge["exclusion_reasons"] for edge in cross.edge_reviews))

    def test_specs_invalid_edges_cancellation_and_time_scope_are_explicit(self):
        for kwargs in ({"length": 0}, {"length": 2}, {"lag_s": -1}, {"padding": "zero"},
                       {"mode": "retrospective"}, {"sample_interval_s": 1, "tolerance_s": .6}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrajectoryWindowSpec(**kwargs)
        with self.assertRaisesRegex(ValueError, "Duplicate project"):
            self.build(rows=[*self.rows, self.rows[0]])
        with self.assertRaisesRegex(ValueError, "unique stable edge"):
            self.build(edges=[*self.edges, self.edges[0]])
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.build(cancelled=lambda: True)
        rejected = self.build(edges=[{**edge, "accepted": False} for edge in self.edges])
        self.assertFalse(any(window["valid"] for window in rejected.windows))
        bounded = self.build(TrajectoryWindowSpec(length=2, sample_interval_s=1, time_start_s=2, time_stop_s=4))
        self.assertEqual([w["anchor"]["observation_id"] for w in bounded.windows if w["valid"]], [4, 5])

    def test_missing_group_split_or_recorded_time_never_silently_crosses(self):
        for field in ("split_id", "group_id", "time_s"):
            rows = [dict(row) for row in self.rows]
            rows[2].pop(field)
            result = self.build(rows=rows)
            self.assertFalse(result.windows[3]["valid"])
            self.assertFalse(result.lag_pairs[2]["valid"])

    def test_saved_window_or_pair_content_cannot_change_under_the_same_identity(self):
        payload = self.build().to_dict()
        payload["windows"][2]["times_s"][0] = -1
        with self.assertRaisesRegex(ValueError, "deterministic identity"):
            TrajectoryWindowResult.from_dict(payload)
        payload = self.build().to_dict()
        payload["lag_pairs"][2]["leaf_type_id"] = self.other
        with self.assertRaisesRegex(ValueError, "resolved exact leaf"):
            TrajectoryWindowResult.from_dict(payload)
        payload = self.build().to_dict()
        payload["segments"][0]["times_s"][0] = 500
        with self.assertRaisesRegex(ValueError, "Segment times"):
            TrajectoryWindowResult.from_dict(payload)
        payload = self.build().to_dict()
        payload["segments"] = []
        with self.assertRaisesRegex(ValueError, "partition the full observation spine"):
            TrajectoryWindowResult.from_dict(payload)

    def test_decimal_timestamps_do_not_create_float_roundoff_gaps(self):
        rows = [{**row, "time_s": i * .1} for i, row in enumerate(self.rows)]
        result = self.build(TrajectoryWindowSpec(length=3, sample_interval_s=.1), rows=rows)
        self.assertTrue(all(window["valid"] for window in result.windows[2:]))


if __name__ == "__main__":
    unittest.main()
