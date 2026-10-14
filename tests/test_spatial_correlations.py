"""Analytic motion surfaces, finite support, topology, and bounded work checks."""

import threading
import unittest
from unittest.mock import patch

import numpy as np

from celltraj2.spatial_correlations import SpatialCorrelationCancelled, compute_motion_surface
from celltraj2.trajectory_index import GraphIndex, Runs
from celltraj2.tracking import SparseAdjacency, TrackGraph


def runs_for(*rows):
    order = np.concatenate(rows).astype(np.int64) if rows else np.array([], dtype=np.int64)
    return Runs(order, np.r_[0, np.cumsum([len(x) for x in rows])].astype(np.int64))


def two_motions(a=(1., 2., 3.), b=(4., 5., 6.)):
    a, b = np.asarray(a), np.asarray(b)
    source = np.zeros(len(a))
    receiver = np.zeros(len(a))
    receiver[0] = 10
    return dict(
        runs=runs_for([0, 1], [2, 3]), times=np.array([0., 1., 0., 1.]),
        positions=np.array([source, source + a, receiver, receiver + b]),
        source_mask=np.array([True, True, False, False]),
        receiver_mask=np.array([False, False, True, True]),
        displacement_interval=1, distance_edges=[0, 20], lag_edges=[-.1, .1], min_pairs=1,
    )


def graph(parents):
    n = len(parents)
    edges = sorted((p - 1, c) for c, p in enumerate(parents) if p)
    counts = np.bincount([p for p, _ in edges], minlength=n)
    assignments = np.zeros(n, dtype=[("observation_id", "i8"), ("parent_observation_id", "i8")])
    assignments["observation_id"] = np.arange(1, n + 1)
    assignments["parent_observation_id"] = parents
    return TrackGraph(
        adjacency=SparseAdjacency(indptr=np.r_[0, np.cumsum(counts)],
                                  indices=np.array([c for _, c in edges], dtype="i8"),
                                  data=np.ones(len(edges)), shape=(n, n)),
        links=np.array([]), assignments=assignments,
    )


class SpatialCorrelationTests(unittest.TestCase):
    def test_radial_singleton_source_forward_motion_sign_and_support(self):
        args = dict(runs=runs_for([0], [1, 2]), times=[0, 0, 1], positions=[[0, 0], [10, 0], [8, 0]],
                    displacement_interval=1, distance_edges=[0, 20], lag_edges=[-.1, .1], min_pairs=1,
                    source_mask=np.array([True, False, False]), receiver_mask=np.array([False, True, True]))
        toward = compute_motion_surface(**args)
        away = compute_motion_surface(**args, radial_sign="away")
        self.assertEqual(toward["values"][0, 0], 2)
        self.assertEqual(away["values"][0, 0], -2)
        for name in ("pair_counts", "source_counts", "receiver_counts", "source_track_counts", "receiver_track_counts"):
            self.assertEqual(toward[name][0, 0], 1)
        self.assertEqual(toward["mean_lags"][0, 0], 0)
        self.assertEqual(toward["mean_distances"][0, 0], 10)
        self.assertEqual(toward["mean_receiver_intervals"][0, 0], 1)
        self.assertTrue(np.isnan(toward["mean_source_intervals"][0, 0]))
        relative = compute_motion_surface(**args, method="relative_radial")
        self.assertEqual(relative["pair_counts"][0, 0], 0)

    def test_relative_radial_subtracts_source_motion(self):
        args = two_motions(a=(1., 0.), b=(-2., 0.))
        self.assertEqual(compute_motion_surface(**args)["values"][0, 0], 2)
        relative = compute_motion_surface(**args, method="relative_radial")
        self.assertEqual(relative["values"][0, 0], 3)
        self.assertEqual(relative["mean_source_intervals"][0, 0], 1)
        self.assertEqual(compute_motion_surface(**args, method="relative_radial", radial_sign="away")["values"][0, 0], -3)

    def test_dot_longitudinal_transverse_sum_and_per_component_mean(self):
        args = two_motions()
        dot = compute_motion_surface(**args, method="full_dot")
        longitudinal = compute_motion_surface(**args, method="longitudinal")
        transverse = compute_motion_surface(**args, method="transverse")
        mean = compute_motion_surface(**args, method="transverse", transverse_mode="mean")
        self.assertEqual(dot["values"][0, 0], 32)
        self.assertEqual(longitudinal["values"][0, 0], 4)
        self.assertEqual(transverse["values"][0, 0], 28)
        self.assertAlmostEqual(mean["values"][0, 0], 14)
        np.testing.assert_allclose(dot["values"], longitudinal["values"] + transverse["values"])

    def test_source_time_and_cross_time_geometry_and_signed_lags(self):
        args = dict(runs=runs_for([0], [1, 2, 3, 4]), times=[0, -1, 0, 1, 2],
                    positions=[[0, 0], [12, 0], [10, 0], [8, 0], [6, 0]], displacement_interval=1,
                    distance_edges=[0, 9, 11, 13], lag_edges=[-1.5, -.5, .5, 1.5], min_pairs=1,
                    source_mask=np.array([True, False, False, False, False]),
                    receiver_mask=np.array([False, True, True, True, True]))
        cross = compute_motion_surface(**args)
        initial = compute_motion_surface(**args, distance_convention="source_time")
        np.testing.assert_equal(cross["pair_counts"], [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        np.testing.assert_equal(initial["pair_counts"], [[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        np.testing.assert_allclose(initial["values"][:, 1], 2)
        np.testing.assert_allclose(initial["mean_lags"][:, 1], [-1, 0, 1])

    def test_source_time_requires_exact_reference_in_receiver_run(self):
        args = dict(runs=runs_for([0], [1, 2]), times=[0, .01, 1.01], positions=[[0, 0], [10, 0], [9, 0]],
                    displacement_interval=1, distance_edges=[0, 20], lag_edges=[-.1, .1], min_pairs=1,
                    source_mask=np.array([True, False, False]), receiver_mask=np.array([False, True, True]))
        cross = compute_motion_surface(**args)
        initial = compute_motion_surface(**args, distance_convention="source_time")
        self.assertEqual(cross["pair_counts"][0, 0], 1)
        self.assertEqual(initial["pair_counts"][0, 0], 0)
        self.assertEqual(initial["exclusion_counts"]["pair_missing_source_time_reference"], 1)

    def test_unique_nearest_endpoint_ties_and_recorded_realized_interval(self):
        common = dict(runs=runs_for([0], [1, 2, 3]), times=[0, 0, .8, 1.3],
                      positions=[[0, 0], [10, 0], [9.2, 0], [8.7, 0]], displacement_interval=1,
                      distance_edges=[0, 20], lag_edges=[-.01, .01], min_pairs=1,
                      source_mask=np.array([True, False, False, False]),
                      receiver_mask=np.array([False, True, True, True]))
        exact = compute_motion_surface(**common)
        nearest = compute_motion_surface(**common, interval_tolerance=.21)
        self.assertEqual(exact["pair_counts"][0, 0], 0)
        self.assertAlmostEqual(nearest["values"][0, 0], .8)
        self.assertAlmostEqual(nearest["mean_receiver_intervals"][0, 0], .8)
        tied = compute_motion_surface(**(common | {"times": [0, 0, .8, 1.2]}), interval_tolerance=.21)
        self.assertEqual(tied["pair_counts"][0, 0], 0)
        self.assertEqual(tied["exclusion_counts"]["receiver_ambiguous_endpoint"], 1)

    def test_decimal_clock_roundoff_does_not_drop_regular_intervals(self):
        common = dict(runs=runs_for([0], [1, 2]), times=[.7, .7, .8], positions=[[0, 0], [10, 0], [9, 0]],
                      displacement_interval=.1, distance_edges=[0, 20], lag_edges=[-.01, .01], min_pairs=1,
                      source_mask=np.array([True, False, False]), receiver_mask=np.array([False, True, True]))
        result = compute_motion_surface(**common)
        self.assertEqual(result["pair_counts"][0, 0], 1)
        missing = compute_motion_surface(**(common | {"times": [.7, .7, .800000001]}))
        self.assertEqual(missing["pair_counts"][0, 0], 0)

    def test_interval_tolerance_still_selects_later_not_origin_endpoint(self):
        common = dict(runs=runs_for([0], [1, 2]), times=[1, 1, 2], positions=[[0, 0], [10, 0], [9, 0]],
                      displacement_interval=1e-20, interval_tolerance=1, distance_edges=[0, 20],
                      lag_edges=[-.01, .01], min_pairs=1, source_mask=np.array([True, False, False]),
                      receiver_mask=np.array([False, True, True]))
        result = compute_motion_surface(**common)
        self.assertEqual(result["values"][0, 0], 1)
        self.assertEqual(result["mean_receiver_intervals"][0, 0], 1)

    def test_cohort_membership_required_through_interior_motion_path(self):
        common = dict(runs=runs_for([0], [1, 2, 3]), times=[0, 0, 1, 2],
                      positions=[[0, 0], [10, 0], [9, 0], [8, 0]], displacement_interval=2,
                      distance_edges=[0, 20], lag_edges=[-.1, .1], min_pairs=1,
                      source_mask=np.array([True, False, False, False]),
                      receiver_mask=np.array([False, True, False, True]))
        result = compute_motion_surface(**common)
        self.assertEqual(result["pair_counts"][0, 0], 0)
        self.assertEqual(result["exclusion_counts"]["receiver_cohort_crossing"], 1)

    def test_radial_source_requires_only_anchor_membership(self):
        common = two_motions(a=(1., 0.), b=(-2., 0.))
        common["source_mask"] = np.array([True, False, False, False])
        self.assertEqual(compute_motion_surface(**common)["pair_counts"][0, 0], 1)
        two = compute_motion_surface(**common, method="full_dot")
        self.assertEqual(two["pair_counts"][0, 0], 0)
        self.assertEqual(two["exclusion_counts"]["source_cohort_crossing"], 1)

    def test_source_time_cohort_path_extends_back_to_reference(self):
        common = dict(runs=runs_for([0], [1, 2, 3, 4]), times=[0, 0, 1, 2, 3],
                      positions=[[0, 0], [10, 0], [9, 0], [8, 0], [7, 0]], displacement_interval=1,
                      distance_edges=[0, 20], lag_edges=[1.5, 2.5], min_pairs=1,
                      source_mask=np.array([True, False, False, False, False]),
                      receiver_mask=np.array([False, True, False, True, True]))
        self.assertEqual(compute_motion_surface(**common)["pair_counts"][0, 0], 1)
        source_time = compute_motion_surface(**common, distance_convention="source_time")
        self.assertEqual(source_time["pair_counts"][0, 0], 0)
        self.assertEqual(source_time["exclusion_counts"]["pair_receiver_reference_cohort_crossing"], 1)

    def test_same_cell_uses_original_ids_across_split_runs(self):
        args = dict(runs=runs_for([0], [1, 2]), times=[0, 1, 2], positions=[[0, 0], [10, 0], [8, 0]],
                    displacement_interval=1, distance_edges=[0, 20], lag_edges=[.5, 1.5], min_pairs=1,
                    source_mask=np.array([True, False, False]), receiver_mask=np.array([False, True, True]))
        fallback = compute_motion_surface(**args)
        same = compute_motion_surface(**args, cell_ids=np.array([7, 7, 7]))
        include = compute_motion_surface(**args, cell_ids=np.array([7, 7, 7]), exclude_same_cell=False)
        self.assertEqual(fallback["pair_counts"][0, 0], 1)
        self.assertEqual(same["pair_counts"][0, 0], 0)
        self.assertEqual(include["pair_counts"][0, 0], 1)
        self.assertEqual(same["exclusion_counts"]["pair_same_cell"], 1)
        self.assertEqual(include["track_count_definition"], "provided_cell_ids")

    def test_lineage_policy_is_independent_of_cell_identity(self):
        args = two_motions(a=(1., 0.), b=(-2., 0.))
        ids = np.array([100, 100, 200, 200])
        lineage = np.array([5, 5, 5, 5])
        kept = compute_motion_surface(**args, cell_ids=ids, lineage_ids=lineage)
        removed = compute_motion_surface(**args, cell_ids=ids, lineage_ids=lineage, exclude_related_lineage=True)
        self.assertEqual(kept["pair_counts"][0, 0], 1)
        self.assertEqual(removed["pair_counts"][0, 0], 0)
        self.assertEqual(removed["exclusion_counts"]["pair_related_lineage"], 1)
        with self.assertRaisesRegex(ValueError, "requires lineage_ids"):
            compute_motion_surface(**args, exclude_related_lineage=True)

    def test_graph_gaps_and_divisions_do_not_supply_motion_endpoints(self):
        index = GraphIndex.from_graph(graph([0, 1, 2, 2, 3, 4]))
        times = np.array([0, 1, 2, 2, 3, 3])
        runs = index.project(np.arange(6), times, np.ones(6, dtype=bool))
        args = dict(runs=runs, times=times, positions=np.c_[np.arange(6.), np.zeros(6)],
                    displacement_interval=2, distance_edges=[0, 20], lag_edges=[-3, 3], min_pairs=1)
        result = compute_motion_surface(**args)
        self.assertEqual(result["eligible_receiver_origins"], 0)
        linear = GraphIndex.from_graph(graph([0, 1, 2, 3, 4]))
        runs = linear.project(np.arange(5), np.arange(5), [True, True, False, True, True])
        result = compute_motion_surface(**(args | {"runs": runs, "times": np.arange(5),
                                                 "positions": np.c_[np.arange(5.), np.zeros(5)]}))
        self.assertEqual(result["eligible_receiver_origins"], 0)

    def test_zero_separation_excluded_zero_motion_valid(self):
        common = two_motions(a=(0., 0.), b=(0., 0.))
        raw = compute_motion_surface(**common)
        self.assertEqual(raw["values"][0, 0], 0)
        constant = compute_motion_surface(**common, method="full_dot", normalization="coefficient")
        self.assertEqual(constant["status"][0, 0], "zero_variance")
        common["positions"] = np.zeros((4, 2))
        zero = compute_motion_surface(**common)
        self.assertEqual(zero["pair_counts"][0, 0], 0)
        self.assertEqual(zero["exclusion_counts"]["pair_zero_separation"], 1)

    def test_missing_motion_does_not_dilute_denominator(self):
        common = dict(runs=runs_for([0], [1, 2], [3, 4]), times=[0, 0, 1, 0, 1],
                      positions=[[0, 0], [10, 0], [8, 0], [12, 0], [np.nan, 0]],
                      displacement_interval=1, distance_edges=[0, 20], lag_edges=[-.1, .1], min_pairs=1,
                      source_mask=np.array([True, False, False, False, False]),
                      receiver_mask=np.array([False, True, True, True, True]))
        result = compute_motion_surface(**common)
        self.assertEqual(result["pair_counts"][0, 0], 1)
        self.assertEqual(result["values"][0, 0], 2)
        self.assertEqual(result["exclusion_counts"]["receiver_nonfinite_motion"], 1)

    def test_bin_edges_final_inclusive_and_sparse_reasons(self):
        args = two_motions(a=(1., 0.), b=(-2., 0.))
        args.update(distance_edges=[0, 5, 10], lag_edges=[-1, 0], min_pairs=2, method="full_dot")
        result = compute_motion_surface(**args)
        np.testing.assert_equal(result["pair_counts"], [[0, 1]])
        np.testing.assert_equal(result["status"], [["no_pairs", "insufficient_pairs"]])
        np.testing.assert_equal(result["receiver_track_counts"], [[0, 1]])
        self.assertTrue(np.isnan(result["values"]).all())

    def test_unique_support_counts_do_not_count_reused_origins_as_tracks(self):
        args = dict(runs=runs_for([0, 1, 2], [3, 4, 5]), times=[0, 1, 2, 0, 1, 2],
                    positions=[[0, 0], [0, 0], [0, 0], [10, 0], [9, 0], [8, 0]],
                    displacement_interval=1, distance_edges=[0, 20], lag_edges=[-2, 2], min_pairs=1,
                    source_mask=np.array([True, True, True, False, False, False]),
                    receiver_mask=np.array([False, False, False, True, True, True]))
        result = compute_motion_surface(**args)
        for name, expected in (("pair_counts", 6), ("source_counts", 3), ("receiver_counts", 2),
                               ("source_track_counts", 1), ("receiver_track_counts", 1)):
            self.assertEqual(result[name][0, 0], expected)
        self.assertEqual(result["values"][0, 0], 1)

    def test_vector_rotation_invariance_and_independent_pair_moment_oracle(self):
        rng = np.random.default_rng(941)
        tracks, clocks, xyz = [], [], []
        for cell in range(5):
            tracks.append(np.arange(4 * cell, 4 * cell + 4))
            clocks.extend([0, 1, 2, 3])
            xyz.extend(np.cumsum(rng.normal(size=(4, 3)), axis=0) + [10 * cell, 0, 0])
        xyz = np.array(xyz)
        args = dict(runs=runs_for(*tracks), times=clocks, positions=xyz, displacement_interval=1,
                    distance_edges=[0, 100], lag_edges=[-2.1, 2.1], min_pairs=1)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pairs = [(i, j) for cell, rows in enumerate(tracks) for i in rows[:-1]
                 for other, target in enumerate(tracks) if other != cell for j in target[:-1]]
        full_a = np.array([xyz[i + 1] - xyz[i] for i, j in pairs])
        full_b = np.array([xyz[j + 1] - xyz[j] for i, j in pairs])
        axes = np.array([xyz[i] - xyz[j] for i, j in pairs])
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        for method in ("full_dot", "longitudinal", "transverse"):
            a, b = full_a.copy(), full_b.copy()
            pa, pb = np.sum(a * axes, axis=1), np.sum(b * axes, axis=1)
            if method == "longitudinal":
                a, b = pa[:, None], pb[:, None]
            elif method == "transverse":
                a -= pa[:, None] * axes
                b -= pb[:, None] * axes
            ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
            for normalization in ("raw", "covariance", "coefficient"):
                original = compute_motion_surface(**args, method=method, normalization=normalization)
                rotated = compute_motion_surface(**(args | {"positions": xyz @ rotation}), method=method, normalization=normalization)
                expected = np.mean(np.sum(a * b, axis=1)) if normalization == "raw" else np.mean(np.sum(ac * bc, axis=1))
                if normalization == "coefficient":
                    expected /= np.sqrt(np.mean(np.sum(ac * ac, axis=1)) * np.mean(np.sum(bc * bc, axis=1)))
                self.assertEqual(original["pair_counts"][0, 0], len(pairs))
                self.assertAlmostEqual(original["values"][0, 0], expected)
                np.testing.assert_allclose(original["values"], rotated["values"], atol=1e-13)

    def test_shared_rotation_invariance_radial_and_relative(self):
        args = two_motions()
        rotation, _ = np.linalg.qr(np.random.default_rng(144).normal(size=(3, 3)))
        for method in ("radial_response", "relative_radial"):
            original = compute_motion_surface(**args, method=method)
            rotated = compute_motion_surface(**(args | {"positions": args["positions"] @ rotation}), method=method)
            np.testing.assert_allclose(original["values"], rotated["values"], atol=1e-13)

    def test_stable_covariance_on_large_motion_baselines_and_readonly_inputs(self):
        xyz, clocks, rows = [], [], []
        for i in range(6):
            xyz.extend([[0, 0], [1e12 + i, 0], [10, 0], [10 - 1e12 + 2 * i, 0]])
            clocks.extend([10 * i, 10 * i + 1, 10 * i, 10 * i + 1])
            rows.extend([[4 * i, 4 * i + 1], [4 * i + 2, 4 * i + 3]])
        xyz, clocks = np.array(xyz), np.array(clocks)
        source = np.tile([True, True, False, False], 6)
        receiver = ~source
        for arr in (xyz, clocks, source, receiver):
            arr.setflags(write=False)
        args = dict(runs=runs_for(*rows), times=clocks, positions=xyz, source_mask=source, receiver_mask=receiver,
                    displacement_interval=1, distance_edges=[0, 20], lag_edges=[-.1, .1], method="full_dot")
        covariance = compute_motion_surface(**args, normalization="covariance")
        coefficient = compute_motion_surface(**args, normalization="coefficient")
        self.assertEqual(coefficient["pair_counts"][0, 0], 6)
        self.assertAlmostEqual(coefficient["values"][0, 0], 1)
        self.assertAlmostEqual(covariance["values"][0, 0], 2 * np.var(np.arange(6)))
        self.assertFalse(xyz.flags.writeable)

    def test_empty_and_invalid_contracts(self):
        empty = compute_motion_surface(runs_for(), [], np.empty((0, 2)), displacement_interval=1,
                                       distance_edges=[0, 1], lag_edges=[0, 1])
        self.assertEqual(empty["status"][0, 0], "no_pairs")
        common = two_motions()
        for override in ({"distance_edges": [-1, 2]}, {"lag_edges": [0, 0]}, {"displacement_interval": 0},
                         {"interval_tolerance": -1}, {"method": "unknown"}, {"normalization": "coefficient"},
                         {"source_mask": [1, 1, 0, 0]}, {"min_pairs": True}, {"times": [0, 0, 0, 1]},
                         {"cell_ids": np.array([0, 1, 2, 2])}, {"lineage_ids": np.array([-1, -1, 0, 0])},
                         {"runs": runs_for([0, 1], [1, 2, 3])}, {"positions": np.empty((3, 2))}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                compute_motion_surface(**(common | override))
        with self.assertRaisesRegex(ValueError, "at least two"):
            compute_motion_surface(**(common | {"positions": np.ones((4, 1)), "method": "transverse"}))

    def test_grid_and_support_memory_caps(self):
        common = two_motions()
        with self.assertRaisesRegex(ValueError, "accumulator limit"):
            compute_motion_surface(**(common | {"distance_edges": np.arange(100002)}))
        with self.assertRaisesRegex(ValueError, "at least 3.*configured limit 2"):
            compute_motion_surface(**common, max_support_entries=2)

    def test_adjustable_support_budget_preserves_exact_results(self):
        from celltraj2.spatial_correlations import DEFAULT_MAX_SUPPORT_ENTRIES
        self.assertEqual(DEFAULT_MAX_SUPPORT_ENTRIES, 20_000_000)
        # One source-cell, receiver-cell and receiver-origin membership = 3.
        bounded = compute_motion_surface(**two_motions(), max_support_entries=3)
        unlimited = compute_motion_surface(**two_motions(), max_support_entries=0)
        for name in ('values', 'pair_counts', 'source_counts', 'receiver_counts',
                     'source_track_counts', 'receiver_track_counts', 'status'):
            np.testing.assert_equal(bounded[name], unlimited[name])
        for invalid in (-1, True, 2.5, None, float('inf')):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'max_support_entries'):
                compute_motion_surface(**two_motions(), max_support_entries=invalid)

    def test_receiver_pair_chunks_and_cooperative_cancellation(self):
        # More receivers than one pair chunk; no dense pair matrix is needed.
        receivers = 4300
        positions = np.zeros((1 + 2 * receivers, 2))
        positions[1::2, 0], positions[2::2, 0] = 10, 9
        times = np.r_[0, np.tile([0, 1], receivers)]
        common = dict(runs=runs_for([0], *np.arange(1, len(times)).reshape(-1, 2)), times=times,
                      positions=positions, source_mask=np.arange(len(times)) == 0,
                      receiver_mask=np.arange(len(times)) != 0, displacement_interval=1,
                      distance_edges=[0, 20], lag_edges=[-.1, .1])
        result = compute_motion_surface(**common)
        self.assertEqual(result["pair_counts"][0, 0], receivers)
        self.assertEqual(result["source_counts"][0, 0], 1)
        self.assertEqual(result["receiver_track_counts"][0, 0], receivers)
        self.assertEqual(result["values"][0, 0], 1)
        event = threading.Event()
        messages = []
        def progress(message):
            messages.append(message)
            if "Spatial motion pairs" in message:
                event.set()
        with self.assertRaisesRegex(SpatialCorrelationCancelled, "cancelled"):
            compute_motion_surface(**common, progress=progress, cancelled=event)
        self.assertTrue(any("Spatial motion pairs" in message for message in messages))
        from celltraj2 import spatial_correlations
        merge = spatial_correlations._grouped_moments
        event.clear()
        chunks = []
        def stop_after_chunk(*args):
            chunks.append(len(args[0]))
            result = merge(*args)
            event.set()
            return result
        with patch("celltraj2.spatial_correlations._grouped_moments", stop_after_chunk):
            with self.assertRaises(SpatialCorrelationCancelled):
                compute_motion_surface(**common, cancelled=event)
        self.assertEqual(len(chunks), 1)
        self.assertLess(chunks[0], receivers)


if __name__ == "__main__":
    unittest.main()
