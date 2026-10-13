"""Analytic and independent small-array checks of temporal estimands."""

import threading
import unittest

import numpy as np

from celltraj2.temporal_correlations import (
    TemporalCorrelationCancelled,
    compute_temporal_curve,
)
from celltraj2.tracking import SparseAdjacency, TrackGraph
from celltraj2.trajectory_index import GraphIndex, Runs


def runs_for(*rows):
    lengths = [len(row) for row in rows]
    order = np.concatenate(rows).astype(np.int64) if rows else np.array([], dtype=np.int64)
    return Runs(order, np.r_[0, np.cumsum(lengths)].astype(np.int64))


def graph(parents):
    n = len(parents)
    edges = sorted((p - 1, c) for c, p in enumerate(parents) if p)
    counts = np.bincount([p for p, _ in edges], minlength=n)
    assignments = np.zeros(n, dtype=[("observation_id", "i8"), ("parent_observation_id", "i8")])
    assignments["observation_id"] = np.arange(1, n + 1)
    assignments["parent_observation_id"] = parents
    return TrackGraph(
        adjacency=SparseAdjacency(
            indptr=np.r_[0, np.cumsum(counts)],
            indices=np.array([c for _, c in edges], dtype="i8"),
            data=np.ones(len(edges)), shape=(n, n),
        ), links=np.array([]), assignments=assignments,
    )


class TemporalCorrelationTests(unittest.TestCase):
    def test_constant_velocity_msd_uses_all_origins(self):
        time = np.arange(5.)
        result = compute_temporal_curve(
            runs_for(np.arange(5)), time, lag_edges=[-.25, .25, 1.25, 2.25, 3.25, 4.25],
            method="msd", positions=np.c_[2 * time, -time], min_pairs=1,
        )
        np.testing.assert_allclose(result["values"], [0, 5, 20, 45, 80])
        np.testing.assert_equal(result["pair_counts"], [5, 4, 3, 2, 1])
        np.testing.assert_equal(result["track_counts"], 1)
        np.testing.assert_allclose(result["mean_lags"], range(5))
        self.assertEqual(result["normalization"], "raw")

    def test_signed_shifted_cross_and_swap_identity(self):
        a = np.array([3., -1, 2, 0, 5, -4, 1, 8])
        b = np.r_[np.nan, a[:-1]]
        common = dict(runs=runs_for(np.arange(len(a))), times=np.arange(len(a)),
                      lag_edges=np.arange(-3.5, 4.5), method="crosscorrelation")
        ab = compute_temporal_curve(**common, feature_a=a, feature_b=b)
        ba = compute_temporal_curve(**common, feature_a=b, feature_b=a)
        self.assertAlmostEqual(ab["values"][4], 1.)
        self.assertLess(ab["values"][2], .9)
        np.testing.assert_allclose(ab["values"], ba["values"][::-1])
        np.testing.assert_equal(ab["pair_counts"], ba["pair_counts"][::-1])

    def test_zero_lag_and_constant_normalized_undefined(self):
        common = dict(runs=runs_for(np.arange(4)), times=np.arange(4), lag_edges=[-.1, .1])
        result = compute_temporal_curve(**common, feature_a=[1, 4, 2, 8])
        self.assertEqual(result["pair_counts"][0], 4)
        self.assertAlmostEqual(result["values"][0], 1.)
        fixed = compute_temporal_curve(**common, feature_a=[1, 4, 2, 8], normalization="fixed")
        self.assertAlmostEqual(fixed["values"][0], 1.)
        for normalization, reason in [("coefficient", "zero_variance"), ("fixed", "zero_reference_variance")]:
            result = compute_temporal_curve(**common, feature_a=np.ones(4), normalization=normalization)
            self.assertTrue(np.isnan(result["values"][0]))
            self.assertEqual(result["status"][0], reason)
        raw = compute_temporal_curve(**common, feature_a=np.ones(4), normalization="raw")
        self.assertEqual(raw["values"][0], 1.)

    def test_feature_nan_removes_endpoints_without_compressing_time(self):
        result = compute_temporal_curve(
            runs_for([0, 1, 2]), [0, 1, 2], lag_edges=[.5, 1.5, 2.5],
            feature_a=[1., np.nan, 3.], normalization="raw", min_pairs=1,
        )
        np.testing.assert_equal(result["pair_counts"], [0, 1])
        self.assertTrue(np.isnan(result["values"][0]))
        self.assertEqual(result["values"][1], 3.)
        self.assertEqual(result["mean_lags"][1], 2.)

    def test_finite_vector_pairs_use_finite_denominator(self):
        result = compute_temporal_curve(
            runs_for([0, 1, 2]), [0, 1, 2], lag_edges=[.5, 1.5],
            method="crosscorrelation", feature_a=[[1, 2], [3, 4], [5, 6]],
            feature_b=[[0, 1], [np.nan, 8], [2, 3]], normalization="raw", min_pairs=1,
        )
        self.assertEqual(result["pair_counts"][0], 1)
        self.assertEqual(result["values"][0], 18.)

    def test_min_pairs_preserves_support_and_reason(self):
        result = compute_temporal_curve(
            runs_for([0, 1]), [0, 1], lag_edges=[.5, 1.5, 2.5], feature_a=[2, 3], normalization="raw",
        )
        np.testing.assert_equal(result["status"], ["insufficient_pairs", "no_pairs"])
        np.testing.assert_equal(result["pair_counts"], [1, 0])
        np.testing.assert_equal(result["track_counts"], [1, 0])
        self.assertTrue(np.isnan(result["values"]).all())

    def test_empty_runs_and_all_nan(self):
        for runs, times, features in [(runs_for(), [], []), (runs_for([0, 1]), [0, 1], [np.nan, np.inf])]:
            result = compute_temporal_curve(runs, times, lag_edges=[-.5, .5, 1.5], feature_a=features)
            self.assertTrue(np.isnan(result["values"]).all())
            np.testing.assert_equal(result["pair_counts"], 0)
            np.testing.assert_equal(result["status"], "no_pairs")

    def test_graph_project_division_and_scope_gaps_are_not_joined(self):
        index = GraphIndex.from_graph(graph([0, 1, 2, 2, 3, 4]))
        times = np.array([0, 1, 2, 2, 3, 3])
        runs = index.project(np.arange(6), times, np.ones(6, dtype=bool))
        result = compute_temporal_curve(runs, times, lag_edges=[.5, 1.5, 2.5],
                                        feature_a=np.arange(6), normalization="raw", min_pairs=1)
        np.testing.assert_equal(result["pair_counts"], [3, 0])
        np.testing.assert_equal(result["track_counts"], [3, 0])
        line = GraphIndex.from_graph(graph([0, 1, 2, 3, 4]))
        runs = line.project(np.arange(5), np.arange(5), [True, True, False, True, True])
        result = compute_temporal_curve(runs, np.arange(5), lag_edges=[.5, 1.5, 4.5],
                                        feature_a=np.arange(5), normalization="raw", min_pairs=1)
        np.testing.assert_equal(result["pair_counts"], [2, 0])

    def test_irregular_time_exact_bins_and_last_edge(self):
        result = compute_temporal_curve(runs_for([0, 1, 2]), [0, .4, 2],
                                        lag_edges=[0, .5, 1, 2], method="msd", positions=[0, .4, 2], min_pairs=1)
        np.testing.assert_equal(result["pair_counts"], [4, 0, 2])
        np.testing.assert_allclose(result["values"][[0, 2]], [.16 / 4, (1.6 ** 2 + 2 ** 2) / 2])
        np.testing.assert_allclose(result["mean_lags"][[0, 2]], [.1, 1.8])

    def test_invalid_times_rejected_but_unselected_unknown_is_allowed(self):
        for times in ([0, 0, 1], [0, 2, 1], [0, np.nan, 2], [0, np.inf, 2]):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                compute_temporal_curve(runs_for([0, 1, 2]), times, lag_edges=[0, 2], feature_a=[1, 2, 3])
        result = compute_temporal_curve(runs_for([0, 1]), [0, 1, np.nan], lag_edges=[-.1, .1], feature_a=[1, 2, 3])
        self.assertAlmostEqual(result["values"][0], 1.)

    def test_vector_shared_rotation_invariance(self):
        rng = np.random.default_rng(123)
        a, b = rng.normal(size=(2, 12, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        kwargs = dict(runs=runs_for(np.arange(12)), times=np.arange(12), lag_edges=[-.5, .5, 2.5, 4.5], method="crosscorrelation")
        for normalization in ("raw", "covariance", "fixed", "coefficient"):
            original = compute_temporal_curve(**kwargs, feature_a=a, feature_b=b, normalization=normalization)
            rotated = compute_temporal_curve(**kwargs, feature_a=a @ rotation, feature_b=b @ rotation, normalization=normalization)
            np.testing.assert_allclose(original["values"], rotated["values"], atol=1e-13)

    def test_roi_and_track_centering_differ_and_track_coefficient_is_pooled(self):
        a = np.array([0., 1., 100., 101.])
        b = np.array([0., 1., -100., -99.])
        common = dict(runs=runs_for([0, 1], [2, 3]), times=[0, 1, 0, 1],
                      lag_edges=[-.1, .1], method="crosscorrelation", feature_a=a, feature_b=b)
        roi = compute_temporal_curve(**common)
        track = compute_temporal_curve(**common, centering="track")
        self.assertLess(roi["values"][0], -.99)
        self.assertAlmostEqual(track["values"][0], 1.)
        track_fixed = compute_temporal_curve(**common, centering="track", normalization="fixed")
        self.assertAlmostEqual(track_fixed["values"][0], 1.)
        self.assertAlmostEqual(track_fixed["reference_variance_a"], .25)

    def test_fixed_reference_not_lag_local_and_not_clipped(self):
        common = dict(runs=runs_for(np.arange(8)), times=np.arange(8), lag_edges=[6.5, 7.5], feature_a=[5, 0, 0, 0, 0, 0, 0, 5], min_pairs=1)
        fixed = compute_temporal_curve(**common, normalization="fixed")
        local = compute_temporal_curve(**common)
        self.assertAlmostEqual(fixed["values"][0], 3.)
        self.assertEqual(local["status"][0], "zero_variance")

    def test_stable_lag_local_coefficient_on_large_baselines(self):
        a = 1e12 + np.arange(10.)
        b = -1e12 + 2 * np.arange(10.)
        result = compute_temporal_curve(runs_for(np.arange(10)), np.arange(10), lag_edges=[1.5, 2.5],
                                        method="crosscorrelation", feature_a=a, feature_b=b)
        self.assertAlmostEqual(result["values"][0], 1.)

    def test_scalar_pairwise_oracle_with_changing_support(self):
        a = np.array([1., 4, np.nan, 2, 9, -1, 3])
        b = np.array([np.nan, 1, 6, 5, 3, 7, 2])
        times = np.array([0., .3, 1.2, 2.5, 0., 1., 2.2])
        runs = runs_for([0, 1, 2, 3], [4, 5, 6])
        edges = np.array([-3., -.5, .5, 3.])
        for normalization in ("raw", "covariance", "coefficient", "fixed"):
            result = compute_temporal_curve(runs, times, lag_edges=edges, method="crosscorrelation",
                                            feature_a=a, feature_b=b, normalization=normalization, min_pairs=1)
            pairs = [[] for _ in range(3)]
            for rows in ([0, 1, 2, 3], [4, 5, 6]):
                for i in rows:
                    for j in rows:
                        if not np.isfinite(a[i]) or not np.isfinite(b[j]):
                            continue
                        lag = times[j] - times[i]
                        for k in range(3):
                            if edges[k] <= lag < edges[k + 1]:
                                pairs[k].append((a[i], b[j]))
            for k, pair in enumerate(pairs):
                x, y = np.array(pair).T
                self.assertEqual(result["pair_counts"][k], len(x))
                if normalization == "raw":
                    expected = np.mean(x * y)
                elif normalization == "coefficient":
                    expected = np.corrcoef(x, y)[0, 1]
                else:
                    expected = np.mean((x - np.nanmean(a)) * (y - np.nanmean(b)))
                    if normalization == "fixed":
                        expected /= np.nanstd(a) * np.nanstd(b)
                self.assertAlmostEqual(result["values"][k], expected)

    def test_zero_motion_raw_msd_is_valid(self):
        result = compute_temporal_curve(runs_for(np.arange(5)), np.arange(5), lag_edges=[.5, 2.5],
                                        method="msd", positions=np.zeros((5, 3)))
        self.assertEqual(result["values"][0], 0.)
        self.assertEqual(result["pair_counts"][0], 7)

    def test_progress_and_cancellation_during_run(self):
        event = threading.Event()
        messages = []
        def progress(message):
            messages.append(message)
            if "Temporal run" in message:
                event.set()
        with self.assertRaises(TemporalCorrelationCancelled):
            compute_temporal_curve(runs_for(np.arange(500)), np.arange(500), lag_edges=[0, 499],
                                   feature_a=np.arange(500), progress=progress, cancelled=event)
        self.assertTrue(messages)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            compute_temporal_curve(runs_for(), [], lag_edges=[0, 1], feature_a=[], cancelled=lambda: True)

    def test_invalid_contracts(self):
        common = dict(runs=runs_for([0, 1]), times=[0, 1], lag_edges=[0, 1], feature_a=[1, 2])
        for override in ({"lag_edges": [0, 0]}, {"lag_edges": [0, np.inf]},
                         {"min_pairs": 0}, {"min_pairs": True}, {"normalization": "unknown"},
                         {"feature_a": [[1, 2]]}, {"runs": runs_for([0, 1], [1])}):
            with self.assertRaises(ValueError):
                compute_temporal_curve(**(common | override))
        with self.assertRaisesRegex(ValueError, "components"):
            compute_temporal_curve(**(common | {"method": "crosscorrelation", "feature_b": [[1, 2], [3, 4]]}))


if __name__ == "__main__":
    unittest.main()
