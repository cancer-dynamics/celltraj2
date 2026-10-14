"""Independent analytic checks for scalar delayed MI and time-aware diagnostics."""
import threading
import unittest

import numpy as np

from celltraj2.mutual_information import (
    MutualInformationPairLimitExceeded, compute_delayed_mutual_information,
    estimate_mutual_information,
)
from celltraj2.temporal_correlations import TemporalCorrelationCancelled, compute_temporal_curve
from celltraj2.trajectory_index import Runs


def runs_for(*rows):
    return Runs(np.concatenate(rows).astype(np.int64), np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64))


def curve(a, b, **options):
    n = len(a)
    return compute_delayed_mutual_information(
        runs_for(np.arange(n)), np.arange(n), feature_a=a, feature_b=b,
        lag_edges=options.pop('lag_edges', [-.1, .1]), **options)


class MutualInformationTests(unittest.TestCase):
    def test_discrete_analytic_entropy_and_independence_in_bits(self):
        a = np.tile([0., 0., 1., 1.], 100)
        common = dict(kind_a='discrete', kind_b='discrete', units='bits')
        self.assertAlmostEqual(estimate_mutual_information(a, a, **common)['value'], 1.)
        independent = np.tile([0., 1., 0., 1.], 100)
        self.assertAlmostEqual(estimate_mutual_information(a, independent, **common)['value'], 0.)
        self.assertAlmostEqual(estimate_mutual_information(a, a, kind_a='discrete', kind_b='discrete')['value'], np.log(2))

    def test_gaussian_matches_analytic_mi_and_independence_small_bias(self):
        rng = np.random.default_rng(839)
        a, noise = rng.normal(size=(2, 5000))
        rho = .7
        b = rho * a + np.sqrt(1-rho*rho) * noise
        estimate = estimate_mutual_information(a, b)
        self.assertAlmostEqual(estimate['value'], -.5*np.log(1-rho*rho), delta=.05)
        self.assertLess(estimate_mutual_information(a, noise)['value'], .06)

    def test_nonlinear_dependency_detected_when_pearson_is_small(self):
        rng = np.random.default_rng(493)
        a = rng.uniform(-1, 1, 4000)
        b = a*a + rng.normal(scale=.04, size=len(a))
        self.assertLess(abs(np.corrcoef(a, b)[0, 1]), .06)
        self.assertGreater(estimate_mutual_information(a, b)['value'], 1.)

    def test_continuous_scaling_handles_large_finite_coordinates(self):
        a = np.linspace(-1, 1, 100)
        b = a*a + .01*np.sin(np.arange(100))
        ordinary = estimate_mutual_information(a, b, seed=71)
        large = estimate_mutual_information(a*1e308, b, seed=71)
        self.assertAlmostEqual(ordinary['value'], large['value'], places=12)

    def test_mixed_separated_classes_and_swap_symmetry(self):
        rng = np.random.default_rng(89)
        category = np.repeat([2., 7.], 700)
        continuous = category + rng.normal(scale=.15, size=len(category))
        mixed = estimate_mutual_information(category, continuous, kind_a='discrete', units='bits')
        reverse = estimate_mutual_information(continuous, category, kind_b='discrete', units='bits')
        self.assertEqual(mixed['estimator'], 'ross_fixed_k')
        self.assertAlmostEqual(mixed['value'], 1., delta=.01)
        self.assertEqual(mixed['value'], reverse['value'])

    def test_mixed_matches_public_sklearn_reference_when_available(self):
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            self.skipTest('Optional reference implementation is unavailable')
        rng = np.random.default_rng(20)
        d = rng.integers(0, 4, 2000)
        c = d + rng.normal(size=len(d))
        reference = mutual_info_regression(d[:, None], c, discrete_features=True, n_neighbors=4, random_state=3)[0]
        actual = estimate_mutual_information(d, c, kind_a='discrete', neighbors=4, seed=3)
        self.assertAlmostEqual(actual['value'], reference, places=9)

    def test_constant_small_sample_ties_and_rare_classes(self):
        self.assertEqual(estimate_mutual_information(np.ones(5), np.arange(5))['value'], 0.)
        self.assertEqual(estimate_mutual_information(np.arange(3), np.arange(3))['status'], 'insufficient_neighbors')
        rare = estimate_mutual_information([0., 0., 0., 0., 1.], np.arange(5), kind_a='discrete')
        self.assertEqual(rare['status'], 'insufficient_category_support')
        self.assertTrue(np.isnan(rare['value']))
        a = np.tile(np.arange(10), 5)
        first = estimate_mutual_information(a, a*a, seed=35)
        second = estimate_mutual_information(a, a*a, seed=35)
        self.assertEqual(first, second)
        self.assertEqual(first['ties_a'], 40)
        self.assertGreater(first['value'], 1.)

    def test_signed_delay_and_swap_with_finite_endpoints(self):
        rng = np.random.default_rng(12)
        a = rng.integers(0, 2, 1000).astype(float)
        b = np.r_[np.nan, a[:-1]]
        kwargs = dict(kind_a='discrete', kind_b='discrete', units='bits', lag_edges=np.arange(-2.5, 3.5))
        ab, ba = curve(a, b, **kwargs), curve(b, a, **kwargs)
        self.assertGreater(ab['values'][3], .99)
        self.assertLess(ab['values'][1], .01)
        np.testing.assert_allclose(ab['values'], ba['values'][::-1])
        np.testing.assert_equal(ab['pair_counts'], ba['pair_counts'][::-1])

    def test_run_gaps_and_nan_do_not_compress_time(self):
        result = compute_delayed_mutual_information(runs_for([0, 1], [3, 4, 5]), np.arange(6),
            feature_a=[0, 1, 1, 0, np.nan, 1], feature_b=[0, 1, 1, 0, np.nan, 1],
            lag_edges=[.5, 1.5, 2.5, 5.5], kind_a='discrete', kind_b='discrete', min_pairs=1)
        np.testing.assert_equal(result['pair_counts'], [1, 1, 0])
        np.testing.assert_equal(result['track_counts'], [1, 1, 0])
        np.testing.assert_allclose(result['mean_lags'][:2], [1, 2])
        self.assertEqual(result['status'][2], 'no_pairs')

    def test_seeded_circular_null_is_reproducible_and_preserves_missing_support(self):
        rng = np.random.default_rng(954)
        a = rng.normal(size=500)
        b = a*a + rng.normal(scale=.03, size=len(a))
        b[250] = np.nan
        kwargs = dict(seed=6, null_draws=7, null_min_shift=15)
        first, second = curve(a, b, **kwargs), curve(a, b, **kwargs)
        self.assertEqual(first['diagnostics'], second['diagnostics'])
        diagnostic = first['diagnostics'][0]
        self.assertEqual(first['pair_counts'][0], 499)
        self.assertEqual(diagnostic['null_status'], 'ok')
        self.assertEqual(diagnostic['null_draws_valid'], 7)
        self.assertLess(diagnostic['null_mean'], .1)
        self.assertGreater(diagnostic['observed_minus_null'], 1.)
        self.assertTrue(np.isnan(b[250]))

    def test_irregular_or_short_null_support_is_unavailable_not_shuffled(self):
        common = dict(runs=runs_for([0, 1, 2, 3, 4]), feature_a=np.arange(5), feature_b=np.arange(5), lag_edges=[-.1, .1], null_draws=3)
        irregular = compute_delayed_mutual_information(**common, times=[0., 1, 3, 4, 6])
        self.assertTrue(np.isfinite(irregular['values'][0]))
        self.assertEqual(irregular['diagnostics'][0]['null_status'], 'unsupported_segments')
        self.assertIn('irregular_sampling', irregular['diagnostics'][0]['null_reasons'])
        short = compute_delayed_mutual_information(**common, times=np.arange(5), null_min_shift=3)
        self.assertIn('short_finite_segments', short['diagnostics'][0]['null_reasons'])

    def test_pair_storage_budget_fails_explicitly_and_unlimited_is_exact(self):
        a = np.arange(10)
        kwargs = dict(lag_edges=[-20, 20], kind_a='discrete', kind_b='discrete')
        with self.assertRaisesRegex(MutualInformationPairLimitExceeded, '0 means unlimited'):
            curve(a, a, max_pairs=99, **kwargs)
        result = curve(a, a, max_pairs=0, **kwargs)
        self.assertEqual(result['pair_counts'][0], 100)
        self.assertAlmostEqual(result['values'][0], 0.)

    def test_cancel_during_null_and_validate_scalar_inputs(self):
        event = threading.Event()
        def progress(message):
            if 'diagnostic' in message:
                event.set()
        with self.assertRaises(TemporalCorrelationCancelled):
            curve(np.arange(100), np.arange(100), null_draws=10, cancelled=event, progress=progress)
        with self.assertRaisesRegex(ValueError, 'scalar'):
            curve(np.ones((10, 2)), np.arange(10))
        with self.assertRaisesRegex(ValueError, 'kinds'):
            curve(np.arange(10), np.arange(10), kind_a='auto')

    def test_temporal_dispatch_ignores_correlation_normalization(self):
        result = compute_temporal_curve(runs_for(np.arange(20)), np.arange(20), method='mutual_information',
            feature_a=np.tile([0, 1], 10), feature_b=np.tile([0, 1], 10), lag_edges=[-.1, .1],
            normalization='coefficient', centering='track', mi_options=dict(kind_a='discrete', kind_b='discrete', units='bits'))
        self.assertEqual(result['normalization'], 'none')
        self.assertEqual(result['centering'], 'none')
        self.assertAlmostEqual(result['values'][0], 1.)


if __name__ == '__main__':
    unittest.main()
