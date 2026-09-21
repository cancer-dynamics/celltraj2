from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from celltraj2.channel_comparison import compare_channel_values, comparison_selections
from celltraj2.feature_catalog import CHANNEL_COMPARISON_FIELDS, CHANNEL_COMPARISON_METRICS
from celltraj2.features import FeatureSetSpec, expand_intensity_statistics
from celltraj2.feature_extraction import _feature_dependency_paths
from celltraj2.schema import ChannelSpec, ImageSourceSpec, TrajectoryMetadata
from celltraj2.store import TrajectoryStore
from celltraj2.trajectory import Trajectory


class ChannelComparisonTests(unittest.TestCase):
    def compare(self, a, b, **kwargs):
        config = {"metrics": CHANNEL_COMPARISON_METRICS, "fields": CHANNEL_COMPARISON_FIELDS,
                  "statistics": expand_intensity_statistics(["mean", "median", "std", "sum", "area", "percentiles"])}
        config.update(kwargs)
        return compare_channel_values(a, b, **config)

    def test_pixel_ratios_are_not_ratio_of_means(self):
        values = self.compare([2., 9.], [1., 3.])
        self.assertAlmostEqual(values["ratio_mean"], 2.5)
        self.assertAlmostEqual(values["ratio_of_means"], 2.75)
        self.assertAlmostEqual(values["inverse_ratio_mean"], (1/2 + 1/3)/2)
        self.assertAlmostEqual(values["difference_mean"], 3.5)
        self.assertAlmostEqual(values["absolute_difference_mean"], 3.5)
        self.assertAlmostEqual(values["sum_mean"], 7.5)
        self.assertAlmostEqual(values["product_mean"], 14.5)
        self.assertAlmostEqual(values["normalized_difference_mean"], (1/3 + 1/2)/2)
        self.assertAlmostEqual(values["log2_ratio_mean"], np.log2([2., 3.]).mean())
        self.assertEqual(values["ratio_percentile_0"], 2)
        self.assertEqual(values["ratio_percentile_100"], 3)
        self.assertAlmostEqual(values["ratio_percentile_99"], 2.99)
        self.assertEqual(values["ratio_area"], 2)

    def test_finite_pair_and_denominator_policies(self):
        a, b = [1, 2, -4, 8, np.nan, 5], [0, -1, 2, 1, 1, np.inf]
        values = self.compare(a, b)
        self.assertEqual(values["valid_pair_count"], 4)
        self.assertAlmostEqual(values["valid_pair_fraction"], 4/6)
        self.assertEqual(values["valid_ratio_fraction"], .5)
        self.assertEqual(values["ratio_mean"], 3)
        self.assertEqual(values["ratio_area"], 2)
        self.assertEqual(values["log2_ratio_area"], 1)
        self.assertEqual(values["log2_ratio_mean"], 3)
        signed = self.compare(a, b, denominator_policy="absolute")
        self.assertEqual(signed["ratio_area"], 3)
        self.assertAlmostEqual(signed["ratio_mean"], 4/3)
        thresholded = self.compare(a, b, denominator_floor=1)
        self.assertEqual(thresholded["ratio_mean"], -2)
        self.assertEqual(thresholded["ratio_area"], 1)
        self.assertTrue(np.isnan(thresholded["log2_ratio_mean"]))

    def test_empty_and_invalid_populations_are_not_zero_signal(self):
        values = self.compare([1, 2], [0, 0])
        self.assertTrue(np.isnan(values["ratio_mean"]))
        self.assertTrue(np.isnan(values["ratio_sum"]))
        self.assertEqual(values["ratio_area"], 0)
        self.assertEqual(values["valid_ratio_fraction"], 0)
        empty = self.compare([], [])
        self.assertEqual(empty["valid_pair_count"], 0)
        self.assertTrue(np.isnan(empty["valid_pair_fraction"]))
        self.assertTrue(np.isnan(empty["ratio_of_means"]))
        nonfinite = self.compare([np.nan, np.inf], [2, 3])
        self.assertEqual(nonfinite["valid_pair_fraction"], 0)
        self.assertEqual(nonfinite["difference_area"], 0)

    def test_correlations_and_constant_channels(self):
        values = self.compare([1, 2, 3, 4], [1, 4, 9, 16])
        self.assertAlmostEqual(values["spearman_correlation"], 1)
        self.assertLess(values["pearson_correlation"], 1)
        self.assertAlmostEqual(values["pearson_correlation"], np.corrcoef([1,2,3,4], [1,4,9,16])[0,1])
        constant = self.compare([2, 2], [3, 3])
        self.assertTrue(np.isnan(constant["pearson_correlation"]))
        self.assertTrue(np.isnan(constant["spearman_correlation"]))
        self.assertAlmostEqual(constant["cosine_similarity"], 1)
        self.assertTrue(np.isnan(self.compare([0, 0], [3, 3])["cosine_similarity"]))
        filtered = self.compare([1, np.nan, 3], [2, 8, 6])
        self.assertAlmostEqual(filtered["pearson_correlation"], 1)

    def test_configuration_validation(self):
        for floor in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                self.compare([1], [1], denominator_floor=floor)
        with self.assertRaises(ValueError):
            self.compare([1], [1], denominator_policy="guess")
        with self.assertRaises(ValueError):
            self.compare([1], [1, 2])
        for cfg in ({"metrics": [], "fields": []}, {"metrics": ["wrong"]},
                    {"fields": ["ratio"], "statistics": []}, {"fields": ["wrong"]}):
            with self.assertRaises(ValueError):
                comparison_selections(cfg)
        self.assertEqual(comparison_selections({"metrics": ["ratio_of_means"], "fields": [], "statistics": []}), (["ratio_of_means"], [], []))

    def create_movie(self, path, ndim):
        labels = np.zeros((4,4), np.uint16)
        labels[1,1:3] = 1
        labels[2,1:3] = 2
        nuclear = np.zeros_like(labels, bool)
        nuclear[1:3,1] = True
        blocked = np.zeros_like(labels)
        blocked[2,1] = 7
        signal = np.zeros(labels.shape, float)
        signal[1,1:3] = [1,3]
        signal[2,1:3] = [2,4]
        if ndim == 3:
            labels, nuclear, blocked, signal = (np.stack([array, array]) for array in (labels, nuclear, blocked, signal))
        axes = ("Y", "X", "C") if ndim == 2 else ("Z", "Y", "X", "C")
        metadata = TrajectoryMetadata(roi_id="test", dataset_id="test", frame_count=2,
            channels=[ChannelSpec(raw_index=0, display_name="A"), ChannelSpec(raw_index=1, display_name="B")],
            image_source=ImageSourceSpec(source_type="embedded_h5", axes=axes))
        with TrajectoryStore.create(path, metadata=metadata) as store:
            for frame in (1,2):
                store.write_raw_frame(frame, np.stack([10 + 2*signal*frame, 5 + signal*frame**2], axis=-1))
                store.write_label_frame("cells", frame, labels)
                store.write_label_frame("blocked", frame, blocked)
                store.write_mask_frame("nuclear", frame, nuclear)
        with Trajectory(path, mode="r+") as trajectory:
            trajectory.index_observations("cells")

    def test_2d_3d_compartments_and_independent_preprocessing_are_stored(self):
        for ndim in (2,3):
            with self.subTest(ndim=ndim), TemporaryDirectory() as tmp:
                path = Path(tmp)/"comparison.h5"
                self.create_movie(path, ndim)
                background = {"enabled": True, "source_kind": "label", "source_name": "cells", "region": "inverse"}
                normalization = {"method": "match_mean", "scope": "per_frame", "source_kind": "label", "source_name": "cells"}
                feature = {"kind": "channel_comparison", "name": "fret", "channel_a": 0, "channel_b": 1,
                    "metrics": ["ratio_of_means", "valid_pair_count"], "fields": ["ratio", "difference"], "statistics": ["mean", "area", "percentiles"],
                    "compartment": {"include_mask_set": "nuclear", "exclude_label_set": "blocked"},
                    "background_a": background, "background_b": background,
                    "normalization_a": normalization, "normalization_b": normalization}
                with Trajectory(path, mode="r+") as trajectory:
                    result = trajectory.extract_features({"feature_set": "comparison", "object_set": "cells", "features": [feature]})
                    np.testing.assert_allclose(result.values["fret_ratio_mean"][[0,2]], 2)
                    np.testing.assert_allclose(result.values["fret_difference_mean"][[0,2]], 1)
                    self.assertTrue(np.isnan(result.values["fret_ratio_mean"][[1,3]]).all())
                    np.testing.assert_equal(result.values["fret_valid_pair_count"][[1,3]], 0)
                    np.testing.assert_equal(result.values["fret_ratio_area"][[1,3]], 0)
                    schema = trajectory.store.read_json("/object_sets/cells/features/comparison/schema.json")
                    scalar = next(column for column in schema["columns"] if column["name"] == "fret_ratio_of_means")
                    self.assertIsNone(scalar["field"])
                    prep = scalar["preprocessing"]
                    self.assertAlmostEqual(prep["a"]["normalization"]["frame_parameters"]["2"]["scale"], .5)
                    self.assertAlmostEqual(prep["b"]["normalization"]["frame_parameters"]["2"]["scale"], .25)

    def test_pooled_normalization_preserves_temporal_ratio_changes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"pooled.h5"
            self.create_movie(path, 2)
            bg = {"source_kind": "label", "source_name": "cells", "region": "inverse"}
            norm = {"method": "divide_mean", "scope": "all_frames", "source_kind": "label", "source_name": "cells"}
            with Trajectory(path, mode="r+") as trajectory:
                result = trajectory.extract_features({"feature_set": "pooled", "object_set": "cells", "features": [{
                    "kind": "channel_comparison", "name": "c", "channel_a": 0, "channel_b": 1,
                    "metrics": [], "fields": ["ratio"], "statistics": ["mean"],
                    "background_a": bg, "background_b": bg, "normalization_a": norm, "normalization_b": norm}]}, save_outputs=False)
                self.assertAlmostEqual(result.values["c_ratio_mean"][0] / result.values["c_ratio_mean"][2], 2)

    def test_legacy_correlation_column_unchanged(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"legacy.h5"
            self.create_movie(path, 2)
            with Trajectory(path, mode="r+") as trajectory:
                result = trajectory.extract_features({"feature_set": "legacy", "object_set": "cells", "features": [{
                    "kind": "channel_correlation", "name": "old_corr", "channel_a": 0, "channel_b": 1}]}, save_outputs=False)
                self.assertEqual(result.values.dtype.names, ("observation_id", "old_corr"))
                np.testing.assert_allclose(result.values["old_corr"], 1)

    def test_source_dependencies_include_both_preprocessing_sources(self):
        spec = FeatureSetSpec.from_dict({"feature_set": "c", "object_set": "cells", "features": [{
            "kind": "channel_comparison", "background_a": {"source_kind": "mask", "source_name": "bg"},
            "normalization_b": {"source_kind": "label", "source_name": "reference"}}]})
        paths = _feature_dependency_paths(spec, [1])
        self.assertIn("/images", paths)
        self.assertIn("/masks/bg", paths)
        self.assertIn("/labels/reference", paths)


if __name__ == "__main__":
    unittest.main()
