"""Independent scalar oracle for exact accelerated spatial pairing/reductions."""

import unittest
from unittest.mock import patch

import numpy as np

from celltraj2 import spatial_correlations as spatial
from celltraj2.trajectory_index import Runs


def fixture(dimension=3):
    rng = np.random.default_rng(9120)
    cells, frames = 9, 6
    xyz = rng.uniform(-4, 4, (cells, 1, dimension)) + np.cumsum(rng.normal(size=(cells, frames, dimension)), axis=1)
    xyz[1, 2] = np.nan
    xyz[2, 0] = xyz[0, 0]  # Undefined axis at exactly zero separation.
    clock = np.tile(np.array([0, .5, 1, 1.5, 2, 2.5]), cells)
    clock[3 * frames:4 * frames] += .1  # Some exact source-time references absent.
    source = np.ones(cells * frames, bool)
    receiver = source.copy()
    source[4 * frames + 2] = False
    receiver[5 * frames + 3] = False
    return dict(runs=Runs(np.arange(cells * frames), np.arange(cells + 1) * frames),
                times=clock, positions=xyz.reshape(-1, dimension),
                source_mask=source, receiver_mask=receiver,
                cell_ids=np.repeat(np.arange(cells) // 2, frames),
                lineage_ids=np.repeat(np.arange(cells) // 3, frames),
                exclude_related_lineage=True, displacement_interval=1,
                distance_edges=[.25, 2, 4, 6], lag_edges=[-1, -.2, .2, 1], min_pairs=2)


def scalar_pairs(args):
    """Intentionally simple, unindexed pair enumeration on a small fixture."""
    times, xyz = args["times"], args["positions"]
    source, receiver = args["source_mask"], args["receiver_mask"]
    runs, ends, owners = [], {}, {}
    for start, stop in zip(args["runs"].offsets[:-1], args["runs"].offsets[1:]):
        rows = args["runs"].order[start:stop]
        runs.append(rows)
        for local, row in enumerate(rows):
            owners[row] = (rows, local)
            later = rows[local + 1:]
            if len(later):
                errors = abs(times[later] - times[row] - args["displacement_interval"])
                best = errors.min()
                matched = later[errors == best]
                if best < 1e-12 and len(matched) == 1:
                    ends[row] = int(matched[0])

    def motion_ok(row, mask):
        if row not in ends:
            return False
        rows, local = owners[row]
        end = int(np.flatnonzero(rows == ends[row])[0])
        return mask[rows[local:end + 1]].all() and np.isfinite(xyz[[row, ends[row]]]).all()

    result = {}
    radial = args["method"] in ("radial_response", "relative_radial")
    for i in args["runs"].order:
        if args["method"] == "radial_response":
            if not source[i] or not np.isfinite(xyz[i]).all():
                continue
        elif not motion_ok(i, source):
            continue
        for j in args["runs"].order:
            if not motion_ok(j, receiver):
                continue
            if args.get("exclude_same_cell", True) and args["cell_ids"][i] == args["cell_ids"][j]:
                continue
            if args["exclude_related_lineage"] and args["lineage_ids"][i] == args["lineage_ids"][j]:
                continue
            ref = j
            if args["distance_convention"] == "source_time":
                rows, local = owners[j]
                matches = np.flatnonzero(times[rows] == times[i])
                if len(matches) != 1:
                    continue
                reference = int(matches[0])
                end = int(np.flatnonzero(rows == ends[j])[0])
                if not receiver[rows[min(local, reference):max(end, reference) + 1]].all():
                    continue
                ref = rows[reference]
            separation = xyz[i] - xyz[ref]
            distance = np.linalg.norm(separation)
            lag = times[j] - times[i]
            if not np.isfinite(distance) or distance == 0:
                continue
            rb = next((k for k, (left, right) in enumerate(zip(args["distance_edges"][:-1], args["distance_edges"][1:]))
                       if left <= distance < right or (k == len(args["distance_edges"]) - 2 and distance == right)), None)
            tb = next((k for k, (left, right) in enumerate(zip(args["lag_edges"][:-1], args["lag_edges"][1:]))
                       if left <= lag < right or (k == len(args["lag_edges"]) - 2 and lag == right)), None)
            if rb is None or tb is None:
                continue
            axis = separation / distance
            b = xyz[ends[j]] - xyz[j]
            a = xyz[ends[i]] - xyz[i] if i in ends else None
            if radial:
                projected = np.dot(b if args["method"] == "radial_response" else b - a, axis)
                a, b = np.array([1.]), np.array([projected * (-1 if args.get("radial_sign") == "away" else 1)])
            elif args["method"] == "longitudinal":
                a, b = np.array([np.dot(a, axis)]), np.array([np.dot(b, axis)])
            elif args["method"] == "transverse":
                a, b = a - np.dot(a, axis) * axis, b - np.dot(b, axis) * axis
                if args.get("transverse_mode") == "mean":
                    a, b = a / np.sqrt(xyz.shape[1] - 1), b / np.sqrt(xyz.shape[1] - 1)
            result.setdefault((tb, rb), []).append((i, j, a, b, lag, distance))
    return result


class SpatialPerformanceParityTests(unittest.TestCase):
    def assert_oracle(self, args, output):
        pairs = scalar_pairs(args)
        for index in np.ndindex(output["values"].shape):
            rows = pairs.get(index, [])
            self.assertEqual(output["pair_counts"][index], len(rows))
            if not rows:
                self.assertEqual(output["status"][index], "no_pairs")
                continue
            sources, targets = [r[0] for r in rows], [r[1] for r in rows]
            a, b = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
            ma, mb = a.mean(axis=0), b.mean(axis=0)
            da, db = a - ma, b - mb
            va, vb, cov = (np.sum(da * da), np.sum(db * db), np.sum(da * db))
            moment_expected = dict(value_sum=np.sum(a * b), lag_sum=sum(r[4] for r in rows),
                                   distance_sum=sum(r[5] for r in rows), pair_centered_cross_sum=cov,
                                   pair_centered_square_sum_a=va, pair_centered_square_sum_b=vb,
                                   pair_reference_mean_a=ma, pair_reference_mean_b=mb)
            for name, expected in moment_expected.items():
                np.testing.assert_allclose(output["moments"][name][index], expected, rtol=3e-12, atol=3e-12)
            for name, expected in (("source_counts", len(set(sources))), ("receiver_counts", len(set(targets))),
                                   ("source_track_counts", len(set(args["cell_ids"][sources]))),
                                   ("receiver_track_counts", len(set(args["cell_ids"][targets])))):
                self.assertEqual(output[name][index], expected)
            if len(rows) < args["min_pairs"]:
                self.assertEqual(output["status"][index], "insufficient_pairs")
            else:
                norm = args["normalization"]
                expected = np.sum(a * b) / len(rows) if norm == "raw" else cov / len(rows)
                if norm == "coefficient":
                    expected = cov / np.sqrt(va * vb) if va * vb else np.nan
                np.testing.assert_allclose(output["values"][index], expected, atol=2e-12)

    def test_all_estimands_normalizations_geometries_match_scalar_oracle(self):
        for dimension in (2, 3):
            for convention in ("cross_time", "source_time"):
                for method in ("radial_response", "relative_radial", "full_dot", "longitudinal", "transverse"):
                    for normalization in (("raw",) if "radial" in method else ("raw", "covariance", "coefficient")):
                        args = fixture(dimension) | dict(method=method, normalization=normalization,
                                                        distance_convention=convention, radial_sign="away", transverse_mode="mean")
                        with self.subTest(dimension=dimension, method=method, norm=normalization, convention=convention):
                            accelerated = spatial.compute_motion_surface(**args, pair_search="spatial_index")
                            self.assert_oracle(args, accelerated)
                            direct = spatial.compute_motion_surface(**args, pair_search="brute_force")
                            for key in ("pair_counts", "source_counts", "receiver_counts", "source_track_counts",
                                        "receiver_track_counts", "status", "values", "mean_distances", "mean_lags"):
                                np.testing.assert_equal(accelerated[key], direct[key])
                            for reason in accelerated["exclusion_counts"].keys() | direct["exclusion_counts"].keys():
                                self.assertEqual(accelerated["exclusion_counts"].get(reason, 0), direct["exclusion_counts"].get(reason, 0))

    def test_index_is_reused_by_clock_and_reductions_are_bounded_batches(self):
        cells, frames = 100, 5
        xyz = np.random.default_rng(31).normal(size=(cells, frames, 2))
        args = dict(runs=Runs(np.arange(cells * frames), np.arange(cells + 1) * frames),
                    times=np.tile(np.arange(frames), cells), positions=xyz.reshape(-1, 2),
                    displacement_interval=1, distance_edges=[0, 1, 2, 3, 20], lag_edges=[-1.5, -.5, .5, 1.5])
        with patch.object(spatial, "cKDTree", wraps=spatial.cKDTree) as tree, \
                patch.object(spatial, "_grouped_moments", wraps=spatial._grouped_moments) as reductions:
            output = spatial.compute_motion_surface(**args)
        self.assertEqual(tree.call_count, frames)
        self.assertLess(reductions.call_count, 40)
        self.assertGreater(output["pair_counts"].sum(), 50000)
        self.assertTrue(all(len(call.args[0]) < 8192 for call in reductions.call_args_list))

    def test_extreme_geometry_falls_back_without_changing_exclusions(self):
        args = fixture(2)
        args["positions"] *= 1e200
        args.update(distance_edges=[0, 1e201], method="radial_response", distance_convention="source_time")
        indexed = spatial.compute_motion_surface(**args, pair_search="spatial_index")
        direct = spatial.compute_motion_surface(**args, pair_search="brute_force")
        np.testing.assert_equal(indexed["values"], direct["values"])
        self.assertEqual(indexed["exclusion_counts"], direct["exclusion_counts"])


if __name__ == "__main__":
    unittest.main()
