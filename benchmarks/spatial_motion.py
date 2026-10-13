"""Reproducible spatial-kernel timings; optionally compare a saved old module.

Run with the supported scientific Python environment, for example::

    python benchmarks/spatial_motion.py --cells 100 --frames 16 --repeat 3
    python benchmarks/spatial_motion.py --baseline /tmp/old_spatial.py

No H5 files are read or modified. Timings exclude imports/input generation.
"""

import argparse
import importlib.util
import json
import time

import numpy as np

from celltraj2.spatial_correlations import compute_motion_surface
from celltraj2.trajectory_index import Runs


def fixture(cells=100, frames=16, radius=80):
    rng = np.random.default_rng(8043)
    positions = rng.uniform(0, 100, (cells, 1, 2)) + np.cumsum(rng.normal(size=(cells, frames, 2)), axis=1)
    return dict(runs=Runs(np.arange(cells * frames), np.arange(cells + 1) * frames),
                positions=positions.reshape(-1, 2), times=np.tile(np.arange(frames), cells),
                displacement_interval=1, distance_edges=np.linspace(0, radius, 11),
                lag_edges=np.arange(-3.5, 4), method="radial_response", min_pairs=2)


def compare_results(expected, actual):
    for key, value in expected.items():
        if key == "exclusion_counts":
            for reason in value.keys() | actual[key].keys():
                assert value.get(reason, 0) == actual[key].get(reason, 0), (key, reason, value, actual[key])
        elif isinstance(value, dict):
            compare_results(value, actual[key])
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in "fc":
                np.testing.assert_allclose(actual[key], value, rtol=2e-10, atol=2e-10, err_msg=key)
            else:
                np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert actual[key] == value, (key, value, actual[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--convention", choices=("cross_time", "source_time"), default="cross_time")
    args = parser.parse_args()
    implementations = {"optimized": compute_motion_surface}
    if args.baseline:
        spec = importlib.util.spec_from_file_location("celltraj2._spatial_baseline", args.baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        implementations = {"baseline": module.compute_motion_surface, **implementations}
    for radius in (80, 8):
        data = fixture(args.cells, args.frames, radius)
        data["distance_convention"] = args.convention
        expected = None
        measurements = {}
        for name, implementation in implementations.items():
            elapsed = []
            for _ in range(args.repeat):
                started = time.perf_counter()
                result = implementation(**data)
                elapsed.append(time.perf_counter() - started)
            measurements[name] = {"seconds": elapsed, "median_seconds": float(np.median(elapsed)),
                                  "pairs": int(result["pair_counts"].sum())}
            if expected is not None:
                compare_results(expected, result)
            expected = result
        print(json.dumps({"cells": args.cells, "frames": args.frames, "radius": radius,
                          "convention": args.convention, "measurements": measurements}), flush=True)


if __name__ == "__main__":
    main()
