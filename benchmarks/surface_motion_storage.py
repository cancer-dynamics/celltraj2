"""Synthetic result-retention/commit RSS comparison (not an OT solver benchmark).

Run each mode/size in a fresh process, e.g.:
  python benchmarks/surface_motion_storage.py --mode disk --links 1024
Includes result generation, retention, concatenation (memory mode), and copying
into compressed output datasets. No user data or canonical H5 is modified.
"""
import argparse
import json
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
import time

import h5py
import numpy as np

from celltraj2.array_stream import copy_array
from celltraj2.motion_stream import MotionSpool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['memory', 'disk'], required=True)
    parser.add_argument('--links', type=int, default=1024)
    parser.add_argument('--edges', type=int, default=8192)
    args = parser.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(42)
    retained = {name: {} for name in ('transport', 'source_summary', 'target_summary')}
    with TemporaryDirectory(prefix='motion-storage-benchmark-') as tmp:
        with MotionSpool(Path(tmp) / 'scratch.h5') as spool:
            for link in range(args.links):
                transport = dict(source_point_id=rng.integers(1, 512, args.edges),
                                 target_point_id=rng.integers(1, 512, args.edges),
                                 mass=rng.random(args.edges), edge_cost=rng.random(args.edges).astype('f4'),
                                 registered_displacement_zyx=rng.random((args.edges, 3)).astype('f4'))
                def summary():
                    return dict(point_id=np.arange(512), point_mass=rng.random(512),
                                matched_fraction=rng.random(512), quality_flags=np.zeros(512, 'u4'),
                                barycentric_displacement_zyx=rng.random((512, 3)).astype('f4'))
                source, target = summary(), summary()
                if args.mode == 'disk':
                    spool.append(link // 50, transport, source, target)
                else:
                    for group, columns in [('transport', transport), ('source_summary', source), ('target_summary', target)]:
                        for name, values in columns.items():
                            retained[group].setdefault(name, []).append(values)
            if args.mode == 'disk':
                transport, summaries = spool.finish({})
                result = dict(transport=transport, **summaries)
            else:
                result = {group: {name: np.concatenate(blocks) for name, blocks in columns.items()}
                          for group, columns in retained.items()}
            with h5py.File(Path(tmp) / 'committed.h5', 'w') as output:
                for group, columns in result.items():
                    for name, values in columns.items():
                        copy_array(output.require_group(group), name, values)
            print(json.dumps(dict(mode=args.mode, links=args.links,
                                  edges=args.edges * args.links,
                                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                  elapsed_seconds=time.perf_counter() - started,
                                  peak_buffer_mib=spool.peak_buffer_bytes / 1024**2)))


if __name__ == '__main__':
    main()
