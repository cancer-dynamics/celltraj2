"""Run a celltraj2 surface-motion batch job."""

from __future__ import annotations

import argparse
from pathlib import Path

from celltraj2.surface_motion_batch import JsonlReporter, load_surface_motion_job, run_batch_surface_motion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 surface-motion batch job.")
    parser.add_argument("job", type=Path, help="Path to a surface-motion job JSON file.")
    args = parser.parse_args(argv)
    run_batch_surface_motion(load_surface_motion_job(args.job), reporter=JsonlReporter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
