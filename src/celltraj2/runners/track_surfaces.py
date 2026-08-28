"""Run a celltraj2 surface-motion batch job."""

from __future__ import annotations

import argparse
from pathlib import Path

from celltraj2.runners._common import run_reported
from celltraj2.surface_motion_batch import load_surface_motion_job, run_batch_surface_motion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 surface-motion batch job.")
    parser.add_argument("job", type=Path, help="Path to a surface-motion job JSON file.")
    args = parser.parse_args(argv)
    job = load_surface_motion_job(args.job)
    return run_reported(lambda reporter: run_batch_surface_motion(job, reporter=reporter))


if __name__ == "__main__":
    raise SystemExit(main())
