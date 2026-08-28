"""Run a celltraj2 batch centroid-tracking job."""

from __future__ import annotations

import argparse
from pathlib import Path

from celltraj2.runners._common import run_reported
from celltraj2.tracking_batch import load_tracking_job, run_batch_tracking


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 centroid-tracking batch job.")
    parser.add_argument("job", type=Path, help="Path to a celltraj2 tracking job JSON file.")
    args = parser.parse_args(argv)
    job = load_tracking_job(args.job)
    return run_reported(lambda reporter: run_batch_tracking(job, reporter=reporter))


if __name__ == "__main__":
    raise SystemExit(main())
