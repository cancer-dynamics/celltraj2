"""Run a celltraj2 boundary-library batch job."""

from __future__ import annotations

import argparse
from pathlib import Path

from celltraj2.boundary_batch import load_boundary_job, run_batch_boundaries
from celltraj2.runners._common import run_reported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 boundary-library batch job.")
    parser.add_argument("job", type=Path, help="Path to a boundary job JSON file.")
    args = parser.parse_args(argv)
    job = load_boundary_job(args.job)
    return run_reported(lambda reporter: run_batch_boundaries(job, reporter=reporter))


if __name__ == "__main__":
    raise SystemExit(main())
