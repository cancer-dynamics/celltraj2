"""Run a celltraj2 boundary-library batch job."""

from __future__ import annotations

import argparse
from pathlib import Path

from celltraj2.boundary_batch import JsonlReporter, load_boundary_job, run_batch_boundaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 boundary-library batch job.")
    parser.add_argument("job", type=Path, help="Path to a boundary job JSON file.")
    args = parser.parse_args(argv)
    run_batch_boundaries(load_boundary_job(args.job), reporter=JsonlReporter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
