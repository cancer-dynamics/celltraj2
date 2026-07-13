"""Run a celltraj2 batch global-registration job."""

from __future__ import annotations

import argparse
from pathlib import Path

from celltraj2.registration_batch import JsonlReporter, load_registration_job, run_batch_registration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 global-registration batch job.")
    parser.add_argument("job", type=Path, help="Path to a celltraj2 registration job JSON file.")
    args = parser.parse_args(argv)
    job = load_registration_job(args.job)
    run_batch_registration(job, reporter=JsonlReporter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
