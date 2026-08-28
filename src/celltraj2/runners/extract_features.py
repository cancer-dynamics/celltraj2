"""Run a celltraj2 batch feature-extraction job."""

from __future__ import annotations

import argparse
from pathlib import Path

from celltraj2.feature_extraction import load_feature_extraction_job, run_batch_feature_extraction
from celltraj2.runners._common import run_reported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 feature-extraction batch job.")
    parser.add_argument("job", type=Path, help="Path to a celltraj2 feature-extraction job JSON file.")
    args = parser.parse_args(argv)
    job = load_feature_extraction_job(args.job)
    return run_reported(lambda reporter: run_batch_feature_extraction(job, reporter=reporter))


if __name__ == "__main__":
    raise SystemExit(main())
