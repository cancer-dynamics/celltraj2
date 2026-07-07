"""Run a celltraj2 batch object-indexing job."""

from __future__ import annotations

import argparse
from pathlib import Path

from celltraj2.object_indexing import JsonlReporter, load_object_index_job, run_batch_object_indexing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 object-indexing batch job.")
    parser.add_argument("job", type=Path, help="Path to a celltraj2 object-indexing job JSON file.")
    args = parser.parse_args(argv)
    job = load_object_index_job(args.job)
    run_batch_object_indexing(job, reporter=JsonlReporter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
