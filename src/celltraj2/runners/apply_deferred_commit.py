"""Apply a durable result bundle to a previously blocked canonical H5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from celltraj2.deferred_commit import apply_deferred_job
from celltraj2.reporting import JobCancelledError, JsonlReporter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", help="Deferred commit job JSON")
    args = parser.parse_args(argv)
    reporter = JsonlReporter()
    try:
        outcome = apply_deferred_job(
            json.loads(Path(args.job).read_text(encoding="utf-8")),
            reporter=reporter,
        )
    except JobCancelledError as exc:
        reporter.terminal({"event": "job_cancelled", "error": str(exc)})
        return 130
    return 0 if outcome.status in {"committed", "skipped", "deferred"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
