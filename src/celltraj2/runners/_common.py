"""Shared worker-runner lifecycle helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from celltraj2.reporting import JobCancelledError, JsonlReporter


def run_reported(action: Callable[[JsonlReporter], Any]) -> int:
    """Run a worker with standard failure and cooperative-cancel exit codes."""

    reporter = JsonlReporter()
    try:
        summary = action(reporter)
    except JobCancelledError as exc:
        reporter.terminal({"event": "job_cancelled", "error": str(exc)})
        return 130
    return 1 if int(getattr(summary, "failed", 0) or 0) else 0


__all__ = ["run_reported"]
