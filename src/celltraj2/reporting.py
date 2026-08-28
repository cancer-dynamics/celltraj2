"""Structured, durable progress reporting for celltraj2 workers."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO

from celltraj2.schema import utc_now_iso


class JobCancelledError(RuntimeError):
    """Raised cooperatively when SITE requests cancellation of a worker."""


def cancel_request_path() -> Path | None:
    """Return the configured cooperative cancellation request file."""

    value = os.environ.get("CELLTRAJ2_CANCEL_PATH")
    return None if value in (None, "") else Path(str(value))


def cancel_requested() -> bool:
    """Return whether the active SITE job has requested graceful cancellation."""

    path = cancel_request_path()
    return bool(path is not None and path.exists())


def raise_if_cancelled() -> None:
    """Stop at a worker-safe progress boundary when cancellation is requested."""

    path = cancel_request_path()
    if path is not None and path.exists():
        raise JobCancelledError(f"Cancellation requested via {path}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


class JsonlReporter:
    """Flush JSON events to stdout and an optional durable JSONL file."""

    def __init__(self, stream: TextIO | None = None, *, events_path: str | Path | None = None) -> None:
        self.stream = stream or sys.stdout
        configured = events_path or os.environ.get("CELLTRAJ2_EVENTS_PATH")
        self.events_path = None if configured in (None, "") else Path(str(configured))
        if self.events_path is not None:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Mapping[str, Any]) -> None:
        payload = {"timestamp": utc_now_iso(), **dict(event)}
        self._write(payload)
        if str(payload.get("event") or "") not in {
            "job_cancel_requested",
            "job_cancelled",
            "job_completed",
        } and cancel_requested():
            self._write(
                {
                    "timestamp": utc_now_iso(),
                    "event": "job_cancel_requested",
                    "cancel_path": str(cancel_request_path()),
                }
            )
            raise_if_cancelled()

    def _write(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(_json_safe(payload), sort_keys=True) + "\n"
        self.stream.write(line)
        self.stream.flush()
        if self.events_path is not None:
            with self.events_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(line)
                handle.flush()

    def terminal(self, event: Mapping[str, Any]) -> None:
        """Write a terminal event without re-checking the cancellation flag."""

        self._write({"timestamp": utc_now_iso(), **dict(event)})


__all__ = [
    "JobCancelledError",
    "JsonlReporter",
    "cancel_request_path",
    "cancel_requested",
    "raise_if_cancelled",
]
