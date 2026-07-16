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
        line = json.dumps(_json_safe(payload), sort_keys=True) + "\n"
        self.stream.write(line)
        self.stream.flush()
        if self.events_path is not None:
            with self.events_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(line)
                handle.flush()


__all__ = ["JsonlReporter"]
