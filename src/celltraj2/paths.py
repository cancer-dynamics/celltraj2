"""Path and identifier helpers for the celltraj2 H5 contract."""

from __future__ import annotations

import re

FRAME_PREFIX = "frame_"
IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def frame_key(frame: int) -> str:
    """Return the one-based H5 dataset key for a frame."""

    value = int(frame)
    if value < 1:
        raise ValueError("celltraj2 frame ids are one-based; frame must be >= 1")
    return f"{FRAME_PREFIX}{value}"


def parse_frame_key(key: str) -> int:
    """Return the one-based integer frame id from a ``frame_<n>`` key."""

    text = str(key)
    if not text.startswith(FRAME_PREFIX):
        raise ValueError(f"Invalid frame key: {key!r}")
    suffix = text[len(FRAME_PREFIX) :]
    if not suffix.isdigit():
        raise ValueError(f"Invalid frame key: {key!r}")
    return int(suffix)


def frame_sort_key(key: str) -> int:
    """Sort ``frame_<n>`` keys by their numeric frame id."""

    return parse_frame_key(key)


def validate_name(name: str, *, kind: str = "name") -> str:
    """Validate a label/mask/feature set name for use as one H5 path segment."""

    text = str(name).strip()
    if not IDENTIFIER_RE.match(text):
        raise ValueError(
            f"Invalid {kind} {name!r}. Use letters, numbers, underscores, dots, "
            "or dashes, starting with a letter."
        )
    if "/" in text or "\\" in text:
        raise ValueError(f"Invalid {kind} {name!r}: path separators are not allowed")
    return text


def label_frame_path(label_set: str, frame: int) -> str:
    """Return the H5 path for one frame of a label set."""

    return f"/labels/{validate_name(label_set, kind='label set')}/{frame_key(frame)}"


def mask_frame_path(mask_set: str, frame: int) -> str:
    """Return the H5 path for one frame of a mask set."""

    return f"/masks/{validate_name(mask_set, kind='mask set')}/{frame_key(frame)}"


def raw_frame_path(frame: int) -> str:
    """Return the H5 path for one embedded raw image frame."""

    return f"/images/raw/{frame_key(frame)}"


def object_set_path(object_set: str) -> str:
    """Return the H5 path for one object set."""

    return f"/object_sets/{validate_name(object_set, kind='object set')}"


def observations_path(object_set: str) -> str:
    """Return the H5 path for one object set's canonical observations."""

    return f"{object_set_path(object_set)}/observations"


def observation_lookup_frame_path(object_set: str, frame: int) -> str:
    """Return the H5 path for one label-id to observation-id lookup frame."""

    return f"{object_set_path(object_set)}/lookup/{frame_key(frame)}"
