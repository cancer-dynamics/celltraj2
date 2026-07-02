"""Dependency-light metadata models for celltraj2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "0.1.0"

ImageSourceType = Literal["embedded_h5", "roi_ome_zarr", "roi_tiff", "linked_nd2", "in_memory"]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _coerce_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def _dataclass_from_dict(cls: type[Any], data: Mapping[str, Any]) -> Any:
    names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in dict(data).items() if key in names})


@dataclass(frozen=True)
class RoiBounds:
    """Zero-based half-open parent-acquisition ROI bounds."""

    z_start: int = 0
    z_stop: int = 1
    y_start: int = 0
    y_stop: int = 1
    x_start: int = 0
    x_stop: int = 1

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (
            max(0, int(self.z_stop) - int(self.z_start)),
            max(0, int(self.y_stop) - int(self.y_start)),
            max(0, int(self.x_stop) - int(self.x_start)),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoiBounds":
        return _dataclass_from_dict(cls, data)


@dataclass(frozen=True)
class ChannelSpec:
    """One channel copied from SITE metadata or user-defined metadata."""

    raw_index: int
    raw_name: str | None = None
    display_name: str | None = None
    role: str | None = None
    target: str | None = None
    readout: str | None = None
    fluorophore: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelSpec":
        return _dataclass_from_dict(cls, data)


@dataclass(frozen=True)
class RoiSpec:
    """One SITE ROI copied into a celltraj2 file."""

    roi_id: str
    dataset_id: str | None = None
    position_index: int = 0
    position_label: str | None = None
    roi_label: str | None = None
    time_start: int = 0
    time_stop: int | None = None
    bounds: RoiBounds = field(default_factory=RoiBounds)
    storage_mode: str = "linked_nd2"
    artifact_path: Path | None = None
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoiSpec":
        payload = dict(data)
        if isinstance(payload.get("bounds"), Mapping):
            payload["bounds"] = RoiBounds.from_dict(payload["bounds"])
        payload["artifact_path"] = _coerce_path(payload.get("artifact_path"))
        payload["source_path"] = _coerce_path(payload.get("source_path"))
        return _dataclass_from_dict(cls, payload)


@dataclass(frozen=True)
class ImageSourceSpec:
    """How raw ROI image pixels should be accessed."""

    source_type: ImageSourceType
    path: Path | None = None
    axes: tuple[str, ...] = ()
    sizes: dict[str, int] = field(default_factory=dict)
    dtype: str | None = None
    dataset_path: str | None = None
    roi: RoiSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageSourceSpec":
        payload = dict(data)
        payload["path"] = _coerce_path(payload.get("path"))
        payload["axes"] = tuple(str(axis) for axis in payload.get("axes", ()) or ())
        payload["sizes"] = {str(key): int(value) for key, value in dict(payload.get("sizes", {}) or {}).items()}
        if isinstance(payload.get("roi"), Mapping):
            payload["roi"] = RoiSpec.from_dict(payload["roi"])
        return _dataclass_from_dict(cls, payload)


@dataclass(frozen=True)
class SegmentationRunSpec:
    """Configuration/provenance stub for a segmentation run."""

    run_id: str
    label_set: str
    backend: str
    frame_start: int = 1
    frame_stop: int | None = None
    channel_specs: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentationRunSpec":
        return _dataclass_from_dict(cls, data)


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Top-level metadata for one per-ROI analysis H5."""

    roi_id: str
    dataset_id: str
    frame_count: int = 1
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    channels: list[ChannelSpec] = field(default_factory=list)
    roi: RoiSpec | None = None
    image_source: ImageSourceSpec | None = None
    acquisition: dict[str, Any] = field(default_factory=dict)
    treatments: list[dict[str, Any]] = field(default_factory=list)
    notes: str | None = None

    def frame_map(self) -> list[dict[str, int]]:
        """Return local one-based frame to parent zero-based T mapping."""

        start = 0 if self.roi is None else int(self.roi.time_start or 0)
        return [
            {"frame": frame, "parent_time_index": start + frame - 1}
            for frame in range(1, int(self.frame_count) + 1)
        ]

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe(self)
        payload["frame_map"] = self.frame_map()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectoryMetadata":
        payload = dict(data)
        payload["channels"] = [
            ChannelSpec.from_dict(item) if isinstance(item, Mapping) else item
            for item in payload.get("channels", [])
        ]
        if isinstance(payload.get("roi"), Mapping):
            payload["roi"] = RoiSpec.from_dict(payload["roi"])
        if isinstance(payload.get("image_source"), Mapping):
            payload["image_source"] = ImageSourceSpec.from_dict(payload["image_source"])
        payload.pop("frame_map", None)
        return _dataclass_from_dict(cls, payload)


def channels_from_site(items: Sequence[Mapping[str, Any]] | None) -> list[ChannelSpec]:
    """Return channel specs copied from SITE channel dictionaries."""

    channels: list[ChannelSpec] = []
    for item in items or []:
        raw_index = int(item.get("raw_index", len(channels)))
        channels.append(
            ChannelSpec(
                raw_index=raw_index,
                raw_name=item.get("raw_name"),
                display_name=item.get("display_name"),
                role=item.get("role"),
                target=item.get("target"),
                readout=item.get("readout"),
                fluorophore=item.get("fluorophore"),
                category=item.get("category"),
                metadata=dict(item),
            )
        )
    return channels
