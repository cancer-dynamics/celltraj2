"""Headless batch centroid tracking for celltraj2 files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from celltraj2.object_indexing import JsonlReporter
from celltraj2.schema import utc_now_iso
from celltraj2.tracking import default_tracking_run_id, track_minimum_centroid_distance
from celltraj2.trajectory import Trajectory


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {str(key): _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class TrackingFileJob:
    """Centroid-tracking work for one trajectory H5."""

    h5_path: Path
    object_set: str
    track_set: str = "centroid_mindist"
    method: str = "minimum_centroid_distance"
    max_distance: float = 5.0
    coordinate_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    enabled: bool = True
    overwrite: bool = False
    save_outputs: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackingFileJob":
        payload = dict(data)
        path_value = payload.get("h5_path", payload.get("path", payload.get("cell_file")))
        if path_value in (None, ""):
            raise ValueError("Tracking file job requires h5_path")
        object_set = str(payload.get("object_set") or "")
        if not object_set:
            raise ValueError("Tracking file job requires object_set")
        method = str(payload.get("method") or "minimum_centroid_distance").lower()
        aliases = {
            "centroid": "minimum_centroid_distance",
            "mindist": "minimum_centroid_distance",
            "centroid_mindist": "minimum_centroid_distance",
        }
        method = aliases.get(method, method)
        if method != "minimum_centroid_distance":
            raise ValueError(f"Unsupported tracking method {method!r}")
        scale_value = payload.get("coordinate_scale", (1.0, 1.0, 1.0))
        if not isinstance(scale_value, Sequence) or isinstance(scale_value, (str, bytes)) or len(scale_value) != 3:
            raise ValueError("coordinate_scale must contain Z,Y,X values")
        scale = tuple(float(value) for value in scale_value)
        max_distance = float(payload.get("max_distance", payload.get("distcut", 5.0)))
        return cls(
            h5_path=Path(path_value),
            object_set=object_set,
            track_set=str(payload.get("track_set") or "centroid_mindist"),
            method=method,
            max_distance=max_distance,
            coordinate_scale=(scale[0], scale[1], scale[2]),
            enabled=bool(payload.get("enabled", True)),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class TrackingBatchJob:
    """A complete SITE-controlled centroid-tracking job."""

    job_id: str = field(default_factory=default_tracking_run_id)
    files: list[TrackingFileJob] = field(default_factory=list)
    project_root: Path | None = None
    overwrite: bool = False
    save_outputs: bool = True
    fail_fast: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackingBatchJob":
        payload = dict(data)
        files = [
            item if isinstance(item, TrackingFileJob) else TrackingFileJob.from_dict(item)
            for item in payload.get("files", [])
        ]
        root = payload.get("project_root", payload.get("root"))
        return cls(
            job_id=str(payload.get("job_id") or default_tracking_run_id()),
            files=files,
            project_root=None if root in (None, "") else Path(root),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            fail_fast=bool(payload.get("fail_fast", False)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrackingBatchJob":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolved_path(self, file_job: TrackingFileJob) -> Path:
        path = Path(file_job.h5_path)
        if path.is_absolute():
            return path
        if self.project_root is not None:
            return self.project_root / path
        return Path.cwd() / path

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class BatchTrackingSummary:
    """Counts accumulated during a batch tracking run."""

    job_id: str
    files: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    observations: int = 0
    links: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


Reporter = Callable[[Mapping[str, Any]], None]


def load_tracking_job(path: str | Path) -> TrackingBatchJob:
    return TrackingBatchJob.load(path)


def run_batch_tracking(
    job: TrackingBatchJob | Mapping[str, Any],
    *,
    reporter: Reporter | None = None,
) -> BatchTrackingSummary:
    """Run a batch minimum-centroid-distance tracking job."""

    batch_job = job if isinstance(job, TrackingBatchJob) else TrackingBatchJob.from_dict(job)
    emit = reporter or (lambda _event: None)
    summary = BatchTrackingSummary(job_id=batch_job.job_id)
    emit({"event": "job_started", "job_id": batch_job.job_id, "files": len(batch_job.files)})
    for file_job in batch_job.files:
        if not file_job.enabled:
            emit({"event": "file_skipped", "reason": "disabled", "h5_path": str(file_job.h5_path)})
            continue
        h5_path = batch_job.resolved_path(file_job)
        summary.files += 1
        save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
        emit(
            {
                "event": "file_started",
                "job_id": batch_job.job_id,
                "h5_path": str(h5_path),
                "object_set": file_job.object_set,
                "track_set": file_job.track_set,
                "method": file_job.method,
                "max_distance": file_job.max_distance,
                "coordinate_scale": list(file_job.coordinate_scale),
                "save_outputs": save_outputs,
            }
        )
        try:
            _run_file_job(batch_job, file_job, h5_path, summary, emit)
        except Exception as exc:
            summary.failed += 1
            emit({"event": "file_failed", "job_id": batch_job.job_id, "h5_path": str(h5_path), "error": repr(exc)})
            if batch_job.fail_fast:
                raise
    emit({"event": "job_completed", **summary.to_dict()})
    return summary


def _run_file_job(
    batch_job: TrackingBatchJob,
    file_job: TrackingFileJob,
    h5_path: Path,
    summary: BatchTrackingSummary,
    emit: Reporter,
) -> None:
    save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
    overwrite = bool(batch_job.overwrite or file_job.overwrite)
    mode = "r+" if save_outputs else "r"
    with Trajectory(h5_path, mode=mode) as trajectory:
        if not trajectory.object_set(file_job.object_set).has_observations():
            raise FileNotFoundError(f"/object_sets/{file_job.object_set}/observations")
        if save_outputs and trajectory.store.has_track_set(file_job.object_set, file_job.track_set) and not overwrite:
            summary.skipped += 1
            emit(
                {
                    "event": "file_skipped",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "object_set": file_job.object_set,
                    "track_set": file_job.track_set,
                    "reason": "track set already exists",
                }
            )
            return
        result = track_minimum_centroid_distance(
            trajectory,
            file_job.object_set,
            max_distance=file_job.max_distance,
            track_set=file_job.track_set,
            coordinate_scale=file_job.coordinate_scale,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=batch_job.job_id,
            metadata={**batch_job.metadata, **file_job.metadata},
        )
        summary.completed += 1
        summary.observations += result.graph.observation_count
        summary.links += result.link_count
        for frame, counts in result.frame_counts.items():
            emit(
                {
                    "event": "tracking_frame_summary",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "object_set": result.object_set,
                    "track_set": result.track_set,
                    "frame": int(frame),
                    "saved": save_outputs,
                    **counts,
                }
            )
        emit({"event": "file_completed", "job_id": batch_job.job_id, "h5_path": str(h5_path), **result.to_dict()})


__all__ = [
    "BatchTrackingSummary",
    "JsonlReporter",
    "TrackingBatchJob",
    "TrackingFileJob",
    "load_tracking_job",
    "run_batch_tracking",
]
