"""Headless batch object tracking for celltraj2 files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from celltraj2.object_indexing import JsonlReporter
from celltraj2.h5_access import run_with_stale_retries, snapshot_revisions, validate_revisions
from celltraj2.schema import utc_now_iso
from celltraj2.tracking import (
    default_tracking_run_id,
    track_minimum_boundary_ot_cost,
    track_minimum_centroid_distance,
)
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
    """One centroid- or boundary-OT tracking task for a trajectory H5."""

    h5_path: Path
    object_set: str
    track_set: str = "centroid_mindist"
    method: str = "minimum_centroid_distance"
    max_distance: float = 5.0
    coordinate_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    registration_set: str | None = None
    boundary_set: str | None = None
    boundary_source_id: int | None = None
    boundary_source_name: str | None = None
    boundary_source_role: str | None = None
    ot_cost_cutoff: float = float("inf")
    ot_method: str = "emd"
    sinkhorn_regularization: float = 0.05
    max_boundary_points: int | None = 512
    mass_tolerance: float = 1e-12
    save_motion: bool = False
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
            "boundary": "minimum_registered_boundary_ot_cost",
            "boundary_ot": "minimum_registered_boundary_ot_cost",
            "ot": "minimum_registered_boundary_ot_cost",
        }
        method = aliases.get(method, method)
        if method not in {"minimum_centroid_distance", "minimum_registered_boundary_ot_cost"}:
            raise ValueError(f"Unsupported tracking method {method!r}")
        scale_value = payload.get("coordinate_scale", (1.0, 1.0, 1.0))
        if not isinstance(scale_value, Sequence) or isinstance(scale_value, (str, bytes)) or len(scale_value) != 3:
            raise ValueError("coordinate_scale must contain Z,Y,X values")
        scale = tuple(float(value) for value in scale_value)
        max_distance = float(payload.get("max_distance", payload.get("distcut", 5.0)))
        max_boundary_points_value = payload.get("max_boundary_points", 512)
        return cls(
            h5_path=Path(path_value),
            object_set=object_set,
            track_set=str(payload.get("track_set") or "centroid_mindist"),
            method=method,
            max_distance=max_distance,
            coordinate_scale=(scale[0], scale[1], scale[2]),
            registration_set=(
                None if payload.get("registration_set") in (None, "") else str(payload.get("registration_set"))
            ),
            boundary_set=(
                None if payload.get("boundary_set") in (None, "") else str(payload.get("boundary_set"))
            ),
            boundary_source_id=(
                None if payload.get("boundary_source_id") in (None, "")
                else int(payload.get("boundary_source_id"))
            ),
            boundary_source_name=(
                None if payload.get("boundary_source_name") in (None, "")
                else str(payload.get("boundary_source_name"))
            ),
            boundary_source_role=(
                None if payload.get("boundary_source_role") in (None, "")
                else str(payload.get("boundary_source_role"))
            ),
            ot_cost_cutoff=float(payload.get("ot_cost_cutoff", float("inf"))),
            ot_method=str(payload.get("ot_method") or "emd").lower(),
            sinkhorn_regularization=float(payload.get("sinkhorn_regularization", 0.05)),
            max_boundary_points=(
                None if max_boundary_points_value in (None, "") else int(max_boundary_points_value)
            ),
            mass_tolerance=float(payload.get("mass_tolerance", 1e-12)),
            save_motion=bool(payload.get("save_motion", False)),
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
                "distance_unit": str(file_job.metadata.get("distance_unit") or "scaled_coordinate_unit"),
                "registration_set": file_job.registration_set,
                "boundary_set": file_job.boundary_set,
                "boundary_source_name": file_job.boundary_source_name,
                "ot_method": file_job.ot_method if file_job.method == "minimum_registered_boundary_ot_cost" else None,
                "max_boundary_points": (
                    file_job.max_boundary_points
                    if file_job.method == "minimum_registered_boundary_ot_cost" else None
                ),
                "save_outputs": save_outputs,
            }
        )
        try:
            run_with_stale_retries(
                lambda: _run_file_job(batch_job, file_job, h5_path, summary, emit),
                reporter=emit,
                context={"job_id": batch_job.job_id, "h5_path": str(h5_path)},
            )
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
    with Trajectory(
        h5_path,
        mode="r",
        reporter=emit,
        operation="tracking_calculation",
        job_id=batch_job.job_id,
    ) as trajectory:
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
        dependency_paths = [f"/object_sets/{file_job.object_set}/observations"]
        selected_registration = file_job.registration_set or trajectory.store.active_registration_name()
        if selected_registration:
            dependency_paths.append(f"/registrations/{selected_registration}")
        if file_job.method == "minimum_registered_boundary_ot_cost" and file_job.boundary_set:
            dependency_paths.append(f"/boundaries/{file_job.boundary_set}")
        dependencies = snapshot_revisions(trajectory.store, dependency_paths)

        def report_frame(frame_event: Mapping[str, Any]) -> None:
            emit(
                {
                    "event": "tracking_frame_summary",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "object_set": file_job.object_set,
                    "track_set": file_job.track_set,
                    "saved": False,
                    **dict(frame_event),
                }
            )

        if file_job.method == "minimum_centroid_distance":
            result = track_minimum_centroid_distance(
                trajectory,
                file_job.object_set,
                max_distance=file_job.max_distance,
                track_set=file_job.track_set,
                coordinate_scale=file_job.coordinate_scale,
                registration_set=file_job.registration_set,
                overwrite=overwrite,
                save_outputs=False,
                run_id=batch_job.job_id,
                metadata={**batch_job.metadata, **file_job.metadata},
                progress=report_frame,
            )
        else:
            result = track_minimum_boundary_ot_cost(
                trajectory,
                file_job.object_set,
                boundary_set=file_job.boundary_set,
                boundary_source_id=file_job.boundary_source_id,
                boundary_source_name=file_job.boundary_source_name,
                boundary_source_role=file_job.boundary_source_role,
                max_distance=file_job.max_distance,
                ot_cost_cutoff=file_job.ot_cost_cutoff,
                track_set=file_job.track_set,
                registration_set=file_job.registration_set,
                ot_method=file_job.ot_method,
                sinkhorn_regularization=file_job.sinkhorn_regularization,
                max_boundary_points=file_job.max_boundary_points,
                mass_tolerance=file_job.mass_tolerance,
                save_motion=file_job.save_motion,
                overwrite=overwrite,
                save_outputs=False,
                run_id=batch_job.job_id,
                metadata={**batch_job.metadata, **file_job.metadata},
                progress=report_frame,
            )
    track_path = None
    motion_path = None
    if save_outputs:
        with Trajectory(
            h5_path,
            mode="r+",
            reporter=emit,
            operation="tracking_commit",
            job_id=batch_job.job_id,
        ) as trajectory:
            validate_revisions(trajectory.store, dependencies)
            if trajectory.store.has_track_set(result.object_set, result.track_set) and not overwrite:
                summary.skipped += 1
                emit(
                    {
                        "event": "file_skipped",
                        "job_id": batch_job.job_id,
                        "h5_path": str(h5_path),
                        "object_set": result.object_set,
                        "track_set": result.track_set,
                        "reason": "track set was written by another job",
                    }
                )
                return
            track_path = trajectory.store.write_track_graph(
                result.object_set,
                result.track_set,
                adjacency=result.graph.adjacency,
                links=result.graph.links,
                assignments=result.graph.assignments,
                schema=result.graph.schema,
                overwrite=overwrite,
            )
            if result.motion_result is not None:
                motion_path = trajectory.store.write_boundary_motion(
                    result.motion_result.boundary_set,
                    result.motion_result.motion_set,
                    links=result.motion_result.links,
                    transport=result.motion_result.transport,
                    schema=result.motion_result.schema,
                    overwrite=overwrite,
                )
            trajectory.store.write_tracking_run(
                batch_job.job_id,
                {
                    "schema": "celltraj2.tracking_run.v1",
                    "run_id": batch_job.job_id,
                    "status": "completed",
                    "completed_at": utc_now_iso(),
                    "h5_path": str(h5_path),
                    "object_set": result.object_set,
                    "track_set": result.track_set,
                    "method": file_job.method,
                    "observation_count": result.graph.observation_count,
                    "link_count": result.link_count,
                    "frame_counts": result.frame_counts,
                    "track_path": track_path,
                    "motion_path": motion_path,
                    "dependencies": dependencies,
                    "graph_schema": result.graph.schema,
                    "metadata": {**batch_job.metadata, **file_job.metadata},
                },
                overwrite=True,
            )
    summary.completed += 1
    summary.observations += result.graph.observation_count
    summary.links += result.link_count
    emit(
        {
            "event": "file_completed",
            "job_id": batch_job.job_id,
            "h5_path": str(h5_path),
            **result.to_dict(),
            "saved": save_outputs,
            "track_path": track_path,
            "motion_path": motion_path,
        }
    )


__all__ = [
    "BatchTrackingSummary",
    "JsonlReporter",
    "TrackingBatchJob",
    "TrackingFileJob",
    "load_tracking_job",
    "run_batch_tracking",
]
