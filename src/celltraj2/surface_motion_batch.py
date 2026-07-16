"""Headless boundary-motion jobs over existing object track graphs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from celltraj2.object_indexing import JsonlReporter
from celltraj2.h5_access import run_with_stale_retries, snapshot_revisions, validate_revisions
from celltraj2.paths import validate_name
from celltraj2.schema import utc_now_iso
from celltraj2.tracking import compute_boundary_motion, default_tracking_run_id
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
class SurfaceMotionFileJob:
    h5_path: Path
    object_set: str
    track_set: str
    boundary_set: str
    motion_set: str = "surface_ot"
    boundary_source_id: int | None = None
    boundary_source_name: str | None = None
    boundary_source_role: str | None = None
    registration_set: str | None = None
    ot_method: str = "emd"
    sinkhorn_regularization: float = 0.05
    max_boundary_points: int | None = 512
    mass_tolerance: float = 1e-12
    enabled: bool = True
    overwrite: bool = False
    save_outputs: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SurfaceMotionFileJob":
        payload = dict(data)
        path_value = payload.get("h5_path", payload.get("path", payload.get("cell_file")))
        required = {
            "h5_path": path_value,
            "object_set": payload.get("object_set"),
            "track_set": payload.get("track_set"),
            "boundary_set": payload.get("boundary_set"),
        }
        missing = [key for key, value in required.items() if value in (None, "")]
        if missing:
            raise ValueError(f"Surface motion file job requires {', '.join(missing)}")
        max_points = payload.get("max_boundary_points", 512)
        return cls(
            h5_path=Path(path_value),
            object_set=str(payload["object_set"]),
            track_set=str(payload["track_set"]),
            boundary_set=str(payload["boundary_set"]),
            motion_set=str(payload.get("motion_set") or "surface_ot"),
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
            registration_set=(
                None if payload.get("registration_set") in (None, "")
                else str(payload.get("registration_set"))
            ),
            ot_method=str(payload.get("ot_method") or "emd").lower(),
            sinkhorn_regularization=float(payload.get("sinkhorn_regularization", 0.05)),
            max_boundary_points=None if max_points in (None, "") else int(max_points),
            mass_tolerance=float(payload.get("mass_tolerance", 1e-12)),
            enabled=bool(payload.get("enabled", True)),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class SurfaceMotionBatchJob:
    job_id: str = field(default_factory=lambda: default_tracking_run_id().replace("track_", "surface_motion_", 1))
    files: tuple[SurfaceMotionFileJob, ...] = ()
    project_root: Path | None = None
    overwrite: bool = False
    save_outputs: bool = True
    fail_fast: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SurfaceMotionBatchJob":
        payload = dict(data)
        root = payload.get("project_root", payload.get("root"))
        return cls(
            job_id=str(payload.get("job_id") or default_tracking_run_id().replace("track_", "surface_motion_", 1)),
            files=tuple(
                item if isinstance(item, SurfaceMotionFileJob) else SurfaceMotionFileJob.from_dict(item)
                for item in payload.get("files", ())
            ),
            project_root=None if root in (None, "") else Path(root),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            fail_fast=bool(payload.get("fail_fast", False)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SurfaceMotionBatchJob":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolved_path(self, file_job: SurfaceMotionFileJob) -> Path:
        if file_job.h5_path.is_absolute():
            return file_job.h5_path
        return (self.project_root or Path.cwd()) / file_job.h5_path

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class BatchSurfaceMotionSummary:
    job_id: str
    files: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    links: int = 0
    transport_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


Reporter = Callable[[Mapping[str, Any]], None]


def load_surface_motion_job(path: str | Path) -> SurfaceMotionBatchJob:
    return SurfaceMotionBatchJob.load(path)


def run_batch_surface_motion(
    job: SurfaceMotionBatchJob | Mapping[str, Any],
    *,
    reporter: Reporter | None = None,
) -> BatchSurfaceMotionSummary:
    batch_job = job if isinstance(job, SurfaceMotionBatchJob) else SurfaceMotionBatchJob.from_dict(job)
    emit = reporter or (lambda _event: None)
    summary = BatchSurfaceMotionSummary(job_id=batch_job.job_id)
    emit({"event": "job_started", "job_id": batch_job.job_id, "files": len(batch_job.files)})
    for file_job in batch_job.files:
        if not file_job.enabled:
            summary.skipped += 1
            emit({"event": "file_skipped", "h5_path": str(file_job.h5_path), "reason": "disabled"})
            continue
        h5_path = batch_job.resolved_path(file_job)
        summary.files += 1
        save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
        common = {
            "job_id": batch_job.job_id,
            "h5_path": str(h5_path),
            "object_set": file_job.object_set,
            "track_set": file_job.track_set,
            "boundary_set": file_job.boundary_set,
            "motion_set": file_job.motion_set,
            "saved": save_outputs,
        }
        emit(
            {
                **common,
                "event": "file_started",
                "registration_set": file_job.registration_set,
                "ot_method": file_job.ot_method,
                "max_boundary_points": file_job.max_boundary_points,
            }
        )
        try:
            outcome = run_with_stale_retries(
                lambda: _run_surface_motion_file(batch_job, file_job, h5_path, emit, common),
                reporter=emit,
                context=common,
            )
            if outcome is None:
                summary.skipped += 1
                continue
            result, motion_path = outcome
            summary.completed += 1
            summary.links += result.link_count
            summary.transport_edges += result.transport_edge_count
            emit({**common, "event": "file_completed", **result.to_dict(), "saved": save_outputs, "motion_path": motion_path})
        except Exception as exc:
            summary.failed += 1
            emit({**common, "event": "file_failed", "error": repr(exc)})
            if batch_job.fail_fast:
                raise
    emit({"event": "job_completed", **summary.to_dict()})
    return summary


def _run_surface_motion_file(
    batch_job: SurfaceMotionBatchJob,
    file_job: SurfaceMotionFileJob,
    h5_path: Path,
    emit: Reporter,
    common: Mapping[str, Any],
) -> tuple[Any, str | None] | None:
    """Calculate read-only, then hold the canonical H5 only for commit."""

    save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
    overwrite = bool(batch_job.overwrite or file_job.overwrite)
    with Trajectory(
        h5_path,
        mode="r",
        reporter=emit,
        operation="surface_motion_calculation",
        job_id=batch_job.job_id,
    ) as trajectory:
        if (
            save_outputs
            and file_job.motion_set in trajectory.boundary_library(file_job.boundary_set).motion_sets()
            and not overwrite
        ):
            emit({**common, "event": "file_skipped", "reason": "motion set already exists"})
            return None
        selected_registration = file_job.registration_set or trajectory.store.active_registration_name()
        dependency_paths = [
            f"/object_sets/{file_job.object_set}/tracks/{file_job.track_set}",
            f"/boundaries/{file_job.boundary_set}",
            f"/boundaries/{file_job.boundary_set}/motion/{file_job.motion_set}",
        ]
        if selected_registration:
            dependency_paths.append(f"/registrations/{selected_registration}")
        dependencies = snapshot_revisions(trajectory.store, dependency_paths)

        def report_progress(event: Mapping[str, Any]) -> None:
            emit({**common, **dict(event)})

        result = compute_boundary_motion(
            trajectory,
            file_job.object_set,
            file_job.track_set,
            boundary_set=file_job.boundary_set,
            motion_set=file_job.motion_set,
            boundary_source_id=file_job.boundary_source_id,
            boundary_source_name=file_job.boundary_source_name,
            boundary_source_role=file_job.boundary_source_role,
            registration_set=file_job.registration_set,
            ot_method=file_job.ot_method,
            sinkhorn_regularization=file_job.sinkhorn_regularization,
            max_boundary_points=file_job.max_boundary_points,
            mass_tolerance=file_job.mass_tolerance,
            overwrite=overwrite,
            save_outputs=False,
            metadata={**batch_job.metadata, **file_job.metadata},
            progress=report_progress,
        )
        for frame, counts in result.frame_counts.items():
            emit({**common, "event": "surface_motion_frame_summary", "frame": int(frame), **counts})

    motion_path = None
    if save_outputs:
        with Trajectory(
            h5_path,
            mode="r+",
            reporter=emit,
            operation="surface_motion_commit",
            job_id=batch_job.job_id,
        ) as trajectory:
            validate_revisions(trajectory.store, dependencies)
            existing = file_job.motion_set in trajectory.boundary_library(file_job.boundary_set).motion_sets()
            if existing and not overwrite:
                emit({**common, "event": "file_skipped", "reason": "motion set was written by another job"})
                return None
            motion_path = trajectory.store.write_boundary_motion(
                result.boundary_set,
                result.motion_set,
                links=result.links,
                transport=result.transport,
                schema=result.schema,
                overwrite=overwrite,
            )
            trajectory.store.write_json(
                f"/runs/surface_motion/{validate_name(batch_job.job_id, kind='surface motion run')}/run.json",
                {
                    "schema": "celltraj2.surface_motion_run.v1",
                    "job_id": batch_job.job_id,
                    "completed_at": utc_now_iso(),
                    **result.to_dict(),
                    "motion_path": motion_path,
                    "dependencies": dependencies,
                    "boundary_dependency": result.schema.get("boundary_dependency"),
                    "track_dependency": result.schema.get("track_dependency"),
                    "registration_dependency": result.schema.get("registration_dependency"),
                    "metadata": {**batch_job.metadata, **file_job.metadata},
                },
                overwrite=True,
            )
    return result, motion_path


__all__ = [
    "BatchSurfaceMotionSummary",
    "JsonlReporter",
    "SurfaceMotionBatchJob",
    "SurfaceMotionFileJob",
    "load_surface_motion_job",
    "run_batch_surface_motion",
]
