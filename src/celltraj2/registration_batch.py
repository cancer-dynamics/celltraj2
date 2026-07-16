"""Headless batch global registration for celltraj2 files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from celltraj2.object_indexing import JsonlReporter
from celltraj2.h5_access import run_with_stale_retries, snapshot_revisions, validate_revisions
from celltraj2.registration import default_registration_run_id, register_global_translation
from celltraj2.schema import utc_now_iso
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
class RegistrationFileJob:
    """Global-registration work for one trajectory H5."""

    h5_path: Path
    object_set: str
    registration_set: str = "global_registration"
    method: str = "pairwise_symmetric_nearest_neighbor_translation"
    max_shift_per_frame: float = 10.0
    grid_step: float = 1.0
    coordinate_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    distance_unit: str = "pixel"
    contact_transform: bool = False
    contact_r0: float = 100.0
    contact_d0: float = 100.0
    enabled: bool = True
    overwrite: bool = False
    save_outputs: bool = True
    set_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistrationFileJob":
        payload = dict(data)
        path_value = payload.get("h5_path", payload.get("path", payload.get("cell_file")))
        if path_value in (None, ""):
            raise ValueError("Registration file job requires h5_path")
        object_set = str(payload.get("object_set") or "")
        if not object_set:
            raise ValueError("Registration file job requires object_set")
        method = str(payload.get("method") or "pairwise_symmetric_nearest_neighbor_translation").lower()
        aliases = {
            "global_translation": "pairwise_symmetric_nearest_neighbor_translation",
            "pairwise_distance": "pairwise_symmetric_nearest_neighbor_translation",
        }
        method = aliases.get(method, method)
        if method != "pairwise_symmetric_nearest_neighbor_translation":
            raise ValueError(f"Unsupported registration method {method!r}")
        scale_value = payload.get("coordinate_scale", (1.0, 1.0, 1.0))
        if not isinstance(scale_value, Sequence) or isinstance(scale_value, (str, bytes)) or len(scale_value) != 3:
            raise ValueError("coordinate_scale must contain Z,Y,X values")
        scale = tuple(float(value) for value in scale_value)
        metadata = dict(payload.get("metadata") or {})
        return cls(
            h5_path=Path(path_value),
            object_set=object_set,
            registration_set=str(payload.get("registration_set") or "global_registration"),
            method=method,
            max_shift_per_frame=float(payload.get("max_shift_per_frame", payload.get("max_shift", 10.0))),
            grid_step=float(payload.get("grid_step", 1.0)),
            coordinate_scale=(scale[0], scale[1], scale[2]),
            distance_unit=str(payload.get("distance_unit") or metadata.get("distance_unit") or "pixel"),
            contact_transform=bool(payload.get("contact_transform", False)),
            contact_r0=float(payload.get("contact_r0", 100.0)),
            contact_d0=float(payload.get("contact_d0", 100.0)),
            enabled=bool(payload.get("enabled", True)),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            set_active=bool(payload.get("set_active", True)),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class RegistrationBatchJob:
    """A complete SITE-controlled global-registration job."""

    job_id: str = field(default_factory=default_registration_run_id)
    files: list[RegistrationFileJob] = field(default_factory=list)
    project_root: Path | None = None
    overwrite: bool = False
    save_outputs: bool = True
    fail_fast: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistrationBatchJob":
        payload = dict(data)
        files = [
            item if isinstance(item, RegistrationFileJob) else RegistrationFileJob.from_dict(item)
            for item in payload.get("files", [])
        ]
        root = payload.get("project_root", payload.get("root"))
        return cls(
            job_id=str(payload.get("job_id") or default_registration_run_id()),
            files=files,
            project_root=None if root in (None, "") else Path(root),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            fail_fast=bool(payload.get("fail_fast", False)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegistrationBatchJob":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolved_path(self, file_job: RegistrationFileJob) -> Path:
        path = Path(file_job.h5_path)
        if path.is_absolute():
            return path
        if self.project_root is not None:
            return self.project_root / path
        return Path.cwd() / path

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class BatchRegistrationSummary:
    job_id: str
    files: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    frames: int = 0
    estimated_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


Reporter = Callable[[Mapping[str, Any]], None]


def load_registration_job(path: str | Path) -> RegistrationBatchJob:
    return RegistrationBatchJob.load(path)


def run_batch_registration(
    job: RegistrationBatchJob | Mapping[str, Any],
    *,
    reporter: Reporter | None = None,
) -> BatchRegistrationSummary:
    """Run a batch pairwise-centroid global-registration job."""

    batch_job = job if isinstance(job, RegistrationBatchJob) else RegistrationBatchJob.from_dict(job)
    emit = reporter or (lambda _event: None)
    summary = BatchRegistrationSummary(job_id=batch_job.job_id)
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
                "registration_set": file_job.registration_set,
                "method": file_job.method,
                "max_shift_per_frame": file_job.max_shift_per_frame,
                "grid_step": file_job.grid_step,
                "coordinate_scale": list(file_job.coordinate_scale),
                "distance_unit": file_job.distance_unit,
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
    batch_job: RegistrationBatchJob,
    file_job: RegistrationFileJob,
    h5_path: Path,
    summary: BatchRegistrationSummary,
    emit: Reporter,
) -> None:
    save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
    overwrite = bool(batch_job.overwrite or file_job.overwrite)
    with Trajectory(
        h5_path,
        mode="r",
        reporter=emit,
        operation="registration_calculation",
        job_id=batch_job.job_id,
    ) as trajectory:
        if not trajectory.object_set(file_job.object_set).has_observations():
            raise FileNotFoundError(f"/object_sets/{file_job.object_set}/observations")
        if save_outputs and trajectory.store.has_registration_set(file_job.registration_set) and not overwrite:
            summary.skipped += 1
            emit(
                {
                    "event": "file_skipped",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "registration_set": file_job.registration_set,
                    "reason": "registration set already exists",
                }
            )
            return
        dependencies = snapshot_revisions(
            trajectory.store,
            [f"/object_sets/{file_job.object_set}/observations"],
        )

        def report_frame(frame_event: Mapping[str, Any]) -> None:
            emit(
                {
                    "event": "registration_frame_summary",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "registration_set": file_job.registration_set,
                    "save_outputs": save_outputs,
                    **dict(frame_event),
                }
            )

        result = register_global_translation(
            trajectory,
            file_job.object_set,
            registration_set=file_job.registration_set,
            max_shift_per_frame=file_job.max_shift_per_frame,
            grid_step=file_job.grid_step,
            coordinate_scale=file_job.coordinate_scale,
            distance_unit=file_job.distance_unit,
            contact_transform=file_job.contact_transform,
            contact_r0=file_job.contact_r0,
            contact_d0=file_job.contact_d0,
            overwrite=overwrite,
            save_outputs=False,
            set_active=file_job.set_active,
            run_id=batch_job.job_id,
            metadata={**batch_job.metadata, **file_job.metadata},
            progress=report_frame,
        )
    registration_path = None
    active = False
    if save_outputs:
        with Trajectory(
            h5_path,
            mode="r+",
            reporter=emit,
            operation="registration_commit",
            job_id=batch_job.job_id,
        ) as trajectory:
            validate_revisions(trajectory.store, dependencies)
            if trajectory.store.has_registration_set(file_job.registration_set) and not overwrite:
                summary.skipped += 1
                emit(
                    {
                        "event": "file_skipped",
                        "job_id": batch_job.job_id,
                        "h5_path": str(h5_path),
                        "registration_set": file_job.registration_set,
                        "reason": "registration set was written by another job",
                    }
                )
                return
            registration_path = trajectory.store.write_registration_set(result.registration, overwrite=overwrite)
            if file_job.set_active:
                trajectory.store.set_active_registration(
                    result.registration.name,
                    reason="registration_run",
                    run_id=batch_job.job_id,
                )
                active = True
            trajectory.store.write_registration_run(
                batch_job.job_id,
                {
                    "schema": "celltraj2.registration_run.v1",
                    "run_id": batch_job.job_id,
                    "status": "completed",
                    "completed_at": utc_now_iso(),
                    "h5_path": str(h5_path),
                    "object_set": file_job.object_set,
                    "registration_set": result.registration.name,
                    "registration_digest": result.registration.digest,
                    "registration_path": registration_path,
                    "set_active": bool(file_job.set_active),
                    "dependencies": dependencies,
                    "schema_record": result.registration.schema,
                    "metadata": {**batch_job.metadata, **file_job.metadata},
                },
                overwrite=True,
            )
    payload = {
        **result.to_dict(),
        "saved": save_outputs,
        "active": active,
        "registration_path": registration_path,
    }
    summary.completed += 1
    summary.frames += int(payload["frame_count"])
    summary.estimated_frames += int(payload["estimated_frame_count"])
    emit({"event": "file_completed", "job_id": batch_job.job_id, "h5_path": str(h5_path), **payload})


__all__ = [
    "BatchRegistrationSummary",
    "JsonlReporter",
    "RegistrationBatchJob",
    "RegistrationFileJob",
    "load_registration_job",
    "run_batch_registration",
]
