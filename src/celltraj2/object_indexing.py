"""Headless batch object indexing for celltraj2 files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from celltraj2.objects import default_object_index_run_id, index_object_set
from celltraj2.h5_access import run_with_stale_retries, snapshot_revisions, validate_revisions
from celltraj2.reporting import JsonlReporter
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
class ObjectIndexFileJob:
    """Object-indexing work for one trajectory H5."""

    h5_path: Path
    object_set: str = "segmentation"
    source_label_set: str | None = None
    enabled: bool = True
    overwrite: bool = False
    save_outputs: bool = True
    frames: dict[str, Any] = field(default_factory=lambda: {"mode": "all"})
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectIndexFileJob":
        payload = dict(data)
        path_value = payload.get("h5_path", payload.get("path", payload.get("cell_file")))
        if path_value in (None, ""):
            raise ValueError("Object-indexing file job requires h5_path")
        object_set = str(payload.get("object_set") or payload.get("label_set") or "segmentation")
        source_label_set = payload.get("source_label_set")
        if source_label_set in (None, ""):
            source_label_set = payload.get("label_set") or object_set
        frames = payload.get("frames")
        if not isinstance(frames, Mapping):
            frames = {
                "mode": payload.get("frame_mode", "all"),
                "frame_start": payload.get("frame_start"),
                "frame_stop": payload.get("frame_stop"),
                "frame_list": payload.get("frame_list"),
            }
        return cls(
            h5_path=Path(path_value),
            object_set=object_set,
            source_label_set=str(source_label_set),
            enabled=bool(payload.get("enabled", True)),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            frames=dict(frames or {"mode": "all"}),
            metadata=dict(payload.get("metadata") or {}),
        )

    @property
    def source_labels(self) -> str:
        return str(self.source_label_set or self.object_set)

    def frame_numbers(self, frame_count: int, *, available_frames: Sequence[int] | None = None) -> list[int]:
        """Return validated one-based frames for this file."""

        if not self.enabled:
            return []
        count = max(1, int(frame_count or 1))
        frames = dict(self.frames or {})
        mode = str(frames.get("mode") or "all").lower()
        explicit = frames.get("frames")
        if explicit is not None:
            values = _parse_frame_values(explicit)
        elif mode == "list":
            values = _parse_frame_values(frames.get("frame_list", ""))
        elif mode == "range":
            start = int(frames.get("frame_start") or 1)
            stop = int(frames.get("frame_stop") or count)
            values = list(range(start, stop + 1))
        elif available_frames is not None:
            values = [int(frame) for frame in available_frames]
        else:
            values = list(range(1, count + 1))
        invalid = [frame for frame in values if int(frame) < 1 or int(frame) > count]
        if invalid:
            raise ValueError(f"Frame(s) outside 1..{count}: {invalid}")
        return sorted(set(int(frame) for frame in values))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class ObjectIndexBatchJob:
    """A complete SITE-controlled object-indexing job."""

    job_id: str = field(default_factory=default_object_index_run_id)
    files: list[ObjectIndexFileJob] = field(default_factory=list)
    project_root: Path | None = None
    overwrite: bool = False
    save_outputs: bool = True
    fail_fast: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectIndexBatchJob":
        payload = dict(data)
        files = [
            item if isinstance(item, ObjectIndexFileJob) else ObjectIndexFileJob.from_dict(item)
            for item in payload.get("files", [])
        ]
        root = payload.get("project_root", payload.get("root"))
        return cls(
            job_id=str(payload.get("job_id") or default_object_index_run_id()),
            files=files,
            project_root=None if root in (None, "") else Path(root),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            fail_fast=bool(payload.get("fail_fast", False)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ObjectIndexBatchJob":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolved_path(self, file_job: ObjectIndexFileJob) -> Path:
        path = Path(file_job.h5_path)
        if path.is_absolute():
            return path
        if self.project_root is not None:
            return self.project_root / path
        return Path.cwd() / path

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class BatchObjectIndexSummary:
    """Counts accumulated during a batch object-indexing run."""

    job_id: str
    files: int = 0
    frames: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


Reporter = Callable[[Mapping[str, Any]], None]


def load_object_index_job(path: str | Path) -> ObjectIndexBatchJob:
    return ObjectIndexBatchJob.load(path)


def run_batch_object_indexing(
    job: ObjectIndexBatchJob | Mapping[str, Any],
    *,
    reporter: Reporter | None = None,
) -> BatchObjectIndexSummary:
    """Run a batch object-indexing job."""

    batch_job = job if isinstance(job, ObjectIndexBatchJob) else ObjectIndexBatchJob.from_dict(job)
    emit = reporter or (lambda _event: None)
    summary = BatchObjectIndexSummary(job_id=batch_job.job_id)
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
                "source_label_set": file_job.source_labels,
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
    batch_job: ObjectIndexBatchJob,
    file_job: ObjectIndexFileJob,
    h5_path: Path,
    summary: BatchObjectIndexSummary,
    emit: Reporter,
) -> None:
    save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
    overwrite = bool(batch_job.overwrite or file_job.overwrite)
    with Trajectory(
        h5_path,
        mode="r",
        reporter=emit,
        operation="object_index_calculation",
        job_id=batch_job.job_id,
    ) as trajectory:
        available_frames = trajectory.label_frames(file_job.source_labels)
        frames = file_job.frame_numbers(int(trajectory.metadata.frame_count or 1), available_frames=available_frames)
        if save_outputs and trajectory.store.has_observations(file_job.object_set) and not overwrite:
            summary.frames += len(frames)
            summary.skipped += 1
            emit(
                {
                    "event": "file_skipped",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "object_set": file_job.object_set,
                    "reason": "observations already exist",
                }
            )
            return

        dependencies = snapshot_revisions(
            trajectory.store,
            [
                "/metadata/celltraj2.json",
                *[f"/labels/{file_job.source_labels}/frame_{int(frame)}" for frame in frames],
            ],
        )

        def report_frame(frame_event: Mapping[str, Any]) -> None:
            emit(
                {
                    "event": "frame_completed",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "saved": False,
                    **dict(frame_event),
                }
            )

        result = index_object_set(
            trajectory,
            file_job.object_set,
            source_label_set=file_job.source_labels,
            frames=frames,
            overwrite=overwrite,
            save_outputs=False,
            run_id=batch_job.job_id,
            metadata={**batch_job.metadata, **file_job.metadata},
            progress=report_frame,
        )
    lookup_paths: dict[int, str] = {}
    observations_path = None
    if save_outputs:
        with Trajectory(
            h5_path,
            mode="r+",
            reporter=emit,
            operation="object_index_commit",
            job_id=batch_job.job_id,
        ) as trajectory:
            validate_revisions(trajectory.store, dependencies)
            if trajectory.store.has_observations(file_job.object_set) and not overwrite:
                summary.frames += len(result.frames)
                summary.skipped += 1
                emit(
                    {
                        "event": "file_skipped",
                        "job_id": batch_job.job_id,
                        "h5_path": str(h5_path),
                        "object_set": file_job.object_set,
                        "reason": "observations were written by another job",
                    }
                )
                return
            observations_path = trajectory.store.write_observations(
                result.object_set,
                result.observations,
                result.schema,
                source_label_set=result.source_label_set,
                overwrite=overwrite,
                metadata={**batch_job.metadata, **file_job.metadata},
            )
            if overwrite:
                trajectory.store.clear_observation_lookup_frames(result.object_set)
            for frame in result.frames:
                lookup_paths[frame] = trajectory.store.write_observation_lookup_frame(
                    result.object_set,
                    frame,
                    result.lookups[frame],
                    overwrite=overwrite,
                )
            trajectory.store.write_object_indexing_run(
                batch_job.job_id,
                {
                    "schema": "celltraj2.object_indexing_run.v1",
                    "run_id": batch_job.job_id,
                    "status": "completed",
                    "completed_at": utc_now_iso(),
                    "h5_path": str(h5_path),
                    "object_set": result.object_set,
                    "source_label_set": result.source_label_set,
                    "frames": result.frames,
                    "observation_count": result.observation_count,
                    "frame_counts": result.frame_counts,
                    "observations_path": observations_path,
                    "dependencies": dependencies,
                    "metadata": {**batch_job.metadata, **file_job.metadata},
                },
                overwrite=True,
            )
    summary.frames += len(result.frames)
    summary.completed += len(result.frames)
    summary.observations += result.observation_count
    emit(
        {
            "event": "file_completed",
            "job_id": batch_job.job_id,
            "h5_path": str(h5_path),
            **result.to_dict(),
            "saved": save_outputs,
            "observations_path": observations_path,
            "lookup_paths": {str(key): value for key, value in lookup_paths.items()},
        }
    )


def _parse_frame_values(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    frames: list[int] = []
    for part in str(value).replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            start_text, stop_text = text.split("-", 1)
            start = int(start_text.strip())
            stop = int(stop_text.strip())
            frames.extend(range(start, stop + 1))
        else:
            frames.append(int(text))
    return frames
