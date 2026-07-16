"""Headless batch feature extraction for celltraj2 files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from celltraj2.features import FeatureSetSpec, default_feature_extraction_run_id, extract_feature_set
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
class FeatureExtractionFileJob:
    """Feature-extraction work for one trajectory H5."""

    h5_path: Path
    feature_spec: FeatureSetSpec
    enabled: bool = True
    overwrite: bool = False
    save_outputs: bool = True
    frames: dict[str, Any] = field(default_factory=lambda: {"mode": "all"})
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureExtractionFileJob":
        payload = dict(data)
        path_value = payload.get("h5_path", payload.get("path", payload.get("cell_file")))
        if path_value in (None, ""):
            raise ValueError("Feature-extraction file job requires h5_path")
        spec_payload = payload.get("feature_spec")
        if not isinstance(spec_payload, Mapping):
            spec_payload = {
                "feature_set": payload.get("feature_set"),
                "object_set": payload.get("object_set"),
                "source_label_set": payload.get("source_label_set"),
                "features": payload.get("features", []),
                "frames": payload.get("frames", {"mode": "all"}),
                "metadata": payload.get("metadata", {}),
            }
        feature_spec = FeatureSetSpec.from_dict(spec_payload)
        frames = payload.get("frames", feature_spec.frames)
        if not isinstance(frames, Mapping):
            frames = {
                "mode": payload.get("frame_mode", "all"),
                "frame_start": payload.get("frame_start"),
                "frame_stop": payload.get("frame_stop"),
                "frame_list": payload.get("frame_list"),
            }
        return cls(
            h5_path=Path(path_value),
            feature_spec=feature_spec,
            enabled=bool(payload.get("enabled", True)),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            frames=dict(frames or {"mode": "all"}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def frame_numbers(self, frame_count: int, *, available_frames: Sequence[int] | None = None) -> list[int]:
        """Return validated one-based frames for this file."""

        if not self.enabled:
            return []
        count = max(1, int(frame_count or 1))
        frames = dict(self.frames or self.feature_spec.frames or {})
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
class FeatureExtractionBatchJob:
    """A complete SITE-controlled feature-extraction job."""

    job_id: str = field(default_factory=default_feature_extraction_run_id)
    files: list[FeatureExtractionFileJob] = field(default_factory=list)
    project_root: Path | None = None
    overwrite: bool = False
    save_outputs: bool = True
    fail_fast: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureExtractionBatchJob":
        payload = dict(data)
        files = [
            item if isinstance(item, FeatureExtractionFileJob) else FeatureExtractionFileJob.from_dict(item)
            for item in payload.get("files", [])
        ]
        root = payload.get("project_root", payload.get("root"))
        return cls(
            job_id=str(payload.get("job_id") or default_feature_extraction_run_id()),
            files=files,
            project_root=None if root in (None, "") else Path(root),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            fail_fast=bool(payload.get("fail_fast", False)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureExtractionBatchJob":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolved_path(self, file_job: FeatureExtractionFileJob) -> Path:
        path = Path(file_job.h5_path)
        if path.is_absolute():
            return path
        if self.project_root is not None:
            return self.project_root / path
        return Path.cwd() / path

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class BatchFeatureExtractionSummary:
    """Counts accumulated during a batch feature-extraction run."""

    job_id: str
    files: int = 0
    frames: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    features: int = 0
    observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


Reporter = Callable[[Mapping[str, Any]], None]


def load_feature_extraction_job(path: str | Path) -> FeatureExtractionBatchJob:
    return FeatureExtractionBatchJob.load(path)


def run_batch_feature_extraction(
    job: FeatureExtractionBatchJob | Mapping[str, Any],
    *,
    reporter: Reporter | None = None,
) -> BatchFeatureExtractionSummary:
    """Run a batch feature-extraction job."""

    batch_job = job if isinstance(job, FeatureExtractionBatchJob) else FeatureExtractionBatchJob.from_dict(job)
    emit = reporter or (lambda _event: None)
    summary = BatchFeatureExtractionSummary(job_id=batch_job.job_id)
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
                "object_set": file_job.feature_spec.object_set,
                "feature_set": file_job.feature_spec.feature_set,
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
    batch_job: FeatureExtractionBatchJob,
    file_job: FeatureExtractionFileJob,
    h5_path: Path,
    summary: BatchFeatureExtractionSummary,
    emit: Reporter,
) -> None:
    save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
    overwrite = bool(batch_job.overwrite or file_job.overwrite)
    with Trajectory(
        h5_path,
        mode="r",
        reporter=emit,
        operation="feature_calculation",
        job_id=batch_job.job_id,
    ) as trajectory:
        available_frames = trajectory.object_set(file_job.feature_spec.object_set).lookup_frames()
        frames = file_job.frame_numbers(int(trajectory.metadata.frame_count or 1), available_frames=available_frames)
        if save_outputs and trajectory.store.has_feature_set(
            file_job.feature_spec.object_set,
            file_job.feature_spec.feature_set,
        ) and not overwrite:
            summary.frames += len(frames)
            summary.skipped += 1
            emit(
                {
                    "event": "file_skipped",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "object_set": file_job.feature_spec.object_set,
                    "feature_set": file_job.feature_spec.feature_set,
                    "reason": "feature set already exists",
                }
            )
            return

        dependency_paths = _feature_dependency_paths(file_job.feature_spec, frames)
        image_source = trajectory.metadata.image_source
        if str(getattr(image_source, "source_type", "")) == "embedded_h5":
            dependency_paths.extend(
                trajectory.store.raw_frame_path(int(frame)) for frame in frames
            )
        dependencies = snapshot_revisions(trajectory.store, sorted(set(dependency_paths)))

        def report_progress(event: Mapping[str, Any]) -> None:
            payload = dict(event)
            name = str(payload.pop("event", "feature_progress"))
            common = {
                "event": name,
                "job_id": batch_job.job_id,
                "h5_path": str(h5_path),
                "object_set": file_job.feature_spec.object_set,
                "feature_set": file_job.feature_spec.feature_set,
                "saved": False,
            }
            if name == "frame_completed":
                for summary_item in payload.pop("feature_summaries", []) or []:
                    emit({**common, "event": "feature_frame_summary", **payload, **summary_item})
            emit({**common, **payload})

        result = extract_feature_set(
            trajectory,
            file_job.feature_spec,
            frames=frames,
            overwrite=overwrite,
            save_outputs=False,
            run_id=batch_job.job_id,
            metadata={**batch_job.metadata, **file_job.metadata},
            progress=report_progress,
        )
        dependencies.update(
            snapshot_revisions(
                trajectory.store,
                [f"/labels/{result.source_label_set}/frame_{int(frame)}" for frame in result.frames],
            )
        )
    values_path = None
    if save_outputs:
        with Trajectory(
            h5_path,
            mode="r+",
            reporter=emit,
            operation="feature_commit",
            job_id=batch_job.job_id,
        ) as trajectory:
            validate_revisions(trajectory.store, dependencies)
            if trajectory.store.has_feature_set(result.object_set, result.feature_set) and not overwrite:
                summary.frames += len(result.frames)
                summary.skipped += 1
                emit(
                    {
                        "event": "file_skipped",
                        "job_id": batch_job.job_id,
                        "h5_path": str(h5_path),
                        "object_set": result.object_set,
                        "feature_set": result.feature_set,
                        "reason": "feature set was written by another job",
                    }
                )
                return
            values_path = trajectory.store.write_feature_set(
                result.object_set,
                result.feature_set,
                result.values,
                result.schema,
                overwrite=overwrite,
                qc=result.qc,
            )
            trajectory.store.write_feature_extraction_run(
                batch_job.job_id,
                {
                    "schema": "celltraj2.feature_extraction_run.v1",
                    "run_id": batch_job.job_id,
                    "status": "completed",
                    "completed_at": utc_now_iso(),
                    "h5_path": str(h5_path),
                    "object_set": result.object_set,
                    "source_label_set": result.source_label_set,
                    "feature_set": result.feature_set,
                    "frames": result.frames,
                    "feature_count": result.feature_count,
                    "observation_count": result.observation_count,
                    "values_path": values_path,
                    "dependencies": dependencies,
                    "feature_spec": file_job.feature_spec.to_dict(),
                    "metadata": {**batch_job.metadata, **file_job.metadata},
                },
                overwrite=True,
            )
    summary.frames += len(result.frames)
    summary.completed += len(result.frames)
    summary.features += result.feature_count
    summary.observations += result.observation_count
    emit(
        {
            "event": "file_completed",
            "job_id": batch_job.job_id,
            "h5_path": str(h5_path),
            **result.to_dict(),
            "saved": save_outputs,
            "values_path": values_path,
        }
    )


def _feature_dependency_paths(spec: FeatureSetSpec, frames: list[int]) -> list[str]:
    """Return the H5 resources that make one feature calculation valid."""

    paths = {
        "/metadata/celltraj2.json",
        "/sources/image_source.json",
        f"/object_sets/{spec.object_set}/observations",
        *{f"/object_sets/{spec.object_set}/lookup/frame_{int(frame)}" for frame in frames},
    }
    source_label = str(spec.source_label_set or spec.object_set)
    paths.update(f"/labels/{source_label}/frame_{int(frame)}" for frame in frames)

    def visit(value: Any, *, source_kind: str | None = None) -> None:
        if isinstance(value, Mapping):
            kind = str(value.get("source_kind") or source_kind or "")
            for key, item in value.items():
                if key in {"label_set", "exclude_label_set"} and item not in (None, ""):
                    paths.add(f"/labels/{item}")
                elif key in {"mask_set", "include_mask_set", "exclude_mask_set"} and item not in (None, ""):
                    paths.add(f"/masks/{item}")
                elif key == "source_name" and item not in (None, "") and kind in {"label", "mask"}:
                    paths.add(f"/{'labels' if kind == 'label' else 'masks'}/{item}")
                elif key == "boundary_set" and item not in (None, ""):
                    paths.add(f"/boundaries/{item}")
                visit(item, source_kind=kind)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, source_kind=source_kind)

    visit(spec.features)
    return sorted(paths)


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
            frames.extend(range(int(start_text.strip()), int(stop_text.strip()) + 1))
        else:
            frames.append(int(text))
    return frames
