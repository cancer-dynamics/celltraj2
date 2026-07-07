"""Headless batch segmentation execution for celltraj2 files."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, TextIO

from celltraj2.model_input import compose_model_input, model_input_summary, normalized_frame_axes
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


def default_job_id() -> str:
    """Return a H5-safe segmentation job id."""

    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "_").replace(".", "_")
    return f"seg_{stamp}"


@dataclass(frozen=True)
class SegmentationFileJob:
    """Batch segmentation work for one trajectory H5."""

    h5_path: Path
    label_set: str = "segmentation"
    output_kind: str = "labels"
    enabled: bool = True
    overwrite: bool = False
    save_outputs: bool = True
    frames: dict[str, Any] = field(default_factory=lambda: {"mode": "all"})
    backend: dict[str, Any] = field(default_factory=dict)
    model_input: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentationFileJob":
        payload = dict(data)
        path_value = payload.get("h5_path", payload.get("path", payload.get("cell_file")))
        if path_value in (None, ""):
            raise ValueError("Segmentation file job requires h5_path")
        frames = payload.get("frames")
        if not isinstance(frames, Mapping):
            frames = {
                "mode": payload.get("frame_mode", "all"),
                "frame_start": payload.get("frame_start"),
                "frame_stop": payload.get("frame_stop"),
                "frame_list": payload.get("frame_list"),
            }
        model_input = payload.get("model_input") if isinstance(payload.get("model_input"), Mapping) else {}
        if payload.get("channel_specs") is not None:
            model_input = {**dict(model_input), "channel_specs": list(payload.get("channel_specs") or [])}
        output_name = (
            payload.get("output_name")
            or payload.get("segmentation_name")
            or payload.get("label_set")
            or payload.get("mask_set")
            or "segmentation"
        )
        output_kind = str(
            payload.get("output_kind")
            or payload.get("output_type")
            or payload.get("save_as")
            or ("masks" if payload.get("mask_set") and not payload.get("label_set") else "labels")
        )
        return cls(
            h5_path=Path(path_value),
            label_set=str(output_name),
            output_kind=output_kind,
            enabled=bool(payload.get("enabled", True)),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            frames=dict(frames or {"mode": "all"}),
            backend=dict(payload.get("backend") or {}),
            model_input=dict(model_input),
            metadata=dict(payload.get("metadata") or {}),
        )

    @property
    def output_name(self) -> str:
        return str(self.label_set)

    @property
    def output_group(self) -> str:
        value = str(self.output_kind or "labels").strip().lower()
        if value in {"label", "labels", "label_image", "labeled_image"}:
            return "labels"
        if value in {"mask", "masks", "bool", "boolean", "binary_mask"}:
            return "masks"
        raise ValueError(f"Unsupported segmentation output kind: {self.output_kind!r}")

    @property
    def output_h5_path(self) -> str:
        group = self.output_group
        return f"/{group}/{self.output_name}"

    @property
    def channel_specs(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.model_input.get("channel_specs", [])]

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self.backend.get("parameters") or {})

    @property
    def model(self) -> str | None:
        value = self.backend.get("model")
        return None if value in (None, "") else str(value)

    @property
    def do_3d(self) -> bool:
        if "do_3d" in self.model_input:
            return bool(self.model_input["do_3d"])
        if "do_3D" in self.model_input:
            return bool(self.model_input["do_3D"])
        if "do_3d" in self.backend:
            return bool(self.backend["do_3d"])
        if "do_3D" in self.parameters:
            return bool(self.parameters["do_3D"])
        return True

    @property
    def z_index(self) -> int | None:
        value = self.model_input.get("z_index", self.frames.get("z_index"))
        if value in (None, ""):
            return None
        return int(value)

    def frame_numbers(self, frame_count: int) -> list[int]:
        """Return validated one-based frames for this file."""

        if not self.enabled:
            return []
        count = max(1, int(frame_count or 1))
        mode = str(self.frames.get("mode") or "all").lower()
        explicit = self.frames.get("frames")
        if explicit is not None:
            frames = _parse_frame_values(explicit)
        elif mode == "list":
            frames = _parse_frame_values(self.frames.get("frame_list", ""))
        elif mode == "range":
            start = int(self.frames.get("frame_start") or 1)
            stop = int(self.frames.get("frame_stop") or count)
            frames = list(range(start, stop + 1))
        else:
            frames = list(range(1, count + 1))
        invalid = [frame for frame in frames if frame < 1 or frame > count]
        if invalid:
            raise ValueError(f"Frame(s) outside 1..{count}: {invalid}")
        return sorted(set(int(frame) for frame in frames))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class SegmentationBatchJob:
    """A complete SITE-controlled batch segmentation job."""

    job_id: str = field(default_factory=default_job_id)
    files: list[SegmentationFileJob] = field(default_factory=list)
    project_root: Path | None = None
    overwrite: bool = False
    save_outputs: bool = True
    preview_output_path: Path | None = None
    preview_output_dir: Path | None = None
    fail_fast: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentationBatchJob":
        payload = dict(data)
        files = [
            item if isinstance(item, SegmentationFileJob) else SegmentationFileJob.from_dict(item)
            for item in payload.get("files", [])
        ]
        root = payload.get("project_root", payload.get("root"))
        return cls(
            job_id=str(payload.get("job_id") or default_job_id()),
            files=files,
            project_root=None if root in (None, "") else Path(root),
            overwrite=bool(payload.get("overwrite", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            preview_output_path=None
            if payload.get("preview_output_path") in (None, "")
            else Path(payload["preview_output_path"]),
            preview_output_dir=None
            if payload.get("preview_output_dir") in (None, "")
            else Path(payload["preview_output_dir"]),
            fail_fast=bool(payload.get("fail_fast", False)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SegmentationBatchJob":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolved_path(self, file_job: SegmentationFileJob) -> Path:
        path = Path(file_job.h5_path)
        if path.is_absolute():
            return path
        if self.project_root is not None:
            return self.project_root / path
        return Path.cwd() / path

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class SegmentationResult:
    """Labels plus backend metadata returned by a segmentation callable."""

    labels: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSegmentationSummary:
    """Counts accumulated during a batch segmentation run."""

    job_id: str
    files: int = 0
    frames: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


Reporter = Callable[[Mapping[str, Any]], None]
Segmenter = Callable[[Any, SegmentationFileJob, int], Any]


class JsonlReporter:
    """Write progress events as JSON lines."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def __call__(self, event: Mapping[str, Any]) -> None:
        payload = {"timestamp": utc_now_iso(), **dict(event)}
        self.stream.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")
        self.stream.flush()


def load_batch_job(path: str | Path) -> SegmentationBatchJob:
    return SegmentationBatchJob.load(path)


def run_batch_segmentation(
    job: SegmentationBatchJob | Mapping[str, Any],
    segmenter: Segmenter,
    *,
    reporter: Reporter | None = None,
) -> BatchSegmentationSummary:
    """Run a batch segmentation job with an injected segmentation callable."""

    batch_job = job if isinstance(job, SegmentationBatchJob) else SegmentationBatchJob.from_dict(job)
    emit = reporter or (lambda _event: None)
    summary = BatchSegmentationSummary(job_id=batch_job.job_id)
    emit({"event": "job_started", "job_id": batch_job.job_id, "files": len(batch_job.files)})
    for file_job in batch_job.files:
        if not file_job.enabled:
            emit({"event": "file_skipped", "reason": "disabled", "h5_path": str(file_job.h5_path)})
            continue
        h5_path = batch_job.resolved_path(file_job)
        summary.files += 1
        save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
        emit({"event": "file_started", "job_id": batch_job.job_id, "h5_path": str(h5_path), "save_outputs": save_outputs})
        try:
            _run_file_job(batch_job, file_job, h5_path, segmenter, summary, emit)
        except Exception as exc:
            summary.failed += 1
            emit({"event": "file_failed", "job_id": batch_job.job_id, "h5_path": str(h5_path), "error": repr(exc)})
            if batch_job.fail_fast:
                raise
    emit({"event": "job_completed", **summary.to_dict()})
    return summary


def _run_file_job(
    batch_job: SegmentationBatchJob,
    file_job: SegmentationFileJob,
    h5_path: Path,
    segmenter: Segmenter,
    summary: BatchSegmentationSummary,
    emit: Reporter,
) -> None:
    save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
    mode = "r+" if save_outputs else "r"
    with Trajectory(h5_path, mode=mode) as trajectory:
        frame_count = int(trajectory.metadata.frame_count or 1)
        frames = file_job.frame_numbers(frame_count)
        summary.frames += len(frames)
        run_record = {
            "schema": "celltraj2.segmentation_run.v1",
            "run_id": batch_job.job_id,
            "job_id": batch_job.job_id,
            "status": "running",
            "started_at": utc_now_iso(),
            "h5_path": str(h5_path),
            "roi_id": trajectory.metadata.roi_id,
            "dataset_id": trajectory.metadata.dataset_id,
            "output_name": file_job.output_name,
            "output_kind": file_job.output_group,
            "output_h5_path": file_job.output_h5_path,
            "frames": frames,
            "overwrite": bool(batch_job.overwrite or file_job.overwrite),
            "save_outputs": save_outputs,
            "backend": file_job.backend,
            "model_input": file_job.model_input,
            "metadata": file_job.metadata,
        }
        if save_outputs:
            trajectory.write_segmentation_run(batch_job.job_id, run_record, overwrite=True)
        failed_before = int(summary.failed)
        for frame in frames:
            _run_frame(batch_job, file_job, trajectory, frame, segmenter, summary, emit)
        run_record["status"] = "completed_with_errors" if summary.failed > failed_before else "completed"
        run_record["completed_at"] = utc_now_iso()
        if save_outputs:
            trajectory.write_segmentation_run(batch_job.job_id, run_record, overwrite=True)


def _run_frame(
    batch_job: SegmentationBatchJob,
    file_job: SegmentationFileJob,
    trajectory: Trajectory,
    frame: int,
    segmenter: Segmenter,
    summary: BatchSegmentationSummary,
    emit: Reporter,
) -> None:
    overwrite = bool(batch_job.overwrite or file_job.overwrite)
    save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
    if save_outputs and _has_output_frame(trajectory, file_job, frame) and not overwrite:
        summary.skipped += 1
        record = {
            "frame": int(frame),
            "status": "skipped",
            "reason": f"{file_job.output_group} frame exists",
            "output_name": file_job.output_name,
            "output_kind": file_job.output_group,
            "output_h5_path": file_job.output_h5_path,
            "saved": False,
        }
        trajectory.write_segmentation_frame_result(batch_job.job_id, frame, record, overwrite=True)
        emit({"event": "frame_skipped", "job_id": batch_job.job_id, "h5_path": str(trajectory.path), **record})
        return

    try:
        frame_data = trajectory.get_image_data(frame=frame)
        frame_axes = normalized_frame_axes(trajectory.frame_axes(getattr(frame_data, "ndim", 0)), getattr(frame_data, "ndim", 0))
        effective_do_3d = _effective_do_3d_for_frame(file_job, frame_axes)
        effective_file_job = _file_job_with_effective_do_3d(file_job, effective_do_3d)
        model_input = compose_model_input(
            frame_data,
            channel_specs=effective_file_job.channel_specs,
            axes=frame_axes,
            do_3d=effective_file_job.do_3d,
            z_index=effective_file_job.z_index,
            channel_index_map=trajectory.channel_index_map(),
        )
        channel_axis = 1 if effective_file_job.do_3d and getattr(model_input, "ndim", 0) == 4 else 0 if (not effective_file_job.do_3d and getattr(model_input, "ndim", 0) == 3) else None
        result = _coerce_segmentation_result(segmenter(model_input, effective_file_job, frame))
        output_path = None
        if save_outputs:
            output_path = _write_output_frame(trajectory, file_job, frame, result.labels, overwrite=overwrite, batch_job=batch_job)
        preview_output_path = _frame_preview_output_path(batch_job, file_job, trajectory, frame)
        if preview_output_path is not None:
            _write_preview_npz(
                preview_output_path,
                model_input=model_input,
                labels=result.labels,
                file_job=effective_file_job,
                frame=frame,
                metadata={
                    **dict(result.metadata or {}),
                    "frame_axes": list(frame_axes),
                    "requested_do_3D": bool(file_job.do_3d),
                    "effective_do_3D": bool(effective_file_job.do_3d),
                },
            )
        summary.completed += 1
        record = {
            "frame": int(frame),
            "status": "completed",
            "output_name": file_job.output_name,
            "output_kind": file_job.output_group,
            "output_h5_path": file_job.output_h5_path,
            "output_path": output_path,
            "saved": save_outputs,
            "preview_output_path": None if preview_output_path is None else str(preview_output_path),
            "input_summary": model_input_summary(model_input, channel_axis=channel_axis),
            "label_summary": _label_summary(result.labels),
            "backend_metadata": result.metadata,
            "frame_axes": list(frame_axes),
            "requested_do_3D": bool(file_job.do_3d),
            "effective_do_3D": bool(effective_file_job.do_3d),
        }
        if save_outputs:
            trajectory.write_segmentation_frame_result(batch_job.job_id, frame, record, overwrite=True)
        emit({"event": "frame_completed", "job_id": batch_job.job_id, "h5_path": str(trajectory.path), **record})
    except Exception as exc:
        summary.failed += 1
        record = {
            "frame": int(frame),
            "status": "failed",
            "output_name": file_job.output_name,
            "output_kind": file_job.output_group,
            "output_h5_path": file_job.output_h5_path,
            "error": repr(exc),
            "saved": False,
        }
        if save_outputs:
            trajectory.write_segmentation_frame_result(batch_job.job_id, frame, record, overwrite=True)
        emit({"event": "frame_failed", "job_id": batch_job.job_id, "h5_path": str(trajectory.path), **record})
        if batch_job.fail_fast:
            raise


def _coerce_segmentation_result(value: Any) -> SegmentationResult:
    if isinstance(value, SegmentationResult):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        labels, metadata = value
        return SegmentationResult(labels=labels, metadata=dict(metadata or {}))
    return SegmentationResult(labels=value, metadata={})


def _effective_do_3d_for_frame(file_job: SegmentationFileJob, frame_axes: Sequence[str]) -> bool:
    """Return the executable dimensionality for a frame.

    A true 2D source has no Z axis after ``get_image_data``. Treating it as a
    3D Cellpose volume adds a synthetic singleton Z plane and can make Cellpose
    return channel-shaped labels, so 3D requests are downgraded for such frames.
    """

    axes = {str(axis).upper() for axis in frame_axes}
    return bool(file_job.do_3d and "Z" in axes)


def _file_job_with_effective_do_3d(file_job: SegmentationFileJob, do_3d: bool) -> SegmentationFileJob:
    backend = dict(file_job.backend or {})
    parameters = dict(backend.get("parameters") or {})
    parameters["do_3D"] = bool(do_3d)
    if not do_3d:
        parameters.pop("anisotropy", None)
        parameters.pop("flow3D_smooth", None)
        parameters.pop("stitch_threshold", None)
    backend["parameters"] = parameters
    model_input = {**dict(file_job.model_input or {}), "do_3D": bool(do_3d)}
    return replace(file_job, backend=backend, model_input=model_input)


def _has_output_frame(trajectory: Trajectory, file_job: SegmentationFileJob, frame: int) -> bool:
    if file_job.output_group == "masks":
        return trajectory.store.has_mask_frame(file_job.output_name, frame)
    return trajectory.store.has_label_frame(file_job.output_name, frame)


def _write_output_frame(
    trajectory: Trajectory,
    file_job: SegmentationFileJob,
    frame: int,
    labels: Any,
    *,
    overwrite: bool,
    batch_job: SegmentationBatchJob,
) -> str:
    metadata = {"run_id": batch_job.job_id, "backend": file_job.backend, "output_kind": file_job.output_group}
    if file_job.output_group == "masks":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Writing segmentation masks requires numpy") from exc
        mask = np.asarray(labels) > 0
        return trajectory.store.write_mask_frame(
            file_job.output_name,
            frame,
            mask,
            overwrite=overwrite,
            metadata=metadata,
        )
    return trajectory.store.write_label_frame(
        file_job.output_name,
        frame,
        labels,
        overwrite=overwrite,
        metadata=metadata,
    )


def _write_preview_npz(
    path: str | Path,
    *,
    model_input: Any,
    labels: Any,
    file_job: SegmentationFileJob,
    frame: int,
    metadata: Mapping[str, Any],
) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Writing segmentation preview output requires numpy") from exc
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    label_data = np.asarray(labels)
    np.savez_compressed(
        out,
        model_input=np.asarray(model_input),
        labels=label_data,
        mask=label_data > 0,
        frame=np.asarray([int(frame)], dtype=np.int64),
        do_3D=np.asarray([bool(file_job.do_3d)]),
        frame_axes=np.asarray([str(axis) for axis in dict(metadata).get("frame_axes", [])]),
        output_name=np.asarray(file_job.output_name),
        output_kind=np.asarray(file_job.output_group),
        output_h5_path=np.asarray(file_job.output_h5_path),
        backend_metadata=np.asarray(json.dumps(_json_safe(dict(metadata)), sort_keys=True)),
    )


def _frame_preview_output_path(
    batch_job: SegmentationBatchJob,
    file_job: SegmentationFileJob,
    trajectory: Trajectory,
    frame: int,
) -> Path | None:
    if batch_job.preview_output_dir is not None:
        stem = _safe_filename(Path(trajectory.path).name)
        output = _safe_filename(file_job.output_name)
        return Path(batch_job.preview_output_dir) / f"{stem}__{output}__frame_{int(frame)}.npz"
    if batch_job.preview_output_path is not None:
        return Path(batch_job.preview_output_path)
    return None


def _safe_filename(value: str) -> str:
    text = str(value).strip()
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return cleaned.strip("._") or "item"


def _label_summary(labels: Any) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {"shape": list(getattr(labels, "shape", ())), "dtype": str(getattr(labels, "dtype", ""))}
    arr = np.asarray(labels)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "max_label": int(np.max(arr)) if arr.size else 0,
        "unique_count": int(len(np.unique(arr))) if arr.size else 0,
    }


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
