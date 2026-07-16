"""Object observation indexing for celltraj2 label frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

from celltraj2.paths import validate_name
from celltraj2.schema import utc_now_iso


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "celltraj2 object indexing requires numpy. Install with "
            "`python -m pip install -e .[analysis]`."
        ) from exc
    return np


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


def observation_dtype() -> Any:
    """Return the canonical structured dtype for object observations."""

    np = _require_numpy()
    return np.dtype(
        [
            ("observation_id", "<i8"),
            ("frame", "<i4"),
            ("parent_time_index", "<i8"),
            ("label_id", "<i8"),
            ("z_min", "<i4"),
            ("z_max", "<i4"),
            ("y_min", "<i4"),
            ("y_max", "<i4"),
            ("x_min", "<i4"),
            ("x_max", "<i4"),
            ("centroid_z", "<f8"),
            ("centroid_y", "<f8"),
            ("centroid_x", "<f8"),
            ("voxel_count", "<i8"),
            ("quality_flags", "<u4"),
        ]
    )


OBSERVATION_COLUMNS = [
    {
        "name": "observation_id",
        "dtype": "int64",
        "description": "Stable 1-based object-observation id within this object set.",
    },
    {"name": "frame", "dtype": "int32", "description": "One-based local ROI frame number."},
    {
        "name": "parent_time_index",
        "dtype": "int64",
        "description": "Zero-based parent acquisition time index.",
    },
    {"name": "label_id", "dtype": "int64", "description": "Positive label value within the source label frame."},
    {"name": "z_min", "dtype": "int32", "description": "Inclusive local ROI Z start."},
    {"name": "z_max", "dtype": "int32", "description": "Exclusive local ROI Z stop."},
    {"name": "y_min", "dtype": "int32", "description": "Inclusive local ROI Y start."},
    {"name": "y_max", "dtype": "int32", "description": "Exclusive local ROI Y stop."},
    {"name": "x_min", "dtype": "int32", "description": "Inclusive local ROI X start."},
    {"name": "x_max", "dtype": "int32", "description": "Exclusive local ROI X stop."},
    {"name": "centroid_z", "dtype": "float64", "description": "Mean local ROI Z coordinate."},
    {"name": "centroid_y", "dtype": "float64", "description": "Mean local ROI Y coordinate."},
    {"name": "centroid_x", "dtype": "float64", "description": "Mean local ROI X coordinate."},
    {"name": "voxel_count", "dtype": "int64", "description": "Number of positive-label voxels in the observation."},
    {"name": "quality_flags", "dtype": "uint32", "description": "Bit-packed quality flags; zero means no flags set."},
]


@dataclass(frozen=True)
class ObjectIndexResult:
    """Result from indexing one object set."""

    object_set: str
    source_label_set: str
    observations: Any
    lookups: dict[int, Any]
    schema: dict[str, Any]
    frames: list[int]
    frame_counts: dict[int, int] = field(default_factory=dict)
    lookup_paths: dict[int, str] = field(default_factory=dict)
    observations_path: str | None = None
    run_id: str | None = None
    saved: bool = True

    @property
    def observation_count(self) -> int:
        return int(getattr(self.observations, "shape", (0,))[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_set": self.object_set,
            "source_label_set": self.source_label_set,
            "frames": list(self.frames),
            "frame_counts": {str(key): int(value) for key, value in self.frame_counts.items()},
            "lookup_paths": {str(key): value for key, value in self.lookup_paths.items()},
            "observations_path": self.observations_path,
            "observation_count": self.observation_count,
            "run_id": self.run_id,
            "saved": bool(self.saved),
        }


def default_object_index_run_id() -> str:
    """Return a H5-safe object-indexing run id."""

    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "_").replace(".", "_")
    return f"obj_index_{stamp}"


def observation_schema(
    *,
    object_set: str,
    source_label_set: str,
    frames: Sequence[int],
    observation_count: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the JSON schema metadata for an observation table."""

    return {
        "schema": "celltraj2.object_observations.v1",
        "object_set": object_set,
        "source_label_set": source_label_set,
        "observation_id_base": 1,
        "row_alignment": "row_index_zero_based_maps_to_observation_id_minus_1",
        "sort_order": ["frame", "label_id"],
        "frames": [int(frame) for frame in frames],
        "observation_count": int(observation_count),
        "columns": list(OBSERVATION_COLUMNS),
        "metadata": _json_safe(dict(metadata or {})),
    }


def index_object_set(
    trajectory: Any,
    object_set: str,
    *,
    source_label_set: str | None = None,
    frames: Sequence[int] | None = None,
    overwrite: bool = False,
    save_outputs: bool = True,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> ObjectIndexResult:
    """Index one source label set into stable object observations."""

    np = _require_numpy()
    object_name = validate_name(object_set, kind="object set")
    label_name = validate_name(source_label_set or object_name, kind="label set")
    if save_outputs and trajectory.store.has_observations(object_name) and not overwrite:
        raise FileExistsError(f"/object_sets/{object_name}/observations")

    selected_frames = [int(frame) for frame in (frames if frames is not None else trajectory.label_frames(label_name))]
    selected_frames = sorted(dict.fromkeys(selected_frames))
    parent_time = {int(item["frame"]): int(item["parent_time_index"]) for item in trajectory.metadata.frame_map()}
    records: list[tuple[Any, ...]] = []
    lookups: dict[int, Any] = {}
    frame_counts: dict[int, int] = {}
    next_observation_id = 1

    for frame in selected_frames:
        if not trajectory.store.has_label_frame(label_name, frame):
            raise FileNotFoundError(f"/labels/{label_name}/frame_{frame}")
        labels = _normalize_label_frame(trajectory.read_label_frame(label_name, frame), np=np)
        frame_records, lookup, next_observation_id = _index_one_frame(
            labels,
            frame=frame,
            parent_time_index=parent_time.get(frame, frame - 1),
            first_observation_id=next_observation_id,
            np=np,
        )
        records.extend(frame_records)
        lookups[frame] = lookup
        frame_counts[frame] = len(frame_records)
        if progress is not None:
            progress(
                {
                    "frame": int(frame),
                    "object_set": object_name,
                    "source_label_set": label_name,
                    "observation_count": int(len(frame_records)),
                }
            )

    observations = np.asarray(records, dtype=observation_dtype())
    schema = observation_schema(
        object_set=object_name,
        source_label_set=label_name,
        frames=selected_frames,
        observation_count=int(observations.shape[0]),
        metadata=metadata,
    )
    observations_path = None
    lookup_paths: dict[int, str] = {}
    run_name = validate_name(run_id or default_object_index_run_id(), kind="object-indexing run")
    if save_outputs:
        run_record = {
            "schema": "celltraj2.object_indexing_run.v1",
            "run_id": run_name,
            "job_id": run_name,
            "status": "running",
            "started_at": utc_now_iso(),
            "h5_path": str(trajectory.path),
            "roi_id": trajectory.metadata.roi_id,
            "dataset_id": trajectory.metadata.dataset_id,
            "object_set": object_name,
            "source_label_set": label_name,
            "frames": selected_frames,
            "overwrite": bool(overwrite),
            "save_outputs": True,
            "metadata": _json_safe(dict(metadata or {})),
        }
        trajectory.store.write_object_indexing_run(run_name, run_record, overwrite=True)
        observations_path = trajectory.store.write_observations(
            object_name,
            observations,
            schema,
            source_label_set=label_name,
            overwrite=overwrite,
            metadata=metadata,
        )
        if overwrite:
            trajectory.store.clear_observation_lookup_frames(object_name)
        for frame in selected_frames:
            lookup_paths[frame] = trajectory.store.write_observation_lookup_frame(
                object_name,
                frame,
                lookups[frame],
                overwrite=overwrite,
            )
            trajectory.store.write_object_indexing_frame_result(
                run_name,
                frame,
                {
                    "frame": int(frame),
                    "status": "completed",
                    "object_set": object_name,
                    "source_label_set": label_name,
                    "observation_count": int(frame_counts[frame]),
                    "lookup_path": lookup_paths[frame],
                },
                overwrite=True,
            )
        run_record["status"] = "completed"
        run_record["completed_at"] = utc_now_iso()
        run_record["observations_path"] = observations_path
        run_record["observation_count"] = int(observations.shape[0])
        run_record["frame_counts"] = {str(key): int(value) for key, value in frame_counts.items()}
        trajectory.store.write_object_indexing_run(run_name, run_record, overwrite=True)

    return ObjectIndexResult(
        object_set=object_name,
        source_label_set=label_name,
        observations=observations,
        lookups=lookups,
        schema=schema,
        frames=selected_frames,
        frame_counts=frame_counts,
        lookup_paths=lookup_paths,
        observations_path=observations_path,
        run_id=run_name,
        saved=bool(save_outputs),
    )


def _normalize_label_frame(labels: Any, *, np: Any) -> Any:
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Object indexing expects 2D YX or 3D ZYX labels; got shape {arr.shape}")
    return arr


def _index_one_frame(
    labels: Any,
    *,
    frame: int,
    parent_time_index: int,
    first_observation_id: int,
    np: Any,
) -> tuple[list[tuple[Any, ...]], Any, int]:
    positive = labels[labels > 0]
    if positive.size:
        label_ids = [int(value) for value in np.unique(positive)]
        max_label = int(max(label_ids))
    else:
        label_ids = []
        max_label = 0
    lookup = np.zeros(max_label + 1, dtype=np.uint64)
    records: list[tuple[Any, ...]] = []
    observation_id = int(first_observation_id)
    for label_id in label_ids:
        coords = np.argwhere(labels == label_id)
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        centroid = coords.mean(axis=0)
        lookup[label_id] = observation_id
        records.append(
            (
                int(observation_id),
                int(frame),
                int(parent_time_index),
                int(label_id),
                int(mins[0]),
                int(maxs[0]),
                int(mins[1]),
                int(maxs[1]),
                int(mins[2]),
                int(maxs[2]),
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
                int(coords.shape[0]),
                0,
            )
        )
        observation_id += 1
    return records, lookup, observation_id
