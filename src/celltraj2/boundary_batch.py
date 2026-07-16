"""Headless multi-source boundary-library jobs for SITE."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from celltraj2.boundaries import (
    BoundaryGeometryResult,
    BoundaryLibraryResult,
    BoundaryNeighborResult,
    BoundarySourceSpec,
    as_boundary_library_view,
    build_boundary_library,
    compute_boundary_geometry,
    compute_boundary_neighbors,
)
from celltraj2.h5_access import run_with_stale_retries, snapshot_revisions, validate_revisions
from celltraj2.object_indexing import JsonlReporter
from celltraj2.paths import validate_name
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


def default_boundary_run_id() -> str:
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "_").replace(".", "_")
    return "boundaries_" + stamp


def _strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in value if str(item))


@dataclass(frozen=True)
class BoundaryGeometryJob:
    geometry_set: str = "surface_v1"
    knn: int = 40
    backend: str = "auto"
    source_names: tuple[str, ...] = ()
    source_roles: tuple[str, ...] = ()
    enabled: bool = True
    overwrite: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundaryGeometryJob":
        payload = dict(data)
        return cls(
            geometry_set=str(payload.get("geometry_set") or "surface_v1"),
            knn=int(payload.get("knn", 40)),
            backend=str(payload.get("backend") or "auto").lower(),
            source_names=_strings(payload.get("source_names")),
            source_roles=_strings(payload.get("source_roles")),
            enabled=bool(payload.get("enabled", True)),
            overwrite=bool(payload.get("overwrite", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class BoundaryNeighborJob:
    neighbor_set: str = "nearest_external_v1"
    k: int = 1
    source_names: tuple[str, ...] = ()
    source_roles: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    target_roles: tuple[str, ...] = ()
    same_frame: bool = True
    exclude_same_entity: bool = True
    max_distance: float | None = None
    bidirectional: bool = False
    reverse_neighbor_set: str | None = None
    enabled: bool = True
    overwrite: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundaryNeighborJob":
        payload = dict(data)
        max_distance = payload.get("max_distance")
        return cls(
            neighbor_set=str(payload.get("neighbor_set") or "nearest_external_v1"),
            k=int(payload.get("k", 1)),
            source_names=_strings(payload.get("source_names")),
            source_roles=_strings(payload.get("source_roles")),
            target_names=_strings(payload.get("target_names")),
            target_roles=_strings(payload.get("target_roles")),
            same_frame=bool(payload.get("same_frame", True)),
            exclude_same_entity=bool(payload.get("exclude_same_entity", True)),
            max_distance=None if max_distance in (None, "") else float(max_distance),
            bidirectional=bool(payload.get("bidirectional", False)),
            reverse_neighbor_set=(
                None if payload.get("reverse_neighbor_set") in (None, "")
                else str(payload.get("reverse_neighbor_set"))
            ),
            enabled=bool(payload.get("enabled", True)),
            overwrite=bool(payload.get("overwrite", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class BoundaryFileJob:
    h5_path: Path
    boundary_set: str
    sources: tuple[BoundarySourceSpec, ...]
    frames: tuple[int, ...] = ()
    coordinate_scale: tuple[float, float, float] | None = None
    point_spacing: float | None = None
    geometries: tuple[BoundaryGeometryJob, ...] = ()
    neighbors: tuple[BoundaryNeighborJob, ...] = ()
    enabled: bool = True
    overwrite_library: bool = False
    overwrite_derived: bool = False
    reuse_existing: bool = True
    save_outputs: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundaryFileJob":
        payload = dict(data)
        path_value = payload.get("h5_path", payload.get("path", payload.get("cell_file")))
        if path_value in (None, ""):
            raise ValueError("Boundary file job requires h5_path")
        boundary_set = str(payload.get("boundary_set") or "")
        if not boundary_set:
            raise ValueError("Boundary file job requires boundary_set")
        sources = tuple(
            item if isinstance(item, BoundarySourceSpec) else BoundarySourceSpec.from_dict(item)
            for item in payload.get("sources", ())
        )
        if not sources:
            object_set = str(payload.get("object_set") or "")
            if not object_set:
                raise ValueError("Boundary file job requires at least one source")
            sources = (BoundarySourceSpec(kind="object_set", name=object_set, object_set=object_set),)
        coordinate_scale_value = payload.get("coordinate_scale")
        coordinate_scale = None
        if coordinate_scale_value not in (None, ""):
            if not isinstance(coordinate_scale_value, Sequence) or isinstance(coordinate_scale_value, (str, bytes)):
                raise ValueError("coordinate_scale must contain Z,Y,X values")
            values = tuple(float(value) for value in coordinate_scale_value)
            if len(values) != 3:
                raise ValueError("coordinate_scale must contain Z,Y,X values")
            coordinate_scale = (values[0], values[1], values[2])
        point_spacing_value = payload.get("point_spacing")
        point_spacing = None if point_spacing_value in (None, "") else float(point_spacing_value)
        if point_spacing is not None and (not math.isfinite(point_spacing) or point_spacing <= 0):
            raise ValueError("point_spacing must be a finite positive coordinate distance")
        return cls(
            h5_path=Path(path_value),
            boundary_set=boundary_set,
            sources=sources,
            frames=tuple(int(value) for value in payload.get("frames", ()) or ()),
            coordinate_scale=coordinate_scale,
            point_spacing=point_spacing,
            geometries=tuple(
                item if isinstance(item, BoundaryGeometryJob) else BoundaryGeometryJob.from_dict(item)
                for item in payload.get("geometries", payload.get("geometry", ())) or ()
            ),
            neighbors=tuple(
                item if isinstance(item, BoundaryNeighborJob) else BoundaryNeighborJob.from_dict(item)
                for item in payload.get("neighbors", ()) or ()
            ),
            enabled=bool(payload.get("enabled", True)),
            overwrite_library=bool(payload.get("overwrite_library", False)),
            overwrite_derived=bool(payload.get("overwrite_derived", False)),
            reuse_existing=bool(payload.get("reuse_existing", True)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class BoundaryBatchJob:
    job_id: str = field(default_factory=default_boundary_run_id)
    files: tuple[BoundaryFileJob, ...] = ()
    project_root: Path | None = None
    overwrite_library: bool = False
    overwrite_derived: bool = False
    save_outputs: bool = True
    fail_fast: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundaryBatchJob":
        payload = dict(data)
        root = payload.get("project_root", payload.get("root"))
        return cls(
            job_id=str(payload.get("job_id") or default_boundary_run_id()),
            files=tuple(
                item if isinstance(item, BoundaryFileJob) else BoundaryFileJob.from_dict(item)
                for item in payload.get("files", ())
            ),
            project_root=None if root in (None, "") else Path(root),
            overwrite_library=bool(payload.get("overwrite_library", False)),
            overwrite_derived=bool(payload.get("overwrite_derived", False)),
            save_outputs=bool(payload.get("save_outputs", not bool(payload.get("dry_run", False)))),
            fail_fast=bool(payload.get("fail_fast", False)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BoundaryBatchJob":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolved_path(self, file_job: BoundaryFileJob) -> Path:
        if file_job.h5_path.is_absolute():
            return file_job.h5_path
        return (self.project_root or Path.cwd()) / file_job.h5_path

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class BatchBoundarySummary:
    job_id: str
    files: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    entities: int = 0
    points: int = 0
    geometry_sets: int = 0
    neighbor_sets: int = 0
    neighbor_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


Reporter = Callable[[Mapping[str, Any]], None]


def load_boundary_job(path: str | Path) -> BoundaryBatchJob:
    return BoundaryBatchJob.load(path)


def _validate_existing_sources(view: Any, requested: Sequence[BoundarySourceSpec]) -> None:
    existing = view.sources
    if len(existing) != len(requested):
        raise ValueError(
            f"Existing boundary library has {len(existing)} sources; requested {len(requested)}"
        )
    for stored, wanted in zip(existing, requested):
        for key, wanted_value in (
            ("kind", wanted.kind),
            ("name", wanted.name),
            ("object_set", wanted.object_set),
            ("label_set", wanted.label_set),
            ("role", wanted.role),
        ):
            if wanted_value not in (None, "") and str(stored.get(key) or "") != str(wanted_value):
                raise ValueError(
                    f"Existing boundary source {stored.get('name')!r} does not match requested {key}={wanted_value!r}"
                )
        if wanted.frames and [int(value) for value in stored.get("frames", [])] != [
            int(value) for value in wanted.frames
        ]:
            raise ValueError(
                f"Existing boundary source {stored.get('name')!r} has different selected frames"
            )


def _validate_existing_sampling(view: Any, point_spacing: float | None) -> None:
    sampling = dict(view.schema.get("sampling") or {})
    stored_value = sampling.get("point_spacing")
    stored_spacing = None if stored_value in (None, "") else float(stored_value)
    if point_spacing is None and stored_spacing is None:
        return
    if point_spacing is not None and stored_spacing is not None and math.isclose(
        float(point_spacing), stored_spacing, rel_tol=1e-9, abs_tol=1e-12
    ):
        return
    stored_label = "full native resolution" if stored_spacing is None else f"{stored_spacing:g}"
    requested_label = "full native resolution" if point_spacing is None else f"{float(point_spacing):g}"
    raise ValueError(
        f"Existing boundary library uses {stored_label}; requested {requested_label}. "
        "Choose a new boundary-set name or rebuild/overwrite the canonical library."
    )


def _emit_library_frames(emit: Reporter, common: dict[str, Any], view: Any) -> None:
    import numpy as np

    entities = view.entities
    sampling_rows = {
        (int(item.get("source_id") or 0), int(item.get("frame") or 0)): dict(item)
        for item in view.schema.get("sampling_frame_summary", []) or []
        if isinstance(item, Mapping)
    }
    for source in view.sources:
        source_id = int(source["source_id"])
        source_entities = entities[entities["source_id"] == source_id]
        for frame in sorted(int(value) for value in np.unique(source_entities["frame"])):
            rows = source_entities[source_entities["frame"] == frame]
            sampling = sampling_rows.get((source_id, frame), {})
            retained_count = int(np.sum(rows["point_count"]))
            emit(
                {
                    **common,
                    "event": "boundary_frame_summary",
                    "source_id": source_id,
                    "source_name": str(source.get("name") or ""),
                    "source_role": str(source.get("role") or ""),
                    "source_kind": str(source.get("kind") or ""),
                    "frame": frame,
                    "entity_count": int(rows.shape[0]),
                    "point_count": retained_count,
                    "native_point_count": int(sampling.get("native_point_count", retained_count)),
                    "retained_fraction": float(sampling.get("retained_fraction", 1.0)),
                    "empty_entity_count": int(np.sum(rows["point_count"] == 0)),
                }
            )


def _geometry_summary(result: Any, view: Any) -> dict[str, Any]:
    import numpy as np

    values = result.values
    return {
        "point_count": int(view.point_count),
        "finite_normal_count": int(np.sum(np.isfinite(values["normals_zyx"]).all(axis=1))),
        "finite_mean_curvature_count": int(np.sum(np.isfinite(values["mean_curvature"]))),
        "quality_flagged_count": int(np.sum(values["quality_flags"] != 0)),
        "topology_edge_count": int(result.topology_indices.shape[0]),
        "selected_backends": list(result.schema.get("selected_backends") or []),
    }


def _neighbor_frame_events(emit: Reporter, common: dict[str, Any], result: Any, view: Any) -> None:
    import numpy as np

    points = view.read_points(fields=("frame", "boundary_entity_id"))
    frames = np.asarray(points["frame"], dtype=int)
    point_entities = np.asarray(points["boundary_entity_id"], dtype=np.int64)
    counts = np.diff(result.indptr)
    selected_source_ids = result.schema.get("source_ids", "all")
    eligible = np.ones(frames.shape[0], dtype=bool)
    if selected_source_ids != "all":
        entity_sources = {
            int(entity["boundary_entity_id"]): int(entity["source_id"])
            for entity in view.entities
        }
        eligible &= np.isin(
            point_entities,
            [entity_id for entity_id, source_id in entity_sources.items() if source_id in selected_source_ids],
        )
    edge_source_rows = np.repeat(np.arange(frames.shape[0], dtype=np.int64), counts)
    for frame in sorted(int(value) for value in np.unique(frames)):
        rows = np.flatnonzero((frames == frame) & eligible)
        edge_mask = frames[edge_source_rows] == frame
        edge_mask &= eligible[edge_source_rows]
        distances = result.distance[edge_mask]
        emit(
            {
                **common,
                "event": "boundary_neighbor_frame_summary",
                "neighbor_set": result.neighbor_set,
                "frame": frame,
                "source_point_count": int(rows.size),
                "edge_count": int(np.sum(counts[rows])),
                "no_neighbor_count": int(np.sum(counts[rows] == 0)),
                "minimum_distance": None if not distances.size else float(np.min(distances)),
                "mean_distance": None if not distances.size else float(np.mean(distances)),
                "maximum_distance": None if not distances.size else float(np.max(distances)),
            }
        )


def run_batch_boundaries(
    job: BoundaryBatchJob | Mapping[str, Any],
    *,
    reporter: Reporter | None = None,
) -> BatchBoundarySummary:
    """Build boundary libraries and requested native single-frame products."""

    batch_job = job if isinstance(job, BoundaryBatchJob) else BoundaryBatchJob.from_dict(job)
    emit = reporter or (lambda _event: None)
    summary = BatchBoundarySummary(job_id=batch_job.job_id)
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
            "boundary_set": file_job.boundary_set,
            "saved": save_outputs,
        }
        emit(
            {
                **common,
                "event": "file_started",
                "source_count": len(file_job.sources),
                "geometry_count": len(file_job.geometries),
                "neighbor_count": len(file_job.neighbors),
            }
        )
        try:
            run_with_stale_retries(
                lambda: _run_boundary_file(batch_job, file_job, h5_path, summary, emit, common),
                reporter=emit,
                context=common,
            )
        except Exception as exc:
            summary.failed += 1
            emit({**common, "event": "file_failed", "error": repr(exc)})
            if batch_job.fail_fast:
                raise
    emit({"event": "job_completed", **summary.to_dict()})
    return summary


def _run_boundary_file(
    batch_job: BoundaryBatchJob,
    file_job: BoundaryFileJob,
    h5_path: Path,
    summary: BatchBoundarySummary,
    emit: Reporter,
    common: dict[str, Any],
) -> None:
    save_outputs = bool(batch_job.save_outputs and file_job.save_outputs)
    overwrite_library = bool(batch_job.overwrite_library or file_job.overwrite_library)
    overwrite_derived = bool(batch_job.overwrite_derived or file_job.overwrite_derived)
    geometry_results: list[tuple[BoundaryGeometryResult, bool]] = []
    neighbor_results: list[tuple[BoundaryNeighborResult, bool]] = []
    geometry_count = 0
    neighbor_count = 0
    neighbor_edge_count = 0
    with Trajectory(
        h5_path,
        mode="r",
        reporter=emit,
        operation="boundary_calculation",
        job_id=batch_job.job_id,
    ) as trajectory:
        exists = trajectory.store.has_boundary_set(file_job.boundary_set)
        library_result: BoundaryLibraryResult | None = None
        reused = False
        dependencies = snapshot_revisions(
            trajectory.store,
            _boundary_dependency_paths(file_job),
        )
        if exists and file_job.reuse_existing and not overwrite_library:
            view = trajectory.boundary_library(file_job.boundary_set)
            _validate_existing_sources(view, file_job.sources)
            _validate_existing_sampling(view, file_job.point_spacing)
            if file_job.coordinate_scale is not None:
                import numpy as np

                existing_scale = np.asarray(
                    view.schema.get("coordinate_scale_zyx", (1.0, 1.0, 1.0)), dtype=float
                )
                if not np.allclose(existing_scale, file_job.coordinate_scale, rtol=1e-7, atol=1e-10):
                    raise ValueError(
                        f"Existing boundary scale {existing_scale.tolist()} does not match requested "
                        f"{list(file_job.coordinate_scale)}"
                    )
            reused = True
            emit({**common, "event": "boundary_library_reused", "boundary_digest": view.schema.get("boundary_digest")})
        else:
            library_result = build_boundary_library(
                trajectory,
                file_job.boundary_set,
                sources=file_job.sources,
                frames=file_job.frames or None,
                coordinate_scale=file_job.coordinate_scale,
                point_spacing=file_job.point_spacing,
                overwrite=overwrite_library,
                save_outputs=False,
                metadata={**batch_job.metadata, **file_job.metadata, "run_id": batch_job.job_id},
            )
            view = as_boundary_library_view(trajectory, file_job.boundary_set, library_result)
            emit(
                {
                    **common,
                    "event": "boundary_library_completed",
                    "entity_count": library_result.entity_count,
                    "point_count": library_result.point_count,
                    "sampling": dict(library_result.schema.get("sampling") or {}),
                    "boundary_digest": library_result.schema.get("boundary_digest"),
                }
            )

        _emit_library_frames(emit, common, view)
        for geometry_job in file_job.geometries:
            if not geometry_job.enabled:
                continue
            geometry_name = validate_name(geometry_job.geometry_set, kind="boundary geometry set")
            exists_geometry = geometry_name in getattr(view, "geometry_sets", lambda: [])()
            overwrite = bool(overwrite_derived or geometry_job.overwrite)
            if save_outputs and exists_geometry and not overwrite:
                emit({**common, "event": "boundary_geometry_skipped", "geometry_set": geometry_name, "reason": "already exists"})
                continue
            result = compute_boundary_geometry(
                trajectory,
                file_job.boundary_set,
                geometry_set=geometry_name,
                knn=geometry_job.knn,
                backend=geometry_job.backend,  # type: ignore[arg-type]
                source_names=geometry_job.source_names or None,
                source_roles=geometry_job.source_roles or None,
                library=view,
                overwrite=overwrite,
                save_outputs=False,
                metadata={**batch_job.metadata, **file_job.metadata, **geometry_job.metadata},
            )
            geometry_results.append((result, overwrite))
            geometry_count += 1
            emit(
                {
                    **common,
                    "event": "boundary_geometry_completed",
                    "geometry_set": geometry_name,
                    **_geometry_summary(result, view),
                }
            )

        neighbor_jobs: list[tuple[BoundaryNeighborJob, bool]] = []
        for neighbor_job in file_job.neighbors:
            if not neighbor_job.enabled:
                continue
            neighbor_jobs.append((neighbor_job, False))
            if neighbor_job.bidirectional:
                neighbor_jobs.append((neighbor_job, True))
        for neighbor_job, reverse in neighbor_jobs:
            neighbor_name = (
                neighbor_job.reverse_neighbor_set or f"{neighbor_job.neighbor_set}_reverse"
                if reverse else neighbor_job.neighbor_set
            )
            neighbor_name = validate_name(neighbor_name, kind="boundary neighbor set")
            exists_neighbor = neighbor_name in getattr(view, "neighbor_sets", lambda: [])()
            overwrite = bool(overwrite_derived or neighbor_job.overwrite)
            if save_outputs and exists_neighbor and not overwrite:
                emit({**common, "event": "boundary_neighbors_skipped", "neighbor_set": neighbor_name, "reason": "already exists"})
                continue
            source_names = neighbor_job.target_names if reverse else neighbor_job.source_names
            source_roles = neighbor_job.target_roles if reverse else neighbor_job.source_roles
            target_names = neighbor_job.source_names if reverse else neighbor_job.target_names
            target_roles = neighbor_job.source_roles if reverse else neighbor_job.target_roles
            result = compute_boundary_neighbors(
                trajectory,
                file_job.boundary_set,
                neighbor_set=neighbor_name,
                k=neighbor_job.k,
                source_names=source_names or None,
                source_roles=source_roles or None,
                target_names=target_names or None,
                target_roles=target_roles or None,
                same_frame=neighbor_job.same_frame,
                exclude_same_entity=neighbor_job.exclude_same_entity,
                max_distance=neighbor_job.max_distance,
                library=view,
                overwrite=overwrite,
                save_outputs=False,
                metadata={**batch_job.metadata, **file_job.metadata, **neighbor_job.metadata, "reverse": reverse},
            )
            neighbor_results.append((result, overwrite))
            neighbor_count += 1
            neighbor_edge_count += result.edge_count
            _neighbor_frame_events(emit, common, result, view)
            emit(
                {
                    **common,
                    "event": "boundary_neighbors_completed",
                    "neighbor_set": neighbor_name,
                    "edge_count": result.edge_count,
                }
            )

        entity_count = int(view.entities.shape[0])
        point_count = int(view.point_count)
        boundary_digest = view.schema.get("boundary_digest")

    if save_outputs:
        with Trajectory(
            h5_path,
            mode="r+",
            reporter=emit,
            operation="boundary_commit",
            job_id=batch_job.job_id,
        ) as trajectory:
            validate_revisions(trajectory.store, dependencies)
            if library_result is not None:
                trajectory.store.write_boundary_library(
                    library_result.boundary_set,
                    entities=library_result.entities,
                    points=library_result.points,
                    sources=library_result.sources,
                    schema=library_result.schema,
                    overwrite=overwrite_library,
                )
            for result, overwrite in geometry_results:
                trajectory.store.write_boundary_geometry(
                    result.boundary_set,
                    result.geometry_set,
                    values=result.values,
                    topology_indptr=result.topology_indptr,
                    topology_indices=result.topology_indices,
                    schema=result.schema,
                    overwrite=overwrite,
                )
            for result, overwrite in neighbor_results:
                trajectory.store.write_boundary_neighbors(
                    result.boundary_set,
                    result.neighbor_set,
                    indptr=result.indptr,
                    indices=result.indices,
                    distance=result.distance,
                    displacement_zyx=result.displacement_zyx,
                    schema=result.schema,
                    overwrite=overwrite,
                )
            trajectory.store.write_json(
                f"/runs/boundaries/{validate_name(batch_job.job_id, kind='boundary run')}/run.json",
                {
                    "schema": "celltraj2.boundary_run.v1",
                    "job_id": batch_job.job_id,
                    "h5_path": str(h5_path),
                    "boundary_set": file_job.boundary_set,
                    "boundary_digest": boundary_digest,
                    "reused_library": reused,
                    "completed_at": utc_now_iso(),
                    "dependencies": dependencies,
                    "metadata": {**batch_job.metadata, **file_job.metadata},
                },
                overwrite=True,
            )
    summary.completed += 1
    summary.entities += entity_count
    summary.points += point_count
    summary.geometry_sets += geometry_count
    summary.neighbor_sets += neighbor_count
    summary.neighbor_edges += neighbor_edge_count
    emit(
        {
            **common,
            "event": "file_completed",
            "entity_count": entity_count,
            "point_count": point_count,
            "boundary_digest": boundary_digest,
            "reused_library": reused,
        }
    )


def _boundary_dependency_paths(file_job: BoundaryFileJob) -> list[str]:
    # A missing target has revision -1, so another job creating it while this
    # calculation runs is detected before commit too.
    paths: set[str] = {f"/boundaries/{file_job.boundary_set}"}
    for source in file_job.sources:
        if source.kind == "object_set" and source.object_set:
            paths.add(f"/object_sets/{source.object_set}")
            # Object boundaries are rasterized from their source label frames.
            # The precise label set is stored in object_set.json, so tracking
            # the labels root safely covers that indirect dependency.
            paths.add("/labels")
        if source.kind == "label_set" and source.label_set:
            paths.add(f"/labels/{source.label_set}")
        if source.kind == "mask_set" and source.label_set:
            paths.add(f"/masks/{source.label_set}")
    return sorted(paths)


__all__ = [
    "BatchBoundarySummary",
    "BoundaryBatchJob",
    "BoundaryFileJob",
    "BoundaryGeometryJob",
    "BoundaryNeighborJob",
    "JsonlReporter",
    "load_boundary_job",
    "run_batch_boundaries",
]
