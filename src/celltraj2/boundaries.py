"""Native boundary libraries, surface geometry, neighborhoods, and transport maps."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

from celltraj2.paths import validate_name
from celltraj2.registration import registration_calibration
from celltraj2.schema import utc_now_iso


BoundarySourceKind = Literal["object_set", "label_set", "mask_set"]


ENTITY_QUALITY_NO_POINTS = 1 << 0
GEOMETRY_QUALITY_TOO_FEW_POINTS = 1 << 0
GEOMETRY_QUALITY_OPERATOR_FAILED = 1 << 1
GEOMETRY_QUALITY_DISCONNECTED_HINT = 1 << 2
GEOMETRY_QUALITY_NOT_SELECTED = 1 << 3


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "celltraj2 boundary analysis requires numpy. Install with "
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


def boundary_entity_dtype() -> Any:
    """Return the row dtype for boundary entities."""

    np = _require_numpy()
    return np.dtype(
        [
            ("boundary_entity_id", "<i8"),
            ("source_id", "<i4"),
            ("frame", "<i4"),
            ("observation_id", "<i8"),
            ("source_label_id", "<i8"),
            ("point_start", "<i8"),
            ("point_count", "<i8"),
            ("quality_flags", "<u4"),
        ]
    )


def boundary_motion_link_dtype() -> Any:
    """Return the row dtype for entity-to-entity boundary motion links."""

    np = _require_numpy()
    return np.dtype(
        [
            ("motion_link_id", "<i8"),
            ("track_link_id", "<i8"),
            ("source_entity_id", "<i8"),
            ("target_entity_id", "<i8"),
            ("source_observation_id", "<i8"),
            ("target_observation_id", "<i8"),
            ("source_frame", "<i4"),
            ("target_frame", "<i4"),
            ("transport_start", "<i8"),
            ("transport_count", "<i8"),
            ("ot_cost", "<f8"),
            ("transported_mass", "<f8"),
            ("quality_flags", "<u4"),
        ]
    )


def boundary_digest(entities: Any, points: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable digest for boundary-derived result dependencies."""

    np = _require_numpy()
    digest = hashlib.sha256()
    entity_values = np.asarray(entities)
    digest.update(str(entity_values.dtype).encode("utf-8"))
    digest.update(str(tuple(entity_values.shape)).encode("utf-8"))
    digest.update(entity_values.tobytes())
    for key in sorted(points):
        values = np.asarray(points[key])
        digest.update(str(key).encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(tuple(values.shape)).encode("utf-8"))
        digest.update(values.tobytes())
    digest.update(
        json.dumps(_json_safe(list(sources)), sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class BoundarySourceSpec:
    """One label-like source contributing entities to a boundary library."""

    kind: BoundarySourceKind
    name: str
    object_set: str | None = None
    label_set: str | None = None
    frames: tuple[int, ...] = ()
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundarySourceSpec":
        payload = dict(data)
        kind = str(payload.get("kind") or "object_set")
        if kind not in {"object_set", "label_set", "mask_set"}:
            raise ValueError(f"Unsupported boundary source kind {kind!r}")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            name=str(payload.get("name") or payload.get("object_set") or payload.get("label_set") or ""),
            object_set=None if payload.get("object_set") in (None, "") else str(payload["object_set"]),
            label_set=None if payload.get("label_set") in (None, "") else str(payload["label_set"]),
            frames=tuple(int(value) for value in payload.get("frames", ()) or ()),
            role=None if payload.get("role") in (None, "") else str(payload["role"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class BoundaryLibraryResult:
    """Result of assembling one native boundary library."""

    boundary_set: str
    entities: Any
    points: dict[str, Any]
    schema: dict[str, Any]
    sources: list[dict[str, Any]]
    saved: bool
    boundary_path: str | None = None

    @property
    def entity_count(self) -> int:
        return int(self.entities.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.points["point_id"].shape[0])


@dataclass(frozen=True)
class BoundaryGeometryResult:
    """Row-aligned surface geometry calculated for a boundary library."""

    boundary_set: str
    geometry_set: str
    values: dict[str, Any]
    topology_indptr: Any
    topology_indices: Any
    schema: dict[str, Any]
    saved: bool
    geometry_path: str | None = None


@dataclass(frozen=True)
class BoundaryNeighborResult:
    """CSR nearest-neighbor edges over boundary point rows."""

    boundary_set: str
    neighbor_set: str
    indptr: Any
    indices: Any
    distance: Any
    displacement_zyx: Any
    schema: dict[str, Any]
    saved: bool
    neighbor_path: str | None = None

    @property
    def edge_count(self) -> int:
        return int(self.indices.shape[0])


@dataclass(frozen=True)
class BoundaryTransportPlan:
    """Sparse representation of one optimal-transport plan."""

    source_rows: Any
    target_rows: Any
    mass: Any
    edge_cost: Any
    total_cost: float
    method: str


class BoundaryLibraryView:
    """Lazy H5-backed view that slices points by contiguous entity spans."""

    def __init__(self, store: Any, boundary_set: str) -> None:
        self.store = store
        self.name = validate_name(boundary_set, kind="boundary set")
        self.path = f"boundaries/{self.name}"
        if f"{self.path}/entities" not in self.store.h5:
            raise KeyError(f"/{self.path}")
        self._entities = None
        self._schema = None

    @property
    def schema(self) -> dict[str, Any]:
        if self._schema is None:
            self._schema = self.store.read_json(f"/{self.path}/schema.json")
        return dict(self._schema)

    @property
    def sources(self) -> list[dict[str, Any]]:
        return list(self.store.read_json(f"/{self.path}/sources.json"))

    @property
    def entities(self) -> Any:
        if self._entities is None:
            self._entities = self.store.h5[f"{self.path}/entities"][()]
        return self._entities

    @property
    def point_count(self) -> int:
        return int(self.store.h5[f"{self.path}/points/point_id"].shape[0])

    def entity(self, boundary_entity_id: int) -> Any:
        value = int(boundary_entity_id)
        if value < 1 or value > int(self.entities.shape[0]):
            raise IndexError(f"boundary_entity_id {value} is outside 1..{int(self.entities.shape[0])}")
        row = self.entities[value - 1]
        if int(row["boundary_entity_id"]) != value:
            matches = _require_numpy().flatnonzero(self.entities["boundary_entity_id"] == value)
            if not matches.size:
                raise KeyError(value)
            row = self.entities[int(matches[0])]
        return row

    def entity_id_for_observation(self, observation_id: int, *, source_id: int | None = None) -> int:
        np = _require_numpy()
        mask = self.entities["observation_id"] == int(observation_id)
        if source_id is not None:
            mask &= self.entities["source_id"] == int(source_id)
        matches = np.flatnonzero(mask)
        if matches.size != 1:
            raise KeyError(
                f"Expected one boundary entity for observation_id={int(observation_id)}, found {int(matches.size)}"
            )
        return int(self.entities[int(matches[0])]["boundary_entity_id"])

    def entities_for_frame(self, frame: int, *, source_id: int | None = None) -> Any:
        mask = self.entities["frame"] == int(frame)
        if source_id is not None:
            mask &= self.entities["source_id"] == int(source_id)
        return self.entities[mask]

    def point_slice(self, boundary_entity_id: int) -> slice:
        row = self.entity(boundary_entity_id)
        start = int(row["point_start"])
        return slice(start, start + int(row["point_count"]))

    def point_spans(self, boundary_entity_ids: Sequence[int]) -> list[slice]:
        """Return ordered contiguous H5 slices for an entity selection."""

        return [self.point_slice(value) for value in sorted(dict.fromkeys(int(item) for item in boundary_entity_ids))]

    def read_points(
        self,
        boundary_entity_id: int | None = None,
        *,
        rows: slice | Any | None = None,
        fields: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if boundary_entity_id is not None and rows is not None:
            raise ValueError("Pass boundary_entity_id or rows, not both")
        selection = (
            self.point_slice(boundary_entity_id)
            if boundary_entity_id is not None
            else (rows if rows is not None else slice(None))
        )
        group = self.store.h5[f"{self.path}/points"]
        names = list(fields) if fields is not None else sorted(str(key) for key in group.keys())
        return {name: group[name][selection] for name in names}

    def native_positions(self, boundary_entity_id: int) -> Any:
        return self.read_points(boundary_entity_id, fields=("native_position_zyx",))["native_position_zyx"]

    def registered_positions(self, boundary_entity_id: int, registration_set: str | None = None) -> Any:
        np = _require_numpy()
        entity = self.entity(boundary_entity_id)
        points = self.native_positions(boundary_entity_id)
        selected = registration_set or self.store.active_registration_name()
        if selected is None:
            return points.copy()
        registration = self.store.read_registration_set(selected)
        frames = np.full(points.shape[0], int(entity["frame"]), dtype=np.int64)
        return registration.apply_zyx(points, frames)

    def geometry(self, geometry_set: str, boundary_entity_id: int | None = None) -> dict[str, Any]:
        name = validate_name(geometry_set, kind="boundary geometry set")
        selection = self.point_slice(boundary_entity_id) if boundary_entity_id is not None else slice(None)
        group = self.store.h5[f"{self.path}/geometry/{name}"]
        return {
            str(key): group[key][selection]
            for key in group.keys()
            if str(key) != "schema.json" and hasattr(group[key], "shape")
        }

    def geometry_topology(self, geometry_set: str, boundary_entity_id: int) -> dict[str, Any]:
        name = validate_name(geometry_set, kind="boundary geometry set")
        group = self.store.h5[f"{self.path}/geometry/{name}/topology"]
        point_span = self.point_slice(boundary_entity_id)
        indptr = group["indptr"]
        edge_start = int(indptr[point_span.start])
        edge_stop = int(indptr[point_span.stop])
        return {
            "source_point_rows": _require_numpy().repeat(
                _require_numpy().arange(point_span.start, point_span.stop, dtype=_require_numpy().int64),
                _require_numpy().diff(indptr[point_span.start : point_span.stop + 1]),
            ),
            "target_point_rows": group["indices"][edge_start:edge_stop],
        }

    def neighbor_edges(self, neighbor_set: str, boundary_entity_id: int) -> dict[str, Any]:
        name = validate_name(neighbor_set, kind="boundary neighbor set")
        group = self.store.h5[f"{self.path}/neighbors/{name}"]
        point_span = self.point_slice(boundary_entity_id)
        indptr_all = group["indptr"]
        edge_start = int(indptr_all[point_span.start])
        edge_stop = int(indptr_all[point_span.stop])
        return {
            "source_point_rows": _require_numpy().repeat(
                _require_numpy().arange(point_span.start, point_span.stop, dtype=_require_numpy().int64),
                _require_numpy().diff(indptr_all[point_span.start : point_span.stop + 1]),
            ),
            "target_point_rows": group["indices"][edge_start:edge_stop],
            "distance": group["distance"][edge_start:edge_stop],
            "displacement_zyx": group["displacement_zyx"][edge_start:edge_stop],
        }

    def geometry_sets(self) -> list[str]:
        path = f"{self.path}/geometry"
        return [] if path not in self.store.h5 else sorted(str(key) for key in self.store.h5[path].keys())

    def neighbor_sets(self) -> list[str]:
        path = f"{self.path}/neighbors"
        return [] if path not in self.store.h5 else sorted(str(key) for key in self.store.h5[path].keys())

    def motion_sets(self) -> list[str]:
        path = f"{self.path}/motion"
        return [] if path not in self.store.h5 else sorted(str(key) for key in self.store.h5[path].keys())

    def entity_attribute_sets(self) -> list[str]:
        return self.store.list_boundary_entity_attribute_sets(self.name)

    def entity_attributes(self, attribute_set: str) -> Any:
        return self.store.read_boundary_entity_attributes(self.name, attribute_set)


class InMemoryBoundaryLibraryView:
    """Boundary-library view over an unsaved :class:`BoundaryLibraryResult`.

    This mirrors the point/entity access used by geometry, neighbor, and
    transient-tracking calculations so SITE ``Test`` jobs can execute the full
    pipeline without mutating the H5 file.
    """

    def __init__(self, result: BoundaryLibraryResult) -> None:
        self.name = result.boundary_set
        self._entities = result.entities
        self._points = dict(result.points)
        self._schema = dict(result.schema)
        self._sources = [dict(value) for value in result.sources]

    @property
    def schema(self) -> dict[str, Any]:
        return dict(self._schema)

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._sources]

    @property
    def entities(self) -> Any:
        return self._entities

    @property
    def point_count(self) -> int:
        return int(self._points["point_id"].shape[0])

    def entity(self, boundary_entity_id: int) -> Any:
        value = int(boundary_entity_id)
        if value < 1 or value > int(self.entities.shape[0]):
            raise IndexError(f"boundary_entity_id {value} is outside 1..{int(self.entities.shape[0])}")
        row = self.entities[value - 1]
        if int(row["boundary_entity_id"]) != value:
            matches = _require_numpy().flatnonzero(self.entities["boundary_entity_id"] == value)
            if not matches.size:
                raise KeyError(value)
            row = self.entities[int(matches[0])]
        return row

    def point_slice(self, boundary_entity_id: int) -> slice:
        row = self.entity(boundary_entity_id)
        start = int(row["point_start"])
        return slice(start, start + int(row["point_count"]))

    def read_points(
        self,
        boundary_entity_id: int | None = None,
        *,
        rows: slice | Any | None = None,
        fields: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if boundary_entity_id is not None and rows is not None:
            raise ValueError("Pass boundary_entity_id or rows, not both")
        selection = (
            self.point_slice(boundary_entity_id)
            if boundary_entity_id is not None
            else (rows if rows is not None else slice(None))
        )
        names = list(fields) if fields is not None else sorted(self._points)
        return {name: self._points[name][selection] for name in names}


BoundaryDataView = BoundaryLibraryView | InMemoryBoundaryLibraryView


def as_boundary_library_view(
    trajectory: Any,
    boundary_set: str,
    library: BoundaryLibraryResult | BoundaryDataView | None = None,
) -> BoundaryDataView:
    """Resolve an H5-backed or unsaved boundary library through one API."""

    name = validate_name(boundary_set, kind="boundary set")
    if library is None:
        return BoundaryLibraryView(trajectory.store, name)
    if isinstance(library, BoundaryLibraryResult):
        if library.boundary_set != name:
            raise ValueError(
                f"In-memory boundary set {library.boundary_set!r} does not match {name!r}"
            )
        return InMemoryBoundaryLibraryView(library)
    if library.name != name:
        raise ValueError(f"Boundary view {library.name!r} does not match {name!r}")
    return library


def resolve_boundary_source_ids(
    library: BoundaryDataView,
    *,
    source_ids: Sequence[int] | None = None,
    source_names: Sequence[str] | None = None,
    source_roles: Sequence[str] | None = None,
    require_one: bool = False,
) -> set[int] | None:
    """Resolve optional source selectors to stable integer source ids.

    Multiple selector categories are intersected. ``None`` means no source
    filtering; an explicitly supplied selector that matches nothing is an
    error rather than a silently empty calculation.
    """

    selectors_used = any(value is not None for value in (source_ids, source_names, source_roles))
    if not selectors_used:
        if require_one:
            if len(library.sources) != 1:
                raise ValueError("Select exactly one boundary source")
            return {int(library.sources[0]["source_id"])}
        return None
    selected = {int(value["source_id"]) for value in library.sources}
    if source_ids is not None:
        selected &= {int(value) for value in source_ids}
    if source_names is not None:
        names = {str(value) for value in source_names}
        selected &= {
            int(value["source_id"])
            for value in library.sources
            if str(value.get("name") or "") in names
        }
    if source_roles is not None:
        roles = {str(value) for value in source_roles}
        selected &= {
            int(value["source_id"])
            for value in library.sources
            if str(value.get("role") or "") in roles
        }
    if not selected:
        raise ValueError("Boundary source selectors did not match any library source")
    if require_one and len(selected) != 1:
        raise ValueError(f"Expected exactly one boundary source, matched {len(selected)}")
    return selected


def _normalize_zyx(array: Any, *, np: Any) -> tuple[Any, int]:
    values = np.asarray(array)
    if values.ndim == 2:
        return values[np.newaxis, :, :], 2
    if values.ndim == 3:
        return values, 3
    raise ValueError(f"Boundary extraction expects 2D YX or 3D ZYX arrays; got shape {values.shape}")


def _boundary_mask_and_orientation(labels: Any, *, spatial_ndim: int, np: Any) -> tuple[Any, Any]:
    """Return inner face-connected boundary voxels and mask-derived outward hints."""

    positive = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)
    orientation = np.zeros(labels.shape + (3,), dtype=np.float32)
    active_axes = (1, 2) if spatial_ndim == 2 else (0, 1, 2)
    for axis in active_axes:
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        lower_t = tuple(lower)
        upper_t = tuple(upper)
        different = labels[lower_t] != labels[upper_t]
        lower_exposed = positive[lower_t] & different
        upper_exposed = positive[upper_t] & different
        boundary[lower_t] |= lower_exposed
        boundary[upper_t] |= upper_exposed
        orientation[lower_t + (axis,)] += lower_exposed.astype(np.float32)
        orientation[upper_t + (axis,)] -= upper_exposed.astype(np.float32)

        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis] = 0
        last[axis] = -1
        first_t = tuple(first)
        last_t = tuple(last)
        boundary[first_t] |= positive[first_t]
        boundary[last_t] |= positive[last_t]
        orientation[first_t + (axis,)] -= positive[first_t].astype(np.float32)
        orientation[last_t + (axis,)] += positive[last_t].astype(np.float32)

    norm = np.linalg.norm(orientation, axis=-1, keepdims=True)
    orientation = np.divide(orientation, norm, out=np.zeros_like(orientation), where=norm > 0)
    return boundary, orientation


def _resolve_source_specs(
    trajectory: Any,
    sources: Sequence[BoundarySourceSpec | Mapping[str, Any]] | None,
    object_set: str | None,
) -> list[BoundarySourceSpec]:
    if sources is None:
        if object_set is None:
            raise ValueError("Pass sources or object_set")
        specs = [BoundarySourceSpec(kind="object_set", name=object_set, object_set=object_set)]
    else:
        specs = [item if isinstance(item, BoundarySourceSpec) else BoundarySourceSpec.from_dict(item) for item in sources]
        if object_set is not None:
            raise ValueError("Pass sources or object_set, not both")
    resolved: list[BoundarySourceSpec] = []
    for spec in specs:
        name = validate_name(spec.name, kind="boundary source")
        if spec.kind == "object_set":
            object_name = validate_name(spec.object_set or name, kind="object set")
            if not trajectory.store.has_observations(object_name):
                raise FileNotFoundError(f"/object_sets/{object_name}/observations")
            object_metadata = trajectory.store.read_json(f"/object_sets/{object_name}/object_set.json")
            label_name = validate_name(
                spec.label_set or str(object_metadata.get("source_label_set") or object_name),
                kind="label set",
            )
            resolved.append(
                BoundarySourceSpec(
                    kind="object_set",
                    name=name,
                    object_set=object_name,
                    label_set=label_name,
                    frames=spec.frames,
                    role=spec.role,
                    metadata=spec.metadata,
                )
            )
        elif spec.kind == "label_set":
            label_name = validate_name(spec.label_set or name, kind="label set")
            resolved.append(
                BoundarySourceSpec(
                    kind="label_set",
                    name=name,
                    label_set=label_name,
                    frames=spec.frames,
                    role=spec.role,
                    metadata=spec.metadata,
                )
            )
        else:
            mask_name = validate_name(spec.label_set or name, kind="mask set")
            resolved.append(
                BoundarySourceSpec(
                    kind="mask_set",
                    name=name,
                    label_set=mask_name,
                    frames=spec.frames,
                    role=spec.role,
                    metadata=spec.metadata,
                )
            )
    if not resolved:
        raise ValueError("At least one boundary source is required")
    return resolved


def build_boundary_library(
    trajectory: Any,
    boundary_set: str,
    *,
    sources: Sequence[BoundarySourceSpec | Mapping[str, Any]] | None = None,
    object_set: str | None = None,
    frames: Sequence[int] | None = None,
    coordinate_scale: Sequence[float] | None = None,
    overwrite: bool = False,
    save_outputs: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryLibraryResult:
    """Extract immutable native boundaries from one or more label-like sources."""

    np = _require_numpy()
    name = validate_name(boundary_set, kind="boundary set")
    specs = _resolve_source_specs(trajectory, sources, object_set)
    requested_frames = None if frames is None else sorted(dict.fromkeys(int(value) for value in frames))
    calibration = registration_calibration(trajectory.metadata)
    scale = np.asarray(coordinate_scale or calibration["coordinate_scale"], dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("coordinate_scale must contain three finite positive values in Z,Y,X order")

    entity_records: list[tuple[Any, ...]] = []
    point_blocks: dict[str, list[Any]] = {
        "point_id": [],
        "boundary_entity_id": [],
        "frame": [],
        "native_index_zyx": [],
        "native_position_zyx": [],
        "orientation_hint_zyx": [],
    }
    source_records: list[dict[str, Any]] = []
    point_start = 0
    entity_id = 1
    point_id = 1
    spatial_ndim_values: set[int] = set()
    frame_shapes: dict[int, tuple[int, ...]] = {}

    for source_id, spec in enumerate(specs, start=1):
        if spec.kind == "mask_set":
            available = trajectory.mask_frames(str(spec.label_set))
        else:
            available = trajectory.label_frames(str(spec.label_set))
        selected = list(spec.frames) if spec.frames else (requested_frames or available)
        selected = sorted(dict.fromkeys(int(value) for value in selected))
        source_records.append(
            {
                "source_id": source_id,
                **spec.to_dict(),
                "frames": selected,
                "identity": (
                    "observation_id"
                    if spec.kind == "object_set"
                    else ("source_label_id" if spec.kind == "label_set" else "one_entity_per_frame")
                ),
            }
        )
        observations_by_frame: dict[int, Any] = {}
        if spec.kind == "object_set":
            observations = trajectory.store.read_observations(str(spec.object_set))
            for frame in selected:
                observations_by_frame[frame] = observations[observations["frame"] == frame]

        for frame in selected:
            if spec.kind == "mask_set":
                if not trajectory.store.has_mask_frame(str(spec.label_set), frame):
                    raise FileNotFoundError(f"/masks/{spec.label_set}/frame_{frame}")
                raw = trajectory.read_mask_frame(str(spec.label_set), frame)
                labels, spatial_ndim = _normalize_zyx(np.asarray(raw, dtype=bool).astype(np.uint8), np=np)
            else:
                if not trajectory.store.has_label_frame(str(spec.label_set), frame):
                    raise FileNotFoundError(f"/labels/{spec.label_set}/frame_{frame}")
                labels, spatial_ndim = _normalize_zyx(trajectory.read_label_frame(str(spec.label_set), frame), np=np)
            spatial_ndim_values.add(spatial_ndim)
            shape = tuple(int(value) for value in labels.shape)
            prior_shape = frame_shapes.setdefault(int(frame), shape)
            if prior_shape != shape:
                raise ValueError(
                    f"Boundary sources for frame {int(frame)} do not share one native grid: "
                    f"expected {prior_shape}, found {shape} for source {spec.name!r}"
                )
            boundary_mask, orientation = _boundary_mask_and_orientation(labels, spatial_ndim=spatial_ndim, np=np)
            coords_all = np.argwhere(boundary_mask)
            labels_all = labels[boundary_mask].astype(np.int64, copy=False)
            hints_all = orientation[boundary_mask]
            order = np.argsort(labels_all, kind="stable") if labels_all.size else np.empty(0, dtype=np.int64)
            coords_all = coords_all[order]
            labels_all = labels_all[order]
            hints_all = hints_all[order]

            if spec.kind == "object_set":
                entity_items = [
                    (int(row["label_id"]), int(row["observation_id"]))
                    for row in observations_by_frame.get(frame, ())
                ]
            elif spec.kind == "label_set":
                entity_items = [(int(value), 0) for value in np.unique(labels[labels > 0])]
            else:
                entity_items = [(1, 0)] if np.any(labels > 0) else []

            for label_id, observation_id in entity_items:
                left = int(np.searchsorted(labels_all, label_id, side="left"))
                right = int(np.searchsorted(labels_all, label_id, side="right"))
                coords = coords_all[left:right]
                hints = hints_all[left:right]
                count = int(coords.shape[0])
                quality = ENTITY_QUALITY_NO_POINTS if count == 0 else 0
                entity_records.append(
                    (
                        entity_id,
                        source_id,
                        frame,
                        observation_id,
                        label_id,
                        point_start,
                        count,
                        quality,
                    )
                )
                if count:
                    ids = np.arange(point_id, point_id + count, dtype=np.int64)
                    point_blocks["point_id"].append(ids)
                    point_blocks["boundary_entity_id"].append(np.full(count, entity_id, dtype=np.int64))
                    point_blocks["frame"].append(np.full(count, frame, dtype=np.int32))
                    point_blocks["native_index_zyx"].append(coords.astype(np.int32, copy=False))
                    point_blocks["native_position_zyx"].append((coords * scale[None, :]).astype(np.float64))
                    point_blocks["orientation_hint_zyx"].append(hints.astype(np.float32, copy=False))
                    point_id += count
                    point_start += count
                entity_id += 1

    if len(spatial_ndim_values) > 1:
        raise ValueError("All sources in one boundary library must have the same spatial dimensionality")
    spatial_ndim = next(iter(spatial_ndim_values), 3)
    entities = np.asarray(entity_records, dtype=boundary_entity_dtype())
    empty_shapes = {
        "point_id": (0,),
        "boundary_entity_id": (0,),
        "frame": (0,),
        "native_index_zyx": (0, 3),
        "native_position_zyx": (0, 3),
        "orientation_hint_zyx": (0, 3),
    }
    dtypes = {
        "point_id": np.int64,
        "boundary_entity_id": np.int64,
        "frame": np.int32,
        "native_index_zyx": np.int32,
        "native_position_zyx": np.float64,
        "orientation_hint_zyx": np.float32,
    }
    points = {
        key: (np.concatenate(blocks, axis=0) if blocks else np.empty(empty_shapes[key], dtype=dtypes[key]))
        for key, blocks in point_blocks.items()
    }
    schema = {
        "schema": "celltraj2.boundary_library.v1",
        "boundary_set": name,
        "created_at": utc_now_iso(),
        "coordinate_order": ["z", "y", "x"],
        "native_index_coordinate_system": "native_roi_array_index",
        "native_position_coordinate_system": "native_roi_physical",
        "coordinate_scale_zyx": scale.tolist(),
        "distance_unit": str(calibration["distance_unit"] if coordinate_scale is None else "scaled_coordinate_unit"),
        "calibration_source": str(calibration["source"] if coordinate_scale is None else "explicit"),
        "registration_applied": False,
        "spatial_ndim": int(spatial_ndim),
        "entity_id_base": 1,
        "point_id_base": 1,
        "point_row_alignment": "row_index_zero_based_maps_to_point_id_minus_1",
        "entity_point_layout": "contiguous_half_open_point_start_point_count",
        "entity_sort_order": ["source_id", "frame", "source_label_id"],
        "entity_count": int(entities.shape[0]),
        "point_count": int(points["point_id"].shape[0]),
        "boundary_digest": boundary_digest(entities, points, source_records),
        "metadata": _json_safe(dict(metadata or {})),
    }
    path = None
    if save_outputs:
        path = trajectory.store.write_boundary_library(
            name,
            entities=entities,
            points=points,
            sources=source_records,
            schema=schema,
            overwrite=overwrite,
        )
    return BoundaryLibraryResult(
        boundary_set=name,
        entities=entities,
        points=points,
        schema=schema,
        sources=source_records,
        saved=bool(save_outputs),
        boundary_path=path,
    )


def _safe_normalize(values: Any, *, np: Any, eps: float = 1e-12) -> Any:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values, dtype=float), where=norms > eps)


def _local_knn(points: Any, k: int, *, np: Any) -> Any:
    count = int(points.shape[0])
    k_eff = min(max(1, int(k)), max(1, count - 1))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        _, indices = cKDTree(points).query(points, k=min(count, k_eff + 1))
        indices = np.asarray(indices)
        if indices.ndim == 1:
            indices = indices[:, None]
        return indices[:, 1 : k_eff + 1].astype(np.int64, copy=False)
    except ImportError:
        result = np.empty((count, k_eff), dtype=np.int64)
        block = max(1, min(count, 1024))
        for start in range(0, count, block):
            stop = min(count, start + block)
            delta = points[start:stop, None, :] - points[None, :, :]
            distances = np.sum(delta * delta, axis=2)
            rows = np.arange(stop - start)
            distances[rows, np.arange(start, stop)] = np.inf
            candidate = np.argpartition(distances, k_eff - 1, axis=1)[:, :k_eff]
            candidate_distance = np.take_along_axis(distances, candidate, axis=1)
            order = np.argsort(candidate_distance, axis=1, kind="stable")
            result[start:stop] = np.take_along_axis(candidate, order, axis=1)
        return result


def _local_geometry(points: Any, orientation: Any, *, spatial_ndim: int, knn: int, np: Any) -> dict[str, Any]:
    """Deterministic local-PCA fallback with a tangent-plane shape operator."""

    count = int(points.shape[0])
    neighbors = _local_knn(points, knn, np=np)
    normals = np.full((count, 3), np.nan, dtype=float)
    xb = np.full((count, 3), np.nan, dtype=float)
    yb = np.full((count, 3), np.nan, dtype=float)
    k1 = np.full(count, np.nan, dtype=float)
    k2 = np.full(count, np.nan, dtype=float)
    centroid = np.mean(points, axis=0)

    for index in range(count):
        local = points[neighbors[index]] - points[index]
        if spatial_ndim == 2:
            covariance = local[:, 1:].T @ local[:, 1:]
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            tangent_yx = eigenvectors[:, int(np.argmax(eigenvalues))]
            tangent = np.asarray([0.0, tangent_yx[0], tangent_yx[1]])
            normal = np.asarray([0.0, -tangent_yx[1], tangent_yx[0]])
            hint = orientation[index]
            if np.dot(normal, hint if np.linalg.norm(hint) > 0 else points[index] - centroid) < 0:
                normal *= -1.0
            tangent = _safe_normalize(tangent[None, :], np=np)[0]
            normal = _safe_normalize(normal[None, :], np=np)[0]
            out_of_plane = np.asarray([1.0, 0.0, 0.0])
            u = local @ tangent
            w = local @ normal
            design = np.column_stack([0.5 * u * u, u, np.ones_like(u)])
            try:
                coefficient = np.linalg.lstsq(design, w, rcond=None)[0]
                k1[index] = float(coefficient[0])
            except np.linalg.LinAlgError:
                pass
            normals[index], xb[index], yb[index] = normal, tangent, out_of_plane
            continue

        covariance = local.T @ local
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        normal = eigenvectors[:, order[0]]
        tangent_x = eigenvectors[:, order[2]]
        hint = orientation[index]
        if np.dot(normal, hint if np.linalg.norm(hint) > 0 else points[index] - centroid) < 0:
            normal *= -1.0
        tangent_y = np.cross(normal, tangent_x)
        tangent_x = _safe_normalize(tangent_x[None, :], np=np)[0]
        tangent_y = _safe_normalize(tangent_y[None, :], np=np)[0]
        u = local @ tangent_x
        v = local @ tangent_y
        w = local @ normal
        design = np.column_stack([0.5 * u * u, u * v, 0.5 * v * v, u, v, np.ones_like(u)])
        try:
            coefficient = np.linalg.lstsq(design, w, rcond=None)[0]
            shape = np.asarray([[coefficient[0], coefficient[1]], [coefficient[1], coefficient[2]]])
            curvature = np.linalg.eigvalsh(shape)
            k1[index], k2[index] = float(curvature[1]), float(curvature[0])
        except np.linalg.LinAlgError:
            pass
        normals[index], xb[index], yb[index] = normal, tangent_x, tangent_y

    return {
        "normals_zyx": normals,
        "tangent_x_zyx": xb,
        "tangent_y_zyx": yb,
        "principal_curvature_1": k1,
        "principal_curvature_2": k2,
        "edge_index": np.column_stack(
            [np.repeat(np.arange(count, dtype=np.int64), neighbors.shape[1]), neighbors.reshape(-1)]
        ),
    }


def _pcdiff_geometry(points: Any, orientation: Any, *, knn: int, np: Any) -> dict[str, Any]:
    from pcdiff import build_grad_div, estimate_basis, knn_graph  # type: ignore

    edge_index = knn_graph(points, min(int(knn), max(2, points.shape[0] - 1)))
    normals, xb, yb = estimate_basis(points, edge_index, orientation=orientation)
    normals = np.asarray(normals, dtype=float)
    xb = np.asarray(xb, dtype=float)
    yb = np.asarray(yb, dtype=float)
    dot = np.sum(normals * orientation, axis=1)
    flip = np.isfinite(dot) & (dot < 0)
    normals[flip] *= -1.0
    xb[flip] *= -1.0
    grad, _ = build_grad_div(points, normals, xb, yb, edge_index)
    derivatives = []
    for component in range(3):
        derivatives.append(np.asarray(grad @ normals[:, component]).reshape(-1, 2))
    dn_du = np.column_stack([value[:, 0] for value in derivatives])
    dn_dv = np.column_stack([value[:, 1] for value in derivatives])
    shape = np.empty((points.shape[0], 2, 2), dtype=float)
    shape[:, 0, 0] = -np.sum(xb * dn_du, axis=1)
    shape[:, 0, 1] = -np.sum(xb * dn_dv, axis=1)
    shape[:, 1, 0] = -np.sum(yb * dn_du, axis=1)
    shape[:, 1, 1] = -np.sum(yb * dn_dv, axis=1)
    shape = 0.5 * (shape + np.swapaxes(shape, 1, 2))
    curvature = np.linalg.eigvalsh(shape)
    src = np.asarray(edge_index[0], dtype=np.int64).reshape(-1)
    dst = np.asarray(edge_index[1], dtype=np.int64).reshape(-1)
    return {
        "normals_zyx": normals,
        "tangent_x_zyx": xb,
        "tangent_y_zyx": yb,
        "principal_curvature_1": curvature[:, 1],
        "principal_curvature_2": curvature[:, 0],
        "edge_index": np.column_stack([src, dst]),
    }


def compute_boundary_geometry(
    trajectory: Any,
    boundary_set: str,
    *,
    geometry_set: str = "surface_v1",
    knn: int = 40,
    backend: Literal["auto", "pcdiff", "local"] = "auto",
    source_ids: Sequence[int] | None = None,
    source_names: Sequence[str] | None = None,
    source_roles: Sequence[str] | None = None,
    library: BoundaryLibraryResult | BoundaryDataView | None = None,
    overwrite: bool = False,
    save_outputs: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryGeometryResult:
    """Calculate row-aligned normals, tangent frames, and signed curvatures."""

    np = _require_numpy()
    boundary_name = validate_name(boundary_set, kind="boundary set")
    geometry_name = validate_name(geometry_set, kind="boundary geometry set")
    if knn < 2:
        raise ValueError("knn must be >= 2")
    view = as_boundary_library_view(trajectory, boundary_name, library)
    selected_source_ids = resolve_boundary_source_ids(
        view,
        source_ids=source_ids,
        source_names=source_names,
        source_roles=source_roles,
    )
    count = view.point_count
    values = {
        "normals_zyx": np.full((count, 3), np.nan, dtype=np.float32),
        "tangent_x_zyx": np.full((count, 3), np.nan, dtype=np.float32),
        "tangent_y_zyx": np.full((count, 3), np.nan, dtype=np.float32),
        "principal_curvature_1": np.full(count, np.nan, dtype=np.float32),
        "principal_curvature_2": np.full(count, np.nan, dtype=np.float32),
        "mean_curvature": np.full(count, np.nan, dtype=np.float32),
        "gaussian_curvature": np.full(count, np.nan, dtype=np.float32),
        "quality_flags": np.zeros(count, dtype=np.uint32),
    }
    topology_rows: list[Any] = [np.empty(0, dtype=np.int64) for _ in range(count)]
    selected_backends: set[str] = set()
    spatial_ndim = int(view.schema.get("spatial_ndim", 3))
    pcdiff_available = False
    if backend in {"auto", "pcdiff"} and spatial_ndim == 3:
        try:
            import pcdiff  # noqa: F401

            pcdiff_available = True
        except ImportError:
            if backend == "pcdiff":
                raise RuntimeError(
                    "pcdiff geometry was requested but pcdiff is not installed. "
                    "Install celltraj2 with the analysis extra."
                )

    for entity in view.entities:
        start = int(entity["point_start"])
        stop = start + int(entity["point_count"])
        if selected_source_ids is not None and int(entity["source_id"]) not in selected_source_ids:
            values["quality_flags"][start:stop] |= GEOMETRY_QUALITY_NOT_SELECTED
            continue
        entity_count = stop - start
        minimum = 5 if spatial_ndim == 3 else 3
        if entity_count < minimum:
            values["quality_flags"][start:stop] |= GEOMETRY_QUALITY_TOO_FEW_POINTS
            continue
        point_data = view.read_points(
            rows=slice(start, stop),
            fields=("native_position_zyx", "orientation_hint_zyx"),
        )
        points = np.asarray(point_data["native_position_zyx"], dtype=float)
        orientation = np.asarray(point_data["orientation_hint_zyx"], dtype=float)
        zero_hint = np.linalg.norm(orientation, axis=1) <= 1e-12
        if np.any(zero_hint):
            orientation[zero_hint] = points[zero_hint] - np.mean(points, axis=0)
            values["quality_flags"][start:stop][zero_hint] |= GEOMETRY_QUALITY_DISCONNECTED_HINT
        orientation = _safe_normalize(orientation, np=np)
        try:
            if pcdiff_available:
                geometry = _pcdiff_geometry(points, orientation, knn=knn, np=np)
                selected_backends.add("pcdiff_1.0.1_shape_operator")
            else:
                geometry = _local_geometry(points, orientation, spatial_ndim=spatial_ndim, knn=knn, np=np)
                selected_backends.add("local_pca_quadratic_shape_operator")
        except Exception:
            if backend != "auto" or not pcdiff_available:
                raise
            geometry = _local_geometry(points, orientation, spatial_ndim=spatial_ndim, knn=knn, np=np)
            selected_backends.add("local_pca_quadratic_shape_operator")
            values["quality_flags"][start:stop] |= GEOMETRY_QUALITY_OPERATOR_FAILED
        for key in (
            "normals_zyx",
            "tangent_x_zyx",
            "tangent_y_zyx",
            "principal_curvature_1",
            "principal_curvature_2",
        ):
            values[key][start:stop] = geometry[key]
        edge_index = np.asarray(geometry["edge_index"], dtype=np.int64)
        if edge_index.ndim == 2 and edge_index.shape[1] == 2:
            for local_source in range(entity_count):
                local_targets = edge_index[edge_index[:, 0] == local_source, 1]
                valid = (local_targets >= 0) & (local_targets < entity_count)
                topology_rows[start + local_source] = np.unique(local_targets[valid] + start)
        k1 = np.asarray(geometry["principal_curvature_1"])
        k2 = np.asarray(geometry["principal_curvature_2"])
        if spatial_ndim == 2:
            values["mean_curvature"][start:stop] = k1
        else:
            values["mean_curvature"][start:stop] = 0.5 * (k1 + k2)
            values["gaussian_curvature"][start:stop] = k1 * k2

    topology_counts = np.asarray([row.shape[0] for row in topology_rows], dtype=np.int64)
    topology_indptr = np.empty(count + 1, dtype=np.int64)
    topology_indptr[0] = 0
    np.cumsum(topology_counts, out=topology_indptr[1:])
    topology_indices = (
        np.concatenate(topology_rows) if int(topology_indptr[-1]) else np.empty(0, dtype=np.int64)
    )
    schema = {
        "schema": "celltraj2.boundary_geometry.v1",
        "boundary_set": boundary_name,
        "geometry_set": geometry_name,
        "created_at": utc_now_iso(),
        "row_alignment": f"/boundaries/{boundary_name}/points",
        "coordinate_system": "native_roi_physical",
        "registration_dependency": None,
        "spatial_ndim": spatial_ndim,
        "knn": int(knn),
        "requested_backend": backend,
        "selected_backends": sorted(selected_backends),
        "selected_source_ids": (
            "all" if selected_source_ids is None else sorted(selected_source_ids)
        ),
        "normal_orientation": "mask_inside_to_outside_hint_with_centroid_fallback",
        "shape_operator": "minus_surface_gradient_of_oriented_normal",
        "curvature_sign": "positive_when_shape_operator_eigenvalue_is_positive_for_stored_outward_normal",
        "mean_curvature": "0.5*(k1+k2) for 2D manifolds; signed curve curvature for 1D boundaries",
        "gaussian_curvature": "k1*k2 for 2D manifolds; NaN for 1D boundaries",
        "topology": {
            "format": "csr",
            "row_alignment": f"/boundaries/{boundary_name}/points/point_id",
            "indices": "zero_based_target_point_rows",
            "edge_count": int(topology_indices.shape[0]),
        },
        "metadata": _json_safe(dict(metadata or {})),
    }
    path = None
    if save_outputs:
        path = trajectory.store.write_boundary_geometry(
            boundary_name,
            geometry_name,
            values=values,
            topology_indptr=topology_indptr,
            topology_indices=topology_indices,
            schema=schema,
            overwrite=overwrite,
        )
    return BoundaryGeometryResult(
        boundary_set=boundary_name,
        geometry_set=geometry_name,
        values=values,
        topology_indptr=topology_indptr,
        topology_indices=topology_indices,
        schema=schema,
        saved=bool(save_outputs),
        geometry_path=path,
    )


def _query_external_neighbors(
    source_points: Any,
    source_entities: Any,
    target_points: Any,
    target_entities: Any,
    target_rows: Any,
    *,
    k: int,
    exclude_same_entity: bool,
    max_distance: float | None,
    np: Any,
) -> tuple[list[Any], list[Any]]:
    """Return per-source target rows and distances, preferring scipy cKDTree."""

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(target_points)
        result_rows: list[Any] = []
        result_distances: list[Any] = []
        target_count = int(target_points.shape[0])
        initial = min(target_count, max(32, k + 8))
        distance_bound = float(max_distance) if max_distance is not None else np.inf
        distances, local_indices = tree.query(source_points, k=initial, distance_upper_bound=distance_bound)
        if initial == 1:
            distances = distances[:, None]
            local_indices = local_indices[:, None]
        for row_index in range(source_points.shape[0]):
            current_k = initial
            d = np.asarray(distances[row_index]).reshape(-1)
            j = np.asarray(local_indices[row_index]).reshape(-1)
            valid = j < target_count
            if exclude_same_entity:
                valid &= target_entities[np.minimum(j, target_count - 1)] != source_entities[row_index]
            while int(np.sum(valid)) < k and current_k < target_count and max_distance is None:
                current_k = min(target_count, current_k * 2)
                d, j = tree.query(source_points[row_index], k=current_k)
                d = np.asarray(d).reshape(-1)
                j = np.asarray(j).reshape(-1)
                valid = j < target_count
                if exclude_same_entity:
                    valid &= target_entities[np.minimum(j, target_count - 1)] != source_entities[row_index]
            chosen = np.flatnonzero(valid)[:k]
            result_rows.append(target_rows[j[chosen]].astype(np.int64, copy=False))
            result_distances.append(d[chosen].astype(float, copy=False))
        return result_rows, result_distances
    except ImportError:
        result_rows = []
        result_distances = []
        for index, point in enumerate(source_points):
            delta = target_points - point[None, :]
            distance = np.sqrt(np.sum(delta * delta, axis=1))
            valid = np.isfinite(distance)
            if exclude_same_entity:
                valid &= target_entities != source_entities[index]
            if max_distance is not None:
                valid &= distance <= float(max_distance)
            candidates = np.flatnonzero(valid)
            order = np.argsort(distance[candidates], kind="stable")[:k]
            selected = candidates[order]
            result_rows.append(target_rows[selected].astype(np.int64, copy=False))
            result_distances.append(distance[selected].astype(float, copy=False))
        return result_rows, result_distances


def compute_boundary_neighbors(
    trajectory: Any,
    boundary_set: str,
    *,
    neighbor_set: str = "nearest_external_v1",
    k: int = 1,
    source_entity_ids: Sequence[int] | None = None,
    target_entity_ids: Sequence[int] | None = None,
    source_ids: Sequence[int] | None = None,
    source_names: Sequence[str] | None = None,
    source_roles: Sequence[str] | None = None,
    target_ids: Sequence[int] | None = None,
    target_names: Sequence[str] | None = None,
    target_roles: Sequence[str] | None = None,
    same_frame: bool = True,
    exclude_same_entity: bool = True,
    max_distance: float | None = None,
    library: BoundaryLibraryResult | BoundaryDataView | None = None,
    overwrite: bool = False,
    save_outputs: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryNeighborResult:
    """Build a point-row CSR graph of nearest interacting boundary points."""

    np = _require_numpy()
    boundary_name = validate_name(boundary_set, kind="boundary set")
    neighbor_name = validate_name(neighbor_set, kind="boundary neighbor set")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    if max_distance is not None and (not np.isfinite(max_distance) or float(max_distance) <= 0):
        raise ValueError("max_distance must be finite and > 0")
    view = as_boundary_library_view(trajectory, boundary_name, library)
    selected_source_ids = resolve_boundary_source_ids(
        view,
        source_ids=source_ids,
        source_names=source_names,
        source_roles=source_roles,
    )
    selected_target_ids = resolve_boundary_source_ids(
        view,
        source_ids=target_ids,
        source_names=target_names,
        source_roles=target_roles,
    )
    all_points = view.read_points(fields=("boundary_entity_id", "frame", "native_position_zyx"))
    positions = np.asarray(all_points["native_position_zyx"], dtype=float)
    point_entities = np.asarray(all_points["boundary_entity_id"], dtype=np.int64)
    point_frames = np.asarray(all_points["frame"], dtype=np.int32)
    point_count = int(positions.shape[0])
    source_entity_id_set = (
        set(int(value) for value in source_entity_ids) if source_entity_ids is not None else None
    )
    target_entity_id_set = (
        set(int(value) for value in target_entity_ids) if target_entity_ids is not None else None
    )
    source_mask = np.ones(point_count, dtype=bool)
    target_mask = np.ones(point_count, dtype=bool)
    if source_entity_id_set is not None:
        source_mask &= np.isin(point_entities, list(source_entity_id_set))
    if target_entity_id_set is not None:
        target_mask &= np.isin(point_entities, list(target_entity_id_set))
    if selected_source_ids is not None or selected_target_ids is not None:
        entity_source_by_id = np.zeros(int(view.entities.shape[0]) + 1, dtype=np.int32)
        entity_source_by_id[
            np.asarray(view.entities["boundary_entity_id"], dtype=np.int64)
        ] = np.asarray(view.entities["source_id"], dtype=np.int32)
        point_source_ids = entity_source_by_id[point_entities]
        if selected_source_ids is not None:
            source_mask &= np.isin(point_source_ids, list(selected_source_ids))
        if selected_target_ids is not None:
            target_mask &= np.isin(point_source_ids, list(selected_target_ids))

    row_targets: list[Any] = [np.empty(0, dtype=np.int64) for _ in range(point_count)]
    row_distances: list[Any] = [np.empty(0, dtype=float) for _ in range(point_count)]
    frame_groups = sorted(int(value) for value in np.unique(point_frames[source_mask])) if same_frame else [None]
    for frame in frame_groups:
        frame_source = source_mask if frame is None else source_mask & (point_frames == frame)
        frame_target = target_mask if frame is None else target_mask & (point_frames == frame)
        source_rows = np.flatnonzero(frame_source)
        target_rows = np.flatnonzero(frame_target)
        if not source_rows.size or not target_rows.size:
            continue
        targets, distances = _query_external_neighbors(
            positions[source_rows],
            point_entities[source_rows],
            positions[target_rows],
            point_entities[target_rows],
            target_rows,
            k=int(k),
            exclude_same_entity=exclude_same_entity,
            max_distance=max_distance,
            np=np,
        )
        for source_row, selected_rows, selected_distances in zip(source_rows, targets, distances):
            row_targets[int(source_row)] = selected_rows
            row_distances[int(source_row)] = selected_distances

    counts = np.asarray([rows.shape[0] for rows in row_targets], dtype=np.int64)
    indptr = np.empty(point_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = np.concatenate(row_targets) if int(indptr[-1]) else np.empty(0, dtype=np.int64)
    distance = np.concatenate(row_distances).astype(np.float32) if int(indptr[-1]) else np.empty(0, dtype=np.float32)
    source_rows_expanded = np.repeat(np.arange(point_count, dtype=np.int64), counts)
    displacement = (
        positions[indices] - positions[source_rows_expanded]
        if indices.size
        else np.empty((0, 3), dtype=float)
    ).astype(np.float32)
    schema = {
        "schema": "celltraj2.boundary_neighbors.v1",
        "boundary_set": boundary_name,
        "neighbor_set": neighbor_name,
        "created_at": utc_now_iso(),
        "format": "csr",
        "row_alignment": f"/boundaries/{boundary_name}/points",
        "indices": "zero_based_target_point_rows",
        "coordinate_system": "native_roi_physical",
        "registration_dependency": None,
        "k": int(k),
        "same_frame": bool(same_frame),
        "exclude_same_entity": bool(exclude_same_entity),
        "max_distance": None if max_distance is None else float(max_distance),
        "source_entity_ids": (
            "all" if source_entity_id_set is None else sorted(source_entity_id_set)
        ),
        "target_entity_ids": (
            "all" if target_entity_id_set is None else sorted(target_entity_id_set)
        ),
        "source_ids": "all" if selected_source_ids is None else sorted(selected_source_ids),
        "target_ids": "all" if selected_target_ids is None else sorted(selected_target_ids),
        "edge_count": int(indices.shape[0]),
        "metadata": _json_safe(dict(metadata or {})),
    }
    path = None
    if save_outputs:
        path = trajectory.store.write_boundary_neighbors(
            boundary_name,
            neighbor_name,
            indptr=indptr,
            indices=indices,
            distance=distance,
            displacement_zyx=displacement,
            schema=schema,
            overwrite=overwrite,
        )
    return BoundaryNeighborResult(
        boundary_set=boundary_name,
        neighbor_set=neighbor_name,
        indptr=indptr,
        indices=indices,
        distance=distance,
        displacement_zyx=displacement,
        schema=schema,
        saved=bool(save_outputs),
        neighbor_path=path,
    )


def pairwise_distance_matrix(source_points: Any, target_points: Any) -> Any:
    """Return Euclidean distance costs without requiring scipy."""

    np = _require_numpy()
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source_points and target_points must be N x D and M x D")
    delta = source[:, None, :] - target[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def optimal_transport_plan(
    source_points: Any,
    target_points: Any,
    *,
    method: Literal["emd", "sinkhorn"] = "emd",
    regularization: float = 0.05,
    mass_tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> BoundaryTransportPlan:
    """Compute a uniform-mass OT plan and return only non-negligible edges."""

    np = _require_numpy()
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if not source.shape[0] or not target.shape[0]:
        raise ValueError("Optimal transport requires non-empty source and target points")
    cost = pairwise_distance_matrix(source, target)
    a = np.full(source.shape[0], 1.0 / source.shape[0], dtype=float)
    b = np.full(target.shape[0], 1.0 / target.shape[0], dtype=float)
    selected_method = method
    if method == "emd":
        try:
            import ot  # type: ignore

            plan = np.asarray(ot.emd(a, b, cost, numItermax=int(max_iterations)), dtype=float)
            selected_method = "pot.emd"
        except ImportError:
            try:
                from scipy.optimize import linprog  # type: ignore
                from scipy.sparse import lil_matrix  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Exact boundary OT requires POT or scipy. Install celltraj2 with the analysis extra."
                ) from exc
            n_source, n_target = cost.shape
            constraints = lil_matrix((n_source + n_target, n_source * n_target), dtype=float)
            for index in range(n_source):
                constraints[index, index * n_target : (index + 1) * n_target] = 1.0
            for index in range(n_target):
                constraints[n_source + index, index::n_target] = 1.0
            solution = linprog(
                cost.reshape(-1),
                A_eq=constraints.tocsr(),
                b_eq=np.concatenate([a, b]),
                bounds=(0.0, None),
                method="highs",
            )
            if not solution.success:
                raise RuntimeError(f"Boundary OT linear program failed: {solution.message}")
            plan = solution.x.reshape(cost.shape)
            selected_method = "scipy.optimize.linprog_highs"
    elif method == "sinkhorn":
        epsilon = float(regularization)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("regularization must be finite and > 0")
        kernel = np.exp(-cost / epsilon)
        kernel = np.maximum(kernel, np.finfo(float).tiny)
        u = np.ones_like(a)
        v = np.ones_like(b)
        for _ in range(int(max_iterations)):
            previous = u
            u = a / np.maximum(kernel @ v, np.finfo(float).tiny)
            v = b / np.maximum(kernel.T @ u, np.finfo(float).tiny)
            if np.max(np.abs(u - previous)) < 1e-10:
                break
        plan = (u[:, None] * kernel) * v[None, :]
        selected_method = "numpy.sinkhorn"
    else:
        raise ValueError(f"Unsupported OT method {method!r}")
    source_rows, target_rows = np.nonzero(plan > float(mass_tolerance))
    masses = plan[source_rows, target_rows]
    edge_cost = cost[source_rows, target_rows]
    return BoundaryTransportPlan(
        source_rows=source_rows.astype(np.int64),
        target_rows=target_rows.astype(np.int64),
        mass=masses.astype(np.float64),
        edge_cost=edge_cost.astype(np.float64),
        total_cost=float(np.sum(masses * edge_cost)),
        method=selected_method,
    )


def deterministic_point_sample(point_count: int, max_points: int | None) -> Any:
    """Return stable, endpoint-preserving sample rows for reproducible OT costs."""

    np = _require_numpy()
    count = int(point_count)
    if max_points is None or count <= int(max_points):
        return np.arange(count, dtype=np.int64)
    if int(max_points) < 2:
        raise ValueError("max_points must be >= 2 when provided")
    return np.unique(np.rint(np.linspace(0, count - 1, int(max_points))).astype(np.int64))


__all__ = [
    "BoundaryGeometryResult",
    "BoundaryLibraryResult",
    "BoundaryLibraryView",
    "InMemoryBoundaryLibraryView",
    "BoundaryNeighborResult",
    "BoundarySourceSpec",
    "BoundaryTransportPlan",
    "boundary_entity_dtype",
    "boundary_digest",
    "boundary_motion_link_dtype",
    "build_boundary_library",
    "compute_boundary_geometry",
    "compute_boundary_neighbors",
    "resolve_boundary_source_ids",
    "deterministic_point_sample",
    "optimal_transport_plan",
    "pairwise_distance_matrix",
]
