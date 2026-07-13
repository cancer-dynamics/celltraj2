"""Sparse lineage graphs and first-pass centroid tracking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

from celltraj2.paths import validate_name
from celltraj2.schema import utc_now_iso


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "celltraj2 tracking requires numpy. Install with "
            "`python -m pip install -e .[analysis]`."
        ) from exc
    return np


def link_dtype() -> Any:
    """Return the canonical structured dtype for lineage-link metadata."""

    np = _require_numpy()
    return np.dtype(
        [
            ("link_id", "<i8"),
            ("parent_observation_id", "<i8"),
            ("child_observation_id", "<i8"),
            ("source_frame", "<i4"),
            ("target_frame", "<i4"),
            ("centroid_distance", "<f8"),
            ("cost", "<f8"),
            ("confidence", "<f8"),
            ("quality_flags", "<u4"),
        ]
    )


def assignment_dtype() -> Any:
    """Return the row-aligned derived track-assignment dtype."""

    np = _require_numpy()
    return np.dtype(
        [
            ("observation_id", "<i8"),
            ("parent_observation_id", "<i8"),
            ("lineage_id", "<i8"),
            ("tracklet_id", "<i8"),
            ("generation", "<i4"),
            ("depth", "<i4"),
            ("n_children", "<i4"),
        ]
    )


@dataclass(frozen=True)
class SparseAdjacency:
    """Dependency-light CSR representation of a parent-to-child graph."""

    indptr: Any
    indices: Any
    data: Any
    shape: tuple[int, int]

    def to_scipy(self, *, topology: bool = False) -> Any:
        """Return a scipy CSR matrix when scipy is installed.

        With ``topology=True``, nonzero data are normalized to boolean values
        so sparse products operate on connectivity rather than link ids.
        """

        try:
            from scipy.sparse import csr_matrix  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Loading a scipy sparse matrix requires scipy.") from exc
        matrix = csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)
        return matrix.astype(bool) if topology else matrix


class TrackGraph:
    """A rooted, forward-branching lineage graph over object observations."""

    def __init__(
        self,
        *,
        adjacency: SparseAdjacency,
        links: Any,
        assignments: Any,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self.adjacency = adjacency
        self.links = links
        self.assignments = assignments
        self.schema = dict(schema or {})
        self._validate()

    @property
    def observation_count(self) -> int:
        return int(self.adjacency.shape[0])

    def _validate(self) -> None:
        np = _require_numpy()
        n = self.observation_count
        if self.adjacency.shape != (n, n):
            raise ValueError("Track adjacency must be square.")
        if np.asarray(self.adjacency.indptr).shape != (n + 1,):
            raise ValueError("CSR indptr length must equal observation_count + 1.")
        indptr = np.asarray(self.adjacency.indptr, dtype=np.int64)
        indices = np.asarray(self.adjacency.indices, dtype=np.int64)
        data = np.asarray(self.adjacency.data)
        if indptr[0] != 0 or np.any(np.diff(indptr) < 0):
            raise ValueError("CSR indptr must start at zero and be monotonic.")
        if int(indptr[-1]) != int(indices.size) or int(data.size) != int(indices.size):
            raise ValueError("CSR indices and data must match the edge count in indptr.")
        if indices.size and (np.any(indices < 0) or np.any(indices >= n)):
            raise ValueError("CSR child indices are outside the observation matrix.")
        if indices.size and np.any(np.bincount(indices, minlength=n) > 1):
            raise ValueError("Each child observation may have at most one parent.")
        if int(np.asarray(self.assignments).shape[0]) != n:
            raise ValueError("Track assignments must be row-aligned to observations.")
        if n and not np.array_equal(
            np.asarray(self.assignments["observation_id"], dtype=np.int64),
            np.arange(1, n + 1, dtype=np.int64),
        ):
            raise ValueError("Track assignments must retain one-based observation row ids.")

    def children(self, observation_id: int) -> Any:
        """Return one-based direct child observation ids."""

        np = _require_numpy()
        row = _observation_row(observation_id, self.observation_count)
        start = int(self.adjacency.indptr[row])
        stop = int(self.adjacency.indptr[row + 1])
        return np.asarray(self.adjacency.indices[start:stop], dtype=np.int64) + 1

    def parent(self, observation_id: int) -> int | None:
        """Return the unique direct parent observation id, if present."""

        row = _observation_row(observation_id, self.observation_count)
        value = int(self.assignments["parent_observation_id"][row])
        return value if value > 0 else None

    def ancestors(self, observation_id: int, *, include_self: bool = False) -> Any:
        """Return ancestors from root to direct parent."""

        np = _require_numpy()
        current = int(observation_id)
        _observation_row(current, self.observation_count)
        values = [current] if include_self else []
        seen = {current}
        while True:
            parent = self.parent(current)
            if parent is None:
                break
            if parent in seen:
                raise ValueError("Cycle detected in lineage graph.")
            seen.add(parent)
            values.append(parent)
            current = parent
        if include_self:
            return np.asarray(list(reversed(values)), dtype=np.int64)
        return np.asarray(list(reversed(values)), dtype=np.int64)

    def descendants(self, observation_id: int, *, include_self: bool = False) -> Any:
        """Return all forward descendants in breadth-first order."""

        np = _require_numpy()
        root = int(observation_id)
        _observation_row(root, self.observation_count)
        values = [root] if include_self else []
        queue = [root]
        seen = {root}
        while queue:
            current = queue.pop(0)
            for child in self.children(current).tolist():
                value = int(child)
                if value in seen:
                    raise ValueError("Cycle or duplicate edge detected in lineage graph.")
                seen.add(value)
                values.append(value)
                queue.append(value)
        return np.asarray(values, dtype=np.int64)

    def history(self, observation_id: int) -> Any:
        """Return the unique root-to-observation history."""

        return self.ancestors(observation_id, include_self=True)

    def lineage(self, observation_id: int) -> Any:
        """Return all observations in the same rooted lineage family."""

        np = _require_numpy()
        row = _observation_row(observation_id, self.observation_count)
        lineage_id = int(self.assignments["lineage_id"][row])
        return np.asarray(self.assignments["observation_id"][self.assignments["lineage_id"] == lineage_id], dtype=np.int64)

    def selection_tree(self, observation_id: int) -> Any:
        """Return the selected observation's ancestors, self, and descendants."""

        np = _require_numpy()
        history = self.history(observation_id)
        future = self.descendants(observation_id)
        return np.concatenate((history, future))

    def maximal_trajectories(self) -> list[Any]:
        """Return canonical maximal root-to-leaf trajectories."""

        leaves = self.assignments["observation_id"][self.assignments["n_children"] == 0]
        return [self.history(int(observation_id)) for observation_id in leaves]

    def maximal_trajectory_matrix(self) -> Any:
        """Return sparse root-to-leaf membership rows using sparse products.

        Rows correspond to leaf observations in ascending observation order;
        columns correspond to zero-based observation rows. This is the sparse
        matrix form of ``maximal_trajectories`` and avoids tracing each history
        in Python when scipy is available.
        """

        np = _require_numpy()
        try:
            from scipy.sparse import csr_matrix  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Sparse trajectory membership requires scipy.") from exc
        leaf_ids = np.asarray(
            self.assignments["observation_id"][self.assignments["n_children"] == 0],
            dtype=np.int64,
        )
        rows = np.arange(leaf_ids.size, dtype=np.int64)
        frontier = csr_matrix(
            (np.ones(leaf_ids.size, dtype=bool), (rows, leaf_ids - 1)),
            shape=(int(leaf_ids.size), self.observation_count),
        )
        membership = frontier.copy()
        parent_step = self.adjacency.to_scipy(topology=True).transpose().tocsr()
        for _ in range(self.observation_count):
            frontier = (frontier @ parent_step).astype(bool).tocsr()
            if frontier.nnz == 0:
                return membership.astype(bool).tocsr()
            membership = (membership + frontier).astype(bool).tocsr()
        raise ValueError("Cycle detected while computing sparse trajectories.")


@dataclass(frozen=True)
class TrackingResult:
    """Result from tracking one object set."""

    object_set: str
    track_set: str
    graph: TrackGraph
    saved: bool
    track_path: str | None = None
    run_id: str | None = None
    frame_counts: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def link_count(self) -> int:
        return int(self.graph.links.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_set": self.object_set,
            "track_set": self.track_set,
            "observation_count": self.graph.observation_count,
            "link_count": self.link_count,
            "track_path": self.track_path,
            "run_id": self.run_id,
            "frame_counts": {str(key): dict(value) for key, value in self.frame_counts.items()},
            "saved": bool(self.saved),
        }


def default_tracking_run_id() -> str:
    """Return an H5-safe default tracking run id."""

    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "_").replace(".", "_")
    return f"track_{stamp}"


def track_minimum_centroid_distance(
    trajectory: Any,
    object_set: str,
    *,
    max_distance: float,
    track_set: str = "centroid_mindist",
    coordinate_scale: Sequence[float] | None = None,
    registration_set: str | None = None,
    overwrite: bool = False,
    save_outputs: bool = True,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrackingResult:
    """Link each observation to its nearest prior-frame centroid.

    Each child independently chooses at most one parent in exactly the previous
    local frame. Multiple children may choose the same parent, preserving the
    forward-branching behavior of legacy ``get_lineage_mindist``.
    """

    np = _require_numpy()
    object_name = validate_name(object_set, kind="object set")
    track_name = validate_name(track_set, kind="track set")
    cutoff = float(max_distance)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("max_distance must be finite and > 0.")
    observations = trajectory.store.read_observations(object_name)
    n = int(observations.shape[0])
    expected_ids = np.arange(1, n + 1, dtype=np.int64)
    if n and not np.array_equal(np.asarray(observations["observation_id"], dtype=np.int64), expected_ids):
        raise ValueError("Observation rows must be aligned to one-based observation_id values.")

    scale = np.asarray(coordinate_scale if coordinate_scale is not None else (1.0, 1.0, 1.0), dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("coordinate_scale must contain three finite positive values in Z,Y,X order.")
    metadata_payload = dict(metadata or {})
    distance_unit = str(
        metadata_payload.get("distance_unit")
        or ("pixel" if coordinate_scale is None else "scaled_coordinate_unit")
    )
    frames = np.asarray(observations["frame"], dtype=np.int64)
    centroids = np.column_stack(
        [observations["centroid_z"], observations["centroid_y"], observations["centroid_x"]]
    ).astype(float)
    centroids *= scale[np.newaxis, :]
    registration = None
    selected_registration = registration_set
    store = getattr(trajectory, "store", None)
    if store is not None and hasattr(store, "read_registration_set"):
        if selected_registration is None and hasattr(store, "active_registration_name"):
            selected_registration = store.active_registration_name()
        if selected_registration:
            try:
                registration = store.read_registration_set(selected_registration)
            except (FileNotFoundError, KeyError):
                if registration_set is not None:
                    raise
    if registration is not None:
        stored_scale = np.asarray(registration.schema.get("coordinate_scale_zyx", scale), dtype=float)
        if (
            str(registration.schema.get("method")) != "identity"
            and (stored_scale.shape != (3,) or not np.allclose(stored_scale, scale, rtol=1e-7, atol=1e-10))
        ):
            raise ValueError(
                f"Registration set {registration.name!r} uses coordinate_scale_zyx "
                f"{stored_scale.tolist()}, not tracker scale {scale.tolist()}."
            )
        centroids = registration.apply_zyx(centroids, frames)
    registration_dependency = (
        None
        if registration is None
        else {
            "registration_set": registration.name,
            "registration_digest": registration.digest,
            "registration_method": str(registration.schema.get("method") or ""),
        }
    )
    edge_records: list[tuple[Any, ...]] = []
    link_id = 1
    for frame in sorted(int(value) for value in np.unique(frames)):
        if frame <= 1:
            continue
        child_rows = np.flatnonzero(frames == frame)
        parent_rows = np.flatnonzero(frames == frame - 1)
        if not child_rows.size or not parent_rows.size:
            continue
        parent_xyz = centroids[parent_rows]
        for child_row in child_rows:
            deltas = parent_xyz - centroids[int(child_row)]
            distances = np.sqrt(np.sum(deltas * deltas, axis=1))
            distances[~np.isfinite(distances)] = np.inf
            nearest_local = int(np.argmin(distances))
            distance = float(distances[nearest_local])
            if distance >= cutoff:
                continue
            parent_row = int(parent_rows[nearest_local])
            edge_records.append(
                (
                    link_id,
                    parent_row + 1,
                    int(child_row) + 1,
                    frame - 1,
                    frame,
                    distance,
                    distance,
                    np.nan,
                    0,
                )
            )
            link_id += 1

    links = np.asarray(edge_records, dtype=link_dtype())
    adjacency = _csr_from_links(n, links, np=np)
    assignments = _derive_assignments(observations, adjacency, links, np=np)
    schema = {
        "schema": "celltraj2.track_graph.v1",
        "object_set": object_name,
        "track_set": track_name,
        "method": "minimum_centroid_distance",
        "max_distance": cutoff,
        "distance_unit": distance_unit,
        "coordinate_order": ["z", "y", "x"],
        "coordinate_scale": scale.tolist(),
        "registration_dependency": registration_dependency,
        "frame_linkage": "immediately_previous_local_frame_only",
        "parent_invariant": "at_most_one_parent_per_child",
        "child_cardinality": "zero_or_more_children_per_parent",
        "adjacency": {
            "format": "csr",
            "orientation": "row_parent_column_child",
            "index_base": 0,
            "data": "one_based_link_id",
            "shape": [n, n],
        },
        "assignments": {
            "row_alignment": f"/object_sets/{object_name}/observations",
            "lineage_id": "rooted_weak_component_id",
            "tracklet_id": "maximal_non_branching_segment_id",
        },
        "link_count": int(links.shape[0]),
        "observation_count": n,
    }
    graph = TrackGraph(adjacency=adjacency, links=links, assignments=assignments, schema=schema)
    frame_counts: dict[int, dict[str, int]] = {}
    for frame in sorted(int(value) for value in np.unique(frames)):
        object_count = int(np.sum(frames == frame))
        linked_count = int(np.sum(links["target_frame"] == frame)) if links.size else 0
        frame_counts[frame] = {
            "object_count": object_count,
            "linked_count": linked_count,
            "unlinked_count": object_count - linked_count,
        }
    track_path = None
    run_name = validate_name(run_id or default_tracking_run_id(), kind="tracking run")
    if save_outputs:
        track_path = trajectory.store.write_track_graph(
            object_name,
            track_name,
            adjacency=adjacency,
            links=links,
            assignments=assignments,
            schema=schema,
            overwrite=overwrite,
        )
        run_record = {
            "schema": "celltraj2.tracking_run.v1",
            "run_id": run_name,
            "status": "completed",
            "started_at": utc_now_iso(),
            "completed_at": utc_now_iso(),
            "h5_path": str(trajectory.path),
            "roi_id": trajectory.metadata.roi_id,
            "dataset_id": trajectory.metadata.dataset_id,
            "object_set": object_name,
            "track_set": track_name,
            "method": "minimum_centroid_distance",
            "max_distance": cutoff,
            "distance_unit": distance_unit,
            "coordinate_scale": scale.tolist(),
            "registration_dependency": registration_dependency,
            "observation_count": n,
            "link_count": int(links.shape[0]),
            "track_path": track_path,
            "overwrite": bool(overwrite),
            "save_outputs": True,
            "metadata": metadata_payload,
        }
        trajectory.store.write_tracking_run(run_name, run_record, overwrite=True)
        for frame, counts in frame_counts.items():
            trajectory.store.write_tracking_frame_result(
                run_name,
                frame,
                {
                    "frame": int(frame),
                    "status": "completed",
                    "object_set": object_name,
                    "track_set": track_name,
                    **counts,
                },
                overwrite=True,
            )
    return TrackingResult(
        object_set=object_name,
        track_set=track_name,
        graph=graph,
        saved=bool(save_outputs),
        track_path=track_path,
        run_id=run_name,
        frame_counts=frame_counts,
    )


def _observation_row(observation_id: int, observation_count: int) -> int:
    value = int(observation_id)
    if value < 1 or value > int(observation_count):
        raise IndexError(f"observation_id {value} is outside 1..{int(observation_count)}")
    return value - 1


def _csr_from_links(observation_count: int, links: Any, *, np: Any) -> SparseAdjacency:
    rows: list[list[tuple[int, int]]] = [[] for _ in range(observation_count)]
    for link in links:
        parent_row = int(link["parent_observation_id"]) - 1
        child_row = int(link["child_observation_id"]) - 1
        rows[parent_row].append((child_row, int(link["link_id"])))
    indptr = np.zeros(observation_count + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[int] = []
    for row, entries in enumerate(rows):
        for child_row, link_id in sorted(entries):
            indices.append(child_row)
            data.append(link_id)
        indptr[row + 1] = len(indices)
    return SparseAdjacency(
        indptr=indptr,
        indices=np.asarray(indices, dtype=np.int64),
        data=np.asarray(data, dtype=np.int64),
        shape=(observation_count, observation_count),
    )


def _derive_assignments(observations: Any, adjacency: SparseAdjacency, links: Any, *, np: Any) -> Any:
    n = int(observations.shape[0])
    parents = np.zeros(n, dtype=np.int64)
    for link in links:
        child_row = int(link["child_observation_id"]) - 1
        if parents[child_row] != 0:
            raise ValueError("Each observation may have at most one parent.")
        parents[child_row] = int(link["parent_observation_id"])
    n_children = np.diff(adjacency.indptr).astype(np.int32)
    lineage_ids = np.zeros(n, dtype=np.int64)
    tracklet_ids = np.zeros(n, dtype=np.int64)
    generations = np.zeros(n, dtype=np.int32)
    depths = np.zeros(n, dtype=np.int32)
    next_lineage = 1
    next_tracklet = 1
    order = np.lexsort((np.asarray(observations["observation_id"]), np.asarray(observations["frame"])))
    for row_value in order:
        row = int(row_value)
        parent_id = int(parents[row])
        if parent_id == 0:
            lineage_ids[row] = next_lineage
            tracklet_ids[row] = next_tracklet
            next_lineage += 1
            next_tracklet += 1
            continue
        parent_row = parent_id - 1
        if int(observations["frame"][parent_row]) + 1 != int(observations["frame"][row]):
            raise ValueError("Track links must connect immediately consecutive local frames.")
        lineage_ids[row] = lineage_ids[parent_row]
        depths[row] = depths[parent_row] + 1
        if n_children[parent_row] == 1:
            tracklet_ids[row] = tracklet_ids[parent_row]
            generations[row] = generations[parent_row]
        else:
            tracklet_ids[row] = next_tracklet
            next_tracklet += 1
            generations[row] = generations[parent_row] + 1
    assignments = np.zeros(n, dtype=assignment_dtype())
    assignments["observation_id"] = np.arange(1, n + 1, dtype=np.int64)
    assignments["parent_observation_id"] = parents
    assignments["lineage_id"] = lineage_ids
    assignments["tracklet_id"] = tracklet_ids
    assignments["generation"] = generations
    assignments["depth"] = depths
    assignments["n_children"] = n_children
    return assignments


__all__ = [
    "SparseAdjacency",
    "TrackGraph",
    "TrackingResult",
    "assignment_dtype",
    "default_tracking_run_id",
    "link_dtype",
    "track_minimum_centroid_distance",
]
