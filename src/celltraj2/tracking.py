"""Sparse lineage graphs and first-pass centroid tracking."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

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
    motion_path: str | None = None
    run_id: str | None = None
    frame_counts: dict[int, dict[str, int]] = field(default_factory=dict)
    motion_result: BoundaryMotionResult | None = None

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
            "motion_path": self.motion_path,
            "run_id": self.run_id,
            "frame_counts": {str(key): dict(value) for key, value in self.frame_counts.items()},
            "saved": bool(self.saved),
        }


@dataclass(frozen=True)
class BoundaryMotionResult:
    """Point-level surface transport calculated along an object track graph."""

    object_set: str
    track_set: str
    boundary_set: str
    motion_set: str
    links: Any
    transport: dict[str, Any]
    schema: dict[str, Any]
    saved: bool
    point_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    motion_path: str | None = None
    frame_counts: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def link_count(self) -> int:
        return int(self.links.shape[0])

    @property
    def transport_edge_count(self) -> int:
        return int(self.transport["mass"].shape[0])

    @property
    def source_summary_count(self) -> int:
        values = self.point_summaries.get("source_summary", {}).get("point_id")
        return int(values.shape[0]) if hasattr(values, "shape") else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_set": self.object_set,
            "track_set": self.track_set,
            "boundary_set": self.boundary_set,
            "motion_set": self.motion_set,
            "link_count": self.link_count,
            "transport_edge_count": self.transport_edge_count,
            "source_summary_count": self.source_summary_count,
            "motion_path": self.motion_path,
            "saved": self.saved,
            "frame_counts": {str(key): dict(value) for key, value in self.frame_counts.items()},
        }


def track_graph_digest(adjacency: SparseAdjacency, links: Any, assignments: Any) -> str:
    """Return a stable digest for surface-motion and feature dependencies."""

    np = _require_numpy()
    digest = hashlib.sha256()
    for values in (
        adjacency.indptr,
        adjacency.indices,
        adjacency.data,
        np.asarray(adjacency.shape, dtype=np.int64),
        links,
        assignments,
    ):
        array = np.asarray(values)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


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
    progress: Callable[[Mapping[str, Any]], None] | None = None,
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
    if progress is not None and np.any(frames == 1):
        progress(
            {
                "frame": 1,
                "object_count": int(np.sum(frames == 1)),
                "linked_count": 0,
                "unlinked_count": int(np.sum(frames == 1)),
                "candidate_count": 0,
                "ot_score_count": 0,
                "full_plan_count": 0,
            }
        )
    for frame in sorted(int(value) for value in np.unique(frames)):
        if frame <= 1:
            continue
        child_rows = np.flatnonzero(frames == frame)
        parent_rows = np.flatnonzero(frames == frame - 1)
        if not child_rows.size or not parent_rows.size:
            if progress is not None:
                progress(
                    {
                        "frame": int(frame),
                        "object_count": int(child_rows.size),
                        "linked_count": 0,
                        "unlinked_count": int(child_rows.size),
                    }
                )
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
        if progress is not None:
            object_count = int(child_rows.size)
            linked_count = sum(1 for record in edge_records if int(record[4]) == frame)
            progress(
                {
                    "frame": int(frame),
                    "object_count": object_count,
                    "linked_count": int(linked_count),
                    "unlinked_count": object_count - int(linked_count),
                }
            )

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
        "track_digest": track_graph_digest(adjacency, links, assignments),
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


def _transport_pair_samples(
    source_points: Mapping[str, Any],
    target_points: Mapping[str, Any],
    *,
    max_boundary_points: int | None,
    mass_mode: str,
    point_measure: float,
    cost_prealign: str,
    np: Any,
) -> dict[str, Any]:
    from celltraj2.boundaries import common_density_point_samples

    source_sample, target_sample = common_density_point_samples(
        source_points["registered"],
        target_points["registered"],
        max_boundary_points,
        mass_mode=mass_mode,  # type: ignore[arg-type]
        point_measure=point_measure,
    )
    source_cost = np.asarray(source_points["registered"][source_sample.rows], dtype=float)
    target_cost = np.asarray(target_points["registered"][target_sample.rows], dtype=float)
    if cost_prealign == "centroid":
        source_cost = source_cost - np.average(
            source_cost, axis=0, weights=source_sample.weights
        )
        target_cost = target_cost - np.average(
            target_cost, axis=0, weights=target_sample.weights
        )
    elif cost_prealign != "none":
        raise ValueError("cost_prealign must be 'none' or 'centroid'")
    return {
        "source": source_sample,
        "target": target_sample,
        "source_cost": source_cost,
        "target_cost": target_cost,
        "voxel_spacing": max(source_sample.voxel_spacing, target_sample.voxel_spacing),
    }


def _centroid_knn_candidates(
    child_rows: Any,
    parent_rows: Any,
    centroids: Any,
    *,
    k: int,
    max_distance: float,
    np: Any,
) -> dict[int, list[tuple[int, float]]]:
    """Return at most ``k`` nearest registered-centroid parents per child."""

    child_values = np.asarray(child_rows, dtype=np.int64)
    parent_values = np.asarray(parent_rows, dtype=np.int64)
    result = {int(row): [] for row in child_values}
    if not child_values.size or not parent_values.size:
        return result
    parent_positions = np.asarray(centroids[parent_values], dtype=float)
    child_positions = np.asarray(centroids[child_values], dtype=float)
    valid_parent = np.all(np.isfinite(parent_positions), axis=1)
    valid_child = np.all(np.isfinite(child_positions), axis=1)
    valid_parent_rows = parent_values[valid_parent]
    valid_parent_positions = parent_positions[valid_parent]
    valid_child_rows = child_values[valid_child]
    valid_child_positions = child_positions[valid_child]
    if not valid_parent_rows.size or not valid_child_rows.size:
        return result
    neighbor_count = min(int(k), int(valid_parent_rows.size))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(valid_parent_positions)
        try:
            distances, local_rows = tree.query(
                valid_child_positions,
                k=neighbor_count,
                distance_upper_bound=float(max_distance),
                workers=1,
            )
        except TypeError:
            distances, local_rows = tree.query(
                valid_child_positions,
                k=neighbor_count,
                distance_upper_bound=float(max_distance),
            )
        distances = np.asarray(distances)
        local_rows = np.asarray(local_rows)
        if neighbor_count == 1:
            distances = distances[:, None]
            local_rows = local_rows[:, None]
        for child_row, child_distances, child_local_rows in zip(
            valid_child_rows, distances, local_rows, strict=False
        ):
            candidates = [
                (int(valid_parent_rows[int(local_row)]), float(distance))
                for distance, local_row in zip(child_distances, child_local_rows, strict=False)
                if int(local_row) < int(valid_parent_rows.size)
                and np.isfinite(distance)
                and float(distance) < float(max_distance)
            ]
            candidates.sort(key=lambda value: (value[1], value[0]))
            result[int(child_row)] = candidates
        return result
    except ImportError:
        pass

    for child_row, child_position in zip(valid_child_rows, valid_child_positions, strict=False):
        delta = valid_parent_positions - child_position
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        eligible = np.flatnonzero(np.isfinite(distances) & (distances < float(max_distance)))
        order = sorted(
            (int(local_row) for local_row in eligible),
            key=lambda local_row: (float(distances[local_row]), int(valid_parent_rows[local_row])),
        )[:neighbor_count]
        result[int(child_row)] = [
            (int(valid_parent_rows[local_row]), float(distances[local_row]))
            for local_row in order
        ]
    return result


def _transport_point_summary(
    plan: Any,
    *,
    point_ids: Any,
    displacement: Any,
    direction: str,
    np: Any,
) -> dict[str, Any]:
    """Return one confidence-weighted barycentric vector per sampled point."""

    if direction == "source":
        edge_rows = np.asarray(plan.source_rows, dtype=np.int64)
        point_weights = np.asarray(plan.source_weights, dtype=float)
        matched_mass = np.asarray(plan.source_matched_mass, dtype=float)
    elif direction == "target":
        edge_rows = np.asarray(plan.target_rows, dtype=np.int64)
        point_weights = np.asarray(plan.target_weights, dtype=float)
        matched_mass = np.asarray(plan.target_matched_mass, dtype=float)
    else:
        raise ValueError("direction must be source or target")
    ids = np.asarray(point_ids, dtype=np.int64)
    edge_mass = np.asarray(plan.mass, dtype=float)
    weighted = np.zeros((ids.size, 3), dtype=float)
    retained_mass = np.zeros(ids.size, dtype=float)
    if edge_rows.size:
        np.add.at(weighted, edge_rows, edge_mass[:, None] * displacement)
        np.add.at(retained_mass, edge_rows, edge_mass)
    vectors = np.full((ids.size, 3), np.nan, dtype=float)
    mapped = retained_mass > 0
    vectors[mapped] = weighted[mapped] / retained_mass[mapped, None]
    variance_sum = np.zeros(ids.size, dtype=float)
    distance_sum = np.zeros(ids.size, dtype=float)
    if edge_rows.size:
        residual = displacement - vectors[edge_rows]
        np.add.at(variance_sum, edge_rows, edge_mass * np.sum(residual * residual, axis=1))
        np.add.at(distance_sum, edge_rows, edge_mass * np.asarray(plan.edge_cost, dtype=float))
    variance = np.full(ids.size, np.nan, dtype=float)
    mean_distance = np.full(ids.size, np.nan, dtype=float)
    variance[mapped] = variance_sum[mapped] / retained_mass[mapped]
    mean_distance[mapped] = distance_sum[mapped] / retained_mass[mapped]
    matched_fraction = np.divide(
        matched_mass,
        point_weights,
        out=np.zeros_like(matched_mass),
        where=point_weights > 0,
    )
    matched_fraction = np.clip(matched_fraction, 0.0, 1.0)
    return {
        "point_id": ids,
        "point_mass": point_weights.astype(np.float64),
        "matched_mass": matched_mass.astype(np.float64),
        "matched_fraction": matched_fraction.astype(np.float32),
        "barycentric_displacement_zyx": vectors.astype(np.float32),
        "displacement_variance": variance.astype(np.float32),
        "edge_distance_mean": mean_distance.astype(np.float32),
        "quality_flags": np.zeros(ids.size, dtype=np.uint32),
    }


def _append_columns(destination: dict[str, list[Any]], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        destination.setdefault(str(key), []).append(value)


def _concatenate_column_blocks(columns: Mapping[str, list[Any]], *, np: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, blocks in columns.items():
        if blocks:
            output[key] = np.concatenate(blocks, axis=0)
        elif key in {"registered_displacement_zyx", "barycentric_displacement_zyx"}:
            output[key] = np.empty((0, 3), dtype=np.float32)
        elif key.endswith("point_id") or key == "quality_flags":
            output[key] = np.empty(0, dtype=np.int64 if key.endswith("point_id") else np.uint32)
        else:
            output[key] = np.empty(0, dtype=np.float64)
    return output


def compute_boundary_motion(
    trajectory: Any,
    object_set: str,
    track_set: str,
    *,
    boundary_set: str,
    motion_set: str = "surface_ot",
    boundary_source_id: int | None = None,
    boundary_source_name: str | None = None,
    boundary_source_role: str | None = None,
    registration_set: str | None = None,
    ot_method: str = "unbalanced",
    sinkhorn_regularization: float = 0.05,
    unbalanced_reach: float = 2.0,
    partial_mass: float = 0.9,
    max_transport_distance: float | None = 5.0,
    min_source_coverage: float = 0.5,
    min_target_coverage: float = 0.5,
    max_boundary_points: int | None = 512,
    mass_tolerance: float = 1e-12,
    relative_mass_tolerance: float = 1e-3,
    retained_mass_fraction: float = 0.999,
    mass_mode: str = "probability",
    cost_prealign: str = "none",
    overwrite: bool = False,
    save_outputs: bool = True,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> BoundaryMotionResult:
    """Map boundary points across every accepted link in a stored object graph.

    Object tracking owns link selection. This function only calculates the
    point-level transport attached to those links, making surface motion usable
    with centroid, imported, or boundary-OT track graphs.
    """

    from celltraj2.boundaries import (
        BoundaryLibraryView,
        boundary_motion_link_dtype,
        optimal_transport_plan,
        resolve_boundary_source_ids,
    )

    np = _require_numpy()
    object_name = validate_name(object_set, kind="object set")
    track_name = validate_name(track_set, kind="track set")
    boundary_name = validate_name(boundary_set, kind="boundary set")
    motion_name = validate_name(motion_set, kind="boundary motion set")
    if ot_method not in {"emd", "sinkhorn", "unbalanced", "partial"}:
        raise ValueError("ot_method must be emd, sinkhorn, unbalanced, or partial")
    for name, value in (
        ("min_source_coverage", min_source_coverage),
        ("min_target_coverage", min_target_coverage),
    ):
        if not np.isfinite(float(value)) or float(value) < 0 or float(value) > 1:
            raise ValueError(f"{name} must be in [0, 1]")

    graph = trajectory.store.read_track_graph(object_name, track_name)
    boundary = BoundaryLibraryView(trajectory.store, boundary_name)
    selected_source_ids = resolve_boundary_source_ids(
        boundary,
        source_ids=None if boundary_source_id is None else (boundary_source_id,),
        source_names=None if boundary_source_name in (None, "") else (str(boundary_source_name),),
        source_roles=None if boundary_source_role in (None, "") else (str(boundary_source_role),),
        require_one=True,
    )
    selected_source_id = int(next(iter(selected_source_ids or ())))
    source_record = next(
        value for value in boundary.sources if int(value["source_id"]) == selected_source_id
    )
    if str(source_record.get("kind") or "") != "object_set":
        raise ValueError("Boundary motion requires an object_set boundary source")
    if str(source_record.get("object_set") or "") != object_name:
        raise ValueError(
            f"Boundary source {source_record.get('name')!r} belongs to object set "
            f"{source_record.get('object_set')!r}, not {object_name!r}"
        )

    entity_by_observation: dict[int, int] = {}
    for entity in boundary.entities:
        if int(entity["source_id"]) != selected_source_id:
            continue
        observation_id = int(entity["observation_id"])
        if observation_id > 0:
            if observation_id in entity_by_observation:
                raise ValueError(
                    f"Boundary source {source_record.get('name')!r} has duplicate "
                    f"observation_id={observation_id}"
                )
            entity_by_observation[observation_id] = int(entity["boundary_entity_id"])

    scale = np.asarray(boundary.schema.get("coordinate_scale_zyx", (1.0, 1.0, 1.0)), dtype=float)
    selected_registration = registration_set or trajectory.store.active_registration_name()
    registration = None
    if selected_registration:
        registration = trajectory.store.read_registration_set(selected_registration)
        registered_scale = np.asarray(registration.schema.get("coordinate_scale_zyx", scale), dtype=float)
        if (
            str(registration.schema.get("method")) != "identity"
            and (
                registered_scale.shape != (3,)
                or not np.allclose(registered_scale, scale, rtol=1e-7, atol=1e-10)
            )
        ):
            raise ValueError(
                f"Registration set {registration.name!r} uses coordinate_scale_zyx "
                f"{registered_scale.tolist()}, not boundary scale {scale.tolist()}"
            )
    registration_dependency = (
        None
        if registration is None
        else {
            "registration_set": registration.name,
            "registration_digest": registration.digest,
            "registration_method": str(registration.schema.get("method") or ""),
        }
    )
    track_registration_dependency = graph.schema.get("registration_dependency")
    track_registration_digest = str(
        (track_registration_dependency or {}).get("registration_digest") or ""
    )
    motion_registration_digest = str(
        (registration_dependency or {}).get("registration_digest") or ""
    )
    if track_registration_digest != motion_registration_digest:
        raise ValueError(
            "Surface motion registration does not match the selected object track graph: "
            f"track digest={track_registration_digest!r}, motion digest={motion_registration_digest!r}"
        )
    boundary_dependency = {
        "boundary_set": boundary_name,
        "boundary_digest": str(boundary.schema.get("boundary_digest") or ""),
        "source_id": selected_source_id,
        "source_name": str(source_record.get("name") or ""),
        "coordinate_system": "native_roi_physical",
    }
    track_dependency = {
        "object_set": object_name,
        "track_set": track_name,
        "track_digest": str(
            graph.schema.get("track_digest")
            or track_graph_digest(graph.adjacency, graph.links, graph.assignments)
        ),
    }

    point_cache: dict[int, dict[str, Any]] = {}

    def registered_entity_points(entity_id: int) -> dict[str, Any]:
        if entity_id in point_cache:
            return point_cache[entity_id]
        entity = boundary.entity(entity_id)
        point_data = boundary.read_points(entity_id, fields=("point_id", "native_position_zyx"))
        native = np.asarray(point_data["native_position_zyx"], dtype=float)
        registered = native
        if registration is not None:
            registered = registration.apply_zyx(
                native,
                np.full(native.shape[0], int(entity["frame"]), dtype=np.int64),
            )
        result = {
            "point_id": np.asarray(point_data["point_id"], dtype=np.int64),
            "native": native,
            "registered": registered,
        }
        point_cache[entity_id] = result
        return result

    motion_links: list[tuple[Any, ...]] = []
    transport_columns: dict[str, list[Any]] = {
        "source_point_id": [],
        "target_point_id": [],
        "mass": [],
        "edge_cost": [],
        "registered_displacement_zyx": [],
    }
    source_summary_columns: dict[str, list[Any]] = {}
    target_summary_columns: dict[str, list[Any]] = {}
    frame_counts: dict[int, dict[str, Any]] = {}
    transport_start = 0
    source_summary_start = 0
    target_summary_start = 0
    selected_methods: set[str] = set()
    sampling = dict(boundary.schema.get("sampling") or {})
    canonical_spacing = float(sampling.get("effective_point_spacing") or sampling.get("point_spacing") or 1.0)
    spatial_ndim = int(boundary.schema.get("spatial_ndim") or 3)
    point_measure = canonical_spacing ** max(1, spatial_ndim - 1)
    for motion_link_id, link in enumerate(graph.links, start=1):
        parent_observation_id = int(link["parent_observation_id"])
        child_observation_id = int(link["child_observation_id"])
        if parent_observation_id not in entity_by_observation or child_observation_id not in entity_by_observation:
            raise ValueError(
                f"Boundary source {source_record.get('name')!r} lacks an entity for track link "
                f"{parent_observation_id}->{child_observation_id}"
            )
        parent_entity_id = entity_by_observation[parent_observation_id]
        child_entity_id = entity_by_observation[child_observation_id]
        parent_points = registered_entity_points(parent_entity_id)
        child_points = registered_entity_points(child_entity_id)
        if not parent_points["native"].shape[0] or not child_points["native"].shape[0]:
            raise ValueError(
                f"Track link {parent_observation_id}->{child_observation_id} has an empty boundary"
            )
        pair = _transport_pair_samples(
            parent_points,
            child_points,
            max_boundary_points=max_boundary_points,
            mass_mode=mass_mode,
            point_measure=point_measure,
            cost_prealign=cost_prealign,
            np=np,
        )
        plan = optimal_transport_plan(
            pair["source_cost"],
            pair["target_cost"],
            method=ot_method,  # type: ignore[arg-type]
            regularization=sinkhorn_regularization,
            unbalanced_reach=unbalanced_reach,
            partial_mass=partial_mass,
            max_transport_distance=(
                max_transport_distance if ot_method in {"unbalanced", "partial"} else None
            ),
            source_weights=pair["source"].weights,
            target_weights=pair["target"].weights,
            mass_tolerance=mass_tolerance,
            relative_mass_tolerance=relative_mass_tolerance,
            retained_mass_fraction=retained_mass_fraction,
        )
        selected_methods.add(str(plan.method))
        source_local = pair["source"].rows[plan.source_rows]
        target_local = pair["target"].rows[plan.target_rows]
        source_point_ids = parent_points["point_id"][source_local]
        target_point_ids = child_points["point_id"][target_local]
        displacement = child_points["registered"][target_local] - parent_points["registered"][source_local]
        source_summary = _transport_point_summary(
            plan,
            point_ids=parent_points["point_id"][pair["source"].rows],
            displacement=displacement,
            direction="source",
            np=np,
        )
        target_summary = _transport_point_summary(
            plan,
            point_ids=child_points["point_id"][pair["target"].rows],
            displacement=displacement,
            direction="target",
            np=np,
        )
        transport_count = int(plan.mass.shape[0])
        source_summary_count = int(pair["source"].rows.size)
        target_summary_count = int(pair["target"].rows.size)
        source_frame = int(link["source_frame"])
        target_frame = int(link["target_frame"])
        motion_links.append(
            (
                motion_link_id,
                int(link["link_id"]),
                parent_entity_id,
                child_entity_id,
                parent_observation_id,
                child_observation_id,
                source_frame,
                target_frame,
                transport_start,
                transport_count,
                source_summary_start,
                source_summary_count,
                target_summary_start,
                target_summary_count,
                float(plan.total_cost),
                float(plan.transport_cost),
                float(plan.objective),
                float(plan.matched_mean_cost),
                float(plan.transported_mass),
                float(plan.source_coverage),
                float(plan.target_coverage),
                float(plan.dropped_mass),
                float(plan.edge_distance_p50),
                float(plan.edge_distance_p90),
                float(plan.edge_distance_p99),
                float(plan.edge_distance_max),
                int(plan.solver_iterations),
                int(bool(plan.solver_converged)),
                (1 if float(plan.source_coverage) < float(min_source_coverage) else 0)
                | (2 if float(plan.target_coverage) < float(min_target_coverage) else 0)
                | (4 if not plan.solver_converged else 0),
            )
        )
        transport_columns["source_point_id"].append(source_point_ids.astype(np.int64))
        transport_columns["target_point_id"].append(target_point_ids.astype(np.int64))
        transport_columns["mass"].append(np.asarray(plan.mass, dtype=np.float64))
        transport_columns["edge_cost"].append(np.asarray(plan.edge_cost, dtype=np.float32))
        transport_columns["registered_displacement_zyx"].append(np.asarray(displacement, dtype=np.float32))
        _append_columns(source_summary_columns, source_summary)
        _append_columns(target_summary_columns, target_summary)
        transport_start += transport_count
        source_summary_start += source_summary_count
        target_summary_start += target_summary_count
        frame_summary = frame_counts.setdefault(
            target_frame,
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "link_count": 0,
                "transport_edge_count": 0,
                "transported_mass": 0.0,
                "ot_cost_sum": 0.0,
                "source_coverage_sum": 0.0,
                "target_coverage_sum": 0.0,
                "dropped_mass": 0.0,
                "nonconverged_link_count": 0,
            },
        )
        frame_summary["link_count"] += 1
        frame_summary["transport_edge_count"] += transport_count
        frame_summary["transported_mass"] += float(plan.transported_mass)
        frame_summary["ot_cost_sum"] += float(plan.total_cost)
        frame_summary["source_coverage_sum"] += float(plan.source_coverage)
        frame_summary["target_coverage_sum"] += float(plan.target_coverage)
        frame_summary["dropped_mass"] += float(plan.dropped_mass)
        frame_summary["nonconverged_link_count"] += int(not plan.solver_converged)
        if progress is not None:
            progress(
                {
                    "event": "surface_motion_link_summary",
                    "motion_link_id": int(motion_link_id),
                    "track_link_id": int(link["link_id"]),
                    "source_observation_id": parent_observation_id,
                    "target_observation_id": child_observation_id,
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "source_point_count": int(parent_points["native"].shape[0]),
                    "target_point_count": int(child_points["native"].shape[0]),
                    "transport_edge_count": transport_count,
                    "raw_transport_edge_count": int(plan.raw_edge_count),
                    "transported_mass": float(plan.transported_mass),
                    "source_coverage": float(plan.source_coverage),
                    "target_coverage": float(plan.target_coverage),
                    "dropped_mass": float(plan.dropped_mass),
                    "edge_distance_p99": float(plan.edge_distance_p99),
                    "edge_distance_max": float(plan.edge_distance_max),
                    "solver_iterations": int(plan.solver_iterations),
                    "solver_converged": bool(plan.solver_converged),
                    "ot_cost": float(plan.total_cost),
                }
            )

    transport = _concatenate_column_blocks(transport_columns, np=np)
    point_summaries = {
        "source_summary": _concatenate_column_blocks(source_summary_columns, np=np),
        "target_summary": _concatenate_column_blocks(target_summary_columns, np=np),
    }
    for counts in frame_counts.values():
        counts["mean_ot_cost"] = (
            counts.pop("ot_cost_sum") / counts["link_count"] if counts["link_count"] else None
        )
        counts["mean_source_coverage"] = (
            counts.pop("source_coverage_sum") / counts["link_count"] if counts["link_count"] else None
        )
        counts["mean_target_coverage"] = (
            counts.pop("target_coverage_sum") / counts["link_count"] if counts["link_count"] else None
        )
    schema = {
        "schema": "celltraj2.boundary_motion.v2",
        "boundary_set": boundary_name,
        "motion_set": motion_name,
        "created_at": utc_now_iso(),
        "source": f"/object_sets/{object_name}/tracks/{track_name}",
        "boundary_dependency": boundary_dependency,
        "track_dependency": track_dependency,
        "registration_dependency": registration_dependency,
        "point_identity_coordinate_system": "native_boundary_point_id",
        "displacement_coordinate_system": "registered_roi_physical",
        "displacement_definition": "T_target(q_native)-T_source(p_native)",
        "ot_method_requested": ot_method,
        "transport_methods": sorted(selected_methods),
        "solver": {
            "max_iterations": 10000,
            "scaling_relative_tolerance": 1e-8,
            "log_absolute_tolerance": 1e-10,
            "underflow_fallback": "log_domain",
        },
        "sinkhorn_regularization": float(sinkhorn_regularization),
        "unbalanced_reach": float(unbalanced_reach),
        "partial_mass": float(partial_mass),
        "max_transport_distance": (
            None if max_transport_distance is None or ot_method not in {"unbalanced", "partial"}
            else float(max_transport_distance)
        ),
        "minimum_coverage": {
            "source": float(min_source_coverage),
            "target": float(min_target_coverage),
        },
        "mass_tolerance": float(mass_tolerance),
        "relative_mass_tolerance": float(relative_mass_tolerance),
        "retained_mass_fraction": float(retained_mass_fraction),
        "mass_mode": str(mass_mode),
        "cost_prealign": str(cost_prealign),
        "max_boundary_points": max_boundary_points,
        "sampling": {
            "method": "shared_physical_voxel_grid",
            "canonical_point_spacing": canonical_spacing,
            "point_measure": point_measure,
        },
        "point_summary": {
            "source_group": "source_summary",
            "target_group": "target_summary",
            "vector": "mass_weighted_barycentric_displacement",
            "confidence": "matched_fraction",
        },
        "link_quality_flags": {
            "1": "source_coverage_below_threshold",
            "2": "target_coverage_below_threshold",
            "4": "solver_not_converged",
        },
        "motion_link_count": len(motion_links),
        "transport_edge_count": int(transport_start),
        "metadata": dict(metadata or {}),
    }
    links = np.asarray(motion_links, dtype=boundary_motion_link_dtype())
    motion_path = None
    if save_outputs:
        motion_path = trajectory.store.write_boundary_motion(
            boundary_name,
            motion_name,
            links=links,
            transport=transport,
            point_summaries=point_summaries,
            schema=schema,
            overwrite=overwrite,
        )
    return BoundaryMotionResult(
        object_set=object_name,
        track_set=track_name,
        boundary_set=boundary_name,
        motion_set=motion_name,
        links=links,
        transport=transport,
        point_summaries=point_summaries,
        schema=schema,
        saved=bool(save_outputs),
        motion_path=motion_path,
        frame_counts=frame_counts,
    )


def track_minimum_boundary_ot_cost(
    trajectory: Any,
    object_set: str,
    *,
    boundary_set: str | None = None,
    boundary_source_id: int | None = None,
    boundary_source_name: str | None = None,
    boundary_source_role: str | None = None,
    max_distance: float,
    candidate_k: int = 5,
    ot_cost_cutoff: float = float("inf"),
    track_set: str = "boundary_ot",
    motion_set: str | None = None,
    registration_set: str | None = None,
    ot_method: str | None = None,
    score_ot_method: str | None = None,
    winner_ot_method: str | None = None,
    sinkhorn_regularization: float = 0.05,
    unbalanced_reach: float = 2.0,
    partial_mass: float = 0.9,
    max_transport_distance: float | None = 5.0,
    min_source_coverage: float = 0.5,
    min_target_coverage: float = 0.5,
    max_boundary_points: int | None = 512,
    mass_tolerance: float = 1e-12,
    relative_mass_tolerance: float = 1e-3,
    retained_mass_fraction: float = 0.999,
    mass_mode: str = "probability",
    cost_prealign: str = "none",
    save_motion: bool = True,
    overwrite: bool = False,
    save_outputs: bool = True,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> TrackingResult:
    """Track observations by registered boundary OT cost after centroid gating.

    Candidate parents are restricted to the ``candidate_k`` nearest registered
    centroids inside the physical-coordinate radius in the immediately
    preceding local frame. Point identities and stored boundary coordinates
    remain native; only the cost and displacement calculations use the selected
    registration.
    """

    from celltraj2.boundaries import (
        _materialize_transport_plan,
        _optimal_transport_score,
        as_boundary_library_view,
        boundary_motion_link_dtype,
        build_boundary_library,
        resolve_boundary_source_ids,
    )

    np = _require_numpy()
    object_name = validate_name(object_set, kind="object set")
    track_name = validate_name(track_set, kind="track set")
    boundary_name = validate_name(
        boundary_set or f"transient_{object_name}", kind="boundary set"
    )
    motion_name = validate_name(motion_set or track_name, kind="boundary motion set")
    cutoff = float(max_distance)
    centroid_candidate_k = int(candidate_k)
    ot_cutoff = float(ot_cost_cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("max_distance must be finite and > 0")
    if centroid_candidate_k < 1:
        raise ValueError("candidate_k must be >= 1")
    if np.isnan(ot_cutoff) or ot_cutoff <= 0:
        raise ValueError("ot_cost_cutoff must be > 0 and may be infinity")
    candidate_ot_method = str(score_ot_method or ot_method or "emd").lower()
    accepted_winner_ot_method = str(winner_ot_method or ot_method or "unbalanced").lower()
    valid_ot_methods = {"emd", "sinkhorn", "unbalanced", "partial"}
    if candidate_ot_method not in valid_ot_methods:
        raise ValueError(
            "score_ot_method must be emd, sinkhorn, unbalanced, or partial"
        )
    if accepted_winner_ot_method not in valid_ot_methods:
        raise ValueError(
            "winner_ot_method must be emd, sinkhorn, unbalanced, or partial"
        )
    for name, value in (
        ("min_source_coverage", min_source_coverage),
        ("min_target_coverage", min_target_coverage),
    ):
        if not np.isfinite(float(value)) or float(value) < 0 or float(value) > 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if boundary_set is None and save_motion and save_outputs:
        raise ValueError(
            "Persistent boundary motion requires a stored boundary_set; "
            "disable save_motion or build the library first"
        )

    observations = trajectory.store.read_observations(object_name)
    observation_count = int(observations.shape[0])
    expected_ids = np.arange(1, observation_count + 1, dtype=np.int64)
    if observation_count and not np.array_equal(
        np.asarray(observations["observation_id"], dtype=np.int64), expected_ids
    ):
        raise ValueError("Observation rows must align to one-based observation_id values")
    transient_library = None
    if boundary_set is None:
        transient_library = build_boundary_library(
            trajectory,
            boundary_name,
            object_set=object_name,
            save_outputs=False,
            metadata={"purpose": "transient_boundary_ot_tracking"},
        )
    boundary = as_boundary_library_view(trajectory, boundary_name, transient_library)
    boundary_schema = boundary.schema
    selected_source_ids = resolve_boundary_source_ids(
        boundary,
        source_ids=None if boundary_source_id is None else (boundary_source_id,),
        source_names=None if boundary_source_name in (None, "") else (str(boundary_source_name),),
        source_roles=None if boundary_source_role in (None, "") else (str(boundary_source_role),),
        require_one=True,
    )
    selected_source_id = int(next(iter(selected_source_ids or ())))
    source_record = next(
        value for value in boundary.sources if int(value["source_id"]) == selected_source_id
    )
    if str(source_record.get("kind") or "") != "object_set":
        raise ValueError("Boundary OT tracking requires an object_set boundary source")
    if str(source_record.get("object_set") or "") != object_name:
        raise ValueError(
            f"Boundary source {source_record.get('name')!r} belongs to object set "
            f"{source_record.get('object_set')!r}, not {object_name!r}"
        )
    scale = np.asarray(boundary_schema.get("coordinate_scale_zyx", (1.0, 1.0, 1.0)), dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"Boundary set {boundary_name!r} has invalid coordinate_scale_zyx")
    distance_unit = str(boundary_schema.get("distance_unit") or "scaled_coordinate_unit")
    entity_by_observation: dict[int, int] = {}
    source_entities_by_frame: dict[int, list[Any]] = {}
    for entity in boundary.entities:
        if int(entity["source_id"]) != selected_source_id:
            continue
        source_entities_by_frame.setdefault(int(entity["frame"]), []).append(entity)
        observation_id = int(entity["observation_id"])
        if observation_id <= 0:
            continue
        if observation_id in entity_by_observation:
            raise ValueError(
                f"Boundary source {source_record.get('name')!r} has multiple entities for "
                f"observation_id={observation_id}"
            )
        entity_by_observation[observation_id] = int(entity["boundary_entity_id"])
    missing = [int(value) for value in expected_ids if int(value) not in entity_by_observation]
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(
            f"Boundary set {boundary_name!r} is missing entities for observation ids {preview}"
        )

    frames = np.asarray(observations["frame"], dtype=np.int64)
    centroids = np.column_stack(
        [observations["centroid_z"], observations["centroid_y"], observations["centroid_x"]]
    ).astype(float)
    centroids *= scale[None, :]
    selected_registration = registration_set
    if selected_registration is None:
        selected_registration = trajectory.store.active_registration_name()
    registration = None
    if selected_registration:
        registration = trajectory.store.read_registration_set(selected_registration)
        registered_scale = np.asarray(registration.schema.get("coordinate_scale_zyx", scale), dtype=float)
        if (
            str(registration.schema.get("method")) != "identity"
            and (
                registered_scale.shape != (3,)
                or not np.allclose(registered_scale, scale, rtol=1e-7, atol=1e-10)
            )
        ):
            raise ValueError(
                f"Registration set {registration.name!r} uses coordinate_scale_zyx "
                f"{registered_scale.tolist()}, not boundary scale {scale.tolist()}"
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
    boundary_dependency = {
        "boundary_set": boundary_name,
        "boundary_digest": str(boundary_schema.get("boundary_digest") or ""),
        "coordinate_system": "native_roi_physical",
        "stored": boundary_set is not None,
        "source_id": selected_source_id,
        "source_name": str(source_record.get("name") or ""),
    }

    frame_point_cache: dict[int, dict[str, Any]] = {}

    def registered_frame_points(frame: int) -> dict[str, Any]:
        frame_value = int(frame)
        if frame_value in frame_point_cache:
            return frame_point_cache[frame_value]
        entities = sorted(
            source_entities_by_frame.get(frame_value, ()),
            key=lambda value: (int(value["point_start"]), int(value["boundary_entity_id"])),
        )
        entity_slices: dict[int, slice] = {}
        nonempty = [entity for entity in entities if int(entity["point_count"]) > 0]
        if not nonempty:
            result = {
                "point_id": np.empty(0, dtype=np.int64),
                "native": np.empty((0, 3), dtype=float),
                "registered": np.empty((0, 3), dtype=float),
                "entity_slices": {
                    int(entity["boundary_entity_id"]): slice(0, 0) for entity in entities
                },
            }
            frame_point_cache[frame_value] = result
            return result

        block_start = min(int(entity["point_start"]) for entity in nonempty)
        block_stop = max(
            int(entity["point_start"]) + int(entity["point_count"])
            for entity in nonempty
        )
        expected_count = sum(int(entity["point_count"]) for entity in entities)
        if block_stop - block_start == expected_count:
            point_data = boundary.read_points(
                rows=slice(block_start, block_stop),
                fields=("point_id", "native_position_zyx"),
            )
            point_ids = np.asarray(point_data["point_id"], dtype=np.int64)
            native = np.asarray(point_data["native_position_zyx"], dtype=float)
            for entity in entities:
                local_start = int(entity["point_start"]) - block_start
                entity_slices[int(entity["boundary_entity_id"])] = slice(
                    local_start,
                    local_start + int(entity["point_count"]),
                )
        else:
            point_id_blocks: list[Any] = []
            native_blocks: list[Any] = []
            local_start = 0
            for entity in entities:
                entity_id = int(entity["boundary_entity_id"])
                count = int(entity["point_count"])
                entity_slices[entity_id] = slice(local_start, local_start + count)
                if count:
                    point_data = boundary.read_points(
                        entity_id,
                        fields=("point_id", "native_position_zyx"),
                    )
                    point_id_blocks.append(np.asarray(point_data["point_id"], dtype=np.int64))
                    native_blocks.append(np.asarray(point_data["native_position_zyx"], dtype=float))
                    local_start += count
            point_ids = np.concatenate(point_id_blocks) if point_id_blocks else np.empty(0, dtype=np.int64)
            native = np.concatenate(native_blocks, axis=0) if native_blocks else np.empty((0, 3), dtype=float)
        registered = native
        if registration is not None and native.shape[0]:
            registered = registration.apply_zyx(
                native,
                np.full(native.shape[0], frame_value, dtype=np.int64),
            )
        result = {
            "point_id": point_ids,
            "native": native,
            "registered": registered,
            "entity_slices": entity_slices,
        }
        frame_point_cache[frame_value] = result
        return result

    def registered_entity_points(entity_id: int, frame: int) -> dict[str, Any]:
        frame_points = registered_frame_points(frame)
        rows = frame_points["entity_slices"].get(int(entity_id), slice(0, 0))
        return {
            "point_id": frame_points["point_id"][rows],
            "native": frame_points["native"][rows],
            "registered": frame_points["registered"][rows],
        }

    edge_records: list[tuple[Any, ...]] = []
    accepted_plans: list[dict[str, Any]] = []
    sampling = dict(boundary_schema.get("sampling") or {})
    canonical_spacing = float(sampling.get("effective_point_spacing") or sampling.get("point_spacing") or 1.0)
    spatial_ndim = int(boundary_schema.get("spatial_ndim") or 3)
    point_measure = canonical_spacing ** max(1, spatial_ndim - 1)
    link_id = 1
    if progress is not None and np.any(frames == 1):
        progress(
            {
                "frame": 1,
                "object_count": int(np.sum(frames == 1)),
                "linked_count": 0,
                "unlinked_count": int(np.sum(frames == 1)),
            }
        )
    for frame in sorted(int(value) for value in np.unique(frames)):
        if frame <= 1:
            continue
        child_rows = np.flatnonzero(frames == frame)
        parent_rows = np.flatnonzero(frames == frame - 1)
        if not child_rows.size or not parent_rows.size:
            if progress is not None:
                progress(
                    {
                        "frame": int(frame),
                        "object_count": int(child_rows.size),
                        "linked_count": 0,
                        "unlinked_count": int(child_rows.size),
                        "candidate_count": 0,
                        "ot_score_count": 0,
                        "full_plan_count": 0,
                    }
                )
            continue
        candidates_by_child = _centroid_knn_candidates(
            child_rows,
            parent_rows,
            centroids,
            k=centroid_candidate_k,
            max_distance=cutoff,
            np=np,
        )
        frame_candidate_count = sum(len(values) for values in candidates_by_child.values())
        frame_ot_score_count = 0
        frame_full_plan_count = 0
        if frame_candidate_count:
            registered_frame_points(frame - 1)
            registered_frame_points(frame)
        for child_row_value in child_rows:
            child_row = int(child_row_value)
            candidates = candidates_by_child.get(child_row, ())
            if not candidates:
                continue
            child_observation_id = child_row + 1
            child_entity_id = entity_by_observation[child_observation_id]
            child_points = registered_entity_points(child_entity_id, frame)
            if not child_points["native"].shape[0]:
                continue
            best: dict[str, Any] | None = None
            for parent_row, centroid_distance in candidates:
                parent_observation_id = parent_row + 1
                parent_entity_id = entity_by_observation[parent_observation_id]
                parent_points = registered_entity_points(parent_entity_id, frame - 1)
                if not parent_points["native"].shape[0]:
                    continue
                pair = _transport_pair_samples(
                    parent_points,
                    child_points,
                    max_boundary_points=max_boundary_points,
                    mass_mode=mass_mode,
                    point_measure=point_measure,
                    cost_prealign=cost_prealign,
                    np=np,
                )
                score = _optimal_transport_score(
                    pair["source_cost"],
                    pair["target_cost"],
                    method=candidate_ot_method,  # type: ignore[arg-type]
                    regularization=sinkhorn_regularization,
                    unbalanced_reach=unbalanced_reach,
                    partial_mass=partial_mass,
                    max_transport_distance=(
                        max_transport_distance
                        if candidate_ot_method in {"unbalanced", "partial"}
                        else None
                    ),
                    source_weights=pair["source"].weights,
                    target_weights=pair["target"].weights,
                )
                frame_ot_score_count += 1
                if not score.solver_converged:
                    continue
                if (
                    float(score.source_coverage) < float(min_source_coverage)
                    or float(score.target_coverage) < float(min_target_coverage)
                ):
                    continue
                candidate_result = {
                    "parent_row": parent_row,
                    "parent_observation_id": parent_observation_id,
                    "parent_entity_id": parent_entity_id,
                    "child_observation_id": child_observation_id,
                    "child_entity_id": child_entity_id,
                    "centroid_distance": float(centroid_distance),
                    "score": score,
                    "parent_points": parent_points,
                    "child_points": child_points,
                    "pair": pair,
                }
                if best is None or score.total_cost < best["score"].total_cost:
                    best = candidate_result
            if best is None or float(best["score"].total_cost) >= ot_cutoff:
                continue
            best_score = best["score"]
            edge_records.append(
                (
                    link_id,
                    best["parent_observation_id"],
                    best["child_observation_id"],
                    frame - 1,
                    frame,
                    best["centroid_distance"],
                    float(best_score.total_cost),
                    float(min(best_score.source_coverage, best_score.target_coverage)),
                    0,
                )
            )
            if save_motion and boundary_set is not None:
                winner_score = best_score
                if accepted_winner_ot_method != candidate_ot_method:
                    pair = best["pair"]
                    winner_score = _optimal_transport_score(
                        pair["source_cost"],
                        pair["target_cost"],
                        method=accepted_winner_ot_method,  # type: ignore[arg-type]
                        regularization=sinkhorn_regularization,
                        unbalanced_reach=unbalanced_reach,
                        partial_mass=partial_mass,
                        max_transport_distance=(
                            max_transport_distance
                            if accepted_winner_ot_method in {"unbalanced", "partial"}
                            else None
                        ),
                        source_weights=pair["source"].weights,
                        target_weights=pair["target"].weights,
                    )
                plan = _materialize_transport_plan(
                    winner_score,
                    mass_tolerance=mass_tolerance,
                    relative_mass_tolerance=relative_mass_tolerance,
                    retained_mass_fraction=retained_mass_fraction,
                )
                pair = best["pair"]
                parent_points = best["parent_points"]
                child_points = best["child_points"]
                accepted_plans.append(
                    {
                        "track_link_id": link_id,
                        "parent_row": best["parent_row"],
                        "parent_observation_id": best["parent_observation_id"],
                        "parent_entity_id": best["parent_entity_id"],
                        "child_observation_id": best["child_observation_id"],
                        "child_entity_id": best["child_entity_id"],
                        "plan": plan,
                        "quality_flags": (
                            (1 if float(winner_score.source_coverage) < float(min_source_coverage) else 0)
                            | (2 if float(winner_score.target_coverage) < float(min_target_coverage) else 0)
                            | (4 if not winner_score.solver_converged else 0)
                        ),
                        "source_point_id": np.asarray(
                            parent_points["point_id"][pair["source"].rows], dtype=np.int64
                        ).copy(),
                        "target_point_id": np.asarray(
                            child_points["point_id"][pair["target"].rows], dtype=np.int64
                        ).copy(),
                        "source_registered": np.asarray(
                            parent_points["registered"][pair["source"].rows], dtype=float
                        ).copy(),
                        "target_registered": np.asarray(
                            child_points["registered"][pair["target"].rows], dtype=float
                        ).copy(),
                    }
                )
                frame_full_plan_count += 1
            link_id += 1
        if progress is not None:
            object_count = int(child_rows.size)
            linked_count = sum(1 for record in edge_records if int(record[4]) == frame)
            progress(
                {
                    "frame": int(frame),
                    "object_count": object_count,
                    "linked_count": int(linked_count),
                    "unlinked_count": object_count - int(linked_count),
                    "candidate_count": int(frame_candidate_count),
                    "ot_score_count": int(frame_ot_score_count),
                    "full_plan_count": int(frame_full_plan_count),
                }
            )
        for cached_frame in tuple(frame_point_cache):
            if int(cached_frame) != int(frame):
                del frame_point_cache[cached_frame]

    links = np.asarray(edge_records, dtype=link_dtype())
    adjacency = _csr_from_links(observation_count, links, np=np)
    assignments = _derive_assignments(observations, adjacency, links, np=np)
    schema = {
        "schema": "celltraj2.track_graph.v1",
        "object_set": object_name,
        "track_set": track_name,
        "method": "minimum_registered_boundary_ot_cost",
        "max_distance": cutoff,
        "candidate_k": centroid_candidate_k,
        "candidate_selection": {
            "method": "registered_centroid_knn_within_radius",
            "k": centroid_candidate_k,
            "max_distance": cutoff,
        },
        "ot_cost_cutoff": None if not np.isfinite(ot_cutoff) else ot_cutoff,
        "distance_unit": distance_unit,
        "coordinate_order": ["z", "y", "x"],
        "coordinate_scale": scale.tolist(),
        "registration_dependency": registration_dependency,
        "boundary_dependency": boundary_dependency,
        "boundary_source_id": selected_source_id,
        "boundary_source_name": str(source_record.get("name") or ""),
        # Backward-compatible alias: graph link costs and acceptance use the
        # candidate-scoring method, while saved motion may use another method.
        "ot_method_requested": candidate_ot_method,
        "score_ot_method_requested": candidate_ot_method,
        "winner_ot_method_requested": accepted_winner_ot_method,
        "candidate_scoring": {
            "ot_method_requested": candidate_ot_method,
            "determines": ["parent_ranking", "track_link_ot_cost", "ot_cost_cutoff"],
            "max_transport_distance": (
                None
                if max_transport_distance is None
                or candidate_ot_method not in {"unbalanced", "partial"}
                else float(max_transport_distance)
            ),
            "minimum_coverage": {
                "source": float(min_source_coverage),
                "target": float(min_target_coverage),
            },
        },
        "winner_motion": {
            "ot_method_requested": accepted_winner_ot_method,
            "enabled": bool(save_motion and boundary_set is not None),
            "executed_plan_count": len(accepted_plans),
            "affects_track_selection": False,
            "max_transport_distance": (
                None
                if max_transport_distance is None
                or accepted_winner_ot_method not in {"unbalanced", "partial"}
                else float(max_transport_distance)
            ),
        },
        "solver": {
            "max_iterations": 10000,
            "scaling_relative_tolerance": 1e-8,
            "log_absolute_tolerance": 1e-10,
            "underflow_fallback": "log_domain",
        },
        "sinkhorn_regularization": float(sinkhorn_regularization),
        "unbalanced_reach": float(unbalanced_reach),
        "partial_mass": float(partial_mass),
        "max_transport_distance": (
            None
            if max_transport_distance is None
            or candidate_ot_method not in {"unbalanced", "partial"}
            else float(max_transport_distance)
        ),
        "minimum_coverage": {
            "source": float(min_source_coverage),
            "target": float(min_target_coverage),
        },
        "max_boundary_points": max_boundary_points,
        "mass_tolerance": float(mass_tolerance),
        "relative_mass_tolerance": float(relative_mass_tolerance),
        "retained_mass_fraction": float(retained_mass_fraction),
        "mass_mode": str(mass_mode),
        "cost_prealign": str(cost_prealign),
        "sampling": {
            "method": "shared_physical_voxel_grid",
            "canonical_point_spacing": canonical_spacing,
            "point_measure": point_measure,
        },
        "ot_cost": (
            "full_unbalanced_regularized_objective"
            if candidate_ot_method == "unbalanced"
            else "matched_mean_euclidean_transport_distance"
        ),
        "frame_linkage": "immediately_previous_local_frame_only",
        "parent_invariant": "at_most_one_parent_per_child",
        "child_cardinality": "zero_or_more_children_per_parent",
        "adjacency": {
            "format": "csr",
            "orientation": "row_parent_column_child",
            "index_base": 0,
            "data": "one_based_link_id",
            "shape": [observation_count, observation_count],
        },
        "assignments": {
            "row_alignment": f"/object_sets/{object_name}/observations",
            "lineage_id": "rooted_weak_component_id",
            "tracklet_id": "maximal_non_branching_segment_id",
        },
        "link_count": int(links.shape[0]),
        "observation_count": observation_count,
        "track_digest": track_graph_digest(adjacency, links, assignments),
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
    motion_path = None
    motion_result = None
    if save_motion and boundary_set is not None:
        motion_result = _tracking_motion_from_plans(
            object_name=object_name,
            track_name=track_name,
            boundary_name=boundary_name,
            motion_name=motion_name,
            observations=observations,
            accepted_plans=accepted_plans,
            boundary_dependency=boundary_dependency,
            registration_dependency=registration_dependency,
            track_digest=schema["track_digest"],
            max_boundary_points=max_boundary_points,
            transport_settings={
                "ot_method_requested": accepted_winner_ot_method,
                "solver": dict(schema["solver"]),
                "sinkhorn_regularization": float(sinkhorn_regularization),
                "unbalanced_reach": float(unbalanced_reach),
                "partial_mass": float(partial_mass),
                "max_transport_distance": (
                    None
                    if max_transport_distance is None
                    or accepted_winner_ot_method not in {"unbalanced", "partial"}
                    else float(max_transport_distance)
                ),
                "minimum_coverage": {
                    "source": float(min_source_coverage),
                    "target": float(min_target_coverage),
                },
                "mass_tolerance": float(mass_tolerance),
                "relative_mass_tolerance": float(relative_mass_tolerance),
                "retained_mass_fraction": float(retained_mass_fraction),
                "mass_mode": str(mass_mode),
                "cost_prealign": str(cost_prealign),
                "sampling": dict(schema["sampling"]),
            },
            frame_counts=frame_counts,
            np=np,
        )
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
        if motion_result is not None:
            motion_path = trajectory.store.write_boundary_motion(
                boundary_name,
                motion_name,
                links=motion_result.links,
                transport=motion_result.transport,
                point_summaries=motion_result.point_summaries,
                schema=motion_result.schema,
                overwrite=overwrite,
            )
        metadata_payload = dict(metadata or {})
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
            "method": "minimum_registered_boundary_ot_cost",
            "max_distance": cutoff,
            "candidate_k": centroid_candidate_k,
            "score_ot_method": candidate_ot_method,
            "winner_ot_method": accepted_winner_ot_method,
            "ot_cost_cutoff": None if not np.isfinite(ot_cutoff) else ot_cutoff,
            "distance_unit": distance_unit,
            "registration_dependency": registration_dependency,
            "boundary_dependency": boundary_dependency,
            "observation_count": observation_count,
            "link_count": int(links.shape[0]),
            "track_path": track_path,
            "motion_path": motion_path,
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
        motion_path=motion_path,
        run_id=run_name,
        frame_counts=frame_counts,
        motion_result=motion_result,
    )


def _tracking_motion_from_plans(
    *,
    object_name: str,
    track_name: str,
    boundary_name: str,
    motion_name: str,
    observations: Any,
    accepted_plans: list[dict[str, Any]],
    boundary_dependency: Mapping[str, Any],
    registration_dependency: Mapping[str, Any] | None,
    track_digest: str,
    max_boundary_points: int | None,
    transport_settings: Mapping[str, Any],
    frame_counts: dict[int, dict[str, int]],
    np: Any,
) -> BoundaryMotionResult:
    from celltraj2.boundaries import boundary_motion_link_dtype

    motion_links: list[tuple[Any, ...]] = []
    transport_columns: dict[str, list[Any]] = {
        "source_point_id": [],
        "target_point_id": [],
        "mass": [],
        "edge_cost": [],
        "registered_displacement_zyx": [],
    }
    source_summary_columns: dict[str, list[Any]] = {}
    target_summary_columns: dict[str, list[Any]] = {}
    transport_start = 0
    source_summary_start = 0
    target_summary_start = 0
    selected_methods: set[str] = set()
    for motion_link_id, accepted in enumerate(accepted_plans, start=1):
        plan = accepted["plan"]
        selected_methods.add(str(plan.method))
        source_point_ids = np.asarray(accepted["source_point_id"], dtype=np.int64)
        target_point_ids = np.asarray(accepted["target_point_id"], dtype=np.int64)
        source_registered = np.asarray(accepted["source_registered"], dtype=float)
        target_registered = np.asarray(accepted["target_registered"], dtype=float)
        displacement = target_registered[plan.target_rows] - source_registered[plan.source_rows]
        source_summary = _transport_point_summary(
            plan,
            point_ids=source_point_ids,
            displacement=displacement,
            direction="source",
            np=np,
        )
        target_summary = _transport_point_summary(
            plan,
            point_ids=target_point_ids,
            displacement=displacement,
            direction="target",
            np=np,
        )
        transport_count = int(plan.mass.shape[0])
        source_summary_count = int(source_point_ids.size)
        target_summary_count = int(target_point_ids.size)
        motion_links.append(
            (
                motion_link_id,
                accepted["track_link_id"],
                accepted["parent_entity_id"],
                accepted["child_entity_id"],
                accepted["parent_observation_id"],
                accepted["child_observation_id"],
                int(observations[accepted["parent_row"]]["frame"]),
                int(observations[accepted["child_observation_id"] - 1]["frame"]),
                transport_start,
                transport_count,
                source_summary_start,
                source_summary_count,
                target_summary_start,
                target_summary_count,
                float(plan.total_cost),
                float(plan.transport_cost),
                float(plan.objective),
                float(plan.matched_mean_cost),
                float(plan.transported_mass),
                float(plan.source_coverage),
                float(plan.target_coverage),
                float(plan.dropped_mass),
                float(plan.edge_distance_p50),
                float(plan.edge_distance_p90),
                float(plan.edge_distance_p99),
                float(plan.edge_distance_max),
                int(plan.solver_iterations),
                int(bool(plan.solver_converged)),
                int(accepted.get("quality_flags", 0)),
            )
        )
        transport_columns["source_point_id"].append(source_point_ids[plan.source_rows].astype(np.int64))
        transport_columns["target_point_id"].append(target_point_ids[plan.target_rows].astype(np.int64))
        transport_columns["mass"].append(np.asarray(plan.mass, dtype=np.float64))
        transport_columns["edge_cost"].append(np.asarray(plan.edge_cost, dtype=np.float32))
        transport_columns["registered_displacement_zyx"].append(np.asarray(displacement, dtype=np.float32))
        _append_columns(source_summary_columns, source_summary)
        _append_columns(target_summary_columns, target_summary)
        transport_start += transport_count
        source_summary_start += source_summary_count
        target_summary_start += target_summary_count
    transport = _concatenate_column_blocks(transport_columns, np=np)
    point_summaries = {
        "source_summary": _concatenate_column_blocks(source_summary_columns, np=np),
        "target_summary": _concatenate_column_blocks(target_summary_columns, np=np),
    }
    motion_schema = {
        "schema": "celltraj2.boundary_motion.v2",
        "boundary_set": boundary_name,
        "motion_set": motion_name,
        "created_at": utc_now_iso(),
        "source": f"/object_sets/{object_name}/tracks/{track_name}",
        "boundary_dependency": dict(boundary_dependency),
        "track_dependency": {"object_set": object_name, "track_set": track_name, "track_digest": track_digest},
        "registration_dependency": registration_dependency,
        "point_identity_coordinate_system": "native_boundary_point_id",
        "displacement_coordinate_system": "registered_roi_physical",
        "displacement_definition": "T_target(q_native)-T_source(p_native)",
        "transport_methods": sorted(selected_methods),
        "max_boundary_points": max_boundary_points,
        **dict(transport_settings),
        "point_summary": {
            "source_group": "source_summary",
            "target_group": "target_summary",
            "vector": "mass_weighted_barycentric_displacement",
            "confidence": "matched_fraction",
        },
        "link_quality_flags": {
            "1": "source_coverage_below_threshold",
            "2": "target_coverage_below_threshold",
            "4": "solver_not_converged",
        },
        "transport_edge_count": int(transport_start),
    }
    return BoundaryMotionResult(
        object_set=object_name,
        track_set=track_name,
        boundary_set=boundary_name,
        motion_set=motion_name,
        links=np.asarray(motion_links, dtype=boundary_motion_link_dtype()),
        transport=transport,
        point_summaries=point_summaries,
        schema=motion_schema,
        saved=False,
        frame_counts={int(key): dict(value) for key, value in frame_counts.items()},
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
