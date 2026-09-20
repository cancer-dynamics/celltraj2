"""Individual and Voronoi-neighbor centroid kinematics, independent of boundaries."""

from collections import OrderedDict

import numpy as np
from scipy.spatial import ConvexHull, QhullError, Voronoi

from celltraj2.feature_catalog import CENTROID_METRICS
from celltraj2.registration import registration_calibration


def time_calibration(metadata):
    acquisition = dict(getattr(metadata, "acquisition", {}) or {})
    for key, divisor in (("time_interval_s", 3600), ("frame_interval_s", 3600),
                         ("timestep_s", 3600), ("time_step_s", 3600),
                         ("time_interval_ms", 3600000), ("frame_interval_ms", 3600000)):
        try:
            interval = float(acquisition.get(key)) / divisor
        except (TypeError, ValueError):
            continue
        if np.isfinite(interval) and interval > 0:
            return interval, "h"
    return 1.0, "frame"


def voronoi_neighbors(points, lower, upper):
    """ROI-box-clipped shared Voronoi face measures (edge lengths in 2D).

    Reflections across each box face impose the clipping planes exactly. Exterior
    faces are excluded from weights; no artificial jitter or infinite-face cutoff.
    """
    points = np.asarray(points, dtype=float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    n, ndim = points.shape
    if ndim not in (2, 3) or lower.shape != (ndim,) or upper.shape != (ndim,):
        raise ValueError("Voronoi inputs must be 2D or 3D with matching box bounds.")
    if np.any(upper <= lower) or not np.all(np.isfinite(points)):
        raise ValueError("Voronoi coordinates and bounds must be finite and valid.")
    if np.any(points <= lower) or np.any(points >= upper):
        raise ValueError("Centroids must lie strictly inside the ROI box.")
    if len(np.unique(points, axis=0)) != n:
        raise ValueError("Coincident centroids do not define unique Voronoi cells.")
    neighbors = [[] for _ in range(n)]
    if n < 2:
        return neighbors
    sites = [points]
    for axis in range(ndim):
        for bound in (lower[axis], upper[axis]):
            reflected = points.copy()
            reflected[:, axis] = 2 * bound - reflected[:, axis]
            sites.append(reflected)
    diagram = Voronoi(np.vstack(sites))
    for (i, j), ridge in zip(diagram.ridge_points, diagram.ridge_vertices):
        if i >= n or j >= n or -1 in ridge or len(ridge) < ndim:
            continue
        vertices = diagram.vertices[ridge]
        if ndim == 2:
            measure = float(np.linalg.norm(vertices[0] - vertices[1]))
        else:
            centered = vertices - vertices.mean(axis=0)
            _, _, basis = np.linalg.svd(centered, full_matrices=False)
            try:
                measure = float(ConvexHull(centered @ basis[:2].T).volume)
            except QhullError:
                continue
        if measure > 0:
            neighbors[i].append((int(j), measure))
            neighbors[j].append((int(i), measure))
    return neighbors


def collective_values(points, displacement, velocity, neighbors, *, weighting="shared_face"):
    if weighting not in {"shared_face", "uniform"}:
        raise ValueError("Neighbor weighting must be shared_face or uniform.")
    n, ndim = points.shape
    result = {name: np.full(n, np.nan) for name in CENTROID_METRICS[10:]}
    lengths = np.linalg.norm(displacement, axis=1)
    unit = np.divide(displacement, lengths[:, None], out=np.full_like(displacement, np.nan), where=lengths[:, None] > 0)
    for i, adjacency in enumerate(neighbors):
        result["neighbor_count"][i] = len(adjacency)
        if not adjacency:
            continue
        js = np.array([j for j, _ in adjacency], int)
        weights = np.array([w if weighting == "shared_face" else 1.0 for _, w in adjacency])
        finite = np.all(np.isfinite(velocity[js]), axis=1)
        result["valid_neighbor_fraction"][i] = float(weights[finite].sum() / weights.sum())
        def average(values):
            valid = np.isfinite(values)
            return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan
        cosines = unit[js] @ unit[i]
        radial = points[i] - points[js]
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        result["beta"][i] = average(cosines)
        result["alpha"][i] = average(np.sum((unit[i] - unit[js]) * radial, axis=1))
        result["nematic_alignment"][i] = average((ndim * cosines ** 2 - 1) / (ndim - 1))
        result["displacement_dot"][i] = average(displacement[js] @ displacement[i])
        result["neighbor_speed"][i] = average(np.linalg.norm(velocity[js], axis=1))
        result["relative_speed"][i] = average(np.linalg.norm(velocity[js] - velocity[i], axis=1))
        valid = np.all(np.isfinite(unit[js]), axis=1)
        if valid.any():
            result["polarization"][i] = float(np.linalg.norm(np.average(unit[js][valid], axis=0, weights=weights[valid])))
    return result


def compute_centroid_frame(trajectory, labels, *, frame, object_set, feature, cache):
    from celltraj2.features import _slug
    from celltraj2.tracking import track_graph_digest

    track_name = str(feature.get("track_set") or "")
    if not track_name:
        raise ValueError("Centroid motility requires a stored track_set.")
    registration_name = feature.get("registration_set", "auto")
    key = ("centroid", object_set, track_name, registration_name)
    if key not in cache:
        observations = trajectory.store.read_observations(object_set)
        graph = trajectory.read_tracks(object_set, track_name)
        if graph.observation_count != len(observations):
            raise ValueError("Track assignments are not aligned with the active object observations.")
        calibration = registration_calibration(trajectory.metadata)
        scale = np.asarray(calibration["coordinate_scale"])
        native = np.column_stack([observations[f"centroid_{axis}"] for axis in "zyx"]) * scale
        positions = native.copy()
        selected_registration = trajectory.store.active_registration_name() if registration_name == "auto" else registration_name
        registration = trajectory.store.read_registration_set(selected_registration) if selected_registration else None
        if registration is not None:
            stored_scale = registration.schema.get("coordinate_scale_zyx", scale)
            if registration.schema.get("method") != "identity" and not np.allclose(stored_scale, scale):
                raise ValueError("Centroid and registration physical calibrations differ.")
            positions = registration.apply_zyx(positions, observations["frame"])
        interval, time_unit = time_calibration(trajectory.metadata)
        displacement = np.full_like(positions, np.nan)
        velocity = np.full_like(positions, np.nan)
        persistence = np.full(len(positions), np.nan)
        acceleration = np.full(len(positions), np.nan)
        durations = np.full(len(positions), np.nan)
        parents = np.asarray(graph.assignments["parent_observation_id"], dtype=int) - 1
        for i, parent in enumerate(parents):
            if parent < 0:
                continue
            duration = float(observations["frame"][i] - observations["frame"][parent]) * interval
            if duration <= 0:
                raise ValueError("Track parent must precede child in time.")
            durations[i] = duration
            displacement[i] = positions[i] - positions[parent]
            velocity[i] = displacement[i] / duration
        for i, parent in enumerate(parents):
            if parent < 0 or not np.all(np.isfinite(velocity[parent])):
                continue
            denominator = np.linalg.norm(displacement[i]) * np.linalg.norm(displacement[parent])
            if denominator > 0:
                persistence[i] = np.clip(np.dot(displacement[i], displacement[parent]) / denominator, -1, 1)
            acceleration[i] = np.linalg.norm(velocity[i] - velocity[parent]) / ((durations[i] + durations[parent]) / 2)
        cache[key] = dict(observations=observations, native=native, displacement=displacement, velocity=velocity,
                          persistence=persistence, acceleration=acceleration, calibration=calibration,
                          time_unit=time_unit, interval=interval, dependency={
                              "track_set": track_name,
                              "track_graph_digest": track_graph_digest(graph.adjacency, graph.links, graph.assignments),
                              "registration_set": selected_registration,
                              "registration_digest": registration.digest if registration is not None else None,
                          })
    data = cache[key]
    rows = np.flatnonzero(data["observations"]["frame"] == frame)
    ndim = labels.ndim
    native = data["native"][rows, -ndim:]
    displacement = data["displacement"][rows]
    velocity = data["velocity"][rows]
    metrics = list(feature.get("metrics", ("displacement", "speed", "persistence", "beta", "alpha", "neighbor_count")))
    if not metrics or set(metrics) - set(CENTROID_METRICS):
        raise ValueError("Select valid centroid motility metrics.")
    values = {f"displacement_{axis}": displacement[:, j] for j, axis in enumerate("zyx")}
    values.update({f"velocity_{axis}": velocity[:, j] for j, axis in enumerate("zyx")})
    values.update(displacement=np.linalg.norm(displacement, axis=1), speed=np.linalg.norm(velocity, axis=1),
                  persistence=data["persistence"][rows], acceleration=data["acceleration"][rows])
    warnings = []
    if set(metrics) & set(CENTROID_METRICS[10:]):
        scale = np.asarray(data["calibration"]["coordinate_scale"])[-ndim:]
        try:
            neighbors = voronoi_neighbors(native, -.5 * scale, (np.asarray(labels.shape) - .5) * scale)
            values.update(collective_values(native, displacement[:, -ndim:], velocity[:, -ndim:], neighbors,
                                            weighting=str(feature.get("neighbor_weighting", "shared_face"))))
        except (ValueError, QhullError) as exc:
            warnings.append(f"Voronoi neighbors unavailable in frame {frame}: {exc}")
            values.update({metric: np.full(len(rows), np.nan) for metric in CENTROID_METRICS[10:]})
    prefix = _slug(feature.get("name") or "centroid")
    length_unit, time_unit = data["calibration"]["distance_unit"], data["time_unit"]
    def unit(metric):
        if metric == "displacement_dot":
            return f"{length_unit}^2"
        if metric.startswith("displacement"):
            return length_unit
        if metric in {"speed", "neighbor_speed", "relative_speed"} or metric.startswith("velocity"):
            return f"{length_unit}/{time_unit}"
        if metric == "acceleration":
            return f"{length_unit}/{time_unit}^2"
        return "dimensionless"
    columns = OrderedDict((f"{prefix}_{metric}", {
        "name": f"{prefix}_{metric}", "dtype": "float64", "family": "centroid_motility", "metric": metric,
        "unit": unit(metric), "calibration": data["calibration"], "time_unit": time_unit,
        "time_interval": data["interval"], "track_dependency": data["dependency"],
        "neighbor_method": "roi_box_clipped_voronoi", "neighbor_weighting": feature.get("neighbor_weighting", "shared_face"),
        "direction": "incoming_parent_to_child",
    }) for metric in metrics)
    return {"columns": columns, "warnings": warnings, "values_by_label": {
        int(data["observations"]["label_id"][row]): {f"{prefix}_{metric}": float(values[metric][j]) for metric in metrics}
        for j, row in enumerate(rows)
    }}
