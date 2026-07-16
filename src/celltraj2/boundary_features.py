"""Row-aligned object summaries derived from stored boundary products."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from celltraj2.paths import validate_name


_GEOMETRY_KINDS = {"boundary_geometry", "surface_geometry"}
_INTERACTION_KINDS = {"boundary_interaction", "surface_interaction", "boundary_contact"}
_MOTION_KINDS = {"boundary_motion", "surface_motion"}
_MULTIPOLE_KINDS = {"boundary_multipole", "surface_multipole"}


def compute_boundary_feature_frame(
    trajectory: Any,
    *,
    object_set: str,
    frame: int,
    feature: Mapping[str, Any],
    cache: dict[str, Any] | None = None,
    np: Any,
) -> dict[str, Any]:
    """Compute one boundary-derived feature block for one indexed frame."""

    kind = str(feature.get("kind") or feature.get("type") or "").strip().lower()
    context = _boundary_context(
        trajectory,
        object_set=object_set,
        frame=frame,
        feature=feature,
        cache=cache if cache is not None else {},
        np=np,
    )
    if kind in _GEOMETRY_KINDS:
        return _geometry_summary(context, feature=feature, np=np)
    if kind in _INTERACTION_KINDS:
        return _interaction_summary(context, feature=feature, np=np)
    if kind in _MOTION_KINDS:
        return _motion_summary(context, feature=feature, np=np)
    if kind in _MULTIPOLE_KINDS:
        return _multipole_summary(context, feature=feature, np=np)
    raise ValueError(f"Unsupported boundary feature kind: {kind!r}")


def boundary_multipole_magnitudes(
    points_zyx: Any,
    charge: Any,
    *,
    order: int = 2,
    spatial_ndim: int = 3,
) -> Any:
    """Return centered, rotationally invariant angular multipole magnitudes.

    The monopole is the signed mean charge. Higher 2-D orders are Fourier
    amplitudes around the boundary. Higher 3-D orders are the power across the
    complex spherical-harmonic channels for that order. Coordinates are
    centered and converted to directions, so the result is translation and
    radius-scale invariant; charge units are retained.
    """

    import numpy as np

    maximum = int(order)
    if maximum < 0 or maximum > 8:
        raise ValueError("Boundary multipole order must be between 0 and 8")
    points = np.asarray(points_zyx, dtype=float)
    values = np.asarray(charge, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] != values.shape[0]:
        raise ValueError("Boundary multipoles require N x 3 points and N charges")
    valid = np.isfinite(values) & np.all(np.isfinite(points), axis=1)
    if not np.any(valid):
        return np.full(maximum + 1, np.nan, dtype=float)
    points = points[valid]
    values = values[valid]
    centered = points - np.mean(points, axis=0, keepdims=True)
    out = np.full(maximum + 1, np.nan, dtype=float)
    out[0] = float(np.mean(values))
    if maximum == 0:
        return out

    if int(spatial_ndim) <= 2:
        spread = np.nanstd(centered, axis=0)
        axes = np.argsort(spread)[-2:]
        planar = centered[:, axes]
        radius = np.linalg.norm(planar, axis=1)
        directional = radius > 1e-12
        if not np.any(directional):
            return out
        theta = np.arctan2(planar[directional, 1], planar[directional, 0])
        q = values[directional]
        for ell in range(1, maximum + 1):
            moment = np.mean(q * np.exp(-1j * float(ell) * theta))
            out[ell] = float(abs(moment))
        return out

    radius = np.linalg.norm(centered, axis=1)
    directional = radius > 1e-12
    if not np.any(directional):
        return out
    unit = centered[directional] / radius[directional, None]
    q = values[directional]
    cos_theta = np.clip(unit[:, 0], -1.0, 1.0)
    phi = np.arctan2(unit[:, 1], unit[:, 2])
    for ell in range(1, maximum + 1):
        power = 0.0
        for m in range(0, ell + 1):
            legendre = _associated_legendre(ell, m, cos_theta, np=np)
            normalization = math.sqrt(
                ((2 * ell + 1) / (4.0 * math.pi))
                * (math.factorial(ell - m) / math.factorial(ell + m))
            )
            harmonic = normalization * legendre * np.exp(1j * float(m) * phi)
            moment = np.mean(q * np.conjugate(harmonic))
            power += float(abs(moment) ** 2) * (1.0 if m == 0 else 2.0)
        out[ell] = math.sqrt(max(0.0, power)) * math.sqrt(4.0 * math.pi / (2 * ell + 1))
    return out


def _associated_legendre(ell: int, m: int, x: Any, *, np: Any) -> Any:
    pmm = np.ones_like(x, dtype=float)
    if m > 0:
        root = np.sqrt(np.maximum(0.0, 1.0 - x * x))
        factor = 1.0
        for _ in range(1, m + 1):
            pmm *= -factor * root
            factor += 2.0
    if ell == m:
        return pmm
    pmmp1 = x * float(2 * m + 1) * pmm
    if ell == m + 1:
        return pmmp1
    previous = pmm
    current = pmmp1
    for degree in range(m + 2, ell + 1):
        following = (
            float(2 * degree - 1) * x * current - float(degree + m - 1) * previous
        ) / float(degree - m)
        previous, current = current, following
    return current


def _boundary_context(
    trajectory: Any,
    *,
    object_set: str,
    frame: int,
    feature: Mapping[str, Any],
    cache: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    object_name = validate_name(object_set, kind="object set")
    boundary_name = validate_name(
        str(feature.get("boundary_set") or ""), kind="boundary set"
    )
    libraries = cache.setdefault("boundary_libraries", {})
    if boundary_name not in libraries:
        libraries[boundary_name] = trajectory.boundary_library(boundary_name)
    view = libraries[boundary_name]
    source = _resolve_object_source(view, object_name=object_name, feature=feature)
    source_id = int(source["source_id"])
    entities = view.entities_for_frame(int(frame), source_id=source_id)
    dependency = {
        "boundary_set": boundary_name,
        "boundary_digest": str(view.schema.get("boundary_digest") or ""),
        "source_id": source_id,
        "source_name": str(source.get("name") or ""),
        "source_role": str(source.get("role") or ""),
        "object_set": object_name,
        "coordinate_system": "native_roi_physical",
    }
    return {
        "trajectory": trajectory,
        "view": view,
        "cache": cache,
        "frame": int(frame),
        "object_set": object_name,
        "boundary_set": boundary_name,
        "source": source,
        "source_id": source_id,
        "entities": entities,
        "dependency": dependency,
    }


def _resolve_object_source(view: Any, *, object_name: str, feature: Mapping[str, Any]) -> dict[str, Any]:
    source_id = feature.get("boundary_source_id", feature.get("source_id"))
    source_name = feature.get("boundary_source_name", feature.get("source_name"))
    source_role = feature.get("boundary_source_role", feature.get("source_role"))
    candidates: list[dict[str, Any]] = []
    for item in view.sources:
        source = dict(item)
        if source_id not in (None, "") and int(source.get("source_id", -1)) != int(source_id):
            continue
        if source_name not in (None, "") and str(source.get("name") or "") != str(source_name):
            continue
        if source_role not in (None, "") and str(source.get("role") or "") != str(source_role):
            continue
        if source_id in (None, "") and source_name in (None, "") and source_role in (None, ""):
            if str(source.get("kind") or "") != "object_set":
                continue
            if str(source.get("object_set") or "") != object_name:
                continue
        candidates.append(source)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one boundary source for object set {object_name!r}; found "
            f"{[(item.get('source_id'), item.get('name'), item.get('role')) for item in candidates]}"
        )
    selected = candidates[0]
    if str(selected.get("kind") or "") != "object_set" or str(selected.get("object_set") or "") != object_name:
        raise ValueError(
            f"Boundary source {selected.get('name')!r} does not represent object set {object_name!r}"
        )
    return selected


def _geometry_summary(context: Mapping[str, Any], *, feature: Mapping[str, Any], np: Any) -> dict[str, Any]:
    geometry_set = validate_name(str(feature.get("geometry_set") or ""), kind="boundary geometry set")
    fields = _string_list(feature.get("fields", feature.get("field", "mean_curvature")))
    statistics = _string_list(feature.get("statistics", feature.get("stats", ("mean", "std"))))
    if not fields or not statistics:
        raise ValueError("Boundary geometry summaries require fields and statistics")
    prefix = _slug(feature.get("name") or feature.get("prefix") or "surface_geometry")
    schema = _product_schema(context, "geometry", geometry_set)
    _validate_product_source(
        schema,
        key="selected_source_ids",
        context=context,
        product=f"Geometry set {geometry_set!r}",
    )
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for field_name in fields:
        for statistic in statistics:
            column = _summary_column(prefix, field_name, statistic, field_count=len(fields))
            columns[column] = {
                "name": column,
                "dtype": "float64",
                "family": "boundary_geometry",
                "statistic": statistic,
                "boundary_field": field_name,
                "boundary_dependency": dict(context["dependency"]),
                "geometry_dependency": {"geometry_set": geometry_set, "schema": schema},
            }
    values_by_label: dict[int, dict[str, float]] = {}
    for entity in context["entities"]:
        label_id = int(entity["source_label_id"])
        data = context["view"].geometry(
            geometry_set,
            int(entity["boundary_entity_id"]),
            fields=fields,
        )
        row = values_by_label.setdefault(label_id, {})
        for field_name in fields:
            if field_name not in data:
                raise KeyError(
                    f"Geometry field {field_name!r} is not stored in {geometry_set!r}"
                )
            field_values = np.asarray(data[field_name])
            if field_values.ndim != 1:
                raise ValueError(f"Boundary geometry field {field_name!r} is not scalar")
            for statistic in statistics:
                column = _summary_column(prefix, field_name, statistic, field_count=len(fields))
                row[column] = _statistic(field_values, statistic, np=np)
    warnings = [] if len(context["entities"]) else [f"No selected boundary entities in frame {context['frame']}"]
    return {"values_by_label": values_by_label, "columns": columns, "warnings": warnings}


def _interaction_summary(context: Mapping[str, Any], *, feature: Mapping[str, Any], np: Any) -> dict[str, Any]:
    neighbor_set = validate_name(str(feature.get("neighbor_set") or ""), kind="boundary neighbor set")
    metrics = _string_list(
        feature.get(
            "metrics",
            ("contact_fraction", "distance_mean", "distance_min", "neighbor_entity_count"),
        )
    )
    allowed = {
        "contact_fraction",
        "contact_point_count",
        "distance_coverage_fraction",
        "distance_mean",
        "distance_std",
        "distance_median",
        "distance_min",
        "distance_max",
        "neighbor_entity_count",
    }
    unsupported = sorted(set(metrics) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported boundary interaction metric(s): {unsupported}")
    contact_distance = float(feature.get("contact_distance", feature.get("contact_threshold", 1.0)))
    if not math.isfinite(contact_distance) or contact_distance <= 0:
        raise ValueError("Boundary contact_distance must be finite and > 0")
    prefix = _slug(feature.get("name") or feature.get("prefix") or "surface_contact")
    schema = _product_schema(context, "neighbors", neighbor_set)
    _validate_product_source(
        schema,
        key="source_ids",
        context=context,
        product=f"Neighbor set {neighbor_set!r}",
    )
    stored_limit = schema.get("max_distance")
    if stored_limit not in (None, "") and float(stored_limit) + 1e-12 < contact_distance:
        raise ValueError(
            f"Neighbor set {neighbor_set!r} was truncated at {stored_limit}; it cannot measure "
            f"contact_distance={contact_distance}"
        )
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for metric in metrics:
        column = _slug(f"{prefix}_{metric}")
        columns[column] = {
            "name": column,
            "dtype": "float64",
            "family": "boundary_interaction",
            "statistic": metric,
            "contact_distance": contact_distance,
            "distance_unit": str(context["view"].schema.get("coordinate_unit") or "physical"),
            "boundary_dependency": dict(context["dependency"]),
            "neighbor_dependency": {"neighbor_set": neighbor_set, "schema": schema},
        }
    point_entities = _point_entity_ids(context, np=np)
    values_by_label: dict[int, dict[str, float]] = {}
    for entity in context["entities"]:
        label_id = int(entity["source_label_id"])
        distances, target_rows, target_distances = _neighbor_point_distances(
            context, neighbor_set=neighbor_set, entity=entity, np=np
        )
        finite = np.isfinite(distances)
        contact = finite & (distances <= contact_distance)
        entity_count = 0
        if np.any(contact) and target_rows:
            contacted_rows = np.concatenate(
                [
                    target_rows[index][target_distances[index] <= contact_distance]
                    for index in np.flatnonzero(contact)
                    if np.any(target_distances[index] <= contact_distance)
                ]
            ) if any(
                np.any(target_distances[index] <= contact_distance)
                for index in np.flatnonzero(contact)
            ) else np.empty(0, dtype=np.int64)
            if contacted_rows.size:
                entity_count = int(np.unique(point_entities[contacted_rows]).size)
        metric_values = {
            "contact_fraction": float(np.sum(contact) / distances.size) if distances.size else np.nan,
            "contact_point_count": float(np.sum(contact)),
            "distance_coverage_fraction": float(np.sum(finite) / distances.size) if distances.size else np.nan,
            "distance_mean": _statistic(distances, "mean", np=np),
            "distance_std": _statistic(distances, "std", np=np),
            "distance_median": _statistic(distances, "median", np=np),
            "distance_min": _statistic(distances, "min", np=np),
            "distance_max": _statistic(distances, "max", np=np),
            "neighbor_entity_count": float(entity_count),
        }
        values_by_label[label_id] = {
            _slug(f"{prefix}_{metric}"): metric_values[metric] for metric in metrics
        }
    warnings = [] if len(context["entities"]) else [f"No selected boundary entities in frame {context['frame']}"]
    return {"values_by_label": values_by_label, "columns": columns, "warnings": warnings}


def _motion_summary(context: Mapping[str, Any], *, feature: Mapping[str, Any], np: Any) -> dict[str, Any]:
    motion_set = validate_name(str(feature.get("motion_set") or ""), kind="boundary motion set")
    direction = str(feature.get("direction") or "incoming").lower()
    if direction not in {"incoming", "outgoing", "both"}:
        raise ValueError("Boundary motion direction must be incoming, outgoing, or both")
    metrics = _string_list(
        feature.get(
            "metrics",
            (
                "displacement_z_mean",
                "displacement_y_mean",
                "displacement_x_mean",
                "magnitude_mean",
                "normal_mean",
                "mapped_fraction",
                "ot_cost_mean",
            ),
        )
    )
    allowed = {
        "displacement_z_mean", "displacement_y_mean", "displacement_x_mean",
        "displacement_z_std", "displacement_y_std", "displacement_x_std",
        "magnitude_mean", "magnitude_std", "magnitude_min", "magnitude_max",
        "normal_mean", "normal_std", "normal_min", "normal_max",
        "tangential_magnitude_mean", "mapped_fraction", "ot_cost_mean",
        "transported_mass_sum", "motion_link_count",
    }
    unsupported = sorted(set(metrics) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported boundary motion metric(s): {unsupported}")
    geometry_set = feature.get("geometry_set")
    if any(metric.startswith("normal_") or metric.startswith("tangential_") for metric in metrics):
        geometry_set = validate_name(str(geometry_set or ""), kind="boundary geometry set")
    geometry_schema = None
    if geometry_set not in (None, ""):
        geometry_schema = _product_schema(context, "geometry", str(geometry_set))
        _validate_product_source(
            geometry_schema,
            key="selected_source_ids",
            context=context,
            product=f"Geometry set {geometry_set!r}",
        )
    prefix = _slug(feature.get("name") or feature.get("prefix") or "surface_motion")
    motion = _motion_product(context, motion_set=motion_set)
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for metric in metrics:
        column = _slug(f"{prefix}_{metric}")
        product_dependency: dict[str, Any] = {
            "motion_set": motion_set,
            "direction": direction,
            "schema": motion["schema"],
        }
        if geometry_set:
            product_dependency["geometry_set"] = str(geometry_set)
            product_dependency["geometry_schema"] = geometry_schema
        columns[column] = {
            "name": column,
            "dtype": "float64",
            "family": "boundary_motion",
            "statistic": metric,
            "boundary_dependency": dict(context["dependency"]),
            "motion_dependency": product_dependency,
            "registration_dependency": motion["schema"].get("registration_dependency"),
        }
    values_by_label: dict[int, dict[str, float]] = {}
    for entity in context["entities"]:
        data = _motion_point_data(
            context,
            motion_set=motion_set,
            entity=entity,
            direction=direction,
            geometry_set=None if geometry_set in (None, "") else str(geometry_set),
            np=np,
        )
        vectors = data["vectors"]
        finite = np.all(np.isfinite(vectors), axis=1)
        magnitude = np.linalg.norm(vectors, axis=1)
        normal = data.get("normal_displacement")
        tangential = data.get("tangential_magnitude")
        metric_values = {
            "displacement_z_mean": _statistic(vectors[:, 0], "mean", np=np),
            "displacement_y_mean": _statistic(vectors[:, 1], "mean", np=np),
            "displacement_x_mean": _statistic(vectors[:, 2], "mean", np=np),
            "displacement_z_std": _statistic(vectors[:, 0], "std", np=np),
            "displacement_y_std": _statistic(vectors[:, 1], "std", np=np),
            "displacement_x_std": _statistic(vectors[:, 2], "std", np=np),
            "magnitude_mean": _statistic(magnitude, "mean", np=np),
            "magnitude_std": _statistic(magnitude, "std", np=np),
            "magnitude_min": _statistic(magnitude, "min", np=np),
            "magnitude_max": _statistic(magnitude, "max", np=np),
            "normal_mean": _statistic(normal, "mean", np=np),
            "normal_std": _statistic(normal, "std", np=np),
            "normal_min": _statistic(normal, "min", np=np),
            "normal_max": _statistic(normal, "max", np=np),
            "tangential_magnitude_mean": _statistic(tangential, "mean", np=np),
            "mapped_fraction": float(np.sum(finite) / vectors.shape[0]) if vectors.shape[0] else np.nan,
            "ot_cost_mean": _statistic(data["ot_cost"], "mean", np=np),
            "transported_mass_sum": float(np.nansum(data["transported_mass"])),
            "motion_link_count": float(data["link_count"]),
        }
        label_id = int(entity["source_label_id"])
        values_by_label[label_id] = {
            _slug(f"{prefix}_{metric}"): metric_values[metric] for metric in metrics
        }
    warnings = [] if len(context["entities"]) else [f"No selected boundary entities in frame {context['frame']}"]
    return {"values_by_label": values_by_label, "columns": columns, "warnings": warnings}


def _multipole_summary(context: Mapping[str, Any], *, feature: Mapping[str, Any], np: Any) -> dict[str, Any]:
    order = int(feature.get("order", 2))
    if order < 0 or order > 8:
        raise ValueError("Boundary multipole order must be between 0 and 8")
    signal = str(feature.get("signal") or "neighbor_distance").lower()
    allowed = {
        "geometry", "neighbor_distance", "motion_normal_displacement",
        "motion_magnitude", "shape_radial_deviation", "shape_radius",
    }
    if signal not in allowed:
        raise ValueError(f"Unsupported boundary multipole signal: {signal!r}")
    prefix = _slug(feature.get("name") or feature.get("prefix") or "boundary_multipole")
    signal_dependency: dict[str, Any] = {"signal": signal}
    if signal == "geometry":
        geometry_set = validate_name(str(feature.get("geometry_set") or ""), kind="boundary geometry set")
        field_name = str(feature.get("geometry_field") or feature.get("field") or "mean_curvature")
        geometry_schema = _product_schema(context, "geometry", geometry_set)
        _validate_product_source(
            geometry_schema,
            key="selected_source_ids",
            context=context,
            product=f"Geometry set {geometry_set!r}",
        )
        signal_dependency.update(
            geometry_set=geometry_set,
            geometry_field=field_name,
            schema=geometry_schema,
        )
    elif signal == "neighbor_distance":
        neighbor_set = validate_name(str(feature.get("neighbor_set") or ""), kind="boundary neighbor set")
        transform = str(feature.get("distance_transform") or "inverse_distance").lower()
        if transform not in {"distance", "negative_distance", "inverse_distance", "contact_indicator"}:
            raise ValueError(f"Unsupported boundary distance transform: {transform!r}")
        epsilon = float(feature.get("distance_epsilon", 1e-6))
        contact_distance = float(feature.get("contact_distance", 1.0))
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Boundary multipole distance_epsilon must be finite and > 0")
        if not math.isfinite(contact_distance) or contact_distance <= 0:
            raise ValueError("Boundary multipole contact_distance must be finite and > 0")
        neighbor_schema = _product_schema(context, "neighbors", neighbor_set)
        _validate_product_source(
            neighbor_schema,
            key="source_ids",
            context=context,
            product=f"Neighbor set {neighbor_set!r}",
        )
        stored_limit = neighbor_schema.get("max_distance")
        if (
            transform == "contact_indicator"
            and stored_limit not in (None, "")
            and float(stored_limit) + 1e-12 < contact_distance
        ):
            raise ValueError(
                f"Neighbor set {neighbor_set!r} was truncated at {stored_limit}; it cannot measure "
                f"contact_distance={contact_distance}"
            )
        signal_dependency.update(
            neighbor_set=neighbor_set,
            distance_transform=transform,
            distance_epsilon=epsilon,
            contact_distance=contact_distance,
            schema=neighbor_schema,
        )
    elif signal.startswith("motion_"):
        motion_set = validate_name(str(feature.get("motion_set") or ""), kind="boundary motion set")
        direction = str(feature.get("direction") or "incoming").lower()
        geometry_set = feature.get("geometry_set")
        if signal == "motion_normal_displacement":
            geometry_set = validate_name(str(geometry_set or ""), kind="boundary geometry set")
            geometry_schema = _product_schema(context, "geometry", geometry_set)
            _validate_product_source(
                geometry_schema,
                key="selected_source_ids",
                context=context,
                product=f"Geometry set {geometry_set!r}",
            )
        else:
            geometry_schema = None
        signal_dependency.update(
            motion_set=motion_set,
            direction=direction,
            geometry_set=None if geometry_set in (None, "") else str(geometry_set),
            geometry_schema=geometry_schema,
            schema=_motion_product(context, motion_set=motion_set)["schema"],
        )
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for ell in range(order + 1):
        column = _slug(f"{prefix}_l{ell}")
        columns[column] = {
            "name": column,
            "dtype": "float64",
            "family": "boundary_multipole",
            "statistic": f"multipole_l{ell}",
            "multipole_order": ell,
            "maximum_order": order,
            "multipole_basis": "centered_rotationally_invariant_angular_power",
            "boundary_dependency": dict(context["dependency"]),
            "signal_dependency": signal_dependency,
        }
    values_by_label: dict[int, dict[str, float]] = {}
    spatial_ndim = int(context["view"].schema.get("spatial_ndim", 3))
    for entity in context["entities"]:
        points = context["view"].read_points(
            int(entity["boundary_entity_id"]), fields=("native_position_zyx",)
        )["native_position_zyx"]
        points = np.asarray(points, dtype=float)
        if signal == "geometry":
            charge = context["view"].geometry(
                signal_dependency["geometry_set"],
                int(entity["boundary_entity_id"]),
                fields=(signal_dependency["geometry_field"],),
            ).get(signal_dependency["geometry_field"])
            if charge is None:
                raise KeyError(f"Geometry field {signal_dependency['geometry_field']!r} is not stored")
        elif signal == "neighbor_distance":
            charge, _targets, _target_distances = _neighbor_point_distances(
                context,
                neighbor_set=signal_dependency["neighbor_set"],
                entity=entity,
                np=np,
            )
            transform = signal_dependency["distance_transform"]
            if transform == "negative_distance":
                charge = -charge
            elif transform == "inverse_distance":
                charge = 1.0 / (charge + float(signal_dependency["distance_epsilon"]))
            elif transform == "contact_indicator":
                charge = np.where(
                    np.isfinite(charge),
                    (charge <= float(signal_dependency["contact_distance"])).astype(float),
                    np.nan,
                )
        elif signal.startswith("motion_"):
            motion_data = _motion_point_data(
                context,
                motion_set=signal_dependency["motion_set"],
                entity=entity,
                direction=signal_dependency["direction"],
                geometry_set=signal_dependency.get("geometry_set"),
                np=np,
            )
            charge = (
                motion_data["normal_displacement"]
                if signal == "motion_normal_displacement"
                else np.linalg.norm(motion_data["vectors"], axis=1)
            )
        else:
            radius = np.linalg.norm(points - np.mean(points, axis=0, keepdims=True), axis=1)
            charge = radius - np.nanmean(radius) if signal == "shape_radial_deviation" else radius
        moments = boundary_multipole_magnitudes(
            points,
            charge,
            order=order,
            spatial_ndim=spatial_ndim,
        )
        label_id = int(entity["source_label_id"])
        values_by_label[label_id] = {
            _slug(f"{prefix}_l{ell}"): float(moments[ell]) for ell in range(order + 1)
        }
    warnings = [] if len(context["entities"]) else [f"No selected boundary entities in frame {context['frame']}"]
    return {"values_by_label": values_by_label, "columns": columns, "warnings": warnings}


def _neighbor_point_distances(
    context: Mapping[str, Any],
    *,
    neighbor_set: str,
    entity: Any,
    np: Any,
) -> tuple[Any, list[Any], list[Any]]:
    span = context["view"].point_slice(int(entity["boundary_entity_id"]))
    count = int(span.stop - span.start)
    distances = np.full(count, np.nan, dtype=float)
    targets: list[Any] = [np.empty(0, dtype=np.int64) for _ in range(count)]
    target_distances: list[Any] = [np.empty(0, dtype=float) for _ in range(count)]
    edges = context["view"].neighbor_edges(neighbor_set, int(entity["boundary_entity_id"]))
    source_rows = np.asarray(edges["source_point_rows"], dtype=np.int64)
    edge_distances = np.asarray(edges["distance"], dtype=float)
    target_rows = np.asarray(edges["target_point_rows"], dtype=np.int64)
    if not source_rows.size:
        return distances, targets, target_distances
    local = source_rows - int(span.start)
    local_rows, starts = np.unique(local, return_index=True)
    stops = np.concatenate((starts[1:], np.asarray([local.size], dtype=starts.dtype)))
    for local_row, edge_start, edge_stop in zip(local_rows, starts, stops):
        row_distances = edge_distances[int(edge_start) : int(edge_stop)]
        if not row_distances.size:
            continue
        minimum = float(np.nanmin(row_distances))
        distances[int(local_row)] = minimum
        targets[int(local_row)] = target_rows[int(edge_start) : int(edge_stop)]
        target_distances[int(local_row)] = row_distances
    return distances, targets, target_distances


def _motion_product(context: Mapping[str, Any], *, motion_set: str) -> dict[str, Any]:
    key = (context["boundary_set"], motion_set)
    products = context["cache"].setdefault("boundary_motion_products", {})
    if key not in products:
        path = f"boundaries/{context['boundary_set']}/motion/{motion_set}"
        group = context["trajectory"].store.h5[path]
        products[key] = {
            "path": path,
            "links": group["links"][()],
            "schema": context["trajectory"].store.read_json(f"/{path}/schema.json"),
        }
    product = products[key]
    source_id = (product["schema"].get("boundary_dependency") or {}).get("source_id")
    if source_id not in (None, "") and int(source_id) != int(context["source_id"]):
        raise ValueError(
            f"Motion set {motion_set!r} uses boundary source {source_id}, not {context['source_id']}"
        )
    object_set = (product["schema"].get("track_dependency") or {}).get("object_set")
    if object_set not in (None, "") and str(object_set) != str(context["object_set"]):
        raise ValueError(
            f"Motion set {motion_set!r} uses object set {object_set!r}, not {context['object_set']!r}"
        )
    return product


def _motion_point_data(
    context: Mapping[str, Any],
    *,
    motion_set: str,
    entity: Any,
    direction: str,
    geometry_set: str | None,
    np: Any,
) -> dict[str, Any]:
    product = _motion_product(context, motion_set=motion_set)
    direction_value = str(direction).lower()
    if direction_value not in {"incoming", "outgoing", "both"}:
        raise ValueError("Boundary motion direction must be incoming, outgoing, or both")
    entity_id = int(entity["boundary_entity_id"])
    links = product["links"]
    selected: list[tuple[Any, str]] = []
    for link in links:
        if direction_value in {"incoming", "both"} and int(link["target_entity_id"]) == entity_id:
            selected.append((link, "incoming"))
        if direction_value in {"outgoing", "both"} and int(link["source_entity_id"]) == entity_id:
            selected.append((link, "outgoing"))
    point_data = context["view"].read_points(
        entity_id, fields=("point_id", "native_position_zyx")
    )
    point_ids = np.asarray(point_data["point_id"], dtype=np.int64)
    positions = np.asarray(point_data["native_position_zyx"], dtype=float)
    weighted = np.zeros((point_ids.size, 3), dtype=float)
    weights = np.zeros(point_ids.size, dtype=float)
    path = f"{product['path']}/transport"
    group = context["trajectory"].store.h5[path]
    ot_cost: list[float] = []
    transported_mass: list[float] = []
    for link, selected_direction in selected:
        start = int(link["transport_start"])
        stop = start + int(link["transport_count"])
        id_column = "target_point_id" if selected_direction == "incoming" else "source_point_id"
        edge_ids = np.asarray(group[id_column][start:stop], dtype=np.int64)
        mass = np.asarray(group["mass"][start:stop], dtype=float)
        displacement = np.asarray(group["registered_displacement_zyx"][start:stop], dtype=float)
        local = np.searchsorted(point_ids, edge_ids)
        valid = (local >= 0) & (local < point_ids.size)
        valid &= point_ids[np.clip(local, 0, max(0, point_ids.size - 1))] == edge_ids if point_ids.size else False
        if np.any(valid):
            np.add.at(weighted, local[valid], mass[valid, None] * displacement[valid])
            np.add.at(weights, local[valid], mass[valid])
        ot_cost.append(float(link["ot_cost"]))
        transported_mass.append(float(link["transported_mass"]))
    vectors = np.full((point_ids.size, 3), np.nan, dtype=float)
    mapped = weights > 0
    vectors[mapped] = weighted[mapped] / weights[mapped, None]
    result: dict[str, Any] = {
        "positions": positions,
        "vectors": vectors,
        "point_mass": weights,
        "ot_cost": np.asarray(ot_cost, dtype=float),
        "transported_mass": np.asarray(transported_mass, dtype=float),
        "link_count": len(selected),
    }
    if geometry_set:
        normals = context["view"].geometry(
            geometry_set, entity_id, fields=("normals_zyx",)
        ).get("normals_zyx")
        if normals is None:
            raise KeyError(f"Geometry set {geometry_set!r} does not contain normals_zyx")
        normals = np.asarray(normals, dtype=float)
        normal = np.sum(vectors * normals, axis=1)
        tangential_vector = vectors - normal[:, None] * normals
        result["normal_displacement"] = normal
        result["tangential_magnitude"] = np.linalg.norm(tangential_vector, axis=1)
    else:
        result["normal_displacement"] = np.full(point_ids.size, np.nan, dtype=float)
        result["tangential_magnitude"] = np.full(point_ids.size, np.nan, dtype=float)
    return result


def _point_entity_ids(context: Mapping[str, Any], *, np: Any) -> Any:
    key = context["boundary_set"]
    cache = context["cache"].setdefault("boundary_point_entity_ids", {})
    if key not in cache:
        cache[key] = np.asarray(
            context["view"].read_points(fields=("boundary_entity_id",))["boundary_entity_id"],
            dtype=np.int64,
        )
    return cache[key]


def _product_schema(context: Mapping[str, Any], family: str, name: str) -> dict[str, Any]:
    key = (context["boundary_set"], family, name)
    schemas = context["cache"].setdefault("boundary_product_schemas", {})
    if key not in schemas:
        schemas[key] = context["trajectory"].store.read_json(
            f"/boundaries/{context['boundary_set']}/{family}/{name}/schema.json"
        )
    return dict(schemas[key])


def _validate_product_source(
    schema: Mapping[str, Any],
    *,
    key: str,
    context: Mapping[str, Any],
    product: str,
) -> None:
    values = schema.get(key, "all")
    if values == "all":
        return
    selected = {int(value) for value in values}
    if int(context["source_id"]) not in selected:
        raise ValueError(
            f"{product} does not contain boundary source {context['source_id']}"
        )


def _statistic(values: Any, statistic: str, *, np: Any) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(np.nan)
    name = str(statistic).lower()
    if name == "mean":
        return float(np.mean(finite))
    if name == "std":
        return float(np.std(finite))
    if name == "median":
        return float(np.median(finite))
    if name == "min":
        return float(np.min(finite))
    if name == "max":
        return float(np.max(finite))
    if name == "sum":
        return float(np.sum(finite))
    if name.startswith("percentile_"):
        return float(np.percentile(finite, float(name.rsplit("_", 1)[-1])))
    raise ValueError(f"Unsupported boundary statistic: {statistic!r}")


def _summary_column(prefix: str, field: str, statistic: str, *, field_count: int) -> str:
    if field_count == 1:
        return _slug(f"{prefix}_{statistic}")
    return _slug(f"{prefix}_{field}_{statistic}")


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _slug(value: Any) -> str:
    import re

    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_") or "boundary_feature"
    return f"feature_{text}" if text[0].isdigit() else text


__all__ = ["boundary_multipole_magnitudes", "compute_boundary_feature_frame"]
