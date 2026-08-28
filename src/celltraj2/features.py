"""Single-object feature extraction for celltraj2 object sets."""

from __future__ import annotations

import re
import warnings as python_warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

from celltraj2.paths import validate_name
from celltraj2.registration import registration_calibration
from celltraj2.schema import utc_now_iso


INTENSITY_SCALAR_STATISTICS = (
    "mean",
    "sum",
    "median",
    "min",
    "max",
    "std",
    "area",
)
INTENSITY_PERCENTILE_LEVELS = (
    0,
    1,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    99,
    100,
)
INTENSITY_PERCENTILE_STATISTICS = tuple(
    f"percentile_{level}" for level in INTENSITY_PERCENTILE_LEVELS
)
INTENSITY_STATISTIC_OPTIONS = (*INTENSITY_SCALAR_STATISTICS, "percentiles")

REGIONPROPS_SCALAR_PROPERTIES = (
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "solidity",
    "eccentricity",
    "orientation",
    "perimeter",
    "perimeter_crofton",
)

MASK_COMPONENT_METRICS = (
    "mask_area",
    "mask_fraction",
    "component_count",
    "component_density",
    "component_area_mean",
    "component_area_median",
    "component_area_std",
    "component_area_cv",
    "component_area_min",
    "component_area_max",
    "largest_component_fraction",
    "component_equivalent_diameter_mean",
    "component_axis_major_length_mean",
    "component_axis_minor_length_mean",
    "component_elongation_mean",
    "component_roundness_mean",
    "component_extent_mean",
    "component_solidity_mean",
)

DEFAULT_MASK_COMPONENT_METRICS = (
    "mask_area",
    "mask_fraction",
    "component_count",
    "component_area_mean",
    "component_area_std",
    "largest_component_fraction",
    "component_elongation_mean",
    "component_roundness_mean",
)


def expand_intensity_statistics(stats: Sequence[str] | str | None = None) -> list[str]:
    """Expand and validate intensity statistics in stable output-column order."""

    if stats is None:
        requested = ["mean"]
    else:
        requested = [stats] if isinstance(stats, str) else list(stats)
    expanded: list[str] = []
    for item in requested:
        name = str(item).strip().lower()
        if not name:
            continue
        values = INTENSITY_PERCENTILE_STATISTICS if name == "percentiles" else (name,)
        for statistic in values:
            if statistic in INTENSITY_SCALAR_STATISTICS:
                pass
            elif statistic.startswith("percentile_"):
                try:
                    percentile = float(statistic.rsplit("_", 1)[-1])
                except ValueError as exc:
                    raise ValueError(f"Unsupported intensity statistic: {statistic}") from exc
                if not 0.0 <= percentile <= 100.0:
                    raise ValueError(f"Intensity percentile must be within 0..100: {statistic}")
            else:
                raise ValueError(f"Unsupported intensity statistic: {statistic}")
            if statistic not in expanded:
                expanded.append(statistic)
    if not expanded:
        raise ValueError("Intensity feature requires at least one statistic.")
    return expanded


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "celltraj2 feature extraction requires numpy. Install with "
            "`python -m pip install -e .[analysis]`."
        ) from exc
    return np


def _require_regionprops_table() -> Any:
    try:
        from skimage.measure import regionprops_table  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "regionprops features require scikit-image. Install with "
            "`python -m pip install -e .[analysis]`."
        ) from exc
    return regionprops_table


def _require_skimage_measure() -> tuple[Any, Any]:
    try:
        from skimage.measure import label, regionprops  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "mask-component features require scikit-image. Install with "
            "`python -m pip install -e .[analysis]`."
        ) from exc
    return label, regionprops


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


def _slug(value: Any, *, fallback: str = "feature") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"{fallback}_{text}"
    return text


@dataclass(frozen=True)
class FeatureSetSpec:
    """Declarative specification for one row-aligned object feature set."""

    feature_set: str
    object_set: str
    features: list[dict[str, Any]] = field(default_factory=list)
    source_label_set: str | None = None
    frames: dict[str, Any] = field(default_factory=lambda: {"mode": "all"})
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureSetSpec":
        payload = dict(data)
        feature_set = payload.get("feature_set") or payload.get("name")
        object_set = payload.get("object_set")
        if feature_set in (None, ""):
            raise ValueError("FeatureSetSpec requires feature_set")
        if object_set in (None, ""):
            raise ValueError("FeatureSetSpec requires object_set")
        frames = payload.get("frames")
        if not isinstance(frames, Mapping):
            frames = {
                "mode": payload.get("frame_mode", "all"),
                "frame_start": payload.get("frame_start"),
                "frame_stop": payload.get("frame_stop"),
                "frame_list": payload.get("frame_list"),
            }
        return cls(
            feature_set=str(feature_set),
            object_set=str(object_set),
            features=[dict(item) for item in payload.get("features", [])],
            source_label_set=None if payload.get("source_label_set") in (None, "") else str(payload.get("source_label_set")),
            frames=dict(frames or {"mode": "all"}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Result from extracting one feature set from one trajectory H5."""

    feature_set: str
    object_set: str
    source_label_set: str
    values: Any
    schema: dict[str, Any]
    qc: dict[str, Any]
    frames: list[int]
    frame_counts: dict[int, int] = field(default_factory=dict)
    frame_warnings: dict[int, list[str]] = field(default_factory=dict)
    frame_feature_summaries: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    values_path: str | None = None
    run_id: str | None = None
    saved: bool = True

    @property
    def observation_count(self) -> int:
        return int(getattr(self.values, "shape", (0,))[0])

    @property
    def feature_count(self) -> int:
        names = list(getattr(getattr(self.values, "dtype", None), "names", None) or [])
        return len([name for name in names if name != "observation_id"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "object_set": self.object_set,
            "source_label_set": self.source_label_set,
            "frames": list(self.frames),
            "frame_counts": {str(key): int(value) for key, value in self.frame_counts.items()},
            "frame_warnings": {str(key): list(value) for key, value in self.frame_warnings.items()},
            "values_path": self.values_path,
            "observation_count": self.observation_count,
            "feature_count": self.feature_count,
            "run_id": self.run_id,
            "saved": bool(self.saved),
        }


def default_feature_extraction_run_id() -> str:
    """Return a H5-safe feature-extraction run id."""

    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "_").replace(".", "_")
    return f"features_{stamp}"


def regionprops_v1_spec(
    object_set: str,
    *,
    source_label_set: str | None = None,
    feature_set: str = "regionprops_v1",
    properties: Sequence[str] | None = None,
    use_physical_spacing: bool = False,
    frames: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeatureSetSpec:
    """Return a starter regionprops feature-set spec."""

    return FeatureSetSpec(
        feature_set=feature_set,
        object_set=object_set,
        source_label_set=source_label_set,
        frames=dict(frames or {"mode": "all"}),
        features=[
            {
                "kind": "regionprops",
                "prefix": "regionprops",
                "properties": list(properties or ("area", "equivalent_diameter_area", "extent", "solidity")),
                "use_physical_spacing": bool(use_physical_spacing),
            }
        ],
        metadata=dict(metadata or {}),
    )


def mask_components_v1_spec(
    object_set: str,
    *,
    mask_set: str,
    source_label_set: str | None = None,
    feature_set: str = "mask_components_v1",
    name: str | None = None,
    metrics: Sequence[str] | None = None,
    connectivity: int | str = 1,
    use_physical_spacing: bool = False,
    frames: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeatureSetSpec:
    """Return a feature spec for a binary mask restricted to each labeled object."""

    return FeatureSetSpec(
        feature_set=feature_set,
        object_set=object_set,
        source_label_set=source_label_set,
        frames=dict(frames or {"mode": "all"}),
        features=[
            {
                "kind": "mask_components",
                "name": _slug(name or mask_set, fallback="mask"),
                "mask_set": str(mask_set),
                "metrics": list(metrics or DEFAULT_MASK_COMPONENT_METRICS),
                "connectivity": connectivity,
                "use_physical_spacing": bool(use_physical_spacing),
            }
        ],
        metadata=dict(metadata or {}),
    )


def site_signaling_v1_spec(
    object_set: str,
    *,
    signal_channel: Mapping[str, Any] | int | str,
    nuclear_mask_set: str = "nuc",
    nuclear_source_kind: str = "mask",
    source_label_set: str | None = None,
    feature_set: str = "site_v1",
    ratio_order: str = "cyto_over_nuc",
    name: str = "site",
    background: Mapping[str, Any] | None = None,
    frames: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeatureSetSpec:
    """Return a starter intracellular signaling feature-set spec."""

    channel = _channel_selector_payload(signal_channel)
    nuclear_key = "include_label_set" if str(nuclear_source_kind).lower() == "label" else "include_mask_set"
    nuclear_exclude_key = "exclude_label_set" if str(nuclear_source_kind).lower() == "label" else "exclude_mask_set"
    cyto = {"label_set": source_label_set or object_set, nuclear_exclude_key: nuclear_mask_set, "name": "cyto_excluding_nuc"}
    nuc = {"label_set": source_label_set or object_set, nuclear_key: nuclear_mask_set, "name": "nuc"}
    numerator = nuc if str(ratio_order).lower() in {"n_over_c", "nuc_over_cyto", "nucleus_over_cytoplasm"} else cyto
    denominator = cyto if numerator is nuc else nuc
    feature_name = _slug(name or "site", fallback="site")
    background_spec = dict(background) if isinstance(background, Mapping) else None
    intensity_cyto = {
        "kind": "intensity",
        "name": f"{feature_name}_cyto",
        "channel": channel,
        "compartment": cyto,
        "stats": ["mean"],
    }
    intensity_nuc = {
        "kind": "intensity",
        "name": f"{feature_name}_nuc",
        "channel": channel,
        "compartment": nuc,
        "stats": ["mean"],
    }
    ratio = {
        "kind": "compartment_ratio",
        "name": f"{feature_name}_ratio",
        "channel": channel,
        "numerator": numerator,
        "denominator": denominator,
        "stat": "mean",
    }
    if background_spec is not None:
        intensity_cyto["background"] = dict(background_spec)
        intensity_nuc["background"] = dict(background_spec)
        ratio["background"] = dict(background_spec)
    return FeatureSetSpec(
        feature_set=feature_set,
        object_set=object_set,
        source_label_set=source_label_set,
        frames=dict(frames or {"mode": "all"}),
        features=[intensity_cyto, intensity_nuc, ratio],
        metadata=dict(metadata or {}),
    )


def extract_feature_set(
    trajectory: Any,
    spec: FeatureSetSpec | Mapping[str, Any],
    *,
    frames: Sequence[int] | None = None,
    overwrite: bool = False,
    save_outputs: bool = True,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> FeatureExtractionResult:
    """Extract one row-aligned object feature set."""

    np = _require_numpy()
    feature_spec = spec if isinstance(spec, FeatureSetSpec) else FeatureSetSpec.from_dict(spec)
    object_name = validate_name(feature_spec.object_set, kind="object set")
    feature_name = validate_name(feature_spec.feature_set, kind="feature set")
    if save_outputs and trajectory.store.has_feature_set(object_name, feature_name) and not overwrite:
        raise FileExistsError(f"/object_sets/{object_name}/features/{feature_name}")
    if not trajectory.store.has_observations(object_name):
        raise FileNotFoundError(f"/object_sets/{object_name}/observations")

    observations = trajectory.store.read_observations(object_name)
    source_label_set = _source_label_set(trajectory, object_name, feature_spec.source_label_set)
    selected_frames = _selected_frames(
        frames,
        feature_spec.frames,
        frame_count=int(trajectory.metadata.frame_count or 1),
        available_frames=trajectory.object_set(object_name).lookup_frames() or trajectory.label_frames(source_label_set),
    )
    row_values: list[dict[str, float]] = [dict() for _ in range(int(observations.shape[0]))]
    column_schemas: OrderedDict[str, dict[str, Any]] = OrderedDict()
    frame_counts: dict[int, int] = {}
    frame_warnings: dict[int, list[str]] = {}
    frame_feature_summaries: dict[int, list[dict[str, Any]]] = {}
    boundary_feature_cache: dict[str, Any] = {}
    run_name = validate_name(run_id or default_feature_extraction_run_id(), kind="feature-extraction run")

    if save_outputs:
        run_record = {
            "schema": "celltraj2.feature_extraction_run.v1",
            "run_id": run_name,
            "job_id": run_name,
            "status": "running",
            "started_at": utc_now_iso(),
            "h5_path": str(trajectory.path),
            "roi_id": trajectory.metadata.roi_id,
            "dataset_id": trajectory.metadata.dataset_id,
            "object_set": object_name,
            "source_label_set": source_label_set,
            "feature_set": feature_name,
            "frames": selected_frames,
            "overwrite": bool(overwrite),
            "save_outputs": True,
            "feature_spec": feature_spec.to_dict(),
            "metadata": _json_safe(dict(metadata or {})),
        }
        trajectory.store.write_feature_extraction_run(run_name, run_record, overwrite=True)

    for frame in selected_frames:
        if progress is not None:
            progress({"event": "frame_started", "frame": int(frame), "feature_set": feature_name})
        labels = np.asarray(trajectory.read_label_frame(source_label_set, frame))
        lookup = trajectory.object_set(object_name).read_lookup_frame(frame)
        frame_rows_touched: set[int] = set()
        frame_observation_ids = _frame_observation_ids(labels, lookup, np=np)
        feature_summaries: list[dict[str, Any]] = []
        frame_columns_seen: set[str] = set()
        warnings: list[str] = []
        for feature in feature_spec.features:
            feature_kind = str(feature.get("kind") or feature.get("type") or "feature")
            feature_label = str(feature.get("name") or feature.get("prefix") or feature_kind)
            if progress is not None:
                progress(
                    {
                        "event": "feature_started",
                        "frame": int(frame),
                        "feature_set": feature_name,
                        "feature_kind": feature_kind,
                        "feature_name": feature_label,
                    }
                )
            result = _compute_feature_frame(
                trajectory,
                labels,
                frame=frame,
                source_label_set=source_label_set,
                feature=feature,
                object_set=object_name,
                boundary_cache=boundary_feature_cache,
                np=np,
            )
            warnings.extend(result["warnings"])
            for column, schema in result["columns"].items():
                if column in frame_columns_seen:
                    raise ValueError(
                        f"Duplicate feature column name {column!r}. Rename the feature blocks so every output column is unique."
                    )
                frame_columns_seen.add(column)
                if column not in column_schemas:
                    column_schemas[column] = schema
            feature_summaries.extend(
                _feature_frame_summaries(
                    result,
                    lookup,
                    frame_observation_ids=frame_observation_ids,
                    np=np,
                )
            )
            if progress is not None:
                progress(
                    {
                        "event": "feature_completed",
                        "frame": int(frame),
                        "feature_set": feature_name,
                        "feature_kind": feature_kind,
                        "feature_name": feature_label,
                        "columns": list(result["columns"].keys()),
                        "warnings": list(result["warnings"]),
                    }
                )
            for label_id, values in result["values_by_label"].items():
                observation_id = _lookup_observation_id(lookup, int(label_id))
                if observation_id < 1:
                    continue
                row_index = observation_id - 1
                if row_index >= len(row_values):
                    continue
                row_values[row_index].update(values)
                frame_rows_touched.add(row_index)
        frame_counts[frame] = len(frame_rows_touched)
        frame_warnings[frame] = warnings
        frame_feature_summaries[frame] = feature_summaries
        if progress is not None:
            progress(
                {
                    "event": "frame_completed",
                    "frame": int(frame),
                    "feature_set": feature_name,
                    "value_count": int(len(frame_rows_touched)),
                    "feature_summaries": feature_summaries,
                    "warnings": warnings,
                }
            )
        if save_outputs:
            trajectory.store.write_feature_extraction_frame_result(
                run_name,
                frame,
                {
                    "frame": int(frame),
                    "status": "completed",
                    "object_set": object_name,
                    "source_label_set": source_label_set,
                    "feature_set": feature_name,
                    "feature_columns": list(column_schemas.keys()),
                    "value_count": int(len(frame_rows_touched)),
                    "feature_summaries": feature_summaries,
                    "warnings": warnings,
                },
                overwrite=True,
            )

    values = _structured_values(observations, row_values, column_schemas, np=np)
    schema = _feature_schema(
        feature_name=feature_name,
        object_name=object_name,
        source_label_set=source_label_set,
        frames=selected_frames,
        observations=observations,
        columns=column_schemas,
        feature_spec=feature_spec,
        metadata={**feature_spec.metadata, **dict(metadata or {})},
    )
    qc = _feature_qc(values, frame_counts, frame_warnings, np=np)
    values_path = None
    if save_outputs:
        values_path = trajectory.store.write_feature_set(
            object_name,
            feature_name,
            values,
            schema,
            overwrite=overwrite,
            qc=qc,
        )
        run_record["status"] = "completed"
        run_record["completed_at"] = utc_now_iso()
        run_record["values_path"] = values_path
        run_record["feature_count"] = len(column_schemas)
        run_record["observation_count"] = int(values.shape[0])
        run_record["frame_counts"] = {str(key): int(value) for key, value in frame_counts.items()}
        run_record["frame_warnings"] = {str(key): list(value) for key, value in frame_warnings.items()}
        trajectory.store.write_feature_extraction_run(run_name, run_record, overwrite=True)

    return FeatureExtractionResult(
        feature_set=feature_name,
        object_set=object_name,
        source_label_set=source_label_set,
        values=values,
        schema=schema,
        qc=qc,
        frames=selected_frames,
        frame_counts=frame_counts,
        frame_warnings=frame_warnings,
        frame_feature_summaries=frame_feature_summaries,
        values_path=values_path,
        run_id=run_name,
        saved=bool(save_outputs),
    )


def _compute_feature_frame(
    trajectory: Any,
    labels: Any,
    *,
    frame: int,
    source_label_set: str,
    feature: Mapping[str, Any],
    object_set: str,
    boundary_cache: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    kind = str(feature.get("kind") or feature.get("type") or "").lower()
    if kind == "regionprops":
        return _compute_regionprops(trajectory, labels, feature=feature, np=np)
    if kind in {"mask_components", "mask_within_objects"}:
        return _compute_mask_components(
            trajectory,
            labels,
            frame=frame,
            feature=feature,
            np=np,
        )
    if kind == "intensity":
        return _compute_intensity(trajectory, labels, frame=frame, source_label_set=source_label_set, feature=feature, np=np)
    if kind in {"compartment_ratio", "ratio"}:
        return _compute_compartment_ratio(
            trajectory,
            labels,
            frame=frame,
            source_label_set=source_label_set,
            feature=feature,
            np=np,
        )
    if kind in {"channel_correlation", "correlation", "crosscorr"}:
        return _compute_channel_correlation(
            trajectory,
            labels,
            frame=frame,
            source_label_set=source_label_set,
            feature=feature,
            np=np,
        )
    if kind in {
        "boundary_geometry",
        "surface_geometry",
        "boundary_interaction",
        "surface_interaction",
        "boundary_contact",
        "boundary_motion",
        "surface_motion",
        "boundary_multipole",
        "surface_multipole",
    }:
        from celltraj2.boundary_features import compute_boundary_feature_frame

        return compute_boundary_feature_frame(
            trajectory,
            object_set=object_set,
            frame=frame,
            feature=feature,
            cache=boundary_cache,
            np=np,
        )
    raise ValueError(f"Unsupported feature kind: {kind!r}")


def _compute_regionprops(
    trajectory: Any,
    labels: Any,
    *,
    feature: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    regionprops_table = _require_regionprops_table()
    properties = [str(item) for item in feature.get("properties", ["area"])]
    prefix = _slug(feature.get("prefix") or "regionprops")
    values_by_label: dict[int, dict[str, float]] = {}
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    warnings: list[str] = []
    label_image = np.asarray(labels)
    spacing, calibration_schema = _feature_spacing(
        trajectory,
        label_image,
        enabled=_config_bool(feature.get("use_physical_spacing"), default=False),
    )
    spacing_kwargs = {} if spacing is None else {"spacing": spacing}
    for prop in properties:
        try:
            table = regionprops_table(
                label_image,
                properties=("label", prop),
                separator="_",
                **spacing_kwargs,
            )
        except Exception as exc:
            column = _slug(f"{prefix}_{prop}")
            columns[column] = {
                "name": column,
                "dtype": "float64",
                "family": "regionprops",
                "property": prop,
                "status": "unsupported",
                "warning": repr(exc),
                "unit": _regionprops_property_unit(prop, label_image.ndim, calibration_schema),
                "calibration": calibration_schema,
            }
            warnings.append(f"regionprops property {prop!r} could not be computed: {exc!r}")
            continue
        label_values = table.get("label")
        if label_values is None:
            continue
        for key, array in table.items():
            if key == "label":
                continue
            column = _slug(f"{prefix}_{key}")
            columns[column] = {
                "name": column,
                "dtype": "float64",
                "family": "regionprops",
                "property": prop,
                "regionprops_key": str(key),
                "unit": _regionprops_property_unit(prop, label_image.ndim, calibration_schema),
                "calibration": calibration_schema,
            }
            for index, label_id in enumerate(label_values):
                values_by_label.setdefault(int(label_id), {})[column] = _float_or_nan(array[index], np=np)
    return {"values_by_label": values_by_label, "columns": columns, "warnings": warnings}


def _compute_mask_components(
    trajectory: Any,
    labels: Any,
    *,
    frame: int,
    feature: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    label_components, regionprops = _require_skimage_measure()
    mask_set = str(feature.get("mask_set") or "").strip()
    if not mask_set:
        raise ValueError("mask_components feature requires mask_set")
    metrics = [str(item).strip().lower() for item in feature.get("metrics", DEFAULT_MASK_COMPONENT_METRICS)]
    unsupported = [metric for metric in metrics if metric not in MASK_COMPONENT_METRICS]
    if unsupported:
        raise ValueError(f"Unsupported mask-component metrics: {', '.join(unsupported)}")
    if not metrics:
        raise ValueError("mask_components feature requires at least one metric")

    label_image = np.asarray(labels)
    mask = _read_compartment_source_mask(
        trajectory,
        "mask",
        mask_set,
        frame,
        label_image,
        np=np,
    )
    spacing, calibration_schema = _feature_spacing(
        trajectory,
        label_image,
        enabled=_config_bool(feature.get("use_physical_spacing"), default=False),
    )
    spacing_kwargs = {} if spacing is None else {"spacing": spacing}
    connectivity = _component_connectivity(feature.get("connectivity", 1), label_image.ndim)
    prefix = _slug(feature.get("name") or mask_set, fallback="mask")
    measure_unit = _measure_unit(label_image.ndim, calibration_schema)
    length_unit = str(calibration_schema["distance_unit"])
    voxel_measure = float(np.prod(spacing)) if spacing is not None else 1.0

    columns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for metric in metrics:
        column = _slug(f"{prefix}_{metric}")
        columns[column] = {
            "name": column,
            "dtype": "float64",
            "family": "mask_components",
            "metric": metric,
            "mask_set": mask_set,
            "connectivity": connectivity,
            "unit": _mask_component_metric_unit(metric, measure_unit=measure_unit, length_unit=length_unit),
            "calibration": calibration_schema,
        }

    values_by_label: dict[int, dict[str, float]] = {}
    warnings: list[str] = []
    for cell in regionprops(label_image):
        label_id = int(cell.label)
        cell_mask = np.asarray(cell.image, dtype=bool)
        local_mask = np.logical_and(np.asarray(mask[cell.slice], dtype=bool), cell_mask)
        component_labels = label_components(local_mask, connectivity=connectivity)
        components = regionprops(component_labels, **spacing_kwargs)
        values = _mask_component_values(
            cell_mask,
            local_mask,
            components,
            metrics=metrics,
            voxel_measure=voxel_measure,
            np=np,
        )
        values_by_label[label_id] = {
            _slug(f"{prefix}_{metric}"): values[metric]
            for metric in metrics
        }
    return {"values_by_label": values_by_label, "columns": columns, "warnings": warnings}


def _feature_spacing(
    trajectory: Any,
    labels: Any,
    *,
    enabled: bool,
) -> tuple[tuple[float, ...] | None, dict[str, Any]]:
    ndim = int(getattr(labels, "ndim", 0))
    if ndim not in {2, 3}:
        raise ValueError(f"Feature extraction expects 2D or 3D labels, received {ndim}D")
    spatial_axes = ("Z", "Y", "X") if ndim == 3 else ("Y", "X")
    if not enabled:
        return None, {
            "physical_spacing_enabled": False,
            "spatial_axes": list(spatial_axes),
            "spacing": [1.0] * ndim,
            "distance_unit": "pixel" if ndim == 2 else "voxel",
            "source": "index_space",
        }
    calibration = registration_calibration(trajectory.metadata)
    if str(calibration.get("distance_unit")) != "um":
        raise ValueError(
            "Physical spacing was requested, but micron calibration is missing from H5 acquisition metadata."
        )
    scale_zyx = tuple(float(value) for value in calibration["coordinate_scale"])
    spacing = scale_zyx if ndim == 3 else scale_zyx[-2:]
    return spacing, {
        "physical_spacing_enabled": True,
        "spatial_axes": list(spatial_axes),
        "spacing": list(spacing),
        "distance_unit": "um",
        "source": str(calibration.get("source") or "h5_acquisition_metadata"),
    }


def _component_connectivity(value: Any, ndim: int) -> int:
    if str(value).strip().lower() in {"full", "max", "all"}:
        return int(ndim)
    try:
        connectivity = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid component connectivity: {value!r}") from exc
    if not 1 <= connectivity <= ndim:
        raise ValueError(f"Component connectivity must be between 1 and {ndim}, received {connectivity}")
    return connectivity


def _mask_component_values(
    cell_mask: Any,
    local_mask: Any,
    components: Sequence[Any],
    *,
    metrics: Sequence[str],
    voxel_measure: float,
    np: Any,
) -> dict[str, float]:
    cell_area = float(np.count_nonzero(cell_mask)) * voxel_measure
    mask_area = float(np.count_nonzero(local_mask)) * voxel_measure
    count = len(components)
    areas = np.asarray([float(component.area) for component in components], dtype=float)
    area_mean = _nan_stat(areas, "mean", np=np)
    values = {
        "mask_area": mask_area,
        "mask_fraction": mask_area / cell_area if cell_area > 0 else float(np.nan),
        "component_count": float(count),
        "component_density": float(count) / cell_area if cell_area > 0 else float(np.nan),
        "component_area_mean": area_mean,
        "component_area_median": _nan_stat(areas, "median", np=np),
        "component_area_std": _nan_stat(areas, "std", np=np),
        "component_area_cv": _nan_stat(areas, "std", np=np) / area_mean if area_mean > 0 else float(np.nan),
        "component_area_min": _nan_stat(areas, "min", np=np),
        "component_area_max": _nan_stat(areas, "max", np=np),
        "largest_component_fraction": float(np.nanmax(areas) / np.nansum(areas)) if areas.size and np.nansum(areas) > 0 else float(np.nan),
    }
    if "component_equivalent_diameter_mean" in metrics:
        equivalent_diameters = _component_property_values(
            components, "equivalent_diameter_area", np=np
        )
        values["component_equivalent_diameter_mean"] = _nan_stat(
            equivalent_diameters, "mean", np=np
        )
    axis_metrics = {
        "component_axis_major_length_mean",
        "component_axis_minor_length_mean",
        "component_elongation_mean",
        "component_roundness_mean",
    }
    if axis_metrics.intersection(metrics):
        major_lengths = _component_property_values(components, "axis_major_length", np=np)
        minor_lengths = _component_property_values(components, "axis_minor_length", np=np)
        values["component_axis_major_length_mean"] = _nan_stat(major_lengths, "mean", np=np)
        values["component_axis_minor_length_mean"] = _nan_stat(minor_lengths, "mean", np=np)
        elongation = np.divide(
            major_lengths,
            minor_lengths,
            out=np.full(major_lengths.shape, np.nan, dtype=float),
            where=minor_lengths > 0,
        )
        roundness = np.divide(
            minor_lengths,
            major_lengths,
            out=np.full(minor_lengths.shape, np.nan, dtype=float),
            where=major_lengths > 0,
        )
        values["component_elongation_mean"] = _nan_stat(elongation, "mean", np=np)
        values["component_roundness_mean"] = _nan_stat(roundness, "mean", np=np)
    if "component_extent_mean" in metrics:
        extents = _component_property_values(components, "extent", np=np)
        values["component_extent_mean"] = _nan_stat(extents, "mean", np=np)
    if "component_solidity_mean" in metrics:
        solidities = _component_property_values(components, "solidity", np=np)
        values["component_solidity_mean"] = _nan_stat(solidities, "mean", np=np)
    return values


def _component_property_values(components: Sequence[Any], name: str, *, np: Any) -> Any:
    values: list[float] = []
    for component in components:
        try:
            with python_warnings.catch_warnings():
                python_warnings.simplefilter("ignore")
                values.append(float(getattr(component, name)))
        except Exception:
            values.append(float(np.nan))
    return np.asarray(values, dtype=float)


def _nan_stat(values: Any, statistic: str, *, np: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float(np.nan)
    return float(getattr(np, f"nan{statistic}")(array))


def _measure_unit(ndim: int, calibration: Mapping[str, Any]) -> str:
    if calibration["distance_unit"] == "um":
        return f"um^{ndim}"
    return "pixel" if ndim == 2 else "voxel"


def _regionprops_property_unit(prop: str, ndim: int, calibration: Mapping[str, Any]) -> str:
    if prop in {"area", "area_bbox", "area_convex", "area_filled"}:
        return _measure_unit(ndim, calibration)
    if prop in {
        "axis_major_length",
        "axis_minor_length",
        "equivalent_diameter_area",
        "feret_diameter_max",
        "perimeter",
        "perimeter_crofton",
    }:
        return str(calibration["distance_unit"])
    if prop == "orientation":
        return "radian"
    return "dimensionless"


def _mask_component_metric_unit(metric: str, *, measure_unit: str, length_unit: str) -> str:
    if metric in {
        "mask_area",
        "component_area_mean",
        "component_area_median",
        "component_area_std",
        "component_area_min",
        "component_area_max",
    }:
        return measure_unit
    if metric == "component_density":
        return f"1/{measure_unit}"
    if metric in {
        "component_equivalent_diameter_mean",
        "component_axis_major_length_mean",
        "component_axis_minor_length_mean",
    }:
        return length_unit
    return "dimensionless"


def _compute_intensity(
    trajectory: Any,
    labels: Any,
    *,
    frame: int,
    source_label_set: str,
    feature: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    selector = feature.get("channel", feature.get("channel_selector", feature.get("signal_channel", 0)))
    channel = _resolve_channel(trajectory, selector)
    image = _read_channel_image(trajectory, frame, channel, np=np)
    compartment = _compartment_labels(
        trajectory,
        labels,
        frame=frame,
        source_label_set=source_label_set,
        compartment=feature.get("compartment") or {},
        np=np,
    )
    image = _match_spatial(np.asarray(image, dtype=float), compartment["labels"], np=np)
    image, background_schema, background_warnings = _apply_background(
        trajectory,
        image,
        frame=frame,
        background=feature.get("background"),
        np=np,
    )
    stats = expand_intensity_statistics(feature.get("stats", ["mean"]))
    per_label = _per_label_stats(compartment["labels"], image, stats, np=np)
    values_by_label: dict[int, dict[str, float]] = {}
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for stat in stats:
        column = _feature_column_name(
            feature,
            channel,
            compartment["name"],
            stat=stat,
            suffix=None,
            disambiguate_explicit=len(stats) > 1,
        )
        columns[column] = {
            "name": column,
            "dtype": "float64",
            "family": "intensity",
            "statistic": stat,
            "channel": channel["schema"],
            "compartment": compartment["schema"],
        }
        if background_schema is not None:
            columns[column]["background"] = background_schema
        for label_id, values in per_label.items():
            values_by_label.setdefault(label_id, {})[column] = values.get(stat, np.nan)
    return {"values_by_label": values_by_label, "columns": columns, "warnings": [*compartment["warnings"], *background_warnings]}


def _compute_compartment_ratio(
    trajectory: Any,
    labels: Any,
    *,
    frame: int,
    source_label_set: str,
    feature: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    selector = feature.get("channel", feature.get("channel_selector", feature.get("signal_channel", 0)))
    channel = _resolve_channel(trajectory, selector)
    image = _read_channel_image(trajectory, frame, channel, np=np)
    numerator = _compartment_labels(
        trajectory,
        labels,
        frame=frame,
        source_label_set=source_label_set,
        compartment=feature.get("numerator") or {},
        np=np,
    )
    denominator = _compartment_labels(
        trajectory,
        labels,
        frame=frame,
        source_label_set=source_label_set,
        compartment=feature.get("denominator") or {},
        np=np,
    )
    image_num = _match_spatial(np.asarray(image, dtype=float), numerator["labels"], np=np)
    image_den = _match_spatial(np.asarray(image, dtype=float), denominator["labels"], np=np)
    image_num, background_schema, background_warnings = _apply_background(
        trajectory,
        image_num,
        frame=frame,
        background=feature.get("background"),
        np=np,
    )
    image_den, _background_schema_den, background_warnings_den = _apply_background(
        trajectory,
        image_den,
        frame=frame,
        background=feature.get("background"),
        np=np,
    )
    stat = str(feature.get("stat") or feature.get("statistic") or "mean")
    numerator_stats = _per_label_stats(numerator["labels"], image_num, [stat], np=np)
    denominator_stats = _per_label_stats(denominator["labels"], image_den, [stat], np=np)
    column = _feature_column_name(
        feature,
        channel,
        f"{numerator['name']}_over_{denominator['name']}",
        stat=stat,
        suffix="ratio",
    )
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict(
        [
            (
                column,
                {
                    "name": column,
                    "dtype": "float64",
                    "family": "compartment_signal",
                    "statistic": stat,
                    "channel": channel["schema"],
                    "numerator": numerator["schema"],
                    "denominator": denominator["schema"],
                    "ratio_order": "numerator_over_denominator",
                    **({"background": background_schema} if background_schema is not None else {}),
                },
            )
        ]
    )
    values_by_label: dict[int, dict[str, float]] = {}
    label_ids = sorted(set(numerator_stats.keys()) | set(denominator_stats.keys()))
    for label_id in label_ids:
        num = numerator_stats.get(label_id, {}).get(stat, np.nan)
        den = denominator_stats.get(label_id, {}).get(stat, np.nan)
        if np.isnan(den) or den == 0:
            ratio = np.nan
        else:
            ratio = float(num) / float(den)
        values_by_label.setdefault(label_id, {})[column] = ratio
    return {
        "values_by_label": values_by_label,
        "columns": columns,
        "warnings": [*numerator["warnings"], *denominator["warnings"], *background_warnings, *background_warnings_den],
    }


def _compute_channel_correlation(
    trajectory: Any,
    labels: Any,
    *,
    frame: int,
    source_label_set: str,
    feature: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    channel_a = _resolve_channel(trajectory, feature.get("channel_a", feature.get("channel1", 0)))
    channel_b = _resolve_channel(trajectory, feature.get("channel_b", feature.get("channel2", 1)))
    image_a = _read_channel_image(trajectory, frame, channel_a, np=np)
    image_b = _read_channel_image(trajectory, frame, channel_b, np=np)
    compartment = _compartment_labels(
        trajectory,
        labels,
        frame=frame,
        source_label_set=source_label_set,
        compartment=feature.get("compartment") or {},
        np=np,
    )
    image_a = _match_spatial(np.asarray(image_a, dtype=float), compartment["labels"], np=np)
    image_b = _match_spatial(np.asarray(image_b, dtype=float), compartment["labels"], np=np)
    column = _slug(
        feature.get("name")
        or f"{channel_a['name']}_{channel_b['name']}_{compartment['name']}_corr"
    )
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict(
        [
            (
                column,
                {
                    "name": column,
                    "dtype": "float64",
                    "family": "channel_relationship",
                    "statistic": "pearson_correlation",
                    "channel_a": channel_a["schema"],
                    "channel_b": channel_b["schema"],
                    "compartment": compartment["schema"],
                },
            )
        ]
    )
    values_by_label: dict[int, dict[str, float]] = {}
    for label_id in [int(item) for item in np.unique(compartment["labels"]) if int(item) > 0]:
        mask = compartment["labels"] == label_id
        a_values = image_a[mask].astype(float)
        b_values = image_b[mask].astype(float)
        if a_values.size < 2 or b_values.size < 2 or np.nanstd(a_values) == 0 or np.nanstd(b_values) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(a_values, b_values)[0, 1])
        values_by_label.setdefault(label_id, {})[column] = corr
    return {"values_by_label": values_by_label, "columns": columns, "warnings": compartment["warnings"]}


def _source_label_set(trajectory: Any, object_set: str, explicit: str | None) -> str:
    if explicit:
        return validate_name(explicit, kind="label set")
    try:
        schema = trajectory.store.read_observations_schema(object_set)
        value = schema.get("source_label_set") or object_set
    except Exception:
        value = object_set
    return validate_name(str(value), kind="label set")


def _selected_frames(
    explicit: Sequence[int] | None,
    frame_spec: Mapping[str, Any] | None,
    *,
    frame_count: int,
    available_frames: Sequence[int] | None = None,
) -> list[int]:
    if explicit is not None:
        values = [int(frame) for frame in explicit]
    else:
        spec = dict(frame_spec or {"mode": "all"})
        mode = str(spec.get("mode") or "all").lower()
        if spec.get("frames") is not None:
            values = _parse_frame_values(spec.get("frames"))
        elif mode == "list":
            values = _parse_frame_values(spec.get("frame_list", ""))
        elif mode == "range":
            start = int(spec.get("frame_start") or 1)
            stop = int(spec.get("frame_stop") or frame_count)
            values = list(range(start, stop + 1))
        elif available_frames is not None:
            values = [int(frame) for frame in available_frames]
        else:
            values = list(range(1, int(frame_count) + 1))
    invalid = [frame for frame in values if int(frame) < 1 or int(frame) > int(frame_count)]
    if invalid:
        raise ValueError(f"Frame(s) outside 1..{int(frame_count)}: {invalid}")
    return sorted(set(int(frame) for frame in values))


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


def _lookup_observation_id(lookup: Any, label_id: int) -> int:
    if label_id < 0 or label_id >= int(getattr(lookup, "shape", (0,))[0]):
        return 0
    return int(lookup[label_id])


def _frame_observation_ids(labels: Any, lookup: Any, *, np: Any) -> list[int]:
    observation_ids: set[int] = set()
    for label_id in [int(item) for item in np.unique(labels) if int(item) > 0]:
        observation_id = _lookup_observation_id(lookup, label_id)
        if observation_id > 0:
            observation_ids.add(observation_id)
    return sorted(observation_ids)


def _feature_frame_summaries(
    result: Mapping[str, Any],
    lookup: Any,
    *,
    frame_observation_ids: Sequence[int],
    np: Any,
) -> list[dict[str, Any]]:
    observation_index = {int(observation_id): index for index, observation_id in enumerate(frame_observation_ids)}
    object_count = len(observation_index)
    summaries: list[dict[str, Any]] = []
    for column, schema in result["columns"].items():
        values = np.full(object_count, np.nan, dtype=float)
        for label_id, row_values in result["values_by_label"].items():
            observation_id = _lookup_observation_id(lookup, int(label_id))
            index = observation_index.get(observation_id)
            if index is None or column not in row_values:
                continue
            values[index] = _float_or_nan(row_values[column], np=np)
        finite_values = values[np.isfinite(values)]
        summaries.append(
            {
                "feature": str(column),
                "column": str(column),
                "family": str(schema.get("family") or ""),
                "statistic": str(schema.get("statistic") or ""),
                "object_count": int(object_count),
                "nan_count": int(np.isnan(values).sum()),
                "finite_count": int(finite_values.size),
                "mean": None if finite_values.size == 0 else float(np.nanmean(finite_values)),
            }
        )
    return summaries


def _channel_selector_payload(selector: Mapping[str, Any] | int | str) -> dict[str, Any]:
    if isinstance(selector, Mapping):
        return dict(selector)
    if isinstance(selector, int):
        return {"raw_index": int(selector)}
    text = str(selector)
    if text.isdigit():
        return {"raw_index": int(text)}
    return {"name": text}


def _resolve_channel(trajectory: Any, selector: Any) -> dict[str, Any]:
    payload = _channel_selector_payload(selector)
    channels = list(getattr(trajectory.metadata, "channels", []) or [])
    matched = None
    if "raw_index" in payload and payload["raw_index"] is not None:
        raw_index = int(payload["raw_index"])
        for channel in channels:
            if int(channel.raw_index) == raw_index:
                matched = channel
                break
        if matched is None:
            matched = {"raw_index": raw_index}
    else:
        wanted = {str(value).strip().lower() for value in payload.values() if value not in (None, "")}
        fields = ("raw_name", "display_name", "role", "target", "readout", "fluorophore", "category")
        for channel in channels:
            channel_values = {str(getattr(channel, field) or "").strip().lower() for field in fields}
            if wanted & channel_values:
                matched = channel
                break
        if matched is None:
            raise ValueError(f"Could not resolve channel selector: {payload}")

    if isinstance(matched, Mapping):
        raw_index = int(matched.get("raw_index", 0))
        schema = dict(matched)
    else:
        raw_index = int(matched.raw_index)
        schema = matched.to_dict()
    channel_map = trajectory.channel_index_map() or {}
    local_index = int(channel_map.get(raw_index, raw_index))
    name = _channel_name(schema, raw_index)
    schema["local_index"] = local_index
    return {"raw_index": raw_index, "local_index": local_index, "name": name, "schema": schema}


def _channel_name(schema: Mapping[str, Any], raw_index: int) -> str:
    for key in ("readout", "target", "display_name", "raw_name", "fluorophore", "role"):
        value = schema.get(key)
        if value not in (None, ""):
            return _slug(value, fallback="channel")
    return f"ch{int(raw_index)}"


def _read_channel_image(trajectory: Any, frame: int, channel: Mapping[str, Any], *, np: Any) -> Any:
    image = trajectory.get_image_data(frame=frame, channels=int(channel["local_index"]))
    return np.asarray(image)


def _compartment_labels(
    trajectory: Any,
    source_labels: Any,
    *,
    frame: int,
    source_label_set: str,
    compartment: Any,
    np: Any,
) -> dict[str, Any]:
    spec = dict(compartment or {})
    label_set = validate_name(str(spec.get("label_set") or source_label_set), kind="label set")
    labels = np.asarray(source_labels if label_set == source_label_set else trajectory.read_label_frame(label_set, frame))
    labels = np.asarray(labels).copy()
    warnings: list[str] = []
    include_mask = spec.get("include_mask_set") or spec.get("mask_set")
    include_label = spec.get("include_label_set")
    exclude_mask = spec.get("exclude_mask_set")
    exclude_label = spec.get("exclude_label_set")
    if include_mask not in (None, ""):
        mask = _read_compartment_source_mask(trajectory, "mask", str(include_mask), frame, labels, np=np)
        labels = np.where(mask > 0, labels, 0)
    if include_label not in (None, ""):
        mask = _read_compartment_source_mask(trajectory, "label", str(include_label), frame, labels, np=np)
        labels = np.where(mask > 0, labels, 0)
    if exclude_mask not in (None, ""):
        mask = _read_compartment_source_mask(trajectory, "mask", str(exclude_mask), frame, labels, np=np)
        labels = np.where(mask > 0, 0, labels)
    if exclude_label not in (None, ""):
        mask = _read_compartment_source_mask(trajectory, "label", str(exclude_label), frame, labels, np=np)
        labels = np.where(mask > 0, 0, labels)
    name = spec.get("name")
    if name in (None, ""):
        if include_mask not in (None, ""):
            name = str(include_mask)
        elif include_label not in (None, ""):
            name = str(include_label)
        elif exclude_mask not in (None, ""):
            name = f"{label_set}_excluding_{exclude_mask}"
        elif exclude_label not in (None, ""):
            name = f"{label_set}_excluding_{exclude_label}"
        else:
            name = label_set
    schema = {
        "label_set": label_set,
        "include_mask_set": None if include_mask in (None, "") else str(include_mask),
        "include_label_set": None if include_label in (None, "") else str(include_label),
        "exclude_mask_set": None if exclude_mask in (None, "") else str(exclude_mask),
        "exclude_label_set": None if exclude_label in (None, "") else str(exclude_label),
        "name": _slug(name, fallback="compartment"),
    }
    return {"labels": labels, "name": schema["name"], "schema": schema, "warnings": warnings}


def _read_compartment_source_mask(
    trajectory: Any,
    source_kind: str,
    source_name: str,
    frame: int,
    reference: Any,
    *,
    np: Any,
) -> Any:
    if source_kind == "label":
        data = trajectory.read_label_frame(source_name, frame)
    elif source_kind == "mask":
        data = trajectory.read_mask_frame(source_name, frame)
    else:
        raise ValueError(f"Unsupported compartment source kind: {source_kind!r}")
    return _match_spatial(np.asarray(data), reference, np=np) > 0


def _match_spatial(data: Any, reference: Any, *, np: Any) -> Any:
    array = np.asarray(data)
    ref = np.asarray(reference)
    if array.shape == ref.shape:
        return array
    squeezed = np.squeeze(array)
    if squeezed.shape == ref.shape:
        return squeezed
    if ref.ndim == 3 and squeezed.ndim == 2 and ref.shape[0] == 1:
        return squeezed[np.newaxis, :, :]
    if ref.ndim == 2 and squeezed.ndim == 3 and squeezed.shape[0] == 1:
        return squeezed[0]
    raise ValueError(f"Cannot match spatial shape {array.shape} to label shape {ref.shape}")


def _apply_background(trajectory: Any, image: Any, *, frame: int, background: Any, np: Any) -> tuple[Any, dict[str, Any] | None, list[str]]:
    if not isinstance(background, Mapping):
        return image, None, []
    if not _config_bool(background.get("enabled", True), default=True):
        return image, None, []
    source = _background_source(background)
    if source is None:
        return image, None, []
    source_kind, source_name = source
    image_array = np.asarray(image, dtype=float)
    mask = _read_compartment_source_mask(trajectory, source_kind, source_name, frame, image_array, np=np)
    region = _background_region(background)
    if region == "inverse":
        mask = np.logical_not(mask)
    values = image_array[mask > 0]
    finite_values = values[np.isfinite(values)]
    if finite_values.size > 0:
        values = finite_values
    schema = _background_schema(background, source_kind=source_kind, source_name=source_name, region=region)
    if values.size == 0:
        return image_array, schema, [f"background source {source_kind}:{source_name} selected no pixels for frame {frame}"]
    mode = str(schema["mode"]).lower()
    if mode == "mean":
        baseline = float(np.nanmean(values))
    elif mode == "median":
        baseline = float(np.nanmedian(values))
    else:
        percentile = float(schema.get("percentile", 1))
        baseline = float(np.nanpercentile(values, percentile))
    if not np.isfinite(baseline):
        return image_array, schema, [f"background source {source_kind}:{source_name} yielded a non-finite baseline for frame {frame}"]
    return image_array - baseline, schema, []


def _config_bool(value: Any, *, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _background_source(background: Mapping[str, Any]) -> tuple[str, str] | None:
    source_kind = background.get("source_kind") or background.get("kind")
    source_name = background.get("source_name") or background.get("name")
    if source_kind not in (None, "") and source_name not in (None, ""):
        kind = str(source_kind).strip().lower()
        if kind not in {"mask", "label"}:
            raise ValueError(f"Unsupported background source kind: {source_kind!r}")
        return kind, str(source_name)
    for key in ("mask_set", "background_mask_set"):
        if background.get(key) not in (None, ""):
            return "mask", str(background.get(key))
    for key in ("label_set", "background_label_set"):
        if background.get(key) not in (None, ""):
            return "label", str(background.get(key))
    return None


def _background_region(background: Mapping[str, Any]) -> str:
    region = str(background.get("region") or background.get("background_region") or "").strip().lower()
    inverse_region_names = {"inverse", "outside", "opposite", "not_region", "not_source", "background"}
    if _config_bool(background.get("invert", background.get("invert_mask", False)), default=False) or region in inverse_region_names:
        return "inverse"
    return "inside"


def _background_schema(
    background: Mapping[str, Any],
    *,
    source_kind: str,
    source_name: str,
    region: str,
) -> dict[str, Any]:
    legacy_mask_default = (
        background.get("mask_set") not in (None, "")
        and background.get("source_kind") in (None, "")
        and background.get("source_name") in (None, "")
        and background.get("mode") in (None, "")
        and background.get("statistic") in (None, "")
    )
    mode = str(background.get("mode") or background.get("statistic") or ("percentile" if legacy_mask_default else "mean")).strip().lower()
    if mode not in {"mean", "median", "percentile"}:
        raise ValueError(f"Unsupported background mode: {mode!r}")
    schema: dict[str, Any] = {
        "source_kind": source_kind,
        "source_name": source_name,
        "region": region,
        "mode": mode,
    }
    if mode == "percentile":
        schema["percentile"] = float(background.get("percentile", 1))
    return schema


def _per_label_stats(labels: Any, image: Any, stats: Sequence[str], *, np: Any) -> dict[int, dict[str, float]]:
    label_image = np.asarray(labels)
    intensity = np.asarray(image, dtype=float)
    result: dict[int, dict[str, float]] = {}
    for label_id in [int(item) for item in np.unique(label_image) if int(item) > 0]:
        values = intensity[label_image == label_id]
        if values.size == 0:
            continue
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            finite_values = values
        result[label_id] = {}
        for stat in stats:
            stat_name = str(stat)
            if stat_name == "mean":
                value = np.nanmean(finite_values)
            elif stat_name == "sum":
                value = np.nansum(finite_values)
            elif stat_name == "median":
                value = np.nanmedian(finite_values)
            elif stat_name == "min":
                value = np.nanmin(finite_values)
            elif stat_name == "max":
                value = np.nanmax(finite_values)
            elif stat_name == "std":
                value = np.nanstd(finite_values)
            elif stat_name == "area":
                value = finite_values.size
            elif stat_name.startswith("percentile_"):
                value = np.nanpercentile(finite_values, float(stat_name.rsplit("_", 1)[-1]))
            else:
                raise ValueError(f"Unsupported intensity statistic: {stat_name}")
            result[label_id][stat_name] = _float_or_nan(value, np=np)
    return result


def _feature_column_name(
    feature: Mapping[str, Any],
    channel: Mapping[str, Any],
    compartment_name: str,
    *,
    stat: str,
    suffix: str | None,
    disambiguate_explicit: bool = False,
) -> str:
    explicit = feature.get("name")
    if explicit not in (None, ""):
        if not disambiguate_explicit:
            return _slug(explicit)
        pieces = [str(explicit)]
        if stat:
            pieces.append(str(stat))
        if suffix:
            pieces.append(str(suffix))
        return _slug("_".join(pieces))
    pieces = [str(channel["name"]), str(compartment_name)]
    if stat:
        pieces.append(str(stat))
    if suffix:
        pieces.append(str(suffix))
    return _slug("_".join(pieces))


def _float_or_nan(value: Any, *, np: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float(np.nan)


def _structured_values(
    observations: Any,
    row_values: Sequence[Mapping[str, float]],
    columns: Mapping[str, Mapping[str, Any]],
    *,
    np: Any,
) -> Any:
    dtype = [("observation_id", "<i8")]
    dtype.extend((str(column), "<f8") for column in columns.keys())
    values = np.empty(int(observations.shape[0]), dtype=np.dtype(dtype))
    values["observation_id"] = observations["observation_id"]
    for column in columns.keys():
        values[column] = np.nan
    for row_index, row in enumerate(row_values):
        for column, value in row.items():
            if column in values.dtype.names:
                values[column][row_index] = _float_or_nan(value, np=np)
    return values


def _feature_schema(
    *,
    feature_name: str,
    object_name: str,
    source_label_set: str,
    frames: Sequence[int],
    observations: Any,
    columns: Mapping[str, Mapping[str, Any]],
    feature_spec: FeatureSetSpec,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "celltraj2.feature_set.v1",
        "feature_set": feature_name,
        "object_set": object_name,
        "source_label_set": source_label_set,
        "row_alignment": f"/object_sets/{object_name}/observations",
        "observation_id_base": 1,
        "frames": [int(frame) for frame in frames],
        "observation_count": int(observations.shape[0]),
        "missing_policy": "uncomputed_or_unavailable_values_are_nan",
        "columns": [
            {
                "name": "observation_id",
                "dtype": "int64",
                "description": "Stable 1-based observation id copied from the object observation table.",
            },
            *[_json_safe(column) for column in columns.values()],
        ],
        "feature_spec": feature_spec.to_dict(),
        "created_at": utc_now_iso(),
        "metadata": _json_safe(dict(metadata or {})),
    }


def _feature_qc(values: Any, frame_counts: Mapping[int, int], frame_warnings: Mapping[int, Sequence[str]], *, np: Any) -> dict[str, Any]:
    feature_names = [name for name in (values.dtype.names or ()) if name != "observation_id"]
    missing_counts = {
        name: int(np.isnan(values[name]).sum())
        for name in feature_names
    }
    return {
        "schema": "celltraj2.feature_qc.v1",
        "feature_count": len(feature_names),
        "observation_count": int(values.shape[0]),
        "missing_counts": missing_counts,
        "frame_counts": {str(key): int(value) for key, value in frame_counts.items()},
        "frame_warnings": {str(key): list(value) for key, value in frame_warnings.items()},
    }
