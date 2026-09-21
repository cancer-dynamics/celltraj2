"""Compartment-local two-channel scalar and pixelwise comparisons."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from celltraj2.feature_catalog import CHANNEL_COMPARISON_FIELDS, CHANNEL_COMPARISON_METRICS


def comparison_selections(feature: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    from celltraj2.features import expand_intensity_statistics

    def selected(key: str, allowed: Sequence[str], default: Sequence[str]) -> list[str]:
        values = feature.get(key, default)
        if isinstance(values, str):
            values = [values]
        values = list(dict.fromkeys(str(value).strip().lower() for value in values))
        unsupported = set(values) - set(allowed)
        if unsupported:
            raise ValueError(f"Unsupported channel-comparison {key}: {sorted(unsupported)}")
        return values

    metrics = selected("metrics", CHANNEL_COMPARISON_METRICS, ["pearson_correlation"])
    fields = selected("fields", CHANNEL_COMPARISON_FIELDS, ["ratio"])
    if not metrics and not fields:
        raise ValueError("Channel comparison requires at least one object scalar or pixel quantity.")
    statistics = expand_intensity_statistics(feature.get("statistics", ["mean"])) if fields else []
    return metrics, fields, statistics


def _safe_ratio(numerator: Any, denominator: Any, floor: float, policy: str) -> Any:
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    valid = np.isfinite(a) & np.isfinite(b)
    valid &= (b > floor) if policy == "positive" else (np.abs(b) > floor)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = np.divide(a, b, out=np.full(a.shape, np.nan), where=valid)
    return np.where(np.isfinite(result), result, np.nan)


def compare_channel_values(
    a: Any, b: Any, *, metrics: Sequence[str], fields: Sequence[str],
    statistics: Sequence[str], denominator_floor: float = 0.0,
    denominator_policy: str = "positive",
) -> dict[str, float]:
    """Compare paired pixels; ratio-of-means and mean-of-ratios stay distinct.

    Invalid divisions are missing, not zero or epsilon-regularized. All statistics
    except finite-pixel count are NaN for an empty valid population.
    """
    floor = float(denominator_floor)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("Channel comparison denominator_floor must be finite and >= 0.")
    if denominator_policy not in {"positive", "absolute"}:
        raise ValueError("Channel comparison denominator_policy must be positive or absolute.")
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape:
        raise ValueError("Channel comparison requires paired arrays of equal shape.")
    count = len(a)
    paired = np.isfinite(a) & np.isfinite(b)
    a, b = a[paired], b[paired]
    values: dict[str, float] = {}
    ratio = _safe_ratio(a, b, floor, denominator_policy) if "ratio" in fields or "valid_ratio_fraction" in metrics else None
    nonconstant = len(a) >= 2 and np.any(a != a[0]) and np.any(b != b[0])
    for metric in metrics:
        value = np.nan
        if metric == "pearson_correlation" and nonconstant:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                value = np.corrcoef(a, b)[0, 1]
        elif metric == "spearman_correlation" and nonconstant:
            from scipy.stats import spearmanr
            value = spearmanr(a, b).statistic
        elif metric == "cosine_similarity" and len(a):
            # Scale independently before the dot product to avoid overflow.
            scale_a, scale_b = np.max(np.abs(a)), np.max(np.abs(b))
            if scale_a > 0 and scale_b > 0:
                x, y = a / scale_a, b / scale_b
                value = np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1)
        elif metric == "ratio_of_means" and len(a):
            with np.errstate(over="ignore", invalid="ignore"):
                value = _safe_ratio(np.mean(a), np.mean(b), floor, denominator_policy)
        elif metric == "valid_pair_count":
            value = len(a)
        elif metric == "valid_pair_fraction":
            value = len(a) / count if count else np.nan
        elif metric == "valid_ratio_fraction":
            value = np.count_nonzero(np.isfinite(ratio)) / len(a) if len(a) else np.nan
        values[metric] = float(value) if np.isfinite(value) else float("nan")

    for field in fields:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if field == "ratio":
                pixels = ratio
            elif field == "inverse_ratio":
                pixels = _safe_ratio(b, a, floor, denominator_policy)
            elif field == "difference":
                pixels = a - b
            elif field == "absolute_difference":
                pixels = np.abs(a - b)
            elif field == "sum":
                pixels = a + b
            elif field == "product":
                pixels = a * b
            elif field == "normalized_difference":
                pixels = _safe_ratio(a - b, a + b, floor, denominator_policy)
            elif field == "log2_ratio":
                valid = (a > 0) & (b > floor)
                pixels = np.full(a.shape, np.nan)
                pixels[valid] = np.log2(a[valid]) - np.log2(b[valid])
            else:
                raise ValueError(f"Unsupported channel-comparison field: {field}")
        finite = pixels[np.isfinite(pixels)]
        for statistic in statistics:
            value = np.nan
            if statistic == "area":
                value = len(finite)
            elif len(finite):
                with np.errstate(over="ignore", invalid="ignore"):
                    value = (np.percentile(finite, float(statistic.removeprefix("percentile_")))
                             if statistic.startswith("percentile_") else getattr(np, statistic)(finite))
            values[f"{field}_{statistic}"] = float(value) if np.isfinite(value) else float("nan")
    return values


def compute_channel_comparison_frame(
    trajectory: Any, labels: Any, *, frame: int, source_label_set: str,
    feature: Mapping[str, Any], cache: dict[str, Any],
) -> dict[str, Any]:
    from skimage.measure import regionprops
    from celltraj2.features import (_apply_background, _compartment_labels, _match_spatial,
                                   _read_channel_image, _resolve_channel, _slug)
    from celltraj2.image_features import normalize_intensity

    metrics, fields, statistics = comparison_selections(feature)
    floor = float(feature.get("denominator_floor", 0.0))
    policy = str(feature.get("denominator_policy", "positive"))
    # Validate division options even when the selected frame contains no objects.
    compare_channel_values([], [], metrics=[], fields=[], statistics=[], denominator_floor=floor, denominator_policy=policy)
    compartment = _compartment_labels(trajectory, labels, frame=frame, source_label_set=source_label_set,
                                      compartment=feature.get("compartment") or {}, np=np)
    images, channels, preprocessing = {}, {}, {}
    warnings = list(compartment["warnings"])
    for key, default in (("a", 0), ("b", 1)):
        channel = _resolve_channel(trajectory, feature.get(f"channel_{key}", default))
        image = _match_spatial(_read_channel_image(trajectory, frame, channel, np=np), labels, np=np).astype(float)
        config = {"background": feature.get(f"background_{key}"), "normalization": feature.get(f"normalization_{key}")}
        image, background, messages = _apply_background(trajectory, image, frame=frame, background=config["background"], np=np)
        warnings.extend(f"Channel {key.upper()}: {message}" for message in messages)
        image, normalization, messages = normalize_intensity(trajectory, image, frame=frame, channel=channel, feature=config, cache=cache)
        warnings.extend(f"Channel {key.upper()}: {message}" for message in messages)
        images[key], channels[key] = image, channel["schema"]
        preprocessing[key] = {"background": background, "normalization": normalization}
    prefix = _slug(feature.get("name") or "comparison")
    columns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for suffix in [*metrics, *(f"{field}_{stat}" for field in fields for stat in statistics)]:
        field, stat = ((None, suffix) if suffix in metrics else
                       next((field, suffix[len(field) + 1:]) for field in fields if suffix.startswith(f"{field}_")))
        column = f"{prefix}_{suffix}"
        columns[column] = {
            "name": column, "dtype": "float64", "family": "channel_comparison",
            "metric": suffix if suffix in metrics else None, "field": field, "statistic": stat,
            "channel_a": channels["a"], "channel_b": channels["b"], "compartment": compartment["schema"],
            "preprocessing": preprocessing, "denominator_floor": floor, "denominator_policy": policy,
            "finite_policy": "paired_finite_then_quantity_validity", "empty_statistic_policy": "nan_except_area_zero",
            "operation_order": "per_channel_background_then_normalization_then_comparison_then_summary",
        }
    values_by_label = {}
    for region in regionprops(np.asarray(labels)):
        selected = np.asarray(compartment["labels"][region.slice]) == region.label
        values = compare_channel_values(images["a"][region.slice][selected], images["b"][region.slice][selected],
                    metrics=metrics, fields=fields, statistics=statistics, denominator_floor=floor, denominator_policy=policy)
        values_by_label[int(region.label)] = {f"{prefix}_{name}": value for name, value in values.items()}
    return {"columns": columns, "values_by_label": values_by_label, "warnings": warnings}
