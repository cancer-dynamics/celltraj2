"""Masked XY texture and auditable, streaming ROI intensity normalization."""

from collections import OrderedDict
import json

import numpy as np

from celltraj2.feature_catalog import TEXTURE_METRICS


def masked_glcm(image, mask, *, levels=32, distances=(1,), lower_percentile=1, upper_percentile=99,
                intensity_range=None):
    """Directional-mean GLCM statistics using only pairs inside the object mask."""
    from skimage.feature import graycomatrix, graycoprops
    image, mask = np.asarray(image, float), np.asarray(mask, bool)
    mask = mask & np.isfinite(image)
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("Texture requires a 2D image and matching mask.")
    levels = int(levels)
    distances = [int(value) for value in distances]
    if not 2 <= levels <= 256 or not distances or any(value < 1 for value in distances):
        raise ValueError("GLCM requires 2..256 gray levels and positive integer distances.")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Texture quantization percentiles must satisfy 0 <= lower < upper <= 100.")
    result = {name: np.nan for name in TEXTURE_METRICS}
    if not mask.any():
        return result, {"pair_count": 0, "intensity_range": None}
    lo, hi = np.percentile(image[mask], [lower_percentile, upper_percentile]) if intensity_range is None else intensity_range
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        raise ValueError("Texture intensity range must be finite and increasing.")
    quantized = np.zeros(image.shape, dtype=np.uint16)
    quantized[mask] = 1 if hi == lo else 1 + np.floor(np.clip((image[mask] - lo) / (hi - lo), 0, 1) * (levels - 1)).astype(int)
    matrices = graycomatrix(quantized, distances=distances, angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=levels + 1, symmetric=True, normed=False)[1:, 1:, :, :].astype(float)
    counts = matrices.sum(axis=(0, 1))
    valid = counts > 0
    matrices = np.divide(matrices, counts[None, None], out=np.zeros_like(matrices), where=counts[None, None] > 0)
    if valid.any():
        for metric in TEXTURE_METRICS:
            if metric == "entropy":
                terms = np.zeros_like(matrices)
                positive = matrices > 0
                terms[positive] = -matrices[positive] * np.log2(matrices[positive])
                values = terms.sum(axis=(0, 1))
            else:
                values = graycoprops(matrices, "ASM" if metric == "asm" else metric)
            result[metric] = float(np.mean(values[valid]))
    return result, {"pair_count": int(counts.sum() / 2), "intensity_range": [float(lo), float(hi)]}


def compute_texture_frame(trajectory, labels, *, frame, source_label_set, feature):
    from skimage.measure import regionprops
    from celltraj2.features import (_resolve_channel, _read_channel_image, _match_spatial,
                                   _compartment_labels, _apply_background, _slug)
    channel = _resolve_channel(trajectory, feature.get("channel", 0))
    image = _match_spatial(_read_channel_image(trajectory, frame, channel, np=np), labels, np=np)
    compartment = _compartment_labels(trajectory, labels, frame=frame, source_label_set=source_label_set,
                                      compartment=feature.get("compartment") or {}, np=np)
    image, background, warnings = _apply_background(trajectory, image, frame=frame, background=feature.get("background"), np=np)
    metrics = list(feature.get("metrics", ("contrast", "homogeneity", "energy", "correlation", "entropy")))
    if not metrics or set(metrics) - set(TEXTURE_METRICS):
        raise ValueError("Select valid GLCM texture metrics.")
    prefix = _slug(feature.get("name") or "texture")
    config = {"levels": int(feature.get("levels", 32)), "distances": feature.get("distances", [1]),
              "lower_percentile": float(feature.get("lower_percentile", 1)),
              "upper_percentile": float(feature.get("upper_percentile", 99)),
              "intensity_range": feature.get("intensity_range")}
    columns = OrderedDict((f"{prefix}_{metric.lower()}", {
        "name": f"{prefix}_{metric.lower()}", "dtype": "float64", "family": "texture", "metric": metric,
        "channel": channel["schema"], "compartment": compartment["schema"], "background": background,
        "method": "masked_glcm_directional_mean", "slice_selection": "largest_compartment_area_xy_first_tie",
        "angles_degrees": [0, 45, 90, 135], "quantization": config, "unit": "dimensionless",
    }) for metric in metrics)
    for metric, unit in (("slice_z", "voxel_index_zero_based"), ("slice_area", "pixel"), ("pair_count", "count")):
        columns[f"{prefix}_{metric}"] = {"name": f"{prefix}_{metric}", "dtype": "float64", "family": "texture", "metric": metric, "unit": unit}
    values = {}
    for region in regionprops(np.asarray(labels)):
        local = np.asarray(compartment["labels"][region.slice]) == region.label
        pixels = image[region.slice]
        z = np.nan
        if local.ndim == 3:
            index = int(np.argmax(local.sum(axis=(1, 2))))
            z = int(region.slice[0].start) + index if local.any() else np.nan
            local, pixels = local[index], pixels[index]
        texture, qc = masked_glcm(pixels, local, **config)
        row = {f"{prefix}_{metric.lower()}": texture[metric] for metric in metrics}
        row.update({f"{prefix}_slice_z": z, f"{prefix}_slice_area": float(local.sum()), f"{prefix}_pair_count": qc["pair_count"]})
        values[int(region.label)] = row
    return {"values_by_label": values, "columns": columns, "warnings": [*warnings, *compartment["warnings"]]}


def normalize_intensity(trajectory, image, *, frame, channel, feature, cache):
    from celltraj2.features import _read_channel_image, _apply_background, _read_compartment_source_mask

    config = feature.get("normalization")
    if not isinstance(config, dict) or not config.get("enabled", True) or config.get("method", "none") == "none":
        return image, None, []
    method, scope = config.get("method", "zscore"), config.get("scope", "per_frame")
    if method not in {"divide_mean", "zscore", "match_mean", "match_mean_std"}:
        raise ValueError(f"Unsupported intensity normalization method: {method}")
    if scope not in {"per_frame", "all_frames"}:
        raise ValueError("Normalization scope must be per_frame or all_frames.")
    if config.get("region", "region") not in {"region", "inverse"}:
        raise ValueError("Normalization region must be region or inverse.")
    reference_frame = int(config.get("reference_frame", 1))
    if method.startswith("match_") and not 1 <= reference_frame <= int(trajectory.metadata.frame_count):
        raise ValueError("Normalization target frame is outside the movie.")
    key = ("normalization", json.dumps({"channel": channel, "background": feature.get("background"), "config": config}, sort_keys=True, default=str))
    state = cache.setdefault(key, {"stats": {}, "schema": {
        "method": method, "scope": scope, "region": config.get("region", "region"),
        "source_kind": config.get("source_kind"), "source_name": config.get("source_name"),
        "order": "background_then_normalization", "reference_frame": int(config.get("reference_frame", 1)),
        "frame_parameters": {}, "reference_statistics": {},
    }})
    warnings = []

    def statistics(f, current=None):
        if f not in state["stats"]:
            if current is None:
                current = _read_channel_image(trajectory, f, channel, np=np)
                current, _, messages = _apply_background(trajectory, current, frame=f, background=feature.get("background"), np=np)
                warnings.extend(messages)
            pixels = np.asarray(current, float)
            if config.get("source_name"):
                mask = _read_compartment_source_mask(trajectory, config.get("source_kind", "mask"), str(config["source_name"]), f, pixels, np=np)
                if config.get("region") == "inverse":
                    mask = ~mask
                pixels = pixels[mask]
            pixels = pixels[np.isfinite(pixels)]
            state["stats"][f] = {"count": int(pixels.size), "mean": float(pixels.mean()) if pixels.size else np.nan,
                                 "std": float(pixels.std()) if pixels.size else np.nan}
            progress = cache.get("progress")
            if progress:
                progress({"event": "normalization_frame", "frame": f, **state["stats"][f]})
        return state["stats"][f]

    current_stats = statistics(frame, image)
    if scope == "all_frames":
        if "pooled" not in state:
            n, mean, m2 = 0, 0.0, 0.0
            for f in range(1, int(trajectory.metadata.frame_count) + 1):
                s = statistics(f)
                count = s["count"]
                if count:
                    delta = s["mean"] - mean
                    total = n + count
                    m2 += count * s["std"] ** 2 + delta ** 2 * n * count / total
                    mean += delta * count / total
                    n = total
            state["pooled"] = {"count": n, "mean": mean if n else np.nan, "std": float(np.sqrt(m2 / n)) if n else np.nan}
        reference = state["pooled"]
    else:
        reference = current_stats
    target = statistics(int(config.get("reference_frame", 1))) if method.startswith("match_") else None
    denominator = reference["std"] if method in {"zscore", "match_mean_std"} else reference["mean"]
    if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps or (target and not np.isfinite(target["mean"])):
        scale, offset = np.nan, np.nan
        warnings.append(f"Normalization has an empty or zero-width/zero-mean reference in frame {frame}; returning NaN intensities.")
    elif method == "divide_mean":
        scale, offset = 1 / denominator, 0.0
    elif method == "zscore":
        scale, offset = 1 / denominator, -reference["mean"] / denominator
    elif method == "match_mean":
        scale, offset = target["mean"] / denominator, 0.0
    else:
        scale = target["std"] / denominator
        offset = target["mean"] - scale * reference["mean"]
    state["schema"]["frame_parameters"][str(frame)] = {"scale": float(scale), "offset": float(offset), "reference": reference, "target": target}
    state["schema"]["reference_statistics"].update({str(f): s for f, s in state["stats"].items()})
    return image * scale + offset, state["schema"], warnings
