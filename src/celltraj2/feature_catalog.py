"""Shared, dependency-free catalogs for declarative feature builders."""

SUMMARY_STATISTICS = ("mean", "std", "median", "min", "max", "sum")
COMPONENT_STATISTICS = (*SUMMARY_STATISTICS, "cv", "largest")
COMPONENT_FIELDS = (
    "area", "equivalent_diameter", "axis_major_length", "axis_minor_length",
    "elongation", "roundness", "extent", "solidity",
)
COMPONENT_SCALARS = (
    "mask_area", "mask_fraction", "component_count", "component_density",
    "largest_component_fraction",
)
MOTION_FIELDS = (
    "displacement_z", "displacement_y", "displacement_x", "magnitude", "normal",
    "tangential_magnitude", "normal_gradient_squared", "surface_divergence",
    "tangential_vorticity", "tangential_strain_rate",
)
MOTION_SCALARS = ("mapped_fraction", "ot_cost_mean", "transported_mass_sum", "motion_link_count", "derivative_valid_fraction")
CENTROID_METRICS = (
    "displacement_z", "displacement_y", "displacement_x", "displacement",
    "velocity_z", "velocity_y", "velocity_x", "speed", "persistence",
    "acceleration", "neighbor_count", "valid_neighbor_fraction", "beta", "alpha",
    "nematic_alignment", "displacement_dot", "neighbor_speed", "relative_speed",
    "polarization",
)
TEXTURE_METRICS = ("contrast", "dissimilarity", "homogeneity", "asm", "energy", "correlation", "entropy")
CHANNEL_COMPARISON_METRICS = (
    "pearson_correlation", "spearman_correlation", "cosine_similarity", "ratio_of_means",
    "valid_pair_count", "valid_pair_fraction", "valid_ratio_fraction",
)
CHANNEL_COMPARISON_FIELDS = (
    "ratio", "inverse_ratio", "difference", "absolute_difference", "sum", "product",
    "normalized_difference", "log2_ratio",
)


def summary_metrics(feature, *, fields, scalars, statistics=SUMMARY_STATISTICS, prefix="", defaults=()):
    """Expand field/reduction pairs, preserving exact legacy metric requests."""
    legacy = feature.get("metrics", defaults if "fields" not in feature else ())
    if isinstance(legacy, str):
        legacy = [value.strip() for value in legacy.split(",") if value.strip()]
    result = list(legacy)
    requested_fields = list(feature.get("fields", ()))
    requested_stats = list(feature.get("statistics", ("mean",)))
    if set(requested_fields) - set(fields):
        raise ValueError(f"Unsupported quantities: {sorted(set(requested_fields) - set(fields))}")
    if set(requested_stats) - set(statistics):
        raise ValueError(f"Unsupported summary statistics: {sorted(set(requested_stats) - set(statistics))}")
    if requested_fields and not requested_stats:
        raise ValueError("Select at least one summary statistic for the selected quantities.")
    result.extend(f"{prefix}{field}_{stat}" for field in requested_fields for stat in requested_stats)
    allowed = set(scalars) | {f"{prefix}{field}_{stat}" for field in fields for stat in statistics}
    if set(result) - allowed:
        raise ValueError(f"Unsupported metrics: {sorted(set(result) - allowed)}")
    if not result:
        raise ValueError("Select at least one quantity or scalar metric.")
    return list(dict.fromkeys(result))


def component_metrics(feature, defaults=()):
    return summary_metrics(feature, fields=COMPONENT_FIELDS, scalars=COMPONENT_SCALARS,
                           statistics=COMPONENT_STATISTICS, prefix="component_", defaults=defaults)


def motion_metrics(feature, defaults=()):
    return summary_metrics(feature, fields=MOTION_FIELDS, scalars=MOTION_SCALARS, defaults=defaults)
