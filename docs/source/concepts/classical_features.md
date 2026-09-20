# Classical Single-Object Features

These families use `FeatureSetSpec` and the existing observation-aligned compound
HDF5 feature tables. They do not change object IDs or write to tracking/boundary
products. Parameters, calibration, column meanings, and source dependencies are
stored in `schema.json`. SITE's Featurize launcher builds the same specifications.

## Centroid And Collective Motility

```python
trajectory.extract_features({
    "feature_set": "motility_v1", "object_set": "cells",
    "features": [{
        "kind": "centroid_motility", "name": "centroid",
        "track_set": "accepted", "registration_set": "auto",
        "neighbor_weighting": "shared_face",
        "metrics": ["displacement", "speed", "persistence", "acceleration",
                    "beta", "alpha", "nematic_alignment", "neighbor_count",
                    "valid_neighbor_fraction"],
    }],
})
```

Each observation uses its incoming parent-to-child link, including frame gaps
and links after division. Roots have no displacement. `auto` uses active H5
registration; an empty registration name uses native coordinates. Centroids use
calibrated Z/Y/X positions, including axial scaling, or pixels if calibration is
missing. Acquisition timestep metadata supplies hours, otherwise frame intervals.
Track and registration digests are retained.

Individual metrics are displacement components/magnitude, velocity components,
speed, directional persistence (cosine against the parent's incoming step), and
acceleration magnitude. Acceleration compares velocities at their interval
midpoints, accounting for frame gaps. Stationary steps have zero speed but
undefined direction. Components are named `displacement_z/y/x` and `velocity_z/y/x`.

Neighbors are direct Voronoi neighbors of all observed centroids in the current
ROI frame. Reflecting sites across ROI box faces produces box-clipped cells.
Shared edge lengths (2D) or face areas (3D) supply weights; external ROI faces
are excluded. `uniform` weights neighbors equally. This approximates tissue
adjacency, not actual contact: cells separated by empty space can be neighbors.
No boundary library is required. Coincident centroids or failed tessellations
produce NaN neighbor metrics plus warnings, not arbitrary coordinate jitter.

For incoming unit displacement `u_i`, unit separation
`r_ij = (x_i - x_j) / |x_i - x_j|`, and spatial dimension `d`:

| Metric | Pairwise quantity before weighted averaging |
| --- | --- |
| `beta` | `u_i dot u_j`, legacy directional alignment, range [-1, 1] |
| `alpha` | `(u_i - u_j) dot r_ij`, legacy radial product, range [-2, 2]; positive means separating directions |
| `nematic_alignment` | `(d * beta**2 - 1) / (d - 1)` |
| `displacement_dot` | Dot product of unnormalized incoming displacements |
| `neighbor_speed` | Neighbor velocity magnitude |
| `relative_speed` | Magnitude of the difference of the two velocities |

`polarization` is the magnitude of the weighted mean neighbor unit direction.
Weights are renormalized separately over finite pairs for each metric.
`neighbor_count` counts all geometric neighbors; `valid_neighbor_fraction` is
the fraction of weight with finite neighbor velocity. Stationary neighbors count
as valid velocities but have NaN direction products. No-neighbor averages are
NaN and count is zero. Displacement products use full link displacements, not
velocities; unequal gap durations affect amplitudes. Alpha's sign/range follow
the actual legacy implementation, correcting its misleading old docstring.

## Surface-Flow Quantities

```python
feature = {
    "kind": "boundary_motion", "name": "flow",
    "boundary_set": "cell_surfaces", "motion_set": "surface_ot",
    "geometry_set": "geometry", "direction": "incoming", "as_velocity": True,
    "fields": ["normal", "tangential_magnitude", "normal_gradient_squared",
               "surface_divergence", "tangential_vorticity", "tangential_strain_rate"],
    "statistics": ["mean", "std", "min", "max"],
    "metrics": ["mapped_fraction", "derivative_valid_fraction"],
    "derivative_neighbors": 16, "derivative_radius": None,
}
```

Independent fields and reductions produce `<name>_<field>_<statistic>`. Summaries
are mean, population std, median, min, max, and sum over finite surface points,
not area-weighted integrals. Existing explicit `metrics` lists still work without
expanding outputs. Coverage, OT cost, mass, and link count are separate scalars.

For `v = v_n n + v_t`, weighted local linear fits in tangent-plane coordinates
estimate the following, **without** subtracting mean motion or model predictions:

- `normal_gradient_squared`: `|grad_s(v_n)|^2`, KPZ-like normal-flow roughness.
- `surface_divergence`: `div_s(v_t)`, tangential divergence, not divergence of
  the full normal-plus-tangential field.
- `tangential_vorticity`: signed normal curl, with outward normals and a
  right-handed XYZ tangent basis.
- `tangential_strain_rate`: Frobenius norm of the symmetric tangent Jacobian.

Fits use nearest finite mapped points, an optional physical distance radius,
and a same-facing-normal filter. Degenerate/ill-conditioned fits return NaN.
`derivative_valid_fraction` is the fraction of all surface points with a valid
divergence fit. Sparse sampling, folds, and noisy normals warrant checking
neighborhood sensitivity. This local estimator is not a global surface PDE solver.

`as_velocity=True` divides each transport link by its actual elapsed time before
mass-weighted aggregation. Units are length/hour (else length/frame); derivative
units are inverse time, and inverse time squared for roughness. Default remains
displacement for backward compatibility, making these derivatives dimensionless.
Stored geometry normals are required for normal/tangential fields. 2D curves
support normal-gradient squared, divergence, and strain, but have no intrinsic
tangential-vorticity scalar: it is NaN.

## Planar Radial Shape

The existing `boundary_multipole` signals `shape_radius` and
`shape_radial_deviation` work in 2D. Radius about the calibrated object area
centroid is periodically interpolated onto 512 uniform angles; absolute Fourier
amplitudes are stored as `<name>_l0` through the selected order. `l0` is mean
radius or approximately zero for radial deviation. An ellipse has a strong second
harmonic independent of rotation or uneven boundary-point density. 3D multipoles
are unchanged and no separate 2D family is needed.

Radial descriptors retain physical size and do not completely encode holes or
non-star-convex overhangs. Duplicate rays use their outermost radius. Translation
and rotation invariance do not imply scale invariance. Previously saved tables
are not modified; recalculation adopts the uniform-angle 2D shape estimator.

## XY Texture

`kind="texture"` selects a channel, include/exclude compartment sources, optional
background subtraction, gray levels (default 32), and XY pixel-pair distances
(default 1). Quantization uses per-object slice percentiles (default 1 and 99),
or `intensity_range=[low, high]`. Fixed bounds preserve absolute contrast
comparability across objects; per-object bounds emphasize relative texture.

For 3D, use the slice with greatest selected-compartment area, first Z on ties.
Without compartment restrictions this is the largest cell-label slice. There is
no axial co-occurrence or end-slice averaging. QC columns are `<name>_slice_z`
(zero-based local ROI Z index; NaN in 2D/empty compartments), `_slice_area`
(pixels), and `_pair_count` (sum across distances/directions).

GLCM metrics are contrast, dissimilarity, homogeneity, angular second moment
(`asm`), energy, correlation, and base-2 entropy. These are standard
Haralick-style quantities, not the complete legacy 13-feature mahotas vector.
Symmetric matrices are averaged equally over valid distance/direction
combinations at 0, 45, 90, and 135 degrees. Only finite pairs whose endpoints
are inside the compartment contribute. No valid pairs gives NaN; constant-image
correlation follows [scikit-image's GLCM API](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops).

## Intensity Normalization

Normalization is **off by default** and follows background subtraction. It
assumes the reference distribution should be invariant, and can remove genuine
biology as well as bleaching. Reference pixels may be the whole finite ROI, a
stored mask/binarized label, or its inverse. Scope is `per_frame` or `all_frames`
(all movie frames, independent of output frame selection).

For reference mean/std `mu, sigma` and target-frame mean/std `mu0, sigma0`
(`reference_frame=1` by default):

| Method | Transform |
| --- | --- |
| `divide_mean` | `I / mu` |
| `zscore` | `(I - mu) / sigma` |
| `match_mean` | `I * mu0 / mu` |
| `match_mean_std` | `(I - mu) * sigma0 / sigma + mu0` |

Per-frame matching can compensate for bleaching or changing mean/width.
All-frame scope applies one pooled affine transform, preserving temporal shifts.
Pooled mean/variance are exact streaming merges weighted by finite pixel count,
without retaining the movie in memory. `schema.json` records reference
counts/mean/std and every output frame's scale/offset. Empty references or zero
denominators give NaN intensities plus warnings. Negative corrected values are
not clipped. This option currently belongs to Intensity blocks, not ratio blocks
or the SITE Signaling bundle.

## Mask-Component Summaries

Intersect the mask with each object before connected-component labeling. Scalars
are mask area/volume, fraction, component count/density, and largest-component
fraction of the mask. Component `fields` are area, equivalent diameter, major/minor
axes, elongation (major/minor), roundness (minor/major), extent, and solidity.
`statistics` are mean, population std, median, min, max, sum, CV, and `largest`.
Physical spacing applies to regionprops before reduction.

```python
feature = {
    "kind": "mask_components", "name": "mito", "mask_set": "mitochondria",
    "use_physical_spacing": True, "connectivity": 1,
    "metrics": ["mask_fraction", "component_count"],
    "fields": ["area", "roundness"],
    "statistics": ["mean", "min", "max", "largest"],
}
```

`mito_component_roundness_largest` means roundness in the largest component by
physical area/volume, not maximum roundness. Equal-size ties take the lowest
component label. Undefined shapes are NaN; summaries ignore nonfinite values,
but `largest` keeps the largest component's value even if undefined. Empty masks
have zero occupancy/count and NaN component summaries. Sizes retain calibrated
units; fractions, CV, and axis ratios are dimensionless.

Catalogs/expansion live in `celltraj2.feature_catalog`; scientific regression
tests are in `tests/test_classical_features.py`.
