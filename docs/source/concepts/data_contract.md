# Data Contract

The first `celltraj2` analysis contract is frame based. Segmentation and derived
masks are written one local frame at a time, because that is how batch
segmentation runs in practice.

## Frame IDs

Frame ids are one-based and local to the ROI analysis file:

```text
frame_1
frame_2
frame_3
```

`frame_1` is the first ROI timepoint, even though parent ND2/SITE coordinates
remain zero-based in metadata. Static snapshot imaging is a one-frame movie
with `frame_1`.

The H5 stores both:

- user-facing local frame ids, one-based, for paths and navigation, and
- raw parent T indices, zero-based, in metadata and object tables.

## Top-Level H5 Layout

```text
/
  attrs:
    celltraj2_schema_version

/metadata/
  celltraj2.json
  site_manifest.json
  roi.json
  source_links.json
  channels.json
  acquisition.json
  treatments.json

/sources/
  image_source.json

/images/
  raw/
    metadata.json
    frame_1              optional embedded raw image frame
    frame_2

/labels/
  <label_set>/
    metadata.json
    frame_1              integer labels, usually Z,Y,X
    frame_2

/masks/
  <mask_set>/
    metadata.json
    frame_1              bool or uint8 masks, usually Z,Y,X
    frame_2

/registrations/
  active.json             selected transform set for default consumers
  identity/
    frames
    transforms
    frame_status
    pairwise_results
    schema.json
    canvas.json
  <registration_set>/
    ...

/object_sets/
  <object_set>/
    object_set.json
    observations         canonical table, one row per observed object
    observations_schema.json
    lookup/
      frame_1            label_id -> observation_id lookup for UI/analysis
      frame_2
    features/
      <feature_set>/
        values
        schema.json
        qc.json
    tracks/
      <track_set>/
        adjacency/
          indptr
          indices
          data
        assignments
        links
        schema.json

/cells/                  reserved compatibility group, not canonical
/features/               reserved compatibility group, not canonical

/runs/
  object_indexing/
    <run_id>/
      run.json
      frames/              optional legacy/direct-API detail; batch progress is external
        frame_1.json
        frame_2.json
  segmentation/
    <run_id>/
      run.json
      frames/              optional legacy/direct-API detail; batch progress is external
        frame_1.json
        frame_2.json
  feature_extraction/
    <run_id>/
      run.json
      frames/              optional legacy/direct-API detail; batch progress is external
        frame_1.json
        frame_2.json
  registration/
    <run_id>/
      run.json
      frames/              optional legacy/direct-API detail; batch progress is external
        frame_1.json
        frame_2.json
  tracking/
    <run_id>/
      run.json
      frames/              optional legacy/direct-API detail; batch progress is external
        frame_1.json
        frame_2.json
```

## Named Object Sets

Label and mask sets are semantic names, not channel numbers:

```text
/labels/epithelial/frame_1
/labels/immune/frame_1
/labels/nuclei/frame_1
/masks/cyto_immune/frame_1
```

This replaces legacy names such as `msk`, `fmsk`, and `cell_data_m0`.

Pixel labels live under `/labels`. Indexed object information derived from
those labels lives under `/object_sets`:

```text
/labels/tumor_epithelial/frame_1
/object_sets/tumor_epithelial/observations
/object_sets/tumor_epithelial/lookup/frame_1
/object_sets/tumor_epithelial/features/
/object_sets/tumor_epithelial/tracks/
```

An `object_set` is a named population or layer of observed objects. An
`observation` is one object instance in one frame. Object sets usually share
the same name as their source label set, but `object_set.json` records the
`source_label_set` explicitly so a derived object set can still point back to
the image labels it came from.

The observation table is the stable row spine for downstream analysis. Rows are
sorted by frame and then by positive label value. Public ids are one-based:

```text
row 0 -> observation_id 1
row 1 -> observation_id 2
row i -> observation_id i + 1
```

There is no observation `0`; lookup arrays use `0` for background or "not
indexed". Feature tables and track assignment tables are row-aligned to
`/object_sets/<object_set>/observations`.

The first observation table convention includes at least:

- `observation_id`: stable one-based row id within the object set,
- `frame`: one-based local frame number,
- `parent_time_index`: zero-based parent acquisition T index when available,
- `label_id`: integer label value within that frame,
- `z_min`, `z_max`, `y_min`, `y_max`, `x_min`, `x_max`: local ROI half-open
  bounding box coordinates,
- `centroid_z`, `centroid_y`, `centroid_x`: local ROI centroid coordinates,
- `voxel_count`: object size in voxels/pixels,
- `quality_flags`: reserved bit field for curation or indexing issues.

For direct visualization, each indexed frame has a lookup array:

```text
/object_sets/<object_set>/lookup/frame_1
```

where `lookup[label_id] == observation_id`. This lets the SITE ROI viewer map a
clicked label value to the stable observation row without scanning the full
table.

```text
frame + clicked label_id -> observation_id -> observations[observation_id - 1]
```

Feature tables and track assignment tables use that same row alignment:

```text
observation_id -> feature row
observation_id -> track assignment
track_id -> linked observations across frames
```

## Global Registration

Global frame transforms are ROI-level data under `/registrations/`; they are
not owned by a particular track set. Every newly created celltraj2 H5 contains
`/registrations/identity` and selects it in `/registrations/active.json`, so
viewing and downstream analysis always have an explicit coordinate transform.
Existing files without that group are interpreted as identity until a set is
stored.

One registration set contains:

- `frames`: one-based local frame ids;
- `transforms`: one homogeneous matrix per frame;
- `frame_status`: row-aligned status codes such as `reference`, `estimated`,
  `identity`, `inherited`, or `failed`;
- `pairwise_results`: source/target frames, object counts, coarse/refined
  objective values, relative Z/Y/X shift, optimizer status, and quality flags;
- `schema.json`: method, units, axes, calibration, parameters, transform
  direction, completeness, and a stable registration digest;
- `canvas.json`: native/output shape, registered origin, and the additional
  display offset needed to show every transformed frame without cropping.

The canonical matrix direction is:

```text
native ROI physical coordinate -> registered ROI physical coordinate
```

Matrices use homogeneous column-vector notation. A true 2D file stores `Y,X`
3-by-3 matrices; a 3D file stores `Z,Y,X` 4-by-4 matrices. Translation values
are physical microns when `micron_per_pixel` is present. Z uses stored voxel
spacing or `micron_per_pixel * zscale`. Missing physical calibration is an
explicit pixel fallback. The `canvas_offset` is a rendering aid and is kept
separate from the canonical transform so analytical coordinates are not
silently re-originated.

The first computed method is
`pairwise_symmetric_nearest_neighbor_translation`. It uses the indexed object
centroids from consecutive available frames, scales them into physical space,
performs a bounded brute-force translation grid search, and then continuously
refines the best point with scipy optimization. The default objective is the
sum of nearest-neighbor distances in both directions; the legacy-style smooth
contact transform is also available. Relative estimates accumulate into
absolute transforms with the first available object frame as reference.
Unindexed/missing frames between valid anchors inherit the previous absolute
transform and retain an explicit `inherited` status; this permits partial
segmentation while making uncertainty visible. A computed run with no object
observations is rejected.

Registration runs write provenance under `/runs/registration/<run_id>/`.
Test jobs calculate the same matrices and canvas with `save_outputs=false` but
do not change the H5. Saved jobs may make their result active. Consumers should
resolve an explicit registration set first and otherwise use the active set.
Batch jobs stream one `registration_frame_summary` event as soon as each frame
is resolved, with its transform/status and pairwise optimizer diagnostics when
applicable. These live events are screen/log provenance; the final canonical
matrices and pairwise table remain the stored H5 contract.

Stored raw images, labels, masks, bounding boxes, and centroids remain on their
native grids. Viewers apply the active transform at display time, and
between-frame analyses apply it to coordinates before measuring displacement,
distance, or motility. This preserves editable/native segmentation data and
avoids interpolation loss. Any derived result that depends on a registration
must record both `registration_set` and `registration_digest`; a consumer must
not silently co-render it under a different active digest.

## Boundary Libraries

Boundary libraries are ROI-level, named artifacts under `/boundaries/`. They
store immutable geometry extracted from object labels, standalone label sets,
and binary mask surfaces without applying global registration:

```text
/boundaries/<boundary_set>/
  sources.json
  schema.json
  entities
  points/
    point_id
    boundary_entity_id
    frame
    native_index_zyx
    native_position_zyx
    orientation_hint_zyx
  entity_attributes/<attribute_set>/
  geometry/<geometry_set>/
  neighbors/<neighbor_set>/
  motion/<motion_set>/
```

`boundary_entity_id` and `point_id` are one-based identities local to the
boundary set. Every entity owns one contiguous, zero-based half-open point span
described by `point_start` and `point_count`. An entity from an indexed object
set carries its canonical `observation_id`; standalone labels use their native
positive `source_label_id`; masks become explicitly named surface entities.
State, cell type, biological role, and other classifications are row-aligned
entity attributes, never overloaded identity numbers.

Point columns are stored independently so consumers can read only the fields
and entity spans needed from libraries containing millions of points. Native
index coordinates retain the original array grid. Native positions apply the
stored physical Z/Y/X calibration but no registration. The schema contains a
stable `boundary_digest` for geometry, neighbor, tracking, and motion
dependencies.

Geometry sets are point-row-aligned and store oriented normals, tangent frames,
principal/mean/Gaussian curvature, quality flags, and the surface kNN topology
as CSR. The 3D surface backend follows the tissue-kinematic model: `pcdiff`
builds the kNN graph, local basis, and surface differential operators; curvature
comes from the symmetric shape operator `-grad_surface(normal)`. Two-dimensional
boundaries use a deterministic local curve estimator. Mask inside/outside
adjacency supplies the primary normal-orientation hint.

Neighbor sets are also CSR over point rows. Their target indices join directly
back to boundary point and entity metadata, allowing interactions to be sliced
by source, mask role, cell type, state, or frame without duplicating target
coordinates. Same-frame neighbor geometry is native and has no registration
dependency.

Between-frame boundary OT is different: candidate gating, transport cost, and
displacement are calculated in registered physical coordinates. Motion sets
retain native source and target `point_id` values, transport mass/cost, and the
registered displacement
`T_target(q_native) - T_source(p_native)`. Both the boundary set/digest and
registration set/digest are recorded. Changing registration invalidates the
derived track/motion set but never rewrites the boundary library.

## Sparse Lineage Graphs

Tracking topology is stored under
`/object_sets/<object_set>/tracks/<track_set>/adjacency/` as a square CSR sparse
matrix over observation rows. A nonzero at row `parent_observation_id - 1`,
column `child_observation_id - 1` means that the parent is linked to the child.
CSR `indices` are zero-based observation rows; `data` contains the one-based
`link_id` that addresses the matching row of `links`.

The core lineage invariant is:

- a child has zero or one parent;
- a parent has zero or more children;
- links connect observations in immediately consecutive local frames;
- the directed graph is acyclic.

`links` is a compound edge-metadata table containing parent/child observation
ids, source/target frames, centroid distance, tracker cost, confidence, and
quality flags. It does not replace the sparse matrix as the canonical topology.

`assignments` is a derived, row-aligned display/query cache with one row per
observation. It contains `parent_observation_id`, `lineage_id`, `tracklet_id`,
`generation`, `depth`, and `n_children`. The terms are deliberately distinct:

- **lineage**: one rooted, potentially branching family; every observation has
  exactly one lineage id;
- **tracklet**: one maximal non-branching segment; every observation has exactly
  one tracklet id;
- **trajectory**: a root-to-observation history or maximal root-to-leaf path;
  branch ancestors may participate in multiple trajectories, so trajectory ids
  are not a one-value-per-observation assignment.

The sparse graph is sufficient to regenerate `assignments`, obtain children
from CSR rows, obtain the unique parent from the assignment cache or a CSC
view, compute lineage components, trace histories, enumerate maximal
root-to-leaf trajectories, and find the ancestors/descendants of a SITE viewer
selection. `Trajectory.read_tracks(...)` returns the dependency-light graph;
`graph.adjacency.to_scipy()` returns a scipy CSR matrix when scipy is installed.
`graph.maximal_trajectory_matrix()` uses sparse matrix products to return a
leaf-by-observation boolean membership matrix for all maximal root-to-leaf
trajectories; the dependency-light list fallback remains available as
`graph.maximal_trajectories()`.

The first tracker, `track_minimum_centroid_distance`, preserves the legacy
`get_lineage_mindist` behavior: each child independently chooses the nearest
centroid in exactly the preceding frame when its scaled distance is strictly
below `max_distance`. Independent child choices intentionally allow divisions,
over-segmentation, and ambiguous forward branches. A future optimal-transport
boundary tracker will write the same graph contract and populate OT cost in
the edge metadata.

Before comparing centroids, tracking applies the selected registration set in
physical space. An omitted `registration_set` means the H5 active set. Both
`schema.json` and `/runs/tracking/<run_id>/run.json` record a
`registration_dependency` containing the registration name, digest, and
method, in addition to `max_distance`, `distance_unit`, Z/Y/X
`coordinate_scale`, and coordinate order. Link `distance` values therefore
refer to registered coordinates in the recorded unit. SITE supplies physical
calibration automatically from H5 acquisition metadata: with valid
`micron_per_pixel`, X/Y scales are microns per pixel and Z uses stored voxel
spacing or `micron_per_pixel * zscale`, giving `distance_unit=um`. If
`micron_per_pixel` is missing, SITE explicitly uses the pixel fallback
`coordinate_scale=[1, 1, 1]` and `distance_unit=pixel`. Coordinate scaling
remains a backend argument for reproducibility and non-SITE callers, but is not
a SITE user control.

Batch tracking uses the same dry-run and JSONL worker conventions as object
indexing and feature extraction:

```bash
python -m celltraj2.runners.track_centroids tracking_job.json
```

Saved runs record `/runs/tracking/<run_id>/run.json` and one frame summary per
stored observation frame. Test jobs set `save_outputs=false`, calculate the
same graph and linked/unlinked counts, and leave the H5 unchanged. Saved jobs
refuse to replace an existing track set unless `overwrite=true`.

The SITE ROI viewer uses this contract without changing the stored label
frames. For display it may render a binary foreground overlay, so all objects
in the active object set share one chosen color instead of napari assigning a
different color to every integer label. The original `/labels/<label_set>`
array remains the selection source: click position gives `label_id`, the lookup
array gives `observation_id`, and the observation row provides the side-panel
object information. Feature coloring keeps the same selection source, reads a
stored feature column for each `observation_id`, and replaces only the display
overlay with a feature-valued image. Lineage coloring and persistent branch
selection use that same row spine plus the selected track graph.

## Feature Sets

Feature extraction writes row-aligned tables under the active object set:

```text
/object_sets/<object_set>/features/<feature_set>/values
/object_sets/<object_set>/features/<feature_set>/schema.json
/object_sets/<object_set>/features/<feature_set>/qc.json
```

`values` is an HDF5 compound dataset with one row per
`/object_sets/<object_set>/observations` row. The first field is
`observation_id`; every other field is a float feature column. Values that were
not computed or not available are stored as `NaN`. `schema.json` records the
submitted `FeatureSetSpec`, row alignment, source label set, frame list, and a
schema entry for each feature column. `qc.json` stores missing-value counts and
per-frame warning summaries.

Feature set names group related quantities. Current defaults are:

- `site_v1`: SITE intracellular signaling measurements. The SITE launcher
  defaults the feature name to `site`, which expands to `site_cyto`,
  `site_nuc`, and `site_ratio`.
- `regionprops_v1`: scikit-image `regionprops_table`-style object morphology
  columns, currently using a `regionprops_<property>` naming pattern.

Additional feature blocks can write intensity, compartment ratio, channel
correlation, or regionprops-style columns into the same feature set. Output
column names must be unique within a feature set. When a future feature returns
multiple ordered values, use one compound-dataset column per component, with
the stable suffix convention `<feature_name>-1`, `<feature_name>-2`,
`<feature_name>-3`, and so on. The full feature details remain in
`schema.json`, so column names should stay readable but compact.

Compartments are described with stored label or mask sources rather than raw H5
paths. A compartment can start from the object set's source labels and then
optionally include or exclude another label/mask source:

```json
{
  "label_set": "cells",
  "include_mask_set": "nuclei",
  "exclude_label_set": "mitotic_nuclei",
  "name": "nuc"
}
```

This lets any stored `/masks/<name>/frame_<n>` or binarized
`/labels/<name>/frame_<n>` dataset define a nuclear, cytoplasmic, or custom
intracellular compartment. SITE's default signaling ratio uses distinct
compartments by excluding the nuclear source from the cytoplasm. Custom ratios
can also exclude the denominator source from the numerator.

Background subtraction is optional on intensity and compartment-ratio features.
The canonical background mapping selects a stored mask or label source and
chooses either the source region itself or its inverse:

```json
{
  "enabled": true,
  "source_kind": "mask",
  "source_name": "background",
  "region": "inverse",
  "mode": "mean"
}
```

`mode` is usually `mean`; percentile mode is retained for legacy wrappers. The
worker computes the background baseline for each frame, subtracts it from the
image used by the feature, and records the background mapping in each affected
column schema.

## Raw Image Sources

Raw pixels can come from four modes:

```text
embedded_h5      /images/raw/frame_1
roi_ome_zarr     roi_files/<dataset>/<roi_id>.ome.zarr
roi_tiff         roi_files/<dataset>/<roi_id>.tif
linked_nd2       original ND2 plus ROI coordinates
```

The default SITE workflow is `roi_ome_zarr`. The H5 stores the image source
specification in `/sources/image_source.json` and enough SITE metadata in
`/metadata/` to understand the file without opening the SITE project.

H5 files may store ROI cache paths as project-relative paths such as
`roi_files/<dataset>/<roi_id>.ome.zarr`. This keeps SITE local/shared projects
portable while preserving standalone `celltraj2` use. When opening an H5,
`celltraj2` resolves relative image-source paths from the standard
`cell_files/<dataset>/` layout, the H5 parent folder, copied source-link project
roots, and the current working directory. Stale absolute ROI cache paths from
another machine can also be recovered when their `roi_files/...` suffix exists
beside the current project.

When the parent ND2 path changes, update the H5 according to source type.
`linked_nd2` files should update `/sources/image_source.json:path` because that
is where raw pixels are read. `roi_ome_zarr` and `roi_tiff` files should keep
their cache path as the image source and update only the original-source link
metadata in `/metadata/source_links.json`, `/metadata/roi.json`, and the nested
ROI/image-source records in `/metadata/celltraj2.json`.

## Image Axis Contract

Image-source axes describe the backing source or cache. Frame axes describe the
array returned by `Trajectory.get_image_data(frame=...)`. Code that composes
model input or does image analysis should read the frame first, then ask
`Trajectory.frame_axes(frame_data.ndim)` how to interpret the returned array.

`get_image_data(frame=n)` applies the one-based local frame selection and drops
time/position axes. Remaining axes are normalized to spatial/channel order:

```text
3D multichannel frame   Z,Y,X,C
3D single-channel frame Z,Y,X
2D multichannel frame   Y,X,C
2D single-channel frame Y,X
```

SITE ROI OME-Zarr caches are written in `T,C,Z,Y,X` order when Z exists and in
`T,C,Y,X` order for true 2D acquisitions. Do not synthesize a fake `Z` axis for
2D caches. Older H5 files may contain stale source specs such as
`T,C,Z,Y,X` for a 2D cache; the image source repairs this at read time from the
actual OME-Zarr attrs/array dimensionality, but new H5 creation should store
the 2D source axes correctly.

## Partial Segmentation

Frame-based storage supports common partial workflows:

- segment only frames 1, 5, and 10,
- retry `frame_5` without rewriting a full dense label array,
- add a new label set after the first segmentation pass,
- keep static snapshot imaging as `frame_1`.

Missing frame datasets mean "not segmented yet", not "empty labels".

## Object Indexing Runs

Object indexing writes run metadata in the same H5 as the observations:

```text
/runs/object_indexing/<run_id>/run.json
/runs/object_indexing/<run_id>/frames/frame_1.json
```

Indexing is deterministic for a fixed set of label frames. Once
`/object_sets/<object_set>/observations` exists, re-indexing refuses to replace
it unless `overwrite=true` is explicit. This protects the row alignment used by
features, tracks, exports, and ROI-viewer selections.

`save_outputs=false` is a dry-run/test mode. The worker reads labels and emits
counts but opens H5 files read-only and does not write object sets, lookup
arrays, or run metadata.

## Segmentation Runs

Batch segmentation writes run metadata in the same H5 as the labels or masks:

```text
/runs/segmentation/<run_id>/run.json
/runs/segmentation/<run_id>/frames/frame_1.json
```

`run.json` stores the submitted job slice for that H5: output name, output
kind (`labels` or `masks`), frame list, backend/model parameters, channel specs,
overwrite policy, and run status. Each frame JSON stores status (`completed`,
`skipped`, or `failed`), the output path written for completed frames, compact
input/label summaries, backend metadata, and any error text.

The output target is explicit for every file job:

```text
output_name = "cyto_epithelial"
output_kind = "labels"   -> /labels/cyto_epithelial/frame_1
output_kind = "masks"    -> /masks/cyto_epithelial/frame_1
```

`masks` output converts any positive label value to `True`. This supports both
object-label segmentation and pixel/binary classifiers under the same workflow
contract.

`save_outputs=false` is a dry-run/test mode. The worker reads image data,
composes model input, runs the segmenter, and emits progress events, but opens
the H5 read-only and does not write labels, masks, or `/runs/segmentation`
metadata. `preview_output_path` and `preview_output_dir` can be used in either
test or saved runs to write temporary `.npz` bundles for visualization.

If a frame target already exists and `overwrite=false`, the frame is skipped.
If `overwrite=true`, the existing frame dataset is replaced.

## Feature Extraction Runs

Batch feature extraction writes run metadata in the same H5 as the feature
table:

```text
/runs/feature_extraction/<run_id>/run.json
/runs/feature_extraction/<run_id>/frames/frame_1.json
```

`run.json` stores the submitted feature spec, object set, source label set,
frame list, overwrite policy, save/test mode, and final feature count. Each
frame JSON stores status, the feature columns computed for that frame, warnings,
object counts, and `feature_summaries` with mean finite value, finite count,
NaN count, and object count per returned feature column.

If a feature set already exists and `overwrite=false`, saved runs refuse to
replace it. `save_outputs=false` is a dry-run/test mode: the worker reads image,
label, mask, and object-set data, streams summaries, and leaves feature tables
and `/runs/feature_extraction` metadata untouched.

## Model Input Contract

Model input is composed from stored channel specs rather than from ad hoc raw
channel numbers. Each output channel spec can select one or more raw source
channels, normalize them, and combine them into a backend input channel.

Supported normalization modes are `raw`, `lut_full_uint16`, and `full_uint16`.
Supported source-channel combinations are `single`, `mean`, and `max`.

3D jobs emit `Z,Y,X` for one model-input channel or `Z,C,Y,X` for multiple
channels. 2D jobs emit `Y,X` or `C,Y,X`. A 2D job reading from a Z stack needs
a `z_index` unless the stack has only one Z plane.

## Coordinate Rules

Paths use one-based local frame ids. Metadata preserves raw coordinates:

- `roi.position_index`: zero-based SITE/ND2 XY position,
- `roi.time_start`, `roi.time_stop`: zero-based half-open parent T range,
- `roi.bounds`: zero-based half-open parent Z/Y/X bounds,
- `frame_map`: maps local one-based frames to parent zero-based T indices.

This lets the UI match NIS/SITE labels while analysis code remains precise.
