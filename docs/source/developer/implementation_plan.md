# Implementation Plan And Status

This repository starts with the batch segmentation and analysis-file backbone,
not the full legacy `celltraj` feature surface.

## Completed Phase 1: Repository And Contract

- Package skeleton under `src/celltraj2`.
- Frame-based H5 data contract.
- SITE handoff documentation.
- Optional dependency strategy for H5, Zarr, TIFF, and ND2 readers.

## Completed Phase 2: Typed Metadata Models

- `RoiBounds`
- `RoiSpec`
- `ChannelSpec`
- `ImageSourceSpec`
- `TrajectoryMetadata`
- `SegmentationRunSpec`

Models use standard-library dataclasses and JSON-safe dictionaries so they can
be used without importing SITE or Pydantic.

## Completed Phase 3: H5 Store

- `TrajectoryStore.create`
- `TrajectoryStore.open`
- JSON metadata read/write helpers.
- Frame-based raw image, label, and mask writers.
- One-based frame path helpers.

## Completed Phase 4: Image Sources

- Embedded H5 frame source.
- ROI OME-Zarr source.
- ROI TIFF fallback source.
- Linked ND2 source.
- In-memory source for tests and notebooks.
- Frame-axis reporting via `Trajectory.frame_axes()` so callers interpret the
  actual array returned by `get_image_data()`. SITE 3D ROI caches read as
  `Z,Y,X,C` per frame, while true 2D ROI caches read as `Y,X,C`; stale 2D H5
  source specs with an extra `Z` are repaired at read time.

## Completed Phase 5: Trajectory Facade

- `Trajectory.get_image_data(frame=1, ...)`
- `Trajectory.write_label_frame(...)`
- `Trajectory.read_label_frame(...)`
- `Trajectory.write_mask_frame(...)`
- `Trajectory.read_mask_frame(...)`
- label/mask frame discovery.

## Completed Phase 6: SITE Creation Hook

- Create `.ct2.h5` from SITE manifest + ROI JSON.
- Resolve `roi_ome_zarr`, `roi_tiff`, and `linked_nd2` source modes.
- Default output path under `cell_files/<dataset>/<roi_id>.ct2.h5`.

## Completed Phase 7: SITE-Controlled Batch Segmentation

- Dependency-light model-input composition from stored channel specs.
- Model-input composition uses actual frame axes after image read, preserving
  correct channel selection for both 2D and 3D SITE ROI caches.
- JSON batch job schema for H5 paths, frame selections, backend parameters,
  output names/kinds, save/test behavior, preview bundles, and overwrite
  policy.
- Generic `run_batch_segmentation` executor with injectable segmentation
  callable for tests and non-Cellpose backends.
- Optional Cellpose runner:
  `python -m celltraj2.runners.cellpose_segment job.json`.
- Per-run and per-frame provenance under `/runs/segmentation/<run_id>/`.
- Explicit label or mask targets under `/labels/<name>/frame_<n>` or
  `/masks/<name>/frame_<n>`.
- JSONL progress events for SITE job monitoring.

## Completed Phase 8: Object Observation Indexing

- Canonical object-set namespace under `/object_sets/<object_set>/`.
- Stable observation tables under `/object_sets/<object_set>/observations`.
- One-based `observation_id` values with row alignment:
  `row_index == observation_id - 1`.
- Per-frame lookup arrays under
  `/object_sets/<object_set>/lookup/frame_<n>` mapping
  `label_id -> observation_id`.
- `Trajectory.object_set(...).index_observations(...)` facade for notebooks and
  backend users.
- Batch object-indexing executor with dry-run behavior and JSONL events.
- Optional runner:
  `python -m celltraj2.runners.index_objects job.json`.
- Per-run and per-frame provenance under `/runs/object_indexing/<run_id>/`.

## Completed Phase 9: Single-Object Feature Tables

- `FeatureSetSpec` for row-aligned object feature calculation.
- Regionprops, intensity, compartment-ratio, channel-correlation, and SITE
  signaling feature blocks.
- Default SITE signaling feature set `site_v1`, expanding a compact feature name
  such as `site` into `site_cyto`, `site_nuc`, and `site_ratio`.
- Stored mask/label source references for compartment inclusion and exclusion.
- Optional per-frame background subtraction from a stored mask/label source or
  its inverse.
- Compound HDF5 feature tables under
  `/object_sets/<object_set>/features/<feature_set>/values`, with
  `schema.json` and `qc.json` sidecars.
- Per-run and per-frame provenance under `/runs/feature_extraction/<run_id>/`.
- JSONL `feature_frame_summary` events for SITE progress output.
- Optional runner:
  `python -m celltraj2.runners.extract_features job.json`.

## Completed Phase 10: First-Pass Tracks And Interactive Analysis

The first tracking layer builds on the saved observation row spine:

- canonical parent-to-child CSR adjacency storage under
  `/object_sets/<object_set>/tracks/<track_set>/adjacency/`;
- edge metadata in `links` and row-aligned lineage/tracklet caches in
  `assignments`;
- dependency-light ancestry, descendants, lineage, selection-tree, and maximal
  root-to-leaf trajectory queries;
- legacy-compatible minimum-centroid-distance tracking with recorded Z/Y/X
  coordinate scaling;
- JSON batch worker, dry-run output, overwrite protection, JSONL events, and
  `/runs/tracking/` provenance;
- SITE lineage coloring and cross-frame selection consume the same
  `label_id -> observation_id -> row` lookup as feature coloring;
- SITE Test/Run/Queue launcher, track-set selection, and napari tracklet
  rendering;
- SITE derives physical micron calibration automatically from stored
  `micron_per_pixel`, voxel spacing, and `zscale`, with an explicit pixel
  fallback and no user-facing coordinate-scale controls;
- completed first-pass scope: sparse storage/query, minimum-distance worker,
  SITE launcher, lineage coloring, tracklets, and persistent branch selection.

## Completed Phase 11: Global Frame Registration

- Every new H5 receives a stored identity registration and active pointer.
- ROI-level registration sets store native-to-registered homogeneous matrices,
  per-frame status, pairwise estimator results, an uncropped display canvas,
  method/calibration metadata, and a stable digest.
- The first estimator matches legacy cell-position registration: symmetric
  nearest-neighbor distance minimization over a bounded brute-force grid,
  followed by continuous scipy refinement.
- Registration is calibrated automatically in microns, including axial voxel
  spacing for 3D data, with a clearly recorded pixel fallback.
- Missing object frames inherit the last known absolute transform and remain
  visibly marked as inherited rather than being treated as estimated.
- JSON Test/Run/Queue workers write `/runs/registration/` provenance when
  saving.
- Minimum-centroid tracking applies the selected/active registration before
  measuring link distance and records the registration name and digest.
- SITE applies active transforms to raw images, labels, masks, object overlays,
  selection overlays, and tracklet coordinates without resampling stored
  arrays; the navigator exposes the active set and current frame shift.

Future between-frame feature calculations, including motility, must consume
the same registration set and record the same name/digest dependency. More
general affine/non-rigid methods may extend the stored matrix contract later;
they must not create a separate coordinate convention.

## Deferred Next Tracking Pass

- Establish the boundary-analysis/library machinery first.
- Implement optimal-transport boundary tracking using the same sparse graph,
  assignments, run provenance, distance-unit convention, registration
  dependency, and SITE viewer representation; do not introduce a parallel
  trajectory storage shape.
- Populate OT cost/confidence/quality edge metadata while retaining the unique
  backward parent and forward branching invariants.

Additional interactive-analysis work remains separate from this tracking pass:

- launch exploratory notebooks or marimo environments with active H5s,
  treatment groups, feature tables, and replicate metadata preloaded;
- keep a clean backend API that works in notebooks without importing SITE.
