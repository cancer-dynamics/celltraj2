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

## Next Phase: Interactive Analysis, Features, And Tracks

The next layer should build on saved H5 files rather than redesign storage:

- load active H5 sets from SITE or directly from Python;
- inspect metadata/source links, labels, masks, segmentation runs, and image
  access behavior;
- compute first feature tables under
  `/object_sets/<object_set>/features/<feature_set>/`;
- design tracking IDs and frame-to-frame lineage tables under
  `/object_sets/<object_set>/tracks/<track_set>/`;
- launch exploratory notebooks or marimo environments with active H5s,
  treatment groups, and replicate metadata preloaded;
- keep a clean backend API that works in notebooks without importing SITE.
