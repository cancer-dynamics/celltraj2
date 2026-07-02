# Focused Implementation Plan

This repository starts with the batch segmentation and analysis-file backbone,
not the full legacy celltraj feature surface.

## Phase 1: Repository And Contract

- Package skeleton under `src/celltraj2`.
- Frame-based H5 data contract.
- SITE handoff documentation.
- Optional dependency strategy for H5, Zarr, TIFF, and ND2 readers.

## Phase 2: Typed Metadata Models

- `RoiBounds`
- `RoiSpec`
- `ChannelSpec`
- `ImageSourceSpec`
- `TrajectoryMetadata`
- `SegmentationRunSpec`

Models use standard-library dataclasses and JSON-safe dictionaries so they can
be used without importing SITE or Pydantic.

## Phase 3: H5 Store

- `TrajectoryStore.create`
- `TrajectoryStore.open`
- JSON metadata read/write helpers.
- Frame-based raw image, label, and mask writers.
- One-based frame path helpers.

## Phase 4: Image Sources

- Embedded H5 frame source.
- ROI OME-Zarr source.
- ROI TIFF fallback source.
- Linked ND2 source.
- In-memory source for tests and notebooks.

## Phase 5: Trajectory Facade

- `Trajectory.get_image_data(frame=1, ...)`
- `Trajectory.write_label_frame(...)`
- `Trajectory.read_label_frame(...)`
- `Trajectory.write_mask_frame(...)`
- `Trajectory.read_mask_frame(...)`
- label/mask frame discovery.

## Phase 6: SITE Creation Hook

- Create `.ct2.h5` from SITE manifest + ROI JSON.
- Resolve `roi_ome_zarr`, `roi_tiff`, and `linked_nd2` source modes.
- Default output path under `cell_files/<dataset>/<roi_id>.ct2.h5`.

After this, the next layer is a small interactive analysis environment that can
load `cell_files`, inspect metadata/source links, view available labels/masks,
and probe image access frame by frame.
