# celltraj2 Data Contract

This document defines the first `celltraj2` analysis H5 contract. The guiding
rule is simple: segmentation and derived masks are written one local frame at a
time, because that is how batch segmentation actually runs.

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

/cells/
  <label_set>/
    observations         table-like records, added after object indexing
    schema.json

/features/
  <label_set>/
    <feature_set>/
      values
      schema.json

/runs/
  segmentation/
    <run_id>/
      config.json
      provenance.json
      status.json
```

## Named Label Sets

Label sets are semantic names, not channel numbers:

```text
/labels/epithelial/frame_1
/labels/immune/frame_1
/labels/nuclei/frame_1
```

This replaces legacy names such as `msk`, `fmsk`, and `cell_data_m0`.
Downstream cell observations are grouped by the label set they were indexed
from:

```text
/cells/epithelial/observations
/cells/immune/observations
```

The first object table convention will include at least:

- `frame`: one-based local frame number,
- `parent_time_index`: zero-based parent acquisition T index when available,
- `label_id`: integer label value within that frame,
- `cell_id`: stable id after tracking or local id before tracking,
- `bbox_z0`, `bbox_z1`, `bbox_y0`, `bbox_y1`, `bbox_x0`, `bbox_x1`.

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

## Partial Segmentation

Frame-based storage supports common partial workflows:

- segment only frames 1, 5, and 10,
- retry `frame_5` without rewriting a full dense label array,
- add a new label set after the first segmentation pass,
- keep static snapshot imaging as `frame_1`.

Missing frame datasets mean "not segmented yet", not "empty labels".

## Coordinate Rules

Paths use one-based local frame ids. Metadata preserves raw coordinates:

- `roi.position_index`: zero-based SITE/ND2 XY position,
- `roi.time_start`, `roi.time_stop`: zero-based half-open parent T range,
- `roi.bounds`: zero-based half-open parent Z/Y/X bounds,
- `frame_map`: maps local one-based frames to parent zero-based T indices.

This lets the UI match NIS/SITE labels while analysis code remains precise.
