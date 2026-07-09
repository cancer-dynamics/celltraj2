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
    tracks/
      <track_set>/
        assignments
        links
        schema.json

/cells/                  reserved compatibility group, not canonical
/features/               reserved compatibility group, not canonical

/runs/
  object_indexing/
    <run_id>/
      run.json
      frames/
        frame_1.json
        frame_2.json
  segmentation/
    <run_id>/
      run.json
      frames/
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

Features and tracks will build on that same row alignment:

```text
observation_id -> feature row
observation_id -> track assignment
track_id -> linked observations across frames
```

The SITE ROI viewer uses this contract without changing the stored label
frames. For display it may render a binary foreground overlay, so all objects
in the active object set share one chosen color instead of napari assigning a
different color to every integer label. The original `/labels/<label_set>`
array remains the selection source: click position gives `label_id`, the lookup
array gives `observation_id`, and the observation row provides the side-panel
object information. Future feature and lineage coloring should keep this same
row spine and only change the display overlay.

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
