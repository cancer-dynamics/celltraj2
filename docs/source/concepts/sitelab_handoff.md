# SITE Handoff

SITE prepares ROI definitions and image caches. `celltraj2` creates and owns
the per-ROI analysis H5.

## Default Flow

```text
SITE project
  -> ND2 + .site.json manifest
  -> rois/<dataset>.rois.json
  -> roi_files/<dataset>/<roi_id>.ome.zarr
  -> cell_files/<dataset>/<roi_id>.ct2.h5
```

SITE should pass:

- parent SITE manifest JSON,
- ROI JSON record,
- resolved ROI image cache path or linked ND2 path,
- channel metadata,
- acquisition metadata,
- treatment metadata,
- segmentation run configuration.

`celltraj2` writes:

- `/metadata/site_manifest.json`,
- `/metadata/roi.json`,
- `/sources/image_source.json`,
- `/labels/<label_set>/frame_<n>` datasets,
- `/masks/<mask_set>/frame_<n>` datasets,
- segmentation run provenance under `/runs/segmentation/<run_id>/`.

## Batch Segmentation Shape

SITE launches a batch request over accepted ROIs. For each ROI:

1. Resolve output path:
   `cell_files/<dataset>/<roi_id>.ct2.h5`.
2. Ask `celltraj2` to create/open the H5.
3. Ask `celltraj2.Trajectory.get_image_data(frame=...)` for each frame.
4. Run the selected segmentation backend.
5. Write labels with:
   `Trajectory.write_label_frame("epithelial", frame=1, labels=labels)`.
6. Optionally write binary masks with:
   `Trajectory.write_mask_frame("nuclear", frame=1, mask=mask)`.

SITE should not write H5 internals directly. It should use `celltraj2` APIs so
the storage contract stays centralized.

## One-Based Frames

H5 paths use one-based frame ids:

```text
/labels/epithelial/frame_1
```

The parent zero-based time index is stored in metadata. For a ROI whose
`time_start` is 12, `frame_1` maps to parent T index 12.

## Snapshot Imaging

Static imaging should be represented as a one-frame acquisition. SITE and
`celltraj2` should create exactly the same style of H5 as for live imaging,
with only `frame_1`.

## First Test Hook For SITE

```python
from celltraj2.sitelab import create_analysis_h5_from_site_roi

path = create_analysis_h5_from_site_roi(
    roi_json_path="project/rois/sample.rois.json",
    roi_id="sample_XY001_ROI001",
    manifest_path="project/manifests/sample.site.json",
)
```

This creates `project/cell_files/sample/sample_XY001_ROI001.ct2.h5` with
metadata and source links, ready for frame-by-frame segmentation writes.
