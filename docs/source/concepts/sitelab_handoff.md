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
- `/metadata/source_links.json`,
- `/sources/image_source.json`,
- `/labels/<label_set>/frame_<n>` datasets,
- `/masks/<mask_set>/frame_<n>` datasets,
- `/object_sets/<object_set>/observations` after object indexing,
- `/object_sets/<object_set>/lookup/frame_<n>` lookup arrays for ROI-viewer
  selection,
- `/object_sets/<object_set>/features/<feature_set>/values` feature tables,
- `/object_sets/<object_set>/features/<feature_set>/schema.json` feature
  provenance,
- segmentation run provenance under `/runs/segmentation/<run_id>/`.

`/sources/image_source.json` is the executable pixel-access contract. If the
source type is `linked_nd2`, its `path` should be the current ND2 path to open.
If the source type is `roi_ome_zarr` or `roi_tiff`, its `path` should remain the
ROI cache path, preferably stored relative to the SITE project root as
`roi_files/<dataset>/<roi_id>.ome.zarr` or `.tif`. The original parent ND2 link
lives in `/metadata/source_links.json` and the nested ROI `source_path`
metadata. SITE's Data tab can check and repair existing H5 files when a local
ND2 path changes or when an older H5 contains a stale absolute ROI cache path.

This does not make `celltraj2` depend on an open SITE project. Workers and
notebooks still accept absolute H5 paths, and relative image-source paths are
resolved from the H5's current location, standard `cell_files/<dataset>/`
layout, copied source-link metadata, or current working directory.

## Batch Segmentation Shape

SITE launches one headless worker process for a batch request, normally inside
the configured Cellpose environment:

```bash
python -m celltraj2.runners.cellpose_segment segmentation_job.json
```

The job file is JSON and contains H5 paths, frame selections, output targets,
backend/model parameters, and model-input channel specs. The worker emits JSONL
progress events on stdout so SITE can update a progress window without moving
large arrays between Python environments.

SITE has two launch paths for the same job shape:

- serial Test/Run starts one worker process immediately for the selected files;
- queued execution writes one file-level job per H5 to
  `analysis/workflow_jobs.jsonl`, then the SITE Jobs tab launches those jobs
  locally and monitors output.

Queued SITE job artifacts live under
`outputs/workflows/segmentation/<site_job_id>/`. The submitted
`segmentation_job.json` in that folder is still the celltraj2 worker contract.

Set `"save_outputs": false` at the job or file level for preview/dry-run
execution. In that mode the worker opens H5 files read-only, does not write
labels or masks, and does not write `/runs/segmentation` metadata. A preview
caller may provide `"preview_output_path"` to receive a single temporary `.npz`
bundle, or `"preview_output_dir"` to receive one `.npz` bundle per completed
frame. Each bundle contains the model input, labels, boolean mask, and resolved
output target. Set `"overwrite": true` for saved batch runs that should replace
existing frames.

The output target is always explicit. `"output_kind": "labels"` writes integer
labels to `/labels/<output_name>/frame_<n>`. `"output_kind": "masks"` writes
boolean positive-label masks to `/masks/<output_name>/frame_<n>`.

For each ROI/H5, the worker:

1. Resolve output path:
   `cell_files/<dataset>/<roi_id>.ct2.h5`.
2. Open the H5 with `celltraj2.Trajectory`.
3. Ask `celltraj2.Trajectory.get_image_data(frame=...)` for each frame.
4. Ask `celltraj2.Trajectory.frame_axes(frame_data.ndim)` for the returned
   frame axes.
5. Compose Cellpose-ready model input from stored channel specs and the actual
   frame axes.
6. Run the selected segmentation backend in the worker process.
7. Write labels or boolean masks under the requested target:
   `/labels/<output_name>/frame_<n>` or `/masks/<output_name>/frame_<n>`.
8. Record run/frame provenance under:
   `/runs/segmentation/<run_id>/`.

SITE should not write H5 internals directly. It should use `celltraj2` APIs so
the storage contract stays centralized.

The `frame_axes()` call is required because source axes and returned frame axes
are not always the same. SITE 3D ROI OME-Zarr caches are `T,C,Z,Y,X`; true 2D
ROI OME-Zarr caches are `T,C,Y,X` and return frames as `Y,X,C`. Model-input
preview and segmentation workers should therefore never infer the channel axis
only from the stored H5 image-source spec.

Minimal job shape:

```json
{
  "job_id": "seg_20260703_example",
  "project_root": "project",
  "save_outputs": true,
  "overwrite": false,
  "files": [
    {
      "h5_path": "cell_files/sample/sample_XY001_ROI001.ct2.h5",
      "output_name": "cyto_epithelial",
      "output_kind": "labels",
      "frames": {"mode": "range", "frame_start": 1, "frame_stop": 12},
      "backend": {
        "backend_id": "cellpose3",
        "model": "cyto3",
        "parameters": {"do_3D": true, "use_gpu": true}
      },
      "model_input": {
        "channel_specs": [
          {
            "channel_indices": [0],
            "combination": "single",
            "normalization": "lut_full_uint16"
          }
        ]
      }
    }
  ]
}
```

For direct Python execution without Cellpose, tests and notebooks can inject any
callable into `celltraj2.run_batch_segmentation`.

```python
from celltraj2 import SegmentationResult, run_batch_segmentation

def segmenter(model_input, file_job, frame):
    labels = run_my_model(model_input)
    return SegmentationResult(labels=labels, metadata={"engine": "my_model"})

run_batch_segmentation(job_dict, segmenter)
```

## Object Indexing Shape

After a label-producing segmentation run, SITE can launch object indexing with
the same local worker pattern:

```bash
python -m celltraj2.runners.index_objects object_index_job.json
```

Minimal job shape:

```json
{
  "job_id": "obj_index_20260703_example",
  "project_root": "project",
  "save_outputs": true,
  "overwrite": false,
  "files": [
    {
      "h5_path": "cell_files/sample/sample_XY001_ROI001.ct2.h5",
      "object_set": "cyto_epithelial",
      "source_label_set": "cyto_epithelial",
      "frames": {"mode": "all"}
    }
  ]
}
```

The worker reads `/labels/<source_label_set>/frame_<n>`, assigns stable
one-based `observation_id` values sorted by frame and label value, and writes:

```text
/object_sets/<object_set>/observations
/object_sets/<object_set>/observations_schema.json
/object_sets/<object_set>/lookup/frame_<n>
/runs/object_indexing/<run_id>/
```

Once observations exist, rerunning without `"overwrite": true` refuses to
replace the established index. This preserves row alignment for feature tables,
track assignments, exports, and ROI-viewer selections. With `"save_outputs":
false`, the worker opens H5 files read-only and reports counts without writing
object sets or run metadata.

## Feature Extraction Shape

After object indexing, SITE can launch feature extraction with the same worker
shape:

```bash
python -m celltraj2.runners.extract_features feature_job.json
```

Minimal job shape:

```json
{
  "job_id": "features_20260703_example",
  "project_root": "project",
  "save_outputs": true,
  "overwrite": false,
  "files": [
    {
      "h5_path": "cell_files/sample/sample_XY001_ROI001.ct2.h5",
      "feature_spec": {
        "feature_set": "site_v1",
        "object_set": "cyto_epithelial",
        "source_label_set": "cyto_epithelial",
        "frames": {"mode": "all"},
        "features": [
          {
            "kind": "intensity",
            "name": "site_cyto",
            "channel": {"raw_index": 3},
            "stats": ["mean"],
            "compartment": {
              "label_set": "cyto_epithelial",
              "exclude_mask_set": "nuclei",
              "name": "cyto_excluding_nuc"
            },
            "background": {
              "enabled": true,
              "source_kind": "mask",
              "source_name": "background",
              "region": "inverse",
              "mode": "mean"
            }
          }
        ]
      }
    }
  ]
}
```

Each file job contains one `FeatureSetSpec`. A spec writes one row-aligned
feature table under:

```text
/object_sets/<object_set>/features/<feature_set>/values
/object_sets/<object_set>/features/<feature_set>/schema.json
/object_sets/<object_set>/features/<feature_set>/qc.json
/runs/feature_extraction/<run_id>/
```

Supported feature block kinds are `regionprops`, `intensity`,
`compartment_ratio`, and `channel_correlation`. The SITE launcher also exposes a
`SITE Signaling` block that expands to compact `site_cyto`, `site_nuc`, and
`site_ratio` columns inside the default `site_v1` feature set.

Mask and label inputs are referenced by source name and kind, not by H5 path.
The worker resolves those names to `/masks/<name>/frame_<n>` or
`/labels/<name>/frame_<n>` for the active frame. The same source selection is
used for compartment inclusion/exclusion and optional background subtraction.

The worker streams JSONL events for SITE. In addition to file/frame lifecycle
events, `feature_frame_summary` events report the file, frame, feature column,
mean finite value, object count, finite count, and NaN count, which lets the SITE
run window show live per-feature progress.

## Tracking Shape

After choosing an active indexed object set, SITE launches minimum-centroid
tracking through:

```bash
python -m celltraj2.runners.track_centroids tracking_job.json
```

Minimal job shape:

```json
{
  "job_id": "track_20260710_example",
  "project_root": "project",
  "save_outputs": true,
  "files": [
    {
      "h5_path": "cell_files/sample/sample_XY001_ROI001.ct2.h5",
      "object_set": "cyto_epithelial",
      "track_set": "centroid_mindist",
      "method": "minimum_centroid_distance",
      "max_distance": 5.0,
      "coordinate_scale": [3.0, 0.325, 0.325],
      "metadata": {
        "distance_unit": "um",
        "distance_calibration_source": "h5_acquisition_metadata",
        "micron_per_pixel": 0.325,
        "zscale": 9.23076923076923
      }
    }
  ]
}
```

The coordinate scale is ordered Z/Y/X and is derived by SITE per file, not
entered by the user. When `/metadata/acquisition.json` has
`micron_per_pixel`, the maximum distance and link distances are reported in
microns; Z automatically uses stored Z voxel spacing or
`micron_per_pixel * zscale`. Missing `micron_per_pixel` triggers an explicitly
labeled pixel fallback with `[1, 1, 1]`. The worker emits per-frame object,
linked, and unlinked counts. Saved runs write the sparse graph under the object
set and provenance under `/runs/tracking/<run_id>/`; dry runs do neither.

SITE's general interaction rule is the same outside tracking: user-facing
spatial measurements and thresholds should use physical microns whenever
`micron_per_pixel` is available, and report pixels only as a clearly marked
fallback. Native array coordinates may remain pixels/voxels in storage.

## SITE ROI Viewer Consumption

SITE ROI viewers discover the per-ROI H5 next to the extracted ROI cache by
project convention:

```text
roi_files/<dataset>/<roi_id>.ome.zarr
cell_files/<dataset>/<roi_id>.ct2.h5
```

The `SITE objects` panel reads `/labels/<label_set>/frame_<n>`,
`/masks/<mask_set>/frame_<n>`, `/object_sets/<object_set>/object_set.json`, and
stored feature schemas to offer available H5 labels, masks, indexed object
sets, and feature columns. Standalone labels and masks can be loaded directly
with napari's default Labels/Image-layer display. `Add to view` keeps multiple
selected H5 sources as separate layers named from their stored H5 names, and all
added layers refresh on frame changes. Selecting one of those managed napari
layers makes it the current panel target. Indexed object sets display the
current frame as a single-color overlay by default, while retaining the original
integer label array for selection. When the user clicks an indexed object, SITE
resolves:

```text
label_id -> lookup[label_id] -> observation_id -> observations[observation_id - 1]
```

and displays the observation row plus any stored feature values for that
`observation_id`, grouped by feature set. `Color by...` can render the object
set as one color, by continuous object id, or by a selected feature column. For
feature coloring, SITE creates a feature-valued display image for the current
frame; pixels outside indexed objects are `NaN`, and the integer label frame
remains the click-selection source. Stored track sets now use the same
lookup/observation row spine: SITE colors by row-aligned `lineage_id` and uses
the CSR graph plus unique-parent cache to retain an observation's ancestors and
descendants as frames change. Tracklet assignments can also be converted to
napari-ready track rows and split-graph metadata.

The SITE Plot workflow is another read-only consumer of this contract. Object
count uses the number of `/object_sets/<object_set>/observations` rows per
frame. Feature-mean traces group the row-aligned
`features/<feature_set>/values` column by the observation table's `frame`
column. Missing lookup frames remain missing instead of being treated as zero,
and acquisition `time_interval_s` from copied metadata is used for an hours
axis when it is consistent across selected H5 files. SITE also reads
single-object feature distributions from the same row-aligned feature tables:
selected frames are filtered through the observation table's `frame` column,
then finite feature values are pooled by treatment for violin plots. Plot
images and reproducibility sidecars live in SITE outputs, not inside celltraj2
H5 files.

## One-Based Frames

H5 paths use one-based frame ids:

```text
/labels/epithelial/frame_1
/masks/cyto_immune/frame_1
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
