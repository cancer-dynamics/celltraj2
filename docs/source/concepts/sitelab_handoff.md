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
4. Compose Cellpose-ready model input from stored channel specs.
5. Run the selected segmentation backend in the worker process.
6. Write labels or boolean masks under the requested target:
   `/labels/<output_name>/frame_<n>` or `/masks/<output_name>/frame_<n>`.
7. Record run/frame provenance under:
   `/runs/segmentation/<run_id>/`.

SITE should not write H5 internals directly. It should use `celltraj2` APIs so
the storage contract stays centralized.

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
