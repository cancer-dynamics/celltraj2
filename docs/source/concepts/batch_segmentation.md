# Batch Segmentation

`celltraj2` provides the headless backend used by SITE for batch segmentation.
SITE owns the GUI, file selection, treatment organization, and job monitoring.
`celltraj2` owns H5 access, raw image access, model-input composition, saved
labels/masks, and segmentation-run provenance.

## Execution Boundary

The normal SITE Cellpose worker command is:

```bash
python -m celltraj2.runners.cellpose_segment segmentation_job.json
```

That command should be run inside the selected Cellpose environment. The worker
does not import `sitelab`. It receives a JSON job file, opens each `.ct2.h5`
with `Trajectory`, reads image data through `Trajectory.get_image_data()`,
composes backend-ready model input, runs Cellpose, writes labels or masks when
requested, and emits JSONL progress events to stdout.

This avoids sending image frames between the SITE GUI environment and the
Cellpose environment. Large arrays stay inside the worker process.

## Job Shape

A batch job contains global execution settings and one file job per H5:

```json
{
  "job_id": "seg_20260703_example",
  "project_root": "/project",
  "save_outputs": true,
  "overwrite": false,
  "preview_output_dir": "/project/outputs/workflows/segmentation/job/preview_npz",
  "files": [
    {
      "h5_path": "cell_files/sample/sample_XY001_ROI001.ct2.h5",
      "output_name": "cyto_epithelial",
      "output_kind": "labels",
      "frames": {"mode": "range", "frame_start": 1, "frame_stop": 5},
      "backend": {
        "backend_id": "cellpose3",
        "model": "cyto3",
        "parameters": {
          "do_3D": true,
          "use_gpu": true,
          "normalize": true
        }
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

`SegmentationBatchJob` accepts either absolute H5 paths or paths relative to
`project_root`. SITE-submitted jobs should prefer relative paths such as
`cell_files/<dataset>/<roi_id>.ct2.h5` so queued job JSON can move with a
shared/local project copy. Standalone scripts may still submit absolute H5
paths. `SegmentationFileJob` supports frame selections by `all`, `range`,
explicit `frame_list`, or an explicit `frames` array.

## Output Targets

Each file job has an `output_name` and `output_kind`.

`output_kind="labels"` writes integer label images to:

```text
/labels/<output_name>/frame_<n>
```

`output_kind="masks"` converts positive labels to a boolean mask and writes:

```text
/masks/<output_name>/frame_<n>
```

Frame ids are one-based. `frame_1` is the first ROI timepoint, including static
snapshot data represented as a one-frame movie.

If a target frame exists and `overwrite=false`, the frame is skipped and the
skip is reported in the run metadata and JSONL events. If `overwrite=true`, the
existing frame dataset is replaced.

## Save, Test, And Preview Modes

`save_outputs=true` reads and calculates each frame without a read/write
handle. It briefly opens the canonical H5 with `r+` to commit that completed
frame, then closes it. Final run provenance is written under:

```text
/runs/segmentation/<job_id>/run.json
```

Per-frame progress is flushed to the external JSONL event stream instead of
creating progress records inside the H5. See [H5 Access And Job
Logging](h5_access_and_logging.md) for locking, retries, and concurrency.

`save_outputs=false` opens the H5 read-only. The worker still reads images,
composes model input, runs the segmentation callable, and reports events, but
it does not write labels, masks, or run metadata. SITE uses this for Test and
Preview actions.

`preview_output_path` writes one temporary `.npz` bundle for a single-frame
preview. `preview_output_dir` writes one `.npz` bundle per completed frame.
Bundles contain the model input, labels, positive mask, frame number, output
target, and backend metadata. SITE can render those bundles into preview PNGs
without asking the worker to import the GUI stack.

## Model Input

`compose_model_input()` turns a trajectory frame into Cellpose-style input from
stored channel specs.

Supported normalization modes:

- `raw`
- `lut_full_uint16`
- `full_uint16`

Supported source-channel combinations:

- `single`
- `mean`
- `max`

For 3D jobs, one output channel returns `Z,Y,X` and multiple output channels
return `Z,C,Y,X`. For 2D jobs, one output channel returns `Y,X` and multiple
output channels return `C,Y,X`. A 2D job reading from a Z stack must provide
`z_index` unless the stack has exactly one Z plane.

The batch worker must derive model-input axes from the actual frame it just
read, not blindly from `/sources/image_source.json`. In practice this means:

1. `frame_data = trajectory.get_image_data(frame=frame)`
2. `frame_axes = trajectory.frame_axes(frame_data.ndim)`
3. `compose_model_input(frame_data, axes=frame_axes, ...)`

This matters for SITE ROI caches because 3D OME-Zarr ROIs are `T,C,Z,Y,X`,
while 2D OME-Zarr ROIs are `T,C,Y,X` and become `Y,X,C` after a frame is read.
Using stale 3D source axes for a 2D frame shifts the channel axis and causes the
wrong channel to be segmented.

If a job requests 3D mode but the returned frame has no `Z` axis, `celltraj2`
runs that frame as effective 2D and records both `requested_do_3D` and
`effective_do_3D` in the frame event. This protects older configs and backend
defaults from sending true 2D images to Cellpose as singleton-Z volumes.
Cellpose-specific 3D parameters such as anisotropy, flow3D smoothing, and
stitching are only passed when the effective mode is 3D.

## Image Source Modes

The worker reads raw pixels through the image source stored in the H5:

```text
embedded_h5      raw frames stored directly in /images/raw/frame_<n>
roi_ome_zarr     SITE ROI OME-Zarr cache
roi_tiff         TIFF fallback ROI cache
linked_nd2       original ND2 plus stored ROI coordinates
```

The default SITE direction is `roi_ome_zarr` because it supports repeated
timepoint/channel/spatial access without repeatedly slicing the parent ND2.
`linked_nd2` remains available for storage-limited projects. Current SITE ROI
extraction requests a Zarr v2-compatible OME-Zarr group layout so workers with
Zarr 2.x can read caches. If a cache was already written as Zarr v3, use a
worker environment with Zarr 3.x or re-extract the ROI cache with the updated
SITE writer.

## Progress Events

`JsonlReporter` emits newline-delimited JSON events. Important events include:

- `job_started`
- `file_started`
- `frame_completed`
- `frame_skipped`
- `frame_failed`
- `h5_lock_waiting`
- `h5_lock_acquired`
- `commit_stale`
- `file_failed`
- `job_completed`

Events include the job id, H5 path, frame, output name/kind, saved path or
preview path, compact model-input summaries, label summaries, backend metadata,
and error text when applicable.

## Python Injection For Tests And Other Backends

The batch executor is not Cellpose-specific. Tests, notebooks, or future
backends can inject any callable with the same shape:

```python
from celltraj2 import SegmentationResult, run_batch_segmentation

def segmenter(model_input, file_job, frame):
    labels = run_my_model(model_input)
    return SegmentationResult(labels=labels, metadata={"engine": "my_model"})

summary = run_batch_segmentation(job_dict, segmenter)
```

This is the extension point for future non-Cellpose segmentation backends,
pixel classifiers, and analysis-specific mask generators.
