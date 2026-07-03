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
`project_root`. `SegmentationFileJob` supports frame selections by `all`,
`range`, explicit `frame_list`, or an explicit `frames` array.

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

`save_outputs=true` opens the H5 in read/write mode. Completed frames are saved
to `/labels` or `/masks`, and run provenance is written under:

```text
/runs/segmentation/<job_id>/run.json
/runs/segmentation/<job_id>/frames/frame_<n>.json
```

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
`linked_nd2` remains available for storage-limited projects.

## Progress Events

`JsonlReporter` emits newline-delimited JSON events. Important events include:

- `job_started`
- `file_started`
- `frame_completed`
- `frame_skipped`
- `frame_failed`
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
