# Quickstart

The central object is a per-ROI `.ct2.h5` file. It stores metadata and analysis
results while raw pixels usually remain in a linked ROI image cache such as
`roi_files/<dataset>/<roi_id>.ome.zarr`.

## Create From a SITE ROI

The first SITE integration hook creates a `celltraj2` analysis file from a SITE
ROI JSON file and an optional SITE manifest:

```python
from celltraj2.sitelab import create_analysis_h5_from_site_roi

path = create_analysis_h5_from_site_roi(
    roi_json_path="project/rois/sample.rois.json",
    roi_id="sample_XY001_ROI001",
    manifest_path="project/manifests/sample.site.json",
)
```

By default, the output path is:

```text
project/cell_files/sample/sample_XY001_ROI001.ct2.h5
```

## Read Images And Write Labels

Use the `Trajectory` facade for user-facing access:

```python
from celltraj2 import Trajectory

with Trajectory(path) as traj:
    image = traj.get_image_data(frame=1, channels=[0, 1])
    labels = run_segmentation(image)
    traj.write_label_frame("epithelial", frame=1, labels=labels)
```

Frame ids are one-based in the H5 paths and public API. Parent acquisition
coordinates from SITE remain zero-based in metadata.

## Inspect Available Results

```python
with Trajectory(path) as traj:
    print(traj.label_sets())
    print(traj.label_frames("epithelial"))
```

Missing label or mask frame datasets mean "not processed yet", not "empty".

## Run A Dry Batch Segmentation

Batch execution can be driven by any callable. This is the same executor shape
used by SITE before swapping in the Cellpose command-line runner:

```python
import numpy as np

from celltraj2 import SegmentationResult, run_batch_segmentation


def threshold_segmenter(model_input, file_job, frame):
    image = np.asarray(model_input)
    labels = (image > image.mean()).astype(np.int32)
    return SegmentationResult(labels=labels, metadata={"engine": "threshold"})


job = {
    "job_id": "example_test",
    "save_outputs": False,
    "files": [
        {
            "h5_path": str(path),
            "output_name": "example_mask",
            "output_kind": "masks",
            "frames": {"mode": "range", "frame_start": 1, "frame_stop": 1},
            "model_input": {
                "channel_specs": [
                    {"channel_indices": [0], "combination": "single", "normalization": "full_uint16"}
                ]
            },
        }
    ],
}

summary = run_batch_segmentation(job, threshold_segmenter)
print(summary)
```

Set `save_outputs=True` and `overwrite=True` when a saved run should write or
replace `/labels/<name>/frame_<n>` or `/masks/<name>/frame_<n>` datasets.

## Run The Cellpose Worker

Inside a Cellpose environment that also has `celltraj2` installed:

```bash
python -m celltraj2.runners.cellpose_segment segmentation_job.json
```

The worker reads the job JSON, emits JSONL progress events on stdout, and writes
segmentation outputs and `/runs/segmentation` provenance when `save_outputs` is
enabled.
