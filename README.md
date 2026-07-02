# celltraj2

`celltraj2` is the live-cell and snapshot microscopy trajectory-analysis
backend for SITE. It owns per-ROI analysis H5 files, image access across cached
or linked raw data, segmentation label/mask storage, object tables, features,
and later tracking/trajectory analysis.

SITE remains the project GUI and acquisition glue: ND2 inspection, channel and
treatment metadata, ROI creation, ROI OME-Zarr extraction, and batch job
launching. `celltraj2` is the standalone Python backend that can be used by
SITE or directly from notebooks.

## Current Scope

The first pass establishes:

- a frame-based H5 analysis contract,
- one-based local frame ids such as `frame_1`,
- named label and mask sets,
- source metadata that can point to ROI OME-Zarr, ROI TIFF, linked ND2, or
  embedded H5 image frames,
- a small `Trajectory` facade for image and segmentation data access,
- SITE handoff helpers for creating per-ROI `.ct2.h5` files.

Raw image pixels are not copied into the analysis H5 by default. The H5 should
be self-contained in metadata and provenance, while raw pixels are usually read
from `roi_files/<dataset>/<roi_id>.ome.zarr`.

## Layout

```text
project/
  roi_files/<dataset>/<roi_id>.ome.zarr
  cell_files/<dataset>/<roi_id>.ct2.h5
```

Inside the H5, frame-backed data are stored one frame per dataset:

```text
/images/raw/frame_1
/labels/epithelial/frame_1
/masks/nuclear/frame_1
/cells/epithelial/observations
```

Static snapshot imaging is represented as a one-frame acquisition with
`frame_1`.

## Development

Install with the analysis extras in the environment that will run H5/Zarr/ND2
workflows:

```bash
python -m pip install -e ".[analysis,nd2,dev]"
```

Lean environments can still import the package and use the typed contracts;
H5/Zarr/ND2 readers import their optional dependencies only when used.

See:

- `docs/data_contract.md`
- `docs/sitelab_handoff.md`
- `docs/implementation_plan.md`
