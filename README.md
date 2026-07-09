# celltraj2

`celltraj2` is the trajectory-analysis interface being built for SITE and for
standalone Python workflows. It owns per-ROI analysis files, image-source
metadata, frame-based labels and masks, object tables, and the trajectory API
that connects segmentation outputs to feature extraction and downstream
analysis.

The package can be called by `sitelab` or used directly from notebooks and
scripts.

Current core capabilities:

- create per-ROI `.ct2.h5` files from SITE ROI/manifest metadata;
- read raw image frames from embedded H5 data, ROI OME-Zarr caches, TIFF
  fallback caches, or linked ND2 files plus stored ROI coordinates;
- store named frame-based labels and masks under one-based `frame_<n>` paths;
- index object observations and row-aligned lookup tables for ROI viewers;
- compute row-aligned single-object feature tables, including regionprops,
  intensity, compartment ratios, channel correlations, and SITE signaling;
- compose segmentation model input from stored channel specs;
- run dry or saved batch segmentation through an injectable Python callable or
  the Cellpose worker command;
- record segmentation and feature-extraction run/per-frame provenance in the
  H5.

Hosted documentation:

https://cancerdynamics.org/docs/celltraj2/

## Clone

```bash
git clone https://github.com/cancer-dynamics/celltraj2.git
cd celltraj2
```

## Install

Install with the analysis and ND2 extras in the environment that will run
H5/Zarr/ND2 workflows:

```bash
python -m pip install -e ".[analysis,nd2,dev]"
```

Cellpose batch workers can install only the I/O pieces they need, for example:

```bash
python -m pip install -e ".[analysis,nd2]"
python -m celltraj2.runners.cellpose_segment segmentation_job.json
python -m celltraj2.runners.extract_features feature_extraction_job.json
```

Lean environments can still import the package and use the typed metadata
contracts:

```bash
python -m pip install -e .
```

## Documentation Development

Documentation source lives in `docs/source/` and is built with Sphinx, MyST
Markdown, and `sphinx_rtd_theme`.

```bash
python -m pip install -e ".[docs]"
bash docs/make_docs.sh
```

To refresh the hosted documentation copy in the sibling
`cancerdynamics-website` repository:

```bash
python docs/publish_to_cancerdynamics.py --build
```
