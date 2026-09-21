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
- store explicit per-frame global transforms, including a default identity
  set, and estimate drift from object-centroid point clouds with grid plus
  continuous optimization;
- track observations by minimum centroid distance and store lineage topology as
  a parent-to-child CSR sparse matrix with row-aligned lineage/tracklet caches;
- apply active registration before tracking distances and preserve registration
  digest, cutoff units, and Z/Y/X calibration in graph/run provenance;
- build full-native or physically sampled columnar native-coordinate boundary
  libraries from indexed objects, standalone labels, and named mask surfaces
  with stable entity/point ids;
- calculate tissue-model-compatible `pcdiff` surface bases, curvature, CSR
  topology, and nearest external boundary interactions;
- track with registered robust unbalanced/partial boundary transport, hard
  physical displacement support, coverage diagnostics, and barycentric motion
  summaries while storing native
  point correspondences and registration-versioned displacement maps;
- query parents, children, histories, descendants, lineages, and maximal
  root-to-leaf trajectories, including scipy sparse trajectory membership;
- compute row-aligned single-object feature tables, including calibrated
  regionprops, mask-within-object component summaries, intensity, compartment
  ratios, channel correlations, and SITE signaling;
- compose segmentation model input from stored channel specs;
- run dry or saved batch segmentation through an injectable Python callable or
  the Cellpose worker command;
- record segmentation, feature-extraction, and tracking run/per-frame
  provenance in the H5.

Hosted documentation:

https://cancerdynamics.org/docs/celltraj2/

## Clone

```bash
git clone https://github.com/cancer-dynamics/celltraj2.git
cd celltraj2
```

## Create the conda environments

Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
and Git first. From the `celltraj2` repository root:

```bash
conda env create -f environment.yml
conda activate celltraj2
```

[environment.yml](environment.yml) selects Python 3.12 and installs the analysis
stack plus local `celltraj2[analysis,nd2]` through pip in editable mode (`-e`).

For Cellpose segmentation, create either or both separate worker environments:

```bash
conda env create -f environment-cellpose3.yml
conda env create -f environment-cellpose4.yml
```

- [environment-cellpose3.yml](environment-cellpose3.yml): Python 3.11,
  Cellpose 3.1.1, and NumPy 2.0.2.
- [environment-cellpose4.yml](environment-cellpose4.yml): Python 3.12 and
  Cellpose 4.2.1.1.

Both workers include PyTorch and editable `celltraj2[analysis,nd2]`. Activate
`cellpose3` or `cellpose4` to use that version. Conda and pip resolve
platform-specific dependencies; the YAMLs contain no CUDA build pins or
workstation paths.

## Existing environments and local installs

From the repository root in the environment you want to update:

```bash
python -m pip install -e ".[analysis,nd2]"
python -m pip check
```

Add `,dev` to the extras for tests. Lean metadata-only environments can use
`python -m pip install -e .`. If a YAML changes, use
`conda env update -f <environment-file.yml>` from this repo root.

For verification, GPU/model setup, and SITE integration, see the
[installation guide](https://cancerdynamics.org/docs/celltraj2/getting_started/installation.html).

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
