# celltraj2 documentation

`celltraj2` is the frame-based microscopy trajectory-analysis backend for SITE
and standalone Python analysis. It owns per-ROI `.ct2.h5` analysis files,
source-image metadata, segmentation labels and masks, object tables, and the
trajectory facade that higher-level tools use to read images and write analysis
results.

The package is intentionally split from `sitelab`: `sitelab` remains the GUI
and acquisition/project workflow, while `celltraj2` owns the analysis data
contract and Python API that can be used from SITE, notebooks, scripts, and
batch jobs.

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started/installation
getting_started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/data_contract
concepts/sitelab_handoff
```

```{toctree}
:maxdepth: 2
:caption: Developer Guide

developer/implementation_plan
developer/publishing
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```
