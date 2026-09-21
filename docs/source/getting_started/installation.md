# Installation

## Install conda and clone the repository

Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
(for example, Miniforge, Miniconda, or Anaconda) and Git. Open a terminal where
`conda --version` and `git --version` work. On Windows, use a conda-enabled
terminal for a native install, or install conda and run all commands inside
your WSL Linux distribution.

```bash
git clone https://github.com/cancer-dynamics/celltraj2.git
cd celltraj2
```

Keep this checkout: the YAML files install the local package with pip in
editable mode (`-e`), so Python uses the code in this folder.

## Create the analysis environment

From the `celltraj2` repository root:

```bash
conda env create -f environment.yml
conda activate celltraj2
```

This creates a Python 3.12 environment with the analysis stack, H5, Zarr, TIFF,
and ND2 support, plus plotting and common notebook-analysis dependencies. The
YAML runs the equivalent of `python -m pip install -e ".[analysis,nd2]"`;
no separate local package installation is needed.

| Environment | Python | YAML |
| --- | --- | --- |
| `celltraj2` | 3.12 | [environment.yml](https://cancerdynamics.org/docs/celltraj2/environment.yml) |
| `cellpose3` | 3.11 | [environment-cellpose3.yml](https://cancerdynamics.org/docs/celltraj2/environment-cellpose3.yml) |
| `cellpose4` | 3.12 | [environment-cellpose4.yml](https://cancerdynamics.org/docs/celltraj2/environment-cellpose4.yml) |

The files are included in the repository and available from the download links
above. Use them in the repository root so the editable installation can find
`pyproject.toml`.

## Optional Cellpose worker environments

Create either or both environments if you will run Cellpose segmentation.
Keep Cellpose 3 and 4 in separate environments. From the `celltraj2` root:

```bash
conda env create -f environment-cellpose3.yml
conda env create -f environment-cellpose4.yml
```

The recipes select Cellpose 3.1.1 with NumPy 2.0.2, or Cellpose 4.2.1.1, and
install PyTorch and editable `celltraj2[analysis,nd2]`. They include the
dependencies used by SITE's worker checks, including direct ND2 reading.
They do not require a `sitelab` checkout.

Activate the worker you want to use:

```bash
conda activate cellpose3
python -m celltraj2.runners.cellpose_segment --help
```

Use `conda activate cellpose4` for Cellpose 4. These are headless worker
environments; the standalone Cellpose GUI is optional. To add it, install
`"cellpose[gui]==3.1.1"` or `"cellpose[gui]==4.2.1.1"` with `python -m pip install`
in the matching environment.

### GPU setup and model downloads

Start with the YAML. Conda and pip resolve packages for the installation
platform; the recipes contain no workstation paths, CUDA build pins, or GPU
driver settings. The analysis environment does not require PyTorch or CUDA.

Check the selected Cellpose environment:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

If CUDA is unavailable on an NVIDIA machine, check the host driver and follow
the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)
for the appropriate build in that environment. Package installation alone does
not install a host GPU driver. SITE's current Cellpose Resources tests require
CUDA; CPU or Apple MPS support in standalone Cellpose does not satisfy those
tests.

Pretrained Cellpose models may be downloaded on first use. See the
[Cellpose model-directory instructions](https://cellpose.readthedocs.io/en/latest/installation.html#built-in-model-directory)
for cache locations and custom model paths.

## Verify the installation

In each environment you create:

```bash
python -m pip check
python -c "import celltraj2, numpy, h5py, zarr, nd2, xarray, skimage; print('celltraj2 imports OK')"
```

For SITE integration, select these environments in the Command Center's
Resources tab and press **Test** for the corresponding backend. The test checks
worker imports and a small H5 roundtrip; Cellpose tests also check the major
version and GPU. See the
[SITE installation guide](https://cancerdynamics.org/docs/sitelab/getting_started/installation.html)
for the GUI and foundation-feature environments.

## Existing environments and editable updates

To install into an existing analysis or Cellpose environment, or refresh local
dependencies after pulling repository changes, activate that environment,
change to the `celltraj2` root, and run:

```bash
python -m pip install -e ".[analysis,nd2]"
```

Editable installs pick up source edits automatically. Rerun the install when
dependencies or package metadata change. If the YAML itself changes, update the
matching environment from the repo root, for example:

```bash
conda env update -f environment.yml
```

Use the corresponding Cellpose YAML for a worker environment. Do not run
`conda env create` again for an environment that already exists.

For lean environments, install only the pieces you need:

```bash
# H5 and Zarr segmentation I/O only:
python -m pip install -e ".[h5,zarr]"
# Also read directly from linked ND2 files:
python -m pip install -e ".[h5,zarr,nd2]"
# Typed metadata contracts and path helpers only:
python -m pip install -e .
```

Optional dependencies are imported when the corresponding readers or stores are
used. The lean installs do not provide the full SITE backend dependency set.

## Development and documentation

From the activated environment and repository root:

```bash
python -m pip install -e ".[analysis,nd2,dev]"
python -m pytest
```

To build the documentation locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs/source docs/build/html
```

The generated HTML is written to `docs/build/html/`.
